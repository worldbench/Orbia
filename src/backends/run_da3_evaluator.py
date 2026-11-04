#!/usr/bin/env python3
"""Offline resident bridge for ORBIA's DA3 held-out renderer."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable

import numpy as np
from PIL import Image


def _write_result(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _validate_request(request: dict) -> None:
    if (
        request.get("format") != "orbia.da3.request.v1"
        or request.get("reconstruction_inputs") != "context_rgb_only"
        or request.get("align_to_input_ext_scale") is not True
        or request.get("infer_gs") is not True
        or request.get("no_post_optimization") is not True
    ):
        raise ValueError("invalid DA3 request")
    if len(request.get("context", [])) < 2:
        raise ValueError("DA3 requires at least two context views")


def _load_request(path: Path) -> dict:
    request = json.loads(path.read_text(encoding="utf-8"))
    _validate_request(request)
    return request


def _save_renders(
    request: dict,
    output: Path,
    colors: np.ndarray,
    depths: np.ndarray,
    alphas: np.ndarray,
    provenance: dict,
) -> None:
    heldout = request["heldout"]
    if len(colors) != len(heldout) or len(depths) != len(heldout) or len(alphas) != len(heldout):
        raise ValueError("DA3 renderer returned the wrong held-out view count")
    render_root = Path(request["render_root"])
    render_root.mkdir(parents=True, exist_ok=True)
    records = []
    for item, color, depth, alpha in zip(heldout, colors, depths, alphas):
        index = int(item["frame_index"])
        rgb_path = render_root / f"rgb_{index:06d}.png"
        depth_path = render_root / f"depth_{index:06d}.npy"
        alpha_path = render_root / f"alpha_{index:06d}.npy"
        color = np.asarray(color)
        if color.ndim == 3 and color.shape[0] == 3:
            color = color.transpose(1, 2, 0)
        Image.fromarray((np.clip(color, 0, 1) * 255).astype(np.uint8)).save(rgb_path)
        np.save(depth_path, np.asarray(depth, dtype=np.float32).squeeze())
        np.save(alpha_path, np.asarray(alpha, dtype=np.float32).squeeze())
        records.append({
            "frame_index": index,
            "rgb": str(rgb_path.resolve()),
            "depth": str(depth_path.resolve()),
            "alpha": str(alpha_path.resolve()),
        })
    _write_result(output, {
        "status": "ok",
        "renders": records,
        "provenance": provenance,
    })




def _normalize_w2c(extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Match DepthAnything3._normalize_extrinsics for context and novel views."""

    context = np.asarray(extrinsics, dtype=np.float64)
    reference_c2w = np.linalg.inv(context[0])
    normalized = context @ reference_c2w
    camera_positions = np.linalg.inv(normalized)[:, :3, 3]
    distances = np.linalg.norm(camera_positions, axis=-1)
    # torch.median selects the LOWER middle element for even view counts.
    middle = (len(distances) - 1) // 2
    scale = max(float(np.partition(distances, middle)[middle]), 0.1)
    normalized[:, :3, 3] /= scale
    return normalized, reference_c2w, scale


def _safe_render_batch_size(gaussian_count: int, sh_coefficients: int, view_count: int) -> int:
    """Bound uint32 SH coefficient indexing in the pinned gsplat kernel."""
    stride = int(gaussian_count) * int(sh_coefficients) * 3
    if stride <= 0 or view_count <= 0:
        raise ValueError("positive Gaussian, SH and view counts required")
    safe_views = ((1 << 32) - 1) // stride
    if safe_views < 1:
        raise ValueError("one view exceeds gsplat uint32 SH indexing; use a corrected kernel")
    return min(4, int(view_count), safe_views)


# How the held-out views are rendered. DA3 builds the Gaussians by unprojecting
# its depth with its own predicted intrinsics; "given" renders with the request's
# intrinsics (the original behavior), "predicted" with the mean predicted
# context intrinsics, the camera the Gaussians were built for.
RENDER_INTRINSICS = ("given", "predicted")


def _build_real_backend(
    repo: Path, checkpoint: Path, device: str, *, resident_model=None,
    render_intrinsics: str = "given",
) -> Callable[[dict, Path], None]:
    if render_intrinsics not in RENDER_INTRINSICS:
        raise ValueError(f"render_intrinsics must be one of {RENDER_INTRINSICS}")
    if not repo.is_dir():
        raise FileNotFoundError(f"DA3 repository does not exist: {repo}")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"DA3 checkpoint directory does not exist: {checkpoint}")
    for required in ("config.json", "model.safetensors"):
        if not (checkpoint / required).is_file():
            raise FileNotFoundError(f"DA3 checkpoint is missing {required}: {checkpoint}")
    sys.path.insert(0, str(repo / "src"))
    import torch
    from einops import rearrange
    from gsplat import rasterization
    from depth_anything_3.api import DepthAnything3

    model = resident_model
    if model is None:
        model = DepthAnything3.from_pretrained(
            str(checkpoint), local_files_only=True
        ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if render_intrinsics == "predicted":
        # inference() replaces prediction.intrinsics with the given ones; keep
        # a copy of the predicted ones, which the Gaussians are built with.
        convert_to_prediction = model._convert_to_prediction

        def convert_keeping_intrinsics(raw_output):
            prediction = convert_to_prediction(raw_output)
            if prediction.intrinsics is not None:
                prediction.predicted_intrinsics = prediction.intrinsics.copy()
            return prediction

        model._convert_to_prediction = convert_keeping_intrinsics

    def render(request: dict, output: Path, *, context_images=None) -> None:
        context = request["context"]
        heldout = request["heldout"]
        context_w2c = np.asarray(
            [item["extrinsics_w2c"] for item in context], dtype=np.float32
        )
        context_k = np.asarray(
            [item["intrinsics"] for item in context], dtype=np.float32
        )
        normalized_context, reference_c2w, camera_scale = _normalize_w2c(context_w2c)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        inference_started = perf_counter()
        prediction = model.inference(
            image=[context_images[int(item["frame_index"])] if context_images is not None else item["rgb"] for item in context],
            extrinsics=context_w2c,
            intrinsics=context_k,
            process_res=int(request["process_resolution"]),
            process_res_method="upper_bound_resize",
            align_to_input_ext_scale=True,
            infer_gs=True,
            export_dir=None,
        )
        inference_s = perf_counter() - inference_started
        if prediction.gaussians is None:
            raise RuntimeError("DA3 inference returned no Gaussians")

        target_w2c = np.asarray(
            [item["extrinsics_w2c"] for item in heldout], dtype=np.float64
        )
        target_w2c = target_w2c @ reference_c2w
        target_w2c[:, :3, 3] /= camera_scale
        resolution = int(request["resolution"])
        if render_intrinsics == "predicted":
            # From DA3's processed size to the render resolution.
            processed_height, processed_width = prediction.depth.shape[-2:]
            render_k = prediction.predicted_intrinsics.mean(axis=0)
            render_k[0] *= resolution / processed_width
            render_k[1] *= resolution / processed_height
            target_k = np.repeat(render_k[None].astype(np.float32), len(heldout), axis=0)
        else:
            target_k = np.asarray(
                [item["intrinsics"] for item in heldout], dtype=np.float32
            )
        gaussians = prediction.gaussians
        viewmats = torch.from_numpy(target_w2c.astype(np.float32)).to(gaussians.means)
        pixel_k = torch.from_numpy(target_k).to(gaussians.means)
        harmonics = rearrange(
            gaussians.harmonics[0], "g xyz n -> g n xyz"
        ).contiguous()
        degree = math.isqrt(harmonics.shape[-2]) - 1

        def rasterize_views(start: int, stop: int):
            return rasterization(
                means=gaussians.means[0],
                quats=gaussians.rotations[0],
                scales=gaussians.scales[0],
                opacities=gaussians.opacities[0],
                colors=harmonics,
                viewmats=viewmats[start:stop],
                Ks=pixel_k[start:stop],
                backgrounds=torch.zeros((stop - start, 3), device=viewmats.device),
                render_mode="RGB+D",
                width=resolution,
                height=resolution,
                packed=False,
                sh_degree=degree,
            )

        def to_cpu(color, alpha):
            return (
                color[..., :3].detach().float().cpu().numpy(),
                color[..., -1].detach().float().cpu().numpy(),
                alpha.detach().float().cpu().numpy(),
            )

        render_started = perf_counter()
        render_batch_size = _safe_render_batch_size(
            gaussians.means.shape[1], harmonics.shape[-2], len(heldout)
        )
        # The pinned SH kernel uses uint32 offsets. OOM-only fallback is not
        # sufficient: overflow returns corrupted RGB without raising an error.
        while True:
            pieces = []
            try:
                for start in range(0, len(heldout), render_batch_size):
                    colors, alphas, _ = rasterize_views(start, min(start + render_batch_size, len(heldout)))
                    pieces.append(to_cpu(colors, alphas))
                    del colors, alphas
                break
            except RuntimeError as error:
                if "out of memory" not in str(error).lower() or render_batch_size == 1:
                    raise
                pieces.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                render_batch_size = max(1, render_batch_size // 2)
        rgb = np.concatenate([piece[0] for piece in pieces], axis=0)
        depths = np.concatenate([piece[1] for piece in pieces], axis=0)
        alpha = np.concatenate([piece[2] for piece in pieces], axis=0)
        render_s = perf_counter() - render_started
        _save_renders(
            request,
            output,
            rgb,
            depths,
            alpha,
            {
                "implementation": "official_da3",
                "model_name": getattr(model, "model_name", None),
                "resident_backend": True,
                "infer_gs": True,
                "align_to_input_ext_scale": True,
                "camera_source": "vipe_w2c_intrinsics",
                "render_intrinsics": render_intrinsics,
                "context_only_reconstruction": True,
                "context_normalization_scale": camera_scale,
                "normalized_context_count": len(normalized_context),
                "rasterization_batch_size": render_batch_size,
                "inference_s": inference_s,
                "render_s": render_s,
                "peak_allocated_vram_bytes": (
                    int(torch.cuda.max_memory_allocated())
                    if torch.cuda.is_available() else None
                ),
            },
        )

    return render


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--request", type=Path)
    mode.add_argument("--jobs-jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--render-intrinsics", choices=RENDER_INTRINSICS, default="given",
        help="intrinsics for the held-out renders (default: given)",
    )
    args = parser.parse_args()

    if args.request is not None:
        if args.output is None:
            parser.error("--output is required with --request")
        jobs = [{"request": str(args.request), "output": str(args.output)}]
    else:
        assert args.jobs_jsonl is not None
        jobs = []
        for line_number, line in enumerate(
            args.jobs_jsonl.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            job = json.loads(line)
            if not isinstance(job, dict) or not all(
                isinstance(job.get(key), str) and job[key]
                for key in ("request", "output")
            ):
                raise ValueError(
                    f"jobs JSONL line {line_number} requires request/output strings"
                )
            jobs.append(job)

    render = _build_real_backend(
        args.repo.resolve(), args.checkpoint.resolve(), args.device,
        render_intrinsics=args.render_intrinsics,
    )
    records = []
    for job in jobs:
        request_path = Path(job["request"])
        output_path = Path(job["output"])
        started = perf_counter()
        try:
            render(_load_request(request_path), output_path)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            _write_result(output_path, {
                "status": "evaluator_failed", "reason": reason, "renders": [],
            })
            records.append({
                "request": str(request_path), "output": str(output_path),
                "status": "evaluator_failed", "reason": reason,
                "runtime_s": perf_counter() - started,
            })
        else:
            records.append({
                "request": str(request_path), "output": str(output_path),
                "status": "ok", "runtime_s": perf_counter() - started,
            })
    if args.summary is not None:
        failed_count = sum(item["status"] != "ok" for item in records)
        _write_result(args.summary, {
            "format": "orbia.da3.worker.v1",
            "status": "ok" if failed_count == 0 else "incomplete",
            "job_count": len(records),
            "ok_count": len(records) - failed_count,
            "failed_count": failed_count,
            "jobs": records,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
