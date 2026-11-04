#!/usr/bin/env python3
"""Bounded, one-video-at-a-time worker with resident geometry/GS/metrics models."""
from __future__ import annotations

import argparse
import gc
from functools import partial
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--worker-id', default='0')
    parser.add_argument('--metrics', nargs='+', default=None)
    parser.add_argument('--tiers', nargs='+', default=None)
    parser.add_argument('--stage', choices=('geometry', 'precompute', 'full'), default='full')
    parser.add_argument('--input-revision', default='1')
    parser.add_argument('--geometry-root', type=Path, help='read-only geometry from an earlier run')
    parser.add_argument('--extra-site', action='append', default=[])
    parser.add_argument('--gs-render-intrinsics', choices=('given', 'predicted'), default='given')
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError('threads must be positive')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    for path in args.extra_site:
        sys.path.append(path)
    import cv2
    import torch
    import yaml
    cv2.setNumThreads(1)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    from src.metrics.video import decode_video_cached, clear_video_cache
    from src.metrics.contract import EvaluationOptions
    from src.metrics.features import ImageComparator
    from src.metrics.runner import prepare_da3_case, evaluate_case
    from src.backends.run_da3_evaluator import _build_real_backend, _load_request
    from src.backends.run_da3_streaming_evaluator import _read_real_outputs, _materialize_contract

    from src.metrics.da3 import da3_artifact_complete
    from src.metrics.case import load_evaluation_case
    from src.metrics.sampling import geometry_sampling_plan
    from src.metrics.stages import StageCache, file_identity, case_identity, validate_geometry
    from src.metrics.registry import KNOWN_METRICS
    from src.evaluate import metrics_for_tiers, normalize_tiers, merge_reports
    if args.metrics is not None and args.tiers is not None:
        parser.error('choose --tiers or --metrics, not both')
    tiers = normalize_tiers(args.tiers) if args.metrics is None else ()
    support_tiers = tuple(t for t in tiers if t in ('T1', 'T2', 'T6'))
    selected = (('da3_reconstruction',) if args.stage == 'precompute' else
                tuple(args.metrics) if args.metrics else metrics_for_tiers(tiers))
    if set(selected) - KNOWN_METRICS:
        raise ValueError('unknown metrics')
    release_config = yaml.safe_load(args.config.read_text())
    config = release_config.get('evaluator', release_config)
    if config.get('torch_home'):
        os.environ['TORCH_HOME'] = str(config['torch_home'])
    supporting_config = release_config.get('quality', {})
    base_config = dict(config)
    options = EvaluationOptions.from_mapping(config)
    loop_enabled = bool(config.get('da3_streaming_loop_enable', False))
    repo = Path(config['da3_streaming_repo'])
    checkpoint = Path(config['da3_streaming_checkpoint'])
    streaming = repo / 'da3_streaming'
    for path in (repo / 'src', repo, streaming):
        sys.path.insert(0, str(path))
    source = (streaming / 'da3_streaming.py').read_text()
    if not loop_enabled:
        source = source.replace('from loop_utils.loop_detector import LoopDetector', '# loop disabled for evaluator')
    source = source.replace('self.model.inference(images, ref_view_strategy=ref_view_strategy)',
        'self.model.inference(images, ref_view_strategy=ref_view_strategy, process_res='
        + str(int(config.get('da3_geometry_resolution',504))) + ')')
    begin = source.index('        with open(self.config["Weights"]["DA3_CONFIG"]) as f:')
    end = source.index('        self.skyseg_session = None', begin)
    source = source[:begin] + '        self.model = resident_model(self.config, self.device)\n\n' + source[end:]
    module = types.ModuleType('orbia_resident_streaming')
    module.__file__ = str(streaming / 'da3_streaming.py')
    models = {}
    def resident_model(cfg, device):
        if 'geometry' not in models:
            from depth_anything_3.api import DepthAnything3
            from safetensors.torch import load_file
            model = DepthAnything3(**json.loads(Path(cfg['Weights']['DA3_CONFIG']).read_text()))
            model.load_state_dict(load_file(cfg['Weights']['DA3']), strict=False)
            model.input_processor = partial(model.input_processor, num_workers=args.threads)
            models['geometry'] = model.eval().to(device)
            print('GEOMETRY_MODEL_LOAD_COUNT=1', flush=True)
        return models['geometry']
    module.resident_model = resident_model
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    if loop_enabled:
        from src.backends.da3_resident_loop import resident_loop_detector
        module.LoopDetector = resident_loop_detector(module.LoopDetector, options.dino_repo)
    base = yaml.safe_load((streaming / 'configs/base_config.yaml').read_text())
    base['Weights'].update(DA3=str(checkpoint / 'model.safetensors'), DA3_CONFIG=str(checkpoint / 'config.json'))
    base['Model'].update(chunk_size=config.get('da3_streaming_chunk_size', 120), overlap=config.get('da3_streaming_overlap', 30), loop_enable=loop_enabled, delete_temp_files=True, save_depth_conf_result=True)
    if loop_enabled:
        salad = config.get('da3_streaming_salad_checkpoint')
        if not salad or not Path(salad).is_file():
            raise FileNotFoundError('loop-on requires da3_streaming_salad_checkpoint')
        base['Weights']['SALAD'] = str(salad)
    # Same default as the existing bridge.
    base['Model']['align_lib'] = 'triton'
    render = None
    comparator = None if args.stage != 'full' else ImageComparator(backend=options.feature_backend, dino_repo=options.dino_repo, dino_checkpoint=options.dino_checkpoint, lpips_checkpoint=options.lpips_checkpoint, device=options.device)
    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    failed = False
    with (args.output / f'worker_{args.worker_id}_results.jsonl').open('a', buffering=1) as log:
        for case_index, job in enumerate(jobs):
            started = perf_counter()
            artifact_root = args.output / job['model'] if job.get('model') else args.output
            case_output = artifact_root / 'results' / job['case_id']
            timing = {}
            record = {'case_id': job['case_id'], 'cold_start': case_index == 0, 'model':job.get('model')}
            try:
                print('START_CASE', job['case_id'], flush=True)
                torch.cuda.reset_peak_memory_stats()
                info, frames = decode_video_cached(job['video'], int(job['frame_count']))
                timing['decode'] = perf_counter() - started
                geometry = (args.geometry_root or (artifact_root / 'geometry')) / job['case_id']
                cache = StageCache(artifact_root / 'stages' / job['case_id'])
                from src.metrics.precompute import load_precompute_case
                case = load_precompute_case(job['case_root']) if args.stage == 'precompute' else load_evaluation_case(job['case_root'])
                config = dict(base_config)
                options = EvaluationOptions.from_mapping(config)
                if 'anchor_trajectory' in selected:
                    from src.metrics.reference_visibility import prepare_reference_visibility
                    visibility, visibility_status = prepare_reference_visibility(
                        case, info, options, job,
                        options.anchor_tracking_visibility or (args.output.parent/'reference_visibility'))
                    config['anchor_tracking_visibility'] = str(visibility) if visibility else None
                    options = EvaluationOptions.from_mapping(config)
                    record['reference_visibility'] = visibility_status
                plan = geometry_sampling_plan(case, info, options)
                identity = {}
                if args.stage == 'full':
                    identity = {'revision': args.input_revision, 'video':file_identity(job['video']),
                                'case':case_identity(job['case_root']), 'config':config,
                                'implementation':'orbia-1', 'supporting':supporting_config,
                                'group':job.get('group','world'),
                                'control_profile':job.get('control_profile','pose_and_action'),
                                'metric_weights':[file_identity(v) for k,v in sorted(config.items()) if k.endswith('checkpoint') and v and Path(v).is_file()],
                                'quality_weights':[file_identity(v) for k,v in sorted(supporting_config.items()) if k.endswith(('_checkpoint', '_config')) and v and Path(v).is_file()],
                                'source':[file_identity(p) for p in sorted((Path(__file__).resolve().parents[1]/'metrics').glob('*.py'))],
                                'supporting_source':file_identity(Path(__file__).resolve().parents[1]/'metrics'/'supporting.py')}
                from src.metrics.precompute import geometry_identity_for, gs_identity_for
                geometry_identity = geometry_identity_for(job, config, plan, args.input_revision)
                phase_start = perf_counter()
                geometry_ok = lambda: validate_geometry(geometry, plan['frame_indices'], info.frame_count)
                reused = cache.ready('geometry', geometry_identity, geometry_ok)
                if args.geometry_root:
                    source_contract = json.loads((geometry/'complete.json').read_text())
                    if (source_contract.get('case_id') != job['case_id'] or
                        Path(source_contract.get('source_video','')).resolve() != Path(job['video']).resolve()):
                        raise ValueError('external geometry belongs to a different case/video')
                    if not geometry_ok():
                        raise ValueError('external geometry does not match the specified input sampling plan')
                    reused = True
                if not reused:
                    (case_output/'precompute.json').unlink(missing_ok=True)
                    cache.invalidate('geometry')
                    geometry.mkdir(parents=True, exist_ok=True)
                    # Remove depths left over from a different sampling plan.
                    for stale in (geometry/'depth').glob('*.npz'):
                        stale.unlink()
                    with tempfile.TemporaryDirectory(prefix='stream-', dir=geometry) as temp:
                        # Only the neural model is reused across videos.
                        if len(plan['frame_indices']) <= int(config.get('da3_joint_max_frames', 0)):
                            from src.backends.precompute_joint_geometry import run_joint_geometry
                            run_joint_geometry(model=resident_model(base, options.device), frames=frames,
                                info=info, job=job, plan=plan, output=geometry, temp=Path(temp),
                                resolution=int(config.get('da3_geometry_resolution', 504)))
                            record.update(geometry_backend='joint', loop_pair_count=0, loop_constraint_count=0)
                        else:
                            pipeline = module.DA3_Streaming('', temp, base)
                            pipeline.img_list = [frames[i] for i in plan['frame_indices']]
                            if loop_enabled:
                                pipeline.loop_detector.rgb_frames = pipeline.img_list
                            pipeline.process_long_sequence()
                            record['loop_pair_count'] = len(pipeline.loop_list)
                            record['loop_constraint_count'] = len(pipeline.loop_sim3_list)
                            poses, intrinsics, depths = _read_real_outputs(Path(temp), frame_count=len(plan['frame_indices']), video_width=info.width, video_height=info.height)
                            _materialize_contract(job=job, output=geometry, poses_c2w=poses, intrinsics_video=intrinsics, source_depths=depths, provenance={'backend':'da3_streaming', 'checkpoint':str(checkpoint), 'resident':True, 'sampling_plan':plan, 'chunk_size':base['Model']['chunk_size'], 'overlap':base['Model']['overlap'], 'align_lib':'triton', 'loop_closure':loop_enabled, 'loop_pair_count':len(pipeline.loop_list), 'salad_checkpoint':base['Weights'].get('SALAD') if loop_enabled else None})
                            del pipeline
                    if not geometry_ok():
                        raise ValueError('geometry contract validation failed')
                    cache.commit('geometry', geometry_identity, [p for p in geometry.rglob('*') if p.is_file()])
                record['geometry_reused'] = reused
                record['geometry_input_frames'] = len(plan['frame_indices'])
                record['video_frames'] = info.frame_count
                timing['geometry'] = perf_counter() - phase_start
                geometry_peak = torch.cuda.max_memory_allocated()
                gc.collect()
                torch.cuda.empty_cache()
                if args.stage == 'geometry':
                    record['status'] = 'complete'
                    timing.update(gs=0.0, metrics=0.0)
                else:
                    phase_start = perf_counter()
                    gs_root = case_output/'da3'
                    gs_identity = gs_identity_for(geometry_identity, config, plan, info.frame_count)
                    gs_identity['pose_cache'] = [file_identity(p) for p in sorted((geometry/'evaluator_pose_cache').glob('*.npy'))]
                    # Keyed only when changed, so caches from the default renders stay valid.
                    if args.gs_render_intrinsics != 'given':
                        gs_identity['render_intrinsics'] = args.gs_render_intrinsics
                    gs_ok = lambda: da3_artifact_complete(gs_root, context_count=options.da3_context_views, heldout_count=options.da3_heldout_views)
                    gs_reused = cache.ready('gs', gs_identity, gs_ok)
                    if 'da3_reconstruction' in selected and not gs_reused:
                        (case_output/'precompute.json').unlink(missing_ok=True)
                        cache.invalidate('gs')
                        context_images = {}
                        request = prepare_da3_case(preloaded_case=case, decoded_frames=frames, context_images=context_images, case_root=job['case_root'], video_path=job['video'], output_root=case_output, pose_cache=geometry/'evaluator_pose_cache', config=config, video_info=info)
                        if render is None:
                            same_checkpoint = Path(config['da3_checkpoint']).resolve() == checkpoint.resolve()
                            render = _build_real_backend(Path(config['da3_repo']), Path(config['da3_checkpoint']), options.device, resident_model=resident_model(base, options.device) if same_checkpoint else None, render_intrinsics=args.gs_render_intrinsics)
                        render(_load_request(request), gs_root/'bridge_result.json', context_images=context_images)
                        del context_images
                        if not gs_ok():
                            raise ValueError('missing GS bridge result')
                        cache.commit('gs', gs_identity, [p for p in gs_root.rglob('*') if p.is_file()])
                    record['gs_reused'] = gs_reused
                    timing['gs'] = perf_counter() - phase_start
                    torch.cuda.empty_cache()
                    if args.stage == 'precompute':
                        from src.metrics.stages import atomic_json
                        atomic_json(case_output/'precompute.json', dict(status='complete',
                            case_id=job['case_id'], video=file_identity(job['video']),
                            geometry_identity=geometry_identity, gs_identity=gs_identity))
                        record['status'] = 'complete'
                        timing['metrics'] = 0.0
                    else:
                        phase_start = perf_counter()
                        report_context = {**identity, 'geometry':gs_identity['geometry'],
                            'depths':[file_identity(p) for p in sorted((geometry/'depth').glob('*.npz'))]}
                        metric_identity = {**report_context, 'metrics':list(selected), 'tiers':list(tiers),
                            'gs_receipt':file_identity(cache.path('gs')) if 'da3_reconstruction' in selected else None,
                            'anchor_tracker_script':file_identity(Path(__file__).with_name('run_anchor_mask_tracking.py')) if 'anchor_identity' in selected else None,
                            'reference_visibility':([file_identity(p) for p in sorted(options.anchor_tracking_visibility.rglob('*.npz'))] if options.anchor_tracking_visibility else [])}
                        prior = case_output/'report.json'
                        def metric_ok():
                            if not prior.is_file():
                                return False
                            saved = json.loads(prior.read_text())
                            return (saved.get('status') == 'complete' and
                                    set(support_tiers).issubset(saved.get('evaluated_tiers', [])))
                        metrics_reused = cache.ready('metrics', metric_identity, metric_ok)
                        if metrics_reused:
                            report = json.loads(prior.read_text())
                        else:
                            cache.invalidate('metrics')
                            previous = json.loads(prior.read_text()) if prior.is_file() else None
                            comparator.clear_case_cache()
                            report = evaluate_case(case_root=job['case_root'], video_path=job['video'], output_root=case_output, pose_cache=geometry/'evaluator_pose_cache', depth_root=geometry/'depth', config=config, metrics=selected, image_comparator=comparator, require_precomputed_da3=True, video_info=info)
                            if report['status'] == 'complete' and support_tiers:
                                from src.metrics.supporting import calculate
                                try:
                                    report['supporting'] = calculate(case,info,frames,report,geometry/'evaluator_pose_cache',
                                        options,{**supporting_config,'device':options.device},case_output,
                                        control_profile=job.get('control_profile','pose_and_action'),group=job.get('group','world'),
                                        tiers=support_tiers)
                                except Exception as exc:
                                    report.update(status='incomplete', supporting_error=f'{type(exc).__name__}: {exc}')
                            report['evaluated_tiers'] = list(tiers) if report['status'] == 'complete' else []
                            report = merge_reports(previous, report, report_context)
                            report.update(model=job['model'],group=job.get('group','world'),
                                          control_profile=job.get('control_profile','pose_and_action'))
                            from src.metrics.stages import atomic_json
                            atomic_json(prior,report)
                            if report['status'] == 'complete':
                                cache.commit('metrics', metric_identity, [prior, *[case_output/'metrics'/(name+'.json') for name in selected]])
                        record['metrics_reused'] = metrics_reused
                        timing['metrics'] = perf_counter() - phase_start
                        record['status'] = report['status']
                        record['metric_statuses'] = {name:value.get('status') for name,value in report['metrics'].items()}
                record['decode_count'] = 1
                record['peak_gpu_memory_bytes'] = max(geometry_peak, torch.cuda.max_memory_allocated())
            except Exception as exc:
                import traceback
                traceback.print_exc()
                record.update(status='incomplete', reason=f'{type(exc).__name__}: {exc}')
            finally:
                pipeline = None
                clear_video_cache()
                frames = None
                if comparator is not None:
                    comparator.clear_case_cache()
                gc.collect()
            record.update(timing_s=timing, elapsed_s=perf_counter() - started)
            log.write(json.dumps(record) + '\n')
            print('FINISH_CASE', json.dumps(record), flush=True)
            failed = failed or record['status'] != 'complete'
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
