"""Evaluate existing generated videos with the ORBIA metric pipeline."""
import argparse
from pathlib import Path
from src.evaluate import TIERS, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', required=True, help='folder of <case_id>.mp4 files or JSONL manifest')
    parser.add_argument('--dataset', default='data/orbia', help='dataset root (default: data/orbia)')
    parser.add_argument('--config', default='configs/evaluation.local.yaml', help='config written by download_weights.py')
    parser.add_argument('--output', default='results', help='reports and scores (default: results)')
    parser.add_argument('--model', help='model name for a video folder (default: folder name)')
    parser.add_argument('--group', choices=('world', 'ti2v'), default='world', help='model group for a video folder')
    parser.add_argument('--control-profile', choices=('pose_and_action', 'action_only'),
                        default='pose_and_action', help='control scoring for a video folder')
    parser.add_argument('--devices', default='0', help='comma-separated GPU IDs')
    parser.add_argument('--threads', type=int, default=4, help='CPU threads per worker')
    parser.add_argument('--stage', choices=('geometry', 'precompute', 'full'), default='full')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--tiers', '--tier', nargs='+', type=str.upper, choices=TIERS,
                           help='tiers to evaluate (default: all six)')
    selection.add_argument('--metrics', nargs='+', help='advanced: raw metric names without tier summaries')
    parser.add_argument('--geometry-root', help='reuse a previous geometry directory')
    parser.add_argument('--gs-render-intrinsics', choices=('given', 'predicted'), default='given',
                        help='T5 reconstruction: render held-out views with the given intrinsics (default) '
                             "or with DA3's predicted intrinsics, which the Gaussians are built with")
    args = parser.parse_args()
    rows = evaluate(args.dataset, args.videos, args.config, args.output,
             devices=args.devices, threads=args.threads, stage=args.stage,
             metrics=args.metrics, geometry_root=args.geometry_root, tiers=args.tiers,
             model=args.model, group=args.group, control_profile=args.control_profile,
             gs_render_intrinsics=args.gs_render_intrinsics)
    if args.stage == 'full':
        columns = (*TIERS, 'Overall', 'Overall_geo')
        print('Model\t' + '\t'.join(columns))
        for row in rows:
            values = ['-' if row.get(key) is None else f'{row[key]:.2f}' for key in columns]
            print(row['model'] + '\t' + '\t'.join(values))
        print(f'Saved scores to {Path(args.output) / "scores" / "scores.csv"}')


if __name__ == '__main__':
    main()
