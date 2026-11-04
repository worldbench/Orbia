"""Dispatch existing videos to one resident worker per GPU."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from src.metrics.scoring import aggregate_reports
from src.metrics.registry import (DEFAULT_METRICS, TIER1_METRICS, TIER2_METRICS,
                                TIER3_METRICS, TIER4_METRICS, TIER5_METRICS)

ROOT = Path(__file__).resolve().parents[1]

TIERS = ('T1', 'T2', 'T3', 'T4', 'T5', 'T6')
TIER_METRICS = dict(zip(TIERS, (TIER1_METRICS, TIER2_METRICS, TIER3_METRICS,
    TIER4_METRICS, TIER5_METRICS,
    (*TIER1_METRICS, *TIER2_METRICS[:2], *TIER5_METRICS[:3]))))


def normalize_tiers(tiers=None):
    if tiers is None:
        return TIERS
    if isinstance(tiers, str):
        tiers = tiers.replace(',', ' ').split()
    selected = {str(t).upper() for t in tiers}
    if not selected or selected - set(TIERS):
        raise ValueError('tiers must be one or more of T1, T2, T3, T4, T5, T6')
    return tuple(t for t in TIERS if t in selected)


def metrics_for_tiers(tiers=None):
    requested = {'validity'}
    for tier in normalize_tiers(tiers):
        requested.update(TIER_METRICS[tier])
    return tuple(metric for metric in DEFAULT_METRICS if metric in requested)


def merge_reports(previous, current, context):
    """Keep separately computed tiers only when they share the same inputs."""
    if previous and previous.get('evaluation_context') == context:
        current['metrics'] = {**previous.get('metrics', {}), **current['metrics']}
        support = {**previous.get('supporting', {}), **current.get('supporting', {})}
        if support:
            current['supporting'] = support
        current['evaluated_tiers'] = sorted(set(previous.get('evaluated_tiers', [])) |
                                            set(current.get('evaluated_tiers', [])))
        from src.metrics.runner import INCOMPLETE_METRIC_STATUSES
        if any(m.get('status') in INCOMPLETE_METRIC_STATUSES for m in current['metrics'].values()):
            current['status'] = 'incomplete'
    current['evaluation_context'] = context
    return current


def local_path(value):
    return Path(value).expanduser().resolve()


def load_jobs(manifest, dataset, *, model=None, group='world', control_profile='pose_and_action'):
    manifest, dataset = Path(manifest).resolve(), Path(dataset).resolve()
    if manifest.is_dir():
        rows = [dict(model=model or manifest.name, case_id=p.stem, video=str(p),
                     group=group, control_profile=control_profile)
                for p in sorted(manifest.glob('*.mp4'))]
    else:
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    seen, jobs = set(), []
    for line_number,row in enumerate(rows,1):
        for field in ('model','case_id'):
            value = row[field]
            if not value or Path(value).name != value or value in ('.','..'):
                raise ValueError(f'line {line_number}: invalid {field}')
        key = (row['model'],row['case_id'])
        if key in seen:
            raise ValueError('duplicate model/case: '+str(key))
        seen.add(key)
        case_root = dataset/'public'/row['case_id']
        case = json.loads((case_root/'case.json').read_text())
        video = Path(row['video']).expanduser()
        video = local_path(video if video.is_absolute() else manifest.parent/video)
        if not video.is_file():
            raise FileNotFoundError(video)
        group = row.get('group','world')
        profile = row.get('control_profile','pose_and_action')
        if group not in ('world','ti2v') or profile not in ('pose_and_action','action_only'):
            raise ValueError('use group world/ti2v and control_profile pose_and_action/action_only')
        jobs.append(dict(model=row['model'],case_id=row['case_id'],case_root=str(case_root),
                         video=str(video),frame_count=case['frame_count'],group=group,control_profile=profile))
    if not jobs:
        raise ValueError('no videos: provide a JSONL manifest or a folder of <case_id>.mp4 files')
    return jobs


def aggregate_directory(root, output):
    reports = [json.loads(p.read_text()) for p in sorted(Path(root).glob('*/results/*/report.json'))]
    reports = [r for r in reports if r.get('status') == 'complete']
    if not reports:
        raise ValueError('no complete evaluation reports')
    seen = {(r['model'],r['case_id']) for r in reports}
    if len(seen) != len(reports):
        raise ValueError('duplicate reports')
    return aggregate_reports(reports,output)


def evaluate(dataset, videos, config, output, devices='0', threads=4, stage='full', metrics=None,
             geometry_root=None, tiers=None, model=None, group='world', control_profile='pose_and_action',
             gs_render_intrinsics='given'):
    if tiers is not None and metrics is not None:
        raise ValueError('choose --tiers or --metrics, not both')
    selected_tiers = normalize_tiers(tiers) if metrics is None else None
    dataset, videos, config, output = map(local_path,(dataset,videos,config,output))
    import yaml
    configured = yaml.safe_load(config.read_text())
    for section in ('evaluator', 'quality'):
        values = configured.get(section, configured if section == 'evaluator' else {})
        for key, value in values.items():
            if value and isinstance(value, str) and key.endswith(('_repo', '_root', '_checkpoint', '_config', '_python', '_visibility')):
                local_path(value)
    jobs = load_jobs(videos,dataset,model=model,group=group,control_profile=control_profile)
    gpu_ids = [v.strip() for v in devices.split(',') if v.strip()]
    if len(set(gpu_ids)) != len(gpu_ids) or not gpu_ids:
        raise ValueError('devices must be distinct GPU IDs')
    output.mkdir(parents=True,exist_ok=True)
    controls = output/'workers'
    controls.mkdir(exist_ok=True)
    children = []
    try:
        for rank,device in enumerate(gpu_ids):
            partition = jobs[rank::len(gpu_ids)]
            if not partition:
                continue
            job_file = controls/f'jobs_{rank}.jsonl'
            job_file.write_text(''.join(json.dumps(row)+'\n' for row in partition))
            command = [sys.executable,'-m','src.backends.worker','--jobs',str(job_file),
                       '--config',str(config),'--output',str(output),'--worker-id',str(rank),
                       '--threads',str(threads),'--stage',stage]
            if metrics:
                command.extend(['--metrics',*metrics])
            elif selected_tiers:
                command.extend(['--tiers',*selected_tiers])
            if geometry_root:
                command.extend(['--geometry-root',str(local_path(geometry_root))])
            command.extend(['--gs-render-intrinsics',gs_render_intrinsics])
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=device,PYTHONPATH=str(ROOT))
            log = (controls/f'worker_{rank}.log').open('a')
            children.append((subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT),log))
        failures = [process.wait() for process,_ in children]
    finally:
        for process,log in children:
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()
    if any(failures):
        raise RuntimeError('some cases failed; inspect workers/*.log and rerun to resume')
    if stage == 'full':
        return aggregate_directory(output,output/'scores')
    return {'jobs':len(jobs),'stage':stage}
