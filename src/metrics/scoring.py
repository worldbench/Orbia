"""Score normalization and T1--T6 aggregation."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

CONFIG = json.loads(Path(__file__).with_name("scoring.json").read_text())


def mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.mean(values) if values else None


def complete_mean(values):
    values = list(values)
    return mean(values) if values and all(v is not None for v in values) else None


def geometric_mean(values, weights=None):
    """Weighted geometric mean of the present values, as mean() is; 0 if any is 0."""
    values = list(values)
    weights = [1.]*len(values) if weights is None else list(weights)
    pairs = [(float(v), w) for v, w in zip(values, weights) if v is not None and math.isfinite(float(v))]
    if not pairs:
        return None
    if min(v for v, _ in pairs) <= 0:
        return 0.
    return math.exp(sum(w*math.log(v) for v, w in pairs)/sum(w for _, w in pairs))


def complete_geometric_mean(values, weights=None):
    values = list(values)
    return geometric_mean(values, weights) if values and all(v is not None for v in values) else None


def normalize(value, rule):
    if value is None or not math.isfinite(float(value)):
        return None
    value = float(value)
    if rule['mode'] == 'unit':
        return 100*max(0., min(1., value))
    level = max(0., min(1., (value-rule['lo'])/(rule['hi']-rule['lo'])))
    return 100*(level if rule['direction'] == 1 else 1-level)


def error_score(value, bound):
    return normalize(value, {'mode':'linear', 'lo':0., 'hi':bound, 'direction':-1})


def aggregate_tree(tree, values, combine=complete_mean):
    if isinstance(tree, str):
        return values.get(tree)
    return combine(aggregate_tree(t, values, combine) for t in
                   (tree.values() if isinstance(tree, dict) else tree))


def report_leaves(report):
    leaves = {}
    for key, field in CONFIG['fields'].items():
        value = report['metrics'].get(field['metric'], {})
        for part in field['path'].split('.'):
            value = value.get(part) if isinstance(value, dict) else None
        leaves[key] = normalize(value, CONFIG['maps'][key])
    return leaves


def tier3(leaves, geometric=False):
    tree = CONFIG['trees']['T3']
    weights = [('return',.3),('qe',.4),('identity',.3)]
    if geometric:
        parts = {name: aggregate_tree(node, leaves, complete_geometric_mean) for name, node in tree.items()}
        return complete_geometric_mean([parts[n] for n,_ in weights], [w for _,w in weights]), parts
    parts = {name: aggregate_tree(node, leaves) for name, node in tree.items()}
    total = (sum(parts[name]*w for name,w in weights)
             if all(v is not None for v in parts.values()) else None)
    return total, parts


def retention(windows):
    windows = list(windows)
    if len(windows) < 2 or windows[0] is None:
        return None
    rest = [v for v in windows[1:] if v is not None]
    if not rest:
        return None
    return dict(level=mean(windows), retention=100-mean(max(0., windows[0]-v) for v in rest))


def temporal_score(dimensions):
    parts = {k:retention(v) for k,v in dimensions.items()}
    scores = [None if v is None else .3*v['level']+.7*v['retention'] for v in parts.values()]
    return complete_mean(scores), parts


def temporal_geometric(temporal):
    """A case's T6 score with each dimension as level^0.3 * retention^0.7, combined geometrically."""
    parts = (temporal or {}).get('parts') or {}
    return complete_geometric_mean(
        [None if p is None else complete_geometric_mean([p['level'], p['retention']], [.3, .7])
         for p in parts.values()])


def aggregate_reports(reports, output):
    """Normalize each case leaf before model means; missing leaves stay missing.

    Every tier and the overall are also reported with a ``_geo`` suffix: each
    combination of different parts (tier trees, tier parts, T6 level and
    retention, the overall) is a weighted geometric mean with the same weights,
    so a part at 0 makes the combination 0. Means over cases and T6's short and
    long horizon weights stay arithmetic in both.
    """
    by_model = defaultdict(list)
    for report in reports:
        by_model[report['model']].append(report)
    rows, components, geometric = [], [], {}
    for model, items in sorted(by_model.items()):
        groups = {r.get('group', 'world') for r in items}
        profiles = {r.get('control_profile', 'pose_and_action') for r in items}
        if len(groups) != 1 or len(profiles) != 1:
            raise ValueError('model group/control profile differs between cases: '+model)
        values = [report_leaves(r) for r in items]
        leaves = {k:mean(v.get(k) for v in values) for k in CONFIG['maps']}
        t3, t3_parts = tier3(leaves)
        tree = CONFIG['trees']['T4']
        direct, g_direct = [], []
        for r, v in zip(items, values):
            raw = r['metrics'].get('completion_revisit', {}).get('raw', {}).get('direct', {})
            branch_values, g_scores, g_counts = [], [], []
            for branch in ('same_pose','different_pose'):
                n = raw.get(branch, {}).get('expected_pair_count', 0)
                score = aggregate_tree(tree[branch], v)
                if n and score is not None:
                    branch_values.extend([score]*n)
                g_score = aggregate_tree(tree[branch], v, complete_geometric_mean)
                if n and g_score is not None:
                    g_scores.append(g_score)
                    g_counts.append(n)
            direct.append(mean(branch_values))
            g_direct.append(geometric_mean(g_scores, g_counts))
        geom = aggregate_tree(tree['geometry'], leaves)
        extra = [r.get('supporting', {}) for r in items]
        t1_parts = {k:mean(r.get('control', {}).get(k) for r in extra)
                    for k in ('translation','rotation','move','turn')}
        selected = ('move','turn') if next(iter(profiles)) == 'action_only' else tuple(t1_parts)
        t1 = mean(t1_parts[k] for k in selected)
        t2_parts = {k:mean(r.get('quality', {}).get(k) for r in extra)
                    for k in ('aesthetic','imaging','flicker','smoothness','hps')}
        t2 = complete_mean(t2_parts.values())
        t6_horizons = {h:mean(r.get('temporal', {}).get('score') for r in extra
                            if r.get('horizon') == h) for h in ('short','long')}
        t6 = (.3*t6_horizons['short']+.7*t6_horizons['long']
              if all(v is not None for v in t6_horizons.values()) else None)
        tiers = {'T1':t1,'T2':t2,'T3':t3,'T4':complete_mean([mean(direct),geom]),
                 'T5':aggregate_tree(CONFIG['trees']['T5'],leaves),'T6':t6}
        overall = sum(tiers[k]*w for k,w in CONFIG['overall_weights'].items()) if all(
            v is not None for v in tiers.values()) else None
        g_t3, g_t3_parts = tier3(leaves, geometric=True)
        g_geom = aggregate_tree(tree['geometry'], leaves, complete_geometric_mean)
        g_t6_horizons = {h:mean(temporal_geometric(r.get('temporal')) for r in extra
                              if r.get('horizon') == h) for h in ('short','long')}
        g_tiers = {'T1':geometric_mean(t1_parts[k] for k in selected),
                   'T2':complete_geometric_mean(t2_parts.values()), 'T3':g_t3,
                   'T4':complete_geometric_mean([mean(g_direct),g_geom]),
                   'T5':aggregate_tree(CONFIG['trees']['T5'],leaves,complete_geometric_mean),
                   'T6':(.3*g_t6_horizons['short']+.7*g_t6_horizons['long']
                         if all(v is not None for v in g_t6_horizons.values()) else None)}
        weights = CONFIG['overall_weights']
        geometric[model] = {**{k+'_geo':v for k,v in g_tiers.items()},
                            'Overall_geo':complete_geometric_mean([g_tiers[k] for k in weights],
                                                                  list(weights.values()))}
        rows.append(dict(model=model,group=next(iter(groups)),cases=len(items),**tiers,Overall=overall))
        components.append(dict(model=model,control=t1_parts,quality=t2_parts,input_preservation=t3_parts,
                               generated_persistence={'direct':mean(direct),'geometry':geom},
                               temporal=t6_horizons,normalized_leaves=leaves,
                               geometric=dict(input_preservation=g_t3_parts,
                                              generated_persistence={'direct':mean(g_direct),'geometry':g_geom},
                                              temporal=g_t6_horizons),
                               valid_leaf_counts={k:sum(v.get(k) is not None for v in values) for k in leaves}))
    for r in rows:
        for key in (*CONFIG['overall_weights'],'Overall'):
            peers = [p for p in rows if p['group'] == r['group'] and p[key] is not None]
            r[key+'_rank'] = 1+sum(p[key] > r[key] for p in peers) if r[key] is not None else None
    for r in rows:
        r.update(geometric[r['model']])
    for r in rows:
        for key in (*CONFIG['overall_weights'],'Overall'):
            key += '_geo'
            peers = [p for p in rows if p['group'] == r['group'] and p[key] is not None]
            r[key+'_rank'] = 1+sum(p[key] > r[key] for p in peers) if r[key] is not None else None
    rows.sort(key=lambda r:(r['group'], -(r['Overall'] if r['Overall'] is not None else -1), r['model']))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'scores.json').write_text(json.dumps({'rows':rows,'components':components,
        'protocol':CONFIG['version']},indent=2,allow_nan=False)+'\n')
    if rows:
        with (output/'scores.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows
