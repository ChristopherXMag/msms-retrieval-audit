#!/usr/bin/env python3
"""Check supplementary estimates from archived arrays; NumPy only, no model runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_metrics import uniform_ties
WEIGHTS = ('query_weighted', 'connectivity_balanced')
METRICS = ('R1', 'R5', 'R20', 'MRRx100')


def estimate(values, groups):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), values.shape[1]))
    np.add.at(sums, inverse, values)
    return 100 * np.stack((values.mean(0), (sums / counts[:, None]).mean(0)))


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)


def check_cells(values, groups, cells, metrics=METRICS):
    points = estimate(values, groups)
    for i, weight in enumerate(WEIGHTS):
        for j, metric in enumerate(metrics):
            close(points[i, j], cells[weight][metric]['point'])
    if 'queries' in cells:
        assert len(groups) == cells['queries']
        assert len(np.unique(groups)) == cells['connectivities']


def clustering(hit, groups, minimum):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    total = np.bincount(inverse, weights=hit)
    squares = np.bincount(inverse, weights=hit * hit)
    maximum = np.full(len(counts), -np.inf)
    np.maximum.at(maximum, inverse, hit)
    keep = counts >= minimum
    n, t, sq, maximum = (x[keep] for x in (counts, total, squares, maximum))
    N, G = n.sum(), len(n)
    effective_n = (N - (n * n).sum() / N) / (G - 1)
    between = ((t * t / n).sum() - t.sum() ** 2 / N) / (G - 1)
    within = (sq - t * t / n).sum() / (N - G)
    wrong, right = t == 0, t == n
    return dict(ICC1=(between-within)/(between+(effective_n-1)*within),
                all_wrong_molecule_share=wrong.mean(), all_right_molecule_share=right.mean(),
                mixed_molecule_share=(~(wrong | right)).mean(),
                all_wrong_query_share=(n*wrong).sum()/N, baseline=t.sum()/N,
                sibling_oracle=(n*maximum).sum()/N, oracle_gain=((n*maximum).sum()-t.sum())/N)


def ranks(x):
    order = np.argsort(x, kind='stable')
    start = np.r_[0, np.flatnonzero(np.diff(x[order])) + 1]
    end = np.r_[start[1:], len(x)]
    result = np.empty(len(x), dtype=float)
    result[order] = np.repeat((start + end - 1) / 2, end-start)
    return result


def verify(root, output):
    if output.exists():
        raise FileExistsError('Choose a fresh output file')
    report = {'status': 'RUNNING', 'models_loaded': False, 'new_inference': False, 'checks': {}}
    supplement = root / 'outputs/revision_supplement_20261111'
    for split in ('validation', 'test'):
        for item in ('R1a', 'R1b', 'R2', 'R3', 'R4', 'R5a'):
            folder = supplement / item / split
            summary = json.loads((folder/'summary.json').read_text())
            count = 0
            with np.load(folder/'per_query.npz', allow_pickle=False) as z:
                groups = z['connectivities']; ids = z['identifiers']
                assert groups.shape == ids.shape and len(np.unique(ids)) == len(ids)
                if item == 'R1a':
                    for name, outcomes in summary['rows'].items():
                        for outcome, cells in outcomes.items():
                            hit = z[name+'_'+outcome]
                            for label, minimum in (('minimum4', 4), ('all_molecules', 1)):
                                result = clustering(hit, groups, minimum)
                                for key, value in result.items():
                                    close(value, cells[label]['values'][key]); count += 1
                elif item == 'R1b':
                    bins = np.digitize(z['proximity'], [.6, .7, .8, .9])
                    np.testing.assert_array_equal(bins, z['bin_index'])
                    for name, cell in summary['rows'].items():
                        values = z[name]
                        for i, label in enumerate(summary['fixed_bins']):
                            mask = bins == i
                            if mask.any():
                                check_cells(values[mask], groups[mask], cell['bins'][label]['metrics']); count += 8
                    _, inv, n = np.unique(groups, return_inverse=True, return_counts=True)
                    prox = np.bincount(inv, weights=z['proximity']) / n
                    for name, column in (('sccf_ensemble', 0), ('sccf_ensemble', 2), ('selected', 0)):
                        hit = np.bincount(inv, weights=z[name][:, column]) / n
                        key = name + ('_R1' if column == 0 else '_R20')
                        close(np.corrcoef(ranks(prox), ranks(hit))[0, 1], summary['correlations'][key]['rho']); count += 1
                elif item == 'R2':
                    for population, block in summary['populations'].items():
                        index = np.arange(len(ids)) if population == 'full' else z['queries__wrong_seed20260820']
                        assert len(index) == block['queries']
                        for method, controls in block['rows'].items():
                            real = z[method+'__real'][index, :1]
                            for mode, cells in controls.items():
                                q = z['queries__'+mode]; positions = np.searchsorted(q, index)
                                np.testing.assert_array_equal(q[positions], index)
                                value = z[method+'__'+mode][positions, :1]
                                check_cells(value, groups[index], cells['metrics'], ('R1',)); count += 2
                                if mode != 'real':
                                    check_cells(real-value, groups[index], cells['real_minus_control_pp'], ('R1',)); count += 2
                                    ratio = estimate(value, groups[index]) / estimate(real, groups[index])
                                    for j, w in enumerate(WEIGHTS):
                                        close(ratio[j, 0], cells['control_over_real_retention'][w]); count += 1
                    for mode in ('wrong_seed20260820', 'wrong_seed20260821', 'wrong_seed20260822'):
                        q, donors = z['queries__'+mode], z['donors__'+mode]
                        assert np.all(groups[q] != groups[donors])
                elif item == 'R3':
                    for name, cells in summary['rows'].items():
                        check_cells(z[name], groups, cells); count += 8
                    for contrast, cells in summary['contrasts'].items():
                        after, before = contrast.split('_minus_')
                        check_cells(z[after]-z[before], groups, cells); count += 8
                    counts, gt = z['candidate_counts'], z['gt_slots']
                    for name in ('msalign_ensemble', 'embcos_ensemble'):
                        scores = z[name+'_scores']
                        valid = np.arange(scores.shape[1])[None, :] < counts[:, None]
                        truth = scores[np.arange(len(ids)), gt]
                        greater = ((scores > truth[:, None]) & valid).sum(1)
                        equal = ((scores == truth[:, None]) & valid).sum(1)
                        close(uniform_ties(greater, equal), z[name]); count += 1
                        del scores, valid, truth
                elif item == 'R4':
                    for name, cells in summary['rows'].items():
                        before, after = z[name+'_baseline'], z[name+'_fused']
                        for value, key in ((before, 'baseline'), (after, 'fused'), (after-before, 'gain_pp')):
                            check_cells(value, groups, cells[key]); count += 8
                elif item == 'R5a':
                    for name, mask in (('CE_present', np.isfinite(z['collision_energy'])), ('CE_missing', np.isnan(z['collision_energy']))):
                        block = summary['strata'][name]
                        assert int(mask.sum()) == block['queries']
                        for method, cells in block['rows'].items():
                            check_cells(z[method][mask], groups[mask], cells); count += 8
            report['checks'][item+'/'+split] = {'status': 'PASS', 'estimate_checks': count}
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2)+'\n')
            print('PASS', item, split, count, flush=True)
    folder = supplement/'R6_lite/validation/analysis'
    summary = json.loads((folder/'summary.json').read_text())
    with np.load(folder/'per_query_and_bootstrap.npz', allow_pickle=False) as z:
        groups = z['connectivities']; points = {}
        for arm in ('targeted', 'size_matched', 'baseline'):
            values = z[arm]
            for i, label in enumerate(('seed20260621', 'seed20260622', 'two_seed_mean')):
                value = values[i] if i < 2 else values.mean(0)
                for cohort in ('A', 'B_clean', 'B_spill'):
                    mask = np.isin(groups, z[cohort])
                    block = summary['existing_full_train_reference'] if arm == 'baseline' else summary['metrics'][arm]
                    close(estimate(value[mask], groups[mask]), [[block[label][cohort][m] for m in ('recall@1','recall@5','recall@20','mrr')],
                          [block[label][cohort]['connectivity_balanced'][m] for m in ('recall@1','recall@5','recall@20','mrr')]])
        difference = z['targeted']-z['size_matched']
        close(difference, z['paired_differences'])
        for cohort, name in (('A', 'A_targeted_minus_random'), ('B_clean', 'B_clean_targeted_minus_random')):
            mask = np.isin(groups, z[cohort])
            points[name] = np.stack([estimate(v[mask], groups[mask]) for v in (*difference, difference.mean(0))])
        points['difference_in_differences'] = points['A_targeted_minus_random']-points['B_clean_targeted_minus_random']
        for name, point in points.items():
            interval = 100*np.quantile(z['bootstrap_'+name], [.025, .975], axis=0)
            for i, label in enumerate(('seed20260621', 'seed20260622', 'two_seed_mean')):
                for j, w in enumerate(WEIGHTS):
                    for k, metric in enumerate(('R@1','R@5','R@20','MRR')):
                        cell = summary['contrasts'][name][label][w][metric]
                        close(point[i,j,k], cell['estimate_pp'])
                        close(interval[:,i,j,k], cell['ci95_pp'])
        assert len(z['A']) == 200 and len(z['B_clean']) == 2691
    report['checks']['R6_lite/validation'] = {'status': 'PASS', 'scope': 'all cohort/arm/seed estimates and saved-bootstrap contrast quantiles; no retraining'}
    report['status'] = 'PASS'
    report['scope'] = 'R1a clustering estimates; R1b bins/correlations; R2 control points/paired effects; R3 ensembles/ranks; R4 mass fusion points; R5a CE strata; R6 cohort means and archived bootstrap quantiles. Fresh R1 permutation/bootstrap and original solver/training/inference are not run.'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print('ALL PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.data_root, args.output)
