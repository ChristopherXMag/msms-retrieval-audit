#!/usr/bin/env python3
"""Fixed CPU group ablations of D2, reported as post-hoc diagnostics only.

Reuses conditional-OOF training features, exposed calibration/validation and
frozen real/wrong-spectrum anchors. No selection, promotion or test loader.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb
import numpy as np
import torch
from rdkit import rdBase
from tools.reranking import run_official_lightgbm_transfer as d2
from tools.analysis.analyze_stage1_logit_ensemble import contributions, grouped, interval, metrics

DEFAULT = ROOT / 'outputs/lightgbm_strong_endpoint_transfer_20261015'
SEEDS = d2.TREE_SEEDS
DONORS = (20260820, 20260821, 20260822)
NEW_ARMS = ('without_specificity', 'without_mz_loss', 'absolute_only', 'relative_only')
ARMS = ('scalar', 'full') + NEW_ARMS
DONOR_FILE = ROOT / 'outputs/stage1_submission_0926/runs/full_r2_block_l2/seed20260621/blind_controls/spectrum_permutation_exact_mass.per_query.npz'


def write(path, value):
    staged = path.with_suffix(path.suffix + '.staging')
    staged.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    staged.replace(path)


def log(stage, **kwargs):
    print(json.dumps({'stage': stage, **kwargs}), flush=True)


def feature_sets(names):
    assert len(names) == 36 and names[10] == 'feasible_peaks'
    specificity = {'exclusive_intensity_fraction', 'exclusive_peak_fraction', 'unambiguous_peak_fraction'}
    location = {'matched_mean_mz', 'low_mz_intensity_fraction', 'middle_mz_intensity_fraction',
                'high_mz_intensity_fraction', 'small_loss_intensity_fraction'}
    return {
        'scalar': list(range(10)), 'full': list(range(36)),
        'without_specificity': [i for i, n in enumerate(names) if n.removeprefix('relative_') not in specificity],
        'without_mz_loss': [i for i, n in enumerate(names) if n.removeprefix('relative_') not in location],
        'absolute_only': list(range(23)), 'relative_only': list(range(10)) + list(range(23, 36)),
    }


def freeze(parent, out):
    assert not out.exists(), 'Use a new directory; never overwrite a frozen follow-up'
    d2.check_sources(parent)
    summary = json.loads((parent / 'features/summary.json').read_text())
    names = summary['feature_names']; sets = feature_sets(names)
    assert [len(sets[x]) for x in ARMS] == [10, 36, 30, 26, 23, 23]
    paths = [Path(__file__), ROOT / 'tools/analysis/analyze_stage1_logit_ensemble.py',
             parent / 'freeze_manifest.json', parent / 'features/train.npz', parent / 'features/inputs.jsonl',
             parent / 'features/summary.json', parent / 'trees/summary.json', parent / 'trees/calibration.npz',
             parent / 'formal_validation/summary.json', parent / 'formal_validation/scores_ranks.npz',
             d2.D1 / 'd1_inputs_val/inputs.jsonl', d2.D1 / 'd1_val_formal/per_query.npz',
             d2.FORMAL / 'top5_projection_cache/seed20260621/val_top5.pt', DONOR_FILE,
             ROOT / 'outputs/msalign_stage1_full_v15/val_full.metadata.jsonl']
    paths += sorted((parent / 'trees').glob('*.txt'))
    paths += [parent / f'formal_validation/wrong_seed{s}.npz' for s in DONORS]
    out.mkdir(parents=True)
    contract = {
        'date': '2026-10-15 Asia/Shanghai', 'status': 'FROZEN_POST_HOC_DIAGNOSTIC',
        'authorization': 'User requested small group ablations; no top20 expansion or feature search.',
        'parent_contract_sha256': d2.digest(parent / 'freeze_manifest.json'),
        'feature_names': names, 'feature_indices': sets,
        'feature_names_per_arm': {a: [names[i] for i in sets[a]] for a in ARMS},
        'new_fits': '4 groups x3 fixed seeds; reuse the existing full/scalar models',
        'fit': 'Same 5738 supported GT-reachable queries from the fixed 8000 connectivity cohort; same labels, candidate permutations, real pair+coverage init and sample weights.',
        'params': d2.trees.PARAMS, 'num_boost_round': 50, 'model_seeds': list(SEEDS),
        'fixed_support': 'Same all-query fallback for every arm; never reselect support using feature groups.',
        'evaluation': 'All existing 1024 calibration and19429 official-validation queries; both development-exposed. Calibration is diagnostic only: no count/group selection, no continuation gate, no refit.',
        'controls': 'All3 original2155-query/388-connectivity donor mappings. Real pair and real additive coverage anchor fixed; recompute all spectrum-derived scalar/detail features; same real support mask.',
        'comparisons': 'Every new group vs full/scalar/coverage; all arms real-vs-wrong and wrong-vs-anchor. Also (arm_real-scalar_real)-(arm_wrong-scalar_wrong), a paired performance contrast, not an additive causal decomposition.',
        'uncertainty': '10000 paired connectivity-cluster bootstrap draws, query-weighted and connectivity-balanced; fixed fits/selections; exploratory, no multiplicity adjustment.',
        'selection': 'NONE. Report every declared arm and3 seeds; no ranked winner or replacement endpoint.',
        'caveats': ['OOF boundary conditional on published frozen DreaMS features, not their entire pretraining chain.',
                    'feature_fraction=.8 retained: dropping columns also changes feature subsampling; correlated/redundant summaries prevent causal-necessity claims.',
                    'All10 scalar features retained in every evidence ablation; absolute/relative refer only to the26 detail summaries.',
                    'The original D2 gates,0.25pp margin and non-promotion decision remain unchanged.'],
        'official_test_read': False, 'device': 'CPU', 'rdkit': rdBase.rdkitVersion,
        'input_sha256': {str(p.relative_to(ROOT)): d2.digest(p) for p in paths},
    }
    assert lgb.__version__ == '4.6.0' and rdBase.rdkitVersion == '2023.09.6'
    write(out / 'freeze_manifest.json', contract)
    shutil.copyfile(__file__, out / 'source.py')
    log('freeze', sha256=d2.digest(out / 'freeze_manifest.json'), dimensions={a: len(sets[a]) for a in ARMS})


def check(parent, out):
    contract = json.loads((out / 'freeze_manifest.json').read_text())
    for path, value in contract['input_sha256'].items():
        assert d2.digest(ROOT / path) == value, path
    assert contract['parent_contract_sha256'] == d2.digest(parent / 'freeze_manifest.json')
    return contract


def load_models(parent, out, arm):
    directory = parent / 'trees' if arm in ('full', 'scalar') else out / 'models'
    prefix = 'detail' if arm == 'full' else arm
    return [lgb.Booster(model_file=str(directory / f'{prefix}_seed{s}.txt')) for s in SEEDS]


def predict(parent, out, arm, feature, support, anchor):
    contract = json.loads((out / 'freeze_manifest.json').read_text())
    x = feature[..., contract['feature_indices'][arm]]
    models = load_models(parent, out, arm)
    values = np.stack([m.predict(x.reshape(-1, x.shape[-1]), num_threads=2).reshape(-1, 5) for m in models])
    values[:, ~support] = 0
    return anchor + values.mean(0), anchor[None] + values


def fit(parent, out):
    c = check(parent, out); dest = out / 'models'
    assert not dest.exists(); dest.mkdir()
    z = dict(np.load(parent / 'features/train.npz'))
    train = np.flatnonzero((z['parts'] == 'fit') & z['support'] & (z['label'] >= 0))
    cal = np.flatnonzero(z['parts'] == 'calibration')
    assert len(train) == 5738 and len(cal) == 1024
    assert set(z['groups'][train]).isdisjoint(z['groups'][cal])
    order = np.array([np.random.default_rng(20261010 + int(i)).permutation(5) for i in train])
    target = (np.arange(5)[None] == z['label'][:, None]).astype(np.int32)
    labels = np.take_along_axis(target[train], order, 1).flatten()
    init = np.take_along_axis(z['anchor'][train], order, 1).flatten()
    started = time.time()
    for arm in NEW_ARMS:
        x = z['features'][..., c['feature_indices'][arm]]
        xx = np.take_along_axis(x[train], order[:, :, None], 1).reshape(-1, x.shape[-1])
        for seed in SEEDS:
            params = dict(d2.trees.PARAMS, seed=seed, feature_fraction_seed=seed,
                          data_random_seed=seed, bagging_seed=seed)
            data = lgb.Dataset(xx, label=labels, group=np.full(len(train), 5), init_score=init,
                               feature_name=c['feature_names_per_arm'][arm], free_raw_data=False)
            model = lgb.train(params, data, num_boost_round=50)
            model.save_model(str(dest / f'{arm}_seed{seed}.txt'))
            assert model.num_trees() == 50
        log('fit', arm=arm, features=x.shape[-1], queries=len(train), seconds=time.time() - started)
    inputs = d2.records(parent / 'features/inputs.jsonl')
    ci = [inputs[i] for i in cal]
    anchor_r = d2.rank_from(ci, z['label'][cal], z['rank'][cal], z['anchor'][cal])
    payload = {'indices': cal, 'groups': z['groups'][cal], 'anchor_ranks': anchor_r}; report = {}
    for arm in ARMS:
        scores, seeds = predict(parent, out, arm, z['features'][cal], z['support'][cal], z['anchor'][cal])
        ranks = d2.rank_from(ci, z['label'][cal], z['rank'][cal], scores)
        seed_ranks = np.stack([d2.rank_from(ci, z['label'][cal], z['rank'][cal], s) for s in seeds])
        payload.update({f'scores_{arm}': scores, f'ranks_{arm}': ranks, f'seed_ranks_{arm}': seed_ranks})
        report[arm] = {'vs_anchor': d2.trees.calibration(anchor_r, ranks),
                       'seed_metrics_vs_anchor': [d2.trees.calibration(anchor_r, r) for r in seed_ranks]}
    old = np.load(parent / 'trees/calibration.npz')
    assert np.array_equal(payload['scores_full'], old['detail'])
    assert np.array_equal(payload['scores_scalar'], old['scalar'])
    np.savez_compressed(out / 'calibration.npz', **payload)
    write(out / 'fit_summary.json', {'fit_queries': len(train), 'calibration_queries': len(cal),
          'calibration_is_already_exposed': True, 'selection': None, 'arms': report,
          'seconds': time.time() - started,
          'model_sha256': {p.name: d2.digest(p) for p in dest.glob('*.txt')}})


def build_features(out, evidence, name, inputs, base, support, coverage):
    """Bounded chunks; source formulas remain the frozen D2 implementation."""
    path = out / f'features_{name}.npz'
    assert not path.exists()
    n = len(inputs); x = np.empty((n, 5, 36), np.float32); cv = np.empty((n, 5), np.float32)
    statuses = Counter(); start_time = time.time()
    for st in range(0, n, 256):
        d2.memory_guard(); end = min(st + 256, n)
        if coverage is None:
            first = evidence.build(inputs[st:end], base[st:end], coverage_override=np.zeros((end-st, 5), np.float32))
            built = evidence.build(inputs[st:end], base[st:end], coverage_override=first['computed_coverage'])
        else:
            built = evidence.build(inputs[st:end], base[st:end], coverage_override=coverage[st:end])
            assert np.array_equal(built['support'], support[st:end])
            assert float(np.abs(built['computed_coverage']-coverage[st:end]).max()) <= 1e-6
        assert built['names'] == json.loads((out / 'freeze_manifest.json').read_text())['feature_names']
        x[st:end] = built['features']; cv[st:end] = built['coverage']; statuses.update(built['support_counts'])
        if st % 1024 == 0 or end == n:
            log('features', channel=name, queries=end, total=n, seconds=time.time()-start_time, rss_mib=d2.memory_state()['rss_mib'])
    np.savez_compressed(path, features=x, coverage=cv)
    return x, cv, dict(statuses)


def prepare(parent, out):
    check(parent, out)
    assert not (out / 'feature_summary.json').exists()
    z = dict(np.load(parent / 'formal_validation/scores_ranks.npz'))
    old = dict(np.load(d2.D1 / 'd1_val_formal/per_query.npz'))
    inputs = d2.records(d2.D1 / 'd1_inputs_val/inputs.jsonl')
    cache = torch.load(str(d2.FORMAL / 'top5_projection_cache/seed20260621/val_top5.pt'), map_location='cpu', weights_only=False, mmap=True)
    assert z['identifiers'].tolist() == cache['identifiers'] == [q['identifier'] for q in inputs]
    assert np.array_equal(z['candidate_rows'], cache['candidate_global_rows'].numpy())
    base = cache['ensemble_scores'].numpy(); anchor = z['scores_coverage']; support = z['supported']
    assert np.array_equal(anchor, d2.deployment_anchor(old['pair_scores'], old['coverage_local']))
    evidence = d2.Evidence()
    feature, _, statuses = build_features(out, evidence, 'real', inputs, base, support, old['coverage_local'])
    for arm, key in [('full', 'detail'), ('scalar', 'scalar')]:
        score, _ = predict(parent, out, arm, feature, support, anchor)
        assert np.array_equal(score, z[f'scores_{key}']), f'real {arm} differs from frozen D2'
    ii = np.arange(32); perm = np.array([2, 4, 0, 3, 1]); alternate = []
    for i in ii:
        q = dict(inputs[i]); q['candidate_smiles'] = [q['candidate_smiles'][j] for j in perm]
        q['candidate_global_rows'] = [q['candidate_global_rows'][j] for j in perm]; alternate.append(q)
    rebuilt = evidence.build(alternate, base[ii][:, perm], coverage_override=old['coverage_local'][ii][:, perm])
    assert np.array_equal(rebuilt['features'], feature[ii][:, perm])
    for arm in ARMS:
        a, _ = predict(parent, out, arm, rebuilt['features'], support[ii], anchor[ii][:, perm])
        b, _ = predict(parent, out, arm, feature[ii], support[ii], anchor[ii])
        assert np.array_equal(a, b[:, perm]), arm
    donors = dict(np.load(DONOR_FILE)); rec = donors['recipient_rows'].astype(int)
    metadata = d2.records(ROOT / 'outputs/msalign_stage1_full_v15/val_full.metadata.jsonl')
    donor_summary = {}
    for seed in DONORS:
        dd = donors[f'seed{seed}_donor_rows'].astype(int)
        assert set(dd) == set(rec) and np.all(z['groups'][rec] != z['groups'][dd])
        wrong = []
        for i, j in zip(rec, dd):
            assert metadata[i]['adduct'] == metadata[j]['adduct']
            assert round(float(metadata[i]['parent_mass']), 6) == round(float(metadata[j]['parent_mass']), 6)
            q = dict(inputs[i]); q['observed_spectrum'] = inputs[j]['observed_spectrum']; wrong.append(q)
        wf, _, _ = build_features(out, evidence, f'wrong_seed{seed}', wrong, base[rec], support[rec], None)
        frozen = dict(np.load(parent / f'formal_validation/wrong_seed{seed}.npz'))
        score, _ = predict(parent, out, 'full', wf, support[rec], anchor[rec])
        assert np.array_equal(rec, frozen['recipient_rows']) and np.array_equal(dd, frozen['donor_rows'])
        assert np.array_equal(score, frozen['wrong_scores']), f'wrong {seed} differs from frozen D2'
        donor_summary[str(seed)] = {'queries': len(rec), 'connectivities': len(set(z['groups'][rec])), 'full_scores_exact': True}
    write(out / 'feature_summary.json', {'real_queries': len(inputs), 'support_counts': statuses,
          'full_and_scalar_real_scores_exact': True, 'actual_candidate_permutation_exact_all_arms': True,
          'donors': donor_summary, 'frozen_anchor_unchanged': True,
          'feature_file_sha256': {p.name: d2.digest(p) for p in out.glob('features_*.npz')}})


def cluster_contrast(delta, groups, seed):
    sums, sizes, means = grouped(delta, groups); rng = np.random.default_rng(seed)
    qw = np.empty((10000, 4)); cb = np.empty_like(qw)
    for st in range(0, 10000, 128):
        n = min(128, 10000-st); draw = rng.integers(0, len(sizes), size=(n, len(sizes)))
        qw[st:st+n] = sums[draw].sum(1) / sizes[draw].sum(1)[:, None]
        cb[st:st+n] = means[draw].mean(1)
    return {'delta_query_weighted_pp': metrics(delta), 'delta_connectivity_balanced_pp': metrics(means),
            'paired_bootstrap': {'resamples': 10000, 'seed': seed,
                'connectivity_cluster_query_weighted': interval(qw), 'connectivity_balanced': interval(cb)}}


def describe(ranks, groups):
    values = contributions(ranks)
    return {'query_weighted': metrics(values), 'connectivity_balanced': metrics(grouped(values, groups)[2])}


def compare(before, after, groups, seed):
    a = describe(before, groups); b = describe(after, groups)
    rescue = int(np.count_nonzero((before > 1) & (after == 1)))
    damage = int(np.count_nonzero((before == 1) & (after > 1)))
    gap = a['query_weighted']['recall@20'] - a['query_weighted']['recall@1']
    result = cluster_contrast(contributions(after)-contributions(before), groups, seed)
    result.update({'baseline': a, 'after': b, 'rescues': rescue, 'damages': damage,
                   'damage_rate_pct': 100*damage/max(int(np.count_nonzero(before == 1)), 1),
                   'gap_closure_pct': 100*result['delta_query_weighted_pp']['recall@1']/gap if gap else None})
    return result


def evaluate(parent, out):
    check(parent, out)
    assert not (out / 'summary.json').exists()
    z = dict(np.load(parent / 'formal_validation/scores_ranks.npz'))
    inputs = d2.records(d2.D1 / 'd1_inputs_val/inputs.jsonl'); x = np.load(out / 'features_real.npz')['features']
    c = json.loads((out / 'freeze_manifest.json').read_text()); anchor = z['scores_coverage']
    labels = z['labels']; original = z['baseline_ranks']; support = z['supported']; groups = z['groups']
    payload = {'identifiers': z['identifiers'], 'groups': groups}; ranks = {}; rows = {}
    for arm in ARMS:
        score, ss = predict(parent, out, arm, x, support, anchor)
        r = d2.rank_from(inputs, labels, original, score)
        sr = np.stack([d2.rank_from(inputs, labels, original, s) for s in ss]); ranks[arm] = r
        assert np.isfinite(score).all() and np.array_equal(score[~support], anchor[~support])
        assert np.array_equal(r[labels < 0], original[labels < 0])
        assert np.array_equal(r <= 5, original <= 5) and np.array_equal(r <= 20, original <= 20)
        rows[arm] = {'features': len(c['feature_indices'][arm]), **describe(r, groups),
                     'per_seed': {str(seed): describe(v, groups) for seed, v in zip(SEEDS, sr)}}
        payload.update({f'scores_{arm}': score, f'ranks_{arm}': r, f'seed_ranks_{arm}': sr})
    assert np.array_equal(ranks['full'], z['ranks_detail']) and np.array_equal(ranks['scalar'], z['ranks_scalar'])
    np.savez_compressed(out / 'validation.npz', **payload)
    comparisons = {}
    for arm in NEW_ARMS:
        comparisons[arm] = {}
        for baseline in ('full', 'scalar', 'coverage'):
            before = ranks[baseline] if baseline != 'coverage' else z['ranks_coverage']
            comparisons[arm][f'vs_{baseline}'] = compare(before, ranks[arm], groups, 20261015)
            log('comparison', arm=arm, baseline=baseline,
                delta_r1=comparisons[arm][f'vs_{baseline}']['delta_query_weighted_pp']['recall@1'])
    donors = dict(np.load(DONOR_FILE)); rec = donors['recipient_rows'].astype(int)
    wi = [inputs[i] for i in rec]; controls = {}
    for seed in DONORS:
        xf = np.load(out / f'features_wrong_seed{seed}.npz')['features']
        wranks = {}; scores = {}; report = {}
        for arm in ARMS:
            score, ss = predict(parent, out, arm, xf, support[rec], anchor[rec])
            r = d2.rank_from(wi, labels[rec], original[rec], score); wranks[arm] = r; scores[arm] = score
            report[arm] = {
                'real_minus_wrong': compare(r, ranks[arm][rec], groups[rec], seed),
                'wrong_minus_real_anchor': compare(z['ranks_coverage'][rec], r, groups[rec], seed),
                'wrong_uplift_upper95_pp': d2.one_sided_upper(z['ranks_coverage'][rec], r, groups[rec], seed)}
        contrast = {}
        for arm in ('full',) + NEW_ARMS:
            delta = (contributions(ranks[arm][rec])-contributions(ranks['scalar'][rec])
                     - contributions(wranks[arm])+contributions(wranks['scalar']))
            contrast[arm] = cluster_contrast(delta, groups[rec], seed)
        np.savez_compressed(out / f'wrong_seed{seed}.npz', recipient_rows=rec,
                            **{f'ranks_{a}': v for a, v in wranks.items()},
                            **{f'scores_{a}': v for a, v in scores.items()})
        controls[str(seed)] = {'queries': len(rec), 'connectivities': len(set(groups[rec])),
                               'arms': report, 'real_vs_wrong_increment_over_scalar': contrast}
        log('controls', donor_seed=seed)
    check(parent, out)
    write(out / 'summary.json', {'status': 'COMPLETE_POST_HOC_DIAGNOSTIC', 'selected_arm': None,
          'original_D2_gates_unchanged': True, 'promoted': False, 'official_test_read': False,
          'queries': len(inputs), 'connectivities': len(set(groups)), 'rows': rows,
          'comparisons': comparisons, 'wrong_channel': controls,
          'calibration': json.loads((out / 'fit_summary.json').read_text())['arms'],
          'caveats': c['caveats'] + ['All intervals exploratory; no multiplicity correction; no model/group selection from validation.']})
    files = [p for p in out.iterdir() if p.is_file()] + list((out / 'models').glob('*.txt'))
    write(out / 'completion_manifest.json', {str(p.relative_to(out)): d2.digest(p) for p in files})
    log('COMPLETE', selected_arm=None, promoted=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('freeze', 'fit', 'prepare', 'evaluate', 'all'))
    p.add_argument('--parent', type=Path, default=DEFAULT)
    p.add_argument('--out', type=Path)
    a = p.parse_args(); parent = a.parent.resolve(); out = (a.out or parent / 'group_ablation').resolve()
    stages = ('freeze', 'fit', 'prepare', 'evaluate') if a.stage == 'all' else (a.stage,)
    for stage in stages:
        log('starting', phase=stage)
        globals()[stage](parent, out)


if __name__ == '__main__':
    main()
