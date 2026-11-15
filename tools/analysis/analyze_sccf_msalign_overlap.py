#!/usr/bin/env python3
"""CPU-only frozen official-val SCCF/MSAlign error overlap; no fitting or fusion."""
from __future__ import annotations
import argparse
from itertools import combinations
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from glmr.embcos import sha256_file, write_json, uniform_tie_metrics
SEEDS = (20260621, 20260622, 20260623)
E1 = ROOT / 'outputs/msalign_official_e1_20261014'
SCORES = ROOT / 'outputs/stage2_official_datasafe/stage1_candidate_scores/val_three_seed_logits.npz'
ROWS = ROOT / 'outputs/embcos_official_e2_20261014/cache/val.rows.npz'
CELL_NAMES = ('both_correct', 'sccf_only_correct', 'msalign_only_correct', 'both_wrong')


def sources():
    paths = [Path(__file__), ROOT / 'glmr/embcos.py', ROWS, SCORES,
             SCORES.with_suffix('.summary.json'), E1 / 'analysis/summary.json',
             E1 / 'assets/val.metadata.npz']
    for seed in SEEDS:
        paths.extend([E1 / f'seed{seed}/val_ranks.npz', E1 / f'seed{seed}/train_summary.json',
                      ROOT / f'outputs/stage1_submission_0926/runs/full_r2_block_l2/seed{seed}/blind_controls/constant_spectrum_full_val.per_query.npz',
                      E1 / f'prior_followup_20261014/inference/seed{seed}/mean_fixed_both.npz'])
    return paths


def freeze(out):
    assert not out.exists(), 'No overwrite of completed/post-hoc audit'
    contract = {'status': 'INPUTS_AND_ANALYSIS_FROZEN', 'seeds': list(SEEDS),
        'primary': 'same-seed pairs, mean of joint per-query categories over three pairs; not intersection of averaged correctness probabilities and not ensemble',
        'secondary': ['all nine cross-seed pairs', 'within-method seed pairs as training-noise reference',
                      'stable across-seed query and connectivity categories', 'mean-spectrum prior overlap',
                      'GT ranks of the other method on exclusive correct examples', 'unique-top predicted-candidate agreement'],
        'top1_ties': 'require zero fractional GT top1 tie probabilities; otherwise stop rather than introduce a joint-tie assumption',
        'bootstrap_replicates': 10000, 'bootstrap_seed': 20261014,
        'scope': 'post-hoc development audit conditional on existing validation-selected checkpoints; oracle uses GT and is not deployable; no backbone causality or clean claim',
        'test_evaluated': False, 'training_performed': False, 'fusion_performed': False,
        'input_source_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in sources()}}
    out.mkdir(parents=True)
    write_json(out / 'contract.json', contract)
    print('FROZEN', sha256_file(out / 'contract.json'), flush=True)


def joint(s, m):
    assert np.isin(s, [0., 1.]).all() and np.isin(m, [0., 1.]).all()
    return np.stack([s*m, s*(1-m), (1-s)*m, (1-s)*(1-m)], axis=-1)


def derived(v):
    v = np.asarray(v)
    b, so, mo, w = (v[..., i] for i in range(4))
    s = b + so; m = b + mo
    return {**{name: v[..., i] * 100 for i, name in enumerate(CELL_NAMES)},
        'sccf_r1': s * 100, 'msalign_r1': m * 100, 'oracle_r1': (1-w) * 100,
        'oracle_gain_over_msalign_pp': so * 100, 'oracle_gain_over_sccf_pp': mo * 100,
        'sccf_rescue_fraction_of_msalign_errors_pct': so / (so+w) * 100,
        'msalign_rescue_fraction_of_sccf_errors_pct': mo / (mo+w) * 100,
        'error_jaccard_pct': w / (so+mo+w) * 100,
        'correct_jaccard_pct': b / (b+so+mo) * 100,
        'sccf_errors_shared_with_msalign_pct': w / (mo+w) * 100,
        'msalign_errors_shared_with_sccf_pct': w / (so+w) * 100}


def group_values(v, groups):
    _, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(inv)
    totals = np.stack([np.bincount(inv, weights=v[:, j]) for j in range(v.shape[1])], 1)
    return totals, counts, totals/counts[:, None]


def describe(v, groups):
    _, counts, balanced = group_values(v, groups)
    return {'queries': len(v), 'connectivities': len(counts),
        'query_weighted': {k: float(x) for k, x in derived(v.mean(0)).items()},
        'connectivity_balanced': {k: float(x) for k, x in derived(balanced.mean(0)).items()},
        'expected_query_counts': {name: float(v[:, i].sum()) for i, name in enumerate(CELL_NAMES)}}


def bootstrap(v, groups, n, seed):
    totals, counts, means = group_values(v, groups)
    rng = np.random.default_rng(seed)
    draws = {name: np.empty((n, 4)) for name in ('query', 'cluster_query', 'balanced')}
    for start in range(0, n, 32):
        end = min(start+32, n)
        q = rng.integers(0, len(v), size=(end-start, len(v)))
        draws['query'][start:end] = v[q].mean(1)
        g = rng.integers(0, len(counts), size=(end-start, len(counts)))
        draws['cluster_query'][start:end] = totals[g].sum(1)/counts[g].sum(1)[:, None]
        draws['balanced'][start:end] = means[g].mean(1)
    points = {'query': v.mean(0), 'cluster_query': v.mean(0), 'balanced': means.mean(0)}
    return {kind: {k: {'point': float(derived(points[kind])[k]),
                      'ci95': np.quantile(value, [.025, .975]).tolist()}
                   for k, value in derived(values).items()}
            for kind, values in draws.items()}


def run(out):
    started = time.time()
    c = json.loads((out / 'contract.json').read_text())
    for path, digest in c['input_source_sha256'].items():
        assert sha256_file(ROOT / path) == digest, path
    assert not (out / 'summary.json').exists()
    rows = dict(np.load(ROWS, allow_pickle=False))
    ids = rows['identifiers']; groups = rows['connectivities']; lengths = rows['lengths']
    score = dict(np.load(SCORES, allow_pickle=False))
    assert np.array_equal(score['identifiers'], ids) and np.array_equal(score['connectivities'], groups)
    mask = np.arange(256)[None, :] < lengths[:, None]
    assert np.array_equal(rows['candidates'][mask], score['candidate_rows'])
    assert np.array_equal(rows['candidates'][:, 0], rows['permuted_candidates'][np.arange(len(ids)), rows['gt_slots']])
    assert np.array_equal(score['indptr'], np.r_[0, np.cumsum(lengths)])
    assert np.array_equal(lengths, score['candidate_counts'])
    per_seed = {}; sv = []; mv = []; sp = []; mp = []; score_checks = []
    for i, seed in enumerate(SEEDS):
        rec = dict(np.load(E1 / f'seed{seed}/val_ranks.npz', allow_pickle=False))
        assert np.array_equal(rec['identifiers'], ids) and np.array_equal(rec['connectivities'], groups)
        assert np.array_equal(rec['candidate_counts'], lengths) and np.array_equal(rec['gt_slots'], rows['gt_slots'])
        dense = np.full((len(ids), 256), -np.inf, np.float32)
        dense[mask] = score['scores'][i]
        target = dense[:, :1]
        greater = ((dense > target) & mask).sum(1)
        equal = ((dense == target) & mask).sum(1)
        path = ROOT / f'outputs/stage1_submission_0926/runs/full_r2_block_l2/seed{seed}/blind_controls/constant_spectrum_full_val.per_query.npz'
        with np.load(path) as blind:
            assert np.array_equal(blind['recipient_rows'], np.arange(len(ids)))
            assert np.array_equal(blind['connectivity'], groups)
            assert np.array_equal(greater, blind['baseline_greater'])
            assert np.array_equal(equal, blind['baseline_equal'])
            s_prior = uniform_tie_metrics(blind['train_mean_input_greater'], blind['train_mean_input_equal'])
        s = uniform_tie_metrics(greater, equal); m = rec['expected_metrics']
        assert np.array_equal(m, uniform_tie_metrics(rec['greater'], rec['equal']))
        with np.load(E1 / f'prior_followup_20261014/inference/seed{seed}/mean_fixed_both.npz') as z:
            assert np.array_equal(z['identifiers'], ids)
            m_prior = z['expected_metrics']
        j = joint(s[:, 0], m[:, 0]); prior_joint = joint(s_prior[:, 0], m_prior[:, 0])
        maxima_s = dense.max(1); maxima_m = rec['scores'].max(1)
        unique_s = (dense == maxima_s[:, None]).sum(1) == 1
        unique_m = (rec['scores'] == maxima_m[:, None]).sum(1) == 1
        unique = unique_s & unique_m
        s_top = rows['candidates'][np.arange(len(ids)), dense.argmax(1)]
        m_top = rows['permuted_candidates'][np.arange(len(ids)), rec['scores'].argmax(1)]
        same = s_top == m_top
        both_wrong = j[:, 3].astype(bool)
        def exclusive_ranks(weight, other):
            denominator = weight.sum()
            return {'exclusive_correct_count': float(denominator),
                'other_GT_rank_2_to_5_pct': float((weight*other[:, 1]).sum()/denominator*100),
                'other_GT_rank_6_to_20_pct': float((weight*(other[:, 2]-other[:, 1])).sum()/denominator*100),
                'other_GT_outside_top20_pct': float((weight*(1-other[:, 2])).sum()/denominator*100)}
        per_seed[str(seed)] = {'real_overlap': describe(j, groups), 'prior_overlap': describe(prior_joint, groups),
            'unique_top_queries': int(unique.sum()),
            'same_top_candidate_pct_on_unique_top_queries': float(same[unique].mean()*100),
            'both_wrong_unique_top_queries': int((unique & both_wrong).sum()),
            'same_wrong_candidate_pct_on_both_wrong_unique_top_queries': float(same[unique & both_wrong].mean()*100),
            'sccf_exclusive_other_rank': exclusive_ranks(j[:, 1], m),
            'msalign_exclusive_other_rank': exclusive_ranks(j[:, 2], s),
            'sccf_exclusive_blind_correct_count': float((j[:, 1]*s_prior[:, 0]).sum()),
            'msalign_exclusive_blind_correct_count': float((j[:, 2]*m_prior[:, 0]).sum())}
        sv.append(s); mv.append(m); sp.append(s_prior); mp.append(m_prior)
        score_checks.append({'seed': seed, 'sccf_export_vs_frozen_blind_counts': 'exact', 'fractional_GT_top1_ties': 0})
        print('PAIR COMPLETE', seed, json.dumps(per_seed[str(seed)]['real_overlap']['query_weighted']), flush=True)
    sv = np.stack(sv); mv = np.stack(mv); sp = np.stack(sp); mp = np.stack(mp)
    matched = np.stack([joint(sv[i, :, 0], mv[i, :, 0]) for i in range(3)]).mean(0)
    prior = np.stack([joint(sp[i, :, 0], mp[i, :, 0]) for i in range(3)]).mean(0)
    benchmark = json.loads((E1 / 'analysis/summary.json').read_text())
    assert np.isclose(sv[:, :, 0].mean()*100, benchmark['matched_sccf_aggregate']['recall@1'], atol=1e-12)
    assert np.isclose(mv[:, :, 0].mean()*100, benchmark['aggregate']['recall@1'], atol=1e-12)
    cross = {f'{SEEDS[i]}_{SEEDS[j]}': describe(joint(sv[i, :, 0], mv[j, :, 0]), groups) for i in range(3) for j in range(3)}
    within = {}
    for name, values in [('sccf', sv), ('msalign', mv)]:
        records = {}
        for i, j in combinations(range(3), 2):
            v = describe(joint(values[i, :, 0], values[j, :, 0]), groups)
            # Rename first/second-method fields for the within-family reference.
            records[f'{SEEDS[i]}_{SEEDS[j]}'] = {'first_seed': SEEDS[i], 'second_seed': SEEDS[j],
                **{weight: {'first_r1': v[weight]['sccf_r1'], 'second_r1': v[weight]['msalign_r1'],
                            'oracle_r1': v[weight]['oracle_r1'],
                            'gain_over_better_member_pp': v[weight]['oracle_r1']-max(v[weight]['sccf_r1'], v[weight]['msalign_r1']),
                            'error_jaccard_pct': v[weight]['error_jaccard_pct']} for weight in ('query_weighted', 'connectivity_balanced')}}
        within[name] = records
    s1 = sv[:, :, 0]; m1 = mv[:, :, 0]
    stable = {'sccf_correct_all3_msalign_wrong_all3': (s1.min(0)==1)&(m1.max(0)==0),
              'msalign_correct_all3_sccf_wrong_all3': (m1.min(0)==1)&(s1.max(0)==0),
              'both_correct_all6': (s1.min(0)==1)&(m1.min(0)==1),
              'both_wrong_all6': (s1.max(0)==0)&(m1.max(0)==0)}
    connectivity = []
    for group in np.unique(groups):
        mask_g = groups == group
        any_s = bool(s1[:, mask_g].any()); any_m = bool(m1[:, mask_g].any())
        connectivity.append({'connectivity': str(group), 'query_count': int(mask_g.sum()),
            'any_sccf_correct_any_seed': any_s, 'any_msalign_correct_any_seed': any_m,
            'sccf_mean_r1_pct': float(s1[:, mask_g].mean()*100), 'msalign_mean_r1_pct': float(m1[:, mask_g].mean()*100),
            'matched_pair_oracle_r1_pct': float((1-matched[mask_g, 3]).mean()*100)})
    with (out / 'per_connectivity.jsonl').open('w') as f:
        for rec in connectivity: f.write(json.dumps(rec)+'\n')
    metadata = dict(np.load(E1 / 'assets/val.metadata.npz'))
    strata = {'adduct_H': metadata['adduct']==0, 'adduct_Na': metadata['adduct']==1,
              'CE_missing': np.isnan(metadata['collision_energy']), 'CE_present': ~np.isnan(metadata['collision_energy']),
              'cap256': lengths==256, 'below_cap256': lengths<256}
    summary = {'status': 'COMPLETE', 'primary_same_seed_mean': describe(matched, groups),
        'primary_bootstrap': bootstrap(matched, groups, c['bootstrap_replicates'], c['bootstrap_seed']),
        'prior_same_seed_mean': describe(prior, groups), 'per_seed': per_seed,
        'all_cross_seed_pairs': cross, 'within_method_seed_diversity': within,
        'stable_queries': {k: {'queries': int(v.sum()), 'connectivities_with_any': int(len(np.unique(groups[v])))} for k, v in stable.items()},
        'connectivity_any_query_any_seed_categories': {
            'both_have_some_correct': sum(x['any_sccf_correct_any_seed'] and x['any_msalign_correct_any_seed'] for x in connectivity),
            'only_sccf_has_some_correct': sum(x['any_sccf_correct_any_seed'] and not x['any_msalign_correct_any_seed'] for x in connectivity),
            'only_msalign_has_some_correct': sum(x['any_msalign_correct_any_seed'] and not x['any_sccf_correct_any_seed'] for x in connectivity),
            'both_all_wrong_all_queries_all_seeds': sum(not x['any_sccf_correct_any_seed'] and not x['any_msalign_correct_any_seed'] for x in connectivity)},
        'strata': {k: describe(matched[v], groups[v]) for k, v in strata.items()},
        'score_alignment_audit': score_checks, 'elapsed_seconds': time.time()-started,
        'contract_sha256': sha256_file(out/'contract.json'), 'test_evaluated': False, 'training_performed': False, 'fusion_performed': False,
        'scope': c['scope'], 'aggregation': c['primary'],
        'uncertainty_scope': 'query/cluster/balanced 10k bootstrap of mean joint contributions; no training/selection uncertainty or multiplicity adjustment',
        'prior_scope': 'SCCF full-train mean vs E1 full-train mean with fixed CE/adduct; reference inputs differ; descriptive association, not causal spectral attribution',
        'oracle_scope': 'GT-aware choice of one of two existing top1 predictions for each same-seed pair; not shortlist reranking, a trained gate, embedding fusion or actual ensemble'}
    np.savez(out/'per_query.npz', identifiers=ids, connectivities=groups, candidate_counts=lengths,
             sccf_metrics=sv, msalign_metrics=mv, sccf_prior_metrics=sp, msalign_prior_metrics=mp,
             matched_mean_joint=matched, prior_mean_joint=prior, **stable)
    write_json(out/'summary.json', summary)
    write_json(out/'complete_manifest.json', {'status':'COMPLETE','contract_sha256':sha256_file(out/'contract.json'),
        'summary_sha256':sha256_file(out/'summary.json'),'per_query_sha256':sha256_file(out/'per_query.npz'),
        'per_connectivity_sha256':sha256_file(out/'per_connectivity.jsonl'), 'test_evaluated':False,
        'training_performed':False,'fusion_performed':False})
    print('COMPLETE',json.dumps(summary['primary_same_seed_mean']),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run'))
    parser.add_argument('--out', type=Path, default=E1/'error_overlap_20261014')
    args = parser.parse_args()
    if args.action == 'freeze': freeze(args.out)
    else: run(args.out)


if __name__ == '__main__': main()
