#!/usr/bin/env python3
"""Frozen E1 metadata/constant-query audit; official train/val only, no fitting."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from glmr.embcos import sha256_file, write_json, summarize_metrics, uniform_tie_metrics
SEEDS = (20260621, 20260622, 20260623)
ARMS = {
    'mean_fixed_ce': ('mean', 'fixed', 'recipient'),
    'mean_fixed_adduct': ('mean', 'recipient', 'fixed'),
    'mean_fixed_both': ('mean', 'fixed', 'fixed'),
    'zero_fixed_both': ('zero', 'fixed', 'fixed'),
    'real_fixed_both': ('real', 'fixed', 'fixed'),
    'mean_metadata_permuted': ('mean', 'permuted', 'permuted'),
    'real_metadata_permuted': ('real', 'permuted', 'permuted'),
}
CONFIG = ROOT / 'docs/msalign/configs/external_matched_e0_20261014.json'


def guard():
    current = int(Path('/sys/fs/cgroup/memory.current').read_text())
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024
    assert current < 52 * 1024**3, ('cgroup safety gate', current)
    assert available > 64 * 1024**3, ('host-memory safety gate', available)
    return {'cgroup_gib': current / 1024**3, 'host_available_gib': available / 1024**3,
            'process_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def freeze(root, out):
    assert not out.exists(), 'Refusing to overwrite a frozen audit'
    train = dict(np.load(root / 'assets/train.metadata.npz', allow_pickle=False))
    val = dict(np.load(root / 'assets/val.metadata.npz', allow_pickle=False))
    # Most frequent JOINT train tuple; absent CE is represented explicitly.
    tuples = [(int(a), None if np.isnan(ce) else float(ce)) for a, ce in zip(train['adduct'], train['collision_energy'])]
    counts = Counter(tuples)
    modal, frequency = counts.most_common(1)[0]
    assert frequency > counts.most_common(2)[1][1], 'Joint mode must be unique'
    n = len(val['adduct'])
    order = np.random.default_rng(20261014).permutation(n)
    permutation = np.empty(n, np.int64)
    permutation[order] = np.roll(order, 1)
    assert np.array_equal(np.sort(permutation), np.arange(n)) and np.all(permutation != np.arange(n))
    files = [CONFIG, root / 'assets/manifest.json', root / 'assets/train.metadata.npz', root / 'assets/val.metadata.npz',
             ROOT / 'tools/stage1/run_msalign_official.py', ROOT / 'glmr/msalign_external.py',
             ROOT / 'glmr/embcos.py', Path(__file__), ROOT / 'tools/analysis/analyze_msalign_official.py',
             ROOT / 'outputs/embcos_official_e2_20261014/cache/val.rows.npz']
    for seed in SEEDS:
        base = root / f'seed{seed}'
        files.extend([base / 'checkpoint.pt', base / 'train_summary.json', base / 'val_ranks.npz',
                      base / 'controls/train_mean.npz', base / 'controls/zero.npz'])
    digests = {str(p.relative_to(ROOT)): sha256_file(p) for p in files}
    contract = {'status': 'FROZEN_BEFORE_INFERENCE', 'seeds': list(SEEDS), 'arms': ARMS,
        'fixed_tuple_rule': 'unique most frequent joint CE/adduct tuple in official train only',
        'fixed_adduct': modal[0], 'fixed_collision_energy': modal[1], 'train_mode_count': frequency,
        'train_rows': len(tuples), 'val_rows': n, 'metadata_permutation_seed': 20261014,
        'permutation_rule': 'one cycle through seeded shuffled row order; no fixed points; jointly permute CE/adduct',
        'bootstrap_replicates': 10000, 'bootstrap_seed': 20261014,
        'aggregation': 'mean of per-query contributions over three selected seeds, not ensemble',
        'protocol': 'frozen E1 official mass cap256; unchanged candidates, physical order and uniform exact ties',
        'interpretation': 'fixed-query scores test candidate-set/learned prior; metadata interventions are conditional frozen-model stress tests, not retrained causal ablations',
        'no_outcome_based_selection': True, 'test_evaluated': False, 'training_performed': False,
        'input_source_sha256': digests}
    out.mkdir(parents=True)
    np.savez(out / 'metadata_permutation.npz', donor_rows=permutation)
    contract['permutation_sha256'] = sha256_file(out / 'metadata_permutation.npz')
    write_json(out / 'contract.json', contract)
    print(json.dumps(contract), flush=True)


def verify_contract(root, out):
    c = json.loads((out / 'contract.json').read_text())
    assert c['arms'] == {name: list(v) for name, v in ARMS.items()}
    for path, digest in c['input_source_sha256'].items():
        assert sha256_file(ROOT / path) == digest, path
    assert sha256_file(out / 'metadata_permutation.npz') == c['permutation_sha256']
    return c


@contextmanager
def changed_metadata(assets, c, name, permutation):
    import torch
    original = assets.metadata['val']
    _, ce_mode, adduct_mode = ARMS[name]
    changed = {}
    for key, mode in [('collision_energy', ce_mode), ('adduct', adduct_mode)]:
        value = original[key]
        if mode == 'fixed':
            fixed = c['fixed_collision_energy'] if key == 'collision_energy' else c['fixed_adduct']
            changed[key] = torch.full_like(value, float('nan') if fixed is None else fixed)
        elif mode == 'permuted':
            changed[key] = value[torch.as_tensor(permutation[:len(value)], device=assets.device)]
        else:
            changed[key] = value
    assets.metadata['val'] = changed
    try:
        yield changed
    finally:
        assets.metadata['val'] = original


def validate_record(rec, rows):
    for key, source in [('identifiers', 'identifiers'), ('connectivities', 'connectivities'),
                        ('candidate_counts', 'lengths'), ('gt_slots', 'gt_slots')]:
        assert np.array_equal(rec[key], rows[source]), key
    assert np.array_equal(rec['recipient_rows'], np.arange(len(rec['identifiers'])))
    mask = np.arange(256)[None, :] < rec['candidate_counts'][:, None]
    scores = rec['scores']
    assert np.isfinite(scores[mask]).all() and np.isneginf(scores[~mask]).all()
    gt = scores[np.arange(len(scores)), rec['gt_slots']][:, None]
    assert np.array_equal(((scores > gt) & mask).sum(1), rec['greater'])
    assert np.array_equal(((scores == gt) & mask).sum(1), rec['equal'])
    assert np.array_equal(uniform_tie_metrics(rec['greater'], rec['equal']), rec['expected_metrics'])


def infer(root, out, smoke):
    import torch
    from glmr.msalign_external import MSAlign
    from tools.stage1.run_msalign_official import Assets, evaluate
    started = time.time()
    c = verify_contract(root, out)
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    guard()
    target = out / ('smoke' if smoke else 'inference')
    assert not target.exists(), 'No outcome-conditioned rerun/overwrite'
    target.mkdir()
    assets = Assets(root, CONFIG, torch.device('cuda'), val_limit=128 if smoke else 0)
    permutation = np.load(out / 'metadata_permutation.npz')['donor_rows']
    # A reduced cyclic permutation is plumbing-only; full run uses frozen mapping.
    if smoke:
        order = np.random.default_rng(20261014).permutation(128)
        permutation = np.empty(128, np.int64)
        permutation[order] = np.roll(order, 1)
    cfg = json.loads(CONFIG.read_text())
    records = {}
    for seed in SEEDS[:1] if smoke else SEEDS:
        guard()
        model = MSAlign(cfg['e1'], assets.input_dimensions).to(assets.device)
        checkpoint = torch.load(root / f'seed{seed}/checkpoint.pt', map_location=assets.device, weights_only=False)
        assert checkpoint['contract_sha256'] == sha256_file(CONFIG)
        assert checkpoint['cache_manifest_sha256'] == assets.input_digests['cache_manifest']
        assert checkpoint['runner_sha256'] == sha256_file(ROOT / 'tools/stage1/run_msalign_official.py')
        assert checkpoint['module_sha256'] == sha256_file(ROOT / 'glmr/msalign_external.py')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        del checkpoint
        projection = assets.molecules_for_evaluation(model)
        _, real = evaluate(model, assets, projected=projection)
        frozen = dict(np.load(root / f'seed{seed}/val_ranks.npz'))
        if not smoke:
            for key in ('scores', 'greater', 'equal', 'ranks', 'gt_slots'):
                assert np.array_equal(real[key], frozen[key]), ('exact frozen replay', seed, key)
        else:
            assert np.array_equal(real['greater'], frozen['greater'][:128])
            assert np.array_equal(real['equal'], frozen['equal'][:128])
        del frozen, real
        seed_out = target / f'seed{seed}'
        seed_out.mkdir()
        per_seed = {}
        for name, (spectrum, _, _) in ARMS.items():
            t = time.time()
            constant = assets.train_mean if spectrum == 'mean' else torch.zeros_like(assets.train_mean) if spectrum == 'zero' else None
            with changed_metadata(assets, c, name, permutation) as metadata:
                metrics, rec = evaluate(model, assets, projected=projection, constant=constant)
                rec.update({f'input_{k}': v.cpu().numpy() for k, v in metadata.items()})
            validate_record(rec, assets.rows['val'])
            invariant = None
            if name in ('mean_fixed_both', 'zero_fixed_both'):
                # Same molecule must receive the same score across all query pools.
                idx = assets.eval_indices.cpu().numpy()
                mask = assets.eval_mask.cpu().numpy()
                lo = np.full(len(projection), np.inf, np.float32)
                hi = np.full(len(projection), -np.inf, np.float32)
                np.minimum.at(lo, idx[mask], rec['scores'][mask])
                np.maximum.at(hi, idx[mask], rec['scores'][mask])
                spread = float((hi - lo).max())
                assert spread <= 1e-5, ('candidate-only invariance', spread)
                invariant = {'max_same_effective_molecule_score_spread': spread, 'tolerance': 1e-5}
            path = seed_out / f'{name}.npz'
            np.savez(path, **rec)
            per_seed[name] = {'metrics': metrics, 'sha256': sha256_file(path),
                              'elapsed_seconds': time.time() - t, 'candidate_only_invariance': invariant,
                              'resources': guard()}
            print(json.dumps({'seed': seed, 'arm': name, **per_seed[name]}), flush=True)
            del rec
        records[str(seed)] = per_seed
        del model, projection
        torch.cuda.empty_cache()
    write_json(target / 'complete.json', {'status': 'PLUMBING_ONLY' if smoke else 'COMPLETE',
        'seeds': records, 'frozen_replay': 'exact full scores/counts/ranks' if not smoke else '128-query tie counts',
        'elapsed_seconds': time.time() - started, 'cuda_peak_mib': torch.cuda.max_memory_allocated() / 1024**2,
        'resources': guard(), 'contract_sha256': sha256_file(out / 'contract.json'),
        'test_evaluated': False, 'training_performed': False})


def analyze(root, out):
    from tools.analysis.analyze_msalign_official import intervals
    started = time.time()
    c = verify_contract(root, out)
    complete = json.loads((out / 'inference/complete.json').read_text())
    assert complete['status'] == 'COMPLETE' and complete['contract_sha256'] == sha256_file(out / 'contract.json')
    baseline = dict(np.load(root / 'analysis/seed_mean_per_query.npz'))
    groups = baseline['connectivities']
    means = {'real': baseline['msalign'], 'exact_random': baseline['exact_random']}
    digests = {}
    for name in ('train_mean', 'zero', *ARMS):
        vectors = []
        for seed in SEEDS:
            path = root / f'seed{seed}/controls/{name}.npz' if name in ('train_mean', 'zero') else out / f'inference/seed{seed}/{name}.npz'
            if name in ARMS:
                assert sha256_file(path) == complete['seeds'][str(seed)][name]['sha256']
            with np.load(path, allow_pickle=False) as z:
                assert np.array_equal(z['identifiers'], baseline['identifiers'])
                assert np.array_equal(z['connectivities'], groups)
                assert np.array_equal(z['candidate_counts'], baseline['candidate_counts'])
                vectors.append(z['expected_metrics'])
            digests[str(path.relative_to(ROOT))] = sha256_file(path)
        means[name] = np.stack(vectors).mean(0)
    results = {name: summarize_metrics(value, groups) for name, value in means.items()}
    pairs = [('mean_fixed_both', 'exact_random'), ('zero_fixed_both', 'exact_random'),
             ('real', 'train_mean'), ('real', 'real_fixed_both'), ('real', 'real_metadata_permuted'),
             ('train_mean', 'mean_fixed_both'), ('mean_fixed_ce', 'mean_fixed_both'),
             ('mean_fixed_adduct', 'mean_fixed_both'), ('train_mean', 'mean_metadata_permuted')]
    contrasts = {}
    for a, b in pairs:
        contrasts[f'{a}_minus_{b}'] = intervals(means[a] - means[b], groups, c['bootstrap_replicates'], c['bootstrap_seed'])
        print(json.dumps({'paired_analysis_complete': f'{a}_minus_{b}'}), flush=True)
    metadata = dict(np.load(root / 'assets/val.metadata.npz'))
    lengths = baseline['candidate_counts']
    masks = {'adduct_H': metadata['adduct'] == 0, 'adduct_Na': metadata['adduct'] == 1,
             'CE_missing': np.isnan(metadata['collision_energy']), 'CE_present': ~np.isnan(metadata['collision_energy']),
             'candidate_cap256': lengths == 256, 'candidate_below256': lengths < 256}
    strata = {label: {name: summarize_metrics(value[mask], groups[mask]) for name, value in means.items()}
              for label, mask in masks.items() if mask.any()}
    permutation = np.load(out / 'metadata_permutation.npz')['donor_rows']
    ce = metadata['collision_energy']
    metadata_changes = {'joint_adduct_changed': int((metadata['adduct'] != metadata['adduct'][permutation]).sum()),
        'CE_changed_nan_equal': int((~((ce == ce[permutation]) | (np.isnan(ce) & np.isnan(ce[permutation])))).sum()),
        'different_connectivity': int((groups != groups[permutation]).sum())}
    result = {'status': 'COMPLETE', 'metrics': results, 'paired_contrasts': contrasts, 'strata': strata,
        'metadata_permutation_changes': metadata_changes, 'artifact_sha256': digests,
        'contract_sha256': sha256_file(out / 'contract.json'), 'inference_complete_sha256': sha256_file(out / 'inference/complete.json'),
        'baseline_means_sha256': sha256_file(root / 'analysis/seed_mean_per_query.npz'),
        'elapsed_seconds': time.time() - started, 'test_evaluated': False, 'training_performed': False,
        'scope': 'post-hoc validation development audit conditional on three selected E1 checkpoints; no multiple-comparison correction; bootstrap omits selection and training uncertainty; fixed/perm metadata stress is not a retraining ablation'}
    write_json(out / 'summary.json', result)
    np.savez(out / 'seed_mean_per_query.npz', identifiers=baseline['identifiers'], connectivities=groups, candidate_counts=lengths, **means)
    print(json.dumps({'status': 'COMPLETE', 'metrics': results, 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('freeze', 'smoke', 'infer', 'analyze'))
    p.add_argument('--root', type=Path, default=ROOT / 'outputs/msalign_official_e1_20261014')
    p.add_argument('--out', type=Path, default=ROOT / 'outputs/msalign_official_e1_20261014/prior_followup_20261014')
    args = p.parse_args()
    if args.action == 'freeze': freeze(args.root, args.out)
    elif args.action == 'analyze': analyze(args.root, args.out)
    else: infer(args.root, args.out, smoke=args.action == 'smoke')


if __name__ == '__main__':
    main()
