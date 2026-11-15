#!/usr/bin/env python3
"""Freeze and evaluate the approved R6-lite CPU gate; never open official test."""
from __future__ import annotations

import collections
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import torch
from rdkit import rdBase
from glmr.candidate_indices import CandidateIndexStore
from tools.analysis.analyze_sccf_wrong_spectrum_proximity import (
    normalized_raw_features, DEFAULT_FP, DEFAULT_CANDIDATES, DEFAULT_TRAIN,
)
from tools.stage1.train_msalign_projection_smoke import MoleculeEmbeddingStore
from tools.audits.audit_spectrum_permutation import connectivity_identities
from tools.analysis.supplement_official_s7 import guard
from tools.evaluation.official_s7_common import (
    sha, read, write, load, install_no_test_guard, GUARD_EVENTS,
)

OUT = ROOT / 'outputs/revision_supplement_20261111/R6_lite/validation/gate'
VAL = ROOT / 'outputs/msalign_stage1_full_v15/val_full.metadata.jsonl'
ROWS = ROOT / 'outputs/embcos_official_e2_20261014/cache/val.rows.npz'
REFERENCE = ROOT / 'outputs/revision_supplement_20261111/R1b/validation/proximity.npz'
SEED = 20261111
BLOCKS = ['count_morgan', 'count_atom_pair', 'rdkit_path_count', 'maccs']


def entry(path):
    path = Path(path)
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def main():
    install_no_test_guard()
    torch.set_num_threads(2)
    assert rdBase.rdkitVersion == '2025.09.6', rdBase.rdkitVersion
    assert not (OUT / 'summary.json').exists(), 'Gate already frozen; do not overwrite'
    started = time.time()
    resources = {'before': guard()}
    assert shutil.disk_usage(ROOT).free >= 5 * 2**30, 'disk reserve'
    train = [json.loads(x) for x in DEFAULT_TRAIN.read_text().splitlines()]
    val = [json.loads(x) for x in VAL.read_text().splitlines()]
    tg, ta = connectivity_identities(train)
    vg, va = connectivity_identities(val)
    assert not ta['invalid_smiles_fallbacks'] and not va['invalid_smiles_fallbacks']
    rows, reference = load(ROWS), load(REFERENCE)
    ids = np.asarray([x['identifier'] for x in val])
    vg = np.asarray(vg)
    assert np.array_equal(ids, rows['identifiers'])
    assert np.array_equal(vg, rows['connectivities'])
    assert np.array_equal(ids, reference['identifiers'])
    assert np.array_equal(vg, reference['connectivities'])
    index = CandidateIndexStore(DEFAULT_CANDIDATES)
    train_positive = np.asarray([index.get(q['identifier'], 256)[0] for q in train], np.int64)
    val_positive = rows['candidates'][:, 0].astype(np.int64)
    assert all(index.get(q['identifier'], 256)[0] == r for q, r in zip(val, val_positive))
    tr, ti = np.unique(train_positive, return_inverse=True)
    vr, vi = np.unique(val_positive, return_inverse=True)
    tk = {}
    vk = {}
    for row, key in zip(train_positive, tg):
        assert tk.setdefault(int(row), key) == key
    for row, key in zip(val_positive, vg):
        assert vk.setdefault(int(row), key) == key
    train_groups = np.asarray([tk[int(x)] for x in tr])
    val_groups = np.asarray([vk[int(x)] for x in vr])
    assert not set(train_groups) & set(val_groups), 'official connectivity split'
    assert len(tr) == 25046 and len(vr) == 3386 and len(val) == 19429
    store = MoleculeEmbeddingStore(None, 'mean', DEFAULT_FP,
        preload_sparse_features=False, mol_blocks=BLOCKS)
    def features(ix):
        chunks = []
        for start in range(0, len(ix), 256):
            chunks.append(normalized_raw_features(store, ix[start:start+256].tolist(), 'block_l2'))
        return torch.cat(chunks)
    tf, vf = features(tr), features(vr)
    resources['after_features'] = guard()
    with torch.inference_mode():
        similarities = (vf @ tf.T).numpy()
    p0 = similarities.max(axis=1)
    error = float(np.max(np.abs(p0[vi] - reference['proximity'])))
    assert error <= 1e-6, ('baseline proximity reconstruction', error)
    print('BASELINE', len(tr), len(vr), len(set(vg)), 'max_error', error, flush=True)
    groups = np.unique(vg)
    molecular_indices = {key: np.flatnonzero(val_groups == key) for key in groups}
    eligible = np.asarray([key for key in groups if np.min(p0[molecular_indices[key]]) >= .90])
    np.random.default_rng(SEED).shuffle(eligible)
    cohort = eligible[:200]
    assert len(cohort) == 200, ('insufficient eligible cohort', len(eligible))
    a_rows = np.isin(val_groups, cohort)
    removed_groups = np.unique(train_groups[(similarities[a_rows] >= .80).any(axis=0)])
    targeted_removed = np.isin(tg, removed_groups)
    count = int(targeted_removed.sum())
    # Safety is connectivity-wide: no molecular row may be close to any val row.
    unsafe_groups = np.unique(train_groups[(similarities >= .80).any(axis=0)])
    eligible_spectra = np.flatnonzero(~np.isin(tg, unsafe_groups))
    assert len(eligible_spectra) >= count, ('insufficient random-removal pool', count)
    random_removed = np.zeros(len(train), dtype=bool)
    random_removed[np.random.default_rng(SEED).choice(eligible_spectra, count, replace=False)] = True
    assert targeted_removed.sum() == random_removed.sum()
    def remaining_rows(removed):
        counts = np.bincount(ti[~removed], minlength=len(tr))
        return counts > 0
    pT = similarities[:, remaining_rows(targeted_removed)].max(axis=1)
    pR = similarities[:, remaining_rows(random_removed)].max(axis=1)
    clean = np.asarray([key for key in groups if key not in set(cohort)
        and np.all(np.abs(pT[molecular_indices[key]]-p0[molecular_indices[key]]) <= 1e-6)
        and np.all(np.abs(pR[molecular_indices[key]]-p0[molecular_indices[key]]) <= 1e-6)])
    spill = np.setdiff1d(groups, np.concatenate([cohort, clean]))
    mean0 = np.asarray([p0[molecular_indices[key]].mean(dtype=np.float64) for key in groups])
    meanT = np.asarray([pT[molecular_indices[key]].mean(dtype=np.float64) for key in groups])
    meanR = np.asarray([pR[molecular_indices[key]].mean(dtype=np.float64) for key in groups])
    a_mask = np.isin(groups, cohort)
    median_drop = float(np.median(mean0[a_mask] - meanT[a_mask]))
    passed = median_drop >= .10 - 1e-6 and len(clean) >= 1500
    def removal_report(removed):
        touched = set(np.asarray(tg)[removed])
        retained = set(np.asarray(tg)[~removed])
        return {'removed_spectra': int(removed.sum()), 'remaining_spectra': int((~removed).sum()),
            'touched_connectivities': len(touched), 'fully_removed_connectivities': len(touched-retained),
            'remaining_positive_molecular_rows': int(remaining_rows(removed).sum())}
    def group_report(keys):
        mask = np.isin(groups, keys)
        return {'connectivities': len(keys), 'queries': int(np.isin(vg, keys).sum()),
            'baseline_proximity_median': float(np.median(mean0[mask])),
            'targeted_drop_quantiles': np.quantile(mean0[mask]-meanT[mask], [0,.25,.5,.75,1]).tolist(),
            'random_drop_quantiles': np.quantile(mean0[mask]-meanR[mask], [0,.25,.5,.75,1]).tolist()}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/'cohort_and_masks.npz', identifiers=ids, connectivities=vg,
        train_identifiers=np.asarray([q['identifier'] for q in train]), train_connectivities=np.asarray(tg),
        train_positive_rows=train_positive, train_targeted_removed=targeted_removed,
        train_size_matched_removed=random_removed, validation_positive_rows=val_positive,
        unique_validation_positive_rows=vr, unique_validation_connectivities=val_groups,
        p0=p0, p_targeted=pT, p_size_matched=pR,
        connectivity_keys=groups, connectivity_p0=mean0, connectivity_p_targeted=meanT,
        connectivity_p_size_matched=meanR, A=cohort, B_clean=clean, B_spill=spill,
        eligible_A_connectivities=eligible, random_eligible_train_rows=eligible_spectra)
    # Both arm inputs are paired to spectrum rows in original order, including duplicates.
    # The runner can consume these small masked metadata/embedding caches unchanged.
    train_dreams_path = ROOT/'outputs/run_k_official_datasafe_seed20260621/train.no_head.npy'
    dreams = np.load(train_dreams_path, mmap_mode='r')
    assert len(dreams) == len(train)
    if passed:
        for arm, removed in [('targeted', targeted_removed), ('size_matched', random_removed)]:
            keep = np.flatnonzero(~removed)
            with (OUT/f'{arm}.metadata.jsonl').open('w') as handle:
                for i in keep:
                    handle.write(json.dumps(train[int(i)], ensure_ascii=False)+'\n')
            output = np.lib.format.open_memmap(OUT/f'{arm}.dreams.npy', mode='w+',
                dtype=dreams.dtype, shape=(len(keep), dreams.shape[1]))
            for start in range(0,len(keep),4096):
                output[start:start+4096] = dreams[keep[start:start+4096]]
            output.flush()
            del output
            check = np.load(OUT/f'{arm}.dreams.npy', mmap_mode='r')
            for start in range(0,len(keep),4096):
                assert np.array_equal(check[start:start+4096], dreams[keep[start:start+4096]])
            print('MASKED_INPUT', arm, len(keep), flush=True)
    sources = [Path(__file__), ROOT/'to_be_run.md', DEFAULT_TRAIN, VAL, ROWS, REFERENCE,
        DEFAULT_CANDIDATES, DEFAULT_FP, train_dreams_path,
        ROOT/'outputs/run_k_official_datasafe_seed20260621/val.no_head.npy',
        ROOT/'tools/stage1/train_msalign_projection_smoke.py',
        ROOT/'tools/analysis/analyze_sccf_wrong_spectrum_proximity.py',
        ROOT/'tools/audits/audit_spectrum_permutation.py']
    manifest = read(DEFAULT_FP)
    for block in BLOCKS:
        for key in ('indices','values','indptr'):
            sources.append(DEFAULT_FP.parent/manifest['features'][block][key])
    sources.append(ROOT/manifest['base_cache'])
    for key in ('indices','sources'):
        sources.append(DEFAULT_FP.parent/manifest['projection_order'][key])
    sources += [ROOT/'outputs/stage1_submission_0926/assets/heldout_connectivity_exclusion/manifest.json']
    inputs = []
    for path in dict.fromkeys(sources):
        inputs.append(entry(path))
        print('BOUND', inputs[-1]['path'], flush=True)
    outputs = [entry(path) for path in sorted(OUT.iterdir()) if path.is_file()]
    resources['after'] = guard()
    result = dict(status='PASS' if passed else 'STOP_GATE_FAILED',
        time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        gate_criteria={'minimum_A_eligible_proximity_every_GT_row': .90, 'A_connectivities':200,
            'removal_cosine_threshold':.80, 'minimum_A_median_drop':.10,
            'minimum_B_clean_connectivities':1500, 'absolute_tolerance':1e-6},
        cohort_seed=SEED, random_removal_seed=SEED, fixed_fit_seeds=[20260621,20260622],
        bootstrap_draws=10000, bootstrap_seed=SEED, rdkit_version=rdBase.rdkitVersion,
        eligible_A_connectivities=len(eligible), A_median_proximity_drop=median_drop,
        cohorts={name:group_report(keys) for name,keys in [('A',cohort),('B_clean',clean),('B_spill',spill)]},
        removals={'targeted':removal_report(targeted_removed),'size_matched':removal_report(random_removed)},
        random_removal_eligible_spectra=len(eligible_spectra), baseline_max_abs_error=error,
        train_identity_audit=ta, validation_identity_audit=va,
        original_train_spectra=len(train), original_train_connectivities=len(set(tg)),
        original_train_positive_molecular_rows=len(tr), validation_queries=len(val),
        inputs=inputs, outputs=outputs, resources=resources,
        elapsed_seconds=time.time()-started, command=sys.argv, forbidden_read_attempts=GUARD_EVENTS)
    write(OUT/'summary.json', result)
    print('GATE', result['status'], 'A_drop', median_drop, 'B_clean',len(clean),
        'removed',count, 'elapsed',result['elapsed_seconds'], flush=True)

if __name__ == '__main__':
    main()
