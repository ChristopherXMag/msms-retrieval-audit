#!/usr/bin/env python3
"""Additive R6 integrity audit, with an independent sibling-expanded bootstrap check."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.evaluation.official_s7_common import read,write,load,sha,install_no_test_guard,GUARD_EVENTS
from tools.analysis.supplement_official_s7 import guard
from tools.analysis.revision_r6_memory import trim_store
from glmr.embcos import uniform_tie_metrics

OUT=ROOT/'outputs/revision_supplement_20261111/R6_lite/validation'
SEEDS=(20260621,20260622)
ARMS=('targeted','size_matched')
LABELS=('seed20260621','seed20260622','two_seed_mean')
WEIGHTS=('query_weighted','connectivity_balanced')
METRICS=('R@1','R@5','R@20','MRR')


def digest(path):
    value=sha(path)
    # Hashing large immutable inputs should not retain a second hot file cache.
    if Path(path).stat().st_size>512*2**20:
        fd=os.open(path,os.O_RDONLY)
        try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
        finally:os.close(fd)
    return value


def main():
    install_no_test_guard();trim_store();before=guard();started=time.time()
    result=read(OUT/'analysis/summary.json');assert result['status']=='COMPLETE'
    execution=read(OUT/'training_execution.json');assert execution['status']=='COMPLETE'
    assert {(x['seed'],x['arm']) for x in execution['completed']}=={(s,a) for s in SEEDS for a in ARMS}
    gate=read(OUT/'gate/summary.json');assert gate['status']=='PASS'
    masks=load(OUT/'gate/cohort_and_masks.npz')
    assert masks['train_targeted_removed'].sum()==masks['train_size_matched_removed'].sum()==17504
    assert not (masks['train_targeted_removed']&masks['train_size_matched_removed']).any()
    assert len(masks['A'])==200 and len(masks['B_clean'])>=1500
    assert np.array_equal(np.sort(np.concatenate([masks['A'],masks['B_clean'],masks['B_spill']])),masks['connectivity_keys'])
    assert np.max(masks['p_targeted'][np.isin(masks['unique_validation_connectivities'],masks['A'])])<.80
    archive=load(OUT/'analysis/per_query_and_bootstrap.npz')
    assert digest(OUT/'analysis/per_query_and_bootstrap.npz')==result['archive_sha256']
    assert np.array_equal(archive['identifiers'],masks['identifiers'])
    assert np.array_equal(archive['connectivities'],masks['connectivities'])
    assert np.array_equal(archive['paired_differences'],archive['targeted']-archive['size_matched'])
    historical={}
    for path in (OUT/'provenance').iterdir():
        if path.is_file():historical.setdefault(digest(path),path)
    verified_inputs={}
    def verify_binding(path,expected):
        path=Path(path)
        actual=digest(path)
        if actual!=expected:
            assert str(path.relative_to(ROOT)) in ('to_be_run.md','tools/analysis/revision_r6_train.py'),path
            assert expected in historical,('Missing exact historical execution snapshot',path,expected)
            stored=historical[expected]
            verified_inputs[str(path.relative_to(ROOT))+' [historical snapshot]']=dict(sha256=expected,snapshot=str(stored.relative_to(ROOT)))
        else:verified_inputs[str(path.relative_to(ROOT))]=dict(sha256=actual,bytes=path.stat().st_size)
    for item in gate['inputs']:
        verify_binding(ROOT/item['path'],item['sha256'])
        guard()
    for item in gate['outputs']:assert digest(ROOT/item['path'])==item['sha256']
    for path,expected in execution['source_bindings'].items():verify_binding(ROOT/path,expected)
    for path,expected in execution.get('recovery',{}).get('source_bindings',{}).items():verify_binding(ROOT/path,expected)
    for path,entry in read(OUT/'provenance/shared_asset_bindings.json')['files'].items():verify_binding(ROOT/path,entry['sha256'])
    for path,expected in result['inputs'].items():verify_binding(ROOT/path,expected)
    assert digest(ROOT/'tools/analysis/revision_r6_analyze.py')==result['source_sha256']
    assert digest(ROOT/'tools/analysis/revision_r6_memory.py')==read(OUT/'provenance/cache_residency_smoke.json')['source_sha256']
    frozen_args=read(ROOT/'outputs/stage1_submission_0926/runs/full_r2_block_l2/seed20260621/train_summary.json')['args']
    fixed_keys=['epochs','batch_size','eval_batch_size','lr','weight_decay','head_type','hidden_dim','proj_dim',
        'dropout','temperature','freeze_temperature','train_sampling','mol_head_type','mol_block_normalization',
        'mol_blocks','negative_sampling','loss_mode','candidate_pool_limit','candidate_negatives','negative_exclusion_manifest',
        'train_limit','val_limit','resume_checkpoint','materialize_dataset_features','mol_pooling']
    for seed in SEEDS:
        for a,arm in enumerate(ARMS):
            directory=OUT/f'seed{seed}'/arm
            receipt=read(directory/'fit_complete.json');summary=read(directory/'train_summary.json')
            assert digest(directory/'checkpoint.pt')==receipt['checkpoint_sha256']
            assert digest(directory/'train_summary.json')==receipt['summary_sha256']
            assert [row['epoch'] for row in summary['history'] if 'train' in row]==[1]
            assert summary['train_sampling_protocol']['rows_per_epoch']==176615
            for key in fixed_keys:assert summary['args'][key]==frozen_args[key],(seed,arm,key)
            assert summary['args']['seed']==seed
            assert summary['args']['preload_sparse_features'] is False
            command=read(directory/'command.json')
            assert command['source_sha256'] in historical or command['source_sha256']==digest(ROOT/'tools/analysis/revision_r6_train.py')
            validation=read(directory/'validation_summary.json')
            assert validation['state_key']=='model_state_dict' and validation['last_epoch']==1
            assert not validation['forbidden_read_attempts']
            assert validation['source_sha256']==digest(ROOT/'tools/analysis/revision_r6_evaluate.py')
            q=load(directory/'validation_per_query.npz')
            assert digest(directory/'validation_per_query.npz')==validation['per_query_sha256']
            assert np.array_equal(q['identifiers'],masks['identifiers'])
            targets=q['scores'][np.arange(len(q['scores'])),q['gt_slots'],None]
            greater=(q['scores']>targets).sum(1);equal=(q['scores']==targets).sum(1)
            assert np.array_equal(greater,q['greater']) and np.array_equal(equal,q['equal'])
            assert np.array_equal(uniform_tie_metrics(greater,equal),q['expected_metrics'])
            assert np.array_equal(q['expected_metrics'],archive[arm][SEEDS.index(seed)])
    # Independent point estimates operate directly on all sibling queries.
    d=np.concatenate([archive['paired_differences'],archive['paired_differences'].mean(0,keepdims=True)])
    g=archive['connectivities'];keys=np.unique(g)
    siblings={key:np.flatnonzero(g==key) for key in keys}
    cohort_keys={name:sorted(masks[name].tolist()) for name in ('A','B_clean')}
    def direct(selected):
        ix=np.concatenate([siblings[key] for key in selected])
        query=d[:,ix].mean(1)
        balanced=np.stack([d[:,siblings[key]].mean(1) for key in selected]).mean(0)
        return np.stack([query,balanced],axis=1)
    direct_a,direct_b=direct(cohort_keys['A']),direct(cohort_keys['B_clean'])
    point_map={'A_targeted_minus_random':direct_a,'B_clean_targeted_minus_random':direct_b,
        'difference_in_differences':direct_a-direct_b}
    for contrast,point in point_map.items():
        intervals=np.quantile(archive['bootstrap_'+contrast],[.025,.975],axis=0)
        for s,label in enumerate(LABELS):
            for w,weight in enumerate(WEIGHTS):
                for m,metric in enumerate(METRICS):
                    entry=result['contrasts'][contrast][label][weight][metric]
                    assert np.isclose(entry['estimate_pp'],100*point[s,w,m],rtol=0,atol=1e-11)
                    assert np.allclose(entry['ci95_pp'],100*intervals[:,s,w,m],rtol=0,atol=1e-11)
    # Reconstruct the first 16 stratified draws without connectivity sufficient statistics.
    rng=np.random.default_rng(20261111)
    ka,kb=cohort_keys['A'],cohort_keys['B_clean']
    ia=rng.integers(0,len(ka),size=(16,len(ka)))
    ib=rng.integers(0,len(kb),size=(16,len(kb)))
    for draw in range(16):
        actual=direct([ka[i] for i in ia[draw]])-direct([kb[i] for i in ib[draw]])
        assert np.allclose(actual,archive['bootstrap_difference_in_differences'][draw],rtol=0,atol=1e-13)
    base_note=subprocess.check_output(['git','show','14c9298:docs/reranking/REVISION_SUPPLEMENT_20261111.md'],cwd=ROOT)
    assert (ROOT/'docs/reranking/REVISION_SUPPLEMENT_20261111.md').read_bytes().startswith(base_note)
    original_contract=subprocess.check_output(['git','show','14c9298:to_be_run.md'],cwd=ROOT)
    assert (OUT/'provenance/contract_execution.md').read_bytes()==original_contract
    # The failed size-matched startup never entered training and has no checkpoint.
    attempt=OUT/'seed20260621/size_matched/attempt0_before_training'
    assert 'Training projection heads' not in (attempt/'fit.log').read_text()
    assert not (attempt/'checkpoint.pt').exists()
    assert len(list(OUT.glob('seed*/targeted/checkpoint.pt')))+len(list(OUT.glob('seed*/size_matched/checkpoint.pt')))==4
    assert not gate['forbidden_read_attempts'] and not result['forbidden_read_attempts']
    inventory={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=digest(p))
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('inventory.json','verification.json')}
    write(OUT/'analysis/inventory.json',dict(status='COMPLETE',inputs=verified_inputs,outputs=inventory,
        new_output_bytes=sum(x['bytes'] for x in inventory.values()),source_sha256=digest(Path(__file__))))
    checks=['CPU gate, disjoint frozen masks and selected cohort','exactly four completed one-epoch fits; original scientific arguments unchanged',
        'all validation scores and expected tie contributions','independent query/connectivity point estimates and all saved percentile intervals',
        '16 bootstrap draws independently expanded to all sibling queries','exact source snapshots for operational recoveries',
        'original result-note prefix and launch contract preserved','failed startup had no training; completed checkpoints never retrained',
        'no forbidden test-file operations in gate/evaluation/analysis']
    write(OUT/'analysis/verification.json',dict(status='PASS',checks=checks,input_bindings=len(verified_inputs),
        output_files=len(inventory),elapsed_seconds=time.time()-started,resources_before=before,resources_after=guard(),
        forbidden_read_attempts=GUARD_EVENTS,inventory_sha256=digest(OUT/'analysis/inventory.json')))
    print('PASS',len(checks),'checks;',len(verified_inputs),'input bindings;',len(inventory),'output files',flush=True)

if __name__=='__main__':main()
