#!/usr/bin/env python3
"""Export paired R6-lite validation scores with physically shuffled candidates."""
from __future__ import annotations
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import torch
from tools.evaluation.official_s7_common import (
    read,write,load,sha,gpu_preflight,install_no_test_guard,GUARD_EVENTS,
)
from tools.analysis.supplement_official_s7 import guard
from tools.stage1.train_msalign_projection_smoke import (
    MoleculeEmbeddingStore,build_projection_model,checkpoint_model_config,
)
from glmr.embcos import uniform_tie_metrics,summarize_metrics

OUT=ROOT/'outputs/revision_supplement_20261111/R6_lite/validation'
ROWS=ROOT/'outputs/embcos_official_e2_20261014/cache/val.rows.npz'
DREAMS=ROOT/'outputs/run_k_official_datasafe_seed20260621/val.no_head.npy'
FP=ROOT/'outputs/stage1_submission_0926/assets/full_sparse_direct/manifest_historical_projection_order.json'
SEEDS=(20260621,20260622)
ARMS=('targeted','size_matched')


def main():
    install_no_test_guard()
    assert read(OUT/'gate/summary.json')['status']=='PASS'
    assert read(OUT/'training_execution.json')['status']=='COMPLETE'
    from tools.analysis.revision_r6_memory import install,trim_store
    trim_store();install(OUT/'analysis')
    gpu_preflight(OUT/'analysis','before_validation_export')
    rows=load(ROWS);dreams=np.load(DREAMS,mmap_mode='r')
    assert len(dreams)==len(rows['identifiers'])==19429
    gate=load(OUT/'gate/cohort_and_masks.npz')
    assert np.array_equal(rows['identifiers'],gate['identifiers'])
    assert np.array_equal(rows['connectivities'],gate['connectivities'])
    store=MoleculeEmbeddingStore(None,'mean',FP,preload_sparse_features=False,
        mol_blocks=['count_morgan','count_atom_pair','rdkit_path_count','maccs'])
    for seed in SEEDS:
        for arm in ARMS:
            directory=OUT/f'seed{seed}'/arm
            assert not (directory/'validation_summary.json').exists(), 'Do not overwrite an evaluation'
            start=time.time();guard()
            gpu_preflight(directory,'before_validation',require_idle=False)
            checkpoint=directory/'checkpoint.pt'
            state=torch.load(str(checkpoint),map_location='cpu',weights_only=False,mmap=True)
            assert state['last_epoch']==1 and state['args']['seed']==seed
            assert sha(checkpoint)==read(directory/'fit_complete.json')['checkpoint_sha256']
            model=build_projection_model(spec_dim=1024,mol_dim=store.dim,
                mol_block_dims=store.block_dims,mol_block_names=store.block_names,
                config=checkpoint_model_config(state)).to('cuda')
            model.load_state_dict(state['model_state_dict']);model.eval()
            scores=np.full(rows['permuted_candidates'].shape,-np.inf,np.float32)
            greater=np.empty(len(dreams),np.int32);equal=greater.copy()
            with torch.inference_mode():
                for start_row in range(0,len(dreams),64):
                    guard()
                    stop=min(start_row+64,len(dreams))
                    candidates=rows['permuted_candidates'][start_row:stop].copy().astype(np.int64)
                    mask=candidates>=0
                    slots=rows['gt_slots'][start_row:stop]
                    gt=rows['candidates'][start_row:stop,0]
                    assert np.array_equal(candidates[np.arange(stop-start_row),slots],gt)
                    candidates[~mask]=np.broadcast_to(gt[:,None],candidates.shape)[~mask]
                    ix=torch.from_numpy(candidates)
                    unique,inverse=torch.unique(ix.reshape(-1),sorted=True,return_inverse=True)
                    unique,inverse=store.canonicalize_unique_indices(unique,inverse)
                    molecules=model.project_mol(store.gather(unique).to('cuda'))
                    molecules=molecules[inverse.to('cuda')].reshape(len(ix),ix.shape[1],-1)
                    spectrum=model.project_spec(torch.from_numpy(np.array(dreams[start_row:stop],copy=True)).to('cuda'))
                    logits=(torch.einsum('bd,bcd->bc',spectrum,molecules)/model.temperature()).cpu().numpy()
                    assert np.isfinite(logits[mask]).all()
                    logits[~mask]=-np.inf
                    scores[start_row:stop]=logits
                    target=logits[np.arange(len(ix)),slots,None]
                    greater[start_row:stop]=((logits>target)&mask).sum(1)
                    equal[start_row:stop]=((logits==target)&mask).sum(1)
                    if start_row%2048==0:print('VALIDATION',seed,arm,start_row,flush=True)
            values=uniform_tie_metrics(greater,equal)
            assert np.isfinite(values).all() and ((values>=0)&(values<=1)).all()
            # Validate every target's membership and tie counts, independent of its shuffled slot.
            assert (equal>=1).all()
            np.savez_compressed(directory/'validation_per_query.npz',identifiers=rows['identifiers'],
                connectivities=rows['connectivities'],scores=scores,greater=greater,equal=equal,
                expected_metrics=values,candidate_counts=rows['lengths'],gt_slots=rows['gt_slots'])
            summary=dict(status='COMPLETE',seed=seed,arm=arm,last_epoch=1,state_key='model_state_dict',
                candidate_order='Physically permuted, frozen E2 val.rows; GT slots explicit',
                ties='Uniform exact-tie expectation',metrics=summarize_metrics(values,rows['connectivities']),
                exact_GT_ties=int((equal>1).sum()),elapsed_seconds=time.time()-start,
                checkpoint_sha256=sha(checkpoint),rows_sha256=sha(ROWS),dreams_sha256=sha(DREAMS),
                fingerprint_manifest_sha256=sha(FP),source_sha256=sha(Path(__file__)),
                per_query_sha256=sha(directory/'validation_per_query.npz'),
                resources=guard(),forbidden_read_attempts=list(GUARD_EVENTS))
            write(directory/'validation_summary.json',summary)
            print('EVALUATED',seed,arm,summary['metrics'],flush=True)
            del model,state,scores,molecules,spectrum,logits
            torch.cuda.empty_cache()
    write(OUT/'analysis/evaluation_complete.json',dict(status='COMPLETE',
        sources={str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__),ROWS,DREAMS,FP]},
        forbidden_read_attempts=GUARD_EVENTS,time=time.time()))

if __name__=='__main__':main()
