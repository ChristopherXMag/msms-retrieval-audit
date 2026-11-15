#!/usr/bin/env python3
"""Batch-one validation latency for the frozen top20-first/D2-fallback system."""
from __future__ import annotations
import argparse,gc,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.evaluation.official_s7_common import read,write,load,sha,PREP,CONTROLS,install_no_test_guard,gpu_preflight
from tools.evaluation.official_s7_system import System,old
from tools.evaluation.official_s7_numerics import pair_scores
from tools.analysis.supplement_official_s7 import guard
OUT=ROOT/'outputs/official_validation_latency_20261018'

def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),mean_ms=float(a.mean()),median_ms=float(np.median(a)),p90_ms=float(np.quantile(a,.9)),p95_ms=float(np.quantile(a,.95)),min_ms=float(a.min()),max_ms=float(a.max())) if len(a) else None

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queries',type=int,default=128);p.add_argument('--repeats',type=int,default=3);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    install_no_test_guard();OUT.mkdir(parents=True,exist_ok=True);dest=OUT/('smoke' if args.smoke else 'formal');assert not dest.exists();dest.mkdir()
    gpu_preflight(dest,'batch_one');guard();setup=time.perf_counter()
    source=read(old.SOURCE);write(dest/'source.json',source)
    ctx=System(dest/'source.json',PREP/'fresh_real_replay/inputs.jsonl',CONTROLS/'projection_formal',CONTROLS/'train_means.npz',reference=False)
    # No native spectrum/logit/result substitution inside timed inference.
    from tools.data.cache_msalign_dreams_representations import PreTrainedModel,ContrastiveHead,dreams_data,dformats,preprocess_spectrum
    import h5py
    ckpt=ROOT/'external/DreaMS/dreams/models/pretrained/embedding_model_massspecgym.ckpt'
    assert sha(ckpt)=='d9b4e8b920ff367370c5a3fa154eaa951f90bafa58d5224a594424d9c54b1832'
    dreams=PreTrainedModel.from_ckpt(ckpt,ContrastiveHead,n_highest_peaks=100).model.cuda().eval()
    preprocessor=dreams_data.SpectrumPreprocessor(dformat=dformats.DataFormatA(),n_highest_peaks=100)
    rng=np.random.default_rng(20261018);n=ctx.n
    selected=rng.choice(n,size=min(args.queries,n),replace=False);np.save(dest/'sample_rows.npy',selected)
    # Sampling never sees outcomes. Extra branch examples are reported separately, never reweighted as random sample.
    reference=load(CONTROLS/'formal/real.npz');routes=np.where(reference['support20'],0,np.where(reference['route5'],1,2))
    extra=np.concatenate([rng.choice(np.flatnonzero(routes==b),size=min(8,np.sum(routes==b)),replace=False) for b in range(3)])
    all_queries=np.unique(np.r_[selected,extra]);raw={}
    with h5py.File(ROOT/'outputs/msalign_stage1_full_v15/val_full.hdf5','r') as h:
        for q in all_queries:
            spec=np.array(h['spectrum'][int(q)]);valid=(spec[0]>0)&np.isfinite(spec).all(0)&(spec[1]>=0)
            raw[int(q)]=dict(mzs=spec[0,valid].tolist(),intensities=spec[1,valid].tolist(),precursor_mz=ctx.inputs[int(q)]['precursor_mz'])
    setup_seconds=time.perf_counter()-setup
    def synchronize():torch.cuda.synchronize()
    def query(q):
        timer=time.perf_counter();times={}
        def tick(key):
            nonlocal timer
            synchronize();now=time.perf_counter();times[key]=(now-timer)*1000;timer=now
        with torch.inference_mode():
            inp=preprocess_spectrum(raw[q],preprocessor,100.)[None].cuda().to(dtype=dreams.dtype);embedding=dreams(inp,no_head=True).float()
            tick('raw_preprocess_and_dreams')
            a,b=ctx.pool['indptr'][q:q+2];pool=ctx.pool['candidate_rows'][a:b];local=ctx.local[pool];scores=[];projected=[]
            for model,molecules in zip(ctx.models,ctx.projections):
                s=model.project_spec(embedding);m=torch.from_numpy(np.array(molecules[local],copy=True)).cuda();scores.append((torch.einsum('bd,kd->bk',s,m)/model.temperature()).cpu().numpy()[0]);projected.append(s)
            score=np.stack(scores,-1);ensemble=score.astype(np.float64).mean(-1).astype(np.float32)
            order=old.ctl.candidate_order(str(ctx.ids[q]),pool,ensemble,20260926);cand=pool[order[:20]];base=ensemble[order[:20]][None];ss=score[order[:20]][None]
            mol=np.array(ctx.projections[0][ctx.local[cand]],copy=True)[None].astype(np.float16).astype(np.float32);spec=projected[0].cpu().numpy().astype(np.float16).astype(np.float32)
            tick('full_pool_stage1_rank')
            qin=dict(ctx.inputs[q],candidate_global_rows=cand.tolist(),candidate_smiles=[ctx.smiles[int(r)] for r in cand]);support20=old.support(qin,qin['candidate_smiles']);support5=old.support(qin,qin['candidate_smiles'][:5]);route=bool(old.route_to_d2(np.array([support5]),np.array([support20]))[0])
            extent=min(512,ctx.n-(q//512)*512)
            pair=pair_scores(ctx.pair20,spec,mol,ss,base,extent,physical_width=20)
            built=ctx.evidence([qin],base,20);assert bool(built['support'][0])==support20
            detail=ctx.tree(20,'detail',built['features'],built['support'],old.top.d2.deployment_anchor(pair,built['coverage']))
            tick('top20_support_pair_evidence_trees')
            if route:
                in5=dict(qin,candidate_global_rows=cand[:5].tolist(),candidate_smiles=qin['candidate_smiles'][:5]);pair5=old.pair_score(ctx.pair5,spec,mol[:,:5],ss[:,:5],base[:,:5],extent)
                b5=ctx.evidence([in5],base[:,:5],5);d5=ctx.tree(5,'detail',b5['features'],b5['support'],old.top.d2.deployment_anchor(pair5,b5['coverage']));ranked=old.ctl.candidate_order(str(ctx.ids[q]),cand[:5],d5[0],20260926);pred=int(cand[ranked[0]])
            else:
                ranked=old.ctl.candidate_order(str(ctx.ids[q]),cand,detail[0],20260926);pred=int(cand[ranked[0]])
            tick('on_demand_D2_and_final_order')
        return times,dict(query=q,route=1 if route else 0 if support20 else 2,predicted_global_row=pred,expected_top1=int(reference['routed_order'][q,0]),logit_max_abs_error=float(np.max(np.abs(score-ctx.pool['scores'][:,a:b].T))),embedding_max_abs_error=float(np.max(np.abs(embedding.cpu().numpy()[0]-ctx.dreams[q]))))
    # Warmup includes every branch; the formula/substructure memoizer is bounded by the frozen policy.
    for q in extra:query(int(q))
    timings=[];records=[]
    for repeat in range(args.repeats):
        for j,q in enumerate(selected):
            guard();synchronize();started=time.perf_counter();times,record=query(int(q));synchronize();total=(time.perf_counter()-started)*1000
            row=dict(repeat=repeat,total_ms=total,**times,**record);timings.append(row)
            if j%16==0:print('LATENCY',repeat,j,total,guard(),flush=True)
        # Allow other work between repetitions without selecting a timing subset.
    branch_supplement=[]
    for q in extra:
        synchronize();start=time.perf_counter();times,record=query(int(q));synchronize();branch_supplement.append(dict(total_ms=(time.perf_counter()-start)*1000,**times,**record))
    with (dest/'per_query.jsonl').open('w') as f:
        for row in timings:f.write(json.dumps(row)+'\n')
    write(dest/'branch_supplement.json',branch_supplement)
    summary=dict(status='COMPLETE',split='official_validation',official_test_read=False,posthoc_resource_measurement=True,queries=len(selected),repeats=args.repeats,total_observations=len(timings),warmup_queries=len(extra),seed=20261018,
        conditions='serial requests, batch1 raw spectrum preprocessing and data-safe DreaMS, 3 SCCF spectrum projections, cached CPU molecular projections for full supplied pool, deterministic rank, 3 top20 pair heads, rebuilt peak/formula evidence, 3 trees, conditional complete D2; CPU/GPU transfers and synchronizations included; retained frozen physical repeat-padding up to512 in pair heads; 1 PyTorch CPU thread and2 tree threads; no queue/network/disk fetch or candidate search, startup, unseen molecule featurization or projection included',
        molecular_projection_cache='three CPU memory-mapped precomputed molecule projection tables; reused across queries',memoizer_entries=len(ctx.ev.cache),setup_seconds=setup_seconds,
        total=stats([r['total_ms'] for r in timings]),stages={k:stats([r[k] for r in timings]) for k in ('raw_preprocess_and_dreams','full_pool_stage1_rank','top20_support_pair_evidence_trees','on_demand_D2_and_final_order')},
        branches={str(b):stats([r['total_ms'] for r in timings if r['route']==b]) for b in range(3)},branch_supplement={str(b):stats([r['total_ms'] for r in branch_supplement if r['route']==b]) for b in range(3)},
        parity=dict(top1_matches=sum(r['predicted_global_row']==r['expected_top1'] for r in timings),observations=len(timings),logit_max_abs_error=max(r['logit_max_abs_error'] for r in timings),embedding_max_abs_error=max(r['embedding_max_abs_error'] for r in timings)),resources=guard(),gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'tools/evaluation/official_s7_system.py',ROOT/'tools/evaluation/official_s7_numerics.py']},validation_source_sha256=sha(old.SOURCE),dreams_checkpoint_sha256=sha(ckpt))
    write(dest/'summary.json',summary);print('LATENCY_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
