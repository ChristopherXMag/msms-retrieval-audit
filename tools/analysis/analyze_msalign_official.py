#!/usr/bin/env python3
"""Paired seed-mean/cluster uncertainty for frozen E1 and spectrum controls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from glmr.embcos import summarize_metrics,write_json,sha256_file,uniform_tie_metrics
SEEDS=(20260621,20260622,20260623)
METRICS=('recall@1','recall@5','recall@20','mrr')


def intervals(values,groups,replicates=10000,seed=20261014):
    values=np.asarray(values,np.float64)
    if values.ndim==1:values=values[:,None]
    _,inverse=np.unique(groups,return_inverse=True);counts=np.bincount(inverse)
    totals=np.stack([np.bincount(inverse,weights=values[:,j]) for j in range(values.shape[1])],1)
    means=totals/counts[:,None];rng=np.random.default_rng(seed)
    query=np.empty((replicates,values.shape[1]));cluster=np.empty_like(query);balanced=np.empty_like(query)
    for start in range(0,replicates,32):
        end=min(start+32,replicates);idx=rng.integers(0,len(values),size=(end-start,len(values)))
        query[start:end]=values[idx].mean(1)
        ids=rng.integers(0,len(counts),size=(end-start,len(counts)))
        cluster[start:end]=totals[ids].sum(1)/counts[ids].sum(1)[:,None]
        balanced[start:end]=means[ids].mean(1)
    result={}
    for j,name in enumerate(METRICS[:values.shape[1]]):
        result[name]={'query_point_pp':float(values[:,j].mean()*100),
            'query_ci95_pp':(np.quantile(query[:,j],[.025,.975])*100).tolist(),
            'connectivity_cluster_query_ci95_pp':(np.quantile(cluster[:,j],[.025,.975])*100).tolist(),
            'connectivity_balanced_point_pp':float(means[:,j].mean()*100),
            'connectivity_balanced_ci95_pp':(np.quantile(balanced[:,j],[.025,.975])*100).tolist()}
    return {'queries':len(values),'connectivities':len(counts),'replicates':replicates,'seed':seed,'metrics':result}


def load(path):return dict(np.load(path,allow_pickle=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'outputs/msalign_official_e1_20261014')
    args=p.parse_args();root=args.root;out=root/'analysis';out.mkdir(exist_ok=True)
    records={seed:load(root/f'seed{seed}'/'val_ranks.npz') for seed in SEEDS}
    first=records[SEEDS[0]];groups=first['connectivities'];ids=first['identifiers'];n=len(ids)
    for rec in records.values():
        assert np.array_equal(rec['identifiers'],ids) and np.array_equal(rec['connectivities'],groups)
        assert np.array_equal(rec['candidate_counts'],first['candidate_counts'])
        assert np.array_equal(uniform_tie_metrics(rec['greater'],rec['equal']),rec['expected_metrics'])
    per_seed={};baseline=[];sources={}
    for seed,rec in records.items():
        summary=json.loads((root/f'seed{seed}'/'train_summary.json').read_text());assert summary['status']=='TRAINING_COMPLETE' and summary['global_step']==20000
        assert summary['val_ranks_sha256']==sha256_file(root/f'seed{seed}'/'val_ranks.npz')
        control=json.loads((root/f'seed{seed}'/'controls/summary.json').read_text());assert control['status']=='COMPLETE'
        per_seed[str(seed)]={'selected':summary['best'],'real':summarize_metrics(rec['expected_metrics'],groups),
            'elapsed_seconds':summary['elapsed_seconds'],'peak_cuda_mib':summary['peak_cuda_mib'],'controls':control['metrics']}
        bpath=ROOT/f'outputs/stage1_submission_0926/runs/full_r2_block_l2/seed{seed}/val_ranks.json'
        b=json.loads(bpath.read_text());rank=np.asarray(b['metrics']['ranks'],np.int64);assert len(rank)==n
        assert Path(b['metadata']).resolve()==(ROOT/'outputs/msalign_stage1_full_v15/val_full.metadata.jsonl').resolve()
        control_path=ROOT/f'outputs/stage1_submission_0926/runs/full_r2_block_l2/seed{seed}/blind_controls/constant_spectrum_full_val.per_query.npz'
        with np.load(control_path,allow_pickle=False) as z:
            assert np.array_equal(z['recipient_rows'],np.arange(n))
            vectors=uniform_tie_metrics(z['baseline_greater'],z['baseline_equal'])
        manifest=json.loads((root/'assets/manifest.json').read_text())
        assert sha256_file(Path(b['metadata']))==manifest['metadata_source_sha256']['val']
        baseline.append(vectors)
        sources[str(bpath.relative_to(ROOT))]=sha256_file(bpath)
        sources[str(control_path.relative_to(ROOT))]=sha256_file(control_path)
    stack=np.stack([records[s]['expected_metrics'] for s in SEEDS]);mean=stack.mean(0);base=np.stack(baseline).mean(0)
    aggregate=summarize_metrics(mean,groups);sd=stack.mean(1).std(0,ddof=1)*100
    aggregate['seed_sample_sd']={name:float(sd[j]) for j,name in enumerate(METRICS)}
    balanced_seed=np.asarray([[per_seed[str(s)]['real']['connectivity_balanced'][name] for name in METRICS] for s in SEEDS])
    aggregate['connectivity_balanced_seed_sample_sd']={name:float(balanced_seed[:,j].std(ddof=1)) for j,name in enumerate(METRICS)}
    result={'status':'COMPLETE','method':'MSAlign v2 single (matched official/data-safe protocol)','test_evaluated':False,
        'aggregation':'arithmetic mean of three per-query seed contributions; seed metric sample SD; not ensemble',
        'per_seed':per_seed,'aggregate':aggregate,'matched_sccf_aggregate':summarize_metrics(base,groups),
        'absolute_uncertainty':intervals(mean,groups),
        'paired_minus_matched_sccf':intervals(mean-base,groups),
        'comparison_baseline':'same-seed frozen official/data-safe radius2 SCCF, existing frozen blind-control real greater/equal counts, uniform-tie baseline; original ranks/source metadata retained; no literature delta',
        'sccf_seed_directions':{str(s):{name:float((stack[i,:,j]-baseline[i][:,j]).mean()*100) for j,name in enumerate(METRICS)} for i,s in enumerate(SEEDS)},
        'candidate_recall':1.,'candidate_count_min':int(first['candidate_counts'].min()),'candidate_count_max':int(first['candidate_counts'].max()),
        'candidate_count_strata':{},'controls':{},'baseline_artifact_digests':sources,'analysis_sha256':sha256_file(Path(__file__))}
    lengths=first['candidate_counts']
    for label,mask in [('12-127',lengths<128),('128-255',(lengths>=128)&(lengths<256)),('256',lengths==256)]:
        if mask.any():result['candidate_count_strata'][label]=summarize_metrics(mean[mask],groups[mask])
    random=np.stack([np.minimum(k/lengths,1.) for k in (1,5,20)]+[np.asarray([sum(1./np.arange(1,m+1))/m for m in lengths])],-1)
    result['exact_random']=summarize_metrics(random,groups)
    for name in ('zero','train_mean'):
        control=[]
        for seed in SEEDS:
            rec=load(root/f'seed{seed}'/f'controls/{name}.npz');assert np.array_equal(rec['identifiers'],ids)
            control.append(rec['expected_metrics'])
        m=np.stack(control).mean(0)
        result['controls'][name]={'metrics':summarize_metrics(m,groups),'real_minus_control':intervals(mean-m,groups)}
    for donor in (20260820,20260821,20260822):
        control=[];recipient=None
        for seed in SEEDS:
            rec=load(root/f'seed{seed}'/f'controls/wrong_seed{donor}.npz')
            if recipient is None:recipient=rec['recipient_rows']
            assert np.array_equal(rec['recipient_rows'],recipient) and np.array_equal(rec['identifiers'],ids[recipient])
            control.append(rec['expected_metrics'])
        m=np.stack(control).mean(0)
        result['controls'][f'wrong_seed{donor}']={'metrics':summarize_metrics(m,groups[recipient]),
            'eligible_exact_random':summarize_metrics(random[recipient],groups[recipient]),
            'eligible_real':summarize_metrics(mean[recipient],groups[recipient]),
            'real_minus_wrong':intervals(mean[recipient]-m,groups[recipient])}
    # Absolute gains and available R@20--R@1 gap are reported without reranker claims.
    base_r1=float(base[:,0].mean()*100);base_r20=float(base[:,2].mean()*100);delta=float((mean[:,0]-base[:,0]).mean()*100)
    result['sccf_r1_gain_pp']=delta;result['sccf_r20_minus_r1_gap_pp']=base_r20-base_r1
    result['gap_closure_ratio']=delta/(base_r20-base_r1)
    result['gap_closure_scope']='representation/method comparison only; not a fixed-shortlist reranker'
    emb_records=[]
    for seed in SEEDS:
        path=ROOT/f'outputs/embcos_official_e2_20261014/seed{seed}/val_ranks.npz'
        rec=load(path)
        assert np.array_equal(rec['identifiers'],ids) and np.array_equal(rec['connectivities'],groups)
        assert np.array_equal(rec['candidate_counts'],first['candidate_counts']) and np.array_equal(rec['gt_slots'],first['gt_slots'])
        emb_records.append(rec['expected_metrics']);sources[str(path.relative_to(ROOT))]=sha256_file(path)
    embmean=np.stack(emb_records).mean(0)
    result['matched_embcos_aggregate']=summarize_metrics(embmean,groups)
    result['paired_minus_matched_embcos']=intervals(mean-embmean,groups)
    result['embcos_seed_directions']={str(s):{name:float((stack[i,:,j]-emb_records[i][:,j]).mean()*100) for j,name in enumerate(METRICS)} for i,s in enumerate(SEEDS)}
    result['spectrum_replacement_scope']='CE/adduct of recipient fixed; conditional spectral readout; no causal CE/cohort attribution'
    write_json(out/'summary.json',result)
    # Per-query mean contributions persist for downstream frozen report construction.
    np.savez(out/'seed_mean_per_query.npz',identifiers=ids,connectivities=groups,candidate_counts=lengths,
        msalign=mean,sccf=base,embcos=embmean,exact_random=random)
    print(json.dumps({'status':result['status'],'aggregate':aggregate,'paired_minus_matched_sccf':result['paired_minus_matched_sccf']},indent=2),flush=True)


if __name__=='__main__':main()
