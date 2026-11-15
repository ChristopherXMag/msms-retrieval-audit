#!/usr/bin/env python3
"""Fixed R6-lite stratified connectivity bootstrap, conditional on two paired fits."""
from __future__ import annotations
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import rankdata
from tools.evaluation.official_s7_common import read,write,load,sha,install_no_test_guard,GUARD_EVENTS,sccf_seed_contributions
from tools.analysis.supplement_official_s7 import guard
from glmr.embcos import summarize_metrics

OUT=ROOT/'outputs/revision_supplement_20261111/R6_lite/validation'
SEEDS=(20260621,20260622)
LABELS=('seed20260621','seed20260622','two_seed_mean')
ARMS=('targeted','size_matched')
METRICS=('R@1','R@5','R@20','MRR')
WEIGHTS=('query_weighted','connectivity_balanced')
BOOT=10000
SEED=20261111


def correlation(x,y):
    a=rankdata(x,axis=-1);b=rankdata(y,axis=-1)
    a-=a.mean(-1,keepdims=True);b-=b.mean(-1,keepdims=True)
    denominator=np.sqrt((a*a).sum(-1)*(b*b).sum(-1))
    return np.divide((a*b).sum(-1),denominator,out=np.full_like(denominator,np.nan),where=denominator>0)


def dose_response(x,y):
    point=float(correlation(x,y))
    rng=np.random.default_rng(SEED);draws=[]
    for start in range(0,BOOT,32):
        ix=rng.integers(0,len(x),size=(min(32,BOOT-start),len(x)))
        draws.extend(correlation(x[ix],y[ix]).tolist())
    draws=np.asarray(draws);valid=draws[np.isfinite(draws)]
    return dict(spearman_rho=point if np.isfinite(point) else None,
        ci95=np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,
        valid_bootstrap_draws=len(valid),requested_bootstrap_draws=BOOT,
        connectivities=len(x),undefined=not np.isfinite(point),
        x='mean distinct-GT-row p_size_matched minus p_targeted',
        y='mean sibling-query targeted minus size-matched hit; averaged over two seeds')


def main():
    install_no_test_guard();guard();started=time.time()
    assert read(OUT/'analysis/evaluation_complete.json')['status']=='COMPLETE'
    assert not (OUT/'analysis/summary.json').exists(), 'Never overwrite a final analysis'
    gate=read(OUT/'gate/summary.json');assert gate['status']=='PASS'
    g=load(OUT/'gate/cohort_and_masks.npz');groups,inv,counts=np.unique(g['connectivities'],return_inverse=True,return_counts=True)
    assert np.array_equal(groups,g['connectivity_keys'])
    inputs={str((OUT/'gate/summary.json').relative_to(ROOT)):sha(OUT/'gate/summary.json')}
    values={};tie_counts={}
    for arm in ARMS:
        vectors=[]
        for seed in SEEDS:
            path=OUT/f'seed{seed}'/arm/'validation_per_query.npz'
            z=load(path);assert np.array_equal(z['identifiers'],g['identifiers'])
            assert np.array_equal(z['connectivities'],g['connectivities'])
            assert sha(path)==read(path.parent/'validation_summary.json')['per_query_sha256']
            assert z['expected_metrics'].shape==(len(inv),4)
            vectors.append(z['expected_metrics']);inputs[str(path.relative_to(ROOT))]=sha(path)
            tie_counts[f'{arm}/seed{seed}']=int((z['equal']>1).sum())
        values[arm]=np.stack(vectors)
    differences=values['targeted']-values['size_matched']
    all_d=np.concatenate([differences,differences.mean(0,keepdims=True)])
    sums=np.zeros((len(groups),3,4));np.add.at(sums,inv,all_d.transpose(1,0,2))
    means=sums/counts[:,None,None]
    masks={name:np.isin(groups,g[key]) for name,key in [('A','A'),('B_clean','B_clean'),('B_spill','B_spill')]}
    assert sum(mask.astype(int) for mask in masks.values()).min()==1
    assert sum(mask.astype(int) for mask in masks.values()).max()==1
    A=np.flatnonzero(masks['A']);B=np.flatnonzero(masks['B_clean'])
    def estimate(ix):
        return np.stack([sums[ix].sum(0)/counts[ix].sum(),means[ix].mean(0)],axis=1)
    a_point,b_point=estimate(A),estimate(B)
    points={'A_targeted_minus_random':a_point,'B_clean_targeted_minus_random':b_point,
        'difference_in_differences':a_point-b_point}
    rng=np.random.default_rng(SEED)
    boot={key:np.empty((BOOT,3,2,4)) for key in points}
    for start in range(0,BOOT,16):
        size=min(16,BOOT-start)
        ia=A[rng.integers(0,len(A),size=(size,len(A)))]
        ib=B[rng.integers(0,len(B),size=(size,len(B)))]
        def draw(ix):
            return np.stack([sums[ix].sum(1)/counts[ix].sum(1)[:,None,None],means[ix].mean(1)],axis=2)
        a,b=draw(ia),draw(ib)
        boot['A_targeted_minus_random'][start:start+size]=a
        boot['B_clean_targeted_minus_random'][start:start+size]=b
        boot['difference_in_differences'][start:start+size]=a-b
    reports={}
    for contrast,point in points.items():
        intervals=np.quantile(boot[contrast],[.025,.975],axis=0)
        reports[contrast]={label:{weight:{metric:{'estimate_pp':float(100*point[s,w,m]),
            'ci95_pp':(100*intervals[:,s,w,m]).tolist()} for m,metric in enumerate(METRICS)}
            for w,weight in enumerate(WEIGHTS)} for s,label in enumerate(LABELS)}
    group_metrics={}
    for arm in ARMS:
        group_metrics[arm]={}
        for label,array in zip(LABELS,np.concatenate([values[arm],values[arm].mean(0,keepdims=True)])):
            group_metrics[arm][label]={'full_validation':summarize_metrics(array,g['connectivities'])}
            for name,key in [('A','A'),('B_clean','B_clean'),('B_spill','B_spill')]:
                mask=np.isin(g['connectivities'],g[key])
                group_metrics[arm][label][name]=summarize_metrics(array[mask],g['connectivities'][mask])
    # Descriptive original full-training reference: existing scores, never a third refit.
    reference_path=ROOT/'outputs/stage2_official_datasafe/stage1_candidate_scores/val_three_seed_logits.npz'
    ref=load(reference_path)
    assert np.array_equal(ref['identifiers'],g['identifiers'])
    assert np.array_equal(ref['connectivities'],g['connectivities'])
    baseline=np.stack([sccf_seed_contributions(ref,i) for i in (0,1)])
    baseline_report={}
    for label,array in zip(LABELS,np.concatenate([baseline,baseline.mean(0,keepdims=True)])):
        baseline_report[label]={}
        for name,key in [('A','A'),('B_clean','B_clean'),('B_spill','B_spill')]:
            mask=np.isin(g['connectivities'],g[key])
            baseline_report[label][name]=summarize_metrics(array[mask],g['connectivities'][mask])
    inputs[str(reference_path.relative_to(ROOT))]=sha(reference_path)
    dose={}
    differential=g['connectivity_p_size_matched']-g['connectivity_p_targeted']
    for name in ('A','B_spill'):
        ix=np.flatnonzero(masks[name])
        dose[name]={metric:dose_response(differential[ix],means[ix,2,m]) for m,metric in [(2,'R@20'),(0,'R@1')]}
    did_ci=np.quantile(boot['difference_in_differences'][:,2,:,2],[.025,.975],axis=0)
    joint=bool((did_ci[1]<0).all() and (points['difference_in_differences'][:2,:,2]<0).all()
        and (a_point[:2,:,2]<0).all())
    result=dict(status='COMPLETE',date_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        design='Fixed A200 well-covered cohort; targeted versus distant row-count-matched removal; two paired seeds; official validation only',
        primary='R@20 difference-in-differences under both query-weighted and connectivity-balanced estimands',
        secondary='R@1; within-A and B-spill proximity-dose associations',
        primary_joint_directional_criterion_met=joint,
        contrasts=reports,metrics=group_metrics,existing_full_train_reference=baseline_report,
        dose_response=dose,gate=gate['cohorts'],removals=gate['removals'],exact_GT_tie_queries=tie_counts,
        fixed_training_seeds=list(SEEDS),seed_aggregation='Arithmetic mean of per-query contributions; no logit ensemble',
        bootstrap=dict(draws=BOOT,seed=SEED,method='Whole connectivity; independent within fixed A and B-clean strata; all sibling queries, arms, seeds, weightings and outcomes paired',
            intervals='Two-sided 95% percentile, conditional on fixed cohort, single removal draw and fitted models'),
        limitations=['Well-covered selected validation cohort only; development-exposed and post hoc to the original study',
            'Stage 1 intervention only; no final-system or official-test claim',
            'B-clean means unchanged nearest proximity, not unchanged predictions or cancellation of training noise',
            'Two training seeds are a limited stability check; connectivity bootstrap does not estimate seed variance',
            'Training composition changes many learned relationships; proximity is not established as the sole causal mediator',
            'Random arm matches removed spectrum count, not the number of touched or fully removed connectivities',
            'Only positive training spectra are removed; negative candidate pools remain fixed, so this is not a total molecular-exposure ablation',
            'The frozen Stage 1 candidate-prior limitations remain; this intervention does not isolate an MS/MS-specific mechanism',
            'Null/mixed intervals are inconclusive; no significance-triggered extra fits'],
        inputs=inputs,source_sha256=sha(Path(__file__)),elapsed_seconds=time.time()-started,
        resources=guard(),forbidden_read_attempts=GUARD_EVENTS)
    arrays=dict(identifiers=g['identifiers'],connectivities=g['connectivities'],
        group_keys=groups,group_counts=counts,A=g['A'],B_clean=g['B_clean'],B_spill=g['B_spill'],
        targeted=values['targeted'],size_matched=values['size_matched'],baseline=baseline,
        paired_differences=differences,**{f'bootstrap_{k}':v for k,v in boot.items()})
    np.savez_compressed(OUT/'analysis/per_query_and_bootstrap.npz',**arrays)
    result['archive_sha256']=sha(OUT/'analysis/per_query_and_bootstrap.npz')
    write(OUT/'analysis/summary.json',result)
    print('PRIMARY',reports['difference_in_differences']['two_seed_mean'],flush=True)
    print('JOINT_DIRECTIONAL_CRITERION',joint,flush=True)

if __name__=='__main__':main()
