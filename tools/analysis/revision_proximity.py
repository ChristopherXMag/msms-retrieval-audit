#!/usr/bin/env python3
"""R1b association in frozen raw SCCF space; no causal coverage intervention."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from tools.analysis import revision_supplement as c
BINS=['<0.60','[0.60,0.70)','[0.70,0.80)','[0.80,0.90)','>=0.90']


def proximity(split,ids,g,r):
    if split=='test':
        z=c.load(c.ROOT/'outputs/official_s7_posthoc_20261018/proximity.npz')
        assert np.array_equal(z['identifiers'],ids) and np.array_equal(z['connectivities'],g)
        return z['proximity']
    path=c.OUT/'R1b/validation/proximity.npz'
    if path.exists():
        z=c.load(path);assert np.array_equal(z['identifiers'],ids) and np.array_equal(z['connectivities'],g)
        previous=c.OUT/'R1b/validation/summary.json'
        if previous.exists():
            # Reusing the cached proximity must retain its original feature/input bindings.
            old=json.loads(previous.read_text())
            for source,entry in old['inputs'].items():
                if source.endswith('revision_proximity.py') or source==str(previous.relative_to(c.ROOT)):continue
                assert c.sha(c.bind(c.ROOT/source))==entry['sha256'],source
        return z['proximity']
    import torch
    from tools.analysis.analyze_sccf_wrong_spectrum_proximity import normalized_raw_features,nearest_training_similarity,DEFAULT_FP,DEFAULT_CANDIDATES,DEFAULT_TRAIN
    from tools.stage1.train_msalign_projection_smoke import MoleculeEmbeddingStore
    from glmr.candidate_indices import CandidateIndexStore
    from tools.audits.audit_spectrum_permutation import connectivity_identities
    from tools.analysis.supplement_official_s7 import guard
    guard();torch.set_num_threads(2)
    for source in [DEFAULT_FP,DEFAULT_CANDIDATES,DEFAULT_TRAIN]:c.bind(source)
    fm=c.read(DEFAULT_FP)
    for block in ['count_morgan','count_atom_pair','rdkit_path_count','maccs']:
        for key in ['indices','values','indptr']:c.bind(DEFAULT_FP.parent/fm['features'][block][key])
    c.bind(c.ROOT/fm['base_cache'])
    if 'projection_order' in fm:
        for key in ['indices','sources']:c.bind(DEFAULT_FP.parent/fm['projection_order'][key])
    records=[json.loads(x) for x in DEFAULT_TRAIN.read_text().splitlines()];keys,audit=connectivity_identities(records)
    index=CandidateIndexStore(DEFAULT_CANDIDATES);by_row={}
    for q,key in zip(records,keys):by_row.setdefault(index.get(q['identifier'],256)[0],key)
    assert not set(g)&set(by_row.values())
    train_rows=sorted(by_row);val_rows=r['candidates'][:,0];unique=np.unique(val_rows);lookup=dict(zip(val_rows,g))
    assert len(train_rows)==25046
    store=MoleculeEmbeddingStore(None,'mean',DEFAULT_FP,preload_sparse_features=False,mol_blocks=['count_morgan','count_atom_pair','rdkit_path_count','maccs'])
    def features(ix):
        result=[]
        for st in range(0,len(ix),256):
            result.append(normalized_raw_features(store,ix[st:st+256],'block_l2'))
        return torch.cat(result)
    train=features(train_rows);val=features(unique.tolist());guard()
    vector=nearest_training_similarity(val,train,[str(lookup[i]) for i in unique],[by_row[i] for i in train_rows],torch.device('cpu'),128)
    result=vector[np.searchsorted(unique,val_rows)];assert np.isfinite(result).all()
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,identifiers=ids,connectivities=g,proximity=result)
    c.write(path.with_suffix('.json'),{'definition':'nearest official-train positive cosine; raw radius2 SCCF, block L2 then global L2; no learned projection',
                                     'train_positive_rows':len(train_rows),'unique_validation_positive_rows':len(unique),'train_identity_audit':audit,'resources':guard()})
    c.bind(path);return result


def spearman_bootstrap(x,y):
    def corr(a,b):
        a=rankdata(a,axis=-1);b=rankdata(b,axis=-1);a-=a.mean(-1,keepdims=True);b-=b.mean(-1,keepdims=True)
        return (a*b).sum(-1)/np.sqrt((a*a).sum(-1)*(b*b).sum(-1))
    rng=np.random.default_rng(c.SEED);boot=np.empty(c.BOOT)
    for st in range(0,c.BOOT,32):
        ix=rng.integers(0,len(x),size=(min(32,c.BOOT-st),len(x)))
        boot[st:st+len(ix)]=corr(x[ix],y[ix])
    return {'rho':float(corr(x,y)),'ci95':c.clean_interval(boot),'connectivities':len(x),'resamples':c.BOOT}


def residual_icc(hit,g,bin_index):
    unique,inv,n=np.unique(g,return_inverse=True,return_counts=True)
    count=np.zeros((len(unique),5));total=np.zeros_like(count);square=np.zeros_like(count)
    np.add.at(count,(inv,bin_index),1);np.add.at(total,(inv,bin_index),hit);np.add.at(square,(inv,bin_index),hit*hit)
    keep=n>=4;count,total,square=(v[keep] for v in (count,total,square));n=n[keep]
    def calc(count,total,square):
        ns=count.sum(-1);N=ns.sum(-1);G=ns.shape[-1]
        bins_n=count.sum(-2);bins_sum=total.sum(-2)
        mean=np.divide(bins_sum,bins_n,out=np.zeros_like(bins_sum),where=bins_n>0)
        residual_sum=(total-count*mean[...,None,:]).sum(-1)
        residual_square=(square-2*total*mean[...,None,:]+count*mean[...,None,:]**2).sum(-1)
        raw_sum=total.sum(-1);raw_square=square.sum(-1)
        def tau(t,sq):
            k0=(N-(ns*ns).sum(-1)/N)/(G-1)
            msb=((t*t/ns).sum(-1)-t.sum(-1)**2/N)/(G-1)
            msw=(sq-t*t/ns).sum(-1)/(N-G)
            return (msb-msw)/k0
        raw_tau=tau(raw_sum,raw_square);res_tau=tau(residual_sum,residual_square)
        return np.stack([c.icc_sufficient(ns,raw_sum,raw_square),c.icc_sufficient(ns,residual_sum,residual_square),
                         raw_tau,res_tau,1-res_tau/raw_tau],axis=-1)
    observed=calc(count,total,square);rng=np.random.default_rng(c.SEED);samples=np.empty((c.BOOT,5))
    for st in range(0,c.BOOT,32):
        ix=rng.integers(0,len(n),size=(min(32,c.BOOT-st),len(n)))
        samples[st:st+len(ix)]=calc(count[ix],total[ix],square[ix])
    labels=['ICC_raw','ICC_after_bin_mean_removal','between_variance_raw','between_variance_residual','between_variance_fraction_removed']
    return {'groups_minimum4':len(n),'queries':int(n.sum()),'values':{k:(float(v) if np.isfinite(v) else None) for k,v in zip(labels,observed)},
            'ci95':{k:c.clean_interval(samples[:,i]) for i,k in enumerate(labels)},
            'bin_means':'query-weighted means on >=4-connectivity cohort, refitted in each whole-connectivity bootstrap draw'}


def run(split):
    c.bind(Path(__file__));ids,g,r,values=c.all_method_values(split);p=proximity(split,ids,g,r)
    bins=np.digitize(p,[.6,.7,.8,.9]);unique,inv,n,sums,means=c.groups(p,g)
    correlations={}
    for name,outcome in [('sccf_ensemble',0),('sccf_ensemble',2),('selected',0)]:
        accuracy=c.groups(values[name][:,outcome],g)[4][:,0]
        correlations[name+('_R1' if outcome==0 else '_R20')]=spearman_bootstrap(means[:,0],accuracy)
    reports={}
    for name,v in values.items():
        reports[name]={'bin_adjusted_clustering':residual_icc(v[:,0],g,bins),'bins':{}}
        for i,label in enumerate(BINS):
            mask=bins==i;gg=g[mask];hit=v[mask,0]
            if not mask.any():reports[name]['bins'][label]={'queries':0};continue
            _,counts=np.unique(gg,return_counts=True)
            reports[name]['bins'][label]={'metrics':c.stats(v[mask],gg),
                                         'all_molecules':c.cluster_characterization(hit,gg,1) if len(counts)>1 and len(hit)>len(counts) else None,
                                         'minimum4':c.cluster_characterization(hit,gg,4) if np.sum(counts>=4)>1 else None}
        if name.startswith('sccf'):
            reports[name]['R20_bin_adjusted_clustering']=residual_icc(v[:,2],g,bins)
            for i,label in enumerate(BINS):
                mask=bins==i;gg=g[mask];hit=v[mask,2]
                if not mask.any():continue
                _,counts=np.unique(gg,return_counts=True)
                reports[name]['bins'][label]['R20_all_molecules']=c.cluster_characterization(hit,gg,1) if len(counts)>1 and len(hit)>len(counts) else None
                reports[name]['bins'][label]['R20_minimum4']=c.cluster_characterization(hit,gg,4) if np.sum(counts>=4)>1 else None
        print('R1b_ROW',name,flush=True)
    c.finish('R1b',split,{'correlations':correlations,'rows':reports,'headline':correlations,'fixed_bins':BINS,
                         'interpretation':'observational association; R6 not run; no official-split causal coverage claim',
                         'bin_rule':'query-level proximity thresholds; connectivity can span bins for distinct stereoisomer rows; bootstrap always resamples whole connectivity groups',
                         'units':'ICC, variance fractions, Spearman and clustering shares dimensionless; metric rows percent'},
             {'identifiers':ids,'connectivities':g,'proximity':p,'bin_index':bins,**{k:v for k,v in values.items()}})
