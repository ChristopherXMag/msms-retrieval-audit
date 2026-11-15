#!/usr/bin/env python3
"""R4: float64 precursor prior, convex likelihood fusion and connectivity crossfit."""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from tools.analysis import revision_supplement as c


def source_metadata(split, ids, row_smiles):
    path=c.ROOT/'data/source_massspecgym_v15_0dc77f670939718f333cd06e8a7b70fdfc5a39a5/MassSpecGym1.5.tsv'
    expected='50cfdd1d22f79543c59555f9ce43c6893bd788a19a42b21fb4e2e3a54673c06a'
    assert c.sha(path)==expected
    c.bind(path)
    needed=set(ids.tolist());found={};fold='val' if split=='validation' else 'test'
    # Only the requested split's precursor/adduct/identity columns are retained.
    with path.open(newline='') as f:
        for q in csv.DictReader(f,delimiter='\t'):
            if q['fold']!=fold:continue
            if q['identifier'] in needed:found[q['identifier']]=q
    assert set(found)==needed
    metadata=[json.loads(x) for x in c.bind(c.ROOT/f'outputs/msalign_stage1_full_v15/{fold}_full.metadata.jsonl').read_text().splitlines()]
    assert [x['identifier'] for x in metadata]==ids.tolist()
    offsets=c.read(c.ROOT/'outputs/e0_external_contract_20261014/raw_precursor/audit.json')['offsets']
    mass=[]
    for identifier,smi,exported in zip(ids,row_smiles,metadata):
        q=found[str(identifier)]
        assert float(q['precursor_mz'])==float(exported['precursor_mz']) and q['adduct']==exported['adduct']
        assert Chem.MolToSmiles(Chem.MolFromSmiles(q['smiles']),isomericSmiles=True)==smi
        assert exported['canonical_isomeric_smiles']==smi
        assert q['adduct'] in offsets
        mass.append(float(q['precursor_mz'])-offsets[q['adduct']])
    return np.asarray(mass,np.float64),offsets


def exact_masses(candidate_rows, split):
    folder=c.OUT/'R4'/split;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'candidate_exact_masses.npz';wanted=np.unique(candidate_rows[candidate_rows>=0])
    if path.exists():
        z=c.load(path);assert np.array_equal(z['candidate_rows'],wanted)
        return wanted,z['exact_mass']
    source=c.bind(c.ROOT/'outputs/run_k_full_train/full_sparse/full_smiles.txt')
    manifest=c.read(c.ROOT/'outputs/stage1_submission_0926/assets/full_sparse_direct/manifest_historical_projection_order.json')
    assert c.sha(source)==manifest['smiles_sha256']
    masses=np.full(len(wanted),np.nan,np.float64)
    if split=='test':
        old=c.load(c.OUT/'R4/validation/candidate_exact_masses.npz')
        index=np.searchsorted(old['candidate_rows'],wanted);keep=index<len(old['candidate_rows'])
        keep[keep]&=old['candidate_rows'][index[keep]]==wanted[keep]
        masses[keep]=old['exact_mass'][index[keep]]
        print('R4_REUSE_VALIDATION_MASSES',int(keep.sum()),'of',len(wanted),flush=True)
    missing=np.flatnonzero(~np.isfinite(masses));target=set(wanted[missing].tolist());done=0
    if len(target):
        with source.open() as f:
            for i,line in enumerate(f):
                if i not in target:continue
                mol=Chem.MolFromSmiles(line.strip());assert mol is not None,(i,line)
                masses[np.searchsorted(wanted,i)]=Descriptors.ExactMolWt(mol);done+=1
                if done%25000==0:print('R4_EXACT_MASS',split,done,'/',len(target),flush=True)
                if done==len(target):break
    assert done==len(target) and np.isfinite(masses).all() and (masses>0).all()
    np.savez_compressed(path,candidate_rows=wanted,exact_mass=masses)
    c.bind(path)
    return wanted,masses


def fit_fusion(scores,rho,counts,gt,indices):
    scores=np.array(scores[indices],dtype=np.float64,copy=True)
    rho=np.array(rho[indices],dtype=np.float64,copy=True)
    counts=counts[indices];gt=gt[indices];ix=np.arange(len(indices))
    valid=np.arange(scores.shape[1])[None,:]<counts[:,None]
    scores[~valid]=0;rho[~valid]=0
    scores-=scores[ix,gt,None];rho-=rho[ix,gt,None,:]
    def objective(w):
        z=w[0]*scores+np.einsum('ijk,k->ij',rho,w[1:],optimize=False)
        z[~valid]=-np.inf
        denom=logsumexp(z,axis=1);prob=np.exp(z-denom[:,None])
        gradient=np.r_[np.sum(prob*scores)/len(ix),np.einsum('ij,ijk->k',prob,rho,optimize=False)/len(ix)]
        return float(denom.mean()),gradient
    result=minimize(objective,np.array([1.,0.,0.,0.]),jac=True,method='L-BFGS-B',bounds=[(1e-10,None),(0,None),(0,None),(0,None)],
                    options={'maxiter':2000,'ftol':1e-12,'gtol':1e-7,'maxls':50})
    value,gradient=objective(result.x)
    projected=np.where((result.x<=np.array([1e-10,0,0,0])+1e-9)&(gradient>0),0,gradient)
    assert result.success and np.max(np.abs(projected))<2e-5,(result.message,projected,result.x)
    return result.x,{'weights_t_w001_w01_w1':result.x.tolist(),'negative_log_likelihood':value,'iterations':int(result.nit),
                     'success':bool(result.success),'message':str(result.message),'projected_gradient_max':float(np.max(np.abs(projected))),'training_queries':len(ix)}


def pad_sccf(stage, rows, seed=None):
    flat=stage['ensemble_scores'] if seed is None else stage['scores'][seed]
    result=np.zeros(rows['candidates'].shape,np.float64)
    for i,(a,b) in enumerate(zip(stage['indptr'][:-1],stage['indptr'][1:])):
        original=stage['candidate_rows'][a:b];order=rows['permuted_candidates'][i,:b-a]
        assert np.array_equal(original,rows['candidates'][i,:b-a])
        lookup={int(row):j for j,row in enumerate(original)}
        result[i,:b-a]=flat[a:b][[lookup[int(row)] for row in order]]
    return result


def run(split):
    assert rdBase.rdkitVersion=='2025.09.6','Use the pinned RDKit 2025.09.6 chemistry environment'
    c.bind(Path(__file__))
    from tools.analysis.supplement_official_s7 import guard
    resources=guard()
    b=c.base(split);r=c.rows(split);ids=b['identifiers'];g=b['connectivities'];n=len(ids)
    assert np.array_equal(ids,r['identifiers'])
    folder=c.OUT/'R4'/split;folder.mkdir(parents=True,exist_ok=True)
    unique=np.unique(g);shuffled=unique.copy();np.random.default_rng(c.SEED).shuffle(shuffled)
    fold=np.isin(g,shuffled[len(shuffled)//2:]).astype(np.int8)
    contract={'status':'FROZEN_BEFORE_R4_OUTCOMES','fold_seed':c.SEED,'fold_definition':'shuffle sorted unique recomputed connectivities; first floor(G/2) groups in fold0, remainder fold1',
              'objective':'query-mean candidate negative log softmax, convex in four nonnegative scalars; t strictly positive; no R1 selection',
              'sigmas_ppm':[.01,.1,1.],'dtype':'float64','selected_policy':'fuse only the actually executed prefix (20 normally, 5 on frozen D2 fallback); keep frozen tail, membership and R20',
              'test_policy':'one full-validation likelihood refit per row, frozen before any R4 test outcomes','rdkit':rdBase.rdkitVersion}
    contract_path=c.OUT/'R4/contract.json'
    if contract_path.exists():assert c.read(contract_path)==contract
    else:c.write(contract_path,contract)
    if split=='validation':
        assert not set(g[fold==0])&set(g[fold==1]);np.savez_compressed(folder/'folds.npz',identifiers=ids,connectivities=g,fold=fold)
        c.bind(folder/'folds.npz')
    neutral,offsets=source_metadata(split,ids,r['smiles'])
    wanted,masses=exact_masses(r['permuted_candidates'],split)
    valid=np.arange(256)[None,:]<r['lengths'][:,None]
    ix=np.searchsorted(wanted,np.maximum(r['permuted_candidates'],0));ix=np.minimum(ix,len(wanted)-1)
    mass_matrix=masses[ix];ppm=1e6*(neutral[:,None]-mass_matrix)/mass_matrix
    rho=np.exp(-.5*(ppm[:,:,None]/np.array([.01,.1,1.]))**2);rho[~valid]=0
    np.savez_compressed(folder/'mass_inputs.npz',identifiers=ids,neutral_mass=neutral,candidate_ppm=ppm)
    descriptive={};arrays={'identifiers':ids,'connectivities':g,'candidate_counts':r['lengths'],'gt_slots':r['gt_slots']}
    for j,sigma in enumerate([.01,.1,1.]):
        v=c.metrics_scores(rho[:,:,j],r['lengths'],r['gt_slots']);truth=rho[np.arange(n),r['gt_slots'],j]
        tie=((rho[:,:,j]==truth[:,None])&valid).sum(1)/r['lengths']
        descriptive[str(sigma)]={'metrics':c.stats(v,g),'GT_equal_score_candidate_fraction':c.stats(tie,g,('fraction_x100',)),
                                 'GT_mass_error_abs_ppm_median':float(np.median(np.abs(ppm[np.arange(n),r['gt_slots']])))}
        arrays[f'mass_sigma{sigma}']=v;arrays[f'tie_fraction_sigma{sigma}']=tie
    stage=c.load(c.STAGE if split=='validation' else c.TEST/'stage1.npz')
    ensemble=c.load(c.OUT/'R3'/split/'per_query.npz')
    systems=c.load(c.CONTROL/'real.npz' if split=='validation' else c.TEST/'real.npz')
    method_names=[*[f'sccf_seed{s}' for s in c.SEEDS],'sccf_ensemble',*[f'msalign_seed{s}' for s in c.SEEDS],'msalign_ensemble',*[f'embcos_seed{s}' for s in c.SEEDS],'embcos_ensemble','selected']
    reports={};gains={};fullfits={}
    for name in method_names:
        cached=folder/(name+'.json');archive=folder/(name+'.npz')
        if cached.exists() and archive.exists():
            report=c.read(cached);v=c.load(archive);assert np.array_equal(v['identifiers'],ids)
            reports[name]=report;gains[name]=v['fused']-v['baseline'];arrays[name+'_baseline']=v['baseline'];arrays[name+'_fused']=v['fused']
            if split=='validation':fullfits[name]=report['full_validation_fit']
            print('R4_REUSE',name,flush=True);continue
        if name.startswith('sccf'):
            seed=None if name.endswith('ensemble') else c.SEEDS.index(int(name.split('seed')[1]));scores=pad_sccf(stage,r,seed)
        elif name=='selected':scores=np.nan_to_num(systems['selected_scores'],nan=0.)
        elif name.endswith('_ensemble'):scores=ensemble[name+'_scores']
        else:
            method,seed=name.split('_seed');scores=c.external(split,method,int(seed))['scores'].astype(np.float64)
        local_rho=rho;counts=r['lengths'];gt=r['gt_slots'];query=np.arange(n)
        if name=='selected':
            candidate=systems['candidate_rows'];counts=np.where(systems['route5'],5,20).astype(int)
            local_rho=np.empty((n,20,3),np.float64)
            for i in range(n):
                mapping={int(row):j for j,row in enumerate(r['permuted_candidates'][i,:r['lengths'][i]])}
                local_rho[i]=rho[i,[mapping[int(row)] for row in candidate[i]]]
            gt=systems['gt_local'].copy();reachable=(gt>=0)&(gt<counts);query=np.flatnonzero(reachable)
            baseline=b['top20_support_fallback'].copy()
            # Candidate log-softmax requires a represented GT. This secondary fit
            # is conditional on GT in the deployed prefix, disclosed explicitly.
            gt=np.where(reachable,gt,0)
        else:baseline=c.metrics_scores(scores,counts,gt)
        fused=baseline.copy();fits={}
        if split=='validation':
            for holdout in [0,1]:
                train=query[fold[query]!=holdout];test=query[fold[query]==holdout]
                w,fit=fit_fusion(scores,local_rho,counts,gt,train);fits[f'holdout{holdout}']=fit
                fused[test]=c.metrics_scores(w[0]*scores[test]+np.einsum('ijk,k->ij',local_rho[test],w[1:]),counts[test],gt[test])
            w,full=fit_fusion(scores,local_rho,counts,gt,query);fullfits[name]=full
        else:
            saved=c.read(c.OUT/'R4/validation/final_weights.json');full=saved[name];w=np.array(full['weights_t_w001_w01_w1'])
            fused[query]=c.metrics_scores(w[0]*scores[query]+np.einsum('ijk,k->ij',local_rho[query],w[1:]),counts[query],gt[query])
        if name=='selected':
            assert np.array_equal(fused[:,2],baseline[:,2])
        report={'baseline':c.stats(baseline,g),'fused':c.stats(fused,g),'gain_pp':c.stats(fused-baseline,g),'crossfit_fits':fits,
                'full_validation_fit':fullfits[name] if split=='validation' else full,'fit_eligible_queries':len(query),
                'GT_outside_executed_prefix':'secondary selected-system queries keep frozen metrics and are excluded from conditional likelihood fitting' if name=='selected' else None}
        gap=float((baseline[:,2]-baseline[:,0]).mean());report['R1_to_R20_gap_closure_pct']=float(100*(fused[:,0]-baseline[:,0]).mean()/gap) if gap else None
        c.write(cached,report);np.savez_compressed(archive,identifiers=ids,baseline=baseline,fused=fused)
        reports[name]=report;gains[name]=fused-baseline;arrays[name+'_baseline']=baseline;arrays[name+'_fused']=fused
        print('R4_FUSION',name,report['gain_pp']['query_weighted']['R1'],flush=True)
        guard()
    if split=='validation':c.write(folder/'final_weights.json',fullfits)
    contrasts={}
    for a,d in [('msalign_ensemble','sccf_ensemble'),('embcos_ensemble','sccf_ensemble'),('selected','msalign_ensemble')]:
        contrasts[a+'_minus_'+d]=c.stats(gains[a]-gains[d],g)
    for family in ['sccf','msalign','embcos']:
        delta=np.mean([gains[f'{family}_seed{s}'] for s in c.SEEDS],axis=0)
        arrays[family+'_seed_mean_gain']=delta
    for a,d in [('msalign','sccf'),('embcos','sccf')]:
        contrasts[a+'_seed_mean_minus_'+d+'_seed_mean']=c.stats(arrays[a+'_seed_mean_gain']-arrays[d+'_seed_mean_gain'],g)
    c.finish('R4',split,{'mass_only':descriptive,'rows':reports,'paired_differences_of_gains':contrasts,
                        'headline':{k:v['gain_pp']['query_weighted']['R1'] for k,v in reports.items() if k.endswith('ensemble') or k=='selected'},
                        'ion_offsets_da':offsets,'rdkit':rdBase.rdkitVersion,'resources_start':resources,
                        'interpretation':'continuous precursor exact-mass prior within fixed cap256; not MSMS evidence or a GT-formula oracle',
                        'secondary_selected_scope':contract['selected_policy']+'; conditional GT-in-prefix fit, GT-outside-prefix unchanged, never an inference router'},arrays)
