#!/usr/bin/env python3
"""Frozen D2: full-stack OOF trees transferred to the formal ensemble+pair.

No official-test loader. Train-only calibration gates all new formal-val scoring.
All learned fits and GPU stages are serial; source/data manifests bind phases.
"""
from __future__ import annotations
import argparse,copy,gc,hashlib,json,os,random,resource,subprocess,sys,time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import h5py
import lightgbm as lgb
import numpy as np
import torch
from rdkit import Chem,rdBase,RDLogger
from rdkit.Chem import Descriptors,rdMolDescriptors
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from glmr.candidate_indices import CandidateIndexStore,CandidateNegativeExclusion
from glmr.subformula import ion_shift
from tools.reranking.train_official_magma_forward import rows,digest,key
from tools.reranking.train_official_mpfir_oof_retriever import memory_guard,memory_state,BLOCKS,CONFIG
from tools.stage1.train_msalign_projection_smoke import MoleculeEmbeddingStore,build_projection_model
from tools.reranking.audit_official_datasafe_pair_controls import candidate_order
from tools.reranking import train_official_datasafe_pair_reranker as pairlib
from tools.reranking import train_official_lightgbm_pilot as trees
from tools.reranking.train_run_k_e1_fragment_residual import enumerate_fragment_masks,fragment_formula_features
from tools.reranking.eval_official_mpfir_v2_top5 import feasibility,masked_null_support,candidate_relative_score,candidate_status
from tools.reranking.train_official_mpfir_cross_candidate import rank_scores
from tools.analysis.analyze_stage1_logit_ensemble import comparison
from train_spec_encoder.audit_magma_peak_fragment_labels import select_dreams_peaks

SEEDS=(20260621,20260622,20260623);TREE_SEEDS=(20261010,20261011,20261012)
FAMILIES=('mining0','mining1','endpoint');PYTHON=sys.executable
FP=ROOT/'outputs/stage1_submission_0926/assets/full_sparse_direct/manifest_historical_projection_order.json'
CAND=ROOT/'outputs/run_k_full_train/candidate_index_full_v15_mass_cap256_full_order/manifest.json'
META=ROOT/'outputs/msalign_stage1_full_v15/train_full.metadata.jsonl'
DREAMS=ROOT/'outputs/run_k_official_datasafe_seed20260621/train.no_head.npy'
FORMAL=ROOT/'outputs/stage2_official_datasafe'
D1=ROOT/'outputs/mpfir_v2_official_20261002'
RDLogger.DisableLog('rdApp.*')


def write(path,data):path.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
def records(path):return list(rows(path))
def check_sources(out):
    c=json.loads((out/'freeze_manifest.json').read_text())
    for p,h in c['source_sha256'].items():assert digest(ROOT/p)==h,p
    return c

def gpu_check(out,label):
    r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True)
    assert torch.cuda.is_available();memory_guard()
    info={'stage':label,'time':time.time(),'nvidia_smi':r.stdout,'torch_gpu':torch.cuda.get_device_name(0),'resources':memory_state()}
    with (out/'gpu_preflights.jsonl').open('a') as f:f.write(json.dumps(info)+'\n')
    print(json.dumps(info),flush=True)


def freeze(out):
    assert not(out/'freeze_manifest.json').exists()
    assert lgb.__version__=='4.6.0' and rdBase.rdkitVersion=='2023.09.6'
    paths=[Path(__file__),ROOT/'tools/reranking/prepare_official_lightgbm_transfer.py',
      ROOT/'tools/reranking/train_official_mpfir_oof_retriever.py',ROOT/'tools/reranking/train_official_datasafe_pair_reranker.py',
      ROOT/'tools/reranking/train_official_lightgbm_pilot.py',ROOT/'tools/reranking/eval_official_mpfir_v2_top5.py',
      ROOT/'tools/reranking/train_run_k_e1_fragment_residual.py',ROOT/'tools/reranking/audit_official_datasafe_pair_controls.py',
      ROOT/'tools/stage1/train_msalign_projection_smoke.py',ROOT/'glmr/candidate_indices.py',ROOT/'glmr/sparse_features.py',
      ROOT/'train_spec_encoder/audit_magma_peak_fragment_labels.py',META,FP,CAND,DREAMS,
      out/'labels.jsonl',out/'official_train_connectivities.npy',out/'prepare_summary.json']
    scalar,ns=trees.score_features(np.zeros((1,5)),np.zeros((1,5)))
    ev,ne=trees.evidence_features([{'observed_spectrum':np.empty((0,2)),'precursor_mz':100.}],np.zeros((1,5,100,64),bool),np.zeros((1,100)))
    assert len(ns)==10 and len(ne)==26
    contract={'date':'2026-10-15 Asia/Shanghai','scope':'one authorized D2;8000 train/1024 new train-calibration connectivities, natural top5; no official test',
      'upstream':'3 families x3 fresh 1epoch SCCF seeds; batch128 AdamW1e-4 WD1e-4; frozen data-safe no_head/SCCF-r2;heldout GT and all candidate aliases excluded',
      'pair':'per family, original natural-loss pair recipe, hidden256/bottleneck64/dropout.1, AdamW3e-4 WD1e-4,batch1024,5 epochs; internal5fold selection/damage8%, then full complement refit;3 scorer seeds20261001..3',
      'mining':'train families exclude both outer calibration and their prediction fold from all upstream/pair GT and negatives. Tree training candidates exclude outer-cal and inherited heldouts; outer-cal natural cap256 unchanged',
      'tree':'10 scalar control vs ordered36 detail,50trees,LambdaRank;PARAMS unchanged from pilot;3 averaged scorer seeds;fit supported reachable only',
      'feature_mapping':'scalar base=SCCF3seedmean; scalar anchor=SCCF+.25z(coverage); init/deployment=pair+.25z(coverage);zstd floor.02',
      'feature_names':ns+ne,'scalar_feature_names':ns,'params':trees.PARAMS,'tree_seeds':TREE_SEEDS,
      'support':'pilot support:adduct H/Na;all5 valid connected <=60 heavy atoms;isotope-labelled candidates => tree whole-query no-op. No CE/instrument restriction, no native annotations. Outside-top5 retained, no GT insertion',
      'coverage_replay':'formal coverage retains original D1 support/numerics, including isotope rows; conservative tree fallback never alters its anchor. Train unsupported coverage is zero as in pilot',
      'calibration':'fixed50 vs0 detail continuation:deltaR1>=0,deltaMRR>0,damage<=8% vs pair+coverage. Fail stops before any D2 formal-val prediction; no refit. Scalar control fixed50 reported without tuning',
      'transfer_gate':'positive R1 cluster CI lower bounds vs BOTH pair and pair+coverage;MRR>=0 anddamage<=8% versus both;scalar/doubled controls also reported',
      'attribution':'3 frozen exact-parent/adduct/different-connectivity donor assignments;real pair and additive real coverage anchor fixed;replace scalar coverage-derived and26detail features. Require positive real-wrong R1/MRR cluster CI lows andwrong-over-anchor one-sided95% R1 upper<=.25pp;insufficient power=>INCONCLUSIVE',
      'analysis':'10000 paired connectivity bootstrap;query andconnectivity-balanced,R1/R5/R20/MRR,rescue/damage,gap closure,seed directions,support/recall,no-op,equalcoverage/formula diagnostics;no multiplicity adjustment',
      'external':'retain alreadyfrozen E1 method comparison row,not a backbone attribution;formalval alreadydevelopment-exposed',
      'promotion':'no automatic final row/paper promotion;full-system new-channel controls additionally required for any promotedendpoint',
      'no_search':'no extra trees/features/epochs/weights/caps/ppm/shortlist or seed sweep;single formal evaluation pass after calibration',
      'preflight_exposure':'existing official-val artifact schemas/baseline metadata inspected for implementation before fits; no D2 val predictions before calibration',
      'official_test_read':False,'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in paths}}
    write(out/'freeze_manifest.json',contract)
    (out/'source').mkdir()
    for p in paths:
        if p.suffix=='.py':
            dst=out/'source'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    print('D2 FROZEN',digest(out/'freeze_manifest.json'),flush=True)


def upstream(out,family,smoke):
    check_sources(out)
    assert (out/'scan_summary.json').exists()
    for seed in SEEDS[:1] if smoke else SEEDS:
        d=out/'upstream'/family/f'seed{seed}';d.mkdir(parents=True,exist_ok=True)
        for name in ['labels.jsonl','official_train_connectivities.npy']+[f'denylist_{f}' for f in FAMILIES]:
            dest=d/name
            if not dest.exists():dest.symlink_to(out/name,target_is_directory=(out/name).is_dir())
        target=d/(family+('_smoke8' if smoke else ''))
        if (target/'summary.json').exists():continue
        assert not target.exists(),'Incomplete fit must be audited explicitly before resuming'
        gpu_check(out,f'{family}/seed{seed}/smoke{smoke}')
        cmd=[PYTHON,str(ROOT/'tools/reranking/train_official_mpfir_oof_retriever.py'),'--data',str(d),'--model',family,'--seed',str(seed),'--skip-mining']
        if smoke:cmd+=['--smoke-steps','8']
        with (d/('smoke.log' if smoke else 'fit.log')).open('w') as f:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
        summary=json.loads((target/'summary.json').read_text());assert summary['oof_gt_overlap']==summary['negative_overlap']==0
        gc.collect();torch.cuda.empty_cache()


def models_for(out,family,store,device):
    result=[]
    for seed in SEEDS:
        p=out/'upstream'/family/f'seed{seed}'/family/'checkpoint.pt'
        ck=torch.load(str(p),map_location='cpu',weights_only=False)
        m=build_projection_model(spec_dim=1024,mol_dim=store.dim,mol_block_dims=store.block_dims,mol_block_names=BLOCKS,config=CONFIG).to(device)
        m.load_state_dict(ck['model_state_dict']);result.append(m.eval())
    return result


@torch.inference_mode()
def export(out,family,smoke=False):
    check_sources(out);gpu_check(out,'export/'+family)
    d=out/'upstream'/family/('export_smoke' if smoke else 'export')
    assert not d.exists();d.mkdir(parents=True)
    meta=records(META);groups=np.load(out/'official_train_connectivities.npy')
    deny=CandidateNegativeExclusion(out/f'denylist_{family}/manifest.json');outer=CandidateNegativeExclusion(out/'denylist_endpoint/manifest.json')
    excluded=set(deny.metadata['additional_connectivities'])
    train=np.flatnonzero(~np.isin(groups,list(excluded)))
    cohort=records(out/'labels.jsonl')
    pred=[r['official_row'] for r in cohort if (r['pilot_partition']=='calibration' if family=='endpoint' else r['pilot_partition']=='fit' and r['inner_fold']==int(family[-1]))]
    indices=np.r_[train,np.array(pred,dtype=int)]
    if smoke:indices=indices[np.r_[np.arange(min(32,len(train))),len(train)+np.arange(min(32,len(pred)))]];ntrain=min(32,len(train))
    else:ntrain=len(train)
    n=len(indices);assert len(set(indices.tolist()))==n
    cands=CandidateIndexStore(CAND);store=MoleculeEmbeddingStore(None,'mean',FP,mol_blocks=BLOCKS);models=models_for(out,family,store,torch.device('cuda'))
    spec=np.load(DREAMS,mmap_mode='r')
    arrays={name:np.lib.format.open_memmap(d/(name+'.npy'),mode='w+',shape=shape,dtype=dtype) for name,shape,dtype in [
      ('spec',(n,1024),'float16'),('mol',(n,5,1024),'float16'),('seed_scores',(n,5,3),'float32'),('base',(n,5),'float32'),
      ('candidate_rows',(n,5),'int32'),('labels',(n,),'int16'),('ranks',(n,),'int16')]}
    neg_leak=0;start_time=time.time()
    for start in range(0,n,32):
        if start%1024==0:memory_guard()
        rr=indices[start:start+32];pools=[]
        for j,i in enumerate(rr):
            pool=cands.get(meta[int(i)]['identifier'],256)
            # Training complement excludes this family's holdouts; tree-fit
            # prediction excludes the outer calibration identities only.
            if start+j<ntrain:pool=deny.filter_candidates(pool)[0]
            elif family!='endpoint':pool=outer.filter_candidates(pool)[0]
            assert len(pool)>=5
            order=np.random.default_rng(20261015+int(i)).permutation(len(pool));pools.append(np.array(pool,dtype=np.int32)[order])
        width=max(map(len,pools));padded=torch.tensor(np.array([list(p)+[p[0]]*(width-len(p)) for p in pools]),dtype=torch.long)
        unique,inverse=torch.unique(padded.flatten(),sorted=True,return_inverse=True)
        unique,inverse=store.canonicalize_unique_indices(unique,inverse)
        raw=store.gather(unique).to('cuda');sv=torch.from_numpy(np.array(spec[rr],copy=True)).to('cuda');scores=[];primary=None;s0=None
        for k,m in enumerate(models):
            mv=m.project_mol(raw);ss=m.project_spec(sv)
            projected=mv[inverse.to('cuda')].reshape(len(rr),width,1024)
            score=(torch.einsum('bd,bcd->bc',ss,projected)/m.temperature()).cpu().numpy();scores.append(score)
            if k==0:primary=projected;s0=ss
        stack=np.stack(scores,axis=-1);mean=stack.mean(axis=-1,dtype=np.float64).astype(np.float32)
        for j,i in enumerate(rr):
            pos=start+j;pool=pools[j];values=mean[j,:len(pool)];gt=cands.get(meta[int(i)]['identifier'],256)[0]
            order=candidate_order(meta[int(i)]['identifier'],pool,values,20260926);top=order[:5];ids=pool[top]
            rank=int(np.flatnonzero(pool[order]==gt)[0])+1;hit=np.flatnonzero(ids==gt)
            arrays['spec'][pos]=s0[j].cpu().half().numpy();arrays['mol'][pos]=primary[j,torch.from_numpy(top).to('cuda')].cpu().half().numpy()
            arrays['seed_scores'][pos]=stack[j,top];arrays['base'][pos]=values[top];arrays['candidate_rows'][pos]=ids;arrays['labels'][pos]=int(hit[0]) if len(hit) else -1;arrays['ranks'][pos]=rank
            if pos<ntrain:neg_leak+=int(deny.mask[ids[ids!=gt]].sum())
            elif family!='endpoint':neg_leak+=int(outer.mask[ids[ids!=gt]].sum())
        if start%1024==0:print(json.dumps({'export':family,'queries':min(start+32,n),'total':n,'seconds':time.time()-start_time,**memory_state()}),flush=True)
    assert neg_leak==0
    for x in arrays.values():x.flush()
    np.save(d/'official_rows.npy',indices);np.save(d/'groups.npy',groups[indices])
    write(d/'summary.json',{'family':family,'train_queries':ntrain,'prediction_queries':n-ntrain,'queries':n,'negative_leakage':0,
       'outer_holdout_positive_overlap':int(len(set(groups[indices[:ntrain]])&excluded)),'plumbing_only':smoke,
       'seconds':time.time()-start_time,'files_sha256':{p.name:digest(p) for p in d.glob('*.npy')}})
    assert not(set(groups[indices[:ntrain]])&excluded)


def make_split(d,positions):
    ids=np.load(d/'official_rows.npy',mmap_mode='r')[positions];meta=records(META)
    return pairlib.Split([meta[int(i)]['identifier'] for i in ids],np.load(d/'groups.npy',mmap_mode='r')[positions],
       torch.from_numpy(np.load(d/'spec.npy',mmap_mode='r')[positions].copy()),
       torch.from_numpy((np.array(positions)[:,None]*5+np.arange(5)[None]).astype(np.int64)),
       torch.from_numpy(np.load(d/'seed_scores.npy',mmap_mode='r')[positions].copy()),
       torch.from_numpy(np.load(d/'base.npy',mmap_mode='r')[positions].copy()),
       torch.from_numpy(np.load(d/'labels.npy',mmap_mode='r')[positions].astype(np.int64)),
       np.load(d/'ranks.npy',mmap_mode='r')[positions].astype(np.int32),{})


def pair_fit(out,family):
    check_sources(out);gpu_check(out,'pair/'+family);d=out/'upstream'/family/'export';summary=json.loads((d/'summary.json').read_text());assert not summary['plumbing_only']
    result=out/'upstream'/family/'pair';assert not result.exists();result.mkdir()
    n=summary['train_queries'];total=summary['queries'];train=make_split(d,np.arange(n));pred=make_split(d,np.arange(n,total))
    mol=torch.from_numpy(np.load(d/'mol.npy',mmap_mode='r').reshape(-1,1024))
    cfg=SimpleNamespace(hidden_dim=256,bottleneck_dim=64,dropout=.1,lr=3e-4,weight_decay=1e-4,batch_size=1024,max_epochs=5,
      calibration_folds=5,calibration_fold=0,split_seed=20261001,damage_cap=.08,training_balance='natural')
    means=[];hist={};start=time.time()
    for seed in (20261001,20261002,20261003):
        memory_guard();epoch,history=pairlib.select_epoch(train,mol,seed,cfg,torch.device('cuda'))
        model=pairlib.fit_full(train,mol,epoch,seed,cfg,torch.device('cuda'))
        score=pairlib.collect_scores(model,pred,mol,2048,torch.device('cuda'));means.append(score)
        hist[str(seed)]={'epoch':epoch,'history':history};torch.save({'model_state_dict':model.state_dict(),'selected_epochs':epoch,'model_seed':seed},result/f'seed{seed}.pt')
        print(json.dumps({'pair':family,'seed':seed,'selected_epoch':epoch,'seconds':time.time()-start}),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    np.save(result/'prediction_scores.npy',np.mean(np.stack(means),axis=0))
    np.save(result/'prediction_seed_scores.npy',np.stack(means));np.save(result/'official_rows.npy',np.load(d/'official_rows.npy')[n:])
    write(result/'summary.json',{'family':family,'history':hist,'config':vars(cfg),'train_queries':n,'prediction_queries':total-n,
        'outer_holdout_positive_overlap':summary['outer_holdout_positive_overlap'],'negative_overlap':summary['negative_leakage'],'seconds':time.time()-start,
        'export_summary_sha256':digest(d/'summary.json'),'prediction_sha256':digest(result/'prediction_scores.npy')})


class Evidence:
    """Mass-only acceleration of unchanged pilot proposals and feasibility."""
    def __init__(self):self.cache={}
    def molecule(self,s):
        if s not in self.cache:
            status,parent=candidate_status(s);mass=np.zeros(64,np.float32);mask=np.zeros(64,bool);isotope=False;formula='';conn=''
            if status=='supported':
                mol=Chem.MolFromSmiles(s);isotope=any(a.GetIsotope() for a in mol.GetAtoms());formula=rdMolDescriptors.CalcMolFormula(mol);conn=Chem.MolToInchiKey(mol)[:14]
                fragments=enumerate_fragment_masks(mol,64)
                values=[fragment_formula_features(mol,a,c)[2] for a,c in fragments]
                # Match PackedFragmentStore's float32 scalar /1000 *1000.
                mass[:len(values)]=(np.array(values,np.float32)/np.float32(1000))*np.float32(1000);mask[:len(values)]=True
            self.cache[s]=(status,parent,mass,mask,isotope,formula,conn)
        return self.cache[s]
    def build(self,inputs,base,coverage_override=None):
        # Canonical SCCF-rank arithmetic keeps the original pilot values while
        # making physical candidate permutations exactly invariant.
        order=np.array([candidate_order(q['identifier'],np.array(q['candidate_global_rows']),np.asarray(base)[i],20260926) for i,q in enumerate(inputs)])
        inverse=np.argsort(order,axis=1);canonical=[]
        for q,perm in zip(inputs,order):
            value=dict(q);value['candidate_smiles']=[q['candidate_smiles'][j] for j in perm];value['candidate_global_rows']=[q['candidate_global_rows'][j] for j in perm];canonical.append(value)
        new_base=np.take_along_axis(np.asarray(base),order,axis=1)
        new_coverage=np.take_along_axis(coverage_override,order,axis=1) if coverage_override is not None else None
        result=self._build(canonical,new_base,new_coverage)
        for name in ('features','scalar','coverage','computed_coverage','valid'):
            value=result[name];idx=inverse.reshape(inverse.shape+(1,)*(value.ndim-2));result[name]=np.take_along_axis(value,idx,axis=1)
        return result
    def _build(self,inputs,base,coverage_override=None):
        n=len(inputs);valid=np.zeros((n,5,100,64),bool);mz=torch.zeros((n,100));intensity=torch.zeros((n,100));present=torch.zeros((n,100),dtype=torch.bool)
        support=np.ones(n,bool);coverage_support=np.ones(n,bool);reasons=Counter();mass=np.zeros((n,5,64),np.float32);masks=np.zeros((n,5,64),bool);parent=np.zeros((n,5),np.float32)
        for i,q in enumerate(inputs):
            observed=q['observed_spectrum'];a=torch.tensor(observed,dtype=torch.float32)
            if observed:mz[i,:len(a)]=a[:,0];intensity[i,:len(a)]=a[:,1];present[i,:len(a)]=True
            reason='supported'
            if q['adduct'] not in ('[M+H]+','[M+Na]+'):reason='unsupported_adduct'
            for j,s in enumerate(q['candidate_smiles']):
                status,pa,ma,mask,iso,_,_=self.molecule(s)
                if status!='supported':reason=status
                else:parent[i,j]=pa;mass[i,j]=ma;masks[i,j]=mask
            if reason!='supported':support[i]=coverage_support[i]=False
            elif any(self.molecule(s)[4] for s in q['candidate_smiles']):support[i]=False;reason='candidate_isotope_mass_unsupported'
            if not observed or not bool((intensity[i]>0).any()):support[i]=coverage_support[i]=False;reason='invalid_observed_spectrum'
            reasons[reason]+=1
        for st in range(0,n,32):
            ix=np.flatnonzero((coverage_support if coverage_override is not None else support)[st:st+32])+st
            if len(ix):valid[ix],_=feasibility(mz[ix],torch.tensor([inputs[i]['precursor_mz'] for i in ix]),
               torch.tensor([ion_shift(inputs[i]['adduct']) for i in ix]),torch.from_numpy(mass[ix]),torch.from_numpy(parent[ix]),torch.from_numpy(masks[ix]),present[ix])
        cv=candidate_relative_score(masked_null_support(torch.ones(valid.shape),torch.from_numpy(valid)),present,intensity).numpy()
        # Training follows the pilot's conservative isotope fallback. Formal
        # coverage_override preserves its already-frozen original D1 readout.
        computed=cv.copy()
        if coverage_override is None:cv[~support]=0
        else:cv=np.asarray(coverage_override).copy()
        scalar,ns=trees.score_features(base,add_coverage(base,cv));feature_inputs=[dict(q,observed_spectrum=np.asarray(q['observed_spectrum']).reshape(-1,2)) for q in inputs];detail,nd=trees.evidence_features(feature_inputs,valid,intensity.numpy())
        return {'features':np.concatenate([scalar,detail],-1),'scalar':scalar,'names':ns+nd,'coverage':cv,'support':support,
          'computed_coverage':computed,'coverage_support':coverage_support,'valid':valid,'intensity':intensity.numpy(),'support_counts':dict(reasons)}


def add_coverage(base,coverage,multiplier=1.):
    t=torch.as_tensor(coverage);z=(t-t.mean(1,keepdim=True))/t.std(1,unbiased=False,keepdim=True).clamp_min(.02)
    return np.asarray(base)+np.float32(.25*multiplier)*z.numpy()


def feature_parity(out):
    check_sources(out);p=torch.load(str(ROOT/'outputs/mpfir_cross_candidate_official_20261010/features/data.pt'),map_location='cpu',weights_only=False,mmap=True)
    ix=np.arange(min(128,len(p['inputs'])));inputs=[p['inputs'][i] for i in ix];base=p['base'][ix].numpy();anchor=add_coverage(base,p['coverage'][ix].numpy())
    old_scalar,ns=trees.score_features(base,anchor);old_detail,nd=trees.evidence_features(inputs,p['valid'][ix],p['intensity'][ix])
    fresh=Evidence().build(inputs,base)
    errors={'features':float(np.abs(np.concatenate([old_scalar,old_detail],-1)-fresh['features']).max()),
      'coverage':float(np.abs(p['coverage'][ix].numpy()-fresh['coverage']).max()),'valid_disagreements':int(np.count_nonzero(p['valid'][ix].numpy()!=fresh['valid']))}
    assert errors['features']==errors['coverage']==errors['valid_disagreements']==0,errors
    assert fresh['names']==json.loads((out/'freeze_manifest.json').read_text())['feature_names']
    # Actually permute within-list candidate inputs before rebuilding features.
    perm=np.array([2,4,0,3,1]);new=[]
    for q in inputs:
        v=dict(q);v['candidate_smiles']=[q['candidate_smiles'][i] for i in perm];v['candidate_global_rows']=[q['candidate_global_rows'][i] for i in perm];new.append(v)
    alternate=Evidence().build(new,base[:,perm]);assert np.array_equal(alternate['features'],fresh['features'][:,perm])
    write(out/'feature_plumbing_audit.json',{'status':'PASS','queries':len(ix),'errors':errors,'actual_candidate_permutation_exact':True,'rdkit':rdBase.rdkitVersion})


def train_features(out):
    check_sources(out);assert json.loads((out/'feature_plumbing_audit.json').read_text())['status']=='PASS'
    selected=records(out/'labels.jsonl');lookup={r['official_row']:i for i,r in enumerate(selected)};n=len(selected)
    arrays={k:np.zeros(sh,dtype=dt) for k,sh,dt in [('base',(n,5),np.float32),('pair',(n,5),np.float32),('candidates',(n,5),np.int32),('rank',(n,),np.int32),('label',(n,),np.int16)]}
    seen=set()
    for family in FAMILIES:
        d=out/'upstream'/family/'export';summary=json.loads((d/'summary.json').read_text());st=summary['train_queries'];official=np.load(d/'official_rows.npy')[st:]
        pair=np.load(out/'upstream'/family/'pair/prediction_scores.npy')
        for j,r in enumerate(official):
            i=lookup[int(r)];assert i not in seen;seen.add(i)
            for field,source in [('base','base'),('candidates','candidate_rows'),('rank','ranks'),('label','labels')]:arrays[field][i]=np.load(d/(source+'.npy'),mmap_mode='r')[st+j]
            arrays['pair'][i]=pair[j]
    assert len(seen)==n
    wanted=set(arrays['candidates'].flatten().tolist());smiles={}
    with (ROOT/'outputs/run_k_full_train/full_sparse/full_smiles.txt').open() as f:
        for i,line in enumerate(f):
            if i in wanted:smiles[i]=line.rstrip('\n')
    assert set(smiles)==wanted
    inputs=[]
    with h5py.File(ROOT/'outputs/msalign_stage1_full_v15/train_full.hdf5','r') as f:
        for i,r in enumerate(selected):
            j=r['official_row'];title=f['TITLE'][j];title=title.decode() if isinstance(title,bytes) else str(title);assert title==r['identifier']
            observed=f['spectrum'][j];valid=(observed[0]>0)&np.isfinite(observed).all(axis=0)&(observed[1]>=0)
            peak=select_dreams_peaks(observed[0,valid].tolist(),observed[1,valid].tolist(),100)
            inputs.append({'identifier':r['identifier'],'adduct':r['adduct'],'precursor_mz':r['precursor_mz'],
               'observed_spectrum':[[p['mz'],p['relative_intensity']] for p in peak],
               'candidate_global_rows':arrays['candidates'][i].tolist(),'candidate_smiles':[smiles[int(c)] for c in arrays['candidates'][i]]})
    built=Evidence().build(inputs,arrays['base']);anchor=deployment_anchor(arrays['pair'],built['coverage'])
    d=out/'features';assert not d.exists();d.mkdir()
    np.savez_compressed(d/'train.npz',**arrays,features=built['features'],scalar=built['scalar'],coverage=built['coverage'],anchor=anchor,support=built['support'],
       parts=np.array([r['pilot_partition'] for r in selected]),groups=np.array([r['connectivity'] for r in selected]),ids=np.array([r['identifier'] for r in selected]))
    with (d/'inputs.jsonl').open('w') as f:
        for q in inputs:f.write(json.dumps(q)+'\n')
    write(d/'summary.json',{'queries':n,'support_counts':built['support_counts'],'source_sha256':digest(d/'train.npz'),'feature_names':built['names'],
      'full_stack_OOF':True,'outer_calibration_positive_or_negative_overlap':0,'official_test_read':False,'D2_validation_predictions':False})


def rank_from(inputs,labels,baseline,scores):
    return rank_scores(inputs,[{'baseline_rank':int(r),'gt_position':int(g)} for r,g in zip(baseline,labels)],scores)


def fit_trees(out):
    check_sources(out);d=out/'trees';assert not d.exists();d.mkdir();z=dict(np.load(out/'features/train.npz'));inputs=records(out/'features/inputs.jsonl')
    fit=np.flatnonzero((z['parts']=='fit')&z['support']&(z['label']>=0));cal=np.flatnonzero(z['parts']=='calibration')
    assert set(z['groups'][fit]).isdisjoint(z['groups'][cal])
    permutation=np.array([np.random.default_rng(20261010+int(i)).permutation(5) for i in fit]);relevance=(np.arange(5)[None]==z['label'][:,None]).astype(np.int32)
    results={};raw={};before=rank_from([inputs[i] for i in cal],z['label'][cal],z['rank'][cal],z['anchor'][cal]);start=time.time()
    for arm,feature in [('scalar',z['scalar']),('detail',z['features'])]:
        means=[];seedmetrics={}
        for seed in TREE_SEEDS:
            params=dict(trees.PARAMS,seed=seed,feature_fraction_seed=seed,data_random_seed=seed,bagging_seed=seed)
            dataset=lgb.Dataset(np.take_along_axis(feature[fit],permutation[:,:,None],axis=1).reshape(-1,feature.shape[-1]),
              label=np.take_along_axis(relevance[fit],permutation,axis=1).flatten(),group=np.full(len(fit),5),
              init_score=np.take_along_axis(z['anchor'][fit],permutation,axis=1).flatten(),free_raw_data=False)
            model=lgb.train(params,dataset,num_boost_round=50);model.save_model(str(d/f'{arm}_seed{seed}.txt'))
            residual=model.predict(feature[cal].reshape(-1,feature.shape[-1]),num_threads=2).reshape(-1,5);residual[~z['support'][cal]]=0;means.append(residual)
            score=z['anchor'][cal]+residual;r=rank_from([inputs[i] for i in cal],z['label'][cal],z['rank'][cal],score)
            seedmetrics[str(seed)]=trees.calibration(before,r)
            assert np.array_equal(lgb.Booster(model_file=str(d/f'{arm}_seed{seed}.txt')).predict(feature[cal].reshape(-1,feature.shape[-1]),num_threads=2),model.predict(feature[cal].reshape(-1,feature.shape[-1]),num_threads=2))
        score=z['anchor'][cal]+np.mean(np.stack(means),axis=0);r=rank_from([inputs[i] for i in cal],z['label'][cal],z['rank'][cal],score)
        metric=trees.calibration(before,r);results[arm]={'metrics':metric,'seed_metrics':seedmetrics};raw[arm]=score
    metric=results['detail']['metrics'];passed=metric['delta_r1']>=0 and metric['delta_mrr']>0 and metric['damage']<=.08
    np.savez_compressed(d/'calibration.npz',indices=cal,before_ranks=before,**raw)
    write(d/'summary.json',{'status':'CALIBRATION_PASS' if passed else 'STOP_AT_TRAIN_CALIBRATION','detail_continuation_pass':bool(passed),
       'trees':50,'seeds':TREE_SEEDS,'fit_reachable_supported_queries':len(fit),'calibration_queries':len(cal),'arms':results,
       'seconds':time.time()-start,'official_test_read':False,'D2_validation_predictions':False,'model_files_sha256':{p.name:digest(p) for p in d.glob('*.txt')}})
    print(json.dumps(json.loads((d/'summary.json').read_text()),indent=2),flush=True)


def residuals(out,features,support,arm):
    k=features.shape[-1];values=[]
    for seed in TREE_SEEDS:
        model=lgb.Booster(model_file=str(out/'trees'/f'{arm}_seed{seed}.txt'))
        r=model.predict(features.reshape(-1,k),num_threads=2).reshape(-1,5);r[~support]=0;values.append(r)
    return np.mean(np.stack(values),axis=0),np.stack(values)


def deployment_anchor(pair,coverage,multiplier=1.):
    result=np.asarray(pair).copy();v=np.asarray(coverage);z=(v-v.mean(1,keepdims=True))/np.maximum(v.std(1,keepdims=True),.02)
    result+=(np.float32(.25*multiplier)*z).astype(np.float32);return result


def one_sided_upper(before,after,groups,seed):
    unique,inverse=np.unique(groups,return_inverse=True);delta=(after==1).astype(float)-(before==1)
    sums=np.bincount(inverse,weights=delta);sizes=np.bincount(inverse);rng=np.random.default_rng(seed);values=[]
    for st in range(0,10000,128):
        ix=rng.integers(0,len(unique),size=(min(128,10000-st),len(unique)));values.extend((sums[ix].sum(1)/sizes[ix].sum(1)*100).tolist())
    return float(np.quantile(values,.95))


def formal(out):
    check_sources(out);gate=json.loads((out/'trees/summary.json').read_text());assert gate['detail_continuation_pass'],'Calibration failed: no new formal-val scoring'
    dest=out/'formal_validation';assert not dest.exists();dest.mkdir();started=time.time()
    old=dict(np.load(D1/'d1_val_formal/per_query.npz'));inputs=records(D1/'d1_inputs_val/inputs.jsonl')
    cache=torch.load(str(FORMAL/'top5_projection_cache/seed20260621/val_top5.pt'),map_location='cpu',weights_only=False,mmap=True)
    assert old['identifiers'].tolist()==cache['identifiers']==[q['identifier'] for q in inputs]
    base=cache['ensemble_scores'].numpy();label=cache['labels'].numpy();baseline=cache['baseline_ranks'].numpy();groups=np.array(cache['connectivities']);pair=old['pair_scores']
    assert np.array_equal(old['candidate_global_rows'],cache['candidate_global_rows'].numpy())
    p=rank_from(inputs,label,baseline,pair);ref=np.load(FORMAL/'pair_reranker/formal_top5/val_ranks.npz')['reranked_ranks'];assert np.array_equal(p,ref)
    anchor=deployment_anchor(pair,old['coverage_local']);cv_rank=rank_from(inputs,label,baseline,anchor);assert np.array_equal(cv_rank,old['coverage_ranks'])
    ev=Evidence();built=ev.build(inputs,base,coverage_override=old['coverage_local'])
    assert built['names']==json.loads((out/'freeze_manifest.json').read_text())['feature_names']
    supported=built['support'];recomputed_error=float(np.abs(built['computed_coverage']-old['coverage_local']).max());assert recomputed_error<=1e-6,('coverage replay failed',recomputed_error)
    scalar,scalar_seeds=residuals(out,built['scalar'],supported,'scalar');detail,detail_seeds=residuals(out,built['features'],supported,'detail')
    scores={'pair':pair,'coverage':anchor,'doubled_coverage':deployment_anchor(pair,old['coverage_local'],2.),'scalar':anchor+scalar,'detail':anchor+detail}
    ranks={k:rank_from(inputs,label,baseline,v) for k,v in scores.items()}
    assert all(np.array_equal(r<=5,baseline<=5) and np.array_equal(r<=20,baseline<=20) for r in ranks.values())
    report={'status':'COMPLETE_TRANSFER_EVALUATION','queries':len(inputs),'connectivities':len(set(groups)),
      'support':{'supported_queries':int(supported.sum()),'reachable_queries':int((label>=0).sum()),'counts':built['support_counts']},
      'baseline_replay':{'pair_exact':True,'coverage_exact':True,'recomputed_local_max_error':recomputed_error},
      'comparisons':{},'seed_directions':{},'official_test_read':False,'D2_validation_predictions':True}
    for name in ('pair','coverage','doubled_coverage','scalar'):
        report['comparisons']['detail_vs_'+name]=comparison(ranks[name],ranks['detail'],groups,10000,20261015)
    for arm,ss in [('detail',detail_seeds),('scalar',scalar_seeds)]:
        report['seed_directions'][arm]=[]
        for seed,r in zip(TREE_SEEDS,ss):
            rr=rank_from(inputs,label,baseline,anchor+r);report['seed_directions'][arm].append({'seed':seed,**trees.calibration(ranks['coverage'],rr)})
    ii=np.arange(min(128,len(inputs)));perm=np.array([2,4,0,3,1]);permuted=[]
    for i in ii:
        q=dict(inputs[i]);q['candidate_smiles']=[q['candidate_smiles'][j] for j in perm];q['candidate_global_rows']=[q['candidate_global_rows'][j] for j in perm];permuted.append(q)
    rebuilt=ev.build(permuted,base[ii][:,perm],coverage_override=old['coverage_local'][ii][:,perm]);assert np.array_equal(rebuilt['features'],built['features'][ii][:,perm])
    rp,_=residuals(out,rebuilt['features'],rebuilt['support'],'detail');assert np.array_equal(rp,detail[ii][:,perm]);report['permutation_invariance_exact']=True
    checks={}
    for name in ('pair','coverage'):
        comp=report['comparisons']['detail_vs_'+name];ci=comp['paired_bootstrap']['connectivity_cluster_query_weighted']['recall@1']
        checks[name]={'positive_R1_cluster_CI':ci['ci95_low_pp']>0,'MRR_nonnegative':comp['delta_query_weighted_pp']['mrr']>=0,'damage_le_8pct':comp['rank1_transition']['baseline_rank1_damage_rate_pct']<=8.}
    report['transfer_gate_checks']=checks;report['transfer_gate_pass']=all(all(x.values()) for x in checks.values())
    np.savez_compressed(dest/'scores_ranks.npz',identifiers=old['identifiers'],groups=groups,labels=label,baseline_ranks=baseline,supported=supported,
      candidate_rows=old['candidate_global_rows'],**{'scores_'+k:v for k,v in scores.items()},**{'ranks_'+k:v for k,v in ranks.items()})
    hits=built['valid'].any(-1).sum(-1);slices={k:[] for k in ('equal_count','equal_count_same_formula','equal_score','equal_score_same_formula')}
    for i,q in enumerate(inputs):
        gt=int(label[i])
        if gt<0 or not supported[i]:continue
        true=ev.molecule(q['candidate_smiles'][gt]);wrong=[j for j,s in enumerate(q['candidate_smiles']) if j!=gt and ev.molecule(s)[6]!=groups[i]]
        if not wrong:continue
        decoy=max(wrong,key=lambda j:float(old['coverage_local'][i,j]));same=true[5]==ev.molecule(q['candidate_smiles'][decoy])[5]
        count=hits[i,gt]>0 and hits[i,gt]==hits[i,decoy];equal=hits[i,gt]>0 and hits[i,decoy]>0 and abs(old['coverage_local'][i,gt]-old['coverage_local'][i,decoy])<=1e-6
        for name,ok in [('equal_count',count),('equal_score',equal)]:
            if ok:
                value={k:float((s[i,gt]>s[i,decoy])+.5*(s[i,gt]==s[i,decoy])) for k,s in scores.items()};slices[name].append(value)
                if same:slices[name+'_same_formula'].append(value)
    report['descriptive_equal_coverage']={k:{'queries':len(v),'win_rates':{name:float(np.mean([r[name] for r in v])) if v else None for name in scores}} for k,v in slices.items()}
    donor_path=ROOT/'outputs/stage1_submission_0926/runs/full_r2_block_l2/seed20260621/blind_controls/spectrum_permutation_exact_mass.per_query.npz'
    donors=dict(np.load(donor_path));rec=donors['recipient_rows'].astype(int);report['wrong_channel']={};metadata=records(ROOT/'outputs/msalign_stage1_full_v15/val_full.metadata.jsonl')
    for seed in (20260820,20260821,20260822):
        dd=donors[f'seed{seed}_donor_rows'].astype(int);assert set(dd)==set(rec) and np.all(groups[rec]!=groups[dd]);wrong=[]
        for i,j in zip(rec,dd):
            assert metadata[i]['adduct']==metadata[j]['adduct'] and round(float(metadata[i]['parent_mass']),6)==round(float(metadata[j]['parent_mass']),6)
            q=dict(inputs[i]);q['observed_spectrum']=inputs[j]['observed_spectrum'];wrong.append(q)
        w=ev.build(wrong,base[rec],coverage_override=np.zeros((len(rec),5),np.float32));w=ev.build(wrong,base[rec],coverage_override=w['computed_coverage'])
        wr,_=residuals(out,w['features'],supported[rec],'detail');wrong_score=anchor[rec]+wr;wrong_rank=rank_from([inputs[i] for i in rec],label[rec],baseline[rec],wrong_score)
        real_gain=comparison(wrong_rank,ranks['detail'][rec],groups[rec],10000,seed);wrong_gain=comparison(ranks['coverage'][rec],wrong_rank,groups[rec],10000,seed);upper=one_sided_upper(ranks['coverage'][rec],wrong_rank,groups[rec],seed)
        ci=real_gain['paired_bootstrap']['connectivity_cluster_query_weighted'];checks={'real_minus_wrong_R1_CI_low_positive':ci['recall@1']['ci95_low_pp']>0,
          'real_minus_wrong_MRR_CI_low_positive':ci['mrr']['ci95_low_pp']>0,'wrong_R1_uplift_upper95_le_margin':upper<=.25}
        report['wrong_channel'][str(seed)]={'queries':len(rec),'connectivities':len(set(groups[rec])),'real_minus_wrong':real_gain,'wrong_minus_fixed_real_anchor':wrong_gain,
          'wrong_uplift_one_sided95_upper_pp':upper,'margin_pp':.25,'checks':checks,'gate_pass':all(checks.values()),'scope':'realupstream/pair andadditive realcoverage anchor fixed;allnewchannel coverage-derived features recomputed'}
        np.savez_compressed(dest/f'wrong_seed{seed}.npz',recipient_rows=rec,donor_rows=dd,real_ranks=ranks['detail'][rec],wrong_ranks=wrong_rank,wrong_scores=wrong_score)
    report['attribution_gate_pass']=all(x['gate_pass'] for x in report['wrong_channel'].values());report['attribution_status']='PASS' if report['attribution_gate_pass'] else 'INCONCLUSIVE_OR_FAILED_FROZEN_GATE'
    report['status']='TRANSFER_COMPONENT_PASS' if report['transfer_gate_pass'] else 'STOP_TRANSFER_THIS_CONFIGURATION';report['promotion']='NOT_PROMOTED;full-system controls/final row freeze remain separate'
    report['seconds']=time.time()-started;write(dest/'summary.json',report)
    print(json.dumps({'status':report['status'],'attribution':report['attribution_status'],'comparisons':{k:v['delta_query_weighted_pp'] for k,v in report['comparisons'].items()}},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','upstream','export','pair','feature-parity','features','trees','formal'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--family',choices=FAMILIES,default='mining0');p.add_argument('--smoke',action='store_true');a=p.parse_args();a.out=a.out.resolve()
    if a.stage=='freeze':freeze(a.out)
    elif a.stage=='upstream':upstream(a.out,a.family,a.smoke)
    elif a.stage=='export':export(a.out,a.family,a.smoke)
    elif a.stage=='pair':pair_fit(a.out,a.family)
    elif a.stage=='feature-parity':feature_parity(a.out)
    elif a.stage=='features':train_features(a.out)
    elif a.stage=='trees':fit_trees(a.out)
    else:formal(a.out)

if __name__=='__main__':main()
