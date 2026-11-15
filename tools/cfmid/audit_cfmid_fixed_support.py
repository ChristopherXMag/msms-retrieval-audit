#!/usr/bin/env python3
"""Train-only small CFM-ID intensity-value audit with fixed candidates/mass edges.

Phases prepare -> predict -> analyze. prepare freezes the data/source/score and
analysis contract before any teacher prediction on the cohort. This is a
mechanism pilot conditional on the teacher's retained mass support, not a
student fit, forward-model benchmark, or final-system reranking evaluation.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
from rdkit import Chem, RDLogger, rdBase
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from glmr.embcos import sha256_file,write_json

SEED=20261015
UPSTREAM=ROOT/'outputs/mpfir_cross_candidate_official_20261010'
GEOMETRY=ROOT/'outputs/local_spectral_geometry_official_20261015'
SMILES=ROOT/'outputs/run_k_full_train/full_sparse/full_smiles.txt'
RDLogger.DisableLog('rdApp.*')


def rows(path):
    return [json.loads(line) for line in path.open()]


def keyed(text):return hashlib.sha256(f'{SEED}|{text}'.encode()).hexdigest()


def verify(out):
    c=json.loads((out/'contract.json').read_text())
    for name,h in c['input_source_sha256'].items():assert sha256_file(ROOT/name)==h,name
    assert sha256_file(out/'queries.jsonl')==c['queries_sha256']
    assert sha256_file(out/'inference_inputs.jsonl')==c['inference_inputs_sha256']
    return c


def prepare(out):
    assert not (out/'contract.json').exists()
    assert rdBase.rdkitVersion=='2025.09.6'
    metadata=rows(UPSTREAM/'query_metadata.jsonl')
    top=[x for x in rows(UPSTREAM/'endpoint/top5.jsonl') if x['partition']=='evaluation']
    records=rows(GEOMETRY/'train.rows.jsonl')
    assert len(records)==194119 and all(m['fold']=='train' for m in metadata)
    indexed={r['identifier']:r for r in records}
    wanted={i for x in top for i in x['candidate_global_rows']}
    smi={}
    with SMILES.open() as f:
        for i,line in enumerate(f):
            if i in wanted:smi[i]=line.rstrip('\n')
    assert set(smi)==wanted
    exclusions=Counter();pool=[]
    for x in top:
        m=metadata[x['query_row']];r=indexed[x['identifier']]
        assert m['identifier']==x['identifier'] and r['connectivity']==x['connectivity']
        if m['instrument_type']!='QTOF':exclusions['not_QTOF']+=1;continue
        if m['adduct']!='[M+H]+':exclusions['not_protonated']+=1;continue
        if m['collision_energy'] not in (10.,20.,40.):exclusions['CE_not_10_20_40']+=1;continue
        if r['n_retained_peaks']<5:exclusions['query_fewer_than_5_local_peaks']+=1;continue
        structures=[smi[i] for i in x['candidate_global_rows']]
        molecules=[Chem.MolFromSmiles(s) for s in structures]
        reason=None;groups=[]
        for mol in molecules:
            if mol is None:reason='candidate_invalid';break
            if len(Chem.GetMolFrags(mol))!=1:reason='candidate_multicomponent';break
            if not 1<=mol.GetNumHeavyAtoms()<=35:reason='candidate_heavy_atoms_above_35';break
            if any(a.GetAtomicNum() not in (1,6,7,8,9,15,16,17,35,53) for a in mol.GetAtoms()):reason='candidate_element';break
            if any(a.GetIsotope() for a in mol.GetAtoms()):reason='candidate_isotope';break
            if Chem.GetFormalCharge(mol)!=0:reason='candidate_non_neutral';break
            groups.append(Chem.MolToInchiKey(mol)[:14])
        if reason:exclusions[reason]+=1;continue
        if len(set(groups))!=5:exclusions['candidate_duplicate_connectivity']+=1;continue
        pool.append({**x,'metadata':m,'raw_train_row':r['row'],'precursor_mz':r['precursor_mz'],
                     'collision_energy':m['collision_energy'],'energy_index':{10.:0,20.:1,40.:2}[m['collision_energy']],
                     'candidate_smiles':structures,'candidate_connectivities':groups})
    selected=sorted(pool,key=lambda x:keyed(x['identifier']))[:32]
    assert len(selected)==32
    # Strict, deterministic wrong-spectrum donors; no mass-window relaxation.
    donor_index=defaultdict(list)
    for r in records:
        if r['instrument_type']=='QTOF' and r['adduct']=='[M+H]+' and r['collision_energy'] in (10.,20.,40.) and r['n_retained_peaks']>=5:
            donor_index[(r['precursor_mz'],r['collision_energy'])].append(r)
    used=set();selected_ids={q['identifier'] for q in selected}
    inputs=[]
    for q in selected:
        q['candidate_order']=np.random.default_rng(int(keyed(q['identifier']+'order')[:16],16)).permutation(5).tolist()
        donors=[r for r in donor_index[(q['precursor_mz'],q['collision_energy'])]
                if r['connectivity']!=q['connectivity'] and r['identifier'] not in used|selected_ids]
        donor=min(donors,key=lambda r:keyed(q['identifier']+'|'+r['identifier'])) if donors else None
        q['wrong_spectrum_row']=donor['row'] if donor else None
        q['wrong_spectrum_identifier']=donor['identifier'] if donor else None
        if donor:used.add(donor['identifier'])
        inputs.append({k:q[k] for k in ('identifier','precursor_mz','collision_energy','energy_index','candidate_global_rows','candidate_smiles','candidate_order')})
    with (out/'queries.jsonl').open('w') as f:
        for q in selected:f.write(json.dumps(q,sort_keys=True)+'\n')
    with (out/'inference_inputs.jsonl').open('w') as f:
        for q in inputs:f.write(json.dumps(q,sort_keys=True)+'\n')
    paths=[Path(__file__),ROOT/'tools/cfmid/fetch_cfmid_runtime.py',UPSTREAM/'endpoint/top5.jsonl',
           UPSTREAM/'endpoint/summary.json',UPSTREAM/'query_metadata.jsonl',UPSTREAM/'prepare_summary.json',
           UPSTREAM/'freeze_manifest.json',SMILES,GEOMETRY/'train.rows.jsonl',
           GEOMETRY/'train.local_mz.npy',GEOMETRY/'train.local_intensity.npy',GEOMETRY/'inventory_contract.json']
    c={'status':'FROZEN_BEFORE_COHORT_TEACHER_PREDICTIONS','date':'2026-10-15','seed':SEED,
       'scope':'32 hash-selected connectivity-distinct official-train queries, natural fixed top5 from fresh single-seed data-safe SCCF OOF miner; no candidate insertion, no student/fusion fit',
       'teacher':'CFM MSML 4.0.8 from Docker wishartlab/cfmid:4.2.6.0 image sha256:75363ee83b81ed4fdcc3db303456a528466a3fcec838fd978a81f2384802faa4; no MSRB rules',
       'teacher_provenance':'External trained-model membership not bound to official splits; train-only descriptive mechanism diagnostic, not clean/OOF teacher evidence',
       'conditions':'QTOF; [M+H]+; exact published scalar CE10/20/40 mapped to energy0/1/2; CE unit/instrument model not supplied, mapping is an assumption, no CE search',
       'support':'all five neutral connected nonisotopic candidates <=35 heavy atoms, C/H/N/O/F/P/S/Cl/Br/I, distinct connectivities; query >=5 retained peaks; no GT/rank selection',
       'generation':{'probability_pruning':.001,'apply_postproc':0,'annotate_fragments':0,'timeout_seconds_per_candidate':90,'workers':2,'memory_limit_gib_per_child':3},
       'fixed_proposal':'selected-energy positive finite CFM predicted masses below observed precursor-1.5Da; exact duplicate masses summed. Teacher-pruned support, not all chemically feasible fragments. No threshold/rank selected from query matches.',
       'query_peaks':'reuse official-train nonprecursor top100 sqrt/L2 arrays; mz < observed precursor-1.5Da',
       'fixed_match':'20ppm of predicted mass; greedy 1-to-1 edges sorted by absolute ppm residual then query and predicted mz; assignment independent of all intensities, reused unchanged in three arms',
       'score':'dot of sqrt/L2 observed intensity and predicted sqrt/L2 intensity over frozen mass edges; uniform arm substitutes all predicted retained intensities=1; permutation shuffles the exact candidate intensity multiset over same retained masses',
       'arms':['uniform fixed masses','CFM predicted intensities','200 within-candidate random permutations of CFM intensities'],
       'primary':'tie-aware unconditional R@1 teacher minus uniform, and teacher minus mean permutation; exact ties use uniform position expectation, no stable-order GT advantage',
       'secondary':['MRR retaining baseline rank when GT outside top5','reachable-only GT-vs-decoy pairwise win (tie=.5)','binary explained-intensity coverage','same exact precursor/adduct/QTOF/CE different-connectivity wrong-spectrum controls on eligible slice'],
       'fallback':'if any candidate prediction fails/times out/has no nonprecursor masses, whole query retains OOF baseline for all arms; no replacement queries or candidate dropping',
       'intervals':'5000 paired connectivity-cluster bootstrap; each selected query has unique connectivity so equivalent to query bootstrap; descriptive, no multiplicity adjustment or selection uncertainty',
       'randomization':'200 independent within-candidate permutations; finite-sample upper-tail permutation diagnostic on R@1, conditional on weights/support, not an independent cohort test',
       'decision':'small exploratory pilot only; no automatic student training, teacher deployment, larger cohort or parameter sweep. At most directional evidence conditional on teacher support/provenance/CE assumptions.',
       'existing_exposure':'prior cross-candidate/LightGBM results exposed this official-train cohort; no claims of a globally untouched evaluation sample',
       'official_validation_read':False,'official_test_read':False,'training_performed':False,'gpu_used':False,
       'population_counts':{'upstream_evaluation':len(top),'eligible':len(pool),'selected':len(selected),'exclusions':dict(exclusions),
                            'selected_CE':dict(Counter(str(q['collision_energy']) for q in selected)),
                            'strict_wrong_spectrum_donors':sum(q['wrong_spectrum_row'] is not None for q in selected)},
       'queries_sha256':sha256_file(out/'queries.jsonl'),'inference_inputs_sha256':sha256_file(out/'inference_inputs.jsonl'),
       'input_source_sha256':{str(p.relative_to(ROOT)):sha256_file(p) for p in paths}}
    write_json(out/'contract.json',c);print(json.dumps(c['population_counts']),flush=True)


def command(runtime,smiles,dest):
    root=(runtime/'runtime').resolve()
    loader=root/'lib/ld-musl-x86_64.so.1'
    binary=root/'opt/cfm/bin/cfm-predict'
    params=root/'trained_models_cfmid4.0/[M+H]+/param_output.log'
    config=root/'trained_models_cfmid4.0/[M+H]+/param_config.txt'
    libs=':'.join(str(root/p) for p in ('lib','usr/lib','usr/local/lib','opt/cfm/lib'))
    return [str(loader),'--library-path',libs,str(binary),smiles,'0.001',str(params),str(config),'0',str(dest.resolve()),'0','0']


def limit_child():
    resource.setrlimit(resource.RLIMIT_AS,(3*2**30,3*2**30))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_CPU,(90,95))


def predict(out):
    c=verify(out);runtime=out/'teacher_runtime';manifest=json.loads((runtime/'complete.json').read_text())
    assert manifest['image_digest'].endswith('75363ee83b81ed4fdcc3db303456a528466a3fcec838fd978a81f2384802faa4')
    assert not (out/'prediction_complete.json').exists()
    queries=rows(out/'inference_inputs.jsonl');directory=out/'predictions';directory.mkdir(exist_ok=True)
    unique={}
    for q in queries:
        for row,smi in zip(q['candidate_global_rows'],q['candidate_smiles'],strict=True):
            if row in unique:assert unique[row]==smi
            unique[row]=smi
    def work(item):
        row,smi=item;dest=directory/f'{row}.txt';log=directory/f'{row}.log';started=time.time()
        assert not dest.exists() and not log.exists()
        argv=command(runtime,smi,dest);status='unknown'
        try:
            with log.open('w') as f:
                result=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,timeout=90,check=False,
                                      env={**os.environ,'OMP_NUM_THREADS':'1','LC_ALL':'C'},preexec_fn=limit_child)
            status='ok' if result.returncode==0 and dest.exists() and dest.stat().st_size else 'error'
            code=result.returncode
        except subprocess.TimeoutExpired:status='timeout';code=None
        return {'candidate_global_row':row,'smiles':smi,'status':status,'returncode':code,
                'seconds':time.time()-started,'output_sha256':sha256_file(dest) if dest.exists() else None,
                'log_sha256':sha256_file(log),'argv':argv}
    # First two actual candidates serve as a bounded resource/runtime soak.
    items=sorted(unique.items());done=[]
    for item in items[:2]:
        result=work(item);done.append(result);print(json.dumps({k:result[k] for k in ('candidate_global_row','status','seconds')}),flush=True)
        assert result['status']=='ok','Initial CFM runtime soak failed; inspect logs before scaling'
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,result in enumerate(pool.map(work,items[2:]),start=3):
            done.append(result)
            with (out/'prediction_progress.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            if i%10==0:print('predictions',i,'/',len(items),Counter(x['status'] for x in done),flush=True)
    write_json(out/'prediction_complete.json',{'predictions':done,'runtime_manifest_sha256':sha256_file(runtime/'complete.json'),
        'contract_sha256':sha256_file(out/'contract.json'),'counts':dict(Counter(x['status'] for x in done))})


def parse_prediction(path,energy_index,precursor):
    current=None;peaks=defaultdict(float)
    for line in path.open():
        line=line.strip()
        if line.startswith('energy'):
            current=int(line[6:]);continue
        if current!=energy_index or not line or line.startswith('#'):continue
        fields=line.split()
        if len(fields)!=2:continue
        try:mz,intensity=map(float,fields)
        except ValueError:continue
        if np.isfinite(mz) and np.isfinite(intensity) and mz>0 and intensity>0 and mz<precursor-1.5:
            peaks[mz]+=intensity
    mz=np.array(sorted(peaks),dtype=np.float64)
    intensity=np.array([peaks[m] for m in mz],dtype=np.float64)
    return mz,intensity


def match_edges(query_mz,pred_mz):
    potential=[]
    for i,q in enumerate(query_mz):
        for j,p in enumerate(pred_mz):
            ppm=abs(q-p)/p*1e6
            if ppm<=20.:potential.append((ppm,float(q),float(p),i,j))
    used_q=set();used_p=set();qi=[];pj=[]
    for _,_,_,i,j in sorted(potential):
        if i not in used_q and j not in used_p:
            used_q.add(i);used_p.add(j);qi.append(i);pj.append(j)
    return np.array(qi,dtype=int),np.array(pj,dtype=int)


def metrics(scores,gt,baseline_rank):
    if gt<0:return np.array([0.,1./baseline_rank,np.nan])
    value=scores[gt];better=np.sum(scores>value);equal=np.sum(scores==value)
    assert equal>=1
    r1=1./equal if better==0 else 0.
    mrr=float(np.mean(1./np.arange(better+1,better+equal+1)))
    decoy=np.delete(scores,gt);win=float(np.mean((value>decoy)+.5*(value==decoy)))
    return np.array([r1,mrr,win])


def summarize(values):
    return {key:float(np.nanmean(values[:,i])) for i,key in enumerate(('r1','mrr','reachable_pairwise_win'))}


def paired(a,b,indices):
    delta=a-b
    result={}
    for i,key in enumerate(('r1','mrr','reachable_pairwise_win')):
        point=np.nanmean(delta[:,i]);draw=np.nanmean(delta[indices,i],axis=1)
        result[key]={'delta_pp':float(point*100),'ci95_pp':(np.nanquantile(draw,[.025,.975])*100).tolist()}
    return result


def analyze(out):
    start=time.time();c=verify(out);prediction=json.loads((out/'prediction_complete.json').read_text())
    assert prediction['contract_sha256']==sha256_file(out/'contract.json')
    for p in prediction['predictions']:
        if p['output_sha256']:assert sha256_file(out/'predictions'/f'{p["candidate_global_row"]}.txt')==p['output_sha256']
    status={p['candidate_global_row']:p['status'] for p in prediction['predictions']}
    queries=rows(out/'queries.jsonl');mz=np.load(GEOMETRY/'train.local_mz.npy',mmap_mode='r');inten=np.load(GEOMETRY/'train.local_intensity.npy',mmap_mode='r')
    records=rows(GEOMETRY/'train.rows.jsonl');n=len(queries);reps=200
    uniform=np.zeros((n,5));teacher=uniform.copy();shuffled=np.zeros((reps,n,5));baseline=np.zeros((n,3))
    uni_metrics=np.zeros((n,3));tea_metrics=np.zeros((n,3));shu_metrics=np.zeros((reps,n,3));supported=np.ones(n,bool)
    evidence=[];wrong_u=[];wrong_t=[];wrong_indices=[];fixed_edges=[]
    for k,q in enumerate(queries):
        row=q['raw_train_row'];qn=records[row]['n_retained_peaks'];qm=np.asarray(mz[row,:qn],float);qv=np.asarray(inten[row,:qn],float);qv/=np.linalg.norm(qv)
        base_scores=np.array(q['base_scores']);gt=q['gt_position'];rank=q['baseline_rank'];baseline[k]=metrics(base_scores,gt,rank)
        pred=[]
        for cid in q['candidate_global_rows']:
            pm,pv=parse_prediction(out/'predictions'/f'{cid}.txt',q['energy_index'],q['precursor_mz']) if status[cid]=='ok' else (np.array([]),np.array([]))
            pred.append((pm,pv))
        reason='ok' if all(len(pm) for pm,pv in pred) else 'at_least_one_missing_prediction_or_nonprecursor_support'
        supported[k]=reason=='ok';candstats=[];wu=[];wt=[]
        for j,((pm,pv),cid) in enumerate(zip(pred,q['candidate_global_rows'],strict=True)):
            qi,pj=match_edges(qm,pm);t=np.sqrt(pv);u=np.ones(len(pv))
            if len(pv):
                t/=np.linalg.norm(t);u/=np.linalg.norm(u)
                uniform[k,j]=qv[qi]@u[pj];teacher[k,j]=qv[qi]@t[pj]
                rng=np.random.default_rng(int(keyed(q['identifier']+'|'+str(cid)+'|weights')[:16],16))
                for r in range(reps):shuffled[r,k,j]=qv[qi]@rng.permutation(t)[pj]
            candstats.append({'candidate_global_row':cid,'retained_predicted_peaks':len(pm),'matched_peaks':len(qi),
                              'matched_observed_intensity':float((qv[qi]**2).sum()),'matched_predicted_intensity':float((t[pj]**2).sum()) if len(pv) else 0.})
            fixed_edges.append({'identifier':q['identifier'],'candidate_global_row':cid,'query_peak_indices':qi.tolist(),'predicted_peak_indices':pj.tolist(),
                                'retained_predicted_mz':pm.tolist(),'retained_predicted_intensity':pv.tolist(),'query_mz':qm.tolist(),'query_sqrt_L2_intensity':qv.tolist()})
            if q['wrong_spectrum_row'] is not None:
                w=q['wrong_spectrum_row'];wn=records[w]['n_retained_peaks'];wm=np.asarray(mz[w,:wn],float);wv=np.asarray(inten[w,:wn],float);wv/=np.linalg.norm(wv)
                wi,wj=match_edges(wm,pm);wu.append(float(wv[wi]@u[wj]) if len(pv) else 0.);wt.append(float(wv[wi]@t[wj]) if len(pv) else 0.)
        if not supported[k]:
            uniform[k]=base_scores;teacher[k]=base_scores;shuffled[:,k,:]=base_scores
            if wu:wu=base_scores;wt=base_scores
        uni_metrics[k]=metrics(uniform[k],gt,rank);tea_metrics[k]=metrics(teacher[k],gt,rank)
        for r in range(reps):shu_metrics[r,k]=metrics(shuffled[r,k],gt,rank)
        if q['wrong_spectrum_row'] is not None:
            wrong_indices.append(k);wrong_u.append(metrics(np.array(wu),gt,rank));wrong_t.append(metrics(np.array(wt),gt,rank))
        evidence.append({'identifier':q['identifier'],'connectivity':q['connectivity'],'CE':q['collision_energy'],'gt_position':gt,
                         'baseline_rank':rank,'status':reason,'candidate_stats':candstats,'uniform_scores':uniform[k].tolist(),'teacher_scores':teacher[k].tolist()})
    assert len({q['connectivity'] for q in queries})==n
    idx=np.random.default_rng(SEED).integers(0,n,size=(5000,n));mean_shu=shu_metrics.mean(axis=0)
    summary={'status':'COMPLETE_SMALL_EXPLORATORY_PILOT','contract_sha256':sha256_file(out/'contract.json'),
       'queries':n,'supported_queries':int(supported.sum()),'GT_reachable_queries':sum(q['gt_position']>=0 for q in queries),
       'candidate_predictions':prediction['counts'],'metrics':{'upstream_OOF_SCCF':summarize(baseline),'uniform':summarize(uni_metrics),'teacher':summarize(tea_metrics),'shuffle_mean':summarize(mean_shu)},
       'comparisons':{'teacher_minus_uniform':paired(tea_metrics,uni_metrics,idx),'teacher_minus_shuffle_mean':paired(tea_metrics,mean_shu,idx),
                      'teacher_minus_upstream_OOF_SCCF':paired(tea_metrics,baseline,idx)},
       'permutation_r1':{'replicates':reps,'teacher_r1':float(tea_metrics[:,0].mean()),'shuffle_r1_quantiles':np.quantile(shu_metrics[:,:,0].mean(axis=1),[.025,.5,.975]).tolist(),
                         'upper_tail_p':float((1+np.sum(shu_metrics[:,:,0].mean(axis=1)>=tea_metrics[:,0].mean()))/(reps+1))},
       'supported_only_metrics':{'uniform':summarize(uni_metrics[supported]),'teacher':summarize(tea_metrics[supported]),'shuffle_mean':summarize(mean_shu[supported])},
       'counts_by_CE':{},'no_fusion_no_student':True,'official_validation_read':False,'official_test_read':False,'gpu_used':False,'analysis_seconds':time.time()-start}
    for ce in (10.,20.,40.):
        take=np.array([q['collision_energy']==ce for q in queries]);summary['counts_by_CE'][str(ce)]={'n':int(take.sum()),'uniform':summarize(uni_metrics[take]),'teacher':summarize(tea_metrics[take]),'shuffle_mean':summarize(mean_shu[take])}
    if wrong_indices:
        wi=np.array(wrong_indices);wu=np.array(wrong_u);wt=np.array(wrong_t);ii=np.random.default_rng(SEED+1).integers(0,len(wi),size=(5000,len(wi)))
        summary['strict_wrong_spectrum']={'queries':len(wi),'real_uniform':summarize(uni_metrics[wi]),'wrong_uniform':summarize(wu),'real_teacher':summarize(tea_metrics[wi]),'wrong_teacher':summarize(wt),
            'teacher_real_minus_wrong':paired(tea_metrics[wi],wt,ii),'incremental_weight_real_minus_wrong':paired(tea_metrics[wi]-uni_metrics[wi],wt-wu,ii)}
    np.savez_compressed(out/'scores.npz',uniform=uniform,teacher=teacher,shuffle=shuffled,supported=supported,baseline_metrics=baseline,
                        uniform_metrics=uni_metrics,teacher_metrics=tea_metrics,shuffle_metrics=shu_metrics,bootstrap_indices=idx,wrong_indices=np.array(wrong_indices),wrong_uniform=np.array(wrong_u),wrong_teacher=np.array(wrong_t))
    for filename,data in [('per_query.jsonl',evidence),('fixed_mass_edges.jsonl',fixed_edges)]:
        with (out/filename).open('w') as f:
            for x in data:f.write(json.dumps(x,sort_keys=True)+'\n')
    write_json(out/'summary.json',summary)
    files=['contract.json','queries.jsonl','inference_inputs.jsonl','prediction_complete.json','scores.npz','per_query.jsonl','fixed_mass_edges.jsonl','summary.json']
    write_json(out/'complete.json',{'status':'COMPLETE','files_sha256':{p:sha256_file(out/p) for p in files},'official_validation_read':False,'official_test_read':False})
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','predict','analyze'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    {'prepare':prepare,'predict':predict,'analyze':analyze}[a.phase](a.out.resolve())

if __name__=='__main__':main()
