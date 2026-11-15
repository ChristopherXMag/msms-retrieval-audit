#!/usr/bin/env python3
"""One frozen official-test transaction. Default action only checks pretest readiness."""
from __future__ import annotations
import argparse,gc,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.evaluation.official_s7_common import *
from tools.evaluation.official_s7_resources import advise_clean,reclaim_inputs
PYTHON=sys.executable
OUT=ROOT/'outputs/official_s7_test_20261017'
OVERLAY=ROOT/'outputs/e0_external_contract_20261014/runtime_overlay'

def readiness():
    install_no_test_guard();ready=read(PREP/'ready.json');assert ready['status']=='READY_FOR_SEPARATE_OFFICIAL_TEST'
    manifest=read(ROOT/ready['manifest']);assert sha(ROOT/ready['manifest'])==ready['manifest_sha256']
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();assert head==ready['commit']
    from importlib.metadata import version
    for package,expected in read(PREP/'environment.json')['packages'].items():assert version(package)==expected,('Runtime package changed',package)
    for path,entry in manifest['files'].items():
        assert sha(ROOT/path)==entry['sha256'],('Frozen input/source changed',path)
        advise_clean(ROOT/path)
    for path in manifest['source_files']:
        assert subprocess.check_output(['git','diff','HEAD','--',path],cwd=ROOT)==b'',('Uncommitted frozen source',path)
    for path,expected in manifest['submodule_heads'].items():
        assert subprocess.check_output(['git','-C',str(ROOT/path),'rev-parse','HEAD'],text=True).strip()==expected,('Submodule checkout changed',path)
    return ready,manifest

def child(command,out,label,overlay=False):
    from tools.reranking.run_official_lightgbm_transfer import memory_guard
    ready=read(PREP/'ready.json');manifest=read(ROOT/ready['manifest'])
    reclaimed=reclaim_inputs(manifest,include_test=out);write(out/(label+'.resources.json'),dict(clean_files_advised=reclaimed,bytes_changed=False))
    gpu_preflight(out,label);memory_guard()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if overlay:env['PYTHONPATH']=str(OVERLAY)+os.pathsep+str(ROOT)
    with (out/(label+'.log')).open('w') as f:subprocess.run([PYTHON,*command],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def candidate_prior(rows,out):
    import joblib
    from tools.audits import audit_candidate_shortcuts as prior
    base=ROOT/'outputs/stage1_submission_0926/audits/official_candidate_only_20260929'
    needed=np.unique(rows['candidates'][rows['candidates']>=0]);cached=load(base/'candidate_descriptors.npz')
    lookup,values=prior.make_descriptor_lookup(cached['rows'],cached['values'],len(np.load(PREP/'candidate_prior/train_candidate_occurrence.npy')))
    missing=needed[lookup[needed]<0]
    if len(missing):
        from tools.reranking.run_official_lightgbm_top20 import read_smiles
        strings=read_smiles(missing);fresh=np.stack([prior.descriptor(strings[int(i)]) for i in missing]);assert np.isfinite(fresh).all()
        lookup[missing]=len(values)+np.arange(len(missing));values=np.concatenate([values,fresh])
    model=joblib.load(PREP/'candidate_prior/candidate_prior.joblib');occurrence=np.load(PREP/'candidate_prior/train_candidate_occurrence.npy')
    score=prior.model_score_fn(model,lookup,values,occurrence);metrics=[];random=[];g=[];e=[]
    for candidates,n in zip(rows['candidates'],rows['lengths']):
        v,greater,equal=prior.expected_tie_contributions(score(candidates[:n]));metrics.append(v);g.append(greater);e.append(equal)
        ranks=np.arange(1,n+1,dtype=np.float64);random.append([min(k,n)/n for k in (1,5,20)]+[np.mean(1/ranks)])
    result=dict(expected_metrics=np.stack(metrics),random_expected_metrics=np.stack(random),greater=np.array(g,np.int32),equal=np.array(e,np.int32))
    np.savez_compressed(out/'candidate_prior.npz',**result);return result

def sensitivity(ctx,name,result,donors,out):
    """Only actual shortlist/order changes require new scores; retain real GEMM windows."""
    reports={};primary=result['values_selected'];query=result['queries'];n=len(query)
    for j,seed in enumerate(TIE_SEEDS[1:]):
        affected=np.flatnonzero(result['alternate_order'][:,j]);values=primary.copy()
        for window in np.unique(query[affected]//512):
            pos=affected[query[affected]//512==window]
            alternate,_=ctx.evaluate(name,query[pos],None if donors is None else donors[pos],tie_seed=seed)
            values[pos]=alternate['values_selected']
        np.savez_compressed(out/f'{name}.tie_seed{seed}.npz',queries=query,expected_metrics=values,affected_queries=affected)
        reports[str(seed)]=dict(affected_queries=len(affected),conservative_metric_bound_pp=100*len(affected)/max(n,1),
            delta_primary_pp=((values-primary).mean(0)*100).tolist() if n else None)
    return dict(primary_seed=TIE_SEEDS[0],boundary_ties5=int(result['boundary_ties'][:,0].sum()),boundary_ties20=int(result['boundary_ties'][:,1].sum()),
        primary_selected_vs_stage1_R20_pp=float((primary[:,2]-result['values_stage1'][:,2]).mean()*100) if n else None,seeds=reports)

def analysis(out,rows,controls):
    from tools.reranking.audit_official_d3_full_system import cluster_delta
    from tools.analysis.freeze_official_system_s6_e5 import summarize
    data=load(out/'per_query.npz');groups=data['connectivities'];plan=read(S6/'result_rows_plan.json')
    summaries={key:summarize(data[key],groups) for key in (r['row'] for r in plan['rows'])}
    contrasts={}
    for item in plan['primary_contrasts']:
        after,before=item.split(' minus ');contrasts[item]=cluster_delta(data[before],data[after],groups)
    for name,result in controls.items():
        q=result['queries'];contrasts['real minus '+name]=cluster_delta(result['values_selected'],data['top20_support_fallback'][q],groups[q])
    prior=load(out/'candidate_prior.npz');nulls={}
    for name,key in [('candidate_prior','expected_metrics'),('exact_random','random_expected_metrics')]:
        nulls[name]=summarize(prior[key],groups);contrasts['real minus '+name]=cluster_delta(prior[key],data['top20_support_fallback'],groups)
    seed_reports={}
    sccf=load(out/'sccf_seeds.npz')
    for method in ('sccf','e1','e2'):
        vectors=[sccf[f'seed{s}'] if method=='sccf' else load(out/method/f'seed{s}.npz')['expected_metrics'] for s in SEEDS]
        summaries_seed=[summarize(v,groups) for v in vectors]
        from tools.analysis.analyze_stage1_logit_ensemble import grouped
        seed_reports[method]=dict(per_seed=dict(zip(map(str,SEEDS),summaries_seed)),sample_sd_query_pp=np.std(np.array([list(v['query_weighted'].values()) for v in summaries_seed]),axis=0,ddof=1).tolist(),sample_sd_connectivity_balanced_pp=np.std(np.array([list(v['connectivity_balanced'].values()) for v in summaries_seed]),axis=0,ddof=1).tolist())
    transitions={}
    a=data['top5_detail'][:,0];b=data['top20_support_fallback'][:,0];den=a.sum()
    # Exact expected marginal hit rates do not determine joint tie outcomes; disclose sharp bounds.
    transitions['vs_D2']=dict(damage_expected_count_bounds=[float(np.maximum(a-b,0).sum()),float(np.minimum(a,1-b).sum())],
        rescue_expected_count_bounds=[float(np.maximum(b-a,0).sum()),float(np.minimum(b,1-a).sum())],
        damage_rate_pct_bounds=[float(100*np.maximum(a-b,0).sum()/den),float(100*np.minimum(a,1-b).sum()/den)] if den else None)
    real=load(out/'real.npz');strata={}
    for name,mask in [('rank1',real['ranks_stage1']==1),('rank2_5',(real['ranks_stage1']>=2)&(real['ranks_stage1']<=5)),('rank6_20',(real['ranks_stage1']>=6)&(real['ranks_stage1']<=20)),('rank_gt20',real['ranks_stage1']>20)]:
        strata[name]=dict(queries=int(mask.sum()),stage1=summarize(data['sccf_ensemble'][mask],groups[mask]) if mask.any() else None,selected=summarize(data['top20_support_fallback'][mask],groups[mask]) if mask.any() else None)
    closures={}
    for name in ('sccf_ensemble','top5_detail','top20_detail'):
        denom=(data[name][:,2]-data[name][:,0]).mean();closures[name]=dict(baseline_rerankable_gap_pp=float(100*denom),closure_pct=float(100*(data['top20_support_fallback'][:,0]-data[name][:,0]).mean()/denom) if denom else None)
    gap=data['sccf_ensemble'][:,2].mean()-data['sccf_ensemble'][:,0].mean()
    write(out/'results.json',dict(status='COMPLETE',rows=summaries,null_controls=nulls,per_seed=seed_reports,primary_contrasts=contrasts,transitions=transitions,stage1_rank_strata=strata,gap_closure_by_baseline=closures,
        selected_vs_stage1_gap_closure_pct=100*(data['top20_support_fallback'][:,0]-data['sccf_ensemble'][:,0]).mean()/gap if gap else None,
        uncertainty=plan['bootstrap'],claims=plan['claims'],tie_policy='fixed primary shortlist; uniform expectation separately in executed prefix and retained tail',
        seed_metrics_report='per-seed NPZ and arithmetic mean; sample SD of per-seed aggregate metrics separately'))

def worker_gate(stage):
    assert read(OUT/'transaction.json')['status']=='STARTED_ONCE'
    assert not (OUT/'predictions_seal.json').exists() and not (OUT/'completion.json').exists(),'Transaction already sealed'
    descriptor=os.open(OUT/(stage+'.started'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(descriptor,'w') as f:f.write(str(time.time())+'\n');f.flush();os.fsync(f.fileno())

def execute(ready,manifest):
    # No test stat/open before the exclusive transaction marker is durable.
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'transaction.json').exists():raise FileExistsError('An official-test transaction already exists; automatic restart is disabled')
    assert shutil.disk_usage(OUT).free>12*2**30
    gpu_preflight(OUT,'before_test_access')
    fd=os.open(OUT/'transaction.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,'w') as f:
        import json
        json.dump(dict(status='STARTED_ONCE',ready=ready,time=time.time(),automatic_restart=False),f,indent=2);f.flush();os.fsync(f.fileno())
    from tools.evaluation.official_s7_inputs import raw_inputs,metadata_arrays,bins,native_vectors,extend_e1
    from tools.reranking import audit_official_d3_full_system as old
    from glmr.candidate_indices import CandidateIndexStore
    meta=ROOT/'outputs/msalign_stage1_full_v15/test_full.metadata.jsonl';records=old.top.records(meta)
    assert all(q['fold']=='test' for q in records)
    rows=rows_from_records(records,CandidateIndexStore(ROOT/read(CONFIG)['assets']['candidate_manifest']))
    referenced=np.unique(rows['candidates'][rows['candidates']>=0]);union=len(referenced)
    extension_max=int((np.load(E1/'moldeberta_cache/global_to_cache.npy',mmap_mode='r')[referenced]<0).sum())
    native_max=sum(min(union,len(records)*k*3)*2048 for k in (5,20))
    estimate=union*1024*4*3+extension_max*768*4+native_max+len(records)*(10050*4+1024*4*2+1024*2)+2*2**30
    assert shutil.disk_usage(OUT).free>estimate+12*2**30,('Insufficient disk before neural results',estimate)
    write(OUT/'resource_budget.json',dict(candidate_union=union,e1_missing_rows=extension_max,native_vector_upper_bytes=native_max,conservative_working_bytes=estimate,reserve_bytes=12*2**30))
    np.savez_compressed(OUT/'rows.npz',**rows);np.savez(OUT/'metadata.npz',**metadata_arrays(records))
    inputs=raw_inputs(records,ROOT/'outputs/msalign_stage1_full_v15/test_full.hdf5')
    with (OUT/'inputs.jsonl').open('w') as f:
        import json
        for q in inputs:f.write(json.dumps(q)+'\n')
    bins(records,ROOT/'outputs/msalign_stage1_full_v15/test_full.mgf',OUT/'bins.npy')
    donors,audit=make_donors(records,rows['connectivities']);np.savez_compressed(OUT/'donors.npz',**donors);write(OUT/'donor_audit.json',audit)
    checkpoint=read(CONFIG)['assets'];encoder=ROOT/checkpoint['datasafe_checkpoint'];assert sha(encoder)==checkpoint['datasafe_checkpoint_sha256']
    for representation in ('no_head','contrastive'):
        child(['tools/data/cache_msalign_dreams_representations.py','--metadata',str(meta),'--out-npy',str(OUT/f'{representation}.npy'),
            '--out-json',str(OUT/f'{representation}.summary.json'),'--v15-tsv',str(ROOT/'data/MassSpecGym1.5.tsv'),
            '--dreams-ckpt',str(encoder),'--expected-dreams-ckpt-sha256',checkpoint['datasafe_checkpoint_sha256'],
            '--representation',representation,'--batch-size','256','--n-highest-peaks','100','--intensity-scale','100','--device','cuda'],OUT,'dreams_'+representation)
    parent=read(old.SOURCE)
    child(['tools/stage1/export_stage1_candidate_scores.py','--checkpoint',*[str(ROOT/p['path']) for p in parent['checkpoints']],
        '--metadata',str(meta),'--dreams',str(OUT/'no_head.npy'),'--candidate-index-manifest',str(ROOT/parent['candidate_index_manifest']),
        '--fingerprint-cache',str(ROOT/parent['fingerprint_cache']),'--eval-batch-size','64','--tie-seeds',','.join(map(str,TIE_SEEDS)),
        '--out-npz',str(OUT/'stage1.npz'),'--out-json',str(OUT/'stage1.summary.json'),'--device','cuda'],OUT,'stage1')
    finish_from_stage1(ready,manifest,records,rows,donors)

def finish_from_stage1(ready,manifest,records,rows,donors):
    """Unchanged frozen stages, shared with an audited manual memory repair."""
    source=read(OUT/'stage1.summary.json');pool=load(OUT/'stage1.npz');assert pool['identifiers'].tolist()==[q['identifier'] for q in records],'No skipped or inserted test queries'
    child(['tools/evaluation/run_official_s7.py','--native-stage',str(OUT/'transaction.json')],OUT,'native_real_vectors')
    source['native_real_vectors']=str((OUT/'native').relative_to(ROOT));write(OUT/'stage1.native.summary.json',source)
    # The extension uses the pinned Transformers overlay in a clean child process.
    child(['tools/evaluation/run_official_s7.py','--extension-stage',str(OUT/'transaction.json')],OUT,'e1_extension',overlay=True)
    external={}
    for method in ('e1','e2'):
        child(['tools/evaluation/run_official_s7.py','--external-stage',method],OUT,'external_'+method)
        external[method]=load(OUT/method/'seed_mean_per_query.npz')['expected_metrics']
    import torch
    from tools.evaluation.official_s7_system import System
    reclaim_inputs(manifest,include_test=OUT)
    gpu_preflight(OUT,'system');ctx=System(OUT/'stage1.native.summary.json',OUT/'inputs.jsonl',OUT/'projection',CONTROLS/'train_means.npz')
    real,report=ctx.evaluate('real',np.arange(ctx.n));np.savez_compressed(OUT/'real.npz',**real)
    reports={'real':report};ties={'real':sensitivity(ctx,'real',real,None,OUT)};controls={}
    for name in ('zero','train_mean',*(f'wrong_seed{s}' for s in DONOR_SEEDS)):
        q=donors['recipient_rows'] if name.startswith('wrong') else np.arange(ctx.n)
        d=donors[name.replace('wrong_','')+'_donor_rows'] if name.startswith('wrong') else None
        if not len(q):reports[name]=dict(status='NO_ELIGIBLE_DONORS');continue
        result,report=ctx.evaluate(name,q,d);np.savez_compressed(OUT/(name+'.npz'),**result)
        controls[name]=result;reports[name]=report;ties[name]=sensitivity(ctx,name,result,d,OUT)
    comparators=ctx.real_comparators(real);values=[]
    for j in range(3):values.append(sccf_seed_contributions(pool,j))
    np.savez_compressed(OUT/'sccf_seeds.npz',**{f'seed{s}':v for s,v in zip(SEEDS,values)})
    allrows=dict(identifiers=rows['identifiers'],connectivities=rows['connectivities'],**comparators,sccf_seed_mean=np.mean(values,axis=0),
        sccf_ensemble=real['values_stage1'],top20_support_fallback=real['values_selected'],msalign_seed_mean=external['e1'],embcos_seed_mean=external['e2'])
    np.savez_compressed(OUT/'per_query.npz',**allrows)
    del ctx;gc.collect();torch.cuda.empty_cache()
    prior=candidate_prior(rows,OUT);write(OUT/'tie_sensitivity.json',ties);write(OUT/'system_reports.json',reports)
    # Seal predictions and their input/model identities before inferential reporting.
    write(OUT/'predictions_seal.json',dict(status='SEALED_BEFORE_ANALYSIS',ready=ready,
        files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}))
    analysis(OUT,rows,controls)
    write(OUT/'completion.json',dict(status='COMPLETE',one_transaction=True,ready=ready,results_sha256=sha(OUT/'results.json'),predictions_sha256=sha(OUT/'predictions_seal.json')))
    print('OFFICIAL_TEST_TRANSACTION_COMPLETE',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute-official-test',action='store_true')
    parser.add_argument('--extension-stage',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--native-stage',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--external-stage',choices=['e1','e2'],help=argparse.SUPPRESS);args=parser.parse_args()
    if args.external_stage:
        worker_gate('external_'+args.external_stage)
        from tools.evaluation.official_s7_external import infer
        import torch
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.use_deterministic_algorithms(True)
        gpu_preflight(OUT,'external_child_'+args.external_stage)
        if args.external_stage=='e1':infer('e1',OUT/'rows.npz',OUT/'contrastive.npy',OUT/'e1',OUT/'metadata.npz',OUT/'e1_extension')
        else:infer('e2',OUT/'rows.npz',OUT/'bins.npy',OUT/'e2')
        return
    if args.native_stage:
        assert args.native_stage.resolve()==(OUT/'transaction.json').resolve() and read(args.native_stage)['status']=='STARTED_ONCE'
        worker_gate('native')
        from tools.evaluation.official_s7_inputs import native_vectors
        gpu_preflight(OUT,'native_child');native_vectors(OUT/'stage1.summary.json',OUT/'native',allow_gpu=True);return
    if args.extension_stage:
        # Internal child is allowed only under the already-created exclusive transaction.
        assert args.extension_stage.resolve()==(OUT/'transaction.json').resolve()
        transaction=read(args.extension_stage);assert transaction['status']=='STARTED_ONCE'
        worker_gate('e1_extension')
        from tools.evaluation.official_s7_inputs import extend_e1
        gpu_preflight(OUT,'e1_extension_child');extend_e1(load(OUT/'rows.npz'),OUT/'e1_extension');return
    if args.execute_official_test:
        # readiness installs an irreversible no-test hook, so execute in a fresh process after checking it.
        subprocess.run([PYTHON,str(Path(__file__))],cwd=ROOT,check=True)
        ready=read(PREP/'ready.json');manifest=read(ROOT/ready['manifest']);execute(ready,manifest)
    else:
        ready,manifest=readiness();print('PRETEST_READY',ready['commit'],'OFFICIAL_TEST_CONTENT_NOT_READ_BY_THIS_CHECK',flush=True)
if __name__=='__main__':main()
