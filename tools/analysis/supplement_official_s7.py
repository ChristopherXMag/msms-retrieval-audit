#!/usr/bin/env python3
"""Post-hoc S7 metrics from sealed scores only; never performs retrieval inference."""
from __future__ import annotations
import argparse, collections, concurrent.futures, json, os, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.evaluation.official_s7_common import load,read,write,sha,SEEDS
from tools.analysis.report_official_s7 import LABELS,mdtable
from tools.analysis.freeze_official_system_s6_e5 import summarize
OUT=ROOT/'outputs/official_s7_posthoc_20261018'
SEALED=ROOT/'outputs/official_s7_test_20261017'

def guard():
    base=Path('/sys/fs/cgroup');stat=dict(x.split() for x in (base/'memory.stat').read_text().splitlines())
    current=int((base/'memory.current').read_text());limit=int((base/'memory.max').read_text())
    assert limit-current+int(stat['inactive_file'])>=32*2**30,'cgroup reserve'
    assert int((base/'memory.swap.current').read_text())==0,'cgroup swap'
    host=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    assert int(host['MemAvailable'].split()[0])*1024>=32*2**30,'host reserve'
    return dict(limit=limit,current=current,available=limit-current+int(stat['inactive_file']),anon=int(stat['anon']))

def verify():
    receipt=read(ROOT/'docs/reranking/freeze_manifests/s7_test_receipt_20261018.json')
    completion=read(SEALED/'completion.json');seal=read(SEALED/'predictions_seal.json')
    assert sha(SEALED/'results.json')==completion['results_sha256']
    assert sha(SEALED/'predictions_seal.json')==completion['predictions_sha256']
    for name,digest in seal['files'].items():
        assert sha(SEALED/name)==digest,('sealed file modified',name)
    return dict(sealed_files=len(seal['files']),receipt_sha256=sha(ROOT/'docs/reranking/freeze_manifests/s7_test_receipt_20261018.json'),completion_sha256=sha(SEALED/'completion.json'),seal_sha256=sha(SEALED/'predictions_seal.json'))

def extract():
    OUT.mkdir(parents=True,exist_ok=True);v=verify();rows=load(SEALED/'rows.npz');real=load(SEALED/'real.npz');stage=load(SEALED/'stage1.npz');values=load(SEALED/'per_query.npz');n=len(rows['identifiers'])
    # Store every maximizer; averaging their structural distances gives uniform exact-tie expectation.
    predictions={};ties={}
    def add(name,pools,scores):
        result=[]
        for pool,score in zip(pools,scores):
            pool=np.asarray(pool);score=np.asarray(score);assert np.isfinite(score).all()
            ix=np.flatnonzero(score==score.max());assert len(ix)>0
            result.append(pool[ix].astype(int).tolist())
        predictions[name]=result;ties[name]=sum(len(x)>1 for x in result)
    pools=[stage['candidate_rows'][a:b] for a,b in zip(stage['indptr'][:-1],stage['indptr'][1:])]
    for i,s in enumerate(SEEDS):add(f'sccf_seed{s}',pools,[stage['scores'][i,a:b] for a,b in zip(stage['indptr'][:-1],stage['indptr'][1:])])
    add('sccf_ensemble',pools,[stage['ensemble_scores'][a:b] for a,b in zip(stage['indptr'][:-1],stage['indptr'][1:])])
    from tools.reranking.audit_official_d3_full_system import top
    for method in ('e1','e2'):
        for seed in SEEDS:
            z=load(SEALED/f'{method}/seed{seed}.npz')
            assert np.array_equal(z['identifiers'],rows['identifiers'])
            add(f'{method}_seed{seed}',[r[:c] for r,c in zip(rows['permuted_candidates'],rows['lengths'])],[s[:c] for s,c in zip(z['scores'],rows['lengths'])])
    add('top20_pair',real['candidate_rows'],real['pair20'])
    for name,multiplier in [('top20_coverage',1.),('top20_doubled_coverage',2.)]:
        add(name,real['candidate_rows'],top.d2.deployment_anchor(real['pair20'],real['coverage20'],multiplier))
    add('top20_detail',real['candidate_rows'],real['detail20'])
    add('top20_support_fallback',[pool[:5] if route else pool for pool,route in zip(real['candidate_rows'],real['route5'])],[score[:5] if route else score for score,route in zip(real['selected_scores'],real['route5'])])
    gt=rows['candidates'][:,0]
    for name,pred in predictions.items():
        contribution=np.array([float(g in pool)/len(pool) for g,pool in zip(gt,pred)])
        if name in values:expected=values[name][:,0]
        elif name.startswith('sccf_seed'):expected=load(SEALED/'sccf_seeds.npz')[name.removeprefix('sccf_')][:,0]
        else:
            method,seed=name.split('_seed');expected=load(SEALED/f'{method}/seed{seed}.npz')['expected_metrics'][:,0]
        assert np.allclose(contribution,expected,rtol=0,atol=1e-14),('top1 identity/score mismatch',name)
    missing=['top5_pair','top5_coverage','top5_doubled_coverage','top5_scalar','top5_detail','top20_scalar']
    write(OUT/'missing_top1_evidence.json',dict(status='NOT_COMPUTABLE_FROM_SEALED_OUTPUTS',rows=missing,reason='real_comparators saved only GT ranks/metric contributions. pair5_scores exist only for route5; top20 scalar scores/features not serialized. Native arrays are embeddings, not predictions.',no_test_inference_performed=True))
    all_global=set(gt.tolist())
    for pred in predictions.values():
        for pool in pred:all_global.update(pool)
    smiles={}
    source=ROOT/'outputs/run_k_full_train/full_sparse/full_smiles.txt'
    with source.open() as f:
        for i,line in enumerate(f):
            if i in all_global:smiles[i]=line.strip()
    assert set(smiles)==all_global
    assert all(smiles[int(row)]==str(smi) for row,smi in zip(gt,rows['smiles']))
    pairs={};uses={}
    for name,pred in predictions.items():
        uses[name]=[]
        for g,pool in zip(gt,pred):
            keys=[]
            for row in pool:
                pair=sorted((smiles[int(g)],smiles[row]));key=__import__('hashlib').sha256(json.dumps(pair).encode()).hexdigest();pairs[key]=pair;keys.append(key)
            uses[name].append(keys)
    write(OUT/'mces_pairs.json',pairs);write(OUT/'mces_uses.json',uses)
    write(OUT/'extraction_inspection.json',dict(verified=v,files=list(read(SEALED/'predictions_seal.json')['files']),ties=ties))
    np.savez_compressed(OUT/'base_predictions.npz',**{k:np.asarray([x[0] for x in a],np.int32) for k,a in predictions.items()})
    write(OUT/'base_top1_ties.json',predictions)
    print('EXTRACT_BASE',n,ties,flush=True)


def proximity():
    import torch
    from tools.analysis.analyze_sccf_wrong_spectrum_proximity import normalized_raw_features,nearest_training_similarity,DEFAULT_FP,DEFAULT_CANDIDATES,DEFAULT_TRAIN
    from tools.stage1.train_msalign_projection_smoke import MoleculeEmbeddingStore
    from glmr.candidate_indices import CandidateIndexStore
    from tools.audits.audit_spectrum_permutation import connectivity_identities
    guard();torch.set_num_threads(2)
    records=[json.loads(x) for x in DEFAULT_TRAIN.read_text().splitlines()];keys,audit=connectivity_identities(records)
    candidates=CandidateIndexStore(DEFAULT_CANDIDATES);by_row={}
    for q,key in zip(records,keys):by_row.setdefault(candidates.get(q['identifier'],256)[0],key)
    rows=load(SEALED/'rows.npz');test_rows=rows['candidates'][:,0];test_unique=np.unique(test_rows)
    key_lookup=dict(zip(test_rows,rows['connectivities']));train_rows=sorted(by_row)
    assert not set(rows['connectivities'])&set(by_row.values())
    store=MoleculeEmbeddingStore(None,'mean',DEFAULT_FP,preload_sparse_features=False,mol_blocks=['count_morgan','count_atom_pair','rdkit_path_count','maccs'])
    def bounded_features(ix):return torch.cat([normalized_raw_features(store,ix[i:i+256],'block_l2') for i in range(0,len(ix),256)])
    train=bounded_features(train_rows);test=bounded_features(test_unique.tolist());guard()
    p=nearest_training_similarity(test,train,[str(key_lookup[i]) for i in test_unique],[by_row[i] for i in train_rows],torch.device('cpu'),128)
    by=dict(zip(test_unique,p));vector=np.array([by[x] for x in test_rows]);assert np.isfinite(vector).all()
    np.savez_compressed(OUT/'proximity.npz',identifiers=rows['identifiers'],connectivities=rows['connectivities'],proximity=vector)
    write(OUT/'proximity_manifest.json',dict(definition='nearest official-train positive cosine in frozen raw SCCF block_l2 then global L2; no learned projection; every distinct positive row retained',train_positive_rows=len(train_rows),train_connectivities=len(set(by_row.values())),test_positive_rows=len(test_unique),train_identity_audit=audit,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (DEFAULT_FP,DEFAULT_CANDIDATES,DEFAULT_TRAIN)},resources=guard()))
    print('PROXIMITY_COMPLETE',len(train_rows),len(test_unique),float(vector.min()),float(vector.max()),flush=True)


def strata():
    from scipy.stats import spearmanr
    rows=load(SEALED/'rows.npz');values=load(SEALED/'per_query.npz');real=load(SEALED/'real.npz');prox=load(OUT/'proximity.npz');groups=rows['connectivities'];n=len(groups)
    assert np.array_equal(prox['identifiers'],rows['identifiers'])
    inputs=[json.loads(x) for x in (SEALED/'inputs.jsonl').read_text().splitlines()];adduct=np.array([q['adduct'] for q in inputs]);assert [q['identifier'] for q in inputs]==rows['identifiers'].tolist()
    # Fixed numeric bins, declared before viewing stratified outcomes; tails retained even when empty.
    axes=dict(proximity=[('<0.60',prox['proximity']<.6),('[0.60,0.70)',(prox['proximity']>=.6)&(prox['proximity']<.7)),('[0.70,0.80)',(prox['proximity']>=.7)&(prox['proximity']<.8)),('[0.80,0.90)',(prox['proximity']>=.8)&(prox['proximity']<.9)),('>=0.90',prox['proximity']>=.9)],adduct=[(a,adduct==a) for a in sorted(set(adduct))],candidate_count=[('<128',rows['lengths']<128),('[128,256)',(rows['lengths']>=128)&(rows['lengths']<256)),('256',rows['lengths']==256)],route=[('top20_supported',real['support20']),('complete_D2_fallback',real['route5']),('pair_coverage_fallback',~real['support20']&~real['route5'])])
    report={};tex=[];markdown=[]
    methods=('sccf_seed_mean','msalign_seed_mean','embcos_seed_mean','sccf_ensemble','top5_detail','top20_detail','top20_support_fallback')
    for axis,slices in axes.items():
        assert np.all(np.stack([mask for _,mask in slices]).sum(0)==1)
        report[axis]=[];table=[]
        for name,mask in slices:
            count=int(mask.sum());entry=dict(stratum=name,queries=count,connectivities=len(np.unique(groups[mask])),rows={})
            if count:
                for method in methods:entry['rows'][method]=summarize(values[method][mask],groups[mask])
                before=entry['rows']['sccf_ensemble']['query_weighted'];after=entry['rows']['top20_support_fallback']['query_weighted'];gap=before['recall@20']-before['recall@1'];entry['gain_pp']=after['recall@1']-before['recall@1'];entry['gap_closure_pct']=100*entry['gain_pp']/gap if gap else None
                cells=[axis,name,str(count),str(entry['connectivities']),f"{before['recall@1']:.3f}",f"{before['recall@20']:.3f}",f"{after['recall@1']:.3f}",f"{after['recall@5']:.3f}",f"{after['mrr']:.3f}",f"{entry['rows']['top20_support_fallback']['connectivity_balanced']['recall@1']:.3f}"]
                table.append(cells);tex.append(' & '.join(x.replace('_',r'\_').replace('>=',r'$\geq$').replace('<',r'$<$') for x in cells)+r' \\')
            report[axis].append(entry)
        markdown.append('### '+axis+'\n\n'+mdtable(['Axis','Stratum','Queries','Groups','S1 R1','S1 R20','Selected R1','Selected R5','Selected MRR','Selected balanced R1'],table))
    unique,inverse=np.unique(groups,return_inverse=True);counts=np.bincount(inverse)
    mean=lambda a:np.bincount(inverse,weights=a)/counts
    x=mean(prox['proximity']);correlation={key:float(spearmanr(x,mean(values[key][:,col])).statistic) for key,col in [('sccf_ensemble',0),('top20_support_fallback',0)]}
    correlation['stage1_reachability_R20']=float(spearmanr(x,mean(values['sccf_ensemble'][:,2])).statistic)
    write(OUT/'strata.json',dict(status='COMPLETE',posthoc=True,descriptive_only=True,causal_claim=False,multiple_testing_adjustment=False,connectivity_spearman=correlation,axes=report,seal_sha256=sha(SEALED/'predictions_seal.json')))
    (OUT/'strata.md').write_text('\n\n'.join(markdown)+'\n\nConnectivity-level Spearman: '+json.dumps(correlation)+'\n')
    (OUT/'s7_posthoc_strata_rows.tex').write_text('% Post-hoc descriptive; sealed predictions only.\n'+'\n'.join(tex)+'\n')
    print('STRATA_COMPLETE',correlation,flush=True)


def mces_pair(item):
    key,pair=item
    sys.path.insert(0,str(OUT/'vendor/site'))
    from myopic_mces import MCES
    import pulp
    started=time.time()
    if pair[0]==pair[1]:return dict(key=key,distance=0.,seconds=0.,mode=0,status='IDENTICAL')
    try:
        binary=pulp.PULP_CBC_CMD().path
        os.chmod(binary,os.stat(binary).st_mode|0o100)
        result=MCES(*pair,threshold=15,always_stronger_bound=True,solver='COIN_CMD',solver_options=dict(path=binary,msg=False,threads=1,timeLimit=30))
        distance=float(result[1]);mode=int(result[3]);seconds=time.time()-started
        status='TIME_LIMIT' if mode in (5,6) or (seconds>=30 and distance<=15) else 'OK'
        if distance<0:status='UNRESOLVED'
        return dict(key=key,distance=distance,mode=mode,seconds=seconds,status=status)
    except Exception as e:return dict(key=key,status='ERROR',error=repr(e),seconds=time.time()-started)

def mces():
    guard();pairs=read(OUT/'mces_pairs.json');path=OUT/'mces_cache.jsonl'
    done={json.loads(x)['key'] for x in path.read_text().splitlines()} if path.exists() else set()
    pending=[(k,v) for k,v in sorted(pairs.items()) if k not in done]
    limit=int(os.environ.get('S7_MCES_LIMIT','0'));pending=pending[:limit] if limit else pending
    print('MCES_BEGIN',len(pairs),len(done),len(pending),flush=True)
    workers=int(os.environ.get('S7_MCES_WORKERS','2'));assert 1<=workers<=16
    with path.open('a',buffering=1) as f,concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for i,result in enumerate(pool.map(mces_pair,pending,chunksize=1)):
            f.write(json.dumps(result)+'\n')
            if i%100==0:print('MCES_PROGRESS',i+1,'/',len(pending),result['status'],guard(),flush=True)
    write(OUT/'mces_execution.json',dict(total_unique_pairs=len(pairs),completed_before=len(done),attempted_now=len(pending),workers=workers,timeout_seconds=30,source_sha256=sha(Path(__file__)),dependencies={p.name:sha(p) for p in (OUT/'vendor').glob('*.whl')}))

def mces_report():
    rows=load(SEALED/'rows.npz');groups=rows['connectivities'];uses=read(OUT/'mces_uses.json');pairs=read(OUT/'mces_pairs.json')
    cache={json.loads(x)['key']:json.loads(x) for x in (OUT/'mces_cache.jsonl').read_text().splitlines()}
    missing=set(pairs)-set(cache);failed={k:v for k,v in cache.items() if v['status'] not in ('OK','IDENTICAL')}
    write(OUT/'mces_status.json',dict(unique_pairs=len(pairs),missing_pairs=len(missing),failed_pairs=len(failed),failure_statuses=dict(collections.Counter(v['status'] for v in failed.values())),mode_counts=dict(collections.Counter(v.get('mode') for v in cache.values()))))
    if missing or failed:raise RuntimeError(('MCES incomplete; never impute solver failures',len(missing),len(failed)))
    distance={name:np.array([np.mean([cache[k]['distance'] for k in keys]) for keys in a]) for name,a in uses.items()}
    for name,prefix in [('sccf_seed_mean','sccf'),('msalign_seed_mean','e1'),('embcos_seed_mean','e2')]:distance[name]=np.mean([distance[f'{prefix}_seed{s}'] for s in SEEDS],0)
    unique,inverse=np.unique(groups,return_inverse=True);counts=np.bincount(inverse);reports={};md=[];tex=[]
    for name in LABELS:
        if name not in distance:
            md.append([LABELS[name],'--','--']);tex.append(' & '.join(md[-1])+r' \\');continue
        d=distance[name];mean=float(d.mean());balanced=float((np.bincount(inverse,weights=d)/counts).mean());reports[name]=dict(query_weighted=mean,connectivity_balanced=balanced,queries=len(d));md.append([LABELS[name],f'{mean:.4f}',f'{balanced:.4f}']);tex.append(' & '.join(md[-1])+r' \\')
    np.savez_compressed(OUT/'mces_per_query.npz',identifiers=rows['identifiers'],connectivities=groups,**distance)
    write(OUT/'mces.json',dict(status='COMPLETE_FOR_RECOVERABLE_ROWS',posthoc_secondary=True,lower_is_better=True,metric='mean myopic MCES distance at top1, threshold15, always_stronger_bound=True; exact below threshold, valid lower bound may be returned above; uniform exact-top1 ties; seed means average per-seed distances (never score ensembles)',rows=reports,unrecoverable_rows=read(OUT/'missing_top1_evidence.json')['rows'],official_sources={p.name:sha(p) for p in (OUT/'vendor').glob('massspecgym_*.py')},seal_sha256=sha(SEALED/'predictions_seal.json')))
    (OUT/'mces.md').write_text(mdtable(['Endpoint','MCES@1 (lower)','Balanced MCES@1'],md))
    (OUT/'s7_posthoc_mces_rows.tex').write_text('% Post-hoc secondary structural distance; six rows lack sealed top1 identity.\n'+'\n'.join(tex)+'\n')
    print('MCES_REPORT',reports,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['extract','proximity','strata','mces','mces_report']);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);globals()[a.phase]()
if __name__=='__main__':main()
