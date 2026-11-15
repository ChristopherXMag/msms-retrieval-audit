#!/usr/bin/env python3
"""Fixed D3 top20-first complete-system controls; train/validation only.

Freeze -> bounded smoke -> formal run. No training, policy search or test loader.
The supported top5 branch is scored only for top20-unsupported recipients.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, os, resource, shutil, subprocess, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import h5py
import lightgbm as lgb
import numpy as np
import torch
from rdkit import Chem, rdBase
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.reranking import run_official_lightgbm_top20 as top
from tools.reranking import audit_official_datasafe_pair_controls as ctl
from tools.reranking.replay_official_d3_support_fallback import route_to_d2
from tools.analysis.analyze_stage1_logit_ensemble import contributions, grouped, metrics, interval
D2=top.PARENT;D3=top.DEFAULT;S2=top.d2.FORMAL
DEFAULT=ROOT/'outputs/official_d3_full_system_controls_20261017'
FINAL=ROOT/'outputs/official_final_freeze_20261017'
DONORS=top.DONOR
SEEDS=(20260621,20260622,20260623);TIES=ctl.TIE_SEEDS;CACHE_LIMIT=65536
SOURCE=S2/'stage1_candidate_scores/val_three_seed_logits.summary.json'
TRAIN=ROOT/'outputs/run_k_official_datasafe_seed20260621/train.no_head.npy'
TRAIN_H5=ROOT/'outputs/msalign_stage1_full_v15/train_full.hdf5'
TRAIN_META=ROOT/'outputs/msalign_stage1_full_v15/train_full.metadata.jsonl'
FP=top.d2.FP


def sha(p):return ctl.sha256_file(p)
def write(p,x):top.write(p,x)
def log(stage,**x):top.log(stage,**x)
def rel(p):return str(p.relative_to(ROOT))
def read(p):return json.loads(p.read_text())
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def freeze(out):
 assert not out.exists() and not FINAL.exists()
 source=read(SOURCE);ps=read(S2/'pair_reranker/formal_top5/summary.json')
 paths=[SOURCE,ROOT/source['npz'],ROOT/source['metadata'],ROOT/source['dreams'],TRAIN,TRAIN_H5,TRAIN_META,FP,
        DONORS,D2/'freeze_manifest.json',D2/'completion_artifact_manifest.json',D2/'formal_validation/scores_ranks.npz',
        D2/'features/summary.json',D3/'freeze_manifest.json',D3/'completion_manifest.json',D3/'selection_decision.json',
        D3/'formal/report/scores_ranks.npz',D3/'formal/features/features.npz',D3/'formal/features/inputs.jsonl',
        D3/'formal/export/summary.json',D3/'formal/export/vectors.npy',D3/'formal/export/vector_rows.npy',
        D3/'formal/export/candidate_local.npy',D3/'formal/export/spec.npy',D3/'formal/pair/summary.json',
        ROOT/ps['args']['val_top5'],ROOT/ps['args']['candidate_projection'],
        ROOT/'outputs/lightgbm_top20_support_fallback_20261017/per_query.npz',
        ROOT/'outputs/lightgbm_top20_support_fallback_20261017/summary.json']
 paths += [ROOT/x['path'] for x in source['checkpoints']]
 paths += [ROOT/x['path'] for x in ps['checkpoint_files']]
 paths += [D3/f'formal/pair/seed{s}.pt' for s in (20261001,20261002,20261003)]
 paths += [parent/f'trees/detail_seed{s}.txt' for parent in (D2,D3) for s in top.SEEDS]
 sources=[Path(__file__),Path(top.__file__),Path(top.d2.__file__),Path(ctl.__file__),
          ROOT/'tools/reranking/replay_official_d3_support_fallback.py',
          ROOT/'tools/reranking/train_official_lightgbm_pilot.py',
          ROOT/'tools/reranking/train_official_datasafe_pair_reranker.py',
          ROOT/'tools/reranking/eval_official_mpfir_v2_top5.py',
          ROOT/'tools/reranking/train_run_k_e1_fragment_residual.py',
          ROOT/'tools/stage1/train_msalign_projection_smoke.py',
          ROOT/'tools/analysis/analyze_stage1_logit_ensemble.py',
          ROOT/'train_spec_encoder/audit_magma_peak_fragment_labels.py']
 assert rdBase.rdkitVersion=='2023.09.6' and lgb.__version__=='4.6.0'
 out.mkdir(parents=True);FINAL.mkdir(parents=True)
 inputs={rel(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in dict.fromkeys(paths+sources)}
 policy=dict(status='FROZEN_BY_USER',date='2026-10-17 Asia/Shanghai',
   user_authorization='前三个都现在按顺序做了',
   endpoint='D3 top20-first, complete D2 fallback iff support5 AND NOT support20; otherwise original D3.',
   branches=['supported20: frozen D3 pair+coverage+detail','unsupported20 AND supported original Stage1 top5: complete D2 pair+coverage+detail; retain Stage1 tail','unsupported both: original D3 top20 pair+coverage'],
   top5_definition='Original Stage1 top5, before any reranking; independent frozen D2 models and width5 statistics.',
   no_new_training_or_threshold_search=True,original_D3_retained_as_comparator=True,
   original_D3_gate='FAIL damage cap, preserved; later selection and this policy are post-validation decisions.',
   official_test_read=False,inputs=inputs)
 write(FINAL/'endpoint_policy.json',policy)
 c=dict(status='FROZEN_BEFORE_CONTROL_INFERENCE',date=policy['date'],
  endpoint_policy_sha256=sha(FINAL/'endpoint_policy.json'),inputs=inputs,
  user_mean_confirmation='采用上述联合均值定义（推荐）',
  mean={'embedding':'float64 sum over all194119 official-train DreaMS rows / count, cast float32 before projection',
        'peaks':'Every train HDF5 spectrum: same valid-peak filter and DreaMS top100/basepeak normalization as D2. Sum normalized intensity into nearest0.01Da bins floor(mz/.01+.5), average over all train rows including zero bins. Select100 highest mean-intensity bins, break ties by ascending bin m/z, normalize by selected maximum; restore ascending m/z.',
        'scope':'joint constant-input control; mean embedding is not obtained by encoding the mean peak list; not a physical observed spectrum'},
  zero='zero1024 DreaMS input before spectrum projection AND empty observed peaks; valid recipient precursor/adduct retained',
  wrong='three frozen closed one-to-one exact-parent-mass6decimal/same-adduct/different-connectivity derangements. Both embedding and observed peaks use the SAME donor; recipient candidates, precursor and adduct stay fixed.',
  recompute='Full Stage1 pool ranking, top20/top5 membership, support masks, learned pair scores, width-specific coverage/detail and route per intervention; no real support/anchor inherited.',
  numerics='float32 Stage1 projections/logits; float64 seed mean then float32. Primary1024 aligned pair inputs round through frozen fp16. Original real full-pool logits and real pair cache inputs retained after fresh projection parity check; pair scorer uses NumPy float32 mean.',
  feature_names=read(D3/'features/summary.json')['feature_names'],
  tie={'physical_candidate_permutation_seed':20261017,'tie_seeds':TIES,'GT_ties':'uniform random expectation','boundary':'check membership over all3 identity-keyed tie seeds; no hidden GT-first tie credit'},
  analysis={'replicates':10000,'seed':20261017,'primary':'selected complete endpoint real minus zero/mean/each strict wrong donor, on exactly matched recipients','metrics':['recall@1','recall@5','recall@20','mrr'],'weighting':['query','recomputed-connectivity-balanced'],'uncertainty':'whole-connectivity paired bootstrap conditional on fixed fits; no multiplicity/training/selection adjustment','secondary':'original D3 controls and selected-minus-originalD3; exact random; candidate-prior limitations retained'},
  resource={'workers':0,'feature_molecule_cache':CACHE_LIMIT,'projection_batch':256,'inference_batch':32,'host_and_cgroup_available_min_GiB':32,'disk_reserve_GiB':12},
  official_test_read=False)
 write(out/'freeze_manifest.json',c)
 for p in sources:
  dest=out/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 log('policy_and_controls_frozen',contract=sha(out/'freeze_manifest.json'),policy=sha(FINAL/'endpoint_policy.json'))


def verify(out):
 c=read(out/'freeze_manifest.json')
 amendment=out/'runtime_amendments.json'
 if amendment.exists():
  a=read(amendment);assert c['inputs'][a['source_path']]['sha256']==a['initial_sha256']
  c['inputs'][a['source_path']]['sha256']=a['effective_sha256']
  c['inputs'].update(a.get('additional_inputs',{}));c['resource'].update(a.get('resource_override',{}))
 for p,info in c['inputs'].items():assert sha(ROOT/p)==info['sha256'],p
 return c


def preflight(out,label):
 top.d2.memory_guard();assert shutil.disk_usage(out).free>12*2**30
 g=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout
 assert torch.cuda.is_available()
 state=dict(phase=label,gpu=g,name=torch.cuda.get_device_name(0),**top.d2.memory_state())
 with (out/'preflights.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
 log('preflight',**state)


def means(out):
 target=out/'train_means.npz'
 if target.exists():return load(target)
 a=np.load(TRAIN,mmap_mode='r');total=np.zeros(1024,np.float64)
 for st in range(0,len(a),4096):total+=a[st:st+4096].sum(0,dtype=np.float64)
 accumulator={};titles=[]
 with h5py.File(TRAIN_H5,'r') as h:
  assert len(h['spectrum'])==len(a)==194119
  for st in range(0,len(a),1024):
   v=h['spectrum'][st:st+1024];titles.extend(t.decode() if isinstance(t,bytes) else str(t) for t in h['TITLE'][st:st+1024])
   for spec in v:
    ok=(spec[0]>0)&np.isfinite(spec).all(0)&(spec[1]>=0)
    peaks=top.d2.select_dreams_peaks(spec[0,ok].tolist(),spec[1,ok].tolist(),100)
    for p in peaks:
     index=int(np.floor(float(p['mz'])/.01+.5));accumulator[index]=accumulator.get(index,0.)+float(p['relative_intensity'])
   if st%32768==0:log('train_peak_mean',done=st,rows=len(a))
 meta=[json.loads(x) for x in TRAIN_META.read_text().splitlines()]
 assert titles==[q['identifier'] for q in meta] and all(q['fold']=='train' for q in meta)
 keys=np.array(sorted(accumulator));intensity=np.array([accumulator[k]/len(a) for k in keys])
 selected=np.lexsort((keys,-intensity))[:100];selected=selected[np.argsort(keys[selected])]
 peaks=np.stack([keys[selected]*.01,intensity[selected]/intensity[selected].max()],1)
 tmp=out/'train_means.staging.npz';np.savez(tmp,embedding=(total/len(a)).astype(np.float32),peaks=peaks,train_rows=np.array(len(a)))
 tmp.replace(target);log('train_means_complete',sha256=sha(target),peaks=len(peaks))
 return load(target)


class BoundedEvidence(top.Evidence):
 def molecule(self,s):
  value=super().molecule(s)
  if len(self.cache)>CACHE_LIMIT:del self.cache[next(iter(self.cache))]
  return value


def support(q,smiles):
 if q['adduct'] not in ('[M+H]+','[M+Na]+') or not q['observed_spectrum']:return False
 if not any(p[1]>0 for p in q['observed_spectrum']):return False
 for s in smiles:
  m=Chem.MolFromSmiles(s)
  if m is None or m.GetNumAtoms()<1 or len(Chem.GetMolFrags(m))!=1 or m.GetNumHeavyAtoms()>60 or any(a.GetIsotope() for a in m.GetAtoms()):return False
 return True


def build_evidence(ev,inputs,base):
 # Preserve the formal coverage isotope convention separately from tree support.
 z=ev.build(inputs,base,coverage_override=np.zeros_like(base));coverage=z['computed_coverage']
 if np.shape(base)[1]==5:
  # D2 inherits the original D1 GPU coverage. D3's later formal coverage is CPU.
  valid=torch.from_numpy(z['valid']).to('cuda');intensity=torch.zeros((len(inputs),100),device='cuda');present=torch.zeros((len(inputs),100),dtype=torch.bool,device='cuda')
  for i,q in enumerate(inputs):
   count=len(q['observed_spectrum'])
   if count:intensity[i,:count]=torch.tensor([p[1] for p in q['observed_spectrum']],device='cuda');present[i,:count]=True
  # Original D1 used16 supported queries/batch. CUDA reduction shape changes
  # can otherwise shift a subset by one ulp; repeat-padding has no row coupling.
  chunks=[]
  for st in range(0,len(inputs),16):
   v=valid[st:st+16];it=intensity[st:st+16];pm=present[st:st+16];n=len(v)
   def pad(t):return torch.cat([t,t[-1:].expand((16-n,)+t.shape[1:])],0) if n<16 else t
   v,it,pm=[pad(t) for t in (v,it,pm)]
   with torch.inference_mode():chunks.append(top.d2.candidate_relative_score(top.d2.masked_null_support(torch.ones(v.shape,device='cuda'),v),pm,it).cpu().numpy()[:n])
  coverage=np.concatenate(chunks)
 z=ev.build(inputs,base,coverage_override=coverage)
 return z


def pair_score(models,spec,mol,seed_scores,base,extent=None):
 device=next(models[0].parameters()).device
 s=torch.as_tensor(spec,dtype=torch.float32,device=device);m=torch.as_tensor(mol,dtype=torch.float32,device=device)
 b=torch.as_tensor(base,device=device);ss=torch.as_tensor(seed_scores,device=device)
 n=len(b);target=extent or 512
 assert target>=n
 def pad(t):return torch.cat([t,t[-1:].expand((target-n,)+t.shape[1:])],0) if target>n else t
 s,m,b,ss=[pad(t) for t in (s,m,b,ss)];features=top.pair.score_features(ss,b)
 with torch.inference_mode():return np.mean(np.stack([model(s,m,features,b).cpu().numpy() for model in models]),axis=0)[:n]


def tree_score(models,features,supported,anchor):
 values=[]
 for model in models:
  r=model.predict(features.reshape(-1,36),num_threads=2).reshape(features.shape[:2]);r[~supported]=0;values.append(r)
 return anchor+np.mean(np.stack(values),axis=0)


def cluster_delta(before,after,groups,reps=10000):
 delta=after-before;sums,sizes,means=grouped(delta,groups);rng=np.random.default_rng(20261017)
 qw=np.empty((reps,4));bal=np.empty_like(qw)
 for st in range(0,reps,64):
  ix=rng.integers(0,len(sizes),(min(64,reps-st),len(sizes)))
  qw[st:st+len(ix)]=sums[ix].sum(1)/sizes[ix].sum(1)[:,None];bal[st:st+len(ix)]=means[ix].mean(1)
 return dict(delta_query_pp=dict(zip(ctl.paired_bootstrap.__globals__['METRICS'],(delta.mean(0)*100).tolist())) if 'METRICS' in ctl.paired_bootstrap.__globals__ else metrics(delta),
             connectivity_cluster_query_weighted=interval(qw),connectivity_balanced=interval(bal),resamples=reps,seed=20261017)


class Context:
 def __init__(self,out,queries):
  self.out=out;self.source=read(SOURCE);self.pool=load(ROOT/self.source['npz']);self.ids=self.pool['identifiers'];self.groups=self.pool['connectivities']
  self.meta=[json.loads(x) for x in (ROOT/self.source['metadata']).read_text().splitlines()]
  self.inputs=top.records(D3/'formal/features/inputs.jsonl');assert [q['identifier'] for q in self.inputs]==self.ids.tolist()
  self.d2=load(D2/'formal_validation/scores_ranks.npz');self.d3=load(D3/'formal/report/scores_ranks.npz')
  self.cached_evidence20=load(D3/'formal/features/features.npz');self.cached_evidence5=load(D2/'group_ablation/features_real.npz')
  self.route_ref=load(ROOT/'outputs/lightgbm_top20_support_fallback_20261017/per_query.npz')
  self.means=means(out);self.donors=load(DONORS);self.dreams=np.load(ROOT/self.source['dreams'],mmap_mode='r')
  self.query_rows=queries;self.real_spec=[];self.models=[];self.projections=[]
  rows=np.unique(np.concatenate([self.pool['candidate_rows'][self.pool['indptr'][q]:self.pool['indptr'][q+1]] for q in queries]))
  self.local=np.full(int(self.pool['candidate_rows'].max())+1,-1,np.int32);self.local[rows]=np.arange(len(rows))
  self.smiles=top.read_smiles(rows)
  cache=out/('projection_smoke' if len(queries)<len(self.ids) else 'projection_formal');cache.mkdir(exist_ok=True)
  self.device=torch.device('cuda');store=ctl.MoleculeEmbeddingStore(None,'mean',FP,additional_fingerprint_caches=[],preload_sparse_features=False,mol_blocks=self.source['molecule_blocks'])
  for j,entry in enumerate(self.source['checkpoints']):
   payload=torch.load(str(ROOT/entry['path']),map_location='cpu',weights_only=False)
   model=ctl.build_projection_model(spec_dim=1024,mol_dim=store.dim,mol_block_dims=store.block_dims,mol_block_names=store.block_names,config=ctl.checkpoint_model_config(payload)).to(self.device)
   model.load_state_dict(payload['model_state_dict']);model.eval();self.models.append(model)
   target=cache/f'seed{SEEDS[j]}.npy'
   if not target.exists():
    temp=cache/f'seed{SEEDS[j]}.staging.npy';a=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=(len(rows),1024))
    for st in range(0,len(rows),256):
     top.d2.memory_guard();indices=torch.from_numpy(rows[st:st+256].astype(np.int64));ordered,inverse=store.canonicalize_unique_indices(indices,torch.arange(len(indices)))
     with torch.inference_mode():a[st:st+len(indices)]=model.project_mol(store.gather(ordered).to(self.device)).cpu().numpy()[inverse]
     if st%16384==0:log('project',seed=SEEDS[j],done=st,total=len(rows),**top.d2.memory_state())
    a.flush();del a;temp.replace(target)
   self.projections.append(np.load(target,mmap_mode='r'))
   with torch.inference_mode():self.real_spec.append(torch.cat([model.project_spec(torch.from_numpy(np.array(self.dreams[st:st+2048],copy=True)).to(self.device)).cpu() for st in range(0,len(self.dreams),2048)]))
  del store,payload;gc.collect()
  np.save(cache/'candidate_rows.npy',rows)
  ps=read(S2/'pair_reranker/formal_top5/summary.json');self.pair5=[];self.pair20=[]
  for entries,dest in [(ps['checkpoint_files'],self.pair5),([dict(path=rel(D3/f'formal/pair/seed{s}.pt')) for s in (20261001,20261002,20261003)],self.pair20)]:
   for entry in entries:
    payload=torch.load(str(ROOT/entry['path']),map_location='cpu',weights_only=False)
    model=top.pair.PairReranker(1024,11,256,64,.1).to(self.device);model.load_state_dict(payload['model_state_dict']);dest.append(model.eval())
  self.trees5=[lgb.Booster(model_file=str(D2/f'trees/detail_seed{s}.txt')) for s in top.SEEDS]
  self.trees20=[lgb.Booster(model_file=str(D3/f'trees/detail_seed{s}.txt')) for s in top.SEEDS]
  self.ev=BoundedEvidence()
  self.frozen5=torch.load(str(ROOT/ps['args']['val_top5']),map_location='cpu',weights_only=False,mmap=True)
  self.molecule5=torch.load(str(ROOT/ps['args']['candidate_projection']),map_location='cpu',weights_only=False,mmap=True)['embeddings']
  self.frozen20_spec=np.load(D3/'formal/export/spec.npy',mmap_mode='r');self.frozen20_mol=np.load(D3/'formal/export/vectors.npy',mmap_mode='r');self.frozen20_local=np.load(D3/'formal/export/candidate_local.npy',mmap_mode='r')
  self.val_start=read(D3/'formal/export/summary.json')['train_queries']
  # Real raw peaks were frozen with the same HDF5 top100 pipeline.
  self.max_logit_error=0.;self.max_pair20_error=0.;self.max_pair5_error=0.

 def evaluate(self,name,queries,donor_rows=None):
  n=len(queries);ranks={k:np.empty(n,np.int32) for k in ('stage1','original_d3','selected')};vals={k:np.empty((n,4)) for k in ranks}
  saved={k:[] for k in ('candidate_rows','stage1_scores','pair20','coverage20','detail20','support20','support5','route5','routed_order','selected_scores','source_rows','gt_local','pair5_rows','pair5_scores')}
  gt_ties=Counter();boundary_ties=0;permutation_checks=0
  for st in range(0,n,32):
   top.d2.memory_guard();current=queries[st:st+32];count=len(current)
   source=current if donor_rows is None else donor_rows[st:st+32]
   if name in ('zero','train_mean'):
    raw=np.zeros(1024,np.float32) if name=='zero' else self.means['embedding']
    with torch.inference_mode():spectra=[model.project_spec(torch.from_numpy(raw).to(self.device)[None]).expand(count,-1) for model in self.models]
   else:spectra=[a[source].to(self.device) for a in self.real_spec]
   pools=[self.pool['candidate_rows'][self.pool['indptr'][q]:self.pool['indptr'][q+1]] for q in current];width=max(map(len,pools))
   pad=np.stack([np.pad(p,(0,width-len(p)),mode='edge') for p in pools]);local=self.local[pad]
   score=[]
   with torch.inference_mode():
    for model,mol,spec in zip(self.models,self.projections,spectra):
     v=torch.from_numpy(np.array(mol[local],copy=True)).to(self.device);score.append((torch.einsum('bd,bkd->bk',spec,v)/model.temperature()).cpu().numpy())
   score=np.stack(score,-1)
   if name=='real':
    for i,q in enumerate(current):
     a,b=self.pool['indptr'][q:q+2];exact=self.pool['scores'][:,a:b].T
     self.max_logit_error=max(self.max_logit_error,float(np.abs(score[i,:len(pools[i])]-exact).max()));score[i,:len(pools[i])]=exact
   ensemble=score.astype(np.float64).mean(-1).astype(np.float32)
   selected=[];baseline=[];bases=[];ss=[];inputs=[]
   for i,q in enumerate(current):
    identifier=str(self.ids[q]);pool=pools[i];scores=ensemble[i,:len(pool)]
    perm=np.random.default_rng(20261017+int(q)).permutation(len(pool))
    order=perm[ctl.candidate_order(identifier,pool[perm],scores[perm],TIES[0])]
    original=ctl.candidate_order(identifier,pool,scores,TIES[0]);assert np.array_equal(pool[order],pool[original]);permutation_checks+=1
    for tseed in TIES[1:]:
     alternate=ctl.candidate_order(identifier,pool[perm],scores[perm],tseed)
     assert np.array_equal(pool[perm[alternate[:20]]],pool[order[:20]]),'Top20 membership/order tie sensitivity requires explicit alternate evaluation'
    boundary_ties+=int(scores[order[19]]==scores[order[20]])
    baseline.append(int(np.flatnonzero(order==0)[0])+1);selected.append(order[:20]);bases.append(scores[order[:20]]);ss.append(score[i,order[:20]])
    qin=dict(self.inputs[q]);qin['candidate_global_rows']=pool[order[:20]].tolist();qin['candidate_smiles']=[self.smiles[int(row)] for row in pool[order[:20]]]
    qin['observed_spectrum']=[] if name=='zero' else self.means['peaks'].tolist() if name=='train_mean' else self.inputs[int(source[i])]['observed_spectrum']
    inputs.append(qin)
    vals['stage1'][st+i]=ctl.uniform_tie_contributions(scores,0);gt_ties['stage1']+=int((scores==scores[0]).sum()>1)
   selected=np.stack(selected);base=np.asarray(bases,np.float32);ss=np.asarray(ss,np.float32);baseline=np.asarray(baseline,np.int32)
   cand=np.take_along_axis(pad,selected,axis=1);mol=np.array(self.projections[0][self.local[cand]],copy=True).astype(np.float16).astype(np.float32);spec=spectra[0].cpu().numpy().astype(np.float16).astype(np.float32)
   if name=='real':
    assert np.array_equal(cand,self.d3['candidate_rows'][current]);indices=self.val_start+current
    mol=np.array(self.frozen20_mol[self.frozen20_local[indices]],copy=True).astype(np.float32);spec=np.array(self.frozen20_spec[indices],copy=True).astype(np.float32)
   pair20=pair_score(self.pair20,spec,mol,ss,base,min(512,len(self.ids)-(int(current[0])//512)*512) if name=='real' else None)
   if name=='real':self.max_pair20_error=max(self.max_pair20_error,float(np.abs(pair20-self.d3['scores_pair'][current]).max()));assert np.array_equal(pair20,self.d3['scores_pair'][current]),'Real pair20 score replay drift'
   names=read(self.out/'freeze_manifest.json')['feature_names']
   if name=='real':built=dict(features=self.cached_evidence20['features'][current],coverage=self.cached_evidence20['coverage'][current],support=self.cached_evidence20['support'][current],names=names)
   elif name=='zero':built=dict(features=np.zeros((count,20,36),np.float32),coverage=np.zeros((count,20),np.float32),support=np.zeros(count,bool),names=names)
   else:built=build_evidence(self.ev,inputs,base)
   assert built['names']==names
   s20=np.asarray([support(q,q['candidate_smiles']) for q in inputs]);s5=np.asarray([support(q,q['candidate_smiles'][:5]) for q in inputs]);assert np.array_equal(s20,built['support'])
   fallback=route_to_d2(s5,s20);anchor=top.d2.deployment_anchor(pair20,built['coverage']);detail=tree_score(self.trees20,built['features'],s20,anchor)
   if name=='real':
    assert np.array_equal(s20,self.d3['support'][current]);assert np.array_equal(s5,self.d2['supported'][current])
    assert np.array_equal(detail,self.d3['scores_detail'][current]),('Real tree20 replay drift',np.abs(detail-self.d3['scores_detail'][current]).max())
   orders=np.empty_like(cand);selected_scores=detail.copy();pair5out=np.full((count,5),np.nan,np.float32)
   ix=np.flatnonzero(fallback)
   if len(ix):
    in5=[]
    for i in ix:
     q=dict(inputs[i]);q['candidate_smiles']=q['candidate_smiles'][:5];q['candidate_global_rows']=q['candidate_global_rows'][:5];in5.append(q)
    mol5=mol[ix,:5].copy();spec5=spec[ix].copy()
    if name=='real':
     mol5=self.molecule5[self.frozen5['candidate_local_indices'][current[ix]].long()].float().numpy();spec5=self.frozen5['spec_embeddings'][current[ix]].float().numpy()
    ps=pair_score(self.pair5,spec5,mol5,ss[ix,:5],base[ix,:5],min(512,len(self.ids)-(int(current[ix][0])//512)*512) if name=='real' else None)
    if name=='real':b5=dict(features=self.cached_evidence5['features'][current[ix]],coverage=self.cached_evidence5['coverage'][current[ix]],support=s5[ix])
    else:b5=build_evidence(self.ev,in5,base[ix,:5])
    assert b5['support'].all()
    d5=tree_score(self.trees5,b5['features'],b5['support'],top.d2.deployment_anchor(ps,b5['coverage']));pair5out[ix]=ps
    if name=='real':
     self.max_pair5_error=max(self.max_pair5_error,float(np.abs(ps-self.d2['scores_pair'][current[ix]]).max()));assert np.array_equal(ps,self.d2['scores_pair'][current[ix]])
     assert np.array_equal(d5,self.d2['scores_detail'][current[ix]]),'Real D2 branch score replay drift'
    selected_scores[ix,:5]=d5;selected_scores[ix,5:]=np.nan
   for i,q in enumerate(current):
    target=int(self.pool['candidate_rows'][self.pool['indptr'][q]]);label=np.flatnonzero(cand[i]==target);identifier=str(self.ids[q])
    o20=ctl.candidate_order(identifier,cand[i],detail[i],TIES[0]);orders[i]=cand[i,o20]
    r20=baseline[i] if not len(label) else int(np.flatnonzero(o20==label[0])[0])+1
    v20=vals['stage1'][st+i].copy() if not len(label) else ctl.uniform_tie_contributions(detail[i],label[0])
    ranks['original_d3'][st+i]=r20;vals['original_d3'][st+i]=v20
    if len(label):gt_ties['original_d3']+=int((detail[i]==detail[i,label[0]]).sum()>1)
    if fallback[i]:
     order5=ctl.candidate_order(identifier,cand[i,:5],selected_scores[i,:5],TIES[0]);orders[i]=np.r_[cand[i,:5][order5],cand[i,5:]]
     if len(label) and label[0]<5:
      v=ctl.uniform_tie_contributions(selected_scores[i,:5],label[0]);r=int(np.flatnonzero(order5==label[0])[0])+1
      gt_ties['selected']+=int((selected_scores[i,:5]==selected_scores[i,label[0]]).sum()>1)
     else:r=baseline[i];v=vals['stage1'][st+i].copy()
    else:r=r20;v=v20;gt_ties['selected']+=int(len(label)>0 and (detail[i]==detail[i,label[0]]).sum()>1)
    ranks['selected'][st+i]=r;vals['selected'][st+i]=v;ranks['stage1'][st+i]=baseline[i]
    perm=np.arange(20)[::-1];oo=ctl.candidate_order(identifier,cand[i,perm],detail[i,perm],TIES[0]);assert np.array_equal(cand[i,perm][oo],cand[i,o20])
   if name=='real':
    assert np.array_equal(ranks['original_d3'][st:st+count],self.d3['ranks_detail'][current])
    assert np.array_equal(ranks['selected'][st:st+count],self.route_ref['ranks_routed_d3'][current])
   fields=dict(candidate_rows=cand,stage1_scores=base,pair20=pair20,coverage20=built['coverage'],detail20=detail,support20=s20,support5=s5,route5=fallback,routed_order=orders,selected_scores=selected_scores,source_rows=source,gt_local=np.asarray([np.flatnonzero(cand[i]==pools[i][0])[0] if np.any(cand[i]==pools[i][0]) else -1 for i in range(count)]),pair5_rows=cand[:,:5],pair5_scores=pair5out)
   for k,v in fields.items():saved[k].append(v)
   if st%512==0:log('mode',name=name,done=st,total=n,**top.d2.memory_state())
  assert self.max_logit_error<=5e-5
  result=dict(queries=queries,groups=self.groups[queries],identifiers=self.ids[queries],**{k:np.concatenate(v) for k,v in saved.items()},**{'ranks_'+k:v for k,v in ranks.items()},**{'values_'+k:v for k,v in vals.items()})
  assert np.array_equal(ranks['selected']<=20,ranks['stage1']<=20)
  dest=self.out/self.job_name;np.savez_compressed(dest/(name+'.npz'),**result)
  info=dict(queries=n,connectivities=len(np.unique(result['groups'])),support20=int(result['support20'].sum()),support5=int(result['support5'].sum()),top5_fallback=int(result['route5'].sum()),both_unsupported=int((~result['support5']&~result['support20']).sum()),metrics={k:dict(query_weighted=metrics(v),connectivity_balanced=metrics(grouped(v,result['groups'])[2])) for k,v in vals.items()},gt_exact_ties=dict(gt_ties),top20_boundary_ties=boundary_ties,all_candidate_permutations=permutation_checks,real_logit_max_abs_error=self.max_logit_error,real_pair20_max_abs_error=self.max_pair20_error,real_pair5_max_abs_error=self.max_pair5_error,resources=top.d2.memory_state())
  write(dest/(name+'.summary.json'),info);log('mode_complete',name=name,summary=info)
  return result,info


def run(out,limit):
 c=verify(out);name='smoke32' if limit else 'formal';dest=out/name;assert not dest.exists()
 if not limit:assert read(out/'smoke32/completion.json')['status']=='PASS'
 preflight(out,name);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 source=read(SOURCE);n=len(np.load(ROOT/source['dreams'],mmap_mode='r'));queries=np.arange(min(limit,n) if limit else n)
 dest.mkdir();ctx=Context(out,queries);ctx.job_name=name;outputs={};reports={}
 real,info=ctx.evaluate('real',queries);outputs['real']=real;reports['real']=info
 for mode in ('zero','train_mean'):
  result,info=ctx.evaluate(mode,queries);outputs[mode]=result;reports[mode]=info
 rec=ctx.donors['recipient_rows'].astype(int);mask=rec<len(queries)
 for seed in (20260820,20260821,20260822):
  q=rec[mask];d=ctx.donors[f'seed{seed}_donor_rows'][mask].astype(int)
  assert len(np.unique(d))==len(d) and (limit or set(d)==set(q)) and np.all(ctx.groups[q]!=ctx.groups[d])
  for a,b in zip(q,d):assert ctx.meta[a]['adduct']==ctx.meta[b]['adduct'] and round(float(ctx.meta[a]['parent_mass']),6)==round(float(ctx.meta[b]['parent_mass']),6)
  if len(q):
   result,info=ctx.evaluate(f'wrong_seed{seed}',q,d);outputs[f'wrong_seed{seed}']=result;reports[f'wrong_seed{seed}']=info
 for mode,result in outputs.items():
  if mode=='real':continue
  ix=result['queries'];groups=result['groups'];reports[mode]['real_minus_control']={}
  for arm in ('stage1','original_d3','selected'):
   reports[mode]['real_minus_control'][arm]=cluster_delta(result['values_'+arm],real['values_'+arm][ix],groups,100 if limit else 10000)
 random=np.column_stack([np.minimum(k,ctx.pool['candidate_counts'][queries])/ctx.pool['candidate_counts'][queries] for k in (1,5,20)]+[np.array([np.sum(1/np.arange(1,int(count)+1))/count for count in ctx.pool['candidate_counts'][queries]])])
 summary=dict(status='PLUMBING_ONLY' if limit else 'COMPLETE',official_test_read=False,model_fitting=False,policy_sha256=sha(FINAL/'endpoint_policy.json'),contract_sha256=sha(out/'freeze_manifest.json'),modes=reports,exact_random=dict(query_weighted=metrics(random),connectivity_balanced=metrics(grouped(random,ctx.groups[queries])[2])),real_rank_replay_exact=True,real_score_replay_exact=True,controls_recompute_support_shortlists_and_routing=True,mean_definition=c['mean'],workers=0,feature_cache_bound=CACHE_LIMIT,resources=top.d2.memory_state())
 write(dest/'summary.json',summary)
 for p,entry in c['inputs'].items():assert sha(ROOT/p)==entry['sha256'],p
 write(dest/'completion.json',dict(status='PASS',plumbing_only=bool(limit),official_test_read=False,input_hashes_unchanged=True,files_sha256={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
 log('complete',job=name,resources=summary['resources'])


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--phase',choices=['freeze','smoke','formal'],required=True);a=p.parse_args();out=a.out.resolve();assert out.is_relative_to(ROOT/'outputs')
 if a.phase=='freeze':freeze(out)
 else:run(out,32 if a.phase=='smoke' else 0)
if __name__=='__main__':main()
