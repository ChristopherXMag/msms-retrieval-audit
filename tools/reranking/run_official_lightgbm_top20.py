#!/usr/bin/env python3
"""One frozen top20 attempt: per-checkpoint dedup caches, conditional-OOF pair/trees.

Reuses D2 isolation and Stage1 fits. No test loader, hyperparameter sweep, cache
replacement or endpoint promotion. Each subprocess owns a single bounded phase.
"""
from __future__ import annotations
import argparse, gc, json, os, shutil, subprocess, sys, time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import torch
import lightgbm as lgb
from rdkit import rdBase
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.reranking import run_official_lightgbm_transfer as d2
from tools.reranking import train_official_datasafe_pair_reranker as pair
from tools.stage1.train_msalign_projection_smoke import checkpoint_model_config
from tools.analysis.analyze_stage1_logit_ensemble import comparison
K=20
PARENT=ROOT/'outputs/lightgbm_strong_endpoint_transfer_20261015'
DEFAULT=ROOT/'outputs/lightgbm_top20_20261015'
FORMAL_CKPT=ROOT/'outputs/stage1_submission_0926/runs/full_r2_block_l2/seed20260621/checkpoint.pt'
DONOR=ROOT/'outputs/stage1_submission_0926/runs/full_r2_block_l2/seed20260621/blind_controls/spectrum_permutation_exact_mass.per_query.npz'
SEEDS=d2.TREE_SEEDS

def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.staging');tmp.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');tmp.replace(p)
def log(stage,**v):print(json.dumps(dict(stage=stage,**v)),flush=True)
def records(p):return d2.records(p)
def check(out):
 c=json.loads((out/'freeze_manifest.json').read_text())
 for p,h in c['sources'].items():assert d2.digest(ROOT/p)==h,p
 return c

def preflight(out,stage):
 d2.memory_guard()
 assert shutil.disk_usage(out).free>12*2**30,'Less than 12 GiB disk reserve'
 if stage:
  result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],check=True,capture_output=True,text=True)
  assert torch.cuda.is_available();log('gpu_preflight',phase=stage,gpu=result.stdout.strip(),name=torch.cuda.get_device_name(0),**d2.memory_state())

def freeze(out):
 assert not out.exists();d2.check_sources(PARENT)
 assert lgb.__version__=='4.6.0' and rdBase.rdkitVersion=='2023.09.6'
 sources=[Path(__file__),ROOT/'tools/reranking/run_official_lightgbm_transfer.py',ROOT/'tools/reranking/train_official_datasafe_pair_reranker.py',ROOT/'tools/reranking/train_official_lightgbm_pilot.py',ROOT/'tools/reranking/eval_official_mpfir_v2_top5.py',ROOT/'tools/stage1/train_msalign_projection_smoke.py',ROOT/'tools/analysis/analyze_stage1_logit_ensemble.py',ROOT/'tools/reranking/train_official_datasafe_score_fusion.py',ROOT/'tools/reranking/audit_official_datasafe_pair_controls.py',ROOT/'tools/reranking/train_run_k_e1_fragment_residual.py',ROOT/'train_spec_encoder/audit_magma_peak_fragment_labels.py',ROOT/'glmr/candidate_indices.py',ROOT/'glmr/sparse_features.py',ROOT/'tools/reranking/train_official_mpfir_oof_retriever.py']
 inputs=[PARENT/'freeze_manifest.json',PARENT/'labels.jsonl',PARENT/'features/train.npz',PARENT/'trees/calibration.npz',PARENT/'formal_validation/scores_ranks.npz',DONOR,FORMAL_CKPT,d2.FP,d2.CAND]
 inputs += [PARENT/f'upstream/{f}/seed{s}/{f}/checkpoint.pt' for f in d2.FAMILIES for s in d2.SEEDS]
 inputs += [PARENT/f'upstream/{f}/export/summary.json' for f in d2.FAMILIES]
 inputs += [d2.FORMAL/f'stage1_candidate_scores/{split}_three_seed_logits.summary.json' for split in ('train','val')]
 inputs += [PARENT/f'denylist_{f}/manifest.json' for f in d2.FAMILIES]
 out.mkdir(parents=True)
 c=dict(date='2026-10-15',authorization='User authorized one bounded top20 attempt after review; supersedes topK deferral only for this configuration.',
 scope='official train/calibration, conditional formal validation only; no official test',k=K,
 isolation='Reuse exact D2 8000/1024 connectivity cohort, two inner folds and all aliases/positive/negative exclusions; conditional on fixed published data-safe DreaMS features, not the entire pretraining chain.',
 upstream='Reuse all nine D2 SCCF checkpoints unchanged. Formal validation uses frozen original full-train Stage1 checkpoints/scores. No Stage1 retraining.',
 cache='Per-checkpoint sorted unique candidate global row IDs, one 1024-d fp16 vector each, int32 query-to-vector indices. No cross-family sharing or connectivity/formula merging. OOF-filtered pools with5..19 candidates retain their true lengths; padded slots are masked out of pair loss/scores, set features, coverage and LambdaRank groups. No candidate insertion or query dropping. Recompute vectors in fixed batches of 256; original top5 artifacts remain immutable. Full-pool ranking uses original full-pool GPU recipe or cached formal logits, not quantized dedup vectors.',
 pair='For each 3 OOF families and formal full-train endpoint: 3 seeds 20261001..3, original natural-loss pair, hidden256/bottleneck64/dropout.1, AdamW3e-4 WD1e-4; query batch1024 as in top5; inner5fold calibration, up to5epochs/damage8%, original lexicographic epoch rule then full-complement refit. No batch or hyperparameter sweep.',
 features='Keep original ordered10 scalar plus26 detail formulas; all set statistics over20. Fix hardcoded softmax denominator5->K. Std floor.02; coverage lower-median for evenK follows existing torch.median definition. All real candidates (up to20) must be supported; whole-query tree fallback and no GT insertion. Isotope coverage policy preserved for formal endpoint.',
 tree='scalar and detail control: 50 LambdaRank trees, depth4/leaves15/minleaf50/lr.03/L2=10/feature_fraction.8/2threads, 3 seeds averaged. Preserve lambdarank_truncation_level=5: objective still focuses near top; group size20. No hyperparameter or feature search.',
 calibration='Same exposed1024 D2 outer-calibration queries. Main comparator frozen top5 detail, replayed from old cache exactly. Continue iff deltaR1>=0,deltaMRR>0,damage<=8%; report paired cluster intervals and all seeds. Failure stops BEFORE formal top20 export/fitting/evaluation. No refit on this cohort.',
 controls='Top20 Stage1,pair,coverage,doubledcoverage,scalar,detail; main detail20 vs frozen detail5. Frozen3 exact-mass/adduct/different-connectivity val donor mappings. Keep real top20 pair and additive real coverage anchor fixed; replace all new scalar/detail spectrum-derived features; same real support. Scalar wrong controls included.',
 attribution='For each donor seed: real-minus-wrong R1/MRR cluster CI lower>0 and wrong-minus-fixed-real-anchor one-sided95% R1 upper<=0.25pp; uncertainty=>INCONCLUSIVE, no margin relaxation. This does not gate validation entry; it gates interpretation after effect continuation, as in D2. No final endpoint promotion.',
 formal_effect='Require positive paired connectivity-cluster95% R1 lower bound vs frozen top5 detail, MRR>=0 anddamage<=8%; also report matched top20controls without selecting variants.',
 analysis='10000 paired connectivity cluster bootstrap, query and connectivity balanced; R1/R5/R20/MRR, support/fallback, rescue/damage by original rank1/2-5/6-20/>20, gap closure. R20 and all ranks outside originaltop20 must remain unchanged. Validation and calibration development-exposed; no independent confirmation claim.',
 plumbing='Before fitting: K5 exact parity, actual candidate permutation, support/fallback, indexed-vs-expanded molecule reads and scores, GPU bounded smoke/RSS, independently recover oldtop5 ranks. Numerical discrepancies of dedup vectors bounded and reported; never rewrite old caches.',
 official_test_read=False,source_sha256_parent=d2.digest(PARENT/'freeze_manifest.json'),sources={str(p.relative_to(ROOT)):d2.digest(p) for p in sources},inputs={str(p.relative_to(ROOT)):d2.digest(p) for p in inputs})
 write(out/'freeze_manifest.json',c)
 for p in sources:
  dest=out/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 log('freeze',sha256=d2.digest(out/'freeze_manifest.json'))

def store():return d2.MoleculeEmbeddingStore(None,'mean',d2.FP,mol_blocks=d2.BLOCKS)
def primary_model(s,device):
 ck=torch.load(FORMAL_CKPT,map_location='cpu',weights_only=False)
 m=d2.build_projection_model(spec_dim=1024,mol_dim=s.dim,mol_block_dims=s.block_dims,mol_block_names=d2.BLOCKS,config=checkpoint_model_config(ck)).to(device)
 m.load_state_dict(ck['model_state_dict']);return m.eval()

@torch.inference_mode()
def vector_table(out,d,model,s,candidates):
 unique,inverse=np.unique(candidates,return_inverse=True);n=len(unique)
 tmp=d/'vectors.staging.npy';a=np.lib.format.open_memmap(tmp,mode='w+',shape=(n,1024),dtype='float16')
 for st in range(0,n,256):
  if st%16384==0:preflight(out,None)
  raw=s.gather(torch.from_numpy(unique[st:st+256].astype(np.int64))).to('cuda');a[st:st+256]=model.project_mol(raw).cpu().half().numpy()
 a.flush();del a;tmp.replace(d/'vectors.npy')
 np.save(d/'vector_rows.npy',unique.astype(np.int32));np.save(d/'candidate_local.npy',inverse.reshape(candidates.shape).astype(np.int32))
 # Independent index identity replay; no representation merging by connectivity.
 assert np.array_equal(unique[np.load(d/'candidate_local.npy',mmap_mode='r')],candidates)
 log('dedup_vectors',directory=str(d),unique=n,slots=int(candidates.size),fraction=n/candidates.size,GiB=(n*2048+candidates.size*4)/2**30)

@torch.inference_mode()
def export_family(out,family):
 check(out);preflight(out,'export_'+family);d=out/'upstream'/family/'export';assert not d.exists();d.mkdir(parents=True)
 old=PARENT/'upstream'/family/'export';summary=json.loads((old/'summary.json').read_text());indices=np.load(old/'official_rows.npy');n=len(indices);ntrain=summary['train_queries']
 meta=records(d2.META);cands=d2.CandidateIndexStore(d2.CAND);s=store();models=d2.models_for(PARENT,family,s,torch.device('cuda'));spec=np.load(d2.DREAMS,mmap_mode='r')
 deny=d2.CandidateNegativeExclusion(PARENT/f'denylist_{family}/manifest.json');outer=d2.CandidateNegativeExclusion(PARENT/'denylist_endpoint/manifest.json')
 arrays={name:np.lib.format.open_memmap(d/(name+'.npy'),mode='w+',shape=shape,dtype=dt) for name,shape,dt in [('spec',(n,1024),'float16'),('seed_scores',(n,K,3),'float32'),('base',(n,K),'float32'),('candidate_rows',(n,K),'int32'),('labels',(n,),'int16'),('ranks',(n,),'int16'),('candidate_counts',(n,),'int16')]}
 started=time.time();leak=0;maxdiff=0.;same_top5=True
 for start in range(0,n,32):
  if start%1024==0:preflight(out,None)
  rr=indices[start:start+32];pools=[]
  for j,i in enumerate(rr):
   pool=cands.get(meta[int(i)]['identifier'],256)
   if start+j<ntrain:pool=deny.filter_candidates(pool)[0]
   elif family!='endpoint':pool=outer.filter_candidates(pool)[0]
   assert len(pool)>=5
   pools.append(np.asarray(pool,dtype=np.int32)[np.random.default_rng(20261015+int(i)).permutation(len(pool))])
  width=max(map(len,pools));padded=torch.tensor(np.array([list(p)+[p[0]]*(width-len(p)) for p in pools]),dtype=torch.long)
  unique,inverse=torch.unique(padded.flatten(),sorted=True,return_inverse=True);unique,inverse=s.canonicalize_unique_indices(unique,inverse)
  raw=s.gather(unique).to('cuda');sv=torch.from_numpy(np.array(spec[rr],copy=True)).to('cuda');scores=[];s0=None
  for k,m in enumerate(models):
   mv=m.project_mol(raw);ss=m.project_spec(sv);proj=mv[inverse.to('cuda')].reshape(len(rr),width,1024)
   scores.append((torch.einsum('bd,bcd->bc',ss,proj)/m.temperature()).cpu().numpy())
   if k==0:s0=ss
  stack=np.stack(scores,-1);mean=stack.mean(-1,dtype=np.float64).astype(np.float32)
  for j,i in enumerate(rr):
   pos=start+j;pool=pools[j];values=mean[j,:len(pool)];gt=cands.get(meta[int(i)]['identifier'],256)[0];order=d2.candidate_order(meta[int(i)]['identifier'],pool,values,20260926);top=order[:K];ids=pool[top];hit=np.flatnonzero(ids==gt);count=len(ids);pad=K-count
   arrays['spec'][pos]=s0[j].cpu().half().numpy();arrays['candidate_counts'][pos]=count;arrays['seed_scores'][pos]=np.pad(stack[j,top],((0,pad),(0,0)),constant_values=-1e30);arrays['base'][pos]=np.pad(values[top],(0,pad),constant_values=-1e30);arrays['candidate_rows'][pos]=np.pad(ids,(0,pad),constant_values=int(ids[0]));arrays['labels'][pos]=int(hit[0]) if len(hit) else -1;arrays['ranks'][pos]=int(np.flatnonzero(pool[order]==gt)[0])+1
   if pos<ntrain:leak+=int(deny.mask[ids[ids!=gt]].sum())
   elif family!='endpoint':leak+=int(outer.mask[ids[ids!=gt]].sum())
  if start%4096==0:log('export',family=family,done=min(start+32,n),total=n,seconds=time.time()-started,**d2.memory_state())
 for a in arrays.values():a.flush()
 assert leak==0
 assert np.array_equal(arrays['candidate_rows'][:,:5],np.load(old/'candidate_rows.npy',mmap_mode='r'))
 assert np.array_equal(arrays['ranks'],np.load(old/'ranks.npy',mmap_mode='r'))
 maxdiff=float(np.max(np.abs(arrays['base'][:,:5]-np.load(old/'base.npy',mmap_mode='r'))));assert maxdiff<=1e-5,maxdiff
 np.save(d/'official_rows.npy',indices);np.save(d/'groups.npy',np.load(old/'groups.npy'))
 vector_table(out,d,models[0],s,arrays['candidate_rows'])
 # Match reconstructed first5 slot vectors numerically, never silently assuming exactness.
 vec=np.load(d/'vectors.npy',mmap_mode='r');local=np.load(d/'candidate_local.npy',mmap_mode='r');oldmol=np.load(old/'mol.npy',mmap_mode='r');diff=0.
 for st in range(0,n,512):diff=max(diff,float(np.abs(vec[local[st:st+512,:5]].astype(np.float32)-oldmol[st:st+512].astype(np.float32)).max()))
 assert diff<=.001,diff
 write(d/'summary.json',dict(family=family,queries=n,train_queries=ntrain,prediction_queries=n-ntrain,negative_leakage=leak,top5_identity_rank_exact=True,top5_score_max_abs_diff=maxdiff,recomputed_vector_max_abs_diff=diff,seconds=time.time()-started,files={p.name:d2.digest(p) for p in d.glob('*.npy')}))

# The frozen pair library's only width-specific training function allocates
# its prediction buffer at5. Override that function in memory; source is unchanged.
@torch.no_grad()
def collect(model,split,mol,batch_size,device):
 model.eval();output=np.empty((len(split.identifiers),split.candidate_local.shape[1]),np.float32)
 for (batch,) in pair.loader(np.arange(len(split.identifiers)),batch_size,0,False):output[batch.numpy()]=pair.forward_batch(model,split,mol,batch,device).cpu().numpy()
 return output
pair.collect_scores=collect
_original_forward_batch=pair.forward_batch

def forward_masked(model,split,molecule_projection,query_indices,device):
 counts=np.asarray(split.audit['candidate_counts'])[query_indices.numpy()]
 if np.all(counts==K):return _original_forward_batch(model,split,molecule_projection,query_indices,device)
 spec=split.spec[query_indices].float().to(device);local=split.candidate_local[query_indices];mol=molecule_projection[local].float().to(device)
 ss=split.stage1_scores[query_indices].to(device);base=split.ensemble_scores[query_indices].float().to(device);relative=torch.zeros((len(counts),K,11),device=device)
 for count in np.unique(counts):
  ix=torch.from_numpy(np.flatnonzero(counts==count)).to(device);relative[ix,:count]=pair.score_features(ss[ix,:count],base[ix,:count])
 values=model(spec,mol,relative,base);mask=torch.arange(K,device=device)[None]<torch.tensor(counts,device=device)[:,None]
 return values.masked_fill(~mask,-1e30)
pair.forward_batch=forward_masked

# Coverage normalization also ignores padding; the full-width path stays exact.
def anchor_scores(pair_scores,coverage,counts,multiplier=1.):
 values=np.array(pair_scores,copy=True)
 for count in np.unique(counts):
  ix=np.flatnonzero(counts==count);values[ix,:count]=d2.deployment_anchor(pair_scores[ix,:count],coverage[ix,:count],multiplier)
 return values

def split_at(d,indices,identifiers):
 at=lambda name:np.load(d/(name+'.npy'),mmap_mode='r')[indices].copy()
 return pair.Split([identifiers[i] for i in indices],at('groups'),torch.from_numpy(at('spec')),torch.from_numpy(at('candidate_local').astype(np.int64)),torch.from_numpy(at('seed_scores')),torch.from_numpy(at('base')),torch.from_numpy(at('labels').astype(np.int64)),at('ranks').astype(np.int32),{'candidate_counts':at('candidate_counts')})

def fit_pair(out,family):
 check(out);preflight(out,'pair_'+family);folder=out/('formal' if family=='formal' else 'upstream/'+family);d=folder/'export';summ=json.loads((d/'summary.json').read_text());dest=folder/'pair';assert not dest.exists();dest.mkdir()
 n=summ['train_queries'];total=summ['queries'];meta=records(d2.META);rows=np.load(d/'official_rows.npy')
 if family=='formal':ids=[r['identifier'] for r in meta]+[r['identifier'] for r in records(ROOT/'outputs/msalign_stage1_full_v15/val_full.metadata.jsonl')]
 else:ids=[meta[int(i)]['identifier'] for i in rows]
 train=split_at(d,np.arange(n),ids);pred=split_at(d,np.arange(n,total),ids);mol=torch.from_numpy(np.load(d/'vectors.npy',mmap_mode='r'))
 assert set(train.groups).isdisjoint(set(pred.groups))
 cfg=SimpleNamespace(hidden_dim=256,bottleneck_dim=64,dropout=.1,lr=3e-4,weight_decay=1e-4,batch_size=1024,max_epochs=5,calibration_folds=5,calibration_fold=0,split_seed=20261001,damage_cap=.08,training_balance='natural')
 means=[];hist={};started=time.time()
 for seed in (20261001,20261002,20261003):
  preflight(out,'pair_'+family+'_'+str(seed));epoch,history=pair.select_epoch(train,mol,seed,cfg,torch.device('cuda'));model=pair.fit_full(train,mol,epoch,seed,cfg,torch.device('cuda'))
  means.append(collect(model,pred,mol,512,torch.device('cuda')));hist[str(seed)]={'epoch':epoch,'history':history}
  torch.save(dict(model_state_dict=model.state_dict(),selected_epochs=epoch,model_seed=seed),dest/f'seed{seed}.pt')
  log('pair',family=family,seed=seed,epoch=epoch,seconds=time.time()-started,**d2.memory_state());del model;gc.collect();torch.cuda.empty_cache()
 np.save(dest/'prediction_scores.npy',np.mean(np.stack(means),0));np.save(dest/'prediction_seed_scores.npy',np.stack(means));write(dest/'summary.json',dict(family=family,history=hist,config=vars(cfg),train_queries=n,prediction_queries=total-n,seconds=time.time()-started,export_sha256=d2.digest(d/'summary.json')))

class Evidence(d2.Evidence):
 def build(self,inputs,base,coverage_override=None):
  counts=np.array([q.get('candidate_count',len(q['candidate_smiles'])) for q in inputs]);width=np.shape(base)[1]
  if np.all(counts==width):return super().build(inputs,base,coverage_override)
  result=None;reasons=Counter()
  for count in np.unique(counts):
   ix=np.flatnonzero(counts==count);short=[]
   for i in ix:
    q=dict(inputs[i]);q['candidate_smiles']=q['candidate_smiles'][:count];q['candidate_global_rows']=q['candidate_global_rows'][:count];short.append(q)
   cv=None if coverage_override is None else np.asarray(coverage_override)[ix,:count]
   b=super().build(short,np.asarray(base)[ix,:count],cv)
   if result is None:
    result={}
    for key,value in b.items():
     if isinstance(value,np.ndarray):
      shape=(len(inputs),width)+value.shape[2:] if value.ndim>=2 else (len(inputs),)
      result[key]=np.zeros(shape,dtype=value.dtype)
     elif key=='names':result[key]=value
   for key,value in b.items():
    if isinstance(value,np.ndarray):
     if value.ndim>=2:result[key][ix,:count]=value
     else:result[key][ix]=value
   reasons.update(b['support_counts'])
  result['support_counts']=dict(reasons);return result
 def _build(self,inputs,base,coverage_override=None):
  n,k=np.shape(base);valid=np.zeros((n,k,100,64),bool);mz=torch.zeros((n,100));intensity=torch.zeros((n,100));present=torch.zeros((n,100),dtype=torch.bool)
  support=np.ones(n,bool);coverage_support=np.ones(n,bool);reasons=Counter();mass=np.zeros((n,k,64),np.float32);masks=np.zeros((n,k,64),bool);parent=np.zeros((n,k),np.float32)
  for i,q in enumerate(inputs):
   observed=q['observed_spectrum'];a=torch.tensor(observed,dtype=torch.float32)
   if len(observed):mz[i,:len(a)]=a[:,0];intensity[i,:len(a)]=a[:,1];present[i,:len(a)]=True
   reason='supported'
   if q['adduct'] not in ('[M+H]+','[M+Na]+'):reason='unsupported_adduct'
   for j,smi in enumerate(q['candidate_smiles']):
    status,pa,ma,mask,iso,_,_=self.molecule(smi)
    if status!='supported':reason=status
    else:parent[i,j]=pa;mass[i,j]=ma;masks[i,j]=mask
   if reason!='supported':support[i]=coverage_support[i]=False
   elif any(self.molecule(smi)[4] for smi in q['candidate_smiles']):support[i]=False;reason='candidate_isotope_mass_unsupported'
   if not len(observed) or not bool((intensity[i]>0).any()):support[i]=coverage_support[i]=False;reason='invalid_observed_spectrum'
   reasons[reason]+=1
  for st in range(0,n,32):
   ix=np.flatnonzero((coverage_support if coverage_override is not None else support)[st:st+32])+st
   if len(ix):valid[ix],_=d2.feasibility(mz[ix],torch.tensor([inputs[i]['precursor_mz'] for i in ix]),torch.tensor([d2.ion_shift(inputs[i]['adduct']) for i in ix]),torch.from_numpy(mass[ix]),torch.from_numpy(parent[ix]),torch.from_numpy(masks[ix]),present[ix])
  cv=d2.candidate_relative_score(d2.masked_null_support(torch.ones(valid.shape),torch.from_numpy(valid)),present,intensity).numpy();computed=cv.copy()
  if coverage_override is None:cv[~support]=0
  else:cv=np.asarray(coverage_override).copy()
  scalar,ns=d2.trees.score_features(base,d2.add_coverage(base,cv))
  if k!=5:
   # Recompute the normalized exponent channel directly at its actual width.
   for col,values in ((3,base),(8,d2.add_coverage(base,cv))):
    values=np.asarray(values,np.float64);e=np.exp(values-values.max(1,keepdims=True));scalar[:,:,col]=(e/(k*d2.trees.symmetric_mean(e))).astype(np.float32)
  fi=[dict(q,observed_spectrum=np.asarray(q['observed_spectrum']).reshape(-1,2)) for q in inputs];detail,nd=d2.trees.evidence_features(fi,valid,intensity.numpy())
  return dict(features=np.concatenate([scalar,detail],-1),scalar=scalar,names=ns+nd,coverage=cv,computed_coverage=computed,valid=valid,support=support,coverage_support=coverage_support,support_counts=dict(reasons))

def build_chunks(out,inputs,base,name,formal_coverage=False):
 ev=Evidence();n,k=base.shape;features=np.empty((n,k,36),np.float32);coverage=np.empty((n,k),np.float32);support=np.empty(n,bool);statuses=Counter();started=time.time()
 for st in range(0,n,128):
  preflight(out,None);end=min(st+128,n);kwargs={'coverage_override':np.zeros((end-st,k),np.float32)} if formal_coverage else {}
  b=ev.build(inputs[st:end],base[st:end],**kwargs)
  if formal_coverage:
   # Formal coverage retains historical isotope handling; tree support remains conservative.
   b=ev.build(inputs[st:end],base[st:end],coverage_override=b['computed_coverage'])
  features[st:end]=b['features'];coverage[st:end]=b['coverage'];support[st:end]=b['support'];statuses.update(b['support_counts'])
  if st%512==0:log('features',cohort=name,done=end,total=n,seconds=time.time()-started,**d2.memory_state())
 return dict(features=features,coverage=coverage,support=support,support_counts=dict(statuses))

def read_smiles(candidates):
 wanted=set(np.asarray(candidates).ravel().tolist());result={}
 with (ROOT/'outputs/run_k_full_train/full_sparse/full_smiles.txt').open() as f:
  for i,line in enumerate(f):
   if i in wanted:result[i]=line.rstrip('\n')
 assert set(result)==wanted;return result

def training_features(out):
 check(out);dest=out/'features';assert not dest.exists();dest.mkdir()
 selected=records(PARENT/'labels.jsonl');lookup={r['official_row']:i for i,r in enumerate(selected)};n=len(selected)
 arrays={name:np.zeros(shape,dtype=dt) for name,shape,dt in [('base',(n,K),np.float32),('pair',(n,K),np.float32),('candidates',(n,K),np.int32),('rank',(n,),np.int32),('label',(n,),np.int16),('counts',(n,),np.int16)]};seen=set()
 for family in d2.FAMILIES:
  d=out/'upstream'/family/'export';s=json.loads((d/'summary.json').read_text());st=s['train_queries'];official=np.load(d/'official_rows.npy')[st:];ps=np.load(out/'upstream'/family/'pair/prediction_scores.npy')
  for j,r in enumerate(official):
   i=lookup[int(r)];assert i not in seen;seen.add(i)
   for field,source in [('base','base'),('candidates','candidate_rows'),('rank','ranks'),('label','labels'),('counts','candidate_counts')]:arrays[field][i]=np.load(d/(source+'.npy'),mmap_mode='r')[st+j]
   arrays['pair'][i]=ps[j]
 assert len(seen)==n;smiles=read_smiles(arrays['candidates']);old=records(PARENT/'features/inputs.jsonl');inputs=[]
 for i,q in enumerate(old):
  assert q['identifier']==selected[i]['identifier'];v=dict(q);v['candidate_count']=int(arrays['counts'][i]);v['candidate_global_rows']=arrays['candidates'][i].tolist();v['candidate_smiles']=[smiles[int(c)] for c in arrays['candidates'][i]];inputs.append(v)
 b=build_chunks(out,inputs,arrays['base'],'train');anchor=anchor_scores(arrays['pair'],b['coverage'],arrays['counts'])
 np.savez_compressed(dest/'train.npz',**arrays,features=b['features'],coverage=b['coverage'],anchor=anchor,support=b['support'],parts=np.array([r['pilot_partition'] for r in selected]),groups=np.array([r['connectivity'] for r in selected]),ids=np.array([r['identifier'] for r in selected]))
 with (dest/'inputs.jsonl').open('w') as f:
  for q in inputs:f.write(json.dumps(q)+'\n')
 write(dest/'summary.json',dict(queries=n,support_counts=b['support_counts'],feature_names=json.loads((PARENT/'features/summary.json').read_text())['feature_names'],train_sha256=d2.digest(dest/'train.npz')))

def predict(out,arm,feature,support,anchor):
 values=[];dim=10 if arm=='scalar' else 36;x=feature[:,:,:dim]
 for seed in SEEDS:
  m=lgb.Booster(model_file=str(out/'trees'/f'{arm}_seed{seed}.txt'));r=m.predict(x.reshape(-1,dim),num_threads=2).reshape(x.shape[:2]);r[~support]=0;values.append(r)
 v=np.stack(values);return anchor+v.mean(0),anchor[None]+v

def old_calibration():
 z=dict(np.load(PARENT/'features/train.npz'));cal=np.flatnonzero(z['parts']=='calibration');c=np.load(PARENT/'trees/calibration.npz');assert np.array_equal(cal,c['indices'])
 inputs=records(PARENT/'features/inputs.jsonl');r=d2.rank_from([inputs[i] for i in cal],z['label'][cal],z['rank'][cal],c['detail']);return cal,r

def fit_trees(out):
 check(out);dest=out/'trees';assert not dest.exists();dest.mkdir();z=dict(np.load(out/'features/train.npz'));inputs=records(out/'features/inputs.jsonl');fit=np.flatnonzero((z['parts']=='fit')&z['support']&(z['label']>=0));cal=np.flatnonzero(z['parts']=='calibration');oldcal,before=old_calibration();assert np.array_equal(oldcal,cal)
 assert set(z['groups'][fit]).isdisjoint(z['groups'][cal]);permutation=np.array([np.random.default_rng(20261010+int(i)).permutation(K) for i in fit]);target=(np.arange(K)[None]==z['label'][:,None]).astype(np.int32);keep=np.take_along_axis(np.arange(K)[None]<z['counts'][fit,None],permutation,1).ravel()
 names=json.loads((out/'features/summary.json').read_text())['feature_names'];payload={'indices':cal,'before_ranks':before};report={}
 for arm,dim in [('scalar',10),('detail',36)]:
  x=np.take_along_axis(z['features'][fit,:,:dim],permutation[:,:,None],1).reshape(-1,dim)
  for seed in SEEDS:
   params=dict(d2.trees.PARAMS,seed=seed,feature_fraction_seed=seed,data_random_seed=seed,bagging_seed=seed)
   dataset=lgb.Dataset(x[keep],label=np.take_along_axis(target[fit],permutation,1).ravel()[keep],group=z['counts'][fit],init_score=np.take_along_axis(z['anchor'][fit],permutation,1).ravel()[keep],feature_name=names[:dim],free_raw_data=False)
   m=lgb.train(params,dataset,num_boost_round=50);m.save_model(str(dest/f'{arm}_seed{seed}.txt'))
  scores,seed_scores=predict(out,arm,z['features'][cal],z['support'][cal],z['anchor'][cal]);ii=[inputs[i] for i in cal];ranks=d2.rank_from(ii,z['label'][cal],z['rank'][cal],scores);sr=np.stack([d2.rank_from(ii,z['label'][cal],z['rank'][cal],v) for v in seed_scores])
  report[arm]=dict(vs_frozen_top5_detail=comparison(before,ranks,z['groups'][cal],10000,20261015),metrics=d2.trees.calibration(before,ranks),seed_metrics=[d2.trees.calibration(before,r) for r in sr]);payload.update({f'scores_{arm}':scores,f'ranks_{arm}':ranks,f'seed_ranks_{arm}':sr})
 metric=report['detail']['metrics'];passed=metric['delta_r1']>=0 and metric['delta_mrr']>0 and metric['damage']<=.08
 # Exact legacy calibration replay and all-query accounting are prerequisites.
 assert len(cal)==1024 and len(before)==1024
 np.savez_compressed(dest/'calibration.npz',**payload)
 write(dest/'summary.json',dict(status='CALIBRATION_PASS' if passed else 'STOP_AT_TRAIN_CALIBRATION',continuation_pass=bool(passed),fit_queries=len(fit),calibration_queries=len(cal),calibration_support=int(z['support'][cal].sum()),calibration_reachable=int((z['label'][cal]>=0).sum()),arms=report,official_test_read=False,formal_top20_predictions=False))
 log('calibration',passed=bool(passed),detail=metric,fit_queries=len(fit))

@torch.inference_mode()
def formal_export(out):
 check(out);assert json.loads((out/'trees/summary.json').read_text())['continuation_pass'];preflight(out,'formal_export')
 d=out/'formal/export';assert not d.exists();d.mkdir(parents=True)
 summaries={split:json.loads((d2.FORMAL/f'stage1_candidate_scores/{split}_three_seed_logits.summary.json').read_text()) for split in ('train','val')};ntrain=summaries['train']['query_count'];n=ntrain+summaries['val']['query_count']
 arrays={name:np.lib.format.open_memmap(d/(name+'.npy'),mode='w+',shape=shape,dtype=dt) for name,shape,dt in [('spec',(n,1024),'float16'),('seed_scores',(n,K,3),'float32'),('base',(n,K),'float32'),('candidate_rows',(n,K),'int32'),('labels',(n,),'int16'),('ranks',(n,),'int16'),('candidate_counts',(n,),'int16')]};groups=[];offset=0;started=time.time()
 for split in ('train','val'):
  z=np.load(d2.FORMAL/f'stage1_candidate_scores/{split}_three_seed_logits.npz');meta=records(ROOT/f'outputs/msalign_stage1_full_v15/{split}_full.metadata.jsonl');cache=torch.load(d2.FORMAL/f'top5_projection_cache/seed20260621/{split}_top5.pt',map_location='cpu',weights_only=False,mmap=True)
  ids=z['identifiers'];ptr=z['indptr'];allcand=z['candidate_rows'];mean=z['ensemble_scores'];scores=z['scores'];rank=z['ranks_ensemble_mean_logits'];ng=len(ids)
  assert ids.tolist()==cache['identifiers']==[q['identifier'] for q in meta];assert np.array_equal(rank,cache['baseline_ranks'].numpy());groups.extend(z['connectivities'].tolist())
  for i in range(ng):
   a,b=ptr[i:i+2];rows=allcand[a:b];order=d2.candidate_order(str(ids[i]),rows,mean[a:b],20260926);sel=order[:K];gt=allcand[a];hit=np.flatnonzero(rows[sel]==gt);j=offset+i
   assert len(sel)==K;assert np.array_equal(rows[sel[:5]],cache['candidate_global_rows'][i].numpy())
   arrays['candidate_counts'][j]=K;arrays['candidate_rows'][j]=rows[sel];arrays['base'][j]=mean[a:b][sel];arrays['seed_scores'][j]=scores[:,a:b][:,sel].T;arrays['labels'][j]=int(hit[0]) if len(hit) else -1;arrays['ranks'][j]=rank[i]
  arrays['spec'][offset:offset+ng]=cache['spec_embeddings'].numpy();offset+=ng
  del z,cache;gc.collect();log('formal_indices',split=split,queries=ng)
 assert offset==n
 for a in arrays.values():a.flush()
 np.save(d/'groups.npy',np.array(groups));np.save(d/'official_rows.npy',np.r_[np.arange(ntrain),np.arange(n-ntrain)])
 s=store();model=primary_model(s,torch.device('cuda'));vector_table(out,d,model,s,arrays['candidate_rows'])
 write(d/'summary.json',dict(family='formal',queries=n,train_queries=ntrain,prediction_queries=n-ntrain,seconds=time.time()-started,top5_identity_rank_exact=True,files={p.name:d2.digest(p) for p in d.glob('*.npy')}))

def formal_features(out):
 d=out/'formal/export';summ=json.loads((d/'summary.json').read_text());start=summ['train_queries'];c=np.load(d/'candidate_rows.npy',mmap_mode='r')[start:];base=np.load(d/'base.npy',mmap_mode='r')[start:];smiles=read_smiles(c);old=records(d2.D1/'d1_inputs_val/inputs.jsonl');inputs=[]
 for i,q in enumerate(old):
  v=dict(q);v['candidate_global_rows']=c[i].tolist();v['candidate_smiles']=[smiles[int(row)] for row in c[i]];inputs.append(v)
 b=build_chunks(out,inputs,base,'formal_val',formal_coverage=True);dest=out/'formal/features';assert not dest.exists();dest.mkdir()
 np.savez_compressed(dest/'features.npz',features=b['features'],coverage=b['coverage'],support=b['support'])
 with (dest/'inputs.jsonl').open('w') as f:
  for q in inputs:f.write(json.dumps(q)+'\n')
 write(dest/'summary.json',dict(queries=len(inputs),support_counts=b['support_counts']))


def formal_report(out):
 check(out);assert json.loads((out/'trees/summary.json').read_text())['continuation_pass'];d=out/'formal/export';ds=json.loads((d/'summary.json').read_text());start=ds['train_queries'];dest=out/'formal/report';assert not dest.exists();dest.mkdir()
 load=lambda name:np.load(d/(name+'.npy'),mmap_mode='r')[start:].copy()
 base=load('base');labels=load('labels');baseline=load('ranks');groups=load('groups');candidate=load('candidate_rows');inputs=records(out/'formal/features/inputs.jsonl');b=dict(np.load(out/'formal/features/features.npz'));ps=np.load(out/'formal/pair/prediction_scores.npy');anchor=d2.deployment_anchor(ps,b['coverage'])
 scores={'stage1':base,'pair':ps,'coverage':anchor,'doubled_coverage':d2.deployment_anchor(ps,b['coverage'],2.)};seed_ranks={}
 for arm in ('scalar','detail'):
  scores[arm],seeds=predict(out,arm,b['features'],b['support'],anchor);seed_ranks[arm]=np.stack([d2.rank_from(inputs,labels,baseline,s) for s in seeds])
 ranks={k:d2.rank_from(inputs,labels,baseline,s) for k,s in scores.items()};old=dict(np.load(PARENT/'formal_validation/scores_ranks.npz'));assert np.array_equal(baseline,old['baseline_ranks']);assert np.array_equal(candidate[:,:5],old['candidate_rows']);assert np.array_equal(groups,old['groups']);before=old['ranks_detail']
 for r in ranks.values():
  assert np.array_equal(r<=20,baseline<=20);assert np.array_equal(r[baseline>20],baseline[baseline>20])
 comparisons={'detail20_vs_detail5':comparison(before,ranks['detail'],groups,10000,20261015)}
 for arm in ('stage1','pair','coverage','doubled_coverage','scalar'):comparisons['detail20_vs_top20_'+arm]=comparison(ranks[arm],ranks['detail'],groups,10000,20261015)
 main=comparisons['detail20_vs_detail5'];ci=main['paired_bootstrap']['connectivity_cluster_query_weighted']['recall@1'];effect_checks=dict(positive_R1_CI=ci['ci95_low_pp']>0,MRR_nonnegative=main['delta_query_weighted_pp']['mrr']>=0,damage_le_8pct=main['rank1_transition']['baseline_rank1_damage_rate_pct']<=8.)
 strata={}
 for name,mask in [('rank1',baseline==1),('rank2to5',(baseline>=2)&(baseline<=5)),('rank6to20',(baseline>=6)&(baseline<=20)),('rankabove20',baseline>20)]:
  strata[name]=dict(queries=int(mask.sum()),vs_top5_detail=d2.trees.calibration(before[mask],ranks['detail'][mask]))
 report=dict(status='COMPLETE_EFFECT_PASS' if all(effect_checks.values()) else 'COMPLETE_NO_CONFIRMED_TOP20_GAIN',effect_checks=effect_checks,comparisons=comparisons,queries=len(inputs),connectivities=len(set(groups)),supported=int(b['support'].sum()),support_fraction=float(b['support'].mean()),reachable=int((labels>=0).sum()),rank_strata=strata,seed_directions={a:[d2.trees.calibration(before,r) for r in rr] for a,rr in seed_ranks.items()},wrong_channel={},official_test_read=False,promotion='NOT_PROMOTED; original endpoint immutable; full-system controls and final-row freeze separate')
 np.savez_compressed(dest/'scores_ranks.npz',groups=groups,candidate_rows=candidate,labels=labels,baseline_ranks=baseline,support=b['support'],ranks_top5_detail=before,**{'scores_'+k:s for k,s in scores.items()},**{'ranks_'+k:r for k,r in ranks.items()},**{'seed_ranks_'+k:r for k,r in seed_ranks.items()})
 # Real candidate support fixes the control population; donor spectra never select it.
 donors=dict(np.load(DONOR));rec=donors['recipient_rows'].astype(int);metadata=records(ROOT/'outputs/msalign_stage1_full_v15/val_full.metadata.jsonl')
 for seed in (20260820,20260821,20260822):
  dd=donors[f'seed{seed}_donor_rows'].astype(int);assert set(dd)==set(rec) and np.all(groups[rec]!=groups[dd]);wrong=[]
  for i,j in zip(rec,dd):
   assert metadata[i]['adduct']==metadata[j]['adduct'] and round(float(metadata[i]['parent_mass']),6)==round(float(metadata[j]['parent_mass']),6)
   q=dict(inputs[i]);q['observed_spectrum']=inputs[j]['observed_spectrum'];wrong.append(q)
  wb=build_chunks(out,wrong,base[rec],'wrong_'+str(seed),formal_coverage=True);arms={};saved={}
  for arm in ('scalar','detail'):
   ws,_=predict(out,arm,wb['features'],b['support'][rec],anchor[rec]);wr=d2.rank_from([inputs[i] for i in rec],labels[rec],baseline[rec],ws);real=comparison(wr,ranks[arm][rec],groups[rec],10000,seed);wg=comparison(ranks['coverage'][rec],wr,groups[rec],10000,seed);upper=d2.one_sided_upper(ranks['coverage'][rec],wr,groups[rec],seed);ci=real['paired_bootstrap']['connectivity_cluster_query_weighted'];checks=dict(real_wrong_R1_low_positive=ci['recall@1']['ci95_low_pp']>0,real_wrong_MRR_low_positive=ci['mrr']['ci95_low_pp']>0,wrong_uplift_upper_le_025pp=upper<=.25)
   arms[arm]=dict(real_minus_wrong=real,wrong_minus_real_anchor=wg,wrong_uplift_one_sided95_upper_pp=upper,margin_pp=.25,checks=checks,gate_pass=all(checks.values()));saved['scores_'+arm]=ws;saved['ranks_'+arm]=wr
  report['wrong_channel'][str(seed)]=dict(queries=len(rec),connectivities=len(set(groups[rec])),arms=arms)
  np.savez_compressed(dest/f'wrong_seed{seed}.npz',recipient_rows=rec,donor_rows=dd,**saved);write(dest/'summary.partial.json',report)
 report['attribution_gate_pass']=all(v['arms']['detail']['gate_pass'] for v in report['wrong_channel'].values());report['attribution_status']='PASS' if report['attribution_gate_pass'] else 'INCONCLUSIVE_OR_FAILED_FROZEN_GATE';write(dest/'summary.json',report);log('completed',status=report['status'],attribution=report['attribution_status'],main_delta=main['delta_query_weighted_pp'])


def smoke(out):
 check(out);preflight(out,'bounded_smoke');old=records(PARENT/'features/inputs.jsonl')[:32];z=np.load(PARENT/'features/train.npz');base=z['base'][:32];a=d2.Evidence().build(old,base);b=Evidence().build(old,base)
 assert np.array_equal(a['features'],b['features']) and np.array_equal(a['valid'],b['valid']) and np.array_equal(a['support'],b['support'])
 # Actual physical permutation must preserve features and scores.
 perm=np.array([2,4,0,3,1]);changed=[]
 for q in old:
  v=dict(q);v['candidate_global_rows']=[q['candidate_global_rows'][i] for i in perm];v['candidate_smiles']=[q['candidate_smiles'][i] for i in perm];changed.append(v)
 c=Evidence().build(changed,base[:,perm]);assert np.array_equal(b['features'][:,perm],c['features'])
 cache=np.load(PARENT/'upstream/endpoint/export/candidate_rows.npy',mmap_mode='r')[:64];vec=np.random.default_rng(1).normal(size=(len(np.unique(cache)),1024)).astype(np.float16);u,ix=np.unique(cache,return_inverse=True);assert np.array_equal(u[ix.reshape(cache.shape)],cache)
 indexed=torch.from_numpy(vec)[torch.from_numpy(ix.reshape(cache.shape))];assert torch.equal(indexed,torch.from_numpy(vec[ix].reshape(64,5,1024)))
 # Width20 normalization and permutation checks use repeated real candidates
 # only as a plumbing fixture, never as an experiment cohort.
 fixture=[]
 for q in old[:4]:
  v=dict(q);v['candidate_global_rows']=q['candidate_global_rows']*4;v['candidate_smiles']=q['candidate_smiles']*4;fixture.append(v)
 bb20=np.tile(base[:4],(1,4));x20=Evidence().build(fixture,bb20);p20=np.random.default_rng(20261015).permutation(K);changed=[]
 for q in fixture:
  v=dict(q);v['candidate_global_rows']=[q['candidate_global_rows'][i] for i in p20];v['candidate_smiles']=[q['candidate_smiles'][i] for i in p20];changed.append(v)
 y20=Evidence().build(changed,bb20[:,p20]);assert np.array_equal(x20['features'][:,p20],y20['features'])
 assert np.allclose(x20['scalar'][:,:,[3,8]].sum(1),1,atol=1e-6)
 ragged=[dict(q,candidate_count=7) for q in fixture];rb=Evidence().build(ragged,bb20);trim=[dict(q,candidate_global_rows=q['candidate_global_rows'][:7],candidate_smiles=q['candidate_smiles'][:7]) for q in ragged];tb=Evidence().build(trim,bb20[:,:7]);assert np.array_equal(rb['features'][:,:7],tb['features']) and not rb['features'][:,7:].any()
 # Full training-batch GPU smoke with real architecture, no saved model/metric.
 pair.seed_everything(1);model=pair.PairReranker(1024,11,256,64,.1).cuda();spec=torch.randn(1024,1024,device='cuda');mol=torch.randn(1024,K,1024,device='cuda');rel=torch.randn(1024,K,11,device='cuda');bb=torch.randn(1024,K,device='cuda');opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
 for _ in range(3):
  opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(spec,mol,rel,bb),torch.zeros(1024,dtype=torch.long,device='cuda'));loss.backward();opt.step()
 split=pair.Split(['smoke']*2,np.array(['a','b']),torch.randn(2,1024),torch.arange(2*K).reshape(2,K),torch.randn(2,K,3),torch.randn(2,K),torch.tensor([0,1]),np.array([1,2]),{'candidate_counts':np.array([7,K])})
 model.eval();mp=torch.randn(2*K,1024);actual=forward_masked(model,split,mp,torch.tensor([0,1]),torch.device('cuda'))
 with torch.no_grad():
  rel7=pair.score_features(split.stage1_scores[:1,:7].cuda(),split.ensemble_scores[:1,:7].cuda());reference=model(split.spec[:1].cuda(),mp[:7][None].cuda(),rel7,split.ensemble_scores[:1,:7].cuda())
 assert torch.allclose(actual[:1,:7],reference,atol=1e-5) and bool((actual[0,7:]==-1e30).all())
 write(out/'plumbing_smoke.json',dict(status='PASS',K5_feature_parity_exact=True,physical_permutation_exact=True,indexed_dense_reads_exact=True,K20_permutation_exact=True,K20_softmax_sums_to_one=True,gpu_training_batch=1024,steps=3,plumbing_only=True,**d2.memory_state()))
 log('smoke',status='PASS',**d2.memory_state())


def audit_completion(out):
 c=check(out)
 for p,h in c['inputs'].items():assert d2.digest(ROOT/p)==h,p
 files={str(p.relative_to(out)):d2.digest(p) for p in out.rglob('*') if p.is_file() and p.suffix not in ('.log',) and p.name!='completion_manifest.json'}
 write(out/'completion_manifest.json',dict(status='COMPLETE',official_test_read=False,parent_bound_inputs_unchanged=True,files_sha256=files,total_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())))

def run_all(out):
 assert (out/'freeze_manifest.json').exists();check(out);stages=[('smoke',None)]
 for family in d2.FAMILIES:stages.extend([('export',family),('pair',family)])
 stages.extend([('features',None),('trees',None)])
 for stage,family in stages:
  cmd=[d2.PYTHON,__file__,stage,'--out',str(out)]
  if family:cmd+=['--family',family]
  log('launch',phase=stage,family=family)
  with (out/(stage+('_'+family if family else '')+'.log')).open('w') as f:subprocess.run(cmd,check=True,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 if not json.loads((out/'trees/summary.json').read_text())['continuation_pass']:
  write(out/'decision.json',dict(status='STOP_AT_TRAIN_CALIBRATION',official_test_read=False,formal_top20_predictions=False));audit_completion(out);return
 for stage in ('formal-export','pair','formal-features','formal-report'):
  cmd=[d2.PYTHON,__file__,stage,'--out',str(out)]
  if stage=='pair':cmd+=['--family','formal']
  with (out/('formal_'+stage+'.log')).open('w') as f:subprocess.run(cmd,check=True,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 report=json.loads((out/'formal/report/summary.json').read_text());write(out/'decision.json',dict(status=report['status'],attribution=report['attribution_status'],official_test_read=False,promotion='NOT_PROMOTED'));audit_completion(out)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','all','smoke','export','pair','features','trees','formal-export','formal-features','formal-report','audit'));p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--family',choices=(*d2.FAMILIES,'formal'),default='mining0');a=p.parse_args();out=a.out.resolve()
 if a.stage=='freeze':freeze(out)
 elif a.stage=='all':run_all(out)
 elif a.stage=='smoke':smoke(out)
 elif a.stage=='export':export_family(out,a.family)
 elif a.stage=='pair':fit_pair(out,a.family)
 elif a.stage=='features':training_features(out)
 elif a.stage=='trees':fit_trees(out)
 elif a.stage=='formal-export':formal_export(out)
 elif a.stage=='formal-features':formal_features(out)
 elif a.stage=='formal-report':formal_report(out)
 else:audit_completion(out)
if __name__=='__main__':main()
