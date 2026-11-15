#!/usr/bin/env python3
"""Path-only I/O adapter for the frozen top20 runner on installed PyTorch.

Old torch.load(mmap=True) rejects pathlib.Path even though the same filename
as str works. The adapter also restores the inherited formal-train negative-pool filtering
by reusing the original cache builder. No model, feature, gate or parameter changes.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.reranking import run_official_lightgbm_top20 as run
original_load=run.torch.load

def load(filename,*args,**kwargs):
 if isinstance(filename,Path):filename=str(filename)
 return original_load(filename,*args,**kwargs)
run.torch.load=load

def topk_variable(path,top_k,exclusion_mask):
 """Original topk_from_scores policy with storage-only masked padding."""
 import numpy as np
 a=np.load(path);ids=a['identifiers'];groups=a['connectivities'];ptr=a['indptr'];candidate=a['candidate_rows'];score=a['scores'];mean=a['ensemble_scores'];stored=a['ranks_ensemble_mean_logits'];n=len(ids)
 outrows=np.empty((n,top_k),np.int32);outscore=np.full((n,top_k,3),-1e30,np.float32);outmean=np.full((n,top_k),-1e30,np.float32);labels=np.full(n,-1,np.int16);rank=np.empty(n,np.int16);counts=np.empty(n,np.int16);removed=affected=0
 for i in range(n):
  st,en=map(int,ptr[i:i+2]);rows=candidate[st:en];ss=score[:,st:en].T;mm=mean[st:en]
  if exclusion_mask is not None:
   keep=~exclusion_mask[rows];keep[0]=True;nr=int((~keep).sum());removed+=nr;affected+=int(nr>0);rows=rows[keep];ss=ss[keep];mm=mm[keep]
  assert len(rows)>=5
  order=np.argsort(-mm,kind='stable');rank[i]=int(np.flatnonzero(order==0)[0])+1;sel=order[:top_k];count=len(sel);counts[i]=count
  outrows[i]=rows[sel[0]];outrows[i,:count]=rows[sel];outscore[i,:count]=ss[sel];outmean[i,:count]=mm[sel];hit=np.flatnonzero(sel==0)
  if len(hit):labels[i]=int(hit[0])
 if exclusion_mask is None:assert np.array_equal(rank.astype(np.int32),stored)
 return dict(identifiers=ids.tolist(),connectivities=groups.tolist(),candidate_rows=outrows,stage1_scores=outscore,ensemble_scores=outmean,labels=labels,baseline_ranks=rank,candidate_counts=counts,audit=dict(queries=n,top20_reachable=int((labels>=0).sum()),excluded_negative_occurrences=removed,candidate_rows_affected_by_exclusion=affected,underfilled_queries=int((counts<top_k).sum()),minimum_real_candidates=int(counts.min())))

@run.torch.inference_mode()
def export_formal(out):
 """Build top20 from the exact inherited train/val pool and tie policies."""
 import gc,json,time
 import numpy as np
 from tools.reranking import run_official_lightgbm_transfer as d2
 run.check(out);assert json.loads((out/'trees/summary.json').read_text())['continuation_pass'];run.preflight(out,'formal_export')
 d=out/'formal/export';assert not d.exists();d.mkdir(parents=True)
 oldsummary=json.loads((d2.FORMAL/'top5_projection_cache/seed20260621/summary.json').read_text())
 for key in ('negative_exclusion_manifest','negative_exclusion_indices'):
  entry=oldsummary[key];assert d2.digest(ROOT/entry['path'])==entry['sha256']
 manifest_path=ROOT/oldsummary['negative_exclusion_manifest']['path'];m=json.loads(manifest_path.read_text());rows=np.load(ROOT/oldsummary['negative_exclusion_indices']['path']);mask=np.zeros(m['molecule_rows'],bool);mask[rows.astype(np.int64)]=True
 sizes={split:json.loads((d2.FORMAL/f'stage1_candidate_scores/{split}_three_seed_logits.summary.json').read_text())['query_count'] for split in ('train','val')};ntrain=sizes['train'];n=ntrain+sizes['val'];k=run.K
 arrays={name:np.lib.format.open_memmap(d/(name+'.npy'),mode='w+',shape=shape,dtype=dt) for name,shape,dt in [('spec',(n,1024),'float16'),('seed_scores',(n,k,3),'float32'),('base',(n,k),'float32'),('candidate_rows',(n,k),'int32'),('labels',(n,),'int16'),('ranks',(n,),'int16'),('candidate_counts',(n,),'int16')]};groups=[];offset=0;started=time.time();audits={}
 for split in ('train','val'):
  score_path=d2.FORMAL/f'stage1_candidate_scores/{split}_three_seed_logits.npz';assert d2.digest(score_path)==oldsummary[split+'_scores']['sha256']
  z=topk_variable(score_path,k,mask if split=='train' else None)
  cache=run.torch.load(d2.FORMAL/f'top5_projection_cache/seed20260621/{split}_top5.pt',map_location='cpu',weights_only=False,mmap=True)
  meta=run.records(ROOT/f'outputs/msalign_stage1_full_v15/{split}_full.metadata.jsonl')
  assert z['identifiers']==cache['identifiers']==[q['identifier'] for q in meta]
  assert np.array_equal(z['baseline_ranks'].astype(np.int32),cache['baseline_ranks'].numpy())
  assert np.array_equal(z['candidate_rows'][:,:5],cache['candidate_global_rows'].numpy())
  assert np.array_equal(z['stage1_scores'][:,:5],cache['stage1_scores'].numpy())
  assert np.array_equal(z['ensemble_scores'][:,:5],cache['ensemble_scores'].numpy())
  for key in ('excluded_negative_occurrences','candidate_rows_affected_by_exclusion'):assert z['audit'][key]==cache['audit'][key]
  ng=len(z['identifiers']);sl=slice(offset,offset+ng)
  for dst,src in [('seed_scores','stage1_scores'),('base','ensemble_scores'),('candidate_rows','candidate_rows'),('labels','labels'),('ranks','baseline_ranks')]:arrays[dst][sl]=z[src]
  arrays['spec'][sl]=cache['spec_embeddings'].numpy();arrays['candidate_counts'][sl]=z['candidate_counts'];groups.extend(z['connectivities']);audits[split]=z['audit'];offset+=ng
  del z,cache;gc.collect();run.log('formal_indices',split=split,queries=ng,top5_identity_rank_score_exact=True)
 assert offset==n
 for a in arrays.values():a.flush()
 np.save(d/'groups.npy',np.array(groups));np.save(d/'official_rows.npy',np.r_[np.arange(ntrain),np.arange(n-ntrain)])
 s=run.store();model=run.primary_model(s,run.torch.device('cuda'));run.vector_table(out,d,model,s,arrays['candidate_rows'])
 run.write(d/'summary.json',dict(family='formal',queries=n,train_queries=ntrain,prediction_queries=n-ntrain,seconds=time.time()-started,top5_identity_rank_exact=True,top5_scores_exact=True,inherited_train_negative_filter_restored=True,audits=audits,files={p.name:d2.digest(p) for p in d.glob('*.npy')}))
run.formal_export=export_formal
if __name__=='__main__':run.main()
