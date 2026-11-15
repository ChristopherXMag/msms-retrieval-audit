#!/usr/bin/env python3
"""CPU review of frozen official-validation systems; no fitting or test access."""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SEEDS=(20260621,20260622,20260623)
METRICS=('recall@1','recall@5','recall@20','mrr')
E1='outputs/msalign_official_e1_20261014'; E2='outputs/embcos_official_e2_20261014'
S1='outputs/stage1_submission_0926/runs/full_r2_block_l2'
S2='outputs/stage2_official_datasafe'; D2='outputs/lightgbm_strong_endpoint_transfer_20261015'; D3='outputs/lightgbm_top20_20261015'
READS={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def bind(name):
 p=ROOT/name;READS[name]={'sha256':sha(p),'bytes':p.stat().st_size};return p
def js(name):return json.loads(bind(name).read_text())
def npz(name):return dict(np.load(bind(name),allow_pickle=False))
def contribution(r):return np.stack([r<=1,r<=5,r<=20,1/r],1).astype(np.float64)
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def measured(name):return (ROOT/name).stat().st_size

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'outputs/official_system_e4_20261016');args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
 start=time.time(); e=npz(E1+'/analysis/seed_mean_per_query.npz');e2=npz(E2+'/analysis/seed_mean_per_query.npz');d2=npz(D2+'/formal_validation/scores_ranks.npz');d3=npz(D3+'/formal/report/scores_ranks.npz');pool=npz(S2+'/stage1_candidate_scores/val_three_seed_logits.npz');ext=npz(E2+'/cache/val.rows.npz');pair=npz(S2+'/pair_reranker/formal_top5/val_ranks.npz')
 ids=e['identifiers'];groups=e['connectivities'];counts=e['candidate_counts'];n=len(ids)
 assert n==19429 and len(np.unique(groups))==3184
 for x in [e2,d2,pool,pair]:assert np.array_equal(x['identifiers'],ids)
 for x,k in [(e2,'connectivities'),(d2,'groups'),(d3,'groups'),(pool,'connectivities'),(pair,'connectivities'),(ext,'connectivities')]:assert np.array_equal(x[k],groups)
 assert np.array_equal(pool['candidate_counts'],counts) and np.array_equal(ext['lengths'],counts)
 assert np.array_equal(e['sccf'],e2['sccf']) and np.array_equal(e['embcos'],e2['embcos'])
 with bind(D3+'/formal/features/inputs.jsonl').open() as f:recs=[json.loads(x) for x in f]
 assert [r['identifier'] for r in recs]==list(ids)
 gt=ext['permuted_candidates'][np.arange(n),ext['gt_slots']]; full_pool_GT_ties=0
 for i in range(n):
  original=pool['candidate_rows'][pool['indptr'][i]:pool['indptr'][i+1]]
  full_scores=pool['ensemble_scores'][pool['indptr'][i]:pool['indptr'][i+1]]; gt_score=full_scores[original==gt[i]][0]; full_pool_GT_ties+=int((full_scores==gt_score).sum()>1)
  assert np.array_equal(original,ext['candidates'][i,:counts[i]])
  assert np.array_equal(np.sort(original),np.sort(ext['permuted_candidates'][i,:counts[i]]))
  order=np.argsort(-pool['ensemble_scores'][pool['indptr'][i]:pool['indptr'][i+1]],kind='stable')[:20]
  assert np.array_equal(original[order],d3['candidate_rows'][i])
  assert np.array_equal(d2['candidate_rows'][i],d3['candidate_rows'][i,:5])
  if d3['labels'][i]>=0:assert d3['candidate_rows'][i,d3['labels'][i]]==gt[i]
 assert full_pool_GT_ties==0
 assert np.array_equal(d3['ranks_top5_detail'],d2['ranks_detail'])
 assert np.array_equal(d3['ranks_stage1'],pool['ranks_ensemble_mean_logits'])
 src=js(S2+'/stage1_candidate_scores/val_three_seed_logits.summary.json');em=js(E2+'/cache/manifest.json');assets=js(E1+'/assets/manifest.json')
 assert src['candidate_index_manifest_sha256']==em['candidate_manifest_sha256']
 assert assets['candidate_rows']['val']['rows_sha256']==READS[E2+'/cache/val.rows.npz']['sha256']
 rows={'sccf_seed_mean':e['sccf'],'msalign_seed_mean':e['msalign'],'embcos_seed_mean':e['embcos'],'sccf_ensemble':contribution(d3['ranks_stage1']),'top5_pair':contribution(pair['reranked_ranks'])}
 for arm in ['coverage','doubled_coverage','scalar','detail']:rows['top5_'+arm]=contribution(d2['ranks_'+arm])
 for arm in ['pair','coverage','doubled_coverage','scalar','detail']:rows['top20_'+arm]=contribution(d3['ranks_'+arm])
 _,inv=np.unique(groups,return_inverse=True);sizes=np.bincount(inv)
 def group_tot(v):return np.stack([np.bincount(inv,weights=v[:,j]) for j in range(v.shape[1])],1)
 metrics={k:{'query_weighted':dict(zip(METRICS,(v.mean(0)*100).tolist())),'connectivity_balanced':dict(zip(METRICS,((group_tot(v)/sizes[:,None]).mean(0)*100).tolist()))} for k,v in rows.items()}
 # Additional paired contrasts are descriptive post-hoc E4 review, not method selection.
 contrasts=[('top20_detail','msalign_seed_mean'),('top5_detail','msalign_seed_mean'),('top20_detail','sccf_ensemble')]
 delta=np.concatenate([rows[a]-rows[b] for a,b in contrasts],axis=1);tot=group_tot(delta);means=tot/sizes[:,None];rng=np.random.default_rng(20261016);reps=10000;cl=np.empty((reps,delta.shape[1]));bal=np.empty_like(cl)
 for st in range(0,reps,32):
  ix=rng.integers(0,len(sizes),size=(min(32,reps-st),len(sizes)));cl[st:st+len(ix)]=tot[ix].sum(1)/sizes[ix].sum(1)[:,None];bal[st:st+len(ix)]=means[ix].mean(1)
 comps={}
 for j,(a,b) in enumerate(contrasts):
  block={}
  for m,k in enumerate(METRICS):
   col=j*4+m;block[k]={'query_delta_pp':float(delta[:,col].mean()*100),'connectivity_cluster_ci95_pp':(np.quantile(cl[:,col],[.025,.975])*100).tolist(),'connectivity_balanced_delta_pp':float(means[:,col].mean()*100),'connectivity_balanced_ci95_pp':(np.quantile(bal[:,col],[.025,.975])*100).tolist()}
  comps[a+'_vs_'+b]=block
 # Exact ties in the newly compared reachable top20; original full-pool tails preserved.
 ties={}
 reachable=d3['labels']>=0
 for arm in ['stage1','pair','coverage','doubled_coverage','scalar','detail']:
  v=d3['scores_'+arm][reachable];g=v[np.arange(len(v)),d3['labels'][reachable]];ties[arm]=int(((v==g[:,None]).sum(1)>1).sum())
 assert not any(ties.values())
 # Cost records deliberately distinguish process wall time, runner time and unknown latency.
 sccf=[js(f'{S1}/seed{s}/train_summary.json') for s in SEEDS]
 costs={'sccf':{'trainable_parameters_per_model':sccf[0]['parameter_counts']['model']['trainable'],'fixed_parameter_elements_per_model':1,'runner_training_seconds_per_seed':[x['elapsed_seconds'] for x in sccf],'budget':'one all-row epoch, batch128, GT+127; no validation epoch search beyond the fixed one epoch'},'msalign':{},'embcos':{}}
 for key,prefix in [('msalign',E1),('embcos',E2)]:
  tr=[js(f'{prefix}/seed{s}/train_summary.json') for s in SEEDS];res=[js(f'{prefix}/seed{s}_train.resources.json') for s in SEEDS]
  costs[key]={'trainable_parameters_per_model':tr[0]['parameters'],'runner_training_seconds_per_seed':[x['elapsed_seconds'] for x in tr],'training_process_wall_seconds_per_seed':[x['elapsed_seconds'] for x in res],'peak_CUDA_MiB':max(x['peak_cuda_mib'] for x in tr),'timing_scope':'runner includes training and repeated validation; process wall adds startup/input verification; neither is inference latency','selected_steps':[x['best']['global_step'] for x in tr]}
 # Only the four operational r2 blocks, not retained historical Morgan-r3 extras.
 fp=js('outputs/stage1_submission_0926/assets/full_sparse_direct/manifest.json');fpdir=ROOT/'outputs/stage1_submission_0926/assets/full_sparse_direct';blocks={}
 for block in ['count_morgan','count_atom_pair','rdkit_path_count','maccs']:
  b=fp['features'][block];files=[fpdir/b[k] for k in ['indices','values','indptr']];blocks[block]={'bytes':sum(p.stat().st_size for p in files),'rows':b['rows'],'nnz':b['nnz'],'measurement':'logical st_size of existing sparse arrays, shared row text omitted'}
 costs['sccf'].update({'molecule_feature_bytes':sum(b['bytes'] for b in blocks.values()),'molecule_feature_rows':fp['rows'],'feature_blocks':blocks,'feature_construction_seconds':None,'feature_construction_scope':'No complete matched timing retained; existing full-universe four-block sparse storage, not assumed cheapest.'})
 mol=js(E1+'/moldeberta_cache/manifest.json');loading=js(E1+'/moldeberta_cache/loading.json');costs['msalign']['frozen_molecule_encoder_parameters']=loading['parameters'];costs['msalign'].update({'molecule_feature_rows':mol['rows'],'molecule_embedding_bytes':measured(E1+'/moldeberta_cache/embeddings.npy'),'molecule_encoding_seconds':mol['progress']['elapsed_seconds'],'frozen_molecule_checkpoint_bytes':mol['model']['weight_bytes'],'molecule_cache_dtype':'float32 768d; referenced train/val candidate union'})
 costs['embcos'].update({'molecule_feature_rows':em['molecule_rows'],'molecule_embedding_bytes':measured(E2+'/cache/morgan4096.packed.npy'),'feature_construction_seconds':em['elapsed_seconds'],'molecule_cache_dtype':'packed binary4096, full candidate universe'})
 for key,prefix,suffix in [('sccf','outputs/run_k_official_datasafe_seed20260621','no_head'),('msalign',E1+'/dreams_contrastive','')]:
  summaries=[js(prefix+'/'+fold+('.'+suffix if suffix else '')+'.summary.json') for fold in ['train','val']]
  arrays=[prefix+'/'+fold+('.'+suffix if suffix else '')+'.npy' for fold in ['train','val']]
  costs[key]['spectrum_feature_seconds']=sum(x['elapsed_seconds'] for x in summaries);costs[key]['spectrum_feature_bytes']=sum(measured(x) for x in arrays);costs[key]['DreaMS_checkpoint_sha256']=summaries[0]['dreams_ckpt_sha256']
 assert costs['sccf']['DreaMS_checkpoint_sha256']==costs['msalign']['DreaMS_checkpoint_sha256']
 # Inspect only the small already-fitted CPU checkpoint and plain-text tree files.
 import torch
 pc=bind(D3+'/formal/pair/seed20261001.pt');pstate=torch.load(str(pc),map_location='cpu',weights_only=False,mmap=True)['model_state_dict'];pairparams=sum(t.numel() for t in pstate.values())
 treefiles=sorted((ROOT/D3/'trees').glob('detail_seed*.txt'));treecounts=[]
 for p in treefiles:
  bind(str(p.relative_to(ROOT)));leaves=[int(l.split('=')[1]) for l in p.read_text().splitlines() if l.startswith('num_leaves=')];treecounts.append({'file':str(p.relative_to(ROOT)),'trees':len(leaves),'leaves':sum(leaves),'bytes':p.stat().st_size})
 auxiliary=[js(f'{D2}/upstream/{family}/seed{seed}/{family}/summary.json') for family in ['mining0','mining1','endpoint'] for seed in SEEDS]
 pairtime=[js(f'{D3}/upstream/{f}/pair/summary.json')['seconds'] for f in ['mining0','mining1','endpoint']];pairformal=js(D3+'/formal/pair/summary.json')['seconds']
 exports=[js(f'{D3}/upstream/{f}/export/summary.json') for f in ['mining0','mining1','endpoint']]+[js(D3+'/formal/export/summary.json')]
 costs['top20_detail']={'SCCF_models_at_inference':3,'pair_models_at_inference':3,'pair_trainable_parameters_per_model':pairparams,'neural_trainable_parameters_across_inference_models':3*(costs['sccf']['trainable_parameters_per_model']+pairparams),'tree_models_at_inference':3,'trees':treecounts,'auxiliary_OOF_Stage1_fits':9,'auxiliary_Stage1_runner_seconds':sum(x['elapsed_seconds'] for x in auxiliary),'OOF_pair_phase_seconds':pairtime,'formal_pair_phase_seconds':pairformal,'pair_timing_scope':'includes epoch selection, complementary refit and cached-query scoring; not per-query latency','successful_projection_export_bytes':sum(sum((ROOT/D3/(('upstream/'+f+'/export') if f!='formal' else 'formal/export')/p).stat().st_size for p in x['files'] if p.endswith('.npy')) for f,x in zip(['mining0','mining1','endpoint','formal'],exports)),'extra_cost_exclusions':['D2 feature construction, scans and calibration/provenance audits','D3 evidence-feature generation, tree fit wall time not separately retained','all encoder pretraining, setup/debugging/interruption and controls','cold/warm full-system deployment latency and I/O'],'frozen_DreaMS_shared_across_three_models':True}
 js(D3+'/selection_decision.json');js(D3+'/independent_audit.json');js(D2+'/formal_validation/summary.json');js(D3+'/formal/report/summary.json')
 result={'status':'COMPLETE','date':'2026-10-16','official_test_evaluated':False,'queries':n,'connectivities':len(sizes),'candidate_recall':1.,'candidate_count_range':[int(counts.min()),int(counts.max())],'metrics':metrics,'contrasts':comps,'bootstrap':{'replicates':reps,'seed':20261016,'scope':'descriptive post-hoc E4 contrasts, conditional on fixed development-exposed checkpoints; no multiplicity correction, no training/selection uncertainty'},'identity_audit':{'IDs_groups_order_exact':True,'full_candidate_row_order_exact':True,'permuted_candidate_multisets_exact':True,'top20_and_top5_membership_exact':True,'reachable_GT_global_row_exact':True,'ensemble_full_pool_GT_exact_ties':full_pool_GT_ties,'reachable_GT_exact_score_ties':ties,'E1_E2_uniform_tie_expectations_retained':True},'costs':costs,'unmeasured':{'full_system_inference_latency':None,'upstream_pretraining_compute':None,'comparable_total_end_to_end_cost':None,'frozen_encoder_parameter_count':'Not independently recounted; shared DreaMS checkpoint bound by SHA, E1 additional frozen MolDeBERTa checkpoint size/revision reported separately.'},'interpretation':{'single_model_method':'MSAlign exceeds SCCF; molecular backbone effect not isolated','full_system':'D3 top20 detail improves R1/R5/MRR over E1 seed-mean comparator, but has lower R20; different aggregation, inputs and budgets','attribution':'Real feature spectra matter on fixed donor slice; negligible wrong gain and extra matched-coverage chemical discrimination unestablished','adoption':'User accepted original damage-cap failure after results; development endpoint selected, no retroactive prespecified gate PASS','position':'Controlled representation and candidate-evidence integration study, not universal fingerprint superiority, best external method, efficiency or mechanistic fragment explanation claim'},'runner_seconds':time.time()-start}
 write(out/'summary.json',result);np.savez_compressed(out/'per_query.npz',identifiers=ids,connectivities=groups,candidate_counts=counts,**rows)
 write(out/'source_manifest.json',{'files':READS,'analysis_source_sha256':sha(Path(__file__)),'official_test_evaluated':False});print(json.dumps({'status':'COMPLETE','R1_top20_vs_E1':comps['top20_detail_vs_msalign_seed_mean']['recall@1'],'R20_top20_vs_E1':comps['top20_detail_vs_msalign_seed_mean']['recall@20'],'costs':costs,'runner_seconds':result['runner_seconds']},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
