#!/usr/bin/env python3
"""Assemble and verify a fixed S6/E5 evidence package without test access."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from rdkit import Chem, rdBase
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.analysis.analyze_stage1_logit_ensemble import contributions, grouped, metrics, sha256_file
from tools.reranking.audit_official_d3_full_system import cluster_delta, verify
from glmr.embcos import uniform_tie_metrics
SEEDS=(20260621,20260622,20260623)
OUT=ROOT/'outputs/official_final_freeze_20261017'
CONTROLS=ROOT/'outputs/official_d3_full_system_controls_20261017'
D2=ROOT/'outputs/lightgbm_strong_endpoint_transfer_20261015'
D3=ROOT/'outputs/lightgbm_top20_20261015'
E1=ROOT/'outputs/msalign_official_e1_20261014'
E2=ROOT/'outputs/embcos_official_e2_20261014'
S1=ROOT/'outputs/stage1_submission_0926/runs/full_r2_block_l2'
S2=ROOT/'outputs/stage2_official_datasafe'
READS={}


def rel(p):return str(p.relative_to(ROOT))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def bind(p,expected=None,role='evidence'):
 p=p.resolve();assert p.is_relative_to(ROOT)
 assert not any(part.startswith('test_') or part.startswith('test.') or part=='test' for part in p.parts),'No test query artifacts allowed'
 key=rel(p)
 if key not in READS:
  value=sha256_file(p);READS[key]=dict(sha256=value,bytes=p.stat().st_size,role=role)
  print('HASH',key,flush=True)
 if expected is not None:assert READS[key]['sha256']==expected,(key,expected,READS[key]['sha256'])
 return p

def js(p,expected=None,role='evidence'):return json.loads(bind(p,expected,role).read_text())
def npz(p,expected=None):
 with np.load(bind(p,expected),allow_pickle=False) as z:return {k:z[k] for k in z.files}
def summarize(v,groups):return dict(query_weighted=metrics(v),connectivity_balanced=metrics(grouped(v,groups)[2]))


def read_json_source_blocks():return js(S2/'stage1_candidate_scores/val_three_seed_logits.summary.json')['molecule_blocks']


def collect_inputs(out):
 """Hash current files and verify parent result bindings; no model fitting."""
 c=verify(CONTROLS)
 for p,entry in c['inputs'].items():bind(ROOT/p,entry['sha256'],'control_input_or_model')
 bind(CONTROLS/'runtime_amendments.json');bind(CONTROLS/'freeze_manifest.json')
 complete=js(CONTROLS/'formal/completion.json');assert complete['status']=='PASS' and not complete['plumbing_only']
 for file,h in complete['files_sha256'].items():bind(CONTROLS/'formal'/file,h,'complete_system_controls')
 bind(CONTROLS/'train_means.npz')
 for p in sorted((CONTROLS/'projection_formal').glob('*.npy')):
  bind(p,role='complete_control_derived_projection_cache')
  a=np.load(p,mmap_mode='r');assert a.shape[0]==828111
  for st in range(0,len(a),4096):assert np.isfinite(a[st:st+4096]).all()
  if p.name.startswith('seed'):assert a.shape==(828111,1024) and a.dtype==np.float32
 for parent in (E1,E2):
  m=js(parent/'result_manifest.json');assert m['status']=='PASS'
  for p,entry in m['files'].items():bind(ROOT/p,entry['sha256'],'external_frozen_result_or_model')
  for p in [parent/'analysis/summary.json',parent/'analysis/seed_mean_per_query.npz',parent/'execution_complete.json',(parent/'environment.json' if parent==E2 else parent/'seed20260621/train_summary.json')]:bind(p)
 # Frozen feature files: hash in streaming blocks; never materialize large arrays.
 fpdir=ROOT/'outputs/stage1_submission_0926/assets/full_sparse_direct'
 fm=js(fpdir/'manifest_historical_projection_order.json',role='SCCF_feature_contract')
 js(fpdir/'manifest.json',role='SCCF_feature_provenance')
 for block in ('count_morgan','count_atom_pair','rdkit_path_count','maccs'):
  b=fm['features'][block]
  for field in ('indices','values','indptr'):bind(fpdir/b[field],role='SCCF_feature_cache')
 bind(ROOT/fm['base_cache'],fm.get('smiles_sha256'),'molecule_row_identity')
 cm=ROOT/'outputs/run_k_full_train/candidate_index_full_v15_mass_cap256_full_order/manifest.json';candidate=js(cm)
 for field in ('indices','indptr','identifiers'):bind(cm.parent/candidate[field],candidate.get('identifier_sha256') if field=='identifiers' else None,'shared_candidate_index_identity_transductive')
 deny=ROOT/'outputs/stage1_submission_0926/assets/heldout_connectivity_exclusion/manifest.json';dm=js(deny);bind(deny.parent/'excluded_rows.npy',role='inherited_negative_exclusion')
 for parent in (E1,E2):
  if parent==E1:
   a=js(parent/'assets/manifest.json');m=js(parent/'moldeberta_cache/manifest.json')
   for file,h in m['files'].items():bind(parent/'moldeberta_cache'/file,h,'E1_molecule_cache')
   for file,h in m['model']['files'].items():bind(ROOT/m['model']['local_dir']/file,h,'E1_frozen_molecule_encoder_tokenizer')
   for fold in ('train','val'):bind(parent/f'dreams_contrastive/{fold}.npy',a['spectra'][fold]['cache_sha256'],'E1_spectrum_cache')
   for p in [parent/'frozen_policy_inputs_audit.json',parent/'prior_followup_20261014/summary.json',parent/'prior_followup_20261014/integrity_audit.json']:bind(p)
  else:
   a=js(parent/'cache/manifest.json');bind(parent/'cache/morgan4096.packed.npy',a['packed_sha256'],'E2_molecule_cache');bind(parent/'cache/negative_exclusion.npy',a['negative_exclusion_sha256'],'external_effective_negative_exclusion')
   for fold in ('train','val'):bind(parent/f'cache/{fold}.bins.npy',a['spectra'][fold]['sha256'],'E2_spectrum_cache')
  for fold in ('train','val'):
   bind(E2/f'cache/{fold}.rows.npz',a['candidate_rows'][fold]['rows_sha256'] if parent==E1 else a['files'][fold]['rows_sha256'],'external_candidate_rows')
   if parent==E1:bind(parent/f'assets/{fold}.metadata.npz',a['metadata_arrays_sha256'][fold],'E1_metadata_inputs')
 for fold in ('train','val'):bind(ROOT/f'outputs/msalign_stage1_full_v15/{fold}_full.metadata.jsonl',role='official_split_identity')
 shared_encoder=ROOT/'external/DreaMS/dreams/models/pretrained/embedding_model_massspecgym.ckpt'
 expected=js(ROOT/'outputs/run_k_official_datasafe_seed20260621/train.no_head.summary.json')['dreams_ckpt_sha256']
 bind(shared_encoder,expected,'data_safe_spectrum_encoder')
 for seed in SEEDS:
  for file in ('checkpoint.pt','train_summary.json','val_ranks.json'):bind(S1/f'seed{seed}'/file)
  for file in ('constant_spectrum_full_val.json','constant_spectrum_full_val.per_query.npz','spectrum_permutation_exact_mass.json','spectrum_permutation_exact_mass.per_query.npz'):bind(S1/f'seed{seed}/blind_controls'/file)
 adversary=ROOT/'outputs/stage1_submission_0926/audits/official_candidate_only_20260929';assert adversary.exists()
 for p in sorted(adversary.iterdir()):
  if p.is_file():bind(p,role='train_only_candidate_prior_audit')
 for p in [ROOT/'outputs/official_system_e4_20261016/summary.json',ROOT/'outputs/official_system_e4_20261016/per_query.npz',S2/'pair_controls/formal_v3/summary.json',S2/'pair_controls/formal_v3/per_query_ranks.npz',S2/'stage1_candidate_scores/val_three_seed_logits.npz',D2/'formal_validation/summary.json',D3/'formal/report/summary.json']:bind(p)
 # Exact feature order, frozen tree widths/recipe and scorer metadata.
 names=js(D3/'features/summary.json')['feature_names'];assert len(names)==36 and names==js(D2/'features/summary.json')['feature_names']==c['feature_names']
 import lightgbm as lgb
 for parent in (D2,D3):
  for seed in (20261010,20261011,20261012):
   model=lgb.Booster(model_file=str(bind(parent/f'trees/detail_seed{seed}.txt')));assert model.num_feature()==36 and model.num_trees()==50
 write(out/'feature_order.json',dict(ordered_feature_names=names,SCCF_requested_blocks=read_json_source_blocks(),SCCF_frozen_projection_block_order=fm.get('projection_order'),feature_names_sha256=hashlib.sha256(json.dumps(names,separators=(',',':')).encode()).hexdigest(),scalar_columns=10,detail_columns=26,pair_relative_columns=11,shortlist_widths=[5,20],tree_standard_deviation_floor=.02,coverage_weight=.25,formal_coverage='original isotope policy; tree support remains conservative'))
 return c


def identity_audit():
 metas={}
 for fold in ('train','val'):
  p=bind(ROOT/f'outputs/msalign_stage1_full_v15/{fold}_full.metadata.jsonl');rows=[json.loads(x) for x in p.read_text().splitlines()]
  assert all(q['fold']==fold for q in rows)
  strings={q['canonical_isomeric_smiles'] for q in rows};mapping={}
  for s in strings:
   mol=Chem.MolFromSmiles(s);assert mol is not None;mapping[s]=Chem.MolToInchiKey(mol)[:14]
  metas[fold]=(rows,np.array([mapping[q['canonical_isomeric_smiles']] for q in rows]))
 assert set(metas['train'][1]).isdisjoint(set(metas['val'][1]))
 pool=npz(S2/'stage1_candidate_scores/val_three_seed_logits.npz');assert np.array_equal(pool['identifiers'],[q['identifier'] for q in metas['val'][0]])
 assert np.array_equal(pool['connectivities'],metas['val'][1])
 old=npz(D3/'formal/report/scores_ranks.npz');d2=npz(D2/'formal_validation/scores_ranks.npz');ext=npz(E2/'cache/val.rows.npz')
 assert np.array_equal(d2['groups'],pool['connectivities']) and np.array_equal(old['groups'],pool['connectivities'])
 assert np.array_equal(ext['identifiers'],pool['identifiers']) if 'identifiers' in ext else True
 for i in range(len(pool['identifiers'])):
  a,b=pool['indptr'][i:i+2];rows=pool['candidate_rows'][a:b];assert len(rows)==len(set(rows.tolist()))
  assert np.array_equal(rows,ext['candidates'][i,:b-a]) and rows[0]==ext['permuted_candidates'][i,ext['gt_slots'][i]]
  top=rows[np.argsort(-pool['ensemble_scores'][a:b],kind='stable')[:20]]
  assert np.array_equal(top,old['candidate_rows'][i]) and np.array_equal(top[:5],d2['candidate_rows'][i])
 return dict(status='PASS',train_queries=len(metas['train'][0]),val_queries=len(metas['val'][0]),train_connectivities=len(set(metas['train'][1])),val_connectivities=len(set(metas['val'][1])),train_val_connectivity_overlap=0,stored_InChIKey_not_trusted=True,canonical_strings='frozen RDKit2025.09.6 canonical-isomeric strings; no recanonicalization replacing cache IDs',current_connectivity_verifier_rdkit=rdBase.rdkitVersion,candidate_source_order_and_duplicates_exact=True,candidate_recall=1.,transductive_identity_policy='shared unlabeled candidate universe and inherited heldout-negative denylist, no clean-claim inference; no test spectrum/label/rank file loaded')


def analyze_rows(out):
 plan=js(out/'result_rows_plan.json');e4=npz(ROOT/'outputs/official_system_e4_20261016/per_query.npz');base=js(ROOT/'outputs/official_system_e4_20261016/summary.json');control=npz(CONTROLS/'formal/real.npz');groups=e4['connectivities'];ids=e4['identifiers']
 assert np.array_equal(control['identifiers'],ids) and np.array_equal(control['groups'],groups)
 rows={name:e4[name] for name in base['metrics']};rows['top20_support_fallback']=control['values_selected']
 assert np.array_equal(rows['top20_detail'],control['values_original_d3'])
 expected=npz(ROOT/'outputs/lightgbm_top20_support_fallback_20261017/per_query.npz')
 assert np.array_equal(control['ranks_selected'],expected['ranks_routed_d3'])
 stats={name:summarize(v,groups) for name,v in rows.items()}
 for name in base['metrics']:
  for weight in ('query_weighted','connectivity_balanced'):
   np.testing.assert_allclose(list(stats[name][weight].values()),list(base['metrics'][name][weight].values()),atol=1e-6,rtol=0)
 sccf_contrib=[];sccf_seeds=[]
 for seed in SEEDS:
  rec=npz(S1/f'seed{seed}/blind_controls/constant_spectrum_full_val.per_query.npz')
  assert np.array_equal(rec['recipient_rows'],np.arange(len(ids)))
  values=uniform_tie_metrics(rec['baseline_greater'],rec['baseline_equal'])
  sccf_contrib.append(values);sccf_seeds.append(summarize(values,groups))
 assert np.array_equal(np.mean(np.stack(sccf_contrib),0),rows['sccf_seed_mean'])
 stats['sccf_seed_mean']['per_seed']=dict(zip(map(str,SEEDS),sccf_seeds))
 stats['sccf_seed_mean']['sample_SD_query_pp']=dict(zip(('recall@1','recall@5','recall@20','mrr'),np.std(np.array([list(v['query_weighted'].values()) for v in sccf_seeds]),axis=0,ddof=1).tolist()))
 for name in ['msalign','embcos']:
  parent=E1 if name=='msalign' else E2;contrib=[];seed_metrics=[]
  for seed in SEEDS:
   rec=npz(parent/f'seed{seed}/val_ranks.npz');assert np.array_equal(ids,rec['identifiers']) and np.array_equal(groups,rec['connectivities'])
   np.testing.assert_array_equal(uniform_tie_metrics(rec['greater'],rec['equal']),rec['expected_metrics'])
   contrib.append(rec['expected_metrics']);seed_metrics.append(summarize(rec['expected_metrics'],groups))
  assert np.array_equal(np.mean(np.stack(contrib),0),rows[name+'_seed_mean'])
  stats[name+'_seed_mean']['per_seed']=dict(zip(map(str,SEEDS),seed_metrics));stats[name+'_seed_mean']['sample_SD_query_pp']=dict(zip(('recall@1','recall@5','recall@20','mrr'),np.std(np.array([list(v['query_weighted'].values()) for v in seed_metrics]),axis=0,ddof=1).tolist()))
 comps={}
 replay=js(ROOT/'outputs/lightgbm_top20_support_fallback_20261017/summary.json')
 for item in plan['primary_contrasts']:
  a,b=item.split(' minus ')
  if a=='top20_support_fallback' and b in ('top20_detail','top5_detail'):
   key='routed_vs_original_d3' if b=='top20_detail' else 'routed_vs_d2';saved=replay['comparisons'][key];boot=saved['paired_bootstrap']
   comp=dict(delta_query_pp=saved['delta_query_weighted_pp'],connectivity_cluster_query_weighted=boot['connectivity_cluster_query_weighted'],connectivity_balanced=boot['connectivity_balanced'],resamples=boot['resamples'],seed=boot['seed'],reused_from='outputs/lightgbm_top20_support_fallback_20261017/summary.json#'+key)
  elif b=='sccf_seed_mean' and a in ('msalign_seed_mean','embcos_seed_mean'):
   parent=E1 if a=='msalign_seed_mean' else E2;saved=js(parent/'analysis/summary.json')['paired_minus_matched_sccf'];native=saved['metrics']
   def cells(field):return {k:dict(delta_pp=v['query_point_pp'] if field=='connectivity_cluster_query_ci95_pp' else v['connectivity_balanced_point_pp'],ci95_low_pp=v[field][0],ci95_high_pp=v[field][1]) for k,v in native.items()}
   comp=dict(delta_query_pp={k:v['query_point_pp'] for k,v in native.items()},connectivity_cluster_query_weighted=cells('connectivity_cluster_query_ci95_pp'),connectivity_balanced=cells('connectivity_balanced_ci95_pp'),resamples=saved['replicates'],seed=saved['seed'],reused_from=rel(parent/'analysis/summary.json')+'#paired_minus_matched_sccf')
  else:comp=cluster_delta(rows[b],rows[a],groups)
  comps[a+'_vs_'+b]=comp
  if a=='top20_support_fallback' and b in ('top20_detail','top5_detail','sccf_ensemble'):
   b1=(rows[b][:,0]==1);a1=(rows[a][:,0]==1);resc=int((~b1&a1).sum());damage=int((b1&~a1).sum());gap=float((rows[b][:,2]-rows[b][:,0]).mean())
   comps[a+'_vs_'+b]['rank1_transition']=dict(rescues=resc,damages=damage,baseline_correct=int(b1.sum()),damage_rate_pct=float(100*damage/b1.sum()),gap_closure_pct=float(100*(rows[a][:,0]-rows[b][:,0]).mean()/gap))
 write(out/'validation_results.json',dict(status='PASS',metrics=stats,contrasts=comps,official_test_read=False,identity='exact matched19429 queries/3184 connectivities',uncertainty='Reuse existing paired intervals unchanged with their recorded seeds; only missing contrasts use10k whole-connectivity draws seed20261017. Prospective test plan remains fixed20261017. Validation development-exposed; no training/selection/multiplicity adjustment'))
 np.savez_compressed(out/'validation_per_query.npz',identifiers=ids,connectivities=groups,**rows)
 with (out/'result_rows.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['row','main_table','supplement','test','weighting','R1','R5','R20','MRR_times100'])
  for row in plan['rows']:
   for weight in ('query_weighted','connectivity_balanced'):w.writerow([row['row'],row['main_table'],True,True,weight,*stats[row['row']][weight].values()])
 return stats,comps


def control_output_audit(out):
 pool=npz(S2/'stage1_candidate_scores/val_three_seed_logits.npz')
 donor=npz(CONTROLS.parent/'stage1_submission_0926/runs/full_r2_block_l2/seed20260621/blind_controls/spectrum_permutation_exact_mass.per_query.npz')
 means=npz(CONTROLS/'train_means.npz');assert len(means['peaks'])==100 and int(means['train_rows'])==194119
 assert np.all(np.diff(means['peaks'][:,0])>0) and np.isclose(means['peaks'][:,1].max(),1)
 np.testing.assert_allclose(means['peaks'][:,0]*100,np.round(means['peaks'][:,0]*100),atol=1e-8,rtol=0)
 result={}
 for name in ('real','zero','train_mean','wrong_seed20260820','wrong_seed20260821','wrong_seed20260822'):
  z=npz(CONTROLS/'formal'/(name+'.npz'));q=z['queries'];assert len(q)==(2155 if name.startswith('wrong') else 19429)
  assert np.array_equal(z['groups'],pool['connectivities'][q]) and np.array_equal(z['identifiers'],pool['identifiers'][q])
  route=z['support5']&~z['support20'];assert np.array_equal(route,z['route5']) and not np.any(z['support20']&~z['support5'])
  if name=='zero':assert not z['support20'].any() and not z['support5'].any() and np.all(z['coverage20']==0)
  if name.startswith('wrong'):
   seed=int(name.removeprefix('wrong_seed'));assert np.array_equal(q,donor['recipient_rows']) and np.array_equal(z['source_rows'],donor[f'seed{seed}_donor_rows'])
   assert set(q)==set(z['source_rows']) and len(set(z['source_rows']))==len(q) and np.all(z['groups']!=pool['connectivities'][z['source_rows']])
  for i,query in enumerate(q):
   a,b=pool['indptr'][query:query+2];full=pool['candidate_rows'][a:b];c=z['candidate_rows'][i];assert len(set(c))==20 and set(c).issubset(set(full))
   assert set(z['routed_order'][i])==set(c)
   if route[i]:assert np.array_equal(z['routed_order'][i,5:],c[5:]) and np.isnan(z['selected_scores'][i,5:]).all()
   else:assert np.array_equal(z['selected_scores'][i],z['detail20'][i])
   hit=np.flatnonzero(c==full[0]);label=int(hit[0]) if len(hit) else -1;assert label==z['gt_local'][i]
   if label>=0:
    assert z['ranks_selected'][i]==np.flatnonzero(z['routed_order'][i]==full[0])[0]+1
    k=5 if route[i] else 20
    if label<k:
     scores=z['selected_scores'][i,:k];value=scores[label];greater=int((scores>value).sum());equal=int((scores==value).sum());rr=np.arange(greater+1,greater+equal+1,dtype=float)
     expected=np.array([(rr<=1).mean(),(rr<=5).mean(),(rr<=20).mean(),np.mean(1/rr)])
    else:expected=z['values_stage1'][i]
   else:
    assert z['ranks_selected'][i]==z['ranks_stage1'][i] and z['ranks_selected'][i]>20
    expected=z['values_stage1'][i]
   np.testing.assert_array_equal(expected,z['values_selected'][i])
  assert np.array_equal(z['ranks_selected']<=20,z['ranks_stage1']<=20)
  result[name]=dict(status='PASS',queries=len(q),fallback_queries=int(route.sum()),fixed_recipient_pool=True,GT_slot_and_uniform_expectation_exact=True,branch_tail_and_R20_exact=True)
 write(out/'independent_control_artifact_audit.json',dict(status='PASS',modes=result,official_test_read=False,mean_reference_top100_grid_valid=True))
 return result


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT);a=ap.parse_args();out=a.out.resolve();assert out==OUT and not (out/'readiness_decision.json').exists()
 start=time.time();c=collect_inputs(out);identity=identity_audit();write(out/'identity_audit.json',identity)
 control_output_audit(out);stats,comps=analyze_rows(out);controls=js(CONTROLS/'formal/summary.json')
 inventory=dict(status='COMPLETE',exact_random='analytic expectation on every fixed recipient pool; complete-control summary',train_only_candidate_adversary='existing official_candidate_only_20260929 audit, Stage1 candidate-prior stress test; not a complete D3 adversary',
 sccf='all3 frozen checkpoints: zero/train-mean/strict wrong; existing blind controls reused',ensemble='fresh full-system controls emit Stage1 ensemble ranks in every mode',original_top5_pair='existing formal_v3 complete controls; frozen comparator',D2='existing new-feature interventions keep real pair/coverage; component attribution only; selected routed system controls recompute D2 whenever invoked',original_D3='fresh complete controls plus original feature-only controls; scopes kept separate',selected_support_fallback='fresh complete controls recompute both input channels, shortlist/support/routing; no real-mask inheritance',E1='all3 selected checkpoints existing controls; zero/mean originally retain metadata; additional fully-fixed metadata mean/candidate-only stress available',E2='all3 selected checkpoints existing zero/mean/strict wrong controls, raw-spectrum input',official_test_read=False)
 write(out/'control_inventory.json',inventory)
 claims=[dict(claim='selected system is spectrum-sensitive on the strict matched slice',status='conditional on real-minus-wrong cluster lower bounds',artifact=rel(CONTROLS/'formal/summary.json'),scope='same eligible recipients, not whole-val extrapolation'),dict(claim='support fallback improves R1/MRR with R5 cost',status='supported exposed-validation result, no independent confirmation',artifact=rel(out/'validation_results.json'),scope='compared with originalD3; original gate failure retained'),dict(claim='MSAlign exceeds SCCF single-model mean',status='supported matched whole-method comparison',artifact=rel(E1/'analysis/summary.json'),scope='different readout/metadata/molecule/budget, no backbone isolation'),dict(claim='detail evidence improves scalar/coverage controls',status='supported bounded component result',artifact=rel(D3/'formal/report/summary.json'),scope='not matched-coverage fine chemistry or fragment probability'),dict(claim='clean/no-shortcut benchmark, efficiency or learned chemistry explanation',status='NOT_ESTABLISHED',artifact='docs/msalign/OFFICIAL_SYSTEM_REVIEW_E4_20261016.md',scope='candidate priors, source and resource limitations retained')]
 write(out/'claim_artifact_map.json',claims)
 import torch,lightgbm,h5py
 write(out/'environment.json',dict(python=sys.version,platform=platform.platform(),source_git_head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip(),worktree_status=subprocess.run(['git','status','--short'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.splitlines(),source_authority='Current per-file digests and source snapshots, including uncommitted changes; Git HEAD alone is not the frozen implementation.',torch=torch.__version__,cuda=torch.version.cuda,lightgbm=lightgbm.__version__,rdkit=rdBase.rdkitVersion,numpy=np.__version__,h5py=h5py.__version__,feature_cache_generation_rdkit='2025.09.6',control_fragment_runtime_rdkit='2023.09.6',OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),workers=0))
 for p in [Path(__file__),ROOT/'tools/reranking/audit_official_d3_full_system.py',ROOT/'tools/stage1/run_msalign_official.py',ROOT/'tools/stage1/run_embcos_official.py',ROOT/'tools/analysis/analyze_msalign_official.py',ROOT/'tools/analysis/analyze_embcos_official.py',ROOT/'glmr/embcos.py',ROOT/'glmr/msalign_external.py',ROOT/'glmr/candidate_indices.py',ROOT/'glmr/sparse_features.py',ROOT/'glmr/subformula.py',ROOT/'tools/stage1/export_stage1_candidate_scores.py',ROOT/'tools/reranking/train_official_mpfir_oof_retriever.py',ROOT/'tools/reranking/train_official_magma_forward.py',ROOT/'docs/msalign/configs/external_matched_e0_20261014.json']:
  bind(p,role='frozen_evaluation_source');dest=out/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 for p in [out/'endpoint_policy.json',out/'result_rows_plan.json']:bind(p,role='pre_outcome_decision')
 public=js(ROOT/'reproducibility/sccf_stage1/public_repo/ARTIFACT_INVENTORY.all.json')
 reference_matches=0
 for entry in public['files']:
  key=entry.get('source_path');expected=entry.get('source_sha256')
  if key in READS and expected:
   assert READS[key]['sha256']==expected,key
   reference_matches+=1
 write(out/'stage1_public_inventory_crosscheck.json',dict(status='PASS',matched_files=reference_matches))
 write(out/'artifact_inventory.json',dict(status='VERIFIED_CURRENT_BYTES',files=READS,total_logical_bytes=sum(x['bytes'] for x in READS.values()),no_big_cache_duplication=True))
 lower=[]
 for mode,report in controls['modes'].items():
  if mode=='real':continue
  ci=report['real_minus_control']['selected']['connectivity_cluster_query_weighted'];lower.append(ci['recall@1']['ci95_low_pp']>0 and ci['mrr']['ci95_low_pp']>0)
 decision=dict(status='FROZEN_READY_FOR_SEPARATE_S7_TRANSACTION',date='2026-10-17 Asia/Shanghai',endpoint='top20-first with complete D2 support fallback',original_D3_retained=True,official_test_read=False,official_test_execution_authorized_this_turn=False,model_fitting=False,method_and_analysis_frozen=True,control_completion='COMPLETE, not a clean-claim pass',spectrum_sensitive_all_declared_controls=all(lower),identity_and_feature_order='PASS',validation_real_replay='exact frozen ranks and pair/tree scores; fresh Stage1 logit rounding bounded5e-5 with frozen real-score convention',rows_and_aggregation='frozen result_rows_plan.json',unknown_costs_preserved=['upstream pretraining','comparable end-to-end latency','complete development cost'],publication_release_metadata='license/author/citation/DOI tasks are separate and not resolved by this scientific freeze',remaining_action='S7 separately authorized one-read evaluation using frozen rows, controls and analysis; no further tuning',runtime_seconds=time.time()-start)
 write(out/'readiness_decision.json',decision)
 write(out/'completion_manifest.json',dict(status='COMPLETE',official_test_read=False,files_sha256={str(p.relative_to(out)):sha256_file(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='completion_manifest.json'}))
 print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
