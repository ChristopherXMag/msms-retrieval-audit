#!/usr/bin/env python3
"""Select and trace two frozen official-validation explanatory cases; no models/test."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import h5py
import numpy as np
import torch
from rdkit import Chem,rdBase
from rdkit.Chem import rdDepictor,rdMolDescriptors
from tools.evaluation.official_s7_common import install_no_test_guard,GUARD_EVENTS,sha,read,write,load
from tools.analysis.supplement_official_s7 import guard
from tools.reranking import run_official_lightgbm_top20 as top
from tools.reranking import eval_official_mpfir_v2_top5 as match
from tools.reranking.train_run_k_e1_fragment_residual import enumerate_fragment_masks,fragment_formula_features
from tools.reranking.audit_official_d3_full_system import support
from tools.reranking.replay_official_d3_support_fallback import route_to_d2
from tools.reranking.audit_official_datasafe_pair_controls import candidate_order
from train_spec_encoder.audit_magma_peak_fragment_labels import select_dreams_peaks
from glmr.subformula import ion_shift

OUT=ROOT/'outputs/figure5_validation_cases_20261112'
REAL=ROOT/'outputs/official_d3_full_system_controls_20261017/formal/real.npz'
INPUTS=ROOT/'outputs/lightgbm_top20_20261015/formal/features/inputs.jsonl'
FEATURES=ROOT/'outputs/lightgbm_top20_20261015/formal/features/features.npz'
REPORT=ROOT/'outputs/lightgbm_top20_20261015/formal/report/scores_ranks.npz'
POOL=ROOT/'outputs/stage2_official_datasafe/stage1_candidate_scores/val_three_seed_logits.npz'
META=ROOT/'outputs/msalign_stage1_full_v15/val_full.metadata.jsonl'
H5=ROOT/'outputs/msalign_stage1_full_v15/val_full.hdf5'
SEED=20261112


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def json_lines(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
def relative(path):return str(Path(path).relative_to(ROOT))


@lru_cache(maxsize=8192)
def molecule(smiles):
    mol=Chem.MolFromSmiles(smiles);assert mol is not None
    proposals=enumerate_fragment_masks(mol,64)
    mass=np.zeros(64,np.float32);mask=np.zeros(64,bool);details=[]
    for i,(atoms,cuts) in enumerate(proposals):
        elements,counts,raw_mass,boundary=fragment_formula_features(mol,atoms,cuts)
        mass[i]=(np.float32(raw_mass)/np.float32(1000))*np.float32(1000);mask[i]=True
        composition={Chem.GetPeriodicTable().GetElementSymbol(int(e)):int(c) for e,c in zip(elements,counts) if e and c}
        details.append(dict(proposal_index=i,atom_indices=list(atoms),cut_bond_indices=list(cuts),
            parent_atom_hydrogen_composition=composition,raw_atom_sum_mass=raw_mass,
            frozen_float32_mass=float(mass[i]),boundary_features=boundary.tolist()))
    status,parent=match.candidate_status(smiles);assert status=='supported'
    return dict(mol=mol,heavy_atoms=mol.GetNumHeavyAtoms(),formula=rdMolDescriptors.CalcMolFormula(mol),
        connectivity=Chem.MolToInchiKey(mol)[:14],proposals=details,mass=mass,mask=mask,parent=parent)


def feasibility_for(query,smiles):
    n=len(query['observed_spectrum']);observed=np.asarray(query['observed_spectrum'],np.float32)
    mz=torch.zeros((1,100));present=torch.zeros((1,100),dtype=torch.bool)
    mz[0,:n]=torch.from_numpy(observed[:,0]);present[0,:n]=True
    records=[molecule(s) for s in smiles]
    mass=torch.from_numpy(np.stack([r['mass'] for r in records]))[None]
    masks=torch.from_numpy(np.stack([r['mask'] for r in records]))[None]
    parents=torch.tensor([[r['parent'] for r in records]],dtype=torch.float32)
    precursor=torch.tensor([query['precursor_mz']],dtype=torch.float32)
    adduct=torch.tensor([ion_shift(query['adduct'])],dtype=torch.float32)
    valid,audit=match.feasibility(mz,precursor,adduct,mass,parents,masks,present)
    return valid.numpy()[0],audit


def trace_edges(query,smiles):
    """Expose frozen arithmetic; verify every traced boolean against original function."""
    n=len(query['observed_spectrum']);peaks=torch.tensor(query['observed_spectrum'],dtype=torch.float32)
    records=[molecule(s) for s in smiles]
    mass=torch.from_numpy(np.stack([r['mass'] for r in records]))[None]
    masks=torch.from_numpy(np.stack([r['mask'] for r in records]))[None]
    parent=torch.tensor([[r['parent'] for r in records]],dtype=torch.float32)
    mz=torch.zeros((1,100));mz[0,:n]=peaks[:,0]
    present=torch.zeros((1,100),dtype=torch.bool);present[0,:n]=True
    precursor=torch.tensor([query['precursor_mz']],dtype=torch.float32)
    adduct=torch.tensor([ion_shift(query['adduct'])],dtype=torch.float32)
    shifts=mz.new_tensor(match.H_SHIFTS)*float(Chem.GetPeriodicTable().GetMostCommonIsotopeMass(1))
    predicted_direct=mass[:,:,None,:,None]+adduct[:,None,None,None,None]+shifts
    direct_signed=(mz[:,None,:,None,None]-predicted_direct)/predicted_direct.abs().clamp_min(1e-6)*1e6
    da,di=direct_signed.abs().min(-1);ds=direct_signed.gather(-1,di[...,None]).squeeze(-1)
    observed_loss=precursor[:,None]-mz
    predicted_loss=parent[:,:,None,None,None]-mass[:,:,None,:,None]+shifts
    loss_signed=(observed_loss[:,None,:,None,None]-predicted_loss)/predicted_loss.abs().clamp_min(1e-6)*1e6
    la,li=loss_signed.abs().min(-1);ls=loss_signed.gather(-1,li[...,None]).squeeze(-1)
    direct=da<=la;best_abs=torch.where(direct,da,la)
    signed=torch.where(direct,ds,ls);shift_index=torch.where(direct,di,li)
    traced=(best_abs<match.PPM_TOLERANCE)&masks[:,:,None,:]&present[:,None,:,None]
    valid,audit=match.feasibility(mz,precursor,adduct,mass,parent,masks,present)
    assert torch.equal(traced,valid)
    edges=[]
    for c,record in enumerate(records):
        current=[]
        for p,f in np.argwhere(valid.numpy()[0,c,:n]):
            j=int(shift_index[0,c,p,f]);route='direct' if bool(direct[0,c,p,f]) else 'neutral_loss'
            predicted=float((predicted_direct if route=='direct' else predicted_loss)[0,c,0,f,j])
            edge=dict(peak_index=int(p),proposal_index=int(f),route=route,
                hydrogen_adjustment=int(match.H_SHIFTS[j]),hydrogen_shift_index=j,
                signed_ppm_error=float(signed[0,c,p,f]),absolute_ppm_error=float(best_abs[0,c,p,f]),
                observed_mz_float32=float(mz[0,p]),precursor_mz_float32=float(precursor[0]),
                observed_neutral_loss_float32=float(observed_loss[0,p]),
                predicted_direct_mz_or_neutral_loss_float32=predicted,
                proposal_atom_indices=record['proposals'][int(f)]['atom_indices'],
                complementary_atom_indices=sorted(set(range(record['mol'].GetNumAtoms()))-set(record['proposals'][int(f)]['atom_indices'])))
            current.append(edge)
        edges.append(current)
    return valid.numpy()[0],edges,audit


def verify_sources(manifest):
    assert rdBase.rdkitVersion=='2023.09.6'
    assert hashlib.sha256(json.dumps(manifest['rules'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==manifest['rules_sha256']
    for collection in ('input_hashes','source_hashes'):
        for path,info in manifest[collection].items():assert sha(ROOT/path)==info['sha256'],path
    frozen=read(ROOT/'outputs/official_d3_full_system_controls_20261017/freeze_manifest.json')
    amend=read(ROOT/'outputs/official_d3_full_system_controls_20261017/runtime_amendments.json')
    bindings=dict(frozen['inputs'])
    assert bindings[amend['source_path']]['sha256']==amend['initial_sha256']
    bindings[amend['source_path']]={'sha256':amend['effective_sha256']}
    checked=[]
    for path,info in {**manifest['input_hashes'],**manifest['source_hashes']}.items():
        if path in bindings:
            assert info['sha256']==bindings[path]['sha256'],('frozen binding drift',path)
            checked.append(path)
    assert sha(ROOT/'outputs/official_final_freeze_20261017/endpoint_policy.json')==frozen['endpoint_policy_sha256']
    return checked


def rank_info(identifier,candidates,scores,position):
    ordered=candidate_order(identifier,np.asarray(candidates),np.asarray(scores),20260926)
    value=float(scores[position]);greater=int((scores>scores[position]).sum());equal=int((scores==scores[position]).sum())
    return dict(frozen_ordinal=int(np.flatnonzero(ordered==position)[0])+1,
        score=value,greater=greater,equal=equal,rank_interval=[greater+1,greater+equal])


def extract():
    install_no_test_guard();torch.set_num_threads(1);guard()
    manifest=read(OUT/'selection_manifest.json')
    assert manifest['status']=='RULES_FROZEN_BEFORE_SCREENING'
    assert not (OUT/'cases.json').exists()
    bound=verify_sources(manifest)
    manifest['selection_source']={'path':relative(Path(__file__)),'sha256':sha(Path(__file__))}
    manifest['screening_started_at']=now();write(OUT/'selection_manifest.json',manifest)
    r=load(REAL);queries=json_lines(INPUTS);meta=json_lines(META)
    with np.load(POOL,allow_pickle=False) as z:
        pool={key:z[key] for key in ('identifiers','connectivities','indptr','candidate_rows','ensemble_scores')}
    with np.load(FEATURES,allow_pickle=False) as z:cached={key:z[key] for key in ('coverage','support')}
    report=load(REPORT)
    frozen_values=load(ROOT/'outputs/official_final_freeze_20261017/validation_per_query.npz')
    n=len(meta);assert n==len(queries)==len(r['identifiers'])==19429
    assert np.array_equal(r['queries'],np.arange(n)) and np.array_equal(r['source_rows'],np.arange(n))
    assert np.array_equal(r['identifiers'],pool['identifiers'])
    assert np.array_equal(r['identifiers'],frozen_values['identifiers'])
    assert [q['identifier'] for q in queries]==[q['identifier'] for q in meta]==r['identifiers'].tolist()
    assert all(q['fold'] in ('val','validation') for q in meta)
    assert np.array_equal(r['candidate_rows'],np.asarray([q['candidate_global_rows'] for q in queries]))
    assert np.array_equal(r['candidate_rows'],report['candidate_rows'])
    assert np.array_equal(r['gt_local'],report['labels'])
    assert np.array_equal(r['stage1_scores'],report['scores_stage1'])
    assert np.array_equal(r['pair20'],report['scores_pair'])
    assert np.array_equal(r['coverage20'],cached['coverage'])
    assert np.array_equal(r['detail20'],report['scores_detail'])
    assert np.array_equal(r['support20'],cached['support'])
    assert np.array_equal(r['route5'],route_to_d2(r['support5'],r['support20']))
    assert np.array_equal(r['values_selected'],frozen_values['top20_support_fallback'])
    assert np.array_equal(r['selected_scores'][r['support20']],r['detail20'][r['support20']])
    assert not r['route5'][r['support20']].any()
    eligible={'A':[],'B':[]};positions={};funnel=Counter(official_validation=n)
    checked_shortlists=0;mask_tests=0
    for i,q in enumerate(queries):
        a,b=pool['indptr'][i:i+2];candidates=pool['candidate_rows'][a:b];scores=pool['ensemble_scores'][a:b]
        natural=candidate_order(q['identifier'],candidates,scores,20260926)
        assert np.array_equal(candidates[natural[:20]],r['candidate_rows'][i]),('natural shortlist drift',i)
        assert np.array_equal(scores[natural[:20]],r['stage1_scores'][i])
        gt=int(candidates[0]);local=np.flatnonzero(r['candidate_rows'][i]==gt)
        assert (int(local[0]) if len(local) else -1)==int(r['gt_local'][i])
        stage_rank=int(np.flatnonzero(natural==0)[0])+1;assert stage_rank==r['ranks_stage1'][i]
        selected=r['selected_scores'][i];length=5 if r['route5'][i] else 20
        order=candidate_order(q['identifier'],r['candidate_rows'][i,:length],selected[:length],20260926)
        actual=np.r_[r['candidate_rows'][i,:5][order],r['candidate_rows'][i,5:]] if length==5 else r['candidate_rows'][i,order]
        assert np.array_equal(actual,r['routed_order'][i])
        final_rank=int(np.flatnonzero(actual==gt)[0])+1 if len(local) else stage_rank
        assert final_rank==r['ranks_selected'][i]
        checked_shortlists+=1
        if q['adduct']!='[M+H]+':continue
        funnel['protonated']+=1
        if not r['support20'][i]:continue
        funnel['supported20_branch']+=1
        if not len(local):continue
        funnel['GT_in_natural_top20']+=1
        assert q['candidate_smiles'][int(local[0])]==meta[i]['canonical_isomeric_smiles']
        if (scores==scores.max()).sum()!=1 or (selected==selected.max()).sum()!=1:continue
        funnel['rank1_unique_both_stages']+=1
        gt_local=int(local[0]);winner=int(np.argmax(selected));decoy=0 if winner==gt_local else winner
        if winner==gt_local and int(r['candidate_rows'][i,0])==gt:continue
        kind='A' if winner==gt_local else 'B'
        left=molecule(q['candidate_smiles'][gt_local]);right=molecule(q['candidate_smiles'][decoy])
        if left['heavy_atoms']>30 or right['heavy_atoms']>30 or left['connectivity']==right['connectivity']:continue
        funnel[kind+'_structural_eligible']+=1
        if kind=='B':
            if left['formula']!=right['formula']:continue
            funnel['B_same_formula']+=1
            valid,_=feasibility_for(q,[q['candidate_smiles'][gt_local],q['candidate_smiles'][decoy]])
            masks=valid[:,:len(q['observed_spectrum'])].any(-1);mask_tests+=1
            if not np.array_equal(masks[0],masks[1]) or not masks[0].any():continue
        eligible[kind].append(q['identifier']);positions[q['identifier']]=(i,gt_local,decoy)
        if i%1000==0:print('SCREEN',i,dict(funnel),'eligible', {k:len(v) for k,v in eligible.items()},flush=True)
    for kind in eligible:eligible[kind].sort()
    write(OUT/'eligible_queries.json',dict(seed=SEED,eligible_query_ids=eligible,
        ordering='lexicographic IDs; draw A then B from one PCG64 generator'))
    manifest.update(eligible_counts={k:len(v) for k,v in eligible.items()},eligibility_funnel=dict(funnel),
        selection_checks=dict(identity_aligned_queries=n,natural_shortlists_replayed=checked_shortlists,
            final_routing_and_rank_replayed=n,B_per_peak_masks_recomputed=mask_tests,verified_frozen_bindings=bound),
        eligible_queries_sha256=sha(OUT/'eligible_queries.json'))
    if not all(eligible.values()):
        manifest['status']='STOP_EMPTY_CLASS';write(OUT/'selection_manifest.json',manifest)
        raise RuntimeError('A or B has no eligible query; rules must not be silently relaxed')
    rng=np.random.default_rng(SEED);draws={k:int(rng.integers(len(eligible[k]))) for k in ('A','B')}
    selected={k:eligible[k][draws[k]] for k in ('A','B')}
    manifest.update(status='CASES_SELECTED',selected_query_ids=selected,random_indices=draws,
        numpy_version=np.__version__,rng_type=type(rng.bit_generator).__name__)
    write(OUT/'selection_manifest.json',manifest)
    print('SELECTED',selected,'eligible',manifest['eligible_counts'],flush=True)
    cases=[];arrays={};wanted={};ev=top.Evidence()
    with h5py.File(H5,'r') as h5:
        for kind,identifier in selected.items():
            i,gt,decoy=positions[identifier];q=queries[i];row=r['candidate_rows'][i];base=r['stage1_scores'][i]
            assert support(q,q['candidate_smiles']) and not r['route5'][i] and r['support20'][i]
            raw=h5['spectrum'][i];title=h5['TITLE'][i];title=title.decode() if isinstance(title,bytes) else str(title)
            assert title==identifier
            keep=(raw[0]>0)&np.isfinite(raw).all(0)&(raw[1]>=0)
            peaks=select_dreams_peaks(raw[0,keep].tolist(),raw[1,keep].tolist(),100)
            original=np.asarray([[p['mz'],p['relative_intensity']] for p in peaks],np.float64)
            assert np.array_equal(original,np.asarray(q['observed_spectrum'],np.float64))
            before=ev.build([q],base[None],coverage_override=np.zeros((1,20),np.float32))
            evidence=ev.build([q],base[None],coverage_override=before['computed_coverage'])
            assert evidence['support'][0] and np.array_equal(evidence['coverage'][0],r['coverage20'][i])
            with np.load(FEATURES,allow_pickle=False) as z:
                assert np.array_equal(evidence['features'][0],z['features'][i])
            anchor=top.d2.deployment_anchor(r['pair20'][i:i+1],evidence['coverage'])[0]
            assert np.array_equal(anchor,report['scores_coverage'][i])
            valid,edge_lists,fa=trace_edges(q,[q['candidate_smiles'][gt],q['candidate_smiles'][decoy]])
            assert np.array_equal(valid,evidence['valid'][0,[gt,decoy]])
            per_peak=valid[:,:len(peaks)].any(-1)
            if kind=='B':assert np.array_equal(per_peak[0],per_peak[1]) and per_peak[0].any()
            categories=np.where(per_peak[0]&per_peak[1],'both',np.where(per_peak[0],'ground_truth_only',np.where(per_peak[1],'decoy_only','neither')))
            source_indices=np.flatnonzero(keep)
            peak_records=[dict(peak_index=p,source_filtered_peak_index=int(item['source_peak_index']),
                source_hdf5_peak_index=int(source_indices[int(item['source_peak_index'])]),token_index=int(item['token_index']),
                mz=item['mz'],intensity=item['intensity'],relative_intensity=item['relative_intensity'],
                mz_float32=float(np.float32(item['mz'])),relative_intensity_float32=float(np.float32(item['relative_intensity'])),
                ground_truth_feasible=bool(per_peak[0,p]),decoy_feasible=bool(per_peak[1,p]),category=str(categories[p])) for p,item in enumerate(peaks)]
            display=[]
            for j,(role,pos) in enumerate([('ground_truth',gt),('decoy',decoy)]):
                smi=q['candidate_smiles'][pos];m=molecule(smi);mol=Chem.Mol(m['mol']);rdDepictor.Compute2DCoords(mol)
                hypotheses=[]
                for p in sorted(np.flatnonzero(per_peak[j]).tolist(),key=lambda p:(-float(np.float32(peaks[p]['relative_intensity'])),p))[:2]:
                    edge=min([e for e in edge_lists[j] if e['peak_index']==p],key=lambda e:(e['absolute_ppm_error'],e['proposal_index']))
                    hypotheses.append(dict(edge,illustration_index=len(hypotheses)+1,label='mass-feasible hypothesis'))
                scores={name:float(values[pos]) for name,values in [('ensemble',base),('pair',r['pair20'][i]),('anchor',anchor),('final',r['selected_scores'][i]),('coverage',evidence['coverage'][0])]}
                ranks={name:rank_info(identifier,row,values,pos) for name,values in [('ensemble',base),('pair',r['pair20'][i]),('anchor',anchor),('final',r['selected_scores'][i])]}
                display.append(dict(role=role,shortlist_position=pos,global_candidate_row=int(row[pos]),canonical_isomeric_smiles=smi,
                    formula=m['formula'],connectivity=m['connectivity'],heavy_atom_count=m['heavy_atoms'],
                    parent_exact_mass=m['parent'],parent_exact_mass_float32=float(np.float32(m['parent'])),
                    molblock=Chem.MolToMolBlock(mol),atoms=[dict(index=a.GetIdx(),symbol=a.GetSymbol(),
                        formal_charge=a.GetFormalCharge(),isotope=a.GetIsotope(),parent_total_hydrogens=a.GetTotalNumHs(),
                        xy=list(mol.GetConformer().GetAtomPosition(a.GetIdx()))[:2]) for a in mol.GetAtoms()],
                    bonds=[dict(index=bd.GetIdx(),begin=bd.GetBeginAtomIdx(),end=bd.GetEndAtomIdx(),type=str(bd.GetBondType())) for bd in mol.GetBonds()],
                    proposals=m['proposals'],per_peak_feasibility_mask=per_peak[j].tolist(),feasible_edges=edge_lists[j],
                    displayed_hypotheses=hypotheses,highlighted_parent_union_atom_indices=sorted({a for h in hypotheses for a in h['proposal_atom_indices']}),
                    scores=scores,ranks=ranks))
            full20=[]
            for c,smi in enumerate(q['candidate_smiles']):
                wanted[int(row[c])]=smi
                full20.append(dict(shortlist_position=c,global_candidate_row=int(row[c]),smiles=smi,
                    ensemble_score=float(base[c]),pair_score=float(r['pair20'][i,c]),anchor_score=float(anchor[c]),
                    final_score=float(r['selected_scores'][i,c]),coverage_score=float(evidence['coverage'][0,c]),
                    per_peak_feasibility_mask=evidence['valid'][0,c,:len(peaks)].any(-1).tolist()))
            arrays[kind+'_valid']=evidence['valid'][0];arrays[kind+'_coverage']=evidence['coverage'][0]
            pp=torch.tensor(q['observed_spectrum'],dtype=torch.float32)
            present=torch.zeros((1,100),dtype=torch.bool);present[0,:len(pp)]=True
            intensity=torch.zeros((1,100));intensity[0,:len(pp)]=pp[:,1]
            null=match.masked_null_support(torch.ones(evidence['valid'].shape),torch.from_numpy(evidence['valid']))
            medians=null.median(dim=1).values[0,:len(pp)].tolist()
            case=dict(case=kind,title='System rescue' if kind=='A' else 'Unresolved coverage ambiguity',
                identifier=identifier,official_validation_row=i,adduct=q['adduct'],precursor_mz=q['precursor_mz'],
                collision_energy=meta[i].get('collision_energy'),instrument=meta[i].get('instrument_type'),
                route=dict(support20=True,support5=bool(r['support5'][i]),route_to_D2=False,selected_branch='supported top20 / frozen D3',
                    cached_GT_stage1_rank=int(r['ranks_stage1'][i]),cached_GT_final_rank=int(r['ranks_selected'][i])),
                peaks=peak_records,raw_valid_spectrum=dict(mz=raw[0,keep].tolist(),intensity=raw[1,keep].tolist()),
                candidates=display,full_top20=full20,coverage_shortlist_width=20,
                coverage_lower_median_support_per_peak=medians,
                mask_equality=bool(np.array_equal(per_peak[0],per_peak[1])),matched_peak_counts=per_peak.sum(1).astype(int).tolist(),
                peak_category_counts=dict(Counter(categories.tolist())),frozen_numerical_replay=dict(raw_peaks_exact=True,
                    full20_coverage_bitwise_exact=True,full20_36_features_bitwise_exact=True,anchor_bitwise_exact=True,
                    traced_edge_mask_exact=True),feasibility_audit={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in fa.items()})
            cases.append(case)
    found={}
    with (ROOT/'outputs/run_k_full_train/full_sparse/full_smiles.txt').open() as handle:
        for i,line in enumerate(handle):
            if i in wanted:found[i]=line.rstrip('\n')
    assert found==wanted,'Selected candidate row/SMILES mismatch against global registry'
    payload=dict(schema='figure5_validation_cases_v1',created_at=now(),selection_rules_sha256=manifest['rules_sha256'],
        interpretation='Post hoc validation explanatory examples; not performance or chemical-mechanism evidence',
        rdkit_version=rdBase.rdkitVersion,atom_indexing='zero-based RDKit input-SMILES order, preserved in MolBlock and every atom mask',
        hydrogen_policy='Parent atom H counts only; no boundary recapping; frozen +/-2 H adjustments; neutral-loss highlights retained proposal atoms, not confirmed ions or losses',cases=cases)
    write(OUT/'cases.json',payload);np.savez_compressed(OUT/'feasibility_arrays.npz',**arrays)
    summaries=[]
    for case in cases:
        summaries.append({k:case[k] for k in ['case','title','identifier','adduct','precursor_mz','route','coverage_shortlist_width','mask_equality','matched_peak_counts','peak_category_counts','frozen_numerical_replay']})
        summaries[-1]['candidates']=[{k:c[k] for k in ['role','global_candidate_row','canonical_isomeric_smiles','formula','connectivity','heavy_atom_count','scores','ranks']} for c in case['candidates']]
        summaries[-1]['coverage_attribution']='No component attribution from rescue alone; ensemble/pair/anchor/final scores and ranks retained'
    write(OUT/'cases_summary.json',dict(schema='figure5_scalar_summary_v1',contains_spectrum_arrays=False,
        selection_rules_sha256=manifest['rules_sha256'],rdkit_version=rdBase.rdkitVersion,cases=summaries))
    manifest.update(status='SELECTION_AND_FROZEN_MATCHING_COMPLETE',screening_completed_at=now(),
        selected_candidate_identity_registry_check='all selected two-case top20 global rows/SMILES exactly match original full_smiles registry',
        result_hashes={name:sha(OUT/name) for name in ('cases.json','cases_summary.json','feasibility_arrays.npz')},
        forbidden_test_file_operations=GUARD_EVENTS,resources=guard())
    write(OUT/'selection_manifest.json',manifest)
    print('COMPLETE', {c['case']:dict(id=c['identifier'],coverage=[x['scores']['coverage'] for x in c['candidates']],GT_ranks=[c['route']['cached_GT_stage1_rank'],c['route']['cached_GT_final_rank']],matched=c['matched_peak_counts']) for c in cases},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--extract',action='store_true',required=True)
    parser.parse_args();extract()
