#!/usr/bin/env python3
"""Independently verify Figure 5 selections, routed scores, masks and deliverables."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from rdkit import Chem,rdBase
from rdkit.Chem import rdMolDescriptors
from tools.evaluation.official_s7_common import install_no_test_guard,GUARD_EVENTS,read,write,sha,load
from tools.analysis.supplement_official_s7 import guard
from tools.reranking import run_official_lightgbm_top20 as frozen
from tools.reranking import eval_official_mpfir_v2_top5 as matching
from glmr.subformula import ion_shift
from tools.reranking.train_run_k_e1_fragment_residual import enumerate_fragment_masks,fragment_formula_features
from tools.reranking.audit_official_d3_full_system import support
from tools.reranking.audit_official_datasafe_pair_controls import candidate_order


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'outputs/figure5_validation_cases_20261112')
    args=parser.parse_args();out=args.directory.resolve();assert out.is_relative_to(ROOT/'outputs')
    install_no_test_guard();torch.set_num_threads(1);guard();assert rdBase.rdkitVersion=='2023.09.6'
    manifest=read(out/'selection_manifest.json');cases=read(out/'cases.json');small=read(out/'cases_summary.json')
    rendering=read(out/'rendering_manifest.json');eligible=read(out/'eligible_queries.json')['eligible_query_ids']
    rules_hash=hashlib.sha256(json.dumps(manifest['rules'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert rules_hash==manifest['rules_sha256']==cases['selection_rules_sha256']==small['selection_rules_sha256']
    assert manifest['rdkit_version']==cases['rdkit_version']==small['rdkit_version']==rendering['rdkit_version']=='2023.09.6'
    assert sha(out/'eligible_queries.json')==manifest['eligible_queries_sha256']
    rng=np.random.default_rng(20261112)
    for kind in ('A','B'):
        assert eligible[kind]==sorted(set(eligible[kind]))
        assert len(eligible[kind])==manifest['eligible_counts'][kind]
        index=int(rng.integers(len(eligible[kind])))
        assert index==manifest['random_indices'][kind]
        assert eligible[kind][index]==manifest['selected_query_ids'][kind]
    for path,entry in {**manifest['input_hashes'],**manifest['source_hashes']}.items():assert sha(ROOT/path)==entry['sha256'],path
    assert sha(ROOT/manifest['selection_source']['path'])==manifest['selection_source']['sha256']
    for path,digest in manifest['result_hashes'].items():assert sha(out/path)==digest,path
    assert sha(ROOT/rendering['source']['path'])==rendering['source']['sha256']
    real=load(ROOT/'outputs/official_d3_full_system_controls_20261017/formal/real.npz')
    with np.load(ROOT/'outputs/lightgbm_top20_20261015/formal/features/features.npz',allow_pickle=False) as z:
        cached_cov=z['coverage'];cached_support=z['support']
    with np.load(ROOT/'outputs/lightgbm_top20_20261015/formal/report/scores_ranks.npz',allow_pickle=False) as z:
        cached_anchor=z['scores_coverage']
    arrays=load(out/'feasibility_arrays.npz');ev=frozen.Evidence();edge_count=0
    for case,summary in zip(cases['cases'],small['cases']):
        k=case['case'];i=case['official_validation_row'];identifier=case['identifier'];peaks=case['peaks']
        assert identifier==manifest['selected_query_ids'][k]==str(real['identifiers'][i])==summary['identifier']
        assert case['adduct']=='[M+H]+' and case['coverage_shortlist_width']==20
        assert real['support20'][i] and cached_support[i] and not real['route5'][i]
        assert np.array_equal(real['selected_scores'][i],real['detail20'][i])
        full=case['full_top20'];rows=np.asarray([q['global_candidate_row'] for q in full])
        assert np.array_equal(rows,real['candidate_rows'][i])
        base=np.asarray([q['ensemble_score'] for q in full],np.float32)
        assert np.array_equal(base,real['stage1_scores'][i])
        for key,source in [('pair_score',real['pair20']),('final_score',real['selected_scores']),('coverage_score',real['coverage20'])]:
            assert np.array_equal(np.asarray([q[key] for q in full]),source[i]),key
        assert (base==base.max()).sum()==1 and (real['selected_scores'][i]==real['selected_scores'][i].max()).sum()==1
        q=dict(identifier=identifier,adduct=case['adduct'],precursor_mz=case['precursor_mz'],
            candidate_global_rows=rows.tolist(),candidate_smiles=[c['smiles'] for c in full],
            observed_spectrum=[[p['mz'],p['relative_intensity']] for p in peaks])
        assert support(q,q['candidate_smiles'])
        step=ev.build([q],base[None],coverage_override=np.zeros((1,20),np.float32))
        fresh=ev.build([q],base[None],coverage_override=step['computed_coverage'])
        assert np.array_equal(fresh['valid'][0],arrays[k+'_valid'])
        assert np.array_equal(fresh['coverage'][0],arrays[k+'_coverage'])
        assert np.array_equal(fresh['coverage'][0],cached_cov[i])
        anchor=frozen.d2.deployment_anchor(real['pair20'][i:i+1],fresh['coverage'])[0]
        assert np.array_equal(anchor,cached_anchor[i])
        assert np.array_equal(anchor,np.asarray([c['anchor_score'] for c in full],np.float32))
        shown=case['candidates'];gt,decoy=shown
        assert gt['role']=='ground_truth' and decoy['role']=='decoy'
        assert gt['connectivity']!=decoy['connectivity']
        matched=[]
        for c in shown:
            mol=Chem.MolFromSmiles(c['canonical_isomeric_smiles']);mb=Chem.MolFromMolBlock(c['molblock'],removeHs=False)
            assert mol.GetNumHeavyAtoms()==c['heavy_atom_count']<=30
            assert Chem.MolToSmiles(mol)==Chem.MolToSmiles(mb)
            assert Chem.MolToInchiKey(mol)[:14]==c['connectivity']
            assert rdMolDescriptors.CalcMolFormula(mol)==c['formula']
            assert all(a['index']==j and a['symbol']==mol.GetAtomWithIdx(j).GetSymbol() for j,a in enumerate(c['atoms']))
            proposal_list=enumerate_fragment_masks(mol,64)
            assert len(proposal_list)==len(c['proposals'])
            for (atom_set,cuts),proposal in zip(proposal_list,c['proposals']):
                assert list(atom_set)==proposal['atom_indices'] and list(cuts)==proposal['cut_bond_indices']
                mass=fragment_formula_features(mol,atom_set,cuts)[2]
                assert mass==proposal['raw_atom_sum_mass']
                assert float((np.float32(mass)/np.float32(1000))*np.float32(1000))==proposal['frozen_float32_mass']
            pos=c['shortlist_position'];assert rows[pos]==c['global_candidate_row']
            mask=fresh['valid'][0,pos,:len(peaks)].any(-1)
            assert np.array_equal(mask,c['per_peak_feasibility_mask']);matched.append(mask)
            feasible_pairs={(p,f) for p,f in np.argwhere(fresh['valid'][0,pos,:len(peaks)])}
            assert feasible_pairs=={(e['peak_index'],e['proposal_index']) for e in c['feasible_edges']}
            edge_count+=len(feasible_pairs)
            # Independent scalar/vector NumPy arithmetic checks the path/H/ppm trace.
            shifts=np.asarray(matching.H_SHIFTS,np.float32)*np.float32(Chem.GetPeriodicTable().GetMostCommonIsotopeMass(1))
            parent=np.float32(c['parent_exact_mass']);adduct=np.float32(ion_shift(case['adduct']))
            precursor=np.float32(case['precursor_mz'])
            for edge in c['feasible_edges']:
                observed=np.float32(peaks[edge['peak_index']]['mz'])
                mass=np.float32(c['proposals'][edge['proposal_index']]['frozen_float32_mass'])
                direct_prediction=(mass+adduct)+shifts
                loss_prediction=(parent-mass)+shifts
                direct_ppm=(observed-direct_prediction)/np.maximum(np.abs(direct_prediction),np.float32(1e-6))*np.float32(1e6)
                loss_ppm=((precursor-observed)-loss_prediction)/np.maximum(np.abs(loss_prediction),np.float32(1e-6))*np.float32(1e6)
                di=int(np.argmin(np.abs(direct_ppm)));li=int(np.argmin(np.abs(loss_ppm)))
                use_direct=abs(direct_ppm[di])<=abs(loss_ppm[li])
                index=di if use_direct else li
                signed=direct_ppm[index] if use_direct else loss_ppm[index]
                predicted=direct_prediction[index] if use_direct else loss_prediction[index]
                assert edge['route']==('direct' if use_direct else 'neutral_loss')
                assert edge['hydrogen_adjustment']==matching.H_SHIFTS[index]
                assert edge['signed_ppm_error']==float(signed)
                assert edge['predicted_direct_mz_or_neutral_loss_float32']==float(predicted)
            best_peaks=sorted(np.flatnonzero(mask).tolist(),key=lambda p:(-peaks[p]['relative_intensity_float32'],p))[:2]
            assert [h['peak_index'] for h in c['displayed_hypotheses']]==best_peaks
            for h in c['displayed_hypotheses']:
                best=min([e for e in c['feasible_edges'] if e['peak_index']==h['peak_index']],key=lambda e:(e['absolute_ppm_error'],e['proposal_index']))
                for key,value in best.items():assert h[key]==value
                assert h['absolute_ppm_error']<20 and h['hydrogen_adjustment'] in (-2,-1,0,1,2)
                assert h['proposal_atom_indices']==c['proposals'][h['proposal_index']]['atom_indices']
            for name,values in [('ensemble',base),('pair',real['pair20'][i]),('anchor',anchor),('final',real['selected_scores'][i])]:
                order=candidate_order(identifier,rows,values,20260926)
                assert c['ranks'][name]['frozen_ordinal']==int(np.flatnonzero(order==pos)[0])+1
                assert c['scores'][name]==float(values[pos])
        if k=='A':
            assert gt['ranks']['ensemble']['frozen_ordinal']>1 and gt['ranks']['final']['frozen_ordinal']==1
            assert decoy['ranks']['ensemble']['frozen_ordinal']==1
        else:
            assert gt['formula']==decoy['formula'] and decoy['ranks']['final']['frozen_ordinal']==1
            assert gt['ranks']['final']['frozen_ordinal']>1
            assert np.array_equal(matched[0],matched[1]) and matched[0].any()
        for p,peak in enumerate(peaks):
            label='both' if matched[0][p] and matched[1][p] else 'ground_truth_only' if matched[0][p] else 'decoy_only' if matched[1][p] else 'neither'
            assert peak['category']==label
        assert case['route']['cached_GT_stage1_rank']==real['ranks_stage1'][i]
        assert case['route']['cached_GT_final_rank']==real['ranks_selected'][i]
    for filename,digest in rendering['files'].items():assert sha(out/filename)==digest,filename
    pdf=(out/'figure5_validation_cases.pdf').read_bytes()
    assert pdf.startswith(b'%PDF') and b'/Subtype /Image' not in pdf
    assert len(re.findall(rb'/Type\s*/Page\b',pdf))==1
    with Image.open(out/'figure5_validation_cases.png') as image:
        assert list(image.size)==rendering['renderer']['preview_pixels'];image.verify()
    assert len(list((out/'molecules').glob('*_fragment_hypothesis_*.svg')))==sum(len(c['displayed_hypotheses']) for case in cases['cases'] for c in case['candidates'])
    assert not manifest['forbidden_test_file_operations'] and not rendering['forbidden_test_file_operations'] and not GUARD_EVENTS
    assert not small['contains_spectrum_arrays'] and all('peaks' not in c and 'raw_valid_spectrum' not in c for c in small['cases'])
    checks=['pre-screening rule hash and fixed-seed uniform draw reproduced from sorted eligible IDs',
        'all used inputs and frozen matching sources remain hash-identical',
        'actual supported top20 route and stored selected final scores/ranks, not unverified D3 output',
        'selected displayed structures satisfy size/connectivity/formula and rank1 tie rules',
        'fresh complete-top20 feasibility, lower-median coverage and anchor exactly replay frozen arrays',
        'per-peak class masks and all displayed atom/proposal/H/ppm records align with frozen feasible edges',
        'zero-based atom identity, MolBlock and deterministic peak-intensity/proposal tie ordering',
        'SVG file digests, one-page vector PDF with no raster Image XObjects and readable PNG integrity',
        'compact summary excludes spectra; no forbidden test-file operations']
    report=dict(status='PASS',verified_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),checks=checks,
        cases=manifest['selected_query_ids'],eligibility_counts=manifest['eligible_counts'],
        feasible_edges_verified=edge_count,rdkit_version=rdBase.rdkitVersion,
        rules_sha256=rules_hash,selection_execution_source_sha256=manifest['selection_source']['sha256'],
        rendering_source_sha256=rendering['source']['sha256'],verification_source_sha256=sha(Path(__file__)),
        cases_sha256=sha(out/'cases.json'),scalar_summary_sha256=sha(out/'cases_summary.json'),
        rendering_manifest_sha256=sha(out/'rendering_manifest.json'),resources=guard(),forbidden_test_file_operations=GUARD_EVENTS)
    write(out/'verification.json',report)
    manifest.update(status='COMPLETE',verification=dict(path='verification.json',sha256=sha(out/'verification.json')),
        rendering_source=rendering['source'],verification_source=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=sha(Path(__file__))),
        rendering_manifest_sha256=sha(out/'rendering_manifest.json'),caption_sha256=sha(out/'caption.md'),
        completed_at=report['verified_at'])
    write(out/'selection_manifest.json',manifest)
    print('PASS',len(checks),'checks;',edge_count,'feasible edges;',manifest['selected_query_ids'],flush=True)

if __name__=='__main__':main()
