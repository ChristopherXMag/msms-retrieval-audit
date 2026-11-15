#!/usr/bin/env python3
"""Reuse frozen D2 wrong-channel controls and stratify rescue/damage.

Post-hoc validation diagnostics only; no fit, search, inference router or test.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from rdkit import Chem, rdBase, RDLogger
from rdkit.Chem import rdMolDescriptors
from tools.reranking import analyze_official_lightgbm_group_ablation as ab
from tools.reranking import run_official_lightgbm_transfer as d2
from tools.analysis.analyze_stage1_logit_ensemble import contributions, grouped
from tools.reranking.audit_official_datasafe_pair_controls import candidate_order
RDLogger.DisableLog('rdApp.*')
DEFAULT = ROOT / 'outputs/lightgbm_strong_endpoint_transfer_20261015'
EPS = 1e-6
SEED = 20261015


def write(p, x):
    stage = p.with_suffix(p.suffix + '.staging')
    stage.write_text(json.dumps(x, indent=2, sort_keys=True, allow_nan=False) + '\n')
    stage.replace(p)


def log(**x):
    print(json.dumps(x), flush=True)


def freeze(parent, out):
    assert not out.exists()
    inputs = [Path(__file__), ROOT/'tools/reranking/analyze_official_lightgbm_group_ablation.py',
              ROOT/'tools/reranking/audit_official_datasafe_pair_controls.py',
              parent/'freeze_manifest.json', parent/'completion_artifact_manifest.json',
              parent/'formal_validation/summary.json', parent/'formal_validation/scores_ranks.npz',
              parent/'group_ablation/freeze_manifest.json', parent/'group_ablation/completion_manifest.json',
              parent/'group_ablation/summary.json', parent/'group_ablation/validation.npz',
              parent/'group_ablation/features_real.npz', d2.D1/'d1_inputs_val/inputs.jsonl']
    inputs += [parent/f'group_ablation/wrong_seed{s}.npz' for s in ab.DONORS]
    out.mkdir(parents=True)
    c = {'date': '2026-10-15 Asia/Shanghai', 'purpose': 'Two user-requested post-hoc mechanism diagnoses; reuse already-completed scalar wrong-channel controls, add rescue/damage strata.',
         'scope': 'Frozen full36 vs scalar10 PRIMARY; frozen full36 vs pair+coverage SECONDARY; all19429 development-exposed official-validation queries, with7053 supported reachable queries for evidence-gap strata.',
         'no_fit_or_selection': True, 'official_test_read': False,
         'challenger': 'Highest pre-reranking comparator score among the other4 candidate ROWS, identity-keyed tie seed20260926; chosen without full-model scores. Do not exclude same-connectivity rows in the primary diagnostic; flag them and provide different-connectivity sensitivity.',
         'strata': {'stage1_GT_rank': ['1','2','3-5','6-20','21-256'],
                    'coverage_gap': 'GT minus fixed challenger raw coverage score: <−.05, [−.05,−1e-6), absolute<=1e-6, (1e-6,.05], >.05. Square-root-intensity-weighted, median-adjusted feasibility score, not absolute probability.',
                    'exclusive_gap': 'Same fixed bins for GT minus challenger exclusive square-root-intensity fraction, uniqueness among all5 candidates.',
                    'ambiguity_gap': 'GT minus challenger ambiguous matched-peak fraction, where ambiguous=1−unambiguous_peak_fraction. Negative/equal/positive with1e-6 tolerance; if either candidate has no matched peak, a separate no_matched_evidence bin.',
                    'population': ['GT_outside_top5','GT_in_top5_unsupported','GT_in_top5_supported']},
         'same_formula_sensitivity': 'Supported/reachable queries; same formula of GT and fixed challenger, crossed with equal coverage(abs gap<=1e-6 and both matched) or equal positive feasible-peak counts; also exclude same-connectivity challengers. Report actual retrieval and GT-vs-fixed-challenger pairwise score win(.5 ties) separately; other candidates are unconstrained.',
         'report': 'Every stratum: counts/connectivities, baseline correctness opportunities, rescues/damages, rescue/damage rates, share of total rescues/damages, contribution to global query-weighted delta, R1/MRR and balanced differences, cluster intervals. Keep full per-query diagnostics and all transition cases.',
         'uncertainty': '10000 paired connectivity-cluster bootstrap; within-stratum connectivity balanced; no CI if<20 connectivities. Exploratory/no multiplicity correction; bins are descriptive, no routing/threshold tuning.',
         'wrong_channel': 'Reuse saved full/scalar real and3 wrong ranks, retain real pair and coverage anchor. Expose real full−scalar, wrong full−scalar and their paired difference; verify existing difference-of-differences artifact exactly.',
         'caveats': ['The scalar control also contains spectrum-derived coverage features; it is not spectrum-blind.',
                     'Labels/GT formula are diagnostic-only. Same-formula/challenger slices do not isolate the entire candidate set.',
                     'Strata are associated with model inputs and baseline difficulty, not randomized causal feature effects.',
                     'All original D2/ablation gates, predictions, non-promotion decisions and test freeze remain unchanged.'],
         'rdkit': rdBase.rdkitVersion,
         'input_sha256': {str(p.relative_to(ROOT)): d2.digest(p) for p in inputs}}
    assert rdBase.rdkitVersion == '2023.09.6'
    write(out/'freeze_manifest.json', c)
    shutil.copyfile(__file__, out/'source.py')
    log(stage='frozen', sha256=d2.digest(out/'freeze_manifest.json'))


def check(out):
    c = json.loads((out/'freeze_manifest.json').read_text())
    for p, h in c['input_sha256'].items():
        assert d2.digest(ROOT/p) == h, p
    return c


def scalar_interval(delta, groups, seed=SEED):
    sums, sizes, means = grouped(np.asarray(delta)[:, None], groups)
    if len(sizes) < 20:
        return {'connectivities': len(sizes), 'reason': 'fewer than20 clusters; no interval'}
    rng = np.random.default_rng(seed); qw = np.empty(10000); cb = np.empty_like(qw)
    for st in range(0, 10000, 128):
        n = min(128, 10000-st); ix = rng.integers(0, len(sizes), size=(n, len(sizes)))
        qw[st:st+n] = sums[ix, 0].sum(1)/sizes[ix].sum(1)
        cb[st:st+n] = means[ix, 0].mean(1)
    return {'query_weighted_ci95_pp': (np.quantile(qw, [.025,.975])*100).tolist(),
            'connectivity_balanced_ci95_pp': (np.quantile(cb,[.025,.975])*100).tolist()}


def signed(values, eligible, wide=True):
    result = np.full(len(values), 'not_applicable', dtype='<U28')
    if wide:
        for name, mask in [('GT_lower_large',values<-.05), ('GT_lower_small',(values>=-.05)&(values<-EPS)),
                           ('equal',np.abs(values)<=EPS),('GT_higher_small',(values>EPS)&(values<=.05)),
                           ('GT_higher_large',values>.05)]:result[eligible&mask] = name
    else:
        result[eligible&(values<-EPS)]='GT_less_ambiguous'
        result[eligible&(np.abs(values)<=EPS)]='equal_ambiguity'
        result[eligible&(values>EPS)]='GT_more_ambiguous'
    assert np.all(result[eligible] != 'not_applicable')
    return result


def slice_summary(before, after, groups, mask, total_rescues, total_damages, population_n):
    n = int(mask.sum())
    if not n:return {'queries': 0, 'connectivities': 0}
    b = before[mask]; a = after[mask]; g = groups[mask]
    rescue = int(np.count_nonzero((b>1)&(a==1))); damage = int(np.count_nonzero((b==1)&(a>1)))
    ng = len(set(g)); delta = contributions(a)-contributions(b)
    result = {'queries': n,'connectivities': ng,'baseline_wrong': int(np.count_nonzero(b>1)),
              'baseline_correct': int(np.count_nonzero(b==1)), 'rescues': rescue,'damages':damage,
              'net_correct':rescue-damage,'rescue_rate_pct':100*rescue/max(int(np.count_nonzero(b>1)),1),
              'damage_rate_pct':100*damage/max(int(np.count_nonzero(b==1)),1),
              'share_all_rescues_pct':100*rescue/max(total_rescues,1),
              'share_all_damages_pct':100*damage/max(total_damages,1),
              'contribution_to_global_R1_pp':100*(rescue-damage)/population_n,
              'baseline': ab.describe(b,g),'after':ab.describe(a,g)}
    if ng>=20:
        result.update(ab.cluster_contrast(delta,g,SEED))
    else:
        result['delta_query_weighted_pp'] = ab.metrics(delta)
        result['delta_connectivity_balanced_pp'] = ab.metrics(grouped(delta,g)[2])
        result['interval_scope'] = 'fewer than20 clusters; no interval'
    return result


def pair_summary(before, after, labels, decoy, groups, mask):
    ix=np.flatnonzero(mask)
    if not len(ix):return {'queries':0}
    gt=labels[ix];dd=decoy[ix]
    def win(x):return (x[ix,gt]>x[ix,dd]).astype(float)+.5*(x[ix,gt]==x[ix,dd])
    b=win(before);a=win(after);delta=a-b
    return {'queries':len(ix),'baseline_win_pct':float(b.mean()*100),'after_win_pct':float(a.mean()*100),
            'delta_query_weighted_pp':float(delta.mean()*100),
            'delta_connectivity_balanced_pp':float(grouped(delta[:,None],groups[ix])[2].mean()*100),
            'paired_interval':scalar_interval(delta,groups[ix]),'metric':'GT-vs-fixed-challenger pairwise score win; .5 exact ties; not retrieval R1'}


def run(parent,out):
    check(out);started=time.time();assert not (out/'summary.json').exists()
    old=dict(np.load(parent/'formal_validation/scores_ranks.npz'));g=parent/'group_ablation'
    real=dict(np.load(g/'validation.npz'));cache=dict(np.load(g/'features_real.npz'))
    x=cache['features'];coverage=cache['coverage'];groups=old['groups'];labels=old['labels'];n=len(groups)
    names=json.loads((g/'freeze_manifest.json').read_text())['feature_names'];col={s:i for i,s in enumerate(names)}
    assert np.array_equal(real['identifiers'],old['identifiers']) and np.array_equal(real['ranks_full'],old['ranks_detail'])
    inputs=d2.records(d2.D1/'d1_inputs_val/inputs.jsonl');assert [r['identifier'] for r in inputs]==old['identifiers'].tolist()
    prior=json.loads((g/'summary.json').read_text());wrong_report={};rec=None
    for seed in ab.DONORS:
        w=dict(np.load(g/f'wrong_seed{seed}.npz'));rr=w['recipient_rows'];rec=rr if rec is None else rec;assert np.array_equal(rec,rr)
        dreal=contributions(real['ranks_full'][rr])-contributions(real['ranks_scalar'][rr])
        dwrong=contributions(w['ranks_full'])-contributions(w['ranks_scalar'])
        contrast=ab.cluster_contrast(dreal-dwrong,groups[rr],seed)
        assert contrast==prior['wrong_channel'][str(seed)]['real_vs_wrong_increment_over_scalar']['full']
        wrong_report[str(seed)]={'queries':len(rr),'connectivities':len(set(groups[rr])),
            'real_full_minus_scalar':ab.cluster_contrast(dreal,groups[rr],SEED),
            'wrong_full_minus_scalar':ab.cluster_contrast(dwrong,groups[rr],seed),
            'real_minus_wrong_increment_over_scalar':contrast,
            'source':str((g/'summary.json').relative_to(ROOT)), 'reused_controls':True}
    write(out/'scalar_wrong_reuse.json',{'status':'COMPLETE_REUSED_AND_RECONCILED','donors':wrong_report,'official_test_read':False})
    log(stage='wrong_controls_reused',seeds=list(wrong_report))
    eligible=old['supported']&(labels>=0);ids=np.flatnonzero(eligible)
    molecular={}
    for i in ids:
        for s in inputs[i]['candidate_smiles']:
            if s not in molecular:
                mol=Chem.MolFromSmiles(s);assert mol is not None
                molecular[s]=(rdMolDescriptors.CalcMolFormula(mol),Chem.MolToInchiKey(mol)[:14])
        assert molecular[inputs[i]['candidate_smiles'][labels[i]]][1]==groups[i]
    rank=old['baseline_ranks'];rank_bins=np.full(n,'21-256',dtype='<U28')
    rank_bins[rank==1]='1';rank_bins[rank==2]='2';rank_bins[(rank>=3)&(rank<=5)]='3-5';rank_bins[(rank>=6)&(rank<=20)]='6-20'
    population=np.full(n,'GT_outside_top5',dtype='<U28');population[(labels>=0)&~old['supported']]='GT_in_top5_unsupported';population[eligible]='GT_in_top5_supported'
    mechanisms={};perquery=[];cases=[]
    for baseline in ('scalar','coverage'):
        bs=real['scores_scalar'] if baseline=='scalar' else old['scores_coverage'];br=real['ranks_scalar'] if baseline=='scalar' else old['ranks_coverage']
        fs=real['scores_full'];fr=real['ranks_full'];rescue=(br>1)&(fr==1);damage=(br==1)&(fr>1)
        decoy=np.full(n,-1,int);cg=np.full(n,np.nan);ug=np.full(n,np.nan);ag=np.full(n,np.nan)
        same=np.zeros(n,bool);same_conn=np.zeros(n,bool);both=np.zeros(n,bool);equal_count=np.zeros(n,bool)
        for i in ids:
            gt=int(labels[i]);order=candidate_order(inputs[i]['identifier'],old['candidate_rows'][i],bs[i],20260926)
            dd=int(next(j for j in order if j!=gt));decoy[i]=dd
            cg[i]=float(coverage[i,gt]-coverage[i,dd]);ug[i]=float(x[i,gt,col['exclusive_intensity_fraction']]-x[i,dd,col['exclusive_intensity_fraction']])
            both[i]=x[i,gt,col['feasible_peaks']]>0 and x[i,dd,col['feasible_peaks']]>0
            if both[i]:ag[i]=float(x[i,dd,col['unambiguous_peak_fraction']]-x[i,gt,col['unambiguous_peak_fraction']])
            equal_count[i]=both[i] and x[i,gt,col['feasible_peaks']]==x[i,dd,col['feasible_peaks']]
            gt_m=molecular[inputs[i]['candidate_smiles'][gt]];dm=molecular[inputs[i]['candidate_smiles'][dd]]
            same[i]=gt_m[0]==dm[0];same_conn[i]=gt_m[1]==dm[1]
        cb=signed(cg,eligible);ub=signed(ug,eligible);amb=signed(ag,eligible&both,False);amb[eligible&~both]='no_matched_evidence'
        axes={'population':population,'stage1_GT_rank':rank_bins,'coverage_gap':cb,'exclusive_gap':ub,'ambiguity_gap':amb}
        report={'overall':slice_summary(br,fr,groups,np.ones(n,bool),int(rescue.sum()),int(damage.sum()),n),
                'axes':{},'same_formula_sensitivity':{}}
        for axis,bins in axes.items():
            keys=(['GT_outside_top5','GT_in_top5_unsupported','GT_in_top5_supported'] if axis=='population' else ['1','2','3-5','6-20','21-256'] if axis=='stage1_GT_rank' else ['GT_lower_large','GT_lower_small','equal','GT_higher_small','GT_higher_large'] if axis!='ambiguity_gap' else ['GT_less_ambiguous','equal_ambiguity','GT_more_ambiguous','no_matched_evidence'])
            report['axes'][axis]={}
            for key in keys:
                mask=bins==key
                report['axes'][axis][key]=slice_summary(br,fr,groups,mask,int(rescue.sum()),int(damage.sum()),n)
            axis_n=sum(v['queries'] for v in report['axes'][axis].values());assert axis_n==(n if axis in ('population','stage1_GT_rank') else int(eligible.sum()))
            assert sum(v.get('rescues',0) for v in report['axes'][axis].values())==int(rescue.sum())
            assert sum(v.get('damages',0) for v in report['axes'][axis].values())==int(damage.sum())
            log(stage='strata',baseline=baseline,axis=axis)
        slices={'supported_reachable':eligible,'both_have_matches':eligible&both,'same_formula':eligible&same,
                'equal_coverage':eligible&both&(np.abs(cg)<=EPS),
                'equal_coverage_same_formula':eligible&both&(np.abs(cg)<=EPS)&same,
                'equal_positive_count_same_formula':eligible&equal_count&same}
        for key in ('same_formula','equal_coverage_same_formula','equal_positive_count_same_formula'):
            slices[key+'_different_connectivity']=slices[key]&~same_conn
        for name,mask in slices.items():
            report['same_formula_sensitivity'][name]={
                'retrieval':slice_summary(br,fr,groups,mask,int(rescue.sum()),int(damage.sum()),n),
                'fixed_challenger_pairwise':pair_summary(bs,fs,labels,decoy,groups,mask)}
        report['same_connectivity_challenger_queries']=int((eligible&same_conn).sum())
        mechanisms[baseline]=report
        for i in range(n):
            row={'baseline':baseline,'identifier':str(old['identifiers'][i]),'connectivity':str(groups[i]),
                 'stage1_gt_rank':int(rank[i]),'before_rank':int(br[i]),'after_rank':int(fr[i]),
                 'supported':bool(old['supported'][i]),'gt_in_top5':bool(labels[i]>=0),
                 'rescue':bool(rescue[i]),'damage':bool(damage[i]),'challenger_position':int(decoy[i]),
                 'coverage_gap':float(cg[i]) if eligible[i] else None,
                 'exclusive_gap':float(ug[i]) if eligible[i] else None,
                 'ambiguity_gap':float(ag[i]) if eligible[i] and both[i] else None,
                 'coverage_bin':str(cb[i]),'exclusive_bin':str(ub[i]),'ambiguity_bin':str(amb[i]),
                 'same_formula':bool(same[i]) if eligible[i] else None,
                 'same_connectivity':bool(same_conn[i]) if eligible[i] else None}
            perquery.append(row)
            if rescue[i] or damage[i]:
                gt=int(labels[i]);opponent=int(candidate_order(inputs[i]['identifier'],old['candidate_rows'][i],bs[i] if rescue[i] else fs[i],20260926)[0]);assert opponent!=gt
                case=dict(row,event='rescue' if rescue[i] else 'damage',gt_smiles=inputs[i]['candidate_smiles'][gt],
                          fixed_challenger_smiles=inputs[i]['candidate_smiles'][decoy[i]],actual_transition_opponent=inputs[i]['candidate_smiles'][opponent],
                          gt_coverage=float(coverage[i,gt]),opponent_coverage=float(coverage[i,opponent]),
                          gt_exclusive=float(x[i,gt,col['exclusive_intensity_fraction']]),
                          opponent_exclusive=float(x[i,opponent,col['exclusive_intensity_fraction']]),
                          gt_unambiguous=float(x[i,gt,col['unambiguous_peak_fraction']]),
                          opponent_unambiguous=float(x[i,opponent,col['unambiguous_peak_fraction']]))
                cases.append(case)
    for name,rows in [('per_query.csv',perquery),('transition_cases.csv',cases)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    check(out)
    write(out/'summary.json',{'status':'COMPLETE_POST_HOC_DIAGNOSTIC','official_test_read':False,'new_training':False,
          'queries':n,'connectivities':len(set(groups)),'eligible_supported_reachable':int(eligible.sum()),
          'wrong_controls_reused':wrong_report,'comparisons':mechanisms,'seconds':time.time()-started,
          'selected_method':None,'routing_policy_changed':False})
    paths=[p for p in out.rglob('*') if p.is_file() and p.name!='completion_manifest.json']
    write(out/'completion_manifest.json',{str(p.relative_to(out)):d2.digest(p) for p in paths})
    log(stage='COMPLETE',seconds=time.time()-started)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--parent',type=Path,default=DEFAULT);p.add_argument('--out',type=Path)
    a=p.parse_args();parent=a.parent.resolve();out=(a.out or parent/'rescue_diagnosis').resolve()
    (freeze if a.stage=='freeze' else run)(parent,out)


if __name__=='__main__':main()
