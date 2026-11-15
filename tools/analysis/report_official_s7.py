#!/usr/bin/env python3
"""Verify sealed S7 predictions and render the prespecified reporting rows; no inference."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.evaluation.official_s7_common import read,write,load,sha,S6,SEEDS,DONOR_SEEDS,exact
from tools.evaluation.official_s7_resources import advise_clean
from tools.analysis.freeze_official_system_s6_e5 import summarize
METRICS=('recall@1','recall@5','recall@20','mrr')
LABELS=dict(sccf_seed_mean='SCCF single-model mean',msalign_seed_mean='MSAlign single-model mean',embcos_seed_mean='EmbCos single-model mean',sccf_ensemble='SCCF mean-logit ensemble',top5_pair='Top5 pair',top5_coverage='Top5 pair + coverage',top5_doubled_coverage='Top5 doubled coverage',top5_scalar='Top5 scalar tree',top5_detail='D2 top5 detail',top20_pair='Top20 pair',top20_coverage='Top20 pair + coverage',top20_doubled_coverage='Top20 doubled coverage',top20_scalar='Top20 scalar tree',top20_detail='Original D3 top20 detail',top20_support_fallback='Selected support fallback')

def numbers(summary):return [summary['query_weighted'][k] for k in METRICS]+[summary['connectivity_balanced']['recall@1']]
def mdtable(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,row))+' |' for row in rows]])+'\n'
def texrows(rows):return (' '+chr(92)*2+'\n').join(' & '.join(map(str,row)) for row in rows)+'\n'
def fmt(values):return [f'{x:.4f}' for x in values]
def contrast_cell(item,metric,balanced=False,point=None):
    interval=item['connectivity_balanced' if balanced else 'connectivity_cluster_query_weighted'][metric]
    delta=point if point is not None else interval['delta_pp'] if balanced else item['delta_query_pp'][metric]
    return f"{delta:+.4f} [{interval['ci95_low_pp']:+.4f}, {interval['ci95_high_pp']:+.4f}]"

def report(out):
    completion=read(out/'completion.json');assert completion['status']=='COMPLETE'
    assert sha(out/'results.json')==completion['results_sha256']
    assert sha(out/'predictions_seal.json')==completion['predictions_sha256']
    seal=read(out/'predictions_seal.json');assert seal['status']=='SEALED_BEFORE_ANALYSIS'
    for name,expected in seal['files'].items():
        assert sha(out/name)==expected,('Sealed artifact changed',name);advise_clean(out/name)
    provenance=read(out/'test_source_provenance.json')
    for path,entry in provenance['sources'].items():assert sha(ROOT/path)==entry['sha256'],('Test source changed',path)
    result=read(out/'results.json');data=load(out/'per_query.npz');rows=load(out/'rows.npz');plan=read(S6/'result_rows_plan.json')
    groups=data['connectivities'];n=len(groups);assert exact(data['identifiers'],rows['identifiers']) and exact(groups,rows['connectivities'])
    identity=read(out/'test_split_identity_audit.json');assert identity['splits']['test']['queries']==n and identity['splits']['test']['connectivities']==len(np.unique(groups))
    for path,expected in identity['metadata_sha256'].items():assert sha(ROOT/path)==expected,('Identity-audit source changed',path)
    expected={x['row'] for x in plan['rows']};assert set(result['rows'])==expected==set(data)-{'identifiers','connectivities'}
    for key in expected:
        a=data[key];assert a.shape==(n,4) and np.isfinite(a).all() and np.all((a>=0)&(a<=1))
        assert summarize(a,groups)==result['rows'][key],('Summary differs',key)
    real=load(out/'real.npz');route=real['route5']
    assert np.array_equal(route,(~real['support20']) & real['support5'])
    assert exact(real['values_selected'],data['top20_support_fallback'])
    assert exact(real['values_original_d3'],data['top20_detail'])
    assert exact(data['top20_support_fallback'][route],data['top5_detail'][route])
    assert exact(data['top20_support_fallback'][~route],data['top20_detail'][~route])
    destination=out/'report';assert not destination.exists(),'Preserve the first verified report';destination.mkdir()
    comment='% Derived from sealed outputs/official_s7_test_20261017/results.json; never used for fitting.\n'
    method_rows=[[LABELS[x['row']],*fmt(numbers(result['rows'][x['row']]))] for x in plan['rows']]
    main_rows=[r for r,x in zip(method_rows,plan['rows']) if x['main_table']]
    (destination/'s7_system_rows.tex').write_text(comment+texrows(main_rows))
    (destination/'s7_all_rows.tex').write_text(comment+texrows(method_rows))
    balanced_rows=[[LABELS[x['row']],*fmt([result['rows'][x['row']]['connectivity_balanced'][k] for k in METRICS])] for x in plan['rows']]
    (destination/'s7_all_balanced_rows.tex').write_text(comment+texrows(balanced_rows))
    seeds=[]
    for key,label in [('sccf','SCCF'),('e1','MSAlign'),('e2','EmbCos')]:
        item=result['per_seed'][key]
        for seed in SEEDS:seeds.append([label,str(seed),*fmt(numbers(item['per_seed'][str(seed)]))])
    seed_tex=[];mean_tex=[]
    for key,label in [('sccf','SCCF'),('e1','MSAlign'),('e2','EmbCos')]:
        item=result['per_seed'][key]
        for seed in SEEDS:
            for estimand in ('query_weighted','connectivity_balanced'):
                seed_tex.append([label if estimand=='query_weighted' else '',str(seed) if estimand=='query_weighted' else '', 'Query' if estimand=='query_weighted' else 'Balanced',*fmt([item['per_seed'][str(seed)][estimand][k] for k in METRICS])])
        for estimand,sdkey in [('query_weighted','sample_sd_query_pp'),('connectivity_balanced','sample_sd_connectivity_balanced_pp')]:
            mu=np.mean([[x[estimand][k] for k in METRICS] for x in item['per_seed'].values()],axis=0)
            mean_tex.append([label if estimand=='query_weighted' else '', 'Query' if estimand=='query_weighted' else 'Balanced',*[f'${a:.4f}\\pm {b:.4f}$' for a,b in zip(mu,item[sdkey])]])
    (destination/'s7_seed_rows.tex').write_text(comment+texrows(seed_tex));(destination/'s7_seed_sd_rows.tex').write_text(comment+texrows(mean_tex))
    donors=load(out/'donors.npz');q=donors['recipient_rows'];control_rows=[];control_reports={}
    def add_control(name,values,g):
        summary=summarize(values,g);control_reports[name]=dict(queries=len(g),connectivities=len(np.unique(g)),**summary)
        control_rows.append([name,str(len(g)),*fmt(numbers(summary))])
    add_control('Real, full test',data['top20_support_fallback'],groups)
    for mode,label in [('zero','Zero, full test'),('train_mean','Joint mean, full test')]:
        z=load(out/(mode+'.npz'));assert np.array_equal(z['queries'],np.arange(n));add_control(label,z['values_selected'],groups)
    for key,label in [('candidate_prior','Train-only prior'),('exact_random','Exact random')]:control_rows.append([label,str(n),*fmt(numbers(result['null_controls'][key]))])
    if len(q):
        add_control('Real, strict slice',data['top20_support_fallback'][q],groups[q])
        for seed in DONOR_SEEDS:
            z=load(out/f'wrong_seed{seed}.npz');assert np.array_equal(z['queries'],q);add_control(f'Wrong seed {seed}',z['values_selected'],groups[q])
    control_tex=[]
    for row in control_rows:
        label=row[0];summary=control_reports[label] if label in control_reports else result['null_controls']['candidate_prior' if label=='Train-only prior' else 'exact_random']
        for estimand in ('query_weighted','connectivity_balanced'):
            control_tex.append([label if estimand=='query_weighted' else '',row[1] if estimand=='query_weighted' else '', 'Query' if estimand=='query_weighted' else 'Balanced',*fmt([summary[estimand][k] for k in METRICS])])
    (destination/'s7_control_rows.tex').write_text(comment+texrows(control_tex))
    contrasts=[]
    for key in plan['primary_contrasts']:
        z=result['primary_contrasts'][key];after,before=key.split(' minus ')
        contrasts.append([LABELS[after]+' minus '+LABELS[before],*[contrast_cell(z,k) for k in METRICS],contrast_cell(z,'recall@1',True,result['rows'][after]['connectivity_balanced']['recall@1']-result['rows'][before]['connectivity_balanced']['recall@1'])])
    ci_rows=[]
    names={'top20_support_fallback':'Selected','top20_detail':'Original D3','top5_detail':'D2','sccf_ensemble':'SCCF ensemble','msalign_seed_mean':'MSAlign mean','sccf_seed_mean':'SCCF mean','embcos_seed_mean':'EmbCos mean'}
    for key in plan['primary_contrasts']:
        after,before=key.split(' minus ');z=result['primary_contrasts'][key]
        for balanced in (False,True):
            for j,metric in enumerate(METRICS):
                interval=z['connectivity_balanced' if balanced else 'connectivity_cluster_query_weighted'][metric]
                point=(result['rows'][after]['connectivity_balanced'][metric]-result['rows'][before]['connectivity_balanced'][metric]) if balanced else z['delta_query_pp'][metric]
                ci_rows.append([names[after]+' minus '+names[before] if not balanced and j==0 else '',('Balanced' if balanced else 'Query') if j==0 else '',('R@1','R@5','R@20','MRR')[j],f'{point:+.4f}',f"[{interval['ci95_low_pp']:+.4f}, {interval['ci95_high_pp']:+.4f}]"])
    (destination/'s7_contrast_rows.tex').write_text(comment+texrows(ci_rows))
    control_ci=[];control_contrasts=[]
    mapping=[('zero','Zero, full test'),('train_mean','Joint mean, full test'),*[(f'wrong_seed{s}',f'Wrong seed {s}') for s in DONOR_SEEDS],('candidate_prior','Train-only prior'),('exact_random','Exact random')]
    for key,label in mapping:
        if 'real minus '+key not in result['primary_contrasts']:continue
        z=result['primary_contrasts']['real minus '+key]
        control=control_reports[label] if label in control_reports else result['null_controls'][key]
        real_summary=control_reports['Real, strict slice'] if key.startswith('wrong') else result['rows']['top20_support_fallback']
        control_contrasts.append(['Real minus '+label,*[contrast_cell(z,k) for k in METRICS],contrast_cell(z,'recall@1',True,real_summary['connectivity_balanced']['recall@1']-control['connectivity_balanced']['recall@1'])])
        for balanced in (False,True):
            for j,metric in enumerate(METRICS):
                interval=z['connectivity_balanced' if balanced else 'connectivity_cluster_query_weighted'][metric]
                point=(real_summary['connectivity_balanced'][metric]-control['connectivity_balanced'][metric]) if balanced else z['delta_query_pp'][metric]
                control_ci.append(['Real minus '+label if not balanced and j==0 else '',('Balanced' if balanced else 'Query') if j==0 else '',('R@1','R@5','R@20','MRR')[j],f'{point:+.4f}',f"[{interval['ci95_low_pp']:+.4f}, {interval['ci95_high_pp']:+.4f}]"])
    (destination/'s7_control_contrast_rows.tex').write_text(comment+texrows(control_ci))
    ties=read(out/'tie_sensitivity.json')
    for mode,item in ties.items():
        for seed,z in item['seeds'].items():
            if z['delta_primary_pp'] is not None:assert max(map(abs,z['delta_primary_pp']))<=z['conservative_metric_bound_pp']+1e-10
    a=data['top5_detail'][:,0];transitions={}
    for key in ('top20_detail','top20_support_fallback'):
        b=data[key][:,0];den=a.sum();transitions[key]=dict(baseline='D2 top5 detail',baseline_expected_correct=float(den),damage_expected_count_bounds=[float(np.maximum(a-b,0).sum()),float(np.minimum(a,1-b).sum())],rescue_expected_count_bounds=[float(np.maximum(b-a,0).sum()),float(np.minimum(b,1-a).sum())],damage_rate_pct_bounds=[float(100*np.maximum(a-b,0).sum()/den),float(100*np.minimum(a,1-b).sum()/den)])
    headers=['Endpoint','R@1','R@5','R@20','MRR x100','Mol. R@1']
    text='## Frozen real-input rows\n\nAll values are percentages; method seed means are not ensembles.\n\n'+mdtable(headers,method_rows)
    text+='\n## Connectivity-balanced real-input rows\n\n'+mdtable(['Endpoint',*headers[1:5]],balanced_rows)
    text+='\n## Prespecified primary contrasts\n\nDifferences and paired whole-connectivity 95% intervals, in percentage points.\n\n'+mdtable(['Contrast','Query R@1','Query R@5','Query R@20','Query MRR','Balanced R@1'],contrasts)
    text+='\n## Full-system controls\n\nStrict-slice real and wrong rows use identical recipients; do not compare their absolute levels with full-test real.\n\n'+mdtable(['Input/population','Queries',*headers[1:]],control_rows)
    text+='\n## Prespecified real-minus-control contrasts\n\n'+mdtable(['Contrast','Query R@1','Query R@5','Query R@20','Query MRR','Balanced R@1'],control_contrasts)
    text+='\n## Individual seed rows\n\n'+mdtable(['Method','Seed',*headers[1:]],seeds)
    text+='\n## Support, ties and D2-relative damage\n\nSee machine-owned `system_reports.json` and `tie_sensitivity.json` for every mode.\n\n```json\n'+json.dumps(dict(transitions=transitions,tie_sensitivity=ties),indent=2)+'\n```\n'
    (destination/'tables.md').write_text(text)
    write(destination/'verification.json',dict(status='PASS',routing_invariants_pass=True,sealed_files_verified=len(seal['files']),queries=n,connectivities=len(np.unique(groups)),rows_verified=len(expected),completion_sha256=sha(out/'completion.json'),results_sha256=sha(out/'results.json'),prediction_seal_sha256=sha(out/'predictions_seal.json'),strict_recipient_queries=len(q),strict_connectivities=len(np.unique(groups[q])),controls=control_reports,transitions=transitions,derived_files={p.name:sha(p) for p in destination.iterdir() if p.is_file()}))
    print('SEALED_REPORT_VERIFIED',n,len(expected),destination,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'outputs/official_s7_test_20261017');report(p.parse_args().out.resolve())
if __name__=='__main__':main()
