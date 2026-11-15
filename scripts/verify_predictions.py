#!/usr/bin/env python3
"""Recompute primary metrics from archived contributions without model imports."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from audit_metrics import METRICS,paired_cluster_interval,summarize

INPUTS={
 'validation':('outputs/official_final_freeze_20261017/validation_per_query.npz','outputs/official_final_freeze_20261017/validation_results.json'),
 'test':('outputs/official_s7_test_20261017/per_query.npz','outputs/official_s7_test_20261017/results.json'),
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT)
    p.add_argument('--split',choices=('validation','test','both'),default='both')
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=0,help='Optional paired whole-connectivity draws; 0 computes point estimates only')
    p.add_argument('--seed',type=int,default=20261017)
    args=p.parse_args()
    if args.bootstrap<0:p.error('--bootstrap must be nonnegative')
    if args.out_dir.exists():raise FileExistsError('Choose a fresh output directory')
    results={};table=[]
    for split in (('validation','test') if args.split=='both' else (args.split,)):
        archive,reference=INPUTS[split]
        expected=json.loads((args.data_root/reference).read_text())['metrics' if split=='validation' else 'rows']
        with np.load(args.data_root/archive,allow_pickle=False) as z:
            ids=z['identifiers'];groups=z['connectivities']
            if ids.shape!=groups.shape or len(set(ids.tolist()))!=len(ids):
                raise ValueError('Invalid or duplicated query identities')
            rows=set(z.files)-{'identifiers','connectivities'}
            if rows!=set(expected):raise ValueError('Reference and archive row sets differ')
            computed={}
            for row in sorted(rows):
                actual=summarize(z[row],groups)
                for weight in ('query_weighted','connectivity_balanced'):
                    for metric in METRICS:
                        delta=abs(actual[weight][metric]-float(expected[row][weight][metric]))
                        if delta>1.1e-6:raise ValueError(f'{split}/{row}/{weight}/{metric}: mismatch {delta}')
                    table.append([split,row,weight,*[actual[weight][m] for m in METRICS]])
                computed[row]=actual
            result={'queries':len(ids),'connectivities':len(np.unique(groups)),'rows_verified':len(rows),'rows':computed}
            if args.bootstrap:
                for before in ('sccf_ensemble','top20_detail','msalign_seed_mean'):
                    key='top20_support_fallback minus '+before
                    result.setdefault('independent_paired_intervals',{})[key]=paired_cluster_interval(z['top20_support_fallback'],z[before],groups,args.bootstrap,args.seed)
                result['interval_scope']='Independent seeded reproduction from frozen contributions; original analysis-specific interval receipts remain in the archived summaries.'
            results[split]=result
    args.out_dir.mkdir(parents=True)
    (args.out_dir/'verification.json').write_text(json.dumps({'status':'PASS','models_loaded':False,'new_inference':False,'metric_units':'percent; MRR times 100','splits':results},indent=2)+'\n')
    with (args.out_dir/'metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['split','row','weighting',*METRICS]);writer.writerows(table)
    print(json.dumps({s:{k:v for k,v in r.items() if k in ('queries','connectivities','rows_verified')} for s,r in results.items()},indent=2))


if __name__=='__main__':main()
