"""Certified HiGHS solver for the unchanged myopic-mces 1.3.2 threshold15 ILP."""
from __future__ import annotations
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/official_s7_posthoc_20261018'

def solve(item,seconds=30):
    key,pair=item;sys.path.insert(0,str(OUT/'vendor/site'))
    from myopic_mces import construct_graph,apply_filter
    from myopic_mces.MCES_ILP import construct_ILP
    import pulp
    started=time.perf_counter()
    if pair[0]==pair[1]:return dict(key=key,distance=0.,mode=0,status='IDENTICAL',seconds=0.,solver='HiGHS')
    try:
        a,b=map(construct_graph,pair);bound,mode=apply_filter(a,b,15,always_stronger_bound=True,use_bound_zero=False)
        if bound>15:return dict(key=key,distance=float(bound),mode=int(mode),status='OK',seconds=time.perf_counter()-started,solver='HiGHS',certificate='official myopic bound')
        problem=construct_ILP(a,b,threshold=15,no_ilp_threshold=False)
        problem.solve(pulp.HiGHS(msg=False,threads=1,timeLimit=seconds,gapRel=0.,gapAbs=0.))
        info=problem.solverModel.getInfo();state=problem.solverModel.getModelStatus().name
        entry=dict(key=key,seconds=time.perf_counter()-started,solver='HiGHS',solver_model_status=state,filter_bound=float(bound),mip_gap=float(info.mip_gap),dual_bound=float(info.mip_dual_bound))
        if state=='kOptimal' and info.mip_gap==0:
            return dict(entry,distance=float(problem.objective.value()),mode=1,status='OK',certificate='optimal with zero mip gap')
        if state=='kInfeasible':return dict(entry,distance=15.,mode=2,status='OK',certificate='threshold15 ILP infeasible')
        return dict(entry,distance=None,mode=6,status='UNRESOLVED')
    except Exception as e:return dict(key=key,seconds=time.perf_counter()-started,status='ERROR',error=repr(e),solver='HiGHS')


def threshold_certificate(entry,pair):
    """A half-integer objective lower bound above14.5 fixes threshold15 output."""
    import math
    bound=entry.get('dual_bound')
    if entry.get('solver')!='HiGHS' or bound is None or not math.isfinite(bound) or bound<=14.5+1e-6:return None
    sys.path.insert(0,str(OUT/'vendor/site'))
    from myopic_mces import construct_graph,apply_filter
    from myopic_mces.MCES_ILP import construct_ILP
    import pulp
    a,b=map(construct_graph,pair);initial,_=apply_filter(a,b,15,always_stronger_bound=True,use_bound_zero=False)
    if initial>15:return None
    problem=construct_ILP(a,b,threshold=15,no_ilp_threshold=False)
    values=[float(problem.objective.constant),*[float(v) for v in problem.objective.values()]]
    assert all(abs(2*v-round(2*v))<1e-9 for v in values),'Objective is not on the half-integer grid'
    assert all(v.cat==pulp.LpInteger for v in problem.variables()),'Nonintegral objective variable'
    # A valid dual bound>14.5 implies every feasible objective>=15. If no
    # <=15 solution exists, official myopic-mces also returns15, as mode2.
    return dict(entry,status='OK',distance=15.,mode=2,certificate='threshold15 output fixed by certified MIP dual bound and half-integer objective grid',objective_grid=.5,objective_variables=len(problem.variables()),threshold=15,original_attempt_status=entry['status'])

def main():
    import argparse,collections,concurrent.futures,functools,json,math
    from tools.analysis.supplement_official_s7 import guard
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seconds',type=int,default=120);parser.add_argument('--workers',type=int,default=2);args=parser.parse_args();assert 1<=args.workers<=2
    guard();pairs=json.loads((OUT/'mces_pairs.json').read_text());attempts={}
    for name in ('mces_cache.jsonl','highs_smoke.jsonl','mces_retry.jsonl'):
        path=OUT/name
        if not path.exists():continue
        for line in path.read_text().splitlines():
            r=json.loads(line);k=r['key'];previous=attempts.get(k)
            if previous is None or previous['status'] not in ('OK','IDENTICAL'):attempts[k]=r
            elif r['status'] in ('OK','IDENTICAL'):assert abs(previous['distance']-r['distance'])<1e-6
    certificates=[]
    for k,entry in attempts.items():
        if entry['status'] not in ('OK','IDENTICAL'):
            result=threshold_certificate(entry,pairs[k])
            if result:certificates.append(result)
    if certificates:
        with (OUT/'mces_retry.jsonl').open('a') as f:
            for result in certificates:
                f.write(json.dumps(result,allow_nan=False)+'\n');attempts[result['key']]=result
    pending=[(k,pairs[k]) for k,v in attempts.items() if v['status'] not in ('OK','IDENTICAL') and v.get('retry_budget_seconds',0)<args.seconds]
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        if isinstance(value,float) and not math.isfinite(value):return None
        return value
    print('CERTIFIED_RETRY_BEGIN',len(pending),args.seconds,flush=True)
    with (OUT/'mces_retry.jsonl').open('a',buffering=1) as f,concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(functools.partial(solve,seconds=args.seconds),pending):
            r['retry_budget_seconds']=args.seconds;f.write(json.dumps(clean(r),allow_nan=False)+'\n');print('RETRY',r['key'],r['status'],r.get('distance'),guard(),flush=True)
    print('CERTIFIED_RETRY_COMPLETE',flush=True)
if __name__=='__main__':main()
