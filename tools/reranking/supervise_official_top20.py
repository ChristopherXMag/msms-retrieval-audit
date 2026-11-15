#!/usr/bin/env python3
"""Resume complete top20 phases; reclaim only idle read-only SCCF file pages."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.reranking import run_official_lightgbm_top20 as run
from tools.reranking import run_official_lightgbm_transfer as d2
OUT=run.DEFAULT

def state():return d2.memory_state()
def recover(tag):
 before=state();ops=[]
 # No process is using the sparse sources between child phases. Advisory eviction
 # does not change files, array contents, source hashes, or the 32GiB gate.
 paths=sorted((ROOT/'outputs/stage1_submission_0926/assets/full_sparse_direct').glob('*.bin'),key=lambda p:p.stat().st_size,reverse=True)
 for p in paths:
  if state()['cgroup_available_gib']>=40:break
  fd=os.open(p,os.O_RDONLY)
  try:
   for offset in range(0,p.stat().st_size,4*2**30):
    if state()['cgroup_available_gib']>=40:break
    length=min(4*2**30,p.stat().st_size-offset);os.posix_fadvise(fd,offset,length,os.POSIX_FADV_DONTNEED);ops.append(dict(path=str(p.relative_to(ROOT)),offset=offset,length=length))
  finally:os.close(fd)
 after=state()
 with (OUT/'resource_recovery.jsonl').open('a') as f:f.write(json.dumps(dict(stage=tag,before=before,operations=ops,after=after,contents_modified=False,scientific_config_changed=False))+'\n')
 d2.memory_guard()

def phase(stage,family=None):
 tag=stage+('_'+family if family else '');run.check(OUT);recover(tag);cmd=[d2.PYTHON,str(ROOT/'tools/reranking/run_official_top20_compat.py'),stage,'--out',str(OUT)]
 if family:cmd+=['--family',family]
 print(json.dumps(dict(stage=stage,family=family,status='START')),flush=True)
 p=OUT/(tag+'.log')
 if p.exists():
  archive=OUT/'runtime_interruptions';archive.mkdir(exist_ok=True);p.rename(archive/(p.stem+'_'+str(time.time_ns())+'.log'))
 with p.open('w') as f:subprocess.run(cmd,check=True,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 print(json.dumps(dict(stage=stage,family=family,status='COMPLETE')),flush=True)

def main():
 run.check(OUT);run.write(OUT/'supervisor_source.json',{'sha256':d2.digest(Path(__file__)),'no_scientific_changes':True});recover('startup')
 if not (OUT/'plumbing_smoke.json').exists():phase('smoke')
 for family in d2.FAMILIES:
  folder=OUT/'upstream'/family
  if not (folder/'export/summary.json').exists():phase('export',family)
  if not (folder/'pair/summary.json').exists():
   if (folder/'pair').exists():
    assert not list((folder/'pair').iterdir()),'Interrupted nonempty fits require explicit audit'
    (folder/'pair').rmdir()
   phase('pair',family)
 if not (OUT/'features/summary.json').exists():phase('features')
 if not (OUT/'trees/summary.json').exists():phase('trees')
 if not json.loads((OUT/'trees/summary.json').read_text())['continuation_pass']:
  run.write(OUT/'decision.json',dict(status='STOP_AT_TRAIN_CALIBRATION',official_test_read=False,formal_top20_predictions=False))
 else:
  if not (OUT/'formal/export/summary.json').exists():phase('formal-export')
  if not (OUT/'formal/pair/summary.json').exists():phase('pair','formal')
  if not (OUT/'formal/features/summary.json').exists():phase('formal-features')
  if not (OUT/'formal/report/summary.json').exists():phase('formal-report')
  r=json.loads((OUT/'formal/report/summary.json').read_text());run.write(OUT/'decision.json',dict(status=r['status'],attribution=r['attribution_status'],official_test_read=False,promotion='NOT_PROMOTED'))
 recover('independent_audit')
 with (OUT/'independent_audit.log').open('w') as f:subprocess.run([d2.PYTHON,str(ROOT/'tools/analysis/audit_official_top20.py'),'--out',str(OUT)],check=True,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 run.audit_completion(OUT);print(json.dumps(json.loads((OUT/'decision.json').read_text())),flush=True)
if __name__=='__main__':main()
