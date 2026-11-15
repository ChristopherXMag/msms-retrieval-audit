#!/usr/bin/env python3
"""Explicit, audited continuation of the interrupted S7 transaction; never a new test."""
from __future__ import annotations
import argparse,ast,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.evaluation.official_s7_common import *
from tools.evaluation.official_s7_resources import advise_clean,reclaim_inputs
from tools.evaluation import run_official_s7 as driver

def check(amendment_path,expected_sha):
    assert sha(amendment_path)==expected_sha,'Amendment changed'
    amendment=read(amendment_path);assert amendment['status']=='VALIDATED_MEMORY_ONLY_REPAIR'
    ready=read(PREP/'ready.json');manifest=read(ROOT/ready['manifest'])
    assert sha(ROOT/ready['manifest'])==ready['manifest_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==ready['commit']
    from importlib.metadata import version
    for package,expected in read(PREP/'environment.json')['packages'].items():assert version(package)==expected
    allowed=amendment['changed_sources']
    assert set(allowed)=={'tools/stage1/export_stage1_candidate_scores.py','tools/evaluation/run_official_s7.py'}
    for path,entry in manifest['files'].items():
        expected=allowed[path]['after_sha256'] if path in allowed else entry['sha256']
        if path in allowed:assert entry['sha256']==allowed[path]['before_sha256']
        assert sha(ROOT/path)==expected,('Frozen input/source changed',path)
        advise_clean(ROOT/path)
    for path,entry in allowed.items():assert sha(ROOT/entry['original_copy'])==entry['before_sha256']
    assert sha(Path(__file__))==amendment['continuation_source_sha256']
    for path,expected in manifest['submodule_heads'].items():
        assert subprocess.check_output(['git','-C',str(ROOT/path),'rev-parse','HEAD'],text=True).strip()==expected
    for path,expected in amendment['validation_files'].items():assert sha(ROOT/path)==expected
    assert read(ROOT/amendment['validation_replay'])['status']=='PASS'
    original=ast.parse((ROOT/allowed['tools/evaluation/run_official_s7.py']['original_copy']).read_text())
    updated=ast.parse((ROOT/'tools/evaluation/run_official_s7.py').read_text())
    old=next(x for x in original.body if isinstance(x,ast.FunctionDef) and x.name=='execute')
    new=next(x for x in updated.body if isinstance(x,ast.FunctionDef) and x.name=='finish_from_stage1')
    start=next(i for i,x in enumerate(old.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='source' for t in x.targets))
    assert ast.dump(ast.Module(body=old.body[start:],type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=new.body[1:],type_ignores=[]),include_attributes=False)
    assert sha(driver.OUT/'transaction.json')==amendment['original_transaction_sha256']
    assert read(driver.OUT/'transaction.json')['status']=='STARTED_ONCE'
    for path,expected in amendment['retained_inputs'].items():assert sha(driver.OUT/path)==expected
    for name in ('stage1.npz','stage1.summary.json','native.started','e1_extension.started','external_e1.started','external_e2.started','predictions_seal.json','completion.json','memory_repair_continuation.started'):
        assert not (driver.OUT/name).exists(),('Not at the audited pre-Stage1 continuation boundary',name)
    return ready,manifest,amendment

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amendment',type=Path,required=True)
    parser.add_argument('--amendment-sha256',required=True)
    parser.add_argument('--continue-existing-transaction',action='store_true')
    args=parser.parse_args();ready,manifest,amendment=check(args.amendment,args.amendment_sha256)
    if not args.continue_existing_transaction:print('MEMORY_REPAIR_CHECKED; TRANSACTION_NOT_CONTINUED',flush=True);return
    reclaim_inputs(manifest,include_test=driver.OUT);gpu_preflight(driver.OUT,'before_memory_repair_continuation')
    driver.worker_gate('memory_repair_continuation')
    write(driver.OUT/'runtime_memory_amendment.json',amendment)
    from tools.reranking import audit_official_d3_full_system as old
    parent=read(old.SOURCE);meta=ROOT/'outputs/msalign_stage1_full_v15/test_full.metadata.jsonl'
    records=old.top.records(meta);rows=load(driver.OUT/'rows.npz');donors=load(driver.OUT/'donors.npz')
    assert rows['identifiers'].tolist()==[q['identifier'] for q in records]
    driver.child(['tools/stage1/export_stage1_candidate_scores.py','--checkpoint',*[str(ROOT/p['path']) for p in parent['checkpoints']],
        '--metadata',str(meta),'--dreams',str(driver.OUT/'no_head.npy'),'--candidate-index-manifest',str(ROOT/parent['candidate_index_manifest']),
        '--fingerprint-cache',str(ROOT/parent['fingerprint_cache']),'--eval-batch-size','64','--tie-seeds',','.join(map(str,TIE_SEEDS)),
        '--out-npz',str(driver.OUT/'stage1.npz'),'--out-json',str(driver.OUT/'stage1.summary.json'),'--device','cuda',
        '--no-preload-sparse-features'],driver.OUT,'stage1_memory_safe')
    driver.finish_from_stage1(ready,manifest,records,rows,donors)

if __name__=='__main__':main()
