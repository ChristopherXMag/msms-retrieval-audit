#!/usr/bin/env python3
"""Input-only preparation/negative-identity audit for the one-shot D2 transfer."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from rdkit import Chem,rdBase,RDLogger
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.reranking.train_official_magma_forward import rows,digest,key
from tools.reranking.prepare_official_mpfir_cross_candidate import scan,heavy_element_signature,write_json,write_rows
from glmr.candidate_indices import CandidateIndexStore

SEED=20261015


def prepare(out):
    started=time.time();assert not out.exists();out.mkdir(parents=True)
    meta=ROOT/'outputs/msalign_stage1_full_v15/train_full.metadata.jsonl'
    official=list(rows(meta));assert len(official)==194119 and all(x['fold']=='train' for x in official)
    old=ROOT/'outputs/mpfir_cross_candidate_official_20261010/labels.jsonl'
    exposed={x['connectivity'] for x in rows(old) if x['pilot_partition']!='fit'}
    memo={};groups=[];by_group=defaultdict(list);RDLogger.DisableLog('rdApp.*')
    for i,m in enumerate(official):
        s=m['canonical_isomeric_smiles']
        if s not in memo:
            mol=Chem.MolFromSmiles(s);assert mol is not None
            memo[s]=Chem.MolToInchiKey(mol)[:14]
        g=memo[s];groups.append(g);by_group[g].append(i)
    stored=np.load(ROOT/'outputs/mpfir_cross_candidate_official_20261010/official_train_connectivities.npy')
    assert np.array_equal(stored,np.array(groups)),'RDKit identity drift'
    chosen={g:min(ix,key=lambda i:key(SEED,official[i]['identifier'])) for g,ix in by_group.items()}
    cal=sorted(set(chosen)-exposed,key=lambda g:key(SEED,'cal|'+g))[:1024]
    fit=sorted(set(chosen)-set(cal),key=lambda g:key(SEED,'fit|'+g))[:8000]
    assert len(cal)==1024 and len(fit)==8000 and not(set(cal)&set(fit)) and not(set(cal)&exposed)
    selected=[]
    for part,gg in [('fit',fit),('calibration',cal)]:
        for g in gg:
            i=chosen[g];m=official[i]
            selected.append(dict(m,smiles=m['canonical_isomeric_smiles'],connectivity=g,official_row=i,
                pilot_partition=part,inner_fold=int(key(SEED+1,g)[:8],16)%2))
    write_rows(out/'labels.jsonl',selected);write_rows(out/'query_metadata.jsonl',selected)
    np.save(out/'official_train_connectivities.npy',np.array(groups))
    sets={'endpoint':set(cal)}
    for k in (0,1):sets['mining'+str(k)]=set(cal)|{x['connectivity'] for x in selected if x['pilot_partition']=='fit' and x['inner_fold']==k}
    sigs={heavy_element_signature(x['smiles']) for x in selected};assert None not in sigs
    write_json(out/'scan_targets.json',{'sets':{k:sorted(v) for k,v in sets.items()},'signatures':[list(s) for s in sorted(sigs)]})
    cands=CandidateIndexStore(ROOT/'outputs/run_k_full_train/candidate_index_full_v15_mass_cap256_full_order/manifest.json')
    assert all(cands.get(x['identifier'],256)[0]>=0 for x in selected)
    source=[Path(__file__),meta,old,ROOT/'to_be_run.md',ROOT/'tools/reranking/prepare_official_mpfir_cross_candidate.py',
        cands.manifest_path,ROOT/'outputs/run_k_official_datasafe_seed20260621/train.no_head.summary.json',
        ROOT/'outputs/stage1_submission_0926/assets/heldout_connectivity_exclusion/manifest.json',
        ROOT/'tools/reranking/train_official_lightgbm_pilot.py',ROOT/'tools/reranking/train_official_datasafe_pair_reranker.py']
    write_json(out/'prepare_summary.json',{'status':'INPUTS_FROZEN_BEFORE_FITS','seed':SEED,'rdkit':rdBase.rdkitVersion,
        'official_train_rows':len(official),'official_train_connectivities':len(chosen),'fit':len(fit),'calibration':len(cal),
        'prior_exposed_calibration_evaluation_connectivities':len(exposed),'calibration_exposed_overlap':0,
        'inner_fold_counts':dict(Counter(str(x['inner_fold']) for x in selected if x['pilot_partition']=='fit')),
        'sampling':'one hash-selected spectrum per official-train connectivity; no native MAGMa-label restriction or future-rank/support selection',
        'official_validation_read':False,'official_test_read':False,'elapsed_seconds':time.time()-started,
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in source},
        'output_sha256':{n:digest(out/n) for n in ('labels.jsonl','query_metadata.jsonl','official_train_connectivities.npy','scan_targets.json')}})
    print(json.dumps(json.loads((out/'prepare_summary.json').read_text()),indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','scan'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out=a.out.resolve()
    if a.stage=='prepare':prepare(a.out)
    else:scan(a)

if __name__=='__main__':main()
