#!/usr/bin/env python3
"""Final E1 score/rank/tie, policy, replay and selected checkpoint integrity."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from glmr.embcos import sha256_file,write_json,uniform_tie_metrics
OUT=ROOT/'outputs/msalign_official_e1_20261014'


def main():
    result=json.loads((OUT/'analysis/summary.json').read_text());assert result['status']=='COMPLETE' and result['test_evaluated']==False
    source=ROOT/'outputs/embcos_official_e2_20261014/cache';cache=dict(np.load(source/'val.rows.npz'))
    frozen=json.loads((ROOT/'docs/msalign/configs/external_matched_e0_20261014.json').read_text())
    artifacts={};checked=0
    with np.load(ROOT/frozen['assets']['wrong_spectrum_assignments']) as z:
        assignments={key:z[key].copy() for key in ('recipient_rows','seed20260820_donor_rows','seed20260821_donor_rows','seed20260822_donor_rows')}
    for seed in (20260621,20260622,20260623):
        folder=OUT/f'seed{seed}';s=json.loads((folder/'train_summary.json').read_text());c=json.loads((folder/'controls/summary.json').read_text())
        assert s['status']=='TRAINING_COMPLETE' and s['global_step']==s['ceiling']==20000
        assert s['train_rows']==194119 and s['val_rows']==19429 and s['parameters']==13788801
        assert s['source']['runner']==sha256_file(ROOT/'tools/stage1/run_msalign_official.py')
        assert s['source']['module']==sha256_file(ROOT/'glmr/msalign_external.py')
        assert s['checkpoint_sha256']==sha256_file(folder/'checkpoint.pt')==c['checkpoint_sha256']
        assert s['val_ranks_sha256']==sha256_file(folder/'val_ranks.npz')
        assert c['frozen_replay']=='exact scores/counts/ranks passed' and c['status']=='COMPLETE'
        assert s['best']['global_step'] in [1517*i for i in range(1,14)]+[20000]
        with np.load(folder/'val_ranks.npz') as z, np.load(folder/'controls/real.npz') as r:
            for key in ('scores','greater','equal','gt_slots','ranks'):assert np.array_equal(z[key],r[key]),key
        paths=[folder/'val_ranks.npz']+list((folder/'controls').glob('*.npz'))
        for path in paths:
            with np.load(path) as z:
                rows=z['recipient_rows'];scores=z['scores'];length=z['candidate_counts'];slots=z['gt_slots']
                assert np.array_equal(z['identifiers'],cache['identifiers'][rows]) and np.array_equal(z['connectivities'],cache['connectivities'][rows])
                assert np.array_equal(length,cache['lengths'][rows]) and np.array_equal(slots,cache['gt_slots'][rows])
                mask=np.arange(256)[None,:]<length[:,None]
                assert np.isfinite(scores[mask]).all() and np.isneginf(scores[~mask]).all()
                gt=scores[np.arange(len(rows)),slots,None]
                greater=((scores>gt)&mask).sum(1);equal=((scores==gt)&mask).sum(1)
                assert np.array_equal(greater,z['greater']) and np.array_equal(equal,z['equal'])
                assert np.array_equal(uniform_tie_metrics(greater,equal),z['expected_metrics'])
                if path.stem.startswith('wrong_seed'):
                    donor=path.stem.removeprefix('wrong_seed')
                    assert np.array_equal(rows,assignments['recipient_rows']) and np.array_equal(z['spectrum_rows'],assignments[f'seed{donor}_donor_rows'])
                else:assert len(rows)==19429 and np.array_equal(rows,np.arange(19429))
            checked+=1
        for path in [folder/'checkpoint.pt',folder/'checkpoint_last.pt',folder/'train_summary.json',folder/'selected_val.json',folder/'controls/summary.json',*paths]:
            artifacts[str(path.relative_to(ROOT))]={'bytes':path.stat().st_size,'sha256':sha256_file(path)}
    for path in [OUT/'analysis/summary.json',OUT/'analysis/seed_mean_per_query.npz',OUT/'assets/manifest.json',OUT/'frozen_policy_inputs_audit.json',OUT/'runner_semantic_tests.json']:
        artifacts[str(path.relative_to(ROOT))]={'bytes':path.stat().st_size,'sha256':sha256_file(path)}
    write_json(OUT/'result_manifest.json',{'status':'PASS','score_rank_tie_artifacts_verified':checked,'files':artifacts,'test_evaluated':False,
        'scope':'All selected/last checkpoints, exact real replays, candidate slots, finite/masked scores, uniform ties and strict donor rows; CE/adduct recipient-fixed semantics checked in runner tests',
        'audit_script_sha256':sha256_file(Path(__file__))})
    print('PASS:',checked,'score/rank/tie artifacts, exact replays and checkpoint hashes',flush=True)


if __name__=='__main__':main()
