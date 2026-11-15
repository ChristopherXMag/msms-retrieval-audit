#!/usr/bin/env python3
"""Explicitly authorized six-row S7 recovery; immutable original test transaction.

Recover scores by observing the unchanged comparator kernel, validate its full
metric contributions, seal new predictions, then compute only uncached MCES pairs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import functools
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.evaluation.official_s7_common import (
    ROOT, PREP, S6, CONTROLS, SEEDS, read, write, load, sha, exact,
    install_no_test_guard, gpu_preflight,
)
from tools.analysis.supplement_official_s7 import guard

OUT = ROOT / 'outputs/official_s7_mces_completion_20261019'
SEALED = ROOT / 'outputs/official_s7_test_20261017'
BASE = ROOT / 'outputs/official_s7_posthoc_20261018'
ROWS = ('top5_pair', 'top5_coverage', 'top5_doubled_coverage',
        'top5_scalar', 'top5_detail', 'top20_scalar')
VALIDATION = PREP / 'fresh_real_replay'
OK = ('OK', 'IDENTICAL')


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def log(event, **values):
    print(json.dumps(dict(event=event, time=time.time(), **values)), flush=True)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def protect_original():
    """Deny write-mode descriptors anywhere inside the completed S7 transaction."""
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve(strict=False)
        if not path.is_relative_to(SEALED.resolve()):
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        assert not (isinstance(mode, str) and any(x in mode for x in 'wax+')), str(path)
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND), str(path)
    sys.addaudithook(audit)


def plan():
    from tools.evaluation.official_s7_system import old
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = read(ROOT / 'docs/reranking/freeze_manifests/s7_pretest_20261017.json')
    amended = read(ROOT / 'docs/reranking/freeze_manifests/s7_memory_runtime_20261018.json')['changed_sources']
    source = read(old.SOURCE)
    pair = read(old.S2 / 'pair_reranker/formal_top5/summary.json')
    paths = [ROOT / p for p in manifest['source_files']]
    paths += [ROOT / p['path'] for p in source['checkpoints']]
    paths += [ROOT / p['path'] for p in pair['checkpoint_files']]
    paths += [old.D3 / f'formal/pair/seed{s}.pt' for s in (20261001, 20261002, 20261003)]
    paths += [parent / f'trees/{arm}_seed{s}.txt' for parent in (old.D2, old.D3)
              for arm in ('scalar', 'detail') for s in old.top.SEEDS]
    paths += [old.D3 / 'features/summary.json', CONTROLS / 'train_means.npz']
    bindings = {}
    for p in dict.fromkeys(paths):
        rel = relative(p)
        assert rel in manifest['files'], ('Absent frozen binding', rel)
        wanted = amended[rel]['after_sha256'] if rel in amended else manifest['files'][rel]['sha256']
        assert sha(p) == wanted, rel
        bindings[rel] = wanted
    code = relative(__file__)
    bindings[code] = sha(__file__)
    for name in ('mces_cache.jsonl', 'mces.json', 'mces_per_query.npz', 'mces_pairs.json'):
        bindings[relative(BASE / name)] = sha(BASE / name)
    write(OUT / 'plan.json', dict(
        status='FROZEN_BEFORE_ADDITIONAL_TEST_SCORING', date='2026-10-19 Asia/Shanghai',
        user_authorization='现在有服务器 做一下',
        scope='Only recover six existing frozen comparator top1 identities and secondary MCES; no fitting, selection or new controls.',
        rows=list(ROWS), bindings=bindings, validation='Full official-validation per-query all10 comparator metrics must be bitwise equal before test access.',
        test='One exclusive additional scoring pass with original comparator windows, native vectors and uniform exact ties; all10 metric arrays must be bitwise equal to original sealed contributions.',
        numeric_kernel='Unmodified System._comparators_block; capture old.pair_score and System.tree outputs without substituting their computation.',
        outputs=['test/scores.npz', 'test/top1_ties.json', 'test/metrics.npz', 'predictions_seal.json',
                 'mces_pairs.json', 'mces_uses.json', 'mces_cache.jsonl', 'mces.json', 'mces_per_query.npz'],
        original_test_read_only=True, no_new_test_endpoint_or_hyperparameters=True,
        mces='Original threshold15/stronger-bound protocol; reuse successful symmetric pair cache, CBC30s then certified HiGHS retries; never impute failures.',
        resource=dict(workers_initial=2, workers_max=16, available_host_and_cgroup_min_gib=32, swap_bytes=0, disk_reserve_gib=12),
    ))
    dest = OUT / 'source' / code
    dest.parent.mkdir(parents=True)
    shutil.copyfile(__file__, dest)
    log('PLAN_FROZEN', bindings=len(bindings), plan_sha256=sha(OUT / 'plan.json'))


def verify():
    p = read(OUT / 'plan.json')
    for path, wanted in p['bindings'].items():
        assert sha(ROOT / path) == wanted, ('Plan binding changed', path)
    for package, wanted in read(PREP / 'environment.json')['packages'].items():
        assert importlib.metadata.version(package) == wanted, package
    guard()
    assert shutil.disk_usage(OUT).free > 12 * 2**30
    return p


def score(phase):
    if phase != 'test':
        install_no_test_guard()
    protect_original()
    verify()
    if phase == 'test':
        assert read(OUT / 'validation/completion.json')['status'] == 'PASS'
        from tools.analysis.supplement_official_s7 import verify as verify_seal
        log('ORIGINAL_SEAL_VERIFIED', **verify_seal())
    from tools.evaluation.official_s7_system import System, old

    class CaptureSystem(System):
        def tree(self, k, arm, features, support, anchor):
            result = super().tree(k, arm, features, support, anchor)
            if hasattr(self, 'captured'):
                if k == 5:
                    self.captured['top5_coverage'] = np.array(anchor, copy=True)
                    self.captured[f'top5_{arm}'] = np.array(result, copy=True)
                elif arm == 'scalar':
                    self.captured['top20_scalar'] = np.array(result, copy=True)
            return result

        def evidence(self, inputs, base, family):
            result = super().evidence(inputs, base, family)
            if family == 5 and hasattr(self, 'captured'):
                self.captured['coverage5'] = np.array(result['coverage'], copy=True)
            return result

        def _comparators_block(self, current, real, output):
            self.captured = {}
            original = old.pair_score
            calls = []
            def capture_pair(*args, **kwargs):
                result = original(*args, **kwargs)
                calls.append(result)
                return result
            old.pair_score = capture_pair
            try:
                super()._comparators_block(current, real, output)
            finally:
                old.pair_score = original
            assert len(calls) == 1
            self.captured['top5_pair'] = np.array(calls[0], copy=True)
            self.captured['top5_doubled_coverage'] = old.top.d2.deployment_anchor(
                calls[0], self.captured['coverage5'], 2.)
            for name in ROWS:
                recovered[name][current] = self.captured[name]
            for name, array in output.items():
                assert exact(array[current], reference[name][current]), ('Metric mismatch', phase, name, current.tolist())
            completed.extend(current.tolist())
            log('SCORE_PROGRESS', phase=phase, queries=len(completed), total=len(queries), resources=guard())

    dest = OUT / phase
    dest.mkdir(exist_ok=False)
    gpu_preflight(dest, phase)
    if phase == 'test':
        source, inputs, projection = SEALED / 'stage1.native.summary.json', SEALED / 'inputs.jsonl', SEALED / 'projection'
        real = load(SEALED / 'real.npz')
        reference = load(SEALED / 'per_query.npz')
    else:
        source, inputs, projection = VALIDATION / 'source.json', VALIDATION / 'inputs.jsonl', CONTROLS / 'projection_formal'
        real = load(VALIDATION / 'real.npz')
        reference = load(S6 / 'validation_per_query.npz')
    assert all((projection / f'seed{s}.npy').exists() for s in SEEDS), 'Do not rebuild projections'
    start = time.time()
    ctx = CaptureSystem(source, inputs, projection, CONTROLS / 'train_means.npz', reference=False)
    queries = np.arange(512 if phase == 'validation_smoke' else ctx.n)
    recovered = {name: np.full((ctx.n, 5 if name.startswith('top5') else 20), np.nan,
                              np.float64 if name in ('top5_scalar', 'top5_detail', 'top20_scalar') else np.float32) for name in ROWS}
    # In the smoke, keep the original n=19429 and first 512-query numerical window.
    output = {name: np.empty((ctx.n, 4)) for name in reference if name.startswith(('top5_', 'top20_')) and name != 'top20_support_fallback'}
    assert len(output) == 10
    completed = []
    for start_q in range(0, len(queries), 512):
        current = queries[start_q:start_q+512]
        assert np.all(ctx.pool['candidate_counts'][current] >= 20)
        ctx._comparators_block(current, real, output)
    assert completed == queries.tolist()
    top1 = {}
    gt = np.array([ctx.pool['candidate_rows'][a] for a in ctx.pool['indptr'][:-1]])
    for name, array in recovered.items():
        array = array[queries]
        assert np.isfinite(array).all()
        candidates = real['candidate_rows'][queries, :array.shape[1]]
        maxima = array == array.max(1, keepdims=True)
        top1[name] = [pool[mask].tolist() for pool, mask in zip(candidates, maxima)]
        contribution = np.array([float(g in choices) / len(choices) for g, choices in zip(gt[queries], top1[name])])
        assert exact(contribution, reference[name][queries, 0]), ('Top1 tie recovery mismatch', name)
    np.savez_compressed(dest / 'scores.npz', identifiers=ctx.ids[queries], candidate_rows=real['candidate_rows'][queries],
                        **{k: v[queries] for k, v in recovered.items()})
    np.savez_compressed(dest / 'metrics.npz', **{k: v[queries] for k, v in output.items()})
    write(dest / 'top1_ties.json', top1)
    files = {p.name: sha(p) for p in dest.iterdir() if p.is_file()}
    write(dest / 'completion.json', dict(status='PASS', phase=phase, queries=len(queries), all10_metric_arrays='BITWISE_EXACT',
        top1_contributions='BITWISE_EXACT', original_kernel_unchanged=True, elapsed_seconds=time.time()-start,
        official_test_scoring=(phase == 'test'), plan_sha256=sha(OUT / 'plan.json'), files=files,
        gpu_peak_allocated_mib=__import__('torch').cuda.max_memory_allocated()/2**20,
        exact_top1_tie_queries={k: sum(len(v)>1 for v in x) for k, x in top1.items()}))
    if phase == 'test':
        write(OUT / 'predictions_seal.json', dict(status='SEALED_BEFORE_MCES_ANALYSIS',
            additional_test_scoring=True, plan_sha256=sha(OUT / 'plan.json'),
            original_s7_seal_sha256=sha(SEALED / 'predictions_seal.json'),
            files={relative(p): sha(p) for p in dest.iterdir() if p.is_file()}))
    log('SCORING_PASS', phase=phase, elapsed_seconds=time.time()-start)


def pairs():
    verify()
    seal = read(OUT / 'predictions_seal.json')
    for path, wanted in seal['files'].items():
        assert sha(ROOT / path) == wanted
    rows = load(SEALED / 'rows.npz')
    assert np.array_equal(load(OUT / 'test/scores.npz')['identifiers'], rows['identifiers'])
    top1 = read(OUT / 'test/top1_ties.json')
    wanted = set(rows['candidates'][:, 0].tolist())
    for values in top1.values():
        wanted.update(v for choices in values for v in choices)
    smiles = {}
    with (ROOT / 'outputs/run_k_full_train/full_sparse/full_smiles.txt').open() as stream:
        for i, line in enumerate(stream):
            if i in wanted:
                smiles[i] = line.strip()
    assert set(smiles) == wanted
    assert all(smiles[int(i)] == str(s) for i, s in zip(rows['candidates'][:, 0], rows['smiles']))
    table, uses = {}, {}
    for name, predictions in top1.items():
        use = []
        for gt, choices in zip(rows['candidates'][:, 0], predictions):
            keys = []
            for pred in choices:
                pair = sorted((smiles[int(gt)], smiles[pred]))
                key = hashlib.sha256(json.dumps(pair).encode()).hexdigest()
                table[key] = pair
                keys.append(key)
            use.append(keys)
        uses[name] = use
    cache = {}
    for line in (BASE / 'mces_cache.jsonl').read_text().splitlines():
        entry = json.loads(line)
        cache[entry['key']] = entry
    assert all(v['status'] in OK for v in cache.values())
    reused = {k: v for k, v in cache.items() if k in table}
    write(OUT / 'mces_pairs.json', table)
    write(OUT / 'mces_uses.json', uses)
    write(OUT / 'reused_pairs.json', reused)
    write(OUT / 'pair_inventory.json', dict(unique_pairs=len(table), reused_pairs=len(reused), new_pairs=len(set(table)-set(reused)),
        base_cache_sha256=sha(BASE / 'mces_cache.jsonl'), predictions_seal_sha256=sha(OUT / 'predictions_seal.json')))
    log('PAIR_INVENTORY', **read(OUT / 'pair_inventory.json'))


def successful_cache():
    result = read(OUT / 'reused_pairs.json')
    attempts = {}
    path = OUT / 'mces_cache.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            value = json.loads(line)
            attempts[value['key']] = value
            if value['status'] in OK:
                if value['key'] in result:
                    assert abs(result[value['key']]['distance']-value['distance']) < 1e-6
                result[value['key']] = value
    return result, attempts


def mces(args):
    verify()
    table = read(OUT / 'mces_pairs.json')
    done, attempts = successful_cache()
    if args.retry_seconds:
        pending = [(k, table[k]) for k, v in attempts.items() if k not in done]
        from tools.analysis.s7_mces_solver import solve, threshold_certificate
        function = functools.partial(solve, seconds=args.retry_seconds)
    else:
        pending = [(k, v) for k, v in sorted(table.items()) if k not in done and k not in attempts]
        from tools.analysis.supplement_official_s7 import mces_pair
        function = mces_pair
    if args.limit:
        pending = pending[:args.limit]
    assert 1 <= args.workers <= 16
    log('MCES_BEGIN', pending=len(pending), workers=args.workers, retry_seconds=args.retry_seconds, resources=guard())
    start = time.time()
    with (OUT / 'mces_cache.jsonl').open('a', buffering=1) as stream, concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        for i, entry in enumerate(pool.map(function, pending, chunksize=1)):
            if args.retry_seconds and entry['status'] not in OK:
                certificate = threshold_certificate(entry, table[entry['key']])
                if certificate:
                    # Preserve the uncertified attempt as well as its accepted certificate.
                    stream.write(json.dumps(clean(entry), allow_nan=False)+'\n')
                    entry = certificate
            entry['retry_budget_seconds'] = args.retry_seconds
            stream.write(json.dumps(clean(entry), allow_nan=False)+'\n')
            if i % 100 == 0 or entry['status'] not in OK:
                log('MCES_PROGRESS', completed=i+1, total=len(pending), status=entry['status'], resources=guard())
    done, attempts = successful_cache()
    status = dict(unique_pairs=len(table), valid_pairs=len(done), pending_pairs=len(set(table)-set(done)),
                  unresolved_keys=[k for k in attempts if k not in done], last_run_seconds=time.time()-start,
                  workers=args.workers, worker_processes_exited=True)
    write(OUT / f'mces_status_{args.workers}w_{args.retry_seconds}s_{args.limit}limit.json', status)
    log('MCES_COMPLETE', **status)


def report():
    verify()
    done, attempts = successful_cache()
    table = read(OUT / 'mces_pairs.json')
    assert set(table) <= set(done), ('Unresolved MCES', len(set(table)-set(done)))
    rows = load(SEALED / 'rows.npz')
    uses = read(OUT / 'mces_uses.json')
    base = load(BASE / 'mces_per_query.npz')
    distance = {k: v for k, v in base.items() if k not in ('identifiers', 'connectivities')}
    assert np.array_equal(base['identifiers'], rows['identifiers'])
    for name in ROWS:
        distance[name] = np.array([np.mean([done[k]['distance'] for k in keys]) for keys in uses[name]])
    from tools.analysis.report_official_s7 import LABELS, mdtable
    _, inverse = np.unique(rows['connectivities'], return_inverse=True)
    counts = np.bincount(inverse)
    reports, rendered = {}, []
    for name, label in LABELS.items():
        d = distance[name]
        assert len(d) == len(rows['identifiers']) and np.isfinite(d).all()
        reports[name] = dict(query_weighted=float(d.mean()), connectivity_balanced=float((np.bincount(inverse, weights=d)/counts).mean()), queries=len(d))
        rendered.append([label, f"{reports[name]['query_weighted']:.4f}", f"{reports[name]['connectivity_balanced']:.4f}"])
    old_report = read(BASE / 'mces.json')
    for name, value in old_report['rows'].items():
        assert reports[name] == value
    np.savez_compressed(OUT / 'mces_per_query.npz', identifiers=rows['identifiers'], connectivities=rows['connectivities'], **distance)
    write(OUT / 'mces.json', dict(status='COMPLETE_ALL_15_ROWS', posthoc_secondary=True, additional_test_scoring=True,
        rows=reports, unchanged_original_rows=list(old_report['rows']), added_rows=list(ROWS),
        pair_inventory=read(OUT / 'pair_inventory.json'), no_query_exclusions=True, no_solver_imputation=True,
        metric=old_report['metric'], predictions_seal_sha256=sha(OUT / 'predictions_seal.json')))
    (OUT / 'mces.md').write_text(mdtable(['Endpoint', 'MCES@1 (lower)', 'Balanced MCES@1'], rendered)+'\n')
    (OUT / 's7_posthoc_mces_rows.tex').write_text('% Post-hoc MCES: nine original sealed-score rows plus six explicitly authorized recovered-score rows.\n' +
        '\n'.join(' & '.join(row)+r' \\' for row in rendered)+'\n')
    from tools.analysis.supplement_official_s7 import verify as verify_seal
    original = verify_seal()
    write(OUT / 'completion.json', dict(status='COMPLETE', original_s7=original, plan_sha256=sha(OUT / 'plan.json'),
        validation=read(OUT / 'validation/completion.json'), additional_test=read(OUT / 'test/completion.json'),
        report_sha256=sha(OUT / 'mces.json'), per_query_sha256=sha(OUT / 'mces_per_query.npz'), all15_rows=True))
    log('ALL15_MCES_COMPLETE', rows=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['plan', 'validation_smoke', 'validation', 'test', 'pairs', 'mces', 'report'])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--retry-seconds', type=int, default=0)
    args = parser.parse_args()
    if args.phase == 'plan':
        plan()
    elif args.phase in ('validation_smoke', 'validation', 'test'):
        score(args.phase)
    elif args.phase == 'mces':
        mces(args)
    else:
        globals()[args.phase]()


if __name__ == '__main__':
    main()
