#!/usr/bin/env python3
"""Replay one user-authorized D3-to-D2 support fallback on validation only.

No fitting, threshold search, new features, or official-test input. Route only
on inference-available support masks; retain complete D2 or D3 ordering.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.analysis.analyze_stage1_logit_ensemble import (
    comparison, contributions, grouped, metrics, sha256_file,
)
from tools.reranking.audit_official_datasafe_pair_controls import candidate_order

D2 = ROOT / "outputs/lightgbm_strong_endpoint_transfer_20261015"
D3 = ROOT / "outputs/lightgbm_top20_20261015"
DEFAULT_OUT = ROOT / "outputs/lightgbm_top20_support_fallback_20261017"
BOOTSTRAP_SEED = 20261017
RESAMPLES = 10000
TIE_SEED = 20260926


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def route_to_d2(support5, support20):
    """GT-free rule; unsupported in both retains original D3 behavior."""
    assert support5.dtype == support20.dtype == np.bool_
    assert support5.shape == support20.shape
    return support5 & ~support20


def selected_order(rows5, scores5, rows20, scores20, identifiers, use_d2):
    """Return top20 candidate identities; preserve the D2 tail on fallback."""
    result = np.empty_like(rows20)
    for i, identifier in enumerate(identifiers):
        if use_d2[i]:
            order = candidate_order(identifier, rows5[i], scores5[i], TIE_SEED)
            result[i] = np.r_[rows5[i, order], rows20[i, 5:]]
        else:
            order = candidate_order(identifier, rows20[i], scores20[i], TIE_SEED)
            result[i] = rows20[i, order]
    return result


def replay_ranks(rows, scores, labels, baseline):
    # Exact GT ties are asserted absent; count-greater is then independent of
    # slot and tie seed, and is an independent replay of stored GT ranks.
    result = baseline.copy()
    hit = labels >= 0
    values = scores[hit]
    target = values[np.arange(hit.sum()), labels[hit]]
    ties = (values == target[:, None]).sum(1)
    assert np.all(ties == 1), "Uniform-tie evaluation required for tied GT scores"
    result[hit] = 1 + (values > target[:, None]).sum(1)
    assert np.all(np.isfinite(scores))
    return result


def transitions(before, after):
    rescue = int(((before > 1) & (after == 1)).sum())
    damage = int(((before == 1) & (after > 1)).sum())
    correct = int((before == 1).sum())
    return dict(rescues=rescue, damages=damage, net_correct=rescue - damage,
                baseline_correct=correct,
                damage_rate_pct=100 * damage / correct if correct else None)


def summarize(ranks, groups):
    return dict(query_weighted=metrics(contributions(ranks)),
                connectivity_balanced=metrics(grouped(contributions(ranks), groups)[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(ROOT / "outputs")
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    paths = [
        D2 / "formal_validation/scores_ranks.npz",
        D2 / "formal_validation/summary.json",
        D2 / "completion_artifact_manifest.json",
        D2 / "freeze_manifest.json",
        D3 / "formal/report/scores_ranks.npz",
        D3 / "formal/report/summary.json",
        D3 / "formal/features/inputs.jsonl",
        D3 / "freeze_manifest.json",
        D3 / "completion_manifest.json",
        D3 / "independent_audit.json",
        D3 / "selection_decision.json",
        Path(__file__),
        ROOT / "tools/analysis/analyze_stage1_logit_ensemble.py",
        ROOT / "tools/reranking/audit_official_datasafe_pair_controls.py",
        ROOT / "tools/stage1/export_stage1_candidate_scores.py",
    ]
    inputs = {str(p.relative_to(ROOT)): dict(sha256=sha256_file(p), bytes=p.stat().st_size)
              for p in paths}
    contract = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        date="2026-10-17 Asia/Shanghai", status="FIXED_BEFORE_REPLAY",
        authorization="User: 好的，重做，并报告结果与原D3 对比",
        rule="Use complete D2 iff support5 AND NOT support20; else original D3.",
        scope="Official validation, official mass cap256, frozen data-safe components.",
        intervention="Select one complete candidate order per query; no score interpolation.",
        analysis=dict(primary="new minus original D3", secondary="new minus D2",
                      metrics=["recall@1", "recall@5", "recall@20", "mrr"],
                      bootstrap_resamples=RESAMPLES, bootstrap_seed=BOOTSTRAP_SEED,
                      uncertainty="Paired whole-connectivity bootstrap, query and connectivity balanced; existing helper also emits query bootstrap.",
                      transitions="Report rescue/damage with explicit baseline and original-Stage1 rank strata.",
                      seed_diagnostics="D3 scorer seeds with fixed D2 ensemble fallback; not independent end-to-end seeds."),
        exposure="Rule motivated by prior validation support-stratum outcomes; post-hoc development, no independent confirmation or retroactive D3 gate pass.",
        limitations="Full-system spectrum controls remain pending for both original and routed D3. No automatic final-table/test promotion.",
        official_test_read=False, model_fitting=False, hyperparameter_search=False,
        inputs=inputs,
    )
    write(out / "freeze_manifest.json", contract)
    (out / "source").mkdir()
    for p in paths:
        if p.suffix == ".py":
            dst = out / "source" / p.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(p.read_bytes())
    # Verify both parent result hashes against their completed artifact bindings.
    parent2 = json.loads((D2 / "completion_artifact_manifest.json").read_text())
    parent3 = json.loads((D3 / "completion_manifest.json").read_text())["files_sha256"]
    for parent, manifest, rel in [(D2, parent2, "formal_validation/scores_ranks.npz"),
                                   (D3, parent3, "formal/report/scores_ranks.npz")]:
        assert inputs[str((parent / rel).relative_to(ROOT))]["sha256"] == manifest[rel]
    with np.load(D2 / "formal_validation/scores_ranks.npz", allow_pickle=False) as z:
        d2 = {k: z[k] for k in z.files}
    with np.load(D3 / "formal/report/scores_ranks.npz", allow_pickle=False) as z:
        d3 = {k: z[k] for k in z.files}
    with (D3 / "formal/features/inputs.jsonl").open() as handle:
        ids = np.asarray([json.loads(line)["identifier"] for line in handle])
    groups = d3["groups"]
    n = len(ids)
    assert n == 19429 and len(np.unique(groups)) == 3184
    assert np.array_equal(ids, d2["identifiers"])
    assert np.array_equal(groups, d2["groups"])
    assert np.array_equal(d3["baseline_ranks"], d2["baseline_ranks"])
    assert np.array_equal(d3["candidate_rows"][:, :5], d2["candidate_rows"])
    assert np.array_equal(d3["ranks_top5_detail"], d2["ranks_detail"])
    assert np.all((d2["labels"] >= 0) == (d3["baseline_ranks"] <= 5))
    assert np.all((d3["labels"] >= 0) == (d3["baseline_ranks"] <= 20))
    for data in (d2, d3):
        hit = data["labels"] >= 0
        assert np.all(data["labels"][hit] == data["baseline_ranks"][hit] - 1)
        replay = replay_ranks(data["candidate_rows"], data["scores_detail"],
                              data["labels"], data["baseline_ranks"])
        assert np.array_equal(replay, data["ranks_detail"])
    support5, support20 = d2["supported"], d3["support"]
    fallback = route_to_d2(support5, support20)
    assert not np.any(support20 & ~support5)
    original, old = d3["ranks_detail"], d2["ranks_detail"]
    routed = np.where(fallback, old, original)
    order = selected_order(d2["candidate_rows"], d2["scores_detail"],
                           d3["candidate_rows"], d3["scores_detail"], ids, fallback)
    assert np.array_equal(np.sort(order, axis=1), np.sort(d3["candidate_rows"], axis=1))
    hit = d3["labels"] >= 0
    gt = d3["candidate_rows"][hit, d3["labels"][hit]]
    from_order = d3["baseline_ranks"].copy()
    from_order[hit] = np.argmax(order[hit] == gt[:, None], axis=1) + 1
    assert np.array_equal(from_order, routed)
    # Slot permutation must not change either branch's actual candidate order.
    for data in (d2, d3):
        k = data["candidate_rows"].shape[1]
        perm = np.arange(k)[::-1]
        for i, identifier in enumerate(ids):
            a = candidate_order(identifier, data["candidate_rows"][i], data["scores_detail"][i], TIE_SEED)
            b = candidate_order(identifier, data["candidate_rows"][i, perm], data["scores_detail"][i, perm], TIE_SEED)
            assert np.array_equal(data["candidate_rows"][i, a], data["candidate_rows"][i, perm][b])
    assert np.array_equal(routed[~fallback], original[~fallback])
    assert np.array_equal(routed[fallback], old[fallback])
    assert np.array_equal(routed <= 20, d3["baseline_ranks"] <= 20)
    outside = d3["baseline_ranks"] > 20
    assert np.array_equal(routed[outside], d3["baseline_ranks"][outside])
    np.savez_compressed(out / "per_query.npz", identifiers=ids, groups=groups,
                        support5=support5, support20=support20, use_d2=fallback,
                        ranks_d2=old, ranks_original_d3=original, ranks_routed_d3=routed,
                        baseline_ranks=d3["baseline_ranks"], top20_order_rows=order)
    print(json.dumps(dict(stage="integrity_pass", fallback_queries=int(fallback.sum()),
                          metrics=summarize(routed, groups))), flush=True)
    comparisons = {}
    for name, baseline in [("routed_vs_original_d3", original), ("routed_vs_d2", old)]:
        print("Computing " + name, flush=True)
        comparisons[name] = comparison(baseline, routed, groups, RESAMPLES, BOOTSTRAP_SEED)
    masks = dict(supported_both=support5 & support20, lost_support=fallback,
                 unsupported_both=~support5 & ~support20)
    strata = {}
    baseline = d3["baseline_ranks"]
    for name, mask in dict(rank1=baseline == 1, rank2to5=(baseline >= 2) & (baseline <= 5),
                           rank6to20=(baseline >= 6) & (baseline <= 20), rankabove20=baseline > 20).items():
        strata[name] = dict(queries=int(mask.sum()), vs_original_d3=transitions(original[mask], routed[mask]),
                            vs_d2=transitions(old[mask], routed[mask]))
    seeds = []
    for seed, rank in zip((20261010, 20261011, 20261012), d3["seed_ranks_detail"]):
        r = np.where(fallback, old, rank)
        seeds.append(dict(d3_tree_seed=seed, metrics=summarize(r, groups),
                          vs_original_same_tree_seed=transitions(rank, r), vs_d2=transitions(old, r)))
    audit = dict(status="PASS", source_score_to_rank_replay_exact=True,
                 parent_result_hashes_exact=True, identifiers_groups_order_exact=True,
                 top5_is_top20_prefix=True, selection_uses_only_support_masks=True,
                 complete_branch_order_replay_exact=True, top20_candidate_multiset_unchanged=True,
                 actual_candidate_slot_permutation_exact_all_queries=True,
                 no_GT_exact_ties_in_reachable_branch_scores=True,
                 outside_top20_and_R20_unchanged=True, original_D3_gate_unchanged=True)
    # Parent files and scientific code remain byte-identical through the replay.
    assert all(sha256_file(ROOT / p) == info["sha256"] for p, info in inputs.items())
    audit["inputs_unchanged_after_replay"] = True
    write(out / "audit.json", audit)
    result = dict(status="COMPLETE", official_test_read=False, model_fitting=False,
                  queries=n, connectivities=int(len(np.unique(groups))), candidate_recall=1.0,
                  rule=contract["rule"], fallback_queries=int(fallback.sum()),
                  fallback_fraction_pct=float(fallback.mean() * 100),
                  effective_detail_supported_queries=int((support5 | support20).sum()),
                  metrics={name: summarize(r, groups) for name, r in
                           [("d2", old), ("original_d3", original), ("routed_d3", routed)]},
                  comparisons=comparisons, original_rank_strata=strata,
                  support_strata={name: dict(queries=int(mask.sum()),
                                             vs_original_d3=transitions(original[mask], routed[mask]),
                                             vs_d2=transitions(old[mask], routed[mask]))
                                   for name, mask in masks.items()},
                  conditional_scorer_seed_diagnostics=seeds,
                  relative_to_ensemble_gap_closure_pct=float(
                      100 * ((routed == 1).mean() - (baseline == 1).mean()) /
                      ((baseline <= 20).mean() - (baseline == 1).mean())),
                  interpretation=contract["exposure"],
                  remaining_work=contract["limitations"],
                  original_D3_damage_cap_status="FAIL retained; no retroactive gate pass",
                  endpoint_selection="Authorized fixed-rule replay; final row/endpoint freeze remains pending.",
                  runner_seconds=time.time() - started,
                  peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    write(out / "summary.json", result)
    files = {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write(out / "completion_manifest.json", dict(status="COMPLETE", official_test_read=False,
                                                 parent_inputs_unchanged=True, files_sha256=files))
    print(json.dumps(dict(status="COMPLETE", metrics=result["metrics"],
                          comparisons={k: dict(delta=v["delta_query_weighted_pp"],
                                               transitions=v["rank1_transition"],
                                               cluster_CI=v["paired_bootstrap"]["connectivity_cluster_query_weighted"])
                                       for k, v in comparisons.items()},
                          seconds=result["runner_seconds"], peak_RSS_GiB=result["peak_RSS_GiB"]),
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
