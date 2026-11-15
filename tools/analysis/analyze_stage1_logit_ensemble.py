#!/usr/bin/env python3
"""Audit a frozen multi-seed Stage-1 mean-logit ensemble as a Stage-2 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("recall@1", "recall@5", "recall@20", "mrr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-npz", type=Path, required=True)
    parser.add_argument("--score-summary", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs=3, default=(20260621, 20260622, 20260623))
    parser.add_argument("--primary-seed", type=int, default=20260621)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261001)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def contributions(ranks: np.ndarray) -> np.ndarray:
    values = ranks.astype(np.float64)
    return np.stack(
        [values <= 1, values <= 5, values <= 20, 1.0 / values], axis=-1
    ).astype(np.float64)


def metrics(values: np.ndarray) -> dict[str, float]:
    return {
        name: round(float(value), 6)
        for name, value in zip(METRICS, values.mean(axis=0) * 100.0)
    }


def grouped(values: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse).astype(np.float64)
    sums = np.zeros((len(unique), values.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, values)
    return sums, sizes, sums / sizes[:, None]


def interval(samples: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for index, name in enumerate(METRICS):
        column = samples[:, index] * 100.0
        result[name] = {
            "delta_pp": round(float(column.mean()), 6),
            "ci95_low_pp": round(float(np.quantile(column, 0.025)), 6),
            "ci95_high_pp": round(float(np.quantile(column, 0.975)), 6),
            "probability_positive": round(float(np.mean(column > 0)), 6),
        }
    return result


def paired_bootstrap(
    baseline: np.ndarray,
    ensemble: np.ndarray,
    groups: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    delta = contributions(ensemble) - contributions(baseline)
    sums, sizes, means = grouped(delta, groups)
    rng = np.random.default_rng(seed)
    query_samples = np.empty((resamples, len(METRICS)), dtype=np.float64)
    cluster_qw = np.empty_like(query_samples)
    cluster_balanced = np.empty_like(query_samples)
    batch_size = 128
    for start in range(0, resamples, batch_size):
        count = min(batch_size, resamples - start)
        query_draw = rng.integers(0, len(delta), size=(count, len(delta)))
        query_samples[start : start + count] = delta[query_draw].mean(axis=1)
        cluster_draw = rng.integers(0, len(sizes), size=(count, len(sizes)))
        cluster_qw[start : start + count] = (
            sums[cluster_draw].sum(axis=1) / sizes[cluster_draw].sum(axis=1)[:, None]
        )
        cluster_balanced[start : start + count] = means[cluster_draw].mean(axis=1)
    return {
        "resamples": resamples,
        "seed": seed,
        "query": interval(query_samples),
        "connectivity_cluster_query_weighted": interval(cluster_qw),
        "connectivity_balanced": interval(cluster_balanced),
    }


def comparison(
    baseline: np.ndarray,
    ensemble: np.ndarray,
    groups: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_values = contributions(baseline)
    ensemble_values = contributions(ensemble)
    _, _, baseline_group = grouped(baseline_values, groups)
    _, _, ensemble_group = grouped(ensemble_values, groups)
    rescue = (baseline > 1) & (ensemble == 1)
    damage = (baseline == 1) & (ensemble > 1)
    base_metrics = metrics(baseline_values)
    ensemble_metrics = metrics(ensemble_values)
    gap = base_metrics["recall@20"] - base_metrics["recall@1"]
    return {
        "baseline_query_weighted": base_metrics,
        "ensemble_query_weighted": ensemble_metrics,
        "delta_query_weighted_pp": {
            name: round(ensemble_metrics[name] - base_metrics[name], 6) for name in METRICS
        },
        "baseline_connectivity_balanced": metrics(baseline_group),
        "ensemble_connectivity_balanced": metrics(ensemble_group),
        "delta_connectivity_balanced_pp": {
            name: round(float((ensemble_group[:, i] - baseline_group[:, i]).mean() * 100.0), 6)
            for i, name in enumerate(METRICS)
        },
        "rank1_transition": {
            "rescues": int(rescue.sum()),
            "damages": int(damage.sum()),
            "net_correct": int(rescue.sum() - damage.sum()),
            "rescue_damage_ratio": round(float(rescue.sum() / max(int(damage.sum()), 1)), 6),
            "baseline_rank1_damage_rate_pct": round(100.0 * float(damage.sum()) / max(int((baseline == 1).sum()), 1), 6),
            "connectivities_with_net_more_correct": int(np.count_nonzero(ensemble_group[:, 0] > baseline_group[:, 0])),
            "connectivities_with_net_fewer_correct": int(np.count_nonzero(ensemble_group[:, 0] < baseline_group[:, 0])),
        },
        "r1_to_r20_gap_closure_pct": round(
            100.0 * (ensemble_metrics["recall@1"] - base_metrics["recall@1"]) / gap, 6
        ),
        "paired_bootstrap": paired_bootstrap(
            baseline, ensemble, groups, resamples, seed
        ),
    }


def main() -> None:
    args = parse_args()
    if args.primary_seed not in args.seeds:
        raise ValueError("primary-seed must be one of --seeds")
    arrays = np.load(args.scores_npz)
    identifiers = arrays["identifiers"]
    groups = arrays["connectivities"]
    ensemble = arrays["ranks_ensemble_mean_logits"].astype(np.int32)
    comparisons: dict[str, Any] = {}
    for index, seed_value in enumerate(args.seeds):
        comparisons[str(seed_value)] = comparison(
            arrays[f"ranks_seed_{index}"].astype(np.int32),
            ensemble,
            groups,
            args.bootstrap_resamples,
            args.bootstrap_seed + index,
        )
    score_summary = json.loads(args.score_summary.read_text(encoding="utf-8"))
    report = {
        "format": "glmr_stage1_mean_logit_ensemble_audit_v1",
        "status": "frozen_validation_baseline",
        "official_test_read": False,
        "method": "unweighted arithmetic mean of candidate logits from three frozen SCCF checkpoints",
        "selection": "no validation-tuned weights or thresholds",
        "scores_npz": {"path": str(args.scores_npz), "sha256": sha256_file(args.scores_npz)},
        "score_summary": {"path": str(args.score_summary), "sha256": sha256_file(args.score_summary)},
        "input_checkpoint_sha256": score_summary["checkpoints"],
        "queries": len(identifiers),
        "connectivities": int(len(np.unique(groups))),
        "seeds": list(args.seeds),
        "primary_single_seed": args.primary_seed,
        "ensemble_query_weighted": metrics(contributions(ensemble)),
        "ensemble_connectivity_balanced": metrics(grouped(contributions(ensemble), groups)[2]),
        "comparisons": comparisons,
        "interpretation": {
            "role": "strong score-space baseline for Stage 2, not new chemical evidence",
            "next_gate": "a learned candidate-conditioned reranker must beat the frozen mean-logit ensemble under query and connectivity-cluster uncertainty while reporting rank1 damage",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_json),
        "ensemble_query_weighted": report["ensemble_query_weighted"],
        "ensemble_connectivity_balanced": report["ensemble_connectivity_balanced"],
        "primary_comparison": comparisons[str(args.primary_seed)],
    }, indent=2))


if __name__ == "__main__":
    main()
