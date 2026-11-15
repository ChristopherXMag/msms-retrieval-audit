#!/usr/bin/env python3
"""Aggregate available P0.3 val ranks, bootstrap gains, and choose the baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (20260621, 20260622, 20260623)
ARMS = ("control", "balanced", "all_row")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=ROOT / "outputs/run_k_full_train/experiments",
    )
    parser.add_argument(
        "--val-metadata",
        type=Path,
        default=ROOT / "outputs/msalign_formula_group_split_v15/val.metadata.jsonl",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/run_k_full_train/p03_val_decision.json",
    )
    return parser.parse_args()


def metrics(ranks: np.ndarray) -> dict[str, float | int]:
    return {
        "queries": int(len(ranks)),
        "recall@1": round(100 * float((ranks <= 1).mean()), 4),
        "recall@5": round(100 * float((ranks <= 5).mean()), 4),
        "recall@20": round(100 * float((ranks <= 20).mean()), 4),
        "mrr": round(100 * float((1.0 / ranks).mean()), 4),
    }


def group_weighted_metrics(ranks: np.ndarray, groups: np.ndarray) -> dict[str, float | int]:
    unique, inverse = np.unique(groups, return_inverse=True)
    result: dict[str, float | int] = {"groups": int(len(unique))}
    values = {
        "recall@1": (ranks <= 1).astype(np.float64),
        "recall@5": (ranks <= 5).astype(np.float64),
        "recall@20": (ranks <= 20).astype(np.float64),
        "mrr": 1.0 / ranks,
    }
    counts = np.bincount(inverse)
    for name, per_query in values.items():
        per_group = np.bincount(inverse, weights=per_query) / counts
        result[name] = round(100 * float(per_group.mean()), 4)
    return result


def paired_bootstrap(
    reference: np.ndarray,
    comparison: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    per_query = {
        "recall@1": 100.0 * (
            (comparison <= 1).astype(np.float64) - (reference <= 1).astype(np.float64)
        ),
        "recall@20": 100.0 * (
            (comparison <= 20).astype(np.float64) - (reference <= 20).astype(np.float64)
        ),
        "mrr": 100.0 * (1.0 / comparison - 1.0 / reference),
    }
    rng = np.random.default_rng(seed)
    distributions = {name: np.empty(samples, dtype=np.float64) for name in per_query}
    for start in range(0, samples, 128):
        count = min(128, samples - start)
        indices = rng.integers(0, len(reference), size=(count, len(reference)))
        for name, values in per_query.items():
            distributions[name][start : start + count] = values[indices].mean(axis=1)
    result: dict[str, Any] = {}
    for name, values in per_query.items():
        draws = distributions[name]
        result[name] = {
            "delta_percentage_points": round(float(values.mean()), 6),
            "ci95": [
                round(float(np.quantile(draws, 0.025)), 6),
                round(float(np.quantile(draws, 0.975)), 6),
            ],
            "probability_positive": round(float((draws > 0).mean()), 6),
        }
    return {"unit": "query", "samples": samples, "seed": seed, "metrics": result}


def mean_metrics(runs: list[dict[str, Any]], key: str) -> dict[str, float]:
    names = ("recall@1", "recall@5", "recall@20", "mrr")
    return {
        name: round(float(np.mean([run[key][name] for run in runs])), 4)
        for name in names
    }


def main() -> None:
    args = parse_args()
    groups = []
    with args.val_metadata.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                groups.append(str(row.get("inchikey") or "").split("-", 1)[0])
    group_array = np.asarray(groups, dtype=object)

    output: dict[str, Any] = {
        "protocol": {
            "selection": (
                "mean over available completed seeds; when seed counts differ, also report "
                "same-seed deltas against control"
            ),
            "promotion_guard": "mean recall@20 no more than 0.2 pp below matched control",
            "bootstrap": "query-level paired, full arm minus same-seed control",
        },
        "arms": {},
    }
    rank_arrays: dict[tuple[str, int], np.ndarray] = {}
    arm_seeds: dict[str, list[int]] = {}
    for arm in ARMS:
        runs = []
        for seed in SEEDS:
            path = args.experiment_root / arm / f"seed{seed}" / "val_ranks.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            ranks = np.asarray(payload["metrics"]["ranks"], dtype=np.int64)
            if len(ranks) != len(group_array):
                raise ValueError(f"{path}: rank/query count mismatch")
            rank_arrays[(arm, seed)] = ranks
            runs.append(
                {
                    "seed": seed,
                    "path": str(path),
                    "query_weighted": metrics(ranks),
                    "group_weighted": group_weighted_metrics(ranks, group_array),
                }
            )
        if not runs:
            raise FileNotFoundError(f"No completed val ranks found for arm {arm!r}")
        arm_seeds[arm] = [int(run["seed"]) for run in runs]
        output["arms"][arm] = {
            "runs": runs,
            "mean_query_weighted": mean_metrics(runs, "query_weighted"),
            "mean_group_weighted": mean_metrics(runs, "group_weighted"),
        }

    control_mean = output["arms"]["control"]["mean_query_weighted"]
    for arm_index, arm in enumerate(("balanced", "all_row")):
        bootstraps = []
        matched_seeds = sorted(set(arm_seeds[arm]) & set(arm_seeds["control"]))
        if not matched_seeds:
            raise ValueError(f"{arm}: no seed-matched control is available")
        for seed_index, seed in enumerate(matched_seeds):
            bootstraps.append(
                {
                    "seed": seed,
                    **paired_bootstrap(
                        rank_arrays[("control", seed)],
                        rank_arrays[(arm, seed)],
                        args.bootstrap_samples,
                        args.bootstrap_seed + 10 * arm_index + seed_index,
                    ),
                }
            )
        arm_mean = output["arms"][arm]["mean_query_weighted"]
        available_mean_delta = {
            name: round(float(arm_mean[name] - control_mean[name]), 4)
            for name in ("recall@1", "recall@20", "mrr")
        }
        arm_runs = {int(run["seed"]): run for run in output["arms"][arm]["runs"]}
        control_runs = {
            int(run["seed"]): run for run in output["arms"]["control"]["runs"]
        }
        same_seed_delta = {
            name: round(
                float(
                    np.mean(
                        [
                            arm_runs[seed]["query_weighted"][name]
                            - control_runs[seed]["query_weighted"][name]
                            for seed in matched_seeds
                        ]
                    )
                ),
                4,
            )
            for name in ("recall@1", "recall@20", "mrr")
        }
        mean_delta = same_seed_delta
        per_seed_higher = all(
            arm_runs[seed]["query_weighted"][metric]
            > control_runs[seed]["query_weighted"][metric]
            for seed in matched_seeds
            for metric in ("recall@1", "mrr")
        )
        ci_lower_positive = all(
            bootstrap["metrics"][metric]["ci95"][0] > 0
            for bootstrap in bootstraps
            for metric in ("recall@1", "mrr")
        )
        checks = {
            "each_seed_recall1_and_mrr_above_control": per_seed_higher,
            "mean_recall1_gain_at_least_2pp": mean_delta["recall@1"] >= 2.0,
            "mean_mrr_gain_at_least_1pp": mean_delta["mrr"] >= 1.0,
            "mean_recall20_not_below_minus_0.2pp": mean_delta["recall@20"] >= -0.2,
            "each_seed_r1_and_mrr_bootstrap_ci_lower_positive": ci_lower_positive,
        }
        output["arms"][arm]["mean_delta_vs_control_pp"] = mean_delta
        output["arms"][arm]["available_run_mean_delta_vs_control_mean_pp"] = (
            available_mean_delta
        )
        output["arms"][arm]["matched_seeds"] = matched_seeds
        output["arms"][arm]["same_seed_mean_delta_vs_control_pp"] = same_seed_delta
        output["arms"][arm]["paired_bootstrap_vs_control"] = bootstraps
        output["arms"][arm]["practical_success_checks"] = checks
        output["arms"][arm]["practical_success"] = all(checks.values())

    selection_seeds = sorted(set.intersection(*(set(arm_seeds[arm]) for arm in ARMS)))
    if not selection_seeds:
        raise ValueError("No seed is shared by every arm for a matched selection")
    output["protocol"]["selection_seeds"] = selection_seeds
    selection_metrics: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        runs_by_seed = {
            int(run["seed"]): run for run in output["arms"][arm]["runs"]
        }
        selection_metrics[arm] = {
            name: round(
                float(
                    np.mean(
                        [
                            runs_by_seed[seed]["query_weighted"][name]
                            for seed in selection_seeds
                        ]
                    )
                ),
                4,
            )
            for name in ("recall@1", "recall@5", "recall@20", "mrr")
        }
        output["arms"][arm]["matched_selection_query_weighted"] = selection_metrics[arm]

    eligible = []
    selection_control = selection_metrics["control"]
    for arm in ARMS:
        mean = selection_metrics[arm]
        r20_guard = mean["recall@20"] >= selection_control["recall@20"] - 0.2
        output["arms"][arm]["promotion_eligible"] = r20_guard
        if r20_guard:
            eligible.append(arm)
    selected = max(
        eligible,
        key=lambda arm: tuple(
            selection_metrics[arm][name]
            for name in ("mrr", "recall@1", "recall@20")
        ),
    )
    runner_up = max(
        [arm for arm in eligible if arm != selected],
        key=lambda arm: tuple(
            selection_metrics[arm][name]
            for name in ("mrr", "recall@1", "recall@20")
        ),
    )
    runner_up_bootstraps = []
    for seed_index, seed in enumerate(selection_seeds):
        runner_up_bootstraps.append(
            {
                "seed": seed,
                **paired_bootstrap(
                    rank_arrays[(runner_up, seed)],
                    rank_arrays[(selected, seed)],
                    args.bootstrap_samples,
                    args.bootstrap_seed + 100 + seed_index,
                ),
            }
        )
    output["selected_vs_runner_up"] = {
        "selected": selected,
        "runner_up": runner_up,
        "matched_seeds": selection_seeds,
        "paired_bootstrap": runner_up_bootstraps,
    }
    output["selected_downstream_baseline"] = selected
    output["representative_checkpoint_for_error_atlas"] = {
        "arm": selected,
        "seed": arm_seeds[selected][0],
        "rationale": "Use the first locked representative seed available for the selected arm.",
    }
    output["test_policy"] = "Run the selected arm's representative seed once on full test."
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
