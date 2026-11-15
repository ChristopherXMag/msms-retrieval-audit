#!/usr/bin/env python3
"""Build a Stage-2 opportunity atlas from frozen official/data-safe Stage-1 ranks.

The analysis is deliberately label-aware and validation-only: it is for choosing
the Stage-2 training/evaluation design, not for deployment-time routing.  It never
loads official-test metadata, spectra, ranks, or scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glmr.candidate_indices import CandidateIndexStore
from glmr.smiles import profile_smiles


METRICS = ("recall@1", "recall@5", "recall@20", "mrr")
SEEDS = (20260621, 20260622, 20260623)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--val-metadata", type=Path, required=True)
    parser.add_argument("--candidate-index-manifest", type=Path, required=True)
    parser.add_argument("--rank-json", type=Path, nargs=3, required=True)
    parser.add_argument("--rank-seeds", type=int, nargs=3, default=SEEDS)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--negative-exclusion-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261001)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def contributions(ranks: np.ndarray) -> np.ndarray:
    values = ranks.astype(np.float64)
    return np.stack(
        [values <= 1, values <= 5, values <= 20, 1.0 / values], axis=-1
    ).astype(np.float64)


def metric_dict(values: np.ndarray) -> dict[str, float]:
    means = np.asarray(values, dtype=np.float64).mean(axis=0) * 100.0
    return {name: round(float(value), 6) for name, value in zip(METRICS, means)}


def connectivity_balanced(values: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.zeros((len(unique), values.shape[1]), dtype=np.float64)
    counts = np.bincount(inverse).astype(np.float64)
    np.add.at(sums, inverse, values)
    return metric_dict(sums / counts[:, None])


def percentile_interval(samples: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for index, name in enumerate(METRICS):
        column = samples[:, index] * 100.0
        result[name] = {
            "bootstrap_mean_pct": round(float(column.mean()), 6),
            "ci95_low_pct": round(float(np.quantile(column, 0.025)), 6),
            "ci95_high_pct": round(float(np.quantile(column, 0.975)), 6),
        }
    return result


def bootstrap_intervals(
    values: np.ndarray,
    groups: np.ndarray,
    resamples: int,
    seed: int,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Return query, cluster-QW, and connectivity-balanced intervals.

    ``values`` is [query, metric], already averaged across frozen Stage-1 seeds.
    Cluster-QW samples connectivity groups and retains their query multiplicity;
    connectivity-balanced samples one group mean per sampled group.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    unique, inverse = np.unique(groups, return_inverse=True)
    group_count = len(unique)
    group_sizes = np.bincount(inverse).astype(np.float64)
    group_sums = np.zeros((group_count, values.shape[1]), dtype=np.float64)
    np.add.at(group_sums, inverse, values)
    group_means = group_sums / group_sizes[:, None]
    query_samples = np.empty((resamples, values.shape[1]), dtype=np.float64)
    cluster_qw_samples = np.empty_like(query_samples)
    balanced_samples = np.empty_like(query_samples)
    for start in range(0, resamples, batch_size):
        width = min(batch_size, resamples - start)
        query_draw = rng.integers(0, n, size=(width, n))
        query_samples[start : start + width] = values[query_draw].mean(axis=1)
        group_draw = rng.integers(0, group_count, size=(width, group_count))
        drawn_sums = group_sums[group_draw].sum(axis=1)
        drawn_sizes = group_sizes[group_draw].sum(axis=1)
        cluster_qw_samples[start : start + width] = drawn_sums / drawn_sizes[:, None]
        balanced_samples[start : start + width] = group_means[group_draw].mean(axis=1)
    return {
        "resamples": resamples,
        "seed": seed,
        "seed_aggregation": "per-query contribution averaged over the three frozen Stage-1 seeds",
        "query_resampling": percentile_interval(query_samples),
        "connectivity_cluster_query_weighted": percentile_interval(cluster_qw_samples),
        "connectivity_balanced": percentile_interval(balanced_samples),
    }


def candidate_stratum(count: int) -> str:
    if count <= 32:
        return "2-32"
    if count <= 64:
        return "33-64"
    if count <= 128:
        return "65-128"
    if count <= 255:
        return "129-255"
    return "256"


def summarize_mask(mask: np.ndarray, values: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    selected = int(mask.sum())
    if selected == 0:
        return {"queries": 0, "connectivities": 0, "query_weighted": None,
                "connectivity_balanced": None}
    return {
        "queries": selected,
        "query_share_pct": round(100.0 * selected / len(mask), 6),
        "connectivities": int(len(np.unique(groups[mask]))),
        "query_weighted": metric_dict(values[mask]),
        "connectivity_balanced": connectivity_balanced(values[mask], groups[mask]),
    }


def identity_audit(
    train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    def profile(rows: Iterable[dict[str, Any]]) -> tuple[list[str], list[str], int, int]:
        canonical: list[str] = []
        connectivity: list[str] = []
        invalid = 0
        stored_mismatches = 0
        for row in rows:
            item = profile_smiles(
                row.get("canonical_isomeric_smiles") or row.get("raw_smiles")
            )
            if not item.valid or not item.canonical_isomeric or not item.inchi_key_first14:
                invalid += 1
                canonical.append("")
                connectivity.append("")
                continue
            canonical.append(item.canonical_isomeric)
            connectivity.append(item.inchi_key_first14)
            stored = str(row.get("inchikey") or "").split("-", 1)[0]
            stored_mismatches += int(bool(stored) and stored != item.inchi_key_first14)
        return canonical, connectivity, invalid, stored_mismatches

    train_can, train_conn, train_invalid, train_mismatch = profile(train_rows)
    val_can, val_conn, val_invalid, val_mismatch = profile(val_rows)
    train_can_set = set(train_can) - {""}
    val_can_set = set(val_can) - {""}
    train_conn_set = set(train_conn) - {""}
    val_conn_set = set(val_conn) - {""}
    report = {
        "identity_runtime_rdkit_version": rdBase.rdkitVersion,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_unique_canonical_isomeric_smiles": len(train_can_set),
        "val_unique_canonical_isomeric_smiles": len(val_can_set),
        "train_unique_connectivities": len(train_conn_set),
        "val_unique_connectivities": len(val_conn_set),
        "canonical_isomeric_smiles_overlap": len(train_can_set & val_can_set),
        "connectivity_overlap": len(train_conn_set & val_conn_set),
        "invalid_train_rows": train_invalid,
        "invalid_val_rows": val_invalid,
        "stored_inchikey_mismatches_train": train_mismatch,
        "stored_inchikey_mismatches_val": val_mismatch,
    }
    if train_invalid or val_invalid:
        raise ValueError(f"Invalid identity rows: train={train_invalid}, val={val_invalid}")
    if report["connectivity_overlap"]:
        raise ValueError("Official train/validation connectivity overlap is non-zero")
    return report, np.asarray(val_can), np.asarray(val_conn)


def main() -> None:
    args = parse_args()
    seeds = tuple(args.rank_seeds)
    if len(set(seeds)) != 3:
        raise ValueError("Exactly three distinct rank seeds are required")
    train_rows = load_jsonl(args.train_metadata)
    val_rows = load_jsonl(args.val_metadata)
    identifiers = np.asarray([str(row["identifier"]) for row in val_rows])
    if len(set(identifiers.tolist())) != len(identifiers):
        raise ValueError("Validation identifiers are not unique")
    identity, val_canonical, connectivities = identity_audit(train_rows, val_rows)

    store = CandidateIndexStore(args.candidate_index_manifest)
    candidate_counts = np.empty(len(val_rows), dtype=np.int32)
    candidate_rows = np.empty(len(val_rows), dtype=np.int32)
    for index, identifier in enumerate(identifiers):
        row = store.row_for(str(identifier))
        if row is None:
            raise KeyError(f"Candidate pool missing for {identifier}")
        candidate_rows[index] = row
        candidate_counts[index] = store.row_length(row, limit=256)
    if int(candidate_counts.min()) < 2 or int(candidate_counts.max()) > 256:
        raise ValueError("Candidate counts fall outside the frozen [2, 256] protocol")

    rank_matrix = np.empty((len(val_rows), 3), dtype=np.int32)
    per_seed: dict[str, Any] = {}
    rank_files: list[dict[str, Any]] = []
    for column, (seed, path) in enumerate(zip(seeds, args.rank_json, strict=True)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        ranks = np.asarray(payload["metrics"]["ranks"], dtype=np.int32)
        if len(ranks) != len(val_rows):
            raise ValueError(f"Rank length mismatch for seed {seed}: {len(ranks)}")
        if int(ranks.min()) < 1 or bool(np.any(ranks > candidate_counts)):
            raise ValueError(f"Ranks outside candidate pools for seed {seed}")
        rank_matrix[:, column] = ranks
        values = contributions(ranks)
        computed = metric_dict(values)
        stored = payload["metrics"]
        parity = {
            "recall@1": abs(computed["recall@1"] - float(stored["recall@1"])),
            "recall@5": abs(computed["recall@5"] - float(stored["recall@5"])),
            "recall@20": abs(computed["recall@20"] - float(stored["recall@20"])),
            "mrr": abs(computed["mrr"] - float(stored["mrr"])),
        }
        if max(parity.values()) > 5e-4:
            raise ValueError(f"Stored rank metric parity failed for seed {seed}: {parity}")
        per_seed[str(seed)] = {
            "query_weighted": computed,
            "connectivity_balanced": connectivity_balanced(values, connectivities),
            "stored_metric_max_abs_diff_pp": round(max(parity.values()), 9),
        }
        rank_files.append({"seed": seed, "path": str(path), "sha256": sha256_file(path)})

    all_values = np.stack([contributions(rank_matrix[:, col]) for col in range(3)], axis=1)
    seed_mean_values = all_values.mean(axis=1)
    top1_votes = (rank_matrix <= 1).sum(axis=1)
    top5_votes = (rank_matrix <= 5).sum(axis=1)
    top20_votes = (rank_matrix <= 20).sum(axis=1)
    median_rank = np.median(rank_matrix, axis=1)
    min_rank = rank_matrix.min(axis=1)
    max_rank = rank_matrix.max(axis=1)

    masks = {
        "stable_rank1_protect": top1_votes == 3,
        "fragile_rank1_any_seed_only": (top1_votes > 0) & (top1_votes < 3),
        "stable_top5_reachable_not_stable_rank1": (top5_votes == 3) & (top1_votes < 3),
        "stable_rank2_5_all_seeds": (min_rank >= 2) & (max_rank <= 5),
        "stable_top20_reachable_not_stable_top5": (top20_votes == 3) & (top5_votes < 3),
        "stable_rank6_20_all_seeds": (min_rank >= 6) & (max_rank <= 20),
        "top20_unstable_across_seeds": (top20_votes > 0) & (top20_votes < 3),
        "unreachable_top20_all_seeds": top20_votes == 0,
    }
    opportunity = {
        name: summarize_mask(mask, seed_mean_values, connectivities)
        for name, mask in masks.items()
    }

    strata: dict[str, Any] = {}
    labels = np.asarray([candidate_stratum(int(value)) for value in candidate_counts])
    for label in ("2-32", "33-64", "65-128", "129-255", "256"):
        mask = labels == label
        if mask.any():
            item = summarize_mask(mask, seed_mean_values, connectivities)
            item["stage2"] = {
                "stable_rank1_protect_queries": int(np.count_nonzero(mask & masks["stable_rank1_protect"])),
                "stable_rank2_5_queries": int(np.count_nonzero(mask & masks["stable_rank2_5_all_seeds"])),
                "stable_rank6_20_queries": int(np.count_nonzero(mask & masks["stable_rank6_20_all_seeds"])),
            }
            strata[label] = item

    group_rows: dict[str, list[int]] = defaultdict(list)
    for index, connectivity in enumerate(connectivities):
        group_rows[str(connectivity)].append(index)
    molecule_consensus = Counter()
    for indices in group_rows.values():
        votes = top1_votes[np.asarray(indices)]
        if bool(np.all(votes == 3)):
            molecule_consensus["all_queries_stable_right"] += 1
        elif bool(np.all(votes == 0)):
            molecule_consensus["all_queries_wrong_all_seeds"] += 1
        else:
            molecule_consensus["mixed_or_seed_unstable"] += 1

    audit = json.loads(args.candidate_audit.read_text(encoding="utf-8"))
    exclusion = json.loads(args.negative_exclusion_manifest.read_text(encoding="utf-8"))
    if int(audit["structure_audit"]["val_gt_slot_zero_smiles_parity"]["mismatches"]) != 0:
        raise ValueError("Frozen candidate audit reports validation GT-slot mismatch")
    if exclusion.get("heldout_use") != "identity-only; no spectra, scores, ranks, or metrics":
        raise ValueError("Unexpected held-out exclusion scope")

    report = {
        "format": "glmr_official_datasafe_stage2_opportunity_atlas_v1",
        "status": "validation_only_design_evidence",
        "official_test_read": False,
        "protocol": {
            "dataset": "MassSpecGym v1.5 official train/validation",
            "stage1": "frozen official/data-safe radius-2 SCCF",
            "candidate_pool": "official mass cap256; no GT insertion during evaluation",
            "rank_seeds": list(seeds),
            "stage2_scope": "candidate-conditioned reranking within frozen Stage-1 top-k",
        },
        "inputs": {
            "train_metadata": {"path": str(args.train_metadata), "sha256": sha256_file(args.train_metadata)},
            "val_metadata": {"path": str(args.val_metadata), "sha256": sha256_file(args.val_metadata)},
            "candidate_index_manifest": {"path": str(args.candidate_index_manifest), "sha256": sha256_file(args.candidate_index_manifest)},
            "candidate_audit": {"path": str(args.candidate_audit), "sha256": sha256_file(args.candidate_audit)},
            "negative_exclusion_manifest": {"path": str(args.negative_exclusion_manifest), "sha256": sha256_file(args.negative_exclusion_manifest)},
            "rank_files": rank_files,
        },
        "identity_audit": identity,
        "candidate_audit": {
            "queries": len(val_rows),
            "candidate_count_min": int(candidate_counts.min()),
            "candidate_count_median": float(np.median(candidate_counts)),
            "candidate_count_mean": round(float(candidate_counts.mean()), 6),
            "candidate_count_max": int(candidate_counts.max()),
            "gt_slot_zero_exact_matches": int(audit["structure_audit"]["val_gt_slot_zero_smiles_parity"]["exact_matches"]),
            "gt_slot_zero_mismatches": 0,
            "source_duplicate_occurrences_within_cap": 0,
            "heldout_negative_excluded_rows": int(exclusion["excluded_rows"]),
        },
        "per_seed_metrics": per_seed,
        "three_seed_mean": {
            "query_weighted": metric_dict(seed_mean_values),
            "connectivity_balanced": connectivity_balanced(seed_mean_values, connectivities),
        },
        "stage2_ceiling": {
            "mean_r1_to_r5_gap_pp": round(float(metric_dict(seed_mean_values)["recall@5"] - metric_dict(seed_mean_values)["recall@1"]), 6),
            "mean_r1_to_r20_gap_pp": round(float(metric_dict(seed_mean_values)["recall@20"] - metric_dict(seed_mean_values)["recall@1"]), 6),
            "interpretation": "oracle headroom only; a real reranker must trade rescues against damage and cannot insert missing candidates",
        },
        "seed_agreement": {
            "top1_vote_counts": {str(k): int(np.count_nonzero(top1_votes == k)) for k in range(4)},
            "top5_vote_counts": {str(k): int(np.count_nonzero(top5_votes == k)) for k in range(4)},
            "top20_vote_counts": {str(k): int(np.count_nonzero(top20_votes == k)) for k in range(4)},
            "rank_range_median": float(np.median(max_rank - min_rank)),
            "rank_range_p90": float(np.quantile(max_rank - min_rank, 0.9)),
        },
        "stage2_query_sets": opportunity,
        "candidate_count_strata": strata,
        "molecule_consensus": dict(sorted(molecule_consensus.items())),
        "uncertainty": bootstrap_intervals(
            seed_mean_values,
            connectivities,
            args.bootstrap_resamples,
            args.bootstrap_seed,
        ),
        "design_decision": {
            "primary_training_unit": "connectivity-grouped query with frozen top-k candidate list",
            "primary_challenger_slice": "GT in top5, Stage-1 rank >1; retain all rank1 queries as explicit damage controls",
            "secondary_tail_slice": "GT in ranks6-20, trained/evaluated separately from top5",
            "selection_metrics": [
                "query-weighted delta R@1 and MRR",
                "connectivity-balanced delta R@1 and MRR",
                "query and connectivity-cluster bootstrap intervals",
                "rescue, damage, churn, and gap closure",
            ],
            "prohibited_shortcuts": [
                "candidate slot/index as a feature",
                "GT insertion",
                "query GT formula",
                "official-test access before method freeze",
            ],
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for index, identifier in enumerate(identifiers):
            row = {
                "identifier": str(identifier),
                "connectivity": str(connectivities[index]),
                "candidate_count": int(candidate_counts[index]),
                "candidate_count_stratum": str(labels[index]),
                "rank_by_seed": {str(seed): int(rank_matrix[index, col]) for col, seed in enumerate(seeds)},
                "rank_min": int(min_rank[index]),
                "rank_median": float(median_rank[index]),
                "rank_max": int(max_rank[index]),
                "top1_votes": int(top1_votes[index]),
                "top5_votes": int(top5_votes[index]),
                "top20_votes": int(top20_votes[index]),
                "stable_rank1_protect": bool(masks["stable_rank1_protect"][index]),
                "stable_rank2_5": bool(masks["stable_rank2_5_all_seeds"][index]),
                "stable_rank6_20": bool(masks["stable_rank6_20_all_seeds"][index]),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        identifiers=identifiers,
        connectivities=connectivities,
        candidate_rows=candidate_rows,
        candidate_counts=candidate_counts,
        ranks=rank_matrix,
        seeds=np.asarray(seeds, dtype=np.int64),
        top1_votes=top1_votes,
        top5_votes=top5_votes,
        top20_votes=top20_votes,
    )
    print(json.dumps({
        "output_json": str(args.output_json),
        "output_jsonl": str(args.output_jsonl),
        "output_npz": str(args.output_npz),
        "stage2_ceiling": report["stage2_ceiling"],
        "stage2_query_sets": {key: value["queries"] for key, value in opportunity.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
