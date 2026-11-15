#!/usr/bin/env python3
"""Relate SCCF wrong-spectrum retention to training proximity.

This is a post-hoc analysis of the frozen exact-mass/same-adduct spectrum
permutation audit.  It does not rerun retrieval.  Ground-truth training
proximity and donor--recipient similarity are both measured in the exact raw
SCCF block space used by the audited checkpoints (log1p cache values,
checkpoint-declared block normalization, then cosine similarity).

The inferential unit is the RDKit connectivity group.  Per-query outcomes are
first averaged across the two audited checkpoints and the three fixed donor
assignments, then averaged within connectivity.  This avoids treating repeated
spectra, checkpoint seeds, or donor assignments as independent observations.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glmr.candidate_indices import CandidateIndexStore  # noqa: E402
from tools.audits.audit_spectrum_permutation import (  # noqa: E402
    connectivity_identities,
    expected_query_values,
    file_sha256,
)
from tools.stage1.train_msalign_projection_smoke import (  # noqa: E402
    BlockwiseNormalizer,
    MoleculeEmbeddingStore,
    checkpoint_model_config,
    load_jsonl,
)


DEFAULT_RUN_ROOT = ROOT / "outputs/stage1_submission_0926/runs/full_r2_block_l2"
DEFAULT_CHECKPOINTS = (
    DEFAULT_RUN_ROOT / "seed20260621/checkpoint.pt",
    DEFAULT_RUN_ROOT / "seed20260622/checkpoint.pt",
)
DEFAULT_AUDITS = (
    DEFAULT_RUN_ROOT
    / "seed20260621/blind_controls/spectrum_permutation_exact_mass.per_query.npz",
    DEFAULT_RUN_ROOT
    / "seed20260622/blind_controls/spectrum_permutation_exact_mass.per_query.npz",
)
DEFAULT_TRAIN = ROOT / "outputs/msalign_stage1_full_v15/train_full.metadata.jsonl"
DEFAULT_VAL = ROOT / "outputs/msalign_stage1_full_v15/val_full.metadata.jsonl"
DEFAULT_FP = (
    ROOT
    / "outputs/stage1_submission_0926/assets/full_sparse_direct/"
    "manifest_historical_projection_order.json"
)
DEFAULT_CANDIDATES = (
    ROOT
    / "outputs/run_k_full_train/candidate_index_full_v15_mass_cap256_full_order/manifest.json"
)
DEFAULT_OUT = DEFAULT_RUN_ROOT / "analysis/wrong_spectrum_training_proximity.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-metadata", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--fingerprint-cache", type=Path, default=DEFAULT_FP)
    parser.add_argument("--candidate-index-manifest", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--checkpoint", type=Path, action="append", default=None,
        help="Audited checkpoint; repeat twice. Defaults to final r2 seeds 21 and 22.",
    )
    parser.add_argument(
        "--audit-npz", type=Path, action="append", default=None,
        help="Matching exact-mass audit NPZ; repeat in checkpoint order.",
    )
    parser.add_argument("--candidate-pool-limit", type=int, default=256)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260928)
    parser.add_argument("--similarity-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-per-connectivity", type=Path, default=None)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def gt_row(store: CandidateIndexStore, identifier: str, limit: int) -> int:
    values = store.get(str(identifier), limit)
    if not values:
        raise KeyError(f"No candidate row for {identifier}")
    return int(values[0])


def normalized_raw_features(
    mol_store: MoleculeEmbeddingStore,
    rows: list[int],
    mode: str,
) -> torch.Tensor:
    values = mol_store.gather(torch.tensor(rows, dtype=torch.long))
    normalizer = BlockwiseNormalizer(mol_store.block_dims, mode)
    return F.normalize(normalizer(values), p=2, dim=-1, eps=1e-12)


def nearest_training_similarity(
    val_features: torch.Tensor,
    train_features: torch.Tensor,
    val_connectivity: list[str],
    train_connectivity: list[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    train = train_features.to(device)
    columns_by_connectivity: dict[str, list[int]] = defaultdict(list)
    for column, key in enumerate(train_connectivity):
        columns_by_connectivity[key].append(column)
    output: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(val_features), batch_size):
            stop = min(start + batch_size, len(val_features))
            similarity = val_features[start:stop].to(device) @ train.T
            for local, key in enumerate(val_connectivity[start:stop]):
                columns = columns_by_connectivity.get(key)
                if columns:
                    similarity[local, torch.tensor(columns, device=device)] = -torch.inf
            output.append(similarity.max(dim=1).values.cpu())
    return torch.cat(output).numpy()


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else float("nan")


def partial_spearman(
    left: np.ndarray, right: np.ndarray, controls: list[np.ndarray]
) -> float:
    x = rankdata(left).astype(np.float64)
    y = rankdata(right).astype(np.float64)
    design = np.column_stack(
        [np.ones(len(left)), *[rankdata(value).astype(np.float64) for value in controls]]
    )
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.std(x_residual) == 0 or np.std(y_residual) == 0:
        return float("nan")
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def bootstrap_association(
    left: np.ndarray,
    right: np.ndarray,
    controls: list[np.ndarray],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    observed = partial_spearman(left, right, controls) if controls else safe_spearman(left, right)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    valid = 0
    for _ in range(resamples):
        sample = rng.integers(0, len(left), size=len(left))
        value = (
            partial_spearman(left[sample], right[sample], [item[sample] for item in controls])
            if controls
            else safe_spearman(left[sample], right[sample])
        )
        if np.isfinite(value):
            draws[valid] = value
            valid += 1
    draws = draws[:valid]
    return {
        "spearman_rho": observed,
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "bootstrap_resamples": int(valid),
    }


def quantile_assignment(values: np.ndarray, bins: int) -> tuple[np.ndarray, list[float]]:
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))[1:-1]
    return np.digitize(values, edges), [float(value) for value in edges]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(name: str) -> float:
        return float(np.mean([float(row[name]) for row in rows]))

    return {
        "connectivity_groups": len(rows),
        "queries": int(sum(int(row["queries"]) for row in rows)),
        "mean_training_proximity": mean("training_proximity"),
        "mean_donor_gt_similarity": mean("donor_gt_similarity"),
        "mean_candidate_count": mean("mean_candidate_count"),
        "connectivity_balanced": {
            "real_r1": 100.0 * mean("real_r1"),
            "wrong_r1": 100.0 * mean("wrong_r1"),
            "real_minus_wrong_r1_pp": 100.0 * mean("delta_r1"),
            "real_mrr": 100.0 * mean("real_mrr"),
            "wrong_mrr": 100.0 * mean("wrong_mrr"),
            "real_minus_wrong_mrr_pp": 100.0 * mean("delta_mrr"),
        },
    }


def grouped_table(
    rows: list[dict[str, Any]], field: str, bins: int
) -> tuple[list[dict[str, Any]], list[float]]:
    values = np.asarray([float(row[field]) for row in rows])
    assignment, edges = quantile_assignment(values, bins)
    table = []
    for index in range(bins):
        selected = [row for row, group in zip(rows, assignment, strict=True) if group == index]
        summary = summarize_rows(selected)
        summary["bin"] = index + 1
        summary["value_range"] = [
            float(min(row[field] for row in selected)),
            float(max(row[field] for row in selected)),
        ]
        table.append(summary)
    return table, edges


def high_minus_low_bootstrap(
    rows: list[dict[str, Any]], field: str, outcome: str, bins: int, resamples: int, seed: int
) -> dict[str, Any]:
    values = np.asarray([float(row[field]) for row in rows])
    outcomes = np.asarray([float(row[outcome]) for row in rows])
    assignment, _ = quantile_assignment(values, bins)
    low = outcomes[assignment == 0]
    high = outcomes[assignment == bins - 1]
    observed = float(high.mean() - low.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        draws[index] = (
            high[rng.integers(0, len(high), len(high))].mean()
            - low[rng.integers(0, len(low), len(low))].mean()
        )
    return {
        "direction": "highest quantile minus lowest quantile, percentage points",
        "delta_pp": 100.0 * observed,
        "ci95": [100.0 * float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "low_groups": int(len(low)),
        "high_groups": int(len(high)),
    }


def main() -> None:
    args = parse_args()
    checkpoints = list(args.checkpoint or DEFAULT_CHECKPOINTS)
    audit_paths = list(args.audit_npz or DEFAULT_AUDITS)
    if len(checkpoints) != len(audit_paths) or not checkpoints:
        raise ValueError("--checkpoint and --audit-npz must have the same non-zero length")

    checkpoint_payloads = [
        torch.load(path, map_location="cpu", weights_only=False) for path in checkpoints
    ]
    configs = [checkpoint_model_config(payload) for payload in checkpoint_payloads]
    blocks = list(configs[0].get("mol_blocks") or [])
    normalization = str(configs[0].get("mol_block_normalization", "none"))
    for config in configs[1:]:
        if list(config.get("mol_blocks") or []) != blocks:
            raise ValueError("Audited checkpoints use different molecule blocks")
        if str(config.get("mol_block_normalization", "none")) != normalization:
            raise ValueError("Audited checkpoints use different molecule normalization")

    audits = [np.load(path, allow_pickle=False) for path in audit_paths]
    recipients = audits[0]["recipient_rows"].astype(np.int64)
    connectivity = audits[0]["connectivity"].astype(str)
    seeds = sorted(
        int(key.removeprefix("seed").removesuffix("_donor_rows"))
        for key in audits[0].files
        if key.startswith("seed") and key.endswith("_donor_rows")
    )
    for audit in audits[1:]:
        if not np.array_equal(audit["recipient_rows"], recipients):
            raise ValueError("Audit NPZ recipient rows differ")
        if not np.array_equal(audit["connectivity"].astype(str), connectivity):
            raise ValueError("Audit NPZ connectivity labels differ")
        current = sorted(
            int(key.removeprefix("seed").removesuffix("_donor_rows"))
            for key in audit.files
            if key.startswith("seed") and key.endswith("_donor_rows")
        )
        if current != seeds:
            raise ValueError("Audit NPZ permutation seeds differ")

    train_records = load_jsonl(args.train_metadata)
    val_records = load_jsonl(args.val_metadata)
    if recipients.max(initial=-1) >= len(val_records):
        raise IndexError("Audit recipient row falls outside validation metadata")
    train_identity, train_identity_audit = connectivity_identities(train_records)
    val_identity, val_identity_audit = connectivity_identities(val_records)
    if any(val_identity[row] != key for row, key in zip(recipients, connectivity, strict=True)):
        raise ValueError("Audit connectivity labels do not match current validation metadata")

    candidate_store = CandidateIndexStore(args.candidate_index_manifest)
    train_gt_rows = [
        gt_row(candidate_store, str(record["identifier"]), args.candidate_pool_limit)
        for record in train_records
    ]
    # Keep every distinct positive molecule row. Repeated spectra do not add coverage.
    train_by_row: dict[int, str] = {}
    for row, key in zip(train_gt_rows, train_identity, strict=True):
        train_by_row.setdefault(row, key)
    train_rows = sorted(train_by_row)
    train_keys = [train_by_row[row] for row in train_rows]

    needed_val_rows = set(recipients.tolist())
    for seed in seeds:
        needed_val_rows.update(audits[0][f"seed{seed}_donor_rows"].astype(np.int64).tolist())
    val_gt_by_metadata_row = {
        row: gt_row(
            candidate_store,
            str(val_records[row]["identifier"]),
            args.candidate_pool_limit,
        )
        for row in sorted(needed_val_rows)
    }
    val_molecule_rows = sorted(set(val_gt_by_metadata_row.values()))

    mol_store = MoleculeEmbeddingStore(
        None,
        "mean",
        args.fingerprint_cache,
        preload_sparse_features=False,
        mol_blocks=blocks,
    )
    train_features = normalized_raw_features(mol_store, train_rows, normalization)
    val_features = normalized_raw_features(mol_store, val_molecule_rows, normalization)
    val_feature_by_global = {row: index for index, row in enumerate(val_molecule_rows)}

    recipient_global = [val_gt_by_metadata_row[int(row)] for row in recipients]
    unique_recipient_global = sorted(set(recipient_global))
    recipient_unique_position = {row: index for index, row in enumerate(unique_recipient_global)}
    unique_recipient_features = torch.stack(
        [val_features[val_feature_by_global[row]] for row in unique_recipient_global]
    )
    unique_recipient_connectivity = []
    connectivity_by_global: dict[int, str] = {}
    for row, key in zip(recipient_global, connectivity, strict=True):
        prior = connectivity_by_global.setdefault(row, key)
        if prior != key:
            raise ValueError("One molecule-cache row maps to multiple connectivity identities")
    unique_recipient_connectivity = [connectivity_by_global[row] for row in unique_recipient_global]
    device = choose_device(args.device)
    proximity_unique = nearest_training_similarity(
        unique_recipient_features,
        train_features,
        unique_recipient_connectivity,
        train_keys,
        device,
        args.similarity_batch_size,
    )
    proximity = np.asarray(
        [proximity_unique[recipient_unique_position[row]] for row in recipient_global],
        dtype=np.float64,
    )

    donor_similarity_by_seed: dict[int, np.ndarray] = {}
    for seed in seeds:
        donor_rows = audits[0][f"seed{seed}_donor_rows"].astype(np.int64)
        donor_global = [val_gt_by_metadata_row[int(row)] for row in donor_rows]
        recipient_tensor = torch.stack(
            [val_features[val_feature_by_global[row]] for row in recipient_global]
        )
        donor_tensor = torch.stack(
            [val_features[val_feature_by_global[row]] for row in donor_global]
        )
        donor_similarity_by_seed[seed] = torch.sum(
            recipient_tensor * donor_tensor, dim=1
        ).numpy().astype(np.float64)
    donor_similarity = np.mean(list(donor_similarity_by_seed.values()), axis=0)

    real_r1 = []
    real_mrr = []
    wrong_r1 = []
    wrong_mrr = []
    for audit in audits:
        real_values = expected_query_values(audit["baseline_greater"], audit["baseline_equal"])
        real_r1.append(real_values["recall@1"])
        real_mrr.append(real_values["mrr"])
        for seed in seeds:
            wrong_values = expected_query_values(
                audit[f"seed{seed}_own_greater"], audit[f"seed{seed}_own_equal"]
            )
            wrong_r1.append(wrong_values["recall@1"])
            wrong_mrr.append(wrong_values["mrr"])
    real_r1_mean = np.mean(real_r1, axis=0)
    real_mrr_mean = np.mean(real_mrr, axis=0)
    wrong_r1_mean = np.mean(wrong_r1, axis=0)
    wrong_mrr_mean = np.mean(wrong_mrr, axis=0)

    candidate_count = np.asarray(
        [
            candidate_store.row_length(
                int(candidate_store.row_for(str(val_records[int(row)]["identifier"]))),
                args.candidate_pool_limit,
            )
            for row in recipients
        ],
        dtype=np.float64,
    )

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(connectivity):
        grouped[str(key)].append(index)
    connectivity_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        index = np.asarray(grouped[key], dtype=np.int64)
        connectivity_rows.append(
            {
                "connectivity": key,
                "queries": int(len(index)),
                "training_proximity": float(np.mean(proximity[index])),
                "donor_gt_similarity": float(np.mean(donor_similarity[index])),
                "mean_candidate_count": float(np.mean(candidate_count[index])),
                "real_r1": float(np.mean(real_r1_mean[index])),
                "wrong_r1": float(np.mean(wrong_r1_mean[index])),
                "delta_r1": float(np.mean(real_r1_mean[index] - wrong_r1_mean[index])),
                "real_mrr": float(np.mean(real_mrr_mean[index])),
                "wrong_mrr": float(np.mean(wrong_mrr_mean[index])),
                "delta_mrr": float(np.mean(real_mrr_mean[index] - wrong_mrr_mean[index])),
            }
        )

    arrays = {
        name: np.asarray([float(row[name]) for row in connectivity_rows], dtype=np.float64)
        for name in (
            "training_proximity",
            "donor_gt_similarity",
            "mean_candidate_count",
            "real_r1",
            "wrong_r1",
            "delta_r1",
            "real_mrr",
            "wrong_mrr",
            "delta_mrr",
        )
    }
    log_count = np.log(arrays["mean_candidate_count"])
    association_specs = {
        "training_proximity_vs_real_r1": ("training_proximity", "real_r1", []),
        "training_proximity_vs_wrong_r1": ("training_proximity", "wrong_r1", []),
        "training_proximity_vs_delta_r1": ("training_proximity", "delta_r1", []),
        "donor_gt_similarity_vs_wrong_r1": ("donor_gt_similarity", "wrong_r1", []),
        "donor_gt_similarity_vs_delta_r1": ("donor_gt_similarity", "delta_r1", []),
        "training_proximity_vs_donor_gt_similarity": (
            "training_proximity", "donor_gt_similarity", []
        ),
    }
    associations: dict[str, Any] = {}
    for offset, (name, (left, right, _)) in enumerate(association_specs.items()):
        associations[name] = bootstrap_association(
            arrays[left],
            arrays[right],
            [],
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed + offset,
        )
    associations["training_proximity_vs_wrong_r1_adjusted"] = bootstrap_association(
        arrays["training_proximity"],
        arrays["wrong_r1"],
        [arrays["donor_gt_similarity"], log_count],
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed + 20,
    )
    associations["training_proximity_vs_delta_r1_adjusted"] = bootstrap_association(
        arrays["training_proximity"],
        arrays["delta_r1"],
        [arrays["donor_gt_similarity"], log_count],
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed + 21,
    )
    associations["donor_gt_similarity_vs_wrong_r1_adjusted"] = bootstrap_association(
        arrays["donor_gt_similarity"],
        arrays["wrong_r1"],
        [arrays["training_proximity"], log_count],
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed + 22,
    )

    proximity_table, proximity_edges = grouped_table(
        connectivity_rows, "training_proximity", args.bins
    )
    donor_table, donor_edges = grouped_table(
        connectivity_rows, "donor_gt_similarity", args.bins
    )
    overlap = sorted(set(train_identity) & set(connectivity))
    output = {
        "schema": "glmr_sccf_wrong_spectrum_proximity_v1",
        "protocol": {
            "scope": "official validation exact-mass/same-adduct/different-connectivity permutation subset",
            "training_proximity": (
                "cosine from validation GT to nearest distinct official-train positive in the "
                "checkpoint-declared raw SCCF block-normalized space"
            ),
            "donor_gt_similarity": (
                "cosine between recipient GT and wrong-spectrum donor GT in the same raw SCCF space; "
                "averaged over the three fixed donor assignments"
            ),
            "outcomes": (
                "per-query real outcomes averaged over checkpoint seeds; wrong outcomes averaged over "
                "checkpoint seeds and donor assignments; inference after averaging within connectivity"
            ),
            "association": (
                "Spearman on connectivity-level rows; adjusted associations are partial Spearman "
                "controlling the other chemical-similarity axis and log candidate count"
            ),
        },
        "inputs": {
            "train_metadata": str(args.train_metadata),
            "train_metadata_sha256": file_sha256(args.train_metadata),
            "val_metadata": str(args.val_metadata),
            "val_metadata_sha256": file_sha256(args.val_metadata),
            "fingerprint_cache": str(args.fingerprint_cache),
            "fingerprint_cache_manifest_sha256": file_sha256(args.fingerprint_cache),
            "candidate_index_manifest": str(args.candidate_index_manifest),
            "candidate_index_manifest_sha256": file_sha256(args.candidate_index_manifest),
            "checkpoints": [
                {"path": str(path), "sha256": file_sha256(path)} for path in checkpoints
            ],
            "audit_npz": [
                {"path": str(path), "sha256": file_sha256(path)} for path in audit_paths
            ],
        },
        "representation": {
            "blocks": blocks,
            "block_dims": mol_store.block_dims,
            "normalization": normalization,
            "post_normalization_similarity": "cosine",
        },
        "population": {
            "queries": int(len(recipients)),
            "connectivity_groups": len(connectivity_rows),
            "checkpoint_seeds": len(checkpoints),
            "permutation_seeds": seeds,
            "unique_train_positive_rows": len(train_rows),
            "train_recipient_connectivity_overlap": len(overlap),
            "train_identity_audit": train_identity_audit,
            "val_identity_audit": val_identity_audit,
        },
        "overall_connectivity_balanced": summarize_rows(connectivity_rows),
        "associations": associations,
        "training_proximity_quantiles": {
            "edges": proximity_edges,
            "rows": proximity_table,
            "highest_minus_lowest": {
                outcome: high_minus_low_bootstrap(
                    connectivity_rows,
                    "training_proximity",
                    outcome,
                    args.bins,
                    args.bootstrap_resamples,
                    args.bootstrap_seed + 40 + offset,
                )
                for offset, outcome in enumerate(("real_r1", "wrong_r1", "delta_r1"))
            },
        },
        "donor_gt_similarity_quantiles": {
            "edges": donor_edges,
            "rows": donor_table,
            "highest_minus_lowest": {
                outcome: high_minus_low_bootstrap(
                    connectivity_rows,
                    "donor_gt_similarity",
                    outcome,
                    args.bins,
                    args.bootstrap_resamples,
                    args.bootstrap_seed + 50 + offset,
                )
                for offset, outcome in enumerate(("wrong_r1", "delta_r1"))
            },
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    per_connectivity_path = args.out_per_connectivity or args.out.with_suffix(".per_connectivity.jsonl")
    with per_connectivity_path.open("w", encoding="utf-8") as handle:
        for row in connectivity_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {per_connectivity_path}")


if __name__ == "__main__":
    main()
