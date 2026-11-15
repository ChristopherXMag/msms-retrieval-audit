#!/usr/bin/env python3
"""Separate training-coverage failure from target-geometry failure for Run K.

A variance decomposition over repeated spectra shows that Run K correctness is
mostly a property of the molecule rather than of the individual spectrum
(ICC(1) = 0.756; 44.9% of molecules with at least four spectra have every
spectrum wrong). That is consistent with two very different diagnoses:

  coverage   the spectrum tower only learned to map into regions of molecule
             space that the training positives populate, so val molecules far
             from every training molecule are unreachable;
  geometry   the ground truth sits where MS/MS evidence cannot place it,
             independent of how well training covered that region.

They imply different fixes, so this script measures the coverage axis directly.
For every val ground-truth molecule it computes the similarity to the nearest
training positive in both the raw Run K four-block space and the Run K projected
space, then relates that to the molecule's per-spectrum accuracy. Candidate-pool
crowding (s_max_projected) is carried alongside as the competing explanation so
the two can be crossed rather than confounded.

Train positives are the deduplicated ground-truth molecules of the exact 20k
sample Run K trained on. Negatives seen through candidate CE are not counted as
coverage because only positives are ever paired with a spectrum.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glmr.sparse_features import SparseFeatureCache, load_sparse_feature_cache  # noqa: E402
from tools.analysis.analyze_run_k_error_atlas import BLOCKS, load_run_k_model  # noqa: E402
from train_spec_encoder.train_dreams_chemistry_adapter import (  # noqa: E402
    run_k_fingerprint_batch,
)

DEFAULT_TRAIN_METADATA = (
    ROOT / "outputs/msalign_formula_group_split_v15/train_sample20k_seed20260623.metadata.jsonl"
)
DEFAULT_VAL_METADATA = ROOT / "outputs/msalign_formula_group_split_v15/val.metadata.jsonl"
DEFAULT_ALL_QUERIES = ROOT / "outputs/run_k_error_atlas/full_val_20260809/all_queries.jsonl"
DEFAULT_BANDS = (
    ROOT / "outputs/run_k_error_atlas/full_val_20260809/similarity_band_assignment.jsonl"
)
DEFAULT_TOPK = ROOT / "outputs/run_k_glacier_fusion/full_val_top50/run_k_topk.jsonl"
DEFAULT_CHECKPOINT = (
    ROOT / "outputs/msalign_molecule_ablation/run_k_r3_allfp_blockl2_20k_cap256_5ep.pt"
)
DEFAULT_FP = (
    ROOT
    / "outputs/msalign_molecule_ablation/"
    "sparse_fingerprints_formula_group_20k_val_test_cap256/manifest.json"
)
DEFAULT_R3 = (
    ROOT
    / "outputs/msalign_molecule_ablation/"
    "sparse_count_morgan_r3_formula_group_20k_val_test_cap256/manifest.json"
)
DEFAULT_OUT = (
    ROOT / "outputs/run_k_error_atlas/full_val_20260809/train_proximity_vs_reachability.json"
)

BLOCK_DIMS = (2048, 2048, 2048, 167)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_TRAIN_METADATA)
    parser.add_argument("--val-metadata", type=Path, default=DEFAULT_VAL_METADATA)
    parser.add_argument("--all-queries", type=Path, default=DEFAULT_ALL_QUERIES)
    parser.add_argument("--band-assignment", type=Path, default=DEFAULT_BANDS)
    parser.add_argument("--topk-jsonl", type=Path, default=DEFAULT_TOPK)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--state-key",
        choices=("best_model_state_dict", "model_state_dict"),
        default="best_model_state_dict",
    )
    parser.add_argument("--fingerprint-manifest", type=Path, default=DEFAULT_FP)
    parser.add_argument("--r3-manifest", type=Path, default=DEFAULT_R3)
    parser.add_argument("--composite-fingerprint-manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-spectra", type=int, default=4)
    parser.add_argument("--parity-samples", type=int, default=64)
    parser.add_argument("--deciles", type=int, default=10)
    parser.add_argument(
        "--dump-per-molecule",
        type=Path,
        default=None,
        help="Optional jsonl with the per-molecule statistics behind the summary.",
    )
    return parser.parse_args()


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def unique_molecules(path: Path) -> dict[str, str]:
    """Map connectivity key (InChIKey-14) to one canonical isomeric SMILES."""
    molecules: dict[str, str] = {}
    for row in load_jsonl(path):
        key = row.get("inchikey")
        smiles = row.get("canonical_isomeric_smiles")
        if key and smiles and key not in molecules:
            molecules[key] = smiles
    return molecules


def block_normalize(features: torch.Tensor) -> torch.Tensor:
    parts = []
    offset = 0
    for dim in BLOCK_DIMS:
        parts.append(F.normalize(features[:, offset : offset + dim], dim=-1))
        offset += dim
    return F.normalize(torch.cat(parts, dim=-1), dim=-1)


def featurize(smiles_values: list[str], batch: int = 2048) -> torch.Tensor:
    chunks = []
    for start in range(0, len(smiles_values), batch):
        features, valid = run_k_fingerprint_batch(smiles_values[start : start + batch])
        if not all(valid):
            raise ValueError("Encountered an unparseable ground-truth SMILES")
        chunks.append(features)
    return torch.cat(chunks, dim=0)


def parity_check(args: argparse.Namespace) -> dict[str, Any]:
    """Confirm recomputed Run K features match the cached sparse rows."""
    samples: list[tuple[int, str]] = []
    for row in load_jsonl(args.topk_jsonl):
        gt = next((c for c in row.get("candidates") or [] if c.get("is_gt")), None)
        if gt is not None:
            samples.append((int(gt["global_index"]), gt["smiles"]))
        if len(samples) >= args.parity_samples:
            break

    if args.composite_fingerprint_manifest is not None:
        base = load_sparse_feature_cache(args.composite_fingerprint_manifest)
        stores = {name: base for name, _, _ in BLOCKS}
    else:
        base = SparseFeatureCache(args.fingerprint_manifest)
        r3 = SparseFeatureCache(args.r3_manifest)
        stores = {"count_morgan_r3": r3, **{name: base for name, _, _ in BLOCKS[1:]}}
    indices = torch.tensor([index for index, _ in samples], dtype=torch.long)
    cached = torch.cat(
        [stores[name].gather(name, indices).float() for name, _, _ in BLOCKS], dim=-1
    )
    recomputed = featurize([smiles for _, smiles in samples])
    delta = (cached - recomputed).abs()
    return {
        "samples": len(samples),
        "max_abs_difference": float(delta.max()),
        "mean_abs_difference": float(delta.mean()),
        "cosine_min": float(
            F.cosine_similarity(block_normalize(cached), block_normalize(recomputed), dim=-1).min()
        ),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def decile_table(
    values: np.ndarray, accuracy: np.ndarray, extra: dict[str, np.ndarray], deciles: int
) -> list[dict[str, Any]]:
    finite = np.isfinite(values)
    finite_values = values[finite]
    finite_accuracy = accuracy[finite]
    finite_extra = {name: array[finite] for name, array in extra.items()}
    edges = np.quantile(finite_values, np.linspace(0.0, 1.0, deciles + 1))[1:-1]
    assignment = np.digitize(finite_values, edges)
    rows = []
    for index in range(deciles):
        mask = assignment == index
        if mask.sum() < 10:
            continue
        row = {
            "decile": index,
            "molecules": int(mask.sum()),
            "value_range": [float(finite_values[mask].min()), float(finite_values[mask].max())],
            "molecule_weighted_r1": float(finite_accuracy[mask].mean()),
        }
        for name, array in finite_extra.items():
            row[f"mean_{name}"] = float(np.nanmean(array[mask]))
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    parity = parity_check(args)
    if parity["max_abs_difference"] > 1e-4:
        raise SystemExit(f"Run K feature parity failed: {parity}")

    train_molecules = unique_molecules(args.train_metadata)
    val_molecules = unique_molecules(args.val_metadata)
    overlap = sorted(set(train_molecules) & set(val_molecules))

    per_molecule: dict[str, list[bool]] = defaultdict(list)
    parent_mass: dict[str, list[float]] = defaultdict(list)
    peak_counts: dict[str, list[float]] = defaultdict(list)
    for row in load_jsonl(args.all_queries):
        per_molecule[row["gt_group_key"]].append(int(row["rank"]) == 1)
        parent_mass[row["gt_group_key"]].append(float(row["parent_mass"]))
        peak_counts[row["gt_group_key"]].append(float(row["num_peaks"]))

    pool_difficulty: dict[str, list[float]] = defaultdict(list)
    for row in load_jsonl(args.band_assignment):
        pool_difficulty[row["gt_group_key"]].append(float(row["s_max_projected"]))

    keys = [
        key
        for key in val_molecules
        if len(per_molecule.get(key, [])) >= args.min_spectra
    ]
    if not keys:
        raise SystemExit("No val molecules met the minimum spectrum count.")

    model, _ = load_run_k_model(args.checkpoint, device, args.state_key)
    with torch.inference_mode():
        train_features = featurize([train_molecules[key] for key in sorted(train_molecules)])
        val_features = featurize([val_molecules[key] for key in keys])

        train_raw = block_normalize(train_features).to(device)
        val_raw = block_normalize(val_features).to(device)
        train_projected = F.normalize(model.project_mol(train_features.to(device)), dim=-1)
        val_projected = F.normalize(model.project_mol(val_features.to(device)), dim=-1)

        raw_similarity = val_raw @ train_raw.T
        projected_similarity = val_projected @ train_projected.T
        nearest_raw = raw_similarity.max(dim=-1).values.cpu().numpy()
        nearest_projected = projected_similarity.max(dim=-1).values.cpu().numpy()
        top5_raw = raw_similarity.topk(5, dim=-1).values.mean(dim=-1).cpu().numpy()

    accuracy = np.array([float(np.mean(per_molecule[key])) for key in keys])
    spectra = np.array([len(per_molecule[key]) for key in keys])
    s_max_projected = np.array(
        [
            float(np.mean(pool_difficulty[key])) if pool_difficulty.get(key) else float("nan")
            for key in keys
        ]
    )

    extra = {
        "nearest_train_raw": nearest_raw,
        "nearest_train_projected": nearest_projected,
        "s_max_projected": s_max_projected,
        "spectra": spectra.astype(np.float64),
    }

    classes = {
        "all_wrong": accuracy == 0.0,
        "mixed": (accuracy > 0.0) & (accuracy < 1.0),
        "all_right": accuracy == 1.0,
    }
    class_rows = {
        name: {
            "molecules": int(mask.sum()),
            "spectra": int(spectra[mask].sum()),
            "mean_nearest_train_raw": float(nearest_raw[mask].mean()),
            "mean_nearest_train_projected": float(nearest_projected[mask].mean()),
            "mean_top5_train_raw": float(top5_raw[mask].mean()),
            "mean_s_max_projected": float(np.nanmean(s_max_projected[mask])),
        }
        for name, mask in classes.items()
        if mask.sum()
    }

    # Cross the two explanations so neither absorbs the other's variance.
    raw_median = float(np.median(nearest_raw))
    pool_median = float(np.nanmedian(s_max_projected))
    cross = {}
    for near_label, near_mask in (
        ("far_from_train", nearest_raw <= raw_median),
        ("near_train", nearest_raw > raw_median),
    ):
        for pool_label, pool_mask in (
            ("uncrowded_pool", s_max_projected <= pool_median),
            ("crowded_pool", s_max_projected > pool_median),
        ):
            mask = near_mask & pool_mask
            if mask.sum() >= 10:
                cross[f"{near_label}__{pool_label}"] = {
                    "molecules": int(mask.sum()),
                    "molecule_weighted_r1": float(accuracy[mask].mean()),
                }

    summary = {
        "protocol": {
            "train_metadata": str(args.train_metadata),
            "val_metadata": str(args.val_metadata),
            "checkpoint": str(args.checkpoint),
            "state_key": args.state_key,
            "train_positive_molecules": len(train_molecules),
            "val_molecules_analyzed": len(keys),
            "min_spectra_per_molecule": args.min_spectra,
            "train_val_connectivity_overlap": len(overlap),
            "coverage_definition": "nearest deduplicated train positive, not candidate negatives",
        },
        "feature_parity_vs_sparse_cache": parity,
        "correlations_with_molecule_weighted_r1": {
            "spearman_nearest_train_raw": spearman(nearest_raw, accuracy),
            "spearman_nearest_train_projected": spearman(nearest_projected, accuracy),
            "spearman_top5_train_raw": spearman(top5_raw, accuracy),
            "spearman_s_max_projected": spearman(s_max_projected, accuracy),
        },
        "by_outcome_class": class_rows,
        "by_nearest_train_raw_decile": decile_table(nearest_raw, accuracy, extra, args.deciles),
        "by_s_max_projected_decile": decile_table(s_max_projected, accuracy, extra, args.deciles),
        "crossed_median_split": cross,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.dump_per_molecule is not None:
        args.dump_per_molecule.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_per_molecule.open("w", encoding="utf-8") as handle:
            for index, key in enumerate(keys):
                handle.write(
                    json.dumps(
                        {
                            "gt_group_key": key,
                            "molecule_weighted_r1": float(accuracy[index]),
                            "spectra": int(spectra[index]),
                            "nearest_train_raw": float(nearest_raw[index]),
                            "nearest_train_projected": float(nearest_projected[index]),
                            "top5_train_raw": float(top5_raw[index]),
                            "s_max_projected": float(s_max_projected[index]),
                            "mean_parent_mass": float(np.mean(parent_mass[key])),
                            "mean_num_peaks": float(np.mean(peak_counts[key])),
                        }
                    )
                    + "\n"
                )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
