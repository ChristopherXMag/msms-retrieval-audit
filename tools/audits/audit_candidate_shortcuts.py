#!/usr/bin/env python3
"""Audit spectrum-blind shortcuts in the locked MassSpecGym v1.5 candidate pools.

The provided candidate index stores the ground truth in slot zero so that the
Stage1 loss can use a fixed label.  Slot zero is therefore *never* a valid
candidate-only baseline.  All score-based metrics in this script break ties in
expectation, so an all-equal score obtains the random baseline rather than an
artificial 100% R@1 from stable sorting.

The learned baselines see molecular descriptors and train-split candidate
occurrence counts only.  They never receive spectra, DreaMS embeddings, Run K
scores/ranks, query metadata, candidate slot, validation labels, or validation
candidate statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from glmr.candidate_indices import CandidateIndexStore


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = (
    ROOT
    / "outputs/run_k_full_train/candidate_index_full_v15_mass_cap256_full_order/manifest.json"
)
DEFAULT_SMILES = ROOT / "outputs/run_k_full_train/full_sparse/full_smiles.txt"
DEFAULT_TRAIN = ROOT / "outputs/msalign_formula_group_split_v15/train.metadata.jsonl"
DEFAULT_VAL = ROOT / "outputs/msalign_formula_group_split_v15/val.metadata.jsonl"
DEFAULT_OUT = ROOT / "outputs/candidate_shortcut_audit_20260820"

# Keep the feature list explicit: it is part of the audit protocol.
FEATURE_NAMES = (
    "heavy_atoms",
    "hetero_atoms",
    "carbon_atoms",
    "nitrogen_atoms",
    "oxygen_atoms",
    "sulfur_atoms",
    "phosphorus_atoms",
    "halogen_atoms",
    "other_heavy_atoms",
    "aromatic_atoms",
    "rings",
    "aromatic_rings",
    "chiral_centers",
    "unspecified_chiral_centers",
    "rotatable_bonds",
    "fraction_csp3",
    "tpsa",
    "logp",
    "h_bond_donors",
    "h_bond_acceptors",
    "formal_charge_abs",
    "exact_mw",
    "canonical_smiles_length",
    "stereo_marker_count",
)


@dataclass
class Pool:
    gt_row: int
    gt_smiles: str
    connectivity: str
    candidates: np.ndarray
    query_weight: int
    identifiers: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-metadata", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--candidate-index-manifest", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--smiles", type=Path, default=DEFAULT_SMILES)
    parser.add_argument("--candidate-pool-limit", type=int, default=256)
    parser.add_argument(
        "--train-negative-exclusion-manifest",
        type=Path,
        default=None,
        help=(
            "Optional split-safe manifest whose molecule rows are removed from non-GT "
            "training candidates. Validation candidates are never filtered."
        ),
    )
    parser.add_argument(
        "--train-pools",
        type=int,
        default=2048,
        help="Stable train-only pool sample used for descriptor direction/model fitting; 0 means all.",
    )
    parser.add_argument(
        "--val-pools",
        type=int,
        default=0,
        help="Stable validation pool sample; 0 means every unique validation GT pool.",
    )
    parser.add_argument("--negatives-per-train", type=int, default=31)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Descriptor workers; keep bounded because this server's swap is already occupied.",
    )
    parser.add_argument("--descriptor-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--reuse-descriptor-cache",
        type=Path,
        default=None,
        help="Reuse a completed candidate_descriptors.npz after an interrupted summary pass.",
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_metadata(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            identifier = record.get("identifier")
            if identifier is None:
                raise ValueError(f"Metadata row has no identifier: {path}")
            gt_smiles = record.get("canonical_isomeric_smiles")
            if not gt_smiles:
                raise ValueError(f"Metadata row has no canonical_isomeric_smiles: {identifier}")
            records.append((str(identifier), str(gt_smiles)))
    return records


def connectivity_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid GT SMILES: {smiles}")
    return str(Chem.MolToInchiKey(mol)).split("-", 1)[0]


def load_negative_exclusion(
    manifest_path: Path, candidate_manifest: Path, molecule_rows: int
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "glmr_candidate_negative_exclusion_v1":
        raise ValueError(f"Unsupported negative exclusion manifest: {manifest.get('format')}")
    expected_manifest_sha = str(manifest.get("candidate_index_manifest_sha256", ""))
    observed_manifest_sha = file_sha256(candidate_manifest)
    if expected_manifest_sha != observed_manifest_sha:
        raise ValueError(
            "Negative exclusion/candidate manifest mismatch: "
            f"{expected_manifest_sha} != {observed_manifest_sha}"
        )
    indices_path = Path(str(manifest["indices"]))
    if not indices_path.is_absolute():
        indices_path = manifest_path.parent / indices_path
    if file_sha256(indices_path) != str(manifest.get("indices_sha256", "")):
        raise ValueError("Negative exclusion indices SHA256 mismatch")
    rows = np.asarray(np.load(indices_path, allow_pickle=False), dtype=np.int64)
    if len(rows) != int(manifest.get("excluded_rows", -1)):
        raise ValueError("Negative exclusion row count mismatch")
    if len(np.unique(rows)) != len(rows) or bool(np.any(rows < 0)) or bool(np.any(rows >= molecule_rows)):
        raise ValueError("Negative exclusion rows are duplicated or out of range")
    mask = np.zeros(molecule_rows, dtype=bool)
    mask[rows] = True
    return mask, {
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "indices": str(indices_path),
        "indices_sha256": file_sha256(indices_path),
        "excluded_rows": len(rows),
        "scope": manifest.get("scope"),
        "identity_definition": manifest.get("identity_definition"),
    }


def stable_key(value: int, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()


def stable_subset(pools: list[Pool], limit: int, seed: int) -> list[Pool]:
    if limit <= 0 or limit >= len(pools):
        return pools
    return sorted(pools, key=lambda pool: stable_key(pool.gt_row, seed))[:limit]


def build_unique_pools(
    records: Iterable[tuple[str, str]],
    store: CandidateIndexStore,
    limit: int,
    negative_exclusion_mask: np.ndarray | None = None,
) -> tuple[list[Pool], dict[str, Any]]:
    by_signature: dict[tuple[int, bytes], Pool] = {}
    signatures_by_gt: dict[int, set[bytes]] = {}
    stats: Counter[str] = Counter()
    pool_sizes: list[int] = []
    mismatch_examples: list[dict[str, Any]] = []
    for identifier, gt_smiles in records:
        stats["metadata_rows"] += 1
        row = store.row_for(identifier)
        if row is None:
            stats["missing_candidate_rows"] += 1
            continue
        start = int(store.indptr[row])
        end = int(store.indptr[row + 1])
        if limit > 0:
            end = min(end, start + limit)
        candidates = np.asarray(store.indices[start:end], dtype=np.int64)
        if negative_exclusion_mask is not None and len(candidates):
            if bool(negative_exclusion_mask[int(candidates[0])]):
                stats["gt_rows_in_negative_exclusion"] += 1
            remove = negative_exclusion_mask[candidates[1:]]
            removed = int(np.sum(remove))
            if removed:
                stats["candidate_rows_affected_by_exclusion"] += 1
                stats["negative_candidates_excluded"] += removed
                candidates = np.concatenate((candidates[:1], candidates[1:][~remove]))
        if len(candidates) < 2:
            stats["pool_too_small"] += 1
            if negative_exclusion_mask is not None:
                stats["post_exclusion_pool_too_small"] += 1
            continue
        pool_sizes.append(len(candidates))
        if len(np.unique(candidates)) != len(candidates):
            stats["pools_with_duplicate_rows"] += 1
        gt_row = int(candidates[0])
        signature = hashlib.sha256(candidates.tobytes()).digest()
        gt_signatures = signatures_by_gt.setdefault(gt_row, set())
        if gt_signatures and signature not in gt_signatures:
            stats["same_gt_different_pool"] += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append(
                    {
                        "gt_row": gt_row,
                        "different_identifier": identifier,
                        "different_pool_size": len(candidates),
                        "prior_distinct_pool_count": len(gt_signatures),
                    }
                )
        gt_signatures.add(signature)
        key = (gt_row, signature)
        existing = by_signature.get(key)
        if existing is None:
            by_signature[key] = Pool(
                gt_row=gt_row,
                gt_smiles=gt_smiles,
                connectivity=connectivity_from_smiles(gt_smiles),
                candidates=candidates.copy(),
                query_weight=1,
                identifiers=[identifier],
            )
        else:
            existing.query_weight += 1
            if existing.gt_smiles != gt_smiles:
                stats["same_candidate_pool_different_metadata_gt_smiles"] += 1
            if len(existing.identifiers) < 3:
                existing.identifiers.append(identifier)
            if not np.array_equal(existing.candidates, candidates):
                raise AssertionError("Candidate-pool SHA256 collision")
    values = np.asarray(pool_sizes, dtype=np.int64)
    summary: dict[str, Any] = dict(stats)
    summary.update(
        {
            "unique_gt_rows": len(signatures_by_gt),
            "unique_candidate_pools": len(by_signature),
            "gt_rows_with_multiple_candidate_pools": int(
                sum(len(signatures) > 1 for signatures in signatures_by_gt.values())
            ),
            "same_gt_different_pool_examples": mismatch_examples,
            "pool_size": {
                "min": int(values.min()) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
                "mean": float(values.mean()) if len(values) else None,
                "max": int(values.max()) if len(values) else None,
            },
        }
    )
    return list(by_signature.values()), summary


def verify_gt_smiles_parity(
    pools: list[Pool], smiles_by_row: dict[int, str]
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    matched = 0
    for pool in pools:
        observed = smiles_by_row.get(pool.gt_row)
        if observed == pool.gt_smiles:
            matched += 1
        elif len(mismatches) < 10:
            mismatches.append(
                {
                    "gt_row": pool.gt_row,
                    "metadata_canonical_isomeric_smiles": pool.gt_smiles,
                    "candidate_row_smiles": observed,
                    "identifiers": pool.identifiers,
                }
            )
    return {
        "candidate_pools_checked": len(pools),
        "exact_matches": matched,
        "mismatches": len(pools) - matched,
        "mismatch_examples": mismatches,
        "criterion": (
            "metadata canonical_isomeric_smiles must exactly equal full_smiles[candidate_indices[0]]"
        ),
    }


def descriptor(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
    elements = Counter(atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    heavy = int(mol.GetNumHeavyAtoms())
    carbon = int(elements.get("C", 0))
    nitrogen = int(elements.get("N", 0))
    oxygen = int(elements.get("O", 0))
    sulfur = int(elements.get("S", 0))
    phosphorus = int(elements.get("P", 0))
    halogens = sum(int(elements.get(symbol, 0)) for symbol in ("F", "Cl", "Br", "I"))
    known = carbon + nitrogen + oxygen + sulfur + phosphorus + halogens
    values = (
        heavy,
        heavy - carbon,
        carbon,
        nitrogen,
        oxygen,
        sulfur,
        phosphorus,
        halogens,
        heavy - known,
        sum(int(atom.GetIsAromatic()) for atom in mol.GetAtoms()),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAtomStereoCenters(mol),
        rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        abs(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms())),
        Descriptors.ExactMolWt(mol),
        len(smiles),
        smiles.count("@"),
    )
    return np.asarray(values, dtype=np.float32)


def descriptor_chunk(chunk: list[tuple[int, str]]) -> tuple[np.ndarray, np.ndarray]:
    RDLogger.DisableLog("rdApp.*")
    rows = np.asarray([row for row, _smiles in chunk], dtype=np.int64)
    values = np.stack([descriptor(smiles) for _row, smiles in chunk])
    return rows, values


def chunks(values: list[tuple[int, str]], size: int) -> Iterable[list[tuple[int, str]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_selected_smiles(path: Path, selected_rows: np.ndarray) -> list[tuple[int, str]]:
    wanted = np.asarray(selected_rows, dtype=np.int64)
    result: list[tuple[int, str]] = []
    cursor = 0
    with path.open(encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            while cursor < len(wanted) and int(wanted[cursor]) < row:
                raise ValueError(f"SMILES row {int(wanted[cursor])} was not found in {path}")
            if cursor >= len(wanted):
                break
            if int(wanted[cursor]) == row:
                result.append((row, line.rstrip("\n")))
                cursor += 1
    if cursor != len(wanted):
        raise ValueError(f"Only found {cursor}/{len(wanted)} selected SMILES rows")
    return result


def compute_descriptors(
    selected: list[tuple[int, str]], workers: int, chunk_size: int
) -> tuple[np.ndarray, np.ndarray]:
    rows_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    work = list(chunks(selected, chunk_size))
    if workers <= 1:
        iterator = map(descriptor_chunk, work)
        for rows, values in tqdm(iterator, total=len(work), desc="Computing RDKit descriptors"):
            rows_parts.append(rows)
            value_parts.append(values)
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            iterator = pool.imap(descriptor_chunk, work, chunksize=1)
            for rows, values in tqdm(iterator, total=len(work), desc="Computing RDKit descriptors"):
                rows_parts.append(rows)
                value_parts.append(values)
    rows = np.concatenate(rows_parts)
    values = np.concatenate(value_parts)
    order = np.argsort(rows)
    return rows[order], values[order]


def expected_tie_contributions(scores: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return expected R@1/R@5/R@20/MRR under uniform random tie breaking."""
    gt = float(scores[0])
    greater = int(np.sum(scores > gt))
    equal = int(np.sum(scores == gt))
    if equal <= 0:
        raise AssertionError("GT must be tied with itself")
    out = []
    for k in (1, 5, 20):
        out.append(float(np.clip((k - greater) / equal, 0.0, 1.0)))
    reciprocal = sum(1.0 / rank for rank in range(greater + 1, greater + equal + 1)) / equal
    out.append(reciprocal)
    return np.asarray(out, dtype=np.float64), greater, equal


def descriptor_rule_metrics(
    pools: list[Pool], lookup: np.ndarray, values: np.ndarray, query_weighted: bool
) -> dict[str, dict[str, dict[str, Any]]]:
    """Evaluate every scalar descriptor/direction in one candidate-pool pass."""
    totals = {
        "descending": np.zeros((len(FEATURE_NAMES), 4), dtype=np.float64),
        "ascending": np.zeros((len(FEATURE_NAMES), 4), dtype=np.float64),
    }
    denominator = 0.0
    tie_total = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    tie_gt_one = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    harmonic = np.zeros(args_max := 257, dtype=np.float64)
    harmonic[1:] = np.cumsum(1.0 / np.arange(1, args_max, dtype=np.float64))
    for pool in tqdm(pools, desc="Evaluating all scalar candidate-only rules"):
        matrix = pool_features(pool.candidates, lookup, values).astype(np.float64, copy=False)
        gt = matrix[0]
        equal = np.sum(matrix == gt[None, :], axis=0).astype(np.int64)
        greater_desc = np.sum(matrix > gt[None, :], axis=0).astype(np.int64)
        greater_asc = np.sum(matrix < gt[None, :], axis=0).astype(np.int64)
        weight = float(pool.query_weight if query_weighted else 1)
        denominator += weight
        tie_total += weight * equal
        tie_gt_one += weight * (equal > 1)
        for direction, greater in (
            ("descending", greater_desc),
            ("ascending", greater_asc),
        ):
            contribution = np.empty((len(FEATURE_NAMES), 4), dtype=np.float64)
            for metric_index, k in enumerate((1, 5, 20)):
                contribution[:, metric_index] = np.clip((k - greater) / equal, 0.0, 1.0)
            contribution[:, 3] = (harmonic[greater + equal] - harmonic[greater]) / equal
            totals[direction] += weight * contribution
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for feature_index, name in enumerate(FEATURE_NAMES):
        result[name] = {}
        for direction in ("descending", "ascending"):
            metric = 100.0 * totals[direction][feature_index] / max(denominator, 1.0)
            result[name][direction] = {
                "pools": len(pools),
                "effective_queries": int(sum(pool.query_weight for pool in pools))
                if query_weighted
                else len(pools),
                "r1_pct": float(metric[0]),
                "r5_pct": float(metric[1]),
                "r20_pct": float(metric[2]),
                "mrr_pct": float(metric[3]),
                "gt_tie_size_mean": float(tie_total[feature_index] / max(denominator, 1.0)),
                "gt_tie_fraction": float(tie_gt_one[feature_index] / max(denominator, 1.0)),
                "tie_policy": "uniform_random_expected",
            }
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_scores(
    pools: list[Pool], score_fn: Any, query_weighted: bool
) -> tuple[dict[str, Any], np.ndarray]:
    total = np.zeros(4, dtype=np.float64)
    denominator = 0.0
    r1_contributions = np.empty(len(pools), dtype=np.float64)
    gt_tie_sizes: list[int] = []
    for idx, pool in enumerate(pools):
        scores = np.asarray(score_fn(pool.candidates), dtype=np.float64)
        if scores.shape != (len(pool.candidates),):
            raise ValueError(f"Score shape {scores.shape} does not match pool size {len(pool.candidates)}")
        contribution, _greater, equal = expected_tie_contributions(scores)
        weight = float(pool.query_weight if query_weighted else 1)
        total += weight * contribution
        denominator += weight
        r1_contributions[idx] = contribution[0]
        gt_tie_sizes.append(equal)
    values = 100.0 * total / max(denominator, 1.0)
    return (
        {
            "pools": len(pools),
            "effective_queries": int(sum(pool.query_weight for pool in pools))
            if query_weighted
            else len(pools),
            "r1_pct": float(values[0]),
            "r5_pct": float(values[1]),
            "r20_pct": float(values[2]),
            "mrr_pct": float(values[3]),
            "gt_tie_size_mean": float(np.mean(gt_tie_sizes)),
            "gt_tie_size_median": float(np.median(gt_tie_sizes)),
            "gt_tie_fraction": float(np.mean(np.asarray(gt_tie_sizes) > 1)),
            "tie_policy": "uniform_random_expected",
        },
        r1_contributions,
    )


def metric_values_from_contributions(
    contributions: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, float]:
    if len(contributions) == 0:
        return {name: float("nan") for name in ("r1_pct", "r5_pct", "r20_pct", "mrr_pct")}
    values = np.average(contributions, axis=0, weights=weights) if weights is not None else np.mean(contributions, axis=0)
    return {
        name: float(100.0 * value)
        for name, value in zip(("r1_pct", "r5_pct", "r20_pct", "mrr_pct"), values, strict=True)
    }


def candidate_count_and_connectivity_breakdown(
    pools: list[Pool], score_fn: Any
) -> dict[str, Any]:
    contributions = []
    for pool in pools:
        scores = np.asarray(score_fn(pool.candidates), dtype=np.float64)
        contribution, _greater, _equal = expected_tie_contributions(scores)
        contributions.append(contribution)
    matrix = np.stack(contributions)
    query_weights = np.asarray([pool.query_weight for pool in pools], dtype=np.float64)

    by_connectivity: dict[str, list[int]] = defaultdict(list)
    for index, pool in enumerate(pools):
        by_connectivity[pool.connectivity].append(index)
    connectivity_values = []
    for indices in by_connectivity.values():
        local = np.asarray(indices, dtype=np.int64)
        connectivity_values.append(np.average(matrix[local], axis=0, weights=query_weights[local]))

    def count_bin(size: int) -> str:
        if size <= 32:
            return "2-32"
        if size <= 64:
            return "33-64"
        if size <= 128:
            return "65-128"
        if size < 256:
            return "129-255"
        return "256"

    strata: dict[str, Any] = {}
    for label in ("2-32", "33-64", "65-128", "129-255", "256"):
        indices = np.asarray(
            [index for index, pool in enumerate(pools) if count_bin(len(pool.candidates)) == label],
            dtype=np.int64,
        )
        if not len(indices):
            continue
        strata[label] = {
            "unique_pools": int(len(indices)),
            "queries": int(np.sum(query_weights[indices])),
            **metric_values_from_contributions(matrix[indices], query_weights[indices]),
        }
    return {
        "query_weighted": metric_values_from_contributions(matrix, query_weights),
        "connectivity_balanced": {
            "connectivity_groups": len(by_connectivity),
            **metric_values_from_contributions(np.stack(connectivity_values)),
        },
        "candidate_count_strata_query_weighted": strata,
        "candidate_recall_at_pool_cap_pct": 100.0,
        "candidate_recall_denominator_queries": int(np.sum(query_weights)),
        "note": "GT is verified at slot zero before order-free scoring; no missing GT is inserted.",
    }


def weighted_bootstrap_r1(
    contributions: np.ndarray, pools: list[Pool], samples: int, seed: int
) -> dict[str, float]:
    if len(pools) == 0:
        return {}
    weights = np.asarray([pool.query_weight for pool in pools], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 128):
        count = min(128, samples - start)
        indices = rng.integers(0, len(pools), size=(count, len(pools)))
        sampled_weights = weights[indices]
        draws[start : start + count] = (
            100.0
            * np.sum(contributions[indices] * sampled_weights, axis=1)
            / np.sum(sampled_weights, axis=1)
        )
    return {
        "cluster_unit": "unique_(GT_molecule_row,candidate_index_signature)_pool",
        "samples": samples,
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
    }


def metric_pair(pools: list[Pool], score_fn: Any) -> tuple[dict[str, Any], np.ndarray]:
    unique, _ = evaluate_scores(pools, score_fn, query_weighted=False)
    query, contributions = evaluate_scores(pools, score_fn, query_weighted=True)
    return {"unique_pool_weighted": unique, "query_weighted": query}, contributions


def make_descriptor_lookup(
    rows: np.ndarray, values: np.ndarray, molecule_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    if bool(np.isnan(values).any()):
        bad = rows[np.any(np.isnan(values), axis=1)]
        raise ValueError(f"Invalid RDKit descriptors for {len(bad)} candidate rows; first={bad[:5].tolist()}")
    lookup = np.full(molecule_rows, -1, dtype=np.int32)
    lookup[rows] = np.arange(len(rows), dtype=np.int32)
    return lookup, values


def pool_features(indices: np.ndarray, lookup: np.ndarray, values: np.ndarray) -> np.ndarray:
    local = lookup[indices]
    if bool(np.any(local < 0)):
        missing = indices[local < 0][:5]
        raise KeyError(f"Descriptors were not computed for candidate rows {missing.tolist()}")
    return values[local]


def training_examples(
    pools: list[Pool],
    lookup: np.ndarray,
    descriptors: np.ndarray,
    occurrence: np.ndarray,
    negatives_per_pool: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    x_rows: list[np.ndarray] = []
    y: list[int] = []
    for pool in pools:
        negatives = list(range(1, len(pool.candidates)))
        if negatives_per_pool > 0 and len(negatives) > negatives_per_pool:
            negatives = rng.sample(negatives, negatives_per_pool)
        selected = np.asarray([0, *negatives], dtype=np.int64)
        rows = pool.candidates[selected]
        features = pool_features(rows, lookup, descriptors)
        freq = np.log1p(occurrence[rows]).astype(np.float32)[:, None]
        x_rows.append(np.concatenate([features, freq], axis=1))
        y.extend([1] + [0] * len(negatives))
    return np.concatenate(x_rows).astype(np.float32), np.asarray(y, dtype=np.int8)


def model_score_fn(
    model: Any, lookup: np.ndarray, descriptors: np.ndarray, occurrence: np.ndarray
) -> Any:
    def score(indices: np.ndarray) -> np.ndarray:
        features = pool_features(indices, lookup, descriptors)
        freq = np.log1p(occurrence[indices]).astype(np.float32)[:, None]
        matrix = np.concatenate([features, freq], axis=1)
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)

    return score


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    existing = set(args.out_dir.iterdir()) if args.out_dir.exists() else set()
    if existing and args.reuse_descriptor_cache is None:
        raise FileExistsError(f"Refusing to overwrite non-empty audit directory: {args.out_dir}")
    if existing and existing != {args.reuse_descriptor_cache}:
        raise FileExistsError(
            "Descriptor-resume directory contains unexpected files: "
            f"{sorted(str(path) for path in existing)}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    store = CandidateIndexStore(args.candidate_index_manifest)
    train_records = load_metadata(args.train_metadata)
    val_records = load_metadata(args.val_metadata)
    exclusion_mask = None
    exclusion_provenance = None
    if args.train_negative_exclusion_manifest is not None:
        exclusion_mask, exclusion_provenance = load_negative_exclusion(
            args.train_negative_exclusion_manifest,
            args.candidate_index_manifest,
            store.molecule_rows,
        )
    train_all, train_structure = build_unique_pools(
        train_records, store, args.candidate_pool_limit, exclusion_mask
    )
    val_all, val_structure = build_unique_pools(val_records, store, args.candidate_pool_limit)
    if int(train_structure.get("gt_rows_in_negative_exclusion", 0)):
        raise ValueError("A training GT row appears in the held-out negative exclusion denylist")
    inconsistent_gt_smiles = int(
        train_structure.get("same_candidate_pool_different_metadata_gt_smiles", 0)
    ) + int(val_structure.get("same_candidate_pool_different_metadata_gt_smiles", 0))
    if inconsistent_gt_smiles:
        raise ValueError(
            "The same candidate pool is associated with different metadata GT SMILES: "
            f"{inconsistent_gt_smiles} rows"
        )
    train = stable_subset(train_all, args.train_pools, args.seed)
    val = stable_subset(val_all, args.val_pools, args.seed + 1)

    # The occurrence prior is calculated from every unique train GT pool, not
    # from validation and not merely from the descriptor-fitting sample.
    train_occurrence = np.zeros(store.molecule_rows, dtype=np.int32)
    train_gt_occurrence = np.zeros(store.molecule_rows, dtype=np.int32)
    for pool in tqdm(train_all, desc="Counting train-only candidate occurrence"):
        np.add.at(train_occurrence, pool.candidates, 1)
        train_gt_occurrence[pool.gt_row] += 1

    selected_rows = np.unique(
        np.concatenate(
            [
                *(pool.candidates for pool in train),
                *(pool.candidates for pool in val),
                np.asarray([pool.gt_row for pool in train_all], dtype=np.int64),
                np.asarray([pool.gt_row for pool in val_all], dtype=np.int64),
            ]
        )
    )
    gt_rows = np.unique(
        np.asarray([pool.gt_row for pool in [*train_all, *val_all]], dtype=np.int64)
    )
    smiles_by_row = dict(load_selected_smiles(args.smiles, gt_rows))
    train_gt_parity = verify_gt_smiles_parity(train_all, smiles_by_row)
    val_gt_parity = verify_gt_smiles_parity(val_all, smiles_by_row)
    if train_gt_parity["mismatches"] or val_gt_parity["mismatches"]:
        raise ValueError(
            "Candidate slot-zero/metadata GT parity failed: "
            f"train={train_gt_parity['mismatches']} val={val_gt_parity['mismatches']}"
        )
    if args.reuse_descriptor_cache is None:
        selected_smiles = load_selected_smiles(args.smiles, selected_rows)
        descriptor_rows, descriptor_values = compute_descriptors(
            selected_smiles, args.workers, args.descriptor_chunk_size
        )
        np.savez_compressed(
            args.out_dir / "candidate_descriptors.npz",
            rows=descriptor_rows,
            values=descriptor_values,
            feature_names=np.asarray(FEATURE_NAMES),
        )
    else:
        cache = np.load(args.reuse_descriptor_cache, allow_pickle=False)
        descriptor_rows = np.asarray(cache["rows"], dtype=np.int64)
        descriptor_values = np.asarray(cache["values"], dtype=np.float32)
        cached_names = tuple(str(value) for value in cache["feature_names"])
        if cached_names != FEATURE_NAMES:
            raise ValueError(f"Descriptor feature mismatch: {cached_names} != {FEATURE_NAMES}")
        if not np.array_equal(descriptor_rows, selected_rows):
            raise ValueError("Reused descriptor row set does not match the current audit selection")
    lookup, descriptor_values = make_descriptor_lookup(
        descriptor_rows, descriptor_values, store.molecule_rows
    )

    baselines: dict[str, Any] = {}
    random_metrics, random_contributions = metric_pair(
        val, lambda indices: np.zeros(len(indices), dtype=np.float64)
    )
    random_metrics["notes"] = "All candidates tied; metrics are the exact random-tie expectation."
    baselines["random_tie_safe"] = random_metrics

    # This deliberately exposes why slot zero must never be used as evidence.
    slot_metrics, _ = metric_pair(
        val, lambda indices: -np.arange(len(indices), dtype=np.float64)
    )
    slot_metrics["valid_baseline"] = False
    slot_metrics["notes"] = (
        "Candidate construction forces GT into slot zero. This 100% reference is label leakage; "
        "Run K does not receive the slot as a feature."
    )
    baselines["unsafe_forced_slot_zero"] = slot_metrics

    occurrence_metrics, occurrence_contributions = metric_pair(
        val, lambda indices: np.log1p(train_occurrence[indices]).astype(np.float64)
    )
    occurrence_metrics["notes"] = "Descending global candidate occurrence in unique train GT pools only."
    baselines["train_candidate_occurrence_desc"] = occurrence_metrics

    rule_results: dict[str, Any] = {}
    train_rule_metrics = descriptor_rule_metrics(train, lookup, descriptor_values, False)
    val_rule_unique = descriptor_rule_metrics(val, lookup, descriptor_values, False)
    val_rule_query = descriptor_rule_metrics(val, lookup, descriptor_values, True)
    for feature_index, name in enumerate(FEATURE_NAMES):
        def raw_score(indices: np.ndarray, column: int = feature_index) -> np.ndarray:
            return pool_features(indices, lookup, descriptor_values)[:, column]

        train_desc = train_rule_metrics[name]["descending"]
        train_asc = train_rule_metrics[name]["ascending"]
        descending = (train_desc["r1_pct"], train_desc["mrr_pct"]) >= (
            train_asc["r1_pct"],
            train_asc["mrr_pct"],
        )
        direction = "descending" if descending else "ascending"
        val_metrics = {
            "unique_pool_weighted": val_rule_unique[name][direction],
            "query_weighted": val_rule_query[name][direction],
        }
        rule_results[name] = {
            "direction_selected_on_train": "descending" if descending else "ascending",
            "train_unique_pool_weighted_selected": train_desc if descending else train_asc,
            "val": val_metrics,
        }
    baselines["single_descriptor_rules"] = rule_results

    x_train, y_train = training_examples(
        train,
        lookup,
        descriptor_values,
        train_occurrence,
        args.negatives_per_train,
        args.seed,
    )
    feature_names_model = [*FEATURE_NAMES, "log1p_train_candidate_occurrence"]
    linear = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            fit_intercept=True,
            max_iter=500,
            random_state=args.seed,
            solver="lbfgs",
        ),
    )
    linear.fit(x_train, y_train)
    linear_metrics, linear_contributions = metric_pair(
        val, model_score_fn(linear, lookup, descriptor_values, train_occurrence)
    )
    logistic = linear.named_steps["logisticregression"]
    linear_metrics["fit"] = {
        "train_examples": len(y_train),
        "train_positive": int(np.sum(y_train == 1)),
        "train_negative": int(np.sum(y_train == 0)),
        "features": feature_names_model,
        "standardized_coefficients": {
            name: float(value)
            for name, value in zip(feature_names_model, logistic.coef_[0], strict=True)
        },
    }
    baselines["train_only_logistic"] = linear_metrics

    hgb = HistGradientBoostingClassifier(
        class_weight="balanced",
        learning_rate=0.08,
        l2_regularization=1.0,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        random_state=args.seed,
    )
    hgb.fit(x_train, y_train)
    hgb_metrics, hgb_contributions = metric_pair(
        val, model_score_fn(hgb, lookup, descriptor_values, train_occurrence)
    )
    hgb_metrics["fit"] = {
        "train_examples": len(y_train),
        "train_positive": int(np.sum(y_train == 1)),
        "train_negative": int(np.sum(y_train == 0)),
        "features": feature_names_model,
        "iterations": int(hgb.n_iter_),
    }
    baselines["train_only_hist_gradient_boosting"] = hgb_metrics

    valid_candidates = {
        "train_candidate_occurrence_desc": (
            occurrence_metrics["query_weighted"]["r1_pct"],
            occurrence_contributions,
        ),
        "train_only_logistic": (
            linear_metrics["query_weighted"]["r1_pct"],
            linear_contributions,
        ),
        "train_only_hist_gradient_boosting": (
            hgb_metrics["query_weighted"]["r1_pct"],
            hgb_contributions,
        ),
    }
    for name, item in rule_results.items():
        score = float(item["val"]["query_weighted"]["r1_pct"])
        # Recompute only for the eventual best rule below if it wins.
        valid_candidates[f"single_rule:{name}"] = (score, None)
    best_name = max(valid_candidates, key=lambda name: valid_candidates[name][0])
    best_score, best_contributions = valid_candidates[best_name]
    if best_contributions is None:
        feature = best_name.split(":", 1)[1]
        feature_index = FEATURE_NAMES.index(feature)
        direction = rule_results[feature]["direction_selected_on_train"]
        sign = 1.0 if direction == "descending" else -1.0
        _metrics, best_contributions = metric_pair(
            val,
            lambda indices: sign
            * pool_features(indices, lookup, descriptor_values)[:, feature_index],
        )

    result = {
        "schema_version": 1,
        "purpose": "candidate-only shortcut audit; no spectrum-derived inputs",
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "train_metadata_sha256": file_sha256(args.train_metadata),
            "val_metadata_sha256": file_sha256(args.val_metadata),
            "candidate_index_manifest_sha256": file_sha256(args.candidate_index_manifest),
            "molecule_smiles_sha256": store.molecule_smiles_sha256,
            "descriptor_cache_reused": args.reuse_descriptor_cache is not None,
            "descriptor_cache_sha256": file_sha256(
                args.reuse_descriptor_cache
                if args.reuse_descriptor_cache is not None
                else args.out_dir / "candidate_descriptors.npz"
            ),
        },
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "candidate_index": store.metadata,
        "train_negative_exclusion": exclusion_provenance,
        "structure_audit": {
            "train": train_structure,
            "val": val_structure,
            "gt_slot_zero_contract": (
                "Candidate-index construction claims GT is slot zero; independently checked below against metadata."
            ),
            "train_gt_slot_zero_smiles_parity": train_gt_parity,
            "val_gt_slot_zero_smiles_parity": val_gt_parity,
            "candidate_slot_allowed_as_model_feature": False,
            "model_scoring_implication": (
                "Any evaluator using stable sorting on exactly tied scores can be inflated by slot zero; "
                "all metrics here use random expected tie breaking."
            ),
        },
        "audit_scope": {
            "train_unique_pools_total": len(train_all),
            "train_unique_pools_descriptor_fit": len(train),
            "val_unique_pools_total": len(val_all),
            "val_unique_pools_evaluated": len(val),
            "val_query_weight": int(sum(pool.query_weight for pool in val)),
            "unique_candidate_rows_described": len(descriptor_rows),
            "train_candidate_occurrence_nonzero_rows": int(np.sum(train_occurrence > 0)),
            "train_gt_occurrence_nonzero_rows": int(np.sum(train_gt_occurrence > 0)),
            "feature_names": list(FEATURE_NAMES),
        },
        "input_exclusions": [
            "spectra/peaks",
            "DreaMS or any spectrum embedding",
            "Run K scores/ranks/embeddings",
            "candidate slot/order",
            "validation labels during fit or direction selection",
            "validation candidate occurrence statistics",
        ],
        "baselines": baselines,
        "reporting_breakdowns": {
            "random_tie_safe": candidate_count_and_connectivity_breakdown(
                val, lambda indices: np.zeros(len(indices), dtype=np.float64)
            ),
            "train_candidate_occurrence_desc": candidate_count_and_connectivity_breakdown(
                val, lambda indices: np.log1p(train_occurrence[indices]).astype(np.float64)
            ),
            "train_only_logistic": candidate_count_and_connectivity_breakdown(
                val, model_score_fn(linear, lookup, descriptor_values, train_occurrence)
            ),
            "train_only_hist_gradient_boosting": candidate_count_and_connectivity_breakdown(
                val, model_score_fn(hgb, lookup, descriptor_values, train_occurrence)
            ),
        },
        "best_valid_candidate_only_by_val_r1": {
            "name": best_name,
            "query_weighted_r1_pct": float(best_score),
            "cluster_bootstrap_query_weighted_r1": weighted_bootstrap_r1(
                best_contributions, val, samples=2000, seed=args.seed + 10
            ),
            "selection_note": (
                "This max is deliberately adversarial and selected after seeing val across audit baselines; "
                "it is a shortcut stress-test upper signal, not a deployable model estimate."
            ),
        },
        "random_query_weighted_r1_pct": random_metrics["query_weighted"]["r1_pct"],
        "interpretation_guardrail": (
            "These small descriptor models are not a same-capacity Run-K-fingerprint adversary. A low "
            "candidate-only score is necessary but not sufficient to establish spectrum use. Mass/adduct-matched "
            "spectrum permutation and channel-isolated reranker permutation remain required."
        ),
    }
    out_json = args.out_dir / "audit.json"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out_json": str(out_json),
                "train_pools": len(train),
                "val_pools": len(val),
                "val_query_weight": sum(pool.query_weight for pool in val),
                "candidate_rows_described": len(descriptor_rows),
                "random_r1_pct": random_metrics["query_weighted"]["r1_pct"],
                "candidate_occurrence_r1_pct": occurrence_metrics["query_weighted"]["r1_pct"],
                "logistic_r1_pct": linear_metrics["query_weighted"]["r1_pct"],
                "hist_gradient_boosting_r1_pct": hgb_metrics["query_weighted"]["r1_pct"],
                "best_candidate_only": result["best_valid_candidate_only_by_val_r1"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
