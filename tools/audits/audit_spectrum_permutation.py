"""Audit whether locked Run-K retrieval depends on the query MS/MS spectrum.

The primary control permutes DreaMS embeddings only among validation spectra
with the same adduct and the same rounded neutral parent mass, while forbidding
the same RDKit connectivity identity.  Candidate pools stay attached to the
recipient query.  Donors are assigned one-to-one within each stratum, and the
assignment minimizes peak-count mismatch with a small seeded perturbation.

Unlike the historical evaluator, this audit physically randomizes candidate
order and never assumes that the ground truth is at position zero.  Exact score
ties are summarized with optimistic, pessimistic, and uniform-tie expectations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from glmr.candidate_indices import CandidateIndexStore  # noqa: E402
from tools.stage1.train_msalign_projection_smoke import (  # noqa: E402
    MoleculeEmbeddingStore,
    ProjectionDataset,
    build_projection_model,
    checkpoint_model_config,
    choose_device,
    load_dreams,
    load_jsonl,
)

try:
    from rdkit import Chem  # type: ignore  # noqa: E402
except ImportError:  # pragma: no cover
    Chem = None


DEFAULT_METADATA = ROOT / "outputs/msalign_formula_group_split_v15/val.metadata.jsonl"
DEFAULT_DREAMS = ROOT / "outputs/msalign_dreams_no_head_formula_group_masscand/val.no_head.npy"
DEFAULT_TRAIN_DREAMS = (
    ROOT / "outputs/msalign_dreams_no_head_formula_group_masscand/train.no_head.npy"
)
DEFAULT_CHECKPOINT = (
    ROOT / "outputs/run_k_full_train/experiments/all_row/seed20260621/checkpoint.pt"
)
DEFAULT_FINGERPRINTS = ROOT / "outputs/run_k_full_train/full_sparse/manifest.json"
DEFAULT_CANDIDATES = (
    ROOT
    / "outputs/run_k_full_train/candidate_index_full_v15_mass_cap256_full_order/manifest.json"
)
DEFAULT_REFERENCE_RANKS = (
    ROOT / "outputs/run_k_full_train/experiments/all_row/seed20260621/val_ranks.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/shortcut_audit/run_k_spectrum_permutation_exact_mass.json"


def parse_int_csv(value: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated integers: {value!r}") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("At least one permutation seed is required")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--dreams", type=Path, default=DEFAULT_DREAMS)
    parser.add_argument(
        "--train-dreams-for-mean",
        type=Path,
        default=DEFAULT_TRAIN_DREAMS,
        help="Train-only DreaMS cache whose row mean defines the constant-spectrum prior control.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--fingerprint-cache", type=Path, default=DEFAULT_FINGERPRINTS)
    parser.add_argument("--candidate-index-manifest", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--reference-ranks", type=Path, default=DEFAULT_REFERENCE_RANKS)
    parser.add_argument(
        "--state-key",
        choices=("best_model_state_dict", "model_state_dict"),
        default="best_model_state_dict",
    )
    parser.add_argument("--candidate-pool-limit", type=int, default=256)
    parser.add_argument(
        "--evaluation-scope",
        choices=("exact_mass_permutation", "full_val_constant"),
        default="exact_mass_permutation",
        help=(
            "exact_mass_permutation runs all controls on the strict matched subset; "
            "full_val_constant runs real/train-mean/zero on every validation query."
        ),
    )
    parser.add_argument("--mass-round-decimals", type=int, default=6)
    parser.add_argument("--selection-seed", type=int, default=20260820)
    parser.add_argument(
        "--permutation-seeds", type=parse_int_csv, default=[20260820, 20260821, 20260822]
    )
    parser.add_argument(
        "--assignment-jitter",
        type=float,
        default=0.02,
        help="Seeded noise added to |log1p peak-count difference| in the Hungarian cost.",
    )
    parser.add_argument("--candidate-order-seed", type=int, default=20260820)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--max-eval-queries",
        type=int,
        default=0,
        help="Bounded smoke after mapping construction; 0 evaluates every eligible query.",
    )
    parser.add_argument("--spec-projection-batch-size", type=int, default=2048)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    parser.add_argument("--mapping-only", action="store_true")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--out-npz",
        type=Path,
        default=None,
        help="Compact per-query greater/equal counts; defaults beside --out-json.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rdkit_connectivity(smiles: str, fallback: str) -> str:
    if Chem is None:
        raise RuntimeError("RDKit is required for connectivity-safe spectrum permutation")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return fallback
    try:
        return str(Chem.MolToInchiKey(molecule)).split("-", 1)[0]
    except Exception:
        return fallback


def connectivity_identities(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    cache: dict[str, str] = {}
    identities: list[str] = []
    invalid = 0
    stored_mismatches = 0
    examples: list[dict[str, str]] = []
    for record in records:
        smiles = str(record.get("canonical_isomeric_smiles") or "")
        stored = str(record.get("inchikey") or "").split("-", 1)[0]
        if smiles not in cache:
            computed = rdkit_connectivity(smiles, stored or smiles)
            cache[smiles] = computed
            if computed == (stored or smiles) and not stored and computed == smiles:
                invalid += 1
        computed = cache[smiles]
        identities.append(computed)
        if stored and computed != stored:
            stored_mismatches += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "identifier": str(record.get("identifier")),
                        "stored": stored,
                        "computed": computed,
                        "smiles": smiles,
                    }
                )
    return identities, {
        "unique_smiles": len(cache),
        "connectivity_groups": len(set(identities)),
        "invalid_smiles_fallbacks": invalid,
        "stored_vs_rdkit_mismatch_rows": stored_mismatches,
        "stored_vs_rdkit_mismatch_examples": examples,
    }


def stable_select(values: list[int], count: int, seed: int, records: list[dict[str, Any]]) -> list[int]:
    def key(index: int) -> str:
        identifier = str(records[index].get("identifier"))
        return hashlib.sha256(f"{seed}|{identifier}".encode()).hexdigest()

    return sorted(values, key=key)[:count]


def build_balanced_strata(
    records: list[dict[str, Any]],
    identities: list[str],
    decimals: int,
    selection_seed: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    raw: dict[tuple[str, float], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        key = (
            str(record.get("adduct") or ""),
            round(float(record["parent_mass"]), decimals),
        )
        raw[key].append(index)

    selected_strata: list[list[int]] = []
    raw_eligible_rows = 0
    trimmed_rows = 0
    for key in sorted(raw):
        rows = raw[key]
        by_identity: dict[str, list[int]] = defaultdict(list)
        for index in rows:
            by_identity[identities[index]].append(index)
        if len(by_identity) < 2:
            continue
        raw_eligible_rows += len(rows)
        total = len(rows)
        dominant_identity, dominant_rows = max(
            by_identity.items(), key=lambda item: (len(item[1]), item[0])
        )
        other_count = total - len(dominant_rows)
        if len(dominant_rows) > other_count:
            keep_dominant = stable_select(dominant_rows, other_count, selection_seed, records)
            selected = keep_dominant + [
                index
                for identity, indices in by_identity.items()
                if identity != dominant_identity
                for index in indices
            ]
            trimmed_rows += len(dominant_rows) - len(keep_dominant)
        else:
            selected = list(rows)
        if len(selected) >= 2:
            selected_strata.append(sorted(selected))

    return selected_strata, {
        "all_strata": len(raw),
        "multi_connectivity_strata": len(selected_strata),
        "raw_rows_in_multi_connectivity_strata": raw_eligible_rows,
        "balanced_eligible_rows": sum(map(len, selected_strata)),
        "dominant_identity_rows_trimmed": trimmed_rows,
        "mass_round_decimals": decimals,
    }


def make_assignment(
    strata: list[list[int]],
    records: list[dict[str, Any]],
    identities: list[str],
    seed: int,
    jitter: float,
) -> dict[int, int]:
    rng = np.random.default_rng(seed)
    assignment: dict[int, int] = {}
    for rows in strata:
        peak_values = np.log1p(np.asarray([float(records[index]["num_peaks"]) for index in rows]))
        cost = np.abs(peak_values[:, None] - peak_values[None, :])
        if jitter > 0:
            cost += rng.uniform(0.0, jitter, size=cost.shape)
        identity_values = np.asarray([identities[index] for index in rows], dtype=object)
        invalid = identity_values[:, None] == identity_values[None, :]
        cost[invalid] = 1e9
        recipient_pos, donor_pos = linear_sum_assignment(cost)
        if not np.array_equal(recipient_pos, np.arange(len(rows))):
            raise RuntimeError("Hungarian assignment did not cover every recipient")
        for left, right in zip(recipient_pos.tolist(), donor_pos.tolist(), strict=True):
            recipient = rows[left]
            donor = rows[right]
            if identities[recipient] == identities[donor]:
                raise RuntimeError("Connectivity derangement failed")
            assignment[recipient] = donor
    return assignment


def assignment_audit(
    assignment: dict[int, int], records: list[dict[str, Any]], identities: list[str]
) -> dict[str, Any]:
    recipients = sorted(assignment)
    donors = [assignment[index] for index in recipients]
    mass_deltas = np.asarray(
        [abs(float(records[i]["parent_mass"]) - float(records[j]["parent_mass"])) for i, j in zip(recipients, donors, strict=True)]
    )
    mz_ppm = np.asarray(
        [
            abs(float(records[i]["precursor_mz"]) - float(records[j]["precursor_mz"]))
            / max(abs(float(records[i]["precursor_mz"])), 1e-12)
            * 1e6
            for i, j in zip(recipients, donors, strict=True)
        ]
    )
    peak_log_delta = np.asarray(
        [
            abs(math.log1p(float(records[i]["num_peaks"])) - math.log1p(float(records[j]["num_peaks"])))
            for i, j in zip(recipients, donors, strict=True)
        ]
    )
    return {
        "queries": len(recipients),
        "unique_donors": len(set(donors)),
        "one_to_one": len(set(donors)) == len(donors),
        "closed_permutation": set(donors) == set(recipients),
        "fixed_points": sum(i == j for i, j in zip(recipients, donors, strict=True)),
        "same_connectivity_violations": sum(
            identities[i] == identities[j] for i, j in zip(recipients, donors, strict=True)
        ),
        "same_adduct_violations": sum(
            str(records[i].get("adduct")) != str(records[j].get("adduct"))
            for i, j in zip(recipients, donors, strict=True)
        ),
        "parent_mass_delta_max": float(mass_deltas.max(initial=0.0)),
        "precursor_mz_delta_ppm": quantiles(mz_ppm),
        "abs_log1p_peak_count_delta": quantiles(peak_log_delta),
    }


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "min": float(np.min(values)),
        "median": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def projected_vector_comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    cosine = torch.sum(left.float() * right.float(), dim=1).cpu().numpy()
    l2 = torch.linalg.vector_norm(left.float() - right.float(), dim=1).cpu().numpy()
    exact_equal = torch.all(left == right, dim=1).cpu().numpy()
    return {
        "exact_equal_fraction": float(np.mean(exact_equal)),
        "cosine": quantiles(cosine),
        "l2_delta": quantiles(l2),
    }


def seed_assignment_model_change_audit(
    spec_projected: torch.Tensor,
    compact: dict[str, np.ndarray],
    seeds: Iterable[int],
) -> dict[str, Any]:
    seeds = list(seeds)
    if not seeds:
        return {}
    recipients = torch.from_numpy(compact["recipient_rows"].astype(np.int64))
    output: dict[str, Any] = {"each_seed_vs_real_recipient": {}, "seed_pairs": {}}
    for seed in seeds:
        donors = torch.from_numpy(compact[f"seed{seed}_donor_rows"].astype(np.int64))
        output["each_seed_vs_real_recipient"][str(seed)] = projected_vector_comparison(
            spec_projected[recipients], spec_projected[donors]
        )
    for left_pos, left_seed in enumerate(seeds):
        for right_seed in seeds[left_pos + 1 :]:
            left_donors = compact[f"seed{left_seed}_donor_rows"].astype(np.int64)
            right_donors = compact[f"seed{right_seed}_donor_rows"].astype(np.int64)
            left_greater = compact[f"seed{left_seed}_own_greater"].astype(np.int64)
            right_greater = compact[f"seed{right_seed}_own_greater"].astype(np.int64)
            left_equal = compact[f"seed{left_seed}_own_equal"].astype(np.int64)
            right_equal = compact[f"seed{right_seed}_own_equal"].astype(np.int64)
            left_score = compact[f"seed{left_seed}_own_target_score"].astype(np.float64)
            right_score = compact[f"seed{right_seed}_own_target_score"].astype(np.float64)
            pair_key = f"{left_seed}_vs_{right_seed}"
            output["seed_pairs"][pair_key] = {
                "donor_row_difference_fraction": float(np.mean(left_donors != right_donors)),
                "rank_difference_fraction": float(
                    np.mean((left_greater != right_greater) | (left_equal != right_equal))
                ),
                "r1_decision_disagreement_fraction": float(
                    np.mean((left_greater == 0) != (right_greater == 0))
                ),
                "own_target_score_abs_delta": quantiles(np.abs(left_score - right_score)),
                "projected_donor_vectors": projected_vector_comparison(
                    spec_projected[torch.from_numpy(left_donors)],
                    spec_projected[torch.from_numpy(right_donors)],
                ),
            }
    return output


def load_reference_ranks(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    ranks = metrics.get("ranks") if isinstance(metrics, dict) else None
    return [int(value) for value in ranks] if isinstance(ranks, list) else None


def target_counts(
    scores: torch.Tensor, mask: torch.Tensor, target_positions: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    rows = torch.arange(scores.shape[0], device=scores.device)
    target_scores = scores[rows, target_positions]
    greater = ((scores > target_scores[:, None]) & mask).sum(dim=1)
    equal = ((scores == target_scores[:, None]) & mask).sum(dim=1)
    if bool((equal < 1).any()):
        raise RuntimeError("Target candidate was not included in its own tie count")
    return greater.cpu().numpy().astype(np.int64), equal.cpu().numpy().astype(np.int64)


def selected_target_scores(scores: torch.Tensor, target_positions: torch.Tensor) -> np.ndarray:
    rows = torch.arange(scores.shape[0], device=scores.device)
    return scores[rows, target_positions].float().cpu().numpy()


def target_vs_best_other_margin(
    scores: torch.Tensor, mask: torch.Tensor, target_positions: torch.Tensor
) -> np.ndarray:
    rows = torch.arange(scores.shape[0], device=scores.device)
    target_scores = scores[rows, target_positions]
    other_scores = scores.masked_fill(~mask, float("-inf")).clone()
    other_scores[rows, target_positions] = float("-inf")
    best_other = other_scores.max(dim=1).values
    if not bool(torch.isfinite(best_other).all()):
        raise RuntimeError("Every audited query must contain at least one non-target candidate")
    return (target_scores - best_other).float().cpu().numpy()


def margin_summary(values: np.ndarray) -> dict[str, float]:
    summary = quantiles(values.astype(np.float64, copy=False))
    summary["mean"] = float(np.mean(values)) if values.size else 0.0
    summary["positive_fraction"] = float(np.mean(values > 0)) if values.size else 0.0
    return summary


def expected_tie_metrics(greater: np.ndarray, equal: np.ndarray) -> dict[str, Any]:
    if greater.size == 0:
        return {"queries": 0}
    optimistic_rank = greater + 1
    pessimistic_rank = greater + equal
    expected_rank = greater + (equal + 1.0) / 2.0
    result: dict[str, Any] = {
        "queries": int(greater.size),
        "tie_query_fraction": float(np.mean(equal > 1)),
        "mean_tie_size": float(np.mean(equal)),
        "max_tie_size": int(np.max(equal)),
        "expected_mean_rank": float(np.mean(expected_rank)),
        "optimistic_mean_rank": float(np.mean(optimistic_rank)),
        "pessimistic_mean_rank": float(np.mean(pessimistic_rank)),
    }
    harmonic = np.concatenate(([0.0], np.cumsum(1.0 / np.arange(1, int(np.max(pessimistic_rank)) + 1))))
    reciprocal = (harmonic[pessimistic_rank] - harmonic[greater]) / equal
    result["expected_mrr"] = float(np.mean(reciprocal) * 100.0)
    for cutoff in (1, 5, 20, 50, 100):
        expected_hit = np.clip((cutoff - greater) / equal, 0.0, 1.0)
        result[f"expected_recall@{cutoff}"] = float(np.mean(expected_hit) * 100.0)
        result[f"optimistic_recall@{cutoff}"] = float(np.mean(optimistic_rank <= cutoff) * 100.0)
        result[f"pessimistic_recall@{cutoff}"] = float(np.mean(pessimistic_rank <= cutoff) * 100.0)
    return result


def expected_query_values(greater: np.ndarray, equal: np.ndarray) -> dict[str, np.ndarray]:
    """Per-query expectations under a uniform random ordering within exact score ties."""
    max_rank = int(np.max(greater + equal)) if greater.size else 0
    harmonic = np.concatenate(([0.0], np.cumsum(1.0 / np.arange(1, max_rank + 1))))
    values = {
        f"recall@{cutoff}": np.clip((cutoff - greater) / equal, 0.0, 1.0)
        for cutoff in (1, 5, 20, 50, 100)
    }
    values["mrr"] = (harmonic[greater + equal] - harmonic[greater]) / equal
    return values


def molecule_balanced_metrics(
    greater: np.ndarray, equal: np.ndarray, clusters: list[str]
) -> dict[str, Any]:
    if len(greater) != len(clusters):
        raise ValueError("Molecule-balanced metric inputs have incompatible lengths")
    values = expected_query_values(greater, equal)
    cluster_rows: dict[str, list[int]] = defaultdict(list)
    for row, cluster in enumerate(clusters):
        cluster_rows[cluster].append(row)
    names = sorted(cluster_rows)
    output: dict[str, Any] = {
        "connectivity_groups": len(names),
        "queries": len(clusters),
        "aggregation": "mean within RDKit connectivity, then unweighted mean across connectivities",
    }
    for metric, per_query in values.items():
        output[metric] = float(
            np.mean([np.mean(per_query[cluster_rows[name]]) for name in names]) * 100.0
        )
    return output


def paired_cluster_bootstrap(
    baseline_greater: np.ndarray,
    baseline_equal: np.ndarray,
    control_greater: np.ndarray,
    control_equal: np.ndarray,
    clusters: list[str],
    *,
    resamples: int,
    seed: int,
    delta_direction: str = "real spectrum minus control, percentage points",
) -> dict[str, Any]:
    if not (
        len(baseline_greater)
        == len(baseline_equal)
        == len(control_greater)
        == len(control_equal)
        == len(clusters)
    ):
        raise ValueError("Paired cluster bootstrap inputs have incompatible lengths")
    baseline_values = expected_query_values(baseline_greater, baseline_equal)
    control_values = expected_query_values(control_greater, control_equal)
    cluster_rows: dict[str, list[int]] = defaultdict(list)
    for row, cluster in enumerate(clusters):
        cluster_rows[cluster].append(row)
    cluster_names = sorted(cluster_rows)
    counts = np.asarray([len(cluster_rows[name]) for name in cluster_names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cluster_names), size=(resamples, len(cluster_names)))
    output: dict[str, Any] = {
        "clusters": len(cluster_names),
        "queries": len(clusters),
        "resamples": resamples,
        "unit": "RDKit InChIKey connectivity",
        "delta_direction": delta_direction,
    }
    molecule_balanced: dict[str, Any] = {
        "aggregation": "mean within RDKit connectivity, then unweighted mean across connectivities"
    }
    for metric in ("recall@1", "mrr"):
        paired = baseline_values[metric] - control_values[metric]
        sums = np.asarray(
            [float(np.sum(paired[cluster_rows[name]])) for name in cluster_names],
            dtype=np.float64,
        )
        sampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1) * 100.0
        observed = float(np.mean(paired) * 100.0)
        output[metric] = {
            "delta_pp": observed,
            "ci95": [float(value) for value in np.quantile(sampled, [0.025, 0.975])],
            "bootstrap_mean_pp": float(np.mean(sampled)),
        }
        cluster_means = np.asarray(
            [float(np.mean(paired[cluster_rows[name]])) for name in cluster_names],
            dtype=np.float64,
        )
        sampled_balanced = cluster_means[draws].mean(axis=1) * 100.0
        molecule_balanced[metric] = {
            "delta_pp": float(np.mean(cluster_means) * 100.0),
            "ci95": [
                float(value) for value in np.quantile(sampled_balanced, [0.025, 0.975])
            ],
            "bootstrap_mean_pp": float(np.mean(sampled_balanced)),
        }
    output["molecule_balanced"] = molecule_balanced
    return output


def streaming_dreams_mean(path: Path) -> tuple[torch.Tensor, int]:
    if path.suffix == ".npy":
        values = np.load(path, mmap_mode="r")
        # float32 is the cache's native precision and is amply accurate for this
        # negative control.  NumPy's threaded reduction is much faster here than
        # repeatedly materializing float64 chunks from the mmap.
        mean = np.mean(values, axis=0, dtype=np.float32)
        return torch.from_numpy(mean.copy()).float(), int(len(values))
    values = load_dreams(path)
    return values.mean(dim=0), int(len(values))


def projection_batches(
    model: torch.nn.Module,
    dreams: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in tqdm(range(0, len(dreams), batch_size), desc="Projecting all spectra"):
            projected = model.project_spec(dreams[start : start + batch_size].to(device))
            outputs.append(projected.cpu())
    return torch.cat(outputs, dim=0)


def shuffled_candidates(
    dataset: ProjectionDataset, recipients: list[int], seed: int
) -> tuple[list[list[int]], np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[list[int]] = []
    own_positions: list[int] = []
    duplicates = 0
    original_gt_not_first = 0
    for recipient in recipients:
        indices = list(map(int, dataset[recipient]["candidate_indices"]))
        if not indices:
            raise RuntimeError(f"Empty candidate pool for dataset row {recipient}")
        if len(indices) != len(set(indices)):
            duplicates += 1
        own_gt = indices[0]
        permutation = rng.permutation(len(indices))
        shuffled = [indices[int(pos)] for pos in permutation]
        position = shuffled.index(own_gt)
        original_gt_not_first += int(position != 0)
        rows.append(shuffled)
        own_positions.append(position)
    return rows, np.asarray(own_positions, dtype=np.int64), {
        "queries": len(rows),
        "duplicate_candidate_index_rows": duplicates,
        "gt_not_at_position_zero_after_shuffle": original_gt_not_first,
        "gt_position_quantiles": quantiles(np.asarray(own_positions, dtype=np.float64)),
    }


def evaluate_assignments(
    model: torch.nn.Module,
    mol_store: MoleculeEmbeddingStore,
    dataset: ProjectionDataset,
    spec_projected: torch.Tensor,
    recipients: list[int],
    assignments: dict[int, dict[int, int]],
    constant_spec_projected: dict[str, torch.Tensor],
    candidate_order_seed: int,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    candidate_rows, own_positions, order_audit = shuffled_candidates(
        dataset, recipients, candidate_order_seed
    )
    baseline_greater: list[np.ndarray] = []
    baseline_equal: list[np.ndarray] = []
    baseline_target_scores: list[np.ndarray] = []
    baseline_margins: list[np.ndarray] = []
    constant_values: dict[str, dict[str, list[np.ndarray]]] = {
        name: {
            "greater": [],
            "equal": [],
            "target_scores": [],
            "real_residual_greater": [],
            "real_residual_equal": [],
            "real_residual_target_scores": [],
        }
        for name in constant_spec_projected
    }
    per_seed: dict[int, dict[str, Any]] = {
        seed: {
            "own_greater": [],
            "own_equal": [],
            "donor_greater": [],
            "donor_equal": [],
            "own_target_scores": [],
            "own_margins": [],
            "residuals": {
                name: {"greater": [], "equal": [], "target_scores": []}
                for name in constant_spec_projected
            },
        }
        for seed in assignments
    }

    model.eval()
    with torch.inference_mode():
        for start in tqdm(range(0, len(recipients), batch_size), desc="Scoring real/permuted spectra"):
            stop = min(start + batch_size, len(recipients))
            current_recipients = recipients[start:stop]
            current_candidates = candidate_rows[start:stop]
            max_len = max(map(len, current_candidates))
            padded: list[list[int]] = []
            masks: list[list[bool]] = []
            for indices in current_candidates:
                padded.append(indices + [indices[0]] * (max_len - len(indices)))
                masks.append([True] * len(indices) + [False] * (max_len - len(indices)))
            index_tensor = torch.tensor(padded, dtype=torch.long)
            mask = torch.tensor(masks, dtype=torch.bool, device=device)
            unique, inverse = torch.unique(index_tensor.reshape(-1), sorted=True, return_inverse=True)
            unique, inverse = mol_store.canonicalize_unique_indices(unique, inverse)
            unique_features = mol_store.gather(unique).to(device)
            unique_projected = model.project_mol(unique_features)
            candidate_projected = unique_projected[inverse.to(device)].reshape(
                len(current_recipients), max_len, -1
            )

            own_pos = torch.tensor(own_positions[start:stop], dtype=torch.long, device=device)
            real_spec = spec_projected[current_recipients].to(device)
            real_scores_raw = torch.einsum("bd,bcd->bc", real_spec, candidate_projected)
            real_scores = real_scores_raw.masked_fill(~mask, float("-inf"))
            greater, equal = target_counts(real_scores, mask, own_pos)
            baseline_greater.append(greater)
            baseline_equal.append(equal)
            baseline_target_scores.append(selected_target_scores(real_scores, own_pos))
            baseline_margins.append(target_vs_best_other_margin(real_scores, mask, own_pos))

            constant_score_batches: dict[str, torch.Tensor] = {}
            for name, constant_spec in constant_spec_projected.items():
                constant_scores_raw = torch.einsum(
                    "d,bcd->bc", constant_spec.to(device), candidate_projected
                )
                constant_score_batches[name] = constant_scores_raw
                constant_scores = constant_scores_raw.masked_fill(~mask, float("-inf"))
                greater, equal = target_counts(constant_scores, mask, own_pos)
                constant_values[name]["greater"].append(greater)
                constant_values[name]["equal"].append(equal)
                constant_values[name]["target_scores"].append(
                    selected_target_scores(constant_scores, own_pos)
                )
                residual_scores = (real_scores_raw - constant_scores_raw).masked_fill(
                    ~mask, float("-inf")
                )
                residual_greater, residual_equal = target_counts(
                    residual_scores, mask, own_pos
                )
                constant_values[name]["real_residual_greater"].append(residual_greater)
                constant_values[name]["real_residual_equal"].append(residual_equal)
                constant_values[name]["real_residual_target_scores"].append(
                    selected_target_scores(residual_scores, own_pos)
                )

            for seed, mapping in assignments.items():
                donor_rows = [mapping[index] for index in current_recipients]
                donor_spec = spec_projected[donor_rows].to(device)
                permuted_scores_raw = torch.einsum("bd,bcd->bc", donor_spec, candidate_projected)
                permuted_scores = permuted_scores_raw.masked_fill(~mask, float("-inf"))
                greater, equal = target_counts(permuted_scores, mask, own_pos)
                seed_values = per_seed[seed]
                assert isinstance(seed_values["own_greater"], list)
                assert isinstance(seed_values["own_equal"], list)
                seed_values["own_greater"].append(greater)
                seed_values["own_equal"].append(equal)
                seed_values["own_target_scores"].append(
                    selected_target_scores(permuted_scores, own_pos)
                )
                seed_values["own_margins"].append(
                    target_vs_best_other_margin(permuted_scores, mask, own_pos)
                )
                for name, constant_scores_raw in constant_score_batches.items():
                    residual_scores = (permuted_scores_raw - constant_scores_raw).masked_fill(
                        ~mask, float("-inf")
                    )
                    residual_greater, residual_equal = target_counts(
                        residual_scores, mask, own_pos
                    )
                    seed_values["residuals"][name]["greater"].append(residual_greater)
                    seed_values["residuals"][name]["equal"].append(residual_equal)
                    seed_values["residuals"][name]["target_scores"].append(
                        selected_target_scores(residual_scores, own_pos)
                    )

                donor_positions: list[int] = []
                present_rows: list[int] = []
                for local_row, (recipient, donor) in enumerate(
                    zip(current_recipients, donor_rows, strict=True)
                ):
                    donor_gt = int(dataset[donor]["candidate_indices"][0])
                    try:
                        donor_position = current_candidates[local_row].index(donor_gt)
                    except ValueError:
                        continue
                    donor_positions.append(donor_position)
                    present_rows.append(local_row)
                batch_donor_greater = np.full(len(current_recipients), -1, dtype=np.int64)
                batch_donor_equal = np.full(len(current_recipients), -1, dtype=np.int64)
                if present_rows:
                    row_tensor = torch.tensor(present_rows, dtype=torch.long, device=device)
                    donor_pos_tensor = torch.tensor(donor_positions, dtype=torch.long, device=device)
                    donor_greater, donor_equal = target_counts(
                        permuted_scores[row_tensor], mask[row_tensor], donor_pos_tensor
                    )
                    assert isinstance(seed_values["donor_greater"], list)
                    assert isinstance(seed_values["donor_equal"], list)
                    batch_donor_greater[np.asarray(present_rows)] = donor_greater
                    batch_donor_equal[np.asarray(present_rows)] = donor_equal
                seed_values["donor_greater"].append(batch_donor_greater)
                seed_values["donor_equal"].append(batch_donor_equal)

    baseline_g = np.concatenate(baseline_greater)
    baseline_e = np.concatenate(baseline_equal)
    result: dict[str, Any] = {
        "baseline_real_spectrum_own_target": expected_tie_metrics(baseline_g, baseline_e),
        "constant_spectrum_controls": {},
        "residual_score_controls": {},
        "permutation_seeds": {},
    }
    compact: dict[str, np.ndarray] = {
        "recipient_rows": np.asarray(recipients, dtype=np.int32),
        "baseline_greater": baseline_g.astype(np.int16),
        "baseline_equal": baseline_e.astype(np.int16),
        "baseline_target_score": np.concatenate(baseline_target_scores).astype(np.float32),
        "baseline_target_margin": np.concatenate(baseline_margins).astype(np.float32),
    }
    result["baseline_real_spectrum_own_target"]["target_vs_best_other_margin"] = margin_summary(
        compact["baseline_target_margin"]
    )
    for name, values in constant_values.items():
        greater = np.concatenate(values["greater"])
        equal = np.concatenate(values["equal"])
        residual_greater = np.concatenate(values["real_residual_greater"])
        residual_equal = np.concatenate(values["real_residual_equal"])
        result["constant_spectrum_controls"][name] = expected_tie_metrics(greater, equal)
        result["residual_score_controls"][f"real_minus_{name}"] = expected_tie_metrics(
            residual_greater, residual_equal
        )
        compact[f"{name}_greater"] = greater.astype(np.int16)
        compact[f"{name}_equal"] = equal.astype(np.int16)
        compact[f"{name}_target_score"] = np.concatenate(values["target_scores"]).astype(
            np.float32
        )
        compact[f"real_minus_{name}_greater"] = residual_greater.astype(np.int16)
        compact[f"real_minus_{name}_equal"] = residual_equal.astype(np.int16)
        compact[f"real_minus_{name}_target_score"] = np.concatenate(
            values["real_residual_target_scores"]
        ).astype(np.float32)
    for seed, values in per_seed.items():
        own_g = np.concatenate(values["own_greater"])  # type: ignore[arg-type]
        own_e = np.concatenate(values["own_equal"])  # type: ignore[arg-type]
        donor_g_lists = values["donor_greater"]
        donor_e_lists = values["donor_equal"]
        donor_g_all = np.concatenate(donor_g_lists) if donor_g_lists else np.empty(0, dtype=np.int64)  # type: ignore[arg-type]
        donor_e_all = np.concatenate(donor_e_lists) if donor_e_lists else np.empty(0, dtype=np.int64)  # type: ignore[arg-type]
        donor_present = donor_g_all >= 0
        donor_g = donor_g_all[donor_present]
        donor_e = donor_e_all[donor_present]
        result["permutation_seeds"][str(seed)] = {
            "permuted_spectrum_own_target": expected_tie_metrics(own_g, own_e),
            "permuted_spectrum_donor_target": expected_tie_metrics(donor_g, donor_e),
            "donor_target_candidate_coverage": float(np.mean(donor_present)),
            "residual_score_controls": {},
        }
        compact[f"seed{seed}_donor_rows"] = np.asarray(
            [assignments[seed][index] for index in recipients], dtype=np.int32
        )
        compact[f"seed{seed}_own_greater"] = own_g.astype(np.int16)
        compact[f"seed{seed}_own_equal"] = own_e.astype(np.int16)
        compact[f"seed{seed}_own_target_score"] = np.concatenate(
            values["own_target_scores"]
        ).astype(np.float32)
        compact[f"seed{seed}_own_target_margin"] = np.concatenate(
            values["own_margins"]
        ).astype(np.float32)
        margin_delta = (
            compact["baseline_target_margin"] - compact[f"seed{seed}_own_target_margin"]
        )
        compact[f"seed{seed}_real_minus_wrong_target_margin"] = margin_delta.astype(
            np.float32
        )
        result["permutation_seeds"][str(seed)]["permuted_target_vs_best_other_margin"] = (
            margin_summary(compact[f"seed{seed}_own_target_margin"])
        )
        result["permutation_seeds"][str(seed)]["real_minus_wrong_target_margin"] = (
            margin_summary(margin_delta)
        )
        compact[f"seed{seed}_donor_greater"] = donor_g_all.astype(np.int16)
        compact[f"seed{seed}_donor_equal"] = donor_e_all.astype(np.int16)
        for name, residual_values in values["residuals"].items():
            residual_greater = np.concatenate(residual_values["greater"])
            residual_equal = np.concatenate(residual_values["equal"])
            result["permutation_seeds"][str(seed)]["residual_score_controls"][
                f"permuted_minus_{name}"
            ] = expected_tie_metrics(residual_greater, residual_equal)
            compact[f"seed{seed}_minus_{name}_greater"] = residual_greater.astype(np.int16)
            compact[f"seed{seed}_minus_{name}_equal"] = residual_equal.astype(np.int16)
            compact[f"seed{seed}_minus_{name}_target_score"] = np.concatenate(
                residual_values["target_scores"]
            ).astype(np.float32)
    return result, order_audit, compact


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.metadata)
    dreams = load_dreams(args.dreams)
    if len(records) != len(dreams):
        raise ValueError(f"Metadata/DreaMS mismatch: {len(records)} != {len(dreams)}")
    identities, identity_audit = connectivity_identities(records)
    strata, strata_audit = build_balanced_strata(
        records, identities, args.mass_round_decimals, args.selection_seed
    )
    exact_assignments = {
        seed: make_assignment(strata, records, identities, seed, args.assignment_jitter)
        for seed in args.permutation_seeds
    }
    exact_recipients = sorted(next(iter(exact_assignments.values())))
    for seed, mapping in exact_assignments.items():
        if sorted(mapping) != exact_recipients:
            raise RuntimeError(f"Permutation seed {seed} does not use the common recipient set")
    if args.evaluation_scope == "exact_mass_permutation":
        common_recipients = exact_recipients
        assignments = exact_assignments
    else:
        common_recipients = list(range(len(records)))
        assignments = {}
    if args.max_eval_queries > 0:
        common_recipients = common_recipients[: args.max_eval_queries]
        assignments = {
            seed: {index: mapping[index] for index in common_recipients}
            for seed, mapping in assignments.items()
        }

    candidate_store = CandidateIndexStore(args.candidate_index_manifest)
    dataset = ProjectionDataset(
        records,
        dreams,
        {},
        {},
        args.candidate_pool_limit,
        candidate_index_store=candidate_store,
    )
    if len(dataset) != len(records):
        raise RuntimeError(
            "Permutation audit requires one-to-one record/dataset rows; "
            f"got {len(dataset)} vs {len(records)}"
        )

    output: dict[str, Any] = {
        "schema": "glmr_spectrum_permutation_audit_v1",
        "args": {
            key: (
                [str(item) if isinstance(item, Path) else item for item in value]
                if isinstance(value, list)
                else str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in vars(args).items()
        },
        "protocol": {
            "baseline": (
                "Run K checkpoint supplied by --checkpoint; architecture, split, seed, and "
                "training protocol are read from the checkpoint args and recorded under inputs"
            ),
            "recipient_candidates": {
                "manifest": str(args.candidate_index_manifest),
                "pool_limit": args.candidate_pool_limit,
                "construction": candidate_store.metadata.get("construction", "not recorded"),
            },
            "spectrum_control": (
                "one-to-one DreaMS embedding permutation within same adduct and rounded parent mass; "
                "same RDKit connectivity forbidden"
                if args.evaluation_scope == "exact_mass_permutation"
                else "full-validation real spectrum versus train-only mean and zero constant inputs"
            ),
            "candidate_order": "physically shuffled; target position tracked explicitly",
            "ties": "order-free optimistic/pessimistic/uniform-tie expectation",
            "evaluation_scope": args.evaluation_scope,
            "residual_score_controls": (
                "exploratory diagnostic only: subtract a constant-input candidate score from the "
                "real/permuted score; not a primary shortcut verdict or calibrated retrieval policy"
            ),
        },
        "inputs": {
            "metadata": str(args.metadata),
            "metadata_sha256": file_sha256(args.metadata),
            "dreams": str(args.dreams),
            "dreams_sha256": file_sha256(args.dreams),
            "train_dreams_for_mean": str(args.train_dreams_for_mean),
            "train_dreams_for_mean_sha256": file_sha256(args.train_dreams_for_mean),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "fingerprint_cache": str(args.fingerprint_cache),
            "fingerprint_cache_manifest_sha256": file_sha256(args.fingerprint_cache),
            "candidate_index_manifest": str(args.candidate_index_manifest),
            "candidate_index_manifest_sha256": file_sha256(args.candidate_index_manifest),
            "reference_ranks": str(args.reference_ranks),
            "reference_ranks_sha256": (
                file_sha256(args.reference_ranks) if args.reference_ranks.exists() else None
            ),
            "candidate_pool_limit": args.candidate_pool_limit,
        },
        "identity_audit": identity_audit,
        "strata_audit": strata_audit,
        "evaluated_queries": len(common_recipients),
        "assignment_audits": {
            str(seed): assignment_audit(mapping, records, identities)
            for seed, mapping in assignments.items()
        },
        "dataset": dataset.stats,
        "mapping_only": bool(args.mapping_only),
    }

    if not args.mapping_only:
        device = choose_device(args.device)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        output["inputs"]["checkpoint_args"] = {
            key: (
                [str(item) if isinstance(item, Path) else item for item in value]
                if isinstance(value, list)
                else str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in (checkpoint.get("args") or {}).items()
        }
        config = checkpoint_model_config(checkpoint)
        mol_store = MoleculeEmbeddingStore(
            None,
            str(config.get("mol_pooling", "count_morgan_r3_atom_pair_path_maccs")),
            args.fingerprint_cache,
            preload_sparse_features=True,
            mol_blocks=config.get("mol_blocks"),
        )
        model = build_projection_model(
            spec_dim=int(dreams.shape[1]),
            mol_dim=mol_store.dim,
            mol_block_dims=mol_store.block_dims,
            mol_block_names=mol_store.block_names,
            config=config,
        ).to(device)
        state = checkpoint.get(args.state_key)
        if state is None:
            raise KeyError(f"Checkpoint has no {args.state_key}")
        model.load_state_dict(state)
        projected = projection_batches(
            model, dreams, device, args.spec_projection_batch_size
        )
        train_mean, train_mean_rows = streaming_dreams_mean(args.train_dreams_for_mean)
        with torch.inference_mode():
            constant_spec_projected = {
                "train_mean_input": model.project_spec(train_mean[None].to(device))[0].cpu(),
                "zero_input": model.project_spec(torch.zeros_like(train_mean)[None].to(device))[0].cpu(),
            }
        results, order_audit, compact = evaluate_assignments(
            model,
            mol_store,
            dataset,
            projected,
            common_recipients,
            assignments,
            constant_spec_projected,
            args.candidate_order_seed,
            device,
            args.eval_batch_size,
        )
        output["results"] = results
        output["candidate_order_audit"] = order_audit
        output["constant_spectrum_controls"] = {
            "train_mean_rows": train_mean_rows,
            "train_mean_input_l2": float(torch.linalg.vector_norm(train_mean)),
            "definition": (
                "train_mean_input is the per-dimension mean of train.no_head.npy passed through "
                "the locked Run-K spec_head; zero_input is supplemental"
            ),
        }
        cluster_labels = [identities[index] for index in common_recipients]
        compact["connectivity"] = np.asarray(cluster_labels)
        results["baseline_real_spectrum_own_target"]["molecule_balanced"] = (
            molecule_balanced_metrics(
                compact["baseline_greater"].astype(np.int64),
                compact["baseline_equal"].astype(np.int64),
                cluster_labels,
            )
        )
        for name in constant_spec_projected:
            results["constant_spectrum_controls"][name]["molecule_balanced"] = (
                molecule_balanced_metrics(
                    compact[f"{name}_greater"].astype(np.int64),
                    compact[f"{name}_equal"].astype(np.int64),
                    cluster_labels,
                )
            )
            residual_key = f"real_minus_{name}"
            results["residual_score_controls"][residual_key]["molecule_balanced"] = (
                molecule_balanced_metrics(
                    compact[f"{residual_key}_greater"].astype(np.int64),
                    compact[f"{residual_key}_equal"].astype(np.int64),
                    cluster_labels,
                )
            )
        for seed in assignments:
            seed_result = results["permutation_seeds"][str(seed)]
            seed_result["permuted_spectrum_own_target"]["molecule_balanced"] = (
                molecule_balanced_metrics(
                    compact[f"seed{seed}_own_greater"].astype(np.int64),
                    compact[f"seed{seed}_own_equal"].astype(np.int64),
                    cluster_labels,
                )
            )
            for name in constant_spec_projected:
                residual_key = f"permuted_minus_{name}"
                seed_result["residual_score_controls"][residual_key]["molecule_balanced"] = (
                    molecule_balanced_metrics(
                        compact[f"seed{seed}_minus_{name}_greater"].astype(np.int64),
                        compact[f"seed{seed}_minus_{name}_equal"].astype(np.int64),
                        cluster_labels,
                    )
                )
        output["paired_connectivity_cluster_bootstrap"] = {}
        for seed in assignments:
            output["paired_connectivity_cluster_bootstrap"][str(seed)] = paired_cluster_bootstrap(
                compact["baseline_greater"].astype(np.int64),
                compact["baseline_equal"].astype(np.int64),
                compact[f"seed{seed}_own_greater"].astype(np.int64),
                compact[f"seed{seed}_own_equal"].astype(np.int64),
                cluster_labels,
                resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed + seed,
            )
        for name in constant_spec_projected:
            output["paired_connectivity_cluster_bootstrap"][name] = paired_cluster_bootstrap(
                compact["baseline_greater"].astype(np.int64),
                compact["baseline_equal"].astype(np.int64),
                compact[f"{name}_greater"].astype(np.int64),
                compact[f"{name}_equal"].astype(np.int64),
                cluster_labels,
                resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed + int.from_bytes(name.encode()[:4], "little"),
            )
        output["residual_score_paired_bootstrap"] = {}
        for name in constant_spec_projected:
            real_residual_key = f"real_minus_{name}"
            output["residual_score_paired_bootstrap"][
                f"{real_residual_key}_vs_raw_real"
            ] = paired_cluster_bootstrap(
                compact[f"{real_residual_key}_greater"].astype(np.int64),
                compact[f"{real_residual_key}_equal"].astype(np.int64),
                compact["baseline_greater"].astype(np.int64),
                compact["baseline_equal"].astype(np.int64),
                cluster_labels,
                resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed + int.from_bytes(name.encode()[-4:], "little"),
                delta_direction="candidate-prior residual minus raw real, percentage points",
            )
            for seed in assignments:
                perm_residual_key = f"seed{seed}_minus_{name}"
                output["residual_score_paired_bootstrap"][
                    f"real_minus_{name}_vs_seed{seed}_minus_{name}"
                ] = paired_cluster_bootstrap(
                    compact[f"{real_residual_key}_greater"].astype(np.int64),
                    compact[f"{real_residual_key}_equal"].astype(np.int64),
                    compact[f"{perm_residual_key}_greater"].astype(np.int64),
                    compact[f"{perm_residual_key}_equal"].astype(np.int64),
                    cluster_labels,
                    resamples=args.bootstrap_resamples,
                    seed=args.bootstrap_seed + seed + int.from_bytes(name.encode()[:4], "little"),
                    delta_direction="real residual minus permuted residual, percentage points",
                )
                output["residual_score_paired_bootstrap"][
                    f"seed{seed}_minus_{name}_vs_raw_permuted"
                ] = paired_cluster_bootstrap(
                    compact[f"{perm_residual_key}_greater"].astype(np.int64),
                    compact[f"{perm_residual_key}_equal"].astype(np.int64),
                    compact[f"seed{seed}_own_greater"].astype(np.int64),
                    compact[f"seed{seed}_own_equal"].astype(np.int64),
                    cluster_labels,
                    resamples=args.bootstrap_resamples,
                    seed=args.bootstrap_seed + seed + int.from_bytes(name.encode()[-4:], "little"),
                    delta_direction="permuted residual minus raw permuted, percentage points",
                )
        output["seed_assignment_model_change_audit"] = seed_assignment_model_change_audit(
            projected, compact, assignments.keys()
        )
        out_npz = args.out_npz or args.out_json.with_suffix(".per_query.npz")
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, **compact)
        output["per_query_artifact"] = {
            "path": str(out_npz),
            "schema": (
                "recipient_rows/connectivity; baseline, constant, residual and per-seed "
                "greater/equal counts plus own-target scores; donor_rows; donor target "
                "counts=-1 when absent"
            ),
        }
        reference_ranks = load_reference_ranks(args.reference_ranks)
        if reference_ranks is not None and len(reference_ranks) == len(records):
            reference_subset = np.asarray(reference_ranks, dtype=np.int64)[common_recipients]
            reference_full = np.asarray(reference_ranks, dtype=np.int64)
            baseline = results["baseline_real_spectrum_own_target"]
            output["reference_rank_anchor"] = {
                "path": str(args.reference_ranks),
                "full_val": {
                    "queries": len(reference_full),
                    "recall@1": float(np.mean(reference_full <= 1) * 100.0),
                    "recall@5": float(np.mean(reference_full <= 5) * 100.0),
                    "mean_rank": float(np.mean(reference_full)),
                },
                "evaluated_scope": {
                    "scope": args.evaluation_scope,
                    "queries": len(reference_subset),
                    "recall@1": float(np.mean(reference_subset <= 1) * 100.0),
                    "recall@5": float(np.mean(reference_subset <= 5) * 100.0),
                    "mean_rank": float(np.mean(reference_subset)),
                },
                "evaluated_scope_minus_full_r1_pp": float(
                    (np.mean(reference_subset <= 1) - np.mean(reference_full <= 1)) * 100.0
                ),
                "note": (
                    "Historical ranks assume GT position 0; compare with optimistic tie metrics. "
                    f"Audit optimistic R@1={baseline['optimistic_recall@1']:.6f}."
                ),
            }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    printable = dict(output)
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
