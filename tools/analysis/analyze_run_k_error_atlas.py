#!/usr/bin/env python3
"""Build an ODF-free, three-layer error atlas for the Run K baseline.

The atlas separates:
  1. candidate / molecular-structure differences,
  2. Run K raw-fingerprint, projected-molecule, and final-score geometry,
  3. observed peak and precursor-minus-peak evidence.

It deliberately treats GT ranks beyond the exported top-k as recall/long-tail
failures: their GT score and representation pair are not silently imputed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors, rdDepictor, rdFMCS, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glmr.sparse_features import SparseFeatureCache, load_sparse_feature_cache  # noqa: E402
from tools.stage1.train_msalign_projection_smoke import (  # noqa: E402
    build_projection_model,
    checkpoint_model_config,
)


DEFAULT_METADATA = ROOT / "outputs/msalign_formula_group_split_v15/val.metadata.jsonl"
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
DEFAULT_OUT = ROOT / "outputs/run_k_error_atlas/full_val_20260809"

BLOCKS = (
    ("count_morgan_r3", 2048, "morgan_r3_cosine"),
    ("count_atom_pair", 2048, "atom_pair_cosine"),
    ("rdkit_path_count", 2048, "rdkit_path_cosine"),
    ("maccs", 167, "maccs_cosine"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--topk-jsonl", type=Path, default=DEFAULT_TOPK)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--state-key",
        choices=("best_model_state_dict", "model_state_dict"),
        default="best_model_state_dict",
    )
    parser.add_argument("--fingerprint-manifest", type=Path, default=DEFAULT_FP)
    parser.add_argument("--r3-manifest", type=Path, default=DEFAULT_R3)
    parser.add_argument(
        "--composite-fingerprint-manifest",
        type=Path,
        default=None,
        help="Single sparse/composite manifest containing every Run K block.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-peaks", type=int, default=10)
    parser.add_argument("--representatives-per-stratum", type=int, default=2)
    parser.add_argument("--mcs-timeout", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Debug query limit; 0 means all rows.")
    parser.add_argument("--skip-representation", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def rank_bucket(rank: int) -> str:
    if rank == 1:
        return "rank_1"
    if rank <= 5:
        return "rank_2_5"
    if rank <= 20:
        return "rank_6_20"
    if rank <= 50:
        return "rank_21_50"
    if rank <= 100:
        return "rank_51_100"
    return "rank_gt_100"


def mol_profile(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "smiles": smiles}
    iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    noniso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchikey = ""
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, includeCIP=True)
    return {
        "valid": True,
        "smiles": smiles,
        "canonical_isomeric_smiles": iso,
        "canonical_nonisomeric_smiles": noniso,
        "inchikey": inchikey,
        "connectivity_key": inchikey.split("-")[0] if inchikey else noniso,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "exact_mass": float(Descriptors.ExactMolWt(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "bonds": int(mol.GetNumBonds()),
        "rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "fragments": int(len(Chem.GetMolFrags(mol))),
        "stereocenters": int(len(centers)),
        "unassigned_stereocenters": int(sum(label == "?" for _, label in centers)),
    }


def pair_structure(gt: Mapping[str, Any], wrong: Mapping[str, Any]) -> dict[str, Any]:
    if not gt.get("valid") or not wrong.get("valid"):
        return {"structure_category": "invalid_smiles"}
    same_formula = gt["formula"] == wrong["formula"]
    same_connectivity = gt["connectivity_key"] == wrong["connectivity_key"]
    stereoisomer_only = bool(
        same_connectivity
        and gt["canonical_isomeric_smiles"] != wrong["canonical_isomeric_smiles"]
    )
    if stereoisomer_only:
        category = "stereoisomer_only"
    elif same_formula:
        category = "same_formula_constitutional"
    else:
        category = "different_formula_near_mass"
    gt_mass = float(gt["exact_mass"])
    wrong_mass = float(wrong["exact_mass"])
    delta = wrong_mass - gt_mass
    return {
        "same_formula": same_formula,
        "same_connectivity": same_connectivity,
        "stereoisomer_only": stereoisomer_only,
        "structure_category": category,
        "neutral_exact_mass_delta_da": delta,
        "neutral_exact_mass_abs_delta_da": abs(delta),
        "neutral_exact_mass_delta_ppm": 1e6 * delta / gt_mass if gt_mass else None,
        "heavy_atom_delta": int(wrong["heavy_atoms"] - gt["heavy_atoms"]),
        "bond_delta": int(wrong["bonds"] - gt["bonds"]),
        "ring_delta": int(wrong["rings"] - gt["rings"]),
        "aromatic_ring_delta": int(wrong["aromatic_rings"] - gt["aromatic_rings"]),
    }


def major_peaks(
    mzs: Iterable[float],
    intensities: Iterable[float],
    precursor_mz: float | None,
    count: int,
) -> list[dict[str, Any]]:
    values = sorted(
        zip(mzs, intensities, strict=True), key=lambda item: (-float(item[1]), float(item[0]))
    )[:count]
    output = []
    for intensity_rank, (mz, intensity) in enumerate(values, start=1):
        delta = None if precursor_mz is None else float(precursor_mz) - float(mz)
        output.append(
            {
                "intensity_rank": intensity_rank,
                "mz": round(float(mz), 6),
                "relative_intensity": round(float(intensity), 6),
                # This is deliberately not assigned a chemical identity.
                "precursor_minus_peak_mz": round(delta, 6) if delta is not None else None,
            }
        )
    return output


def grouped_distribution(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, float]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(str(row[value]))
    labels = sorted({str(row[value]) for row in rows})
    result = {}
    for label in labels:
        fractions = [sum(item == label for item in items) / len(items) for items in groups.values()]
        result[label] = float(np.mean(fractions)) if fractions else 0.0
    return result


def categorical_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def slice_error_rates(rows: list[dict[str, Any]], key: str, min_count: int = 20) -> dict[str, Any]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(int(row["rank"]) != 1)
    output = {}
    for label, errors in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(errors) >= min_count:
            output[label] = {"queries": len(errors), "top1_error_rate": float(np.mean(errors))}
    return output


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_run_k_model(
    checkpoint: Path,
    device: torch.device,
    state_key: str = "best_model_state_dict",
) -> tuple[torch.nn.Module, float]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint_model_config(payload)
    config.setdefault("hidden_dim", 2048)
    config.setdefault("proj_dim", 1024)
    config.setdefault("dropout", 0.2)
    config.setdefault("head_type", "mlp")
    config.setdefault("temperature", 0.07)
    config.setdefault("mol_block_normalization", "l2")
    model = build_projection_model(
        spec_dim=1024,
        mol_dim=sum(dim for _, dim, _ in BLOCKS),
        mol_block_dims=[dim for _, dim, _ in BLOCKS],
        mol_block_names=[name for name, _, _ in BLOCKS],
        config=config,
    )
    state = payload.get(state_key)
    if state is None:
        raise ValueError(f"No model state found in {checkpoint}")
    model.load_state_dict(state)
    model.eval().to(device)
    return model, float(model.temperature().detach().cpu())


def add_representation_geometry(
    errors: list[dict[str, Any]],
    *,
    fp_manifest: Path,
    r3_manifest: Path,
    composite_manifest: Path | None,
    checkpoint: Path,
    state_key: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    actionable = [row for row in errors if row.get("gt_global_index") is not None]
    if not actionable:
        return {"actionable_errors": 0}

    if composite_manifest is not None:
        base = load_sparse_feature_cache(composite_manifest)
        stores = {name: base for name, _, _ in BLOCKS}
    else:
        base = SparseFeatureCache(fp_manifest)
        r3 = SparseFeatureCache(r3_manifest)
        if base.rows != r3.rows or base.smiles_sha256 != r3.smiles_sha256:
            raise ValueError("Run K sparse feature caches have incompatible molecule row orders")
        stores = {"count_morgan_r3": r3, **{name: base for name, _, _ in BLOCKS[1:]}}
    model, temperature = load_run_k_model(checkpoint, device, state_key)

    with torch.inference_mode():
        for start in range(0, len(actionable), batch_size):
            batch = actionable[start : start + batch_size]
            pair_indices = []
            for row in batch:
                pair_indices.extend([int(row["gt_global_index"]), int(row["top1_global_index"])])
            indices = torch.tensor(pair_indices, dtype=torch.long)
            raw_parts: list[torch.Tensor] = []
            normalized_parts: list[torch.Tensor] = []
            for name, _, output_key in BLOCKS:
                values = stores[name].gather(name, indices).float()
                gt_values, wrong_values = values[0::2], values[1::2]
                similarities = F.cosine_similarity(gt_values, wrong_values, dim=-1)
                for row, similarity in zip(batch, similarities.tolist(), strict=True):
                    row[output_key] = float(similarity)
                raw_parts.append(values)
                normalized_parts.append(F.normalize(values, dim=-1))
            raw = torch.cat(raw_parts, dim=-1)
            normalized = torch.cat(normalized_parts, dim=-1)
            gt_norm, wrong_norm = normalized[0::2], normalized[1::2]
            combined = F.cosine_similarity(gt_norm, wrong_norm, dim=-1)
            projected = model.project_mol(raw.to(device))
            projected_cos = F.cosine_similarity(projected[0::2], projected[1::2], dim=-1).cpu()
            for row, raw_cos, proj_cos in zip(
                batch, combined.tolist(), projected_cos.tolist(), strict=True
            ):
                row["raw_block_l2_concat_cosine"] = float(raw_cos)
                row["projected_molecule_cosine"] = float(proj_cos)
                row["projection_similarity_shift"] = float(proj_cos - raw_cos)
                row["top1_spectrum_molecule_cosine"] = float(row["top1_score"] * temperature)
                row["gt_spectrum_molecule_cosine"] = float(row["gt_score"] * temperature)
    return {
        "actionable_errors": len(actionable),
        "sparse_molecule_rows": base.rows,
        "temperature": temperature,
        "device": str(device),
        "feature_blocks": [name for name, _, _ in BLOCKS],
    }


def mcs_details(gt_smiles: str, wrong_smiles: str, timeout: int) -> dict[str, Any]:
    gt = Chem.MolFromSmiles(gt_smiles)
    wrong = Chem.MolFromSmiles(wrong_smiles)
    if gt is None or wrong is None:
        return {"mcs_valid": False}
    result = rdFMCS.FindMCS(
        [gt, wrong],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
        matchChiralTag=False,
        timeout=max(1, int(timeout)),
    )
    query = Chem.MolFromSmarts(result.smartsString) if result.smartsString else None
    gt_match = gt.GetSubstructMatch(query) if query is not None else ()
    wrong_match = wrong.GetSubstructMatch(query) if query is not None else ()
    return {
        "mcs_valid": bool(query is not None and gt_match and wrong_match),
        "mcs_canceled": bool(result.canceled),
        "mcs_smarts": result.smartsString,
        "mcs_atoms": int(result.numAtoms),
        "mcs_bonds": int(result.numBonds),
        "mcs_gt_atom_fraction": safe_div(result.numAtoms, gt.GetNumHeavyAtoms()),
        "mcs_top1_atom_fraction": safe_div(result.numAtoms, wrong.GetNumHeavyAtoms()),
        "gt_match": list(gt_match),
        "wrong_match": list(wrong_match),
    }


def draw_structure_pair(row: dict[str, Any], svg_path: Path, png_path: Path) -> None:
    gt = Chem.MolFromSmiles(row["gt_smiles"])
    wrong = Chem.MolFromSmiles(row["top1_smiles"])
    if gt is None or wrong is None:
        return
    rdDepictor.Compute2DCoords(gt)
    query = Chem.MolFromSmarts(row.get("mcs_smarts") or "")
    if query is not None and query.GetNumAtoms():
        try:
            rdDepictor.GenerateDepictionMatching2DStructure(wrong, gt, refPatt=query)
        except Exception:
            rdDepictor.Compute2DCoords(wrong)
    else:
        rdDepictor.Compute2DCoords(wrong)
    gt_common = set(row.get("gt_match") or [])
    wrong_common = set(row.get("wrong_match") or [])
    gt_unique = [atom.GetIdx() for atom in gt.GetAtoms() if atom.GetIdx() not in gt_common]
    wrong_unique = [atom.GetIdx() for atom in wrong.GetAtoms() if atom.GetIdx() not in wrong_common]
    gt_colors = {idx: (0.25, 0.55, 0.95) for idx in gt_unique}
    wrong_colors = {idx: (0.95, 0.35, 0.30) for idx in wrong_unique}
    if row.get("stereoisomer_only"):
        for mol, colors in ((gt, gt_colors), (wrong, wrong_colors)):
            for atom in mol.GetAtoms():
                if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                    colors[atom.GetIdx()] = (0.65, 0.35, 0.85)
    def render(drawer: Any) -> Any:
        drawer.drawOptions().legendFontSize = 20
        drawer.DrawMolecules(
            [gt, wrong],
            legends=[f"GT | {row['gt_formula']}", f"Run K top1 | {row['top1_formula']}"],
            highlightAtoms=[gt_unique, wrong_unique],
            highlightAtomColors=[gt_colors, wrong_colors],
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    svg_path.write_text(render(rdMolDraw2D.MolDraw2DSVG(1000, 430, 500, 430)), encoding="utf-8")
    png_path.write_bytes(render(rdMolDraw2D.MolDraw2DCairo(1000, 430, 500, 430)))


def draw_spectrum(row: dict[str, Any], path: Path) -> None:
    peaks = row["major_peaks"]
    fig, axis = plt.subplots(figsize=(9, 4.2))
    mzs = [peak["mz"] for peak in peaks]
    intensities = [peak["relative_intensity"] for peak in peaks]
    axis.vlines(mzs, 0, intensities, color="#283593", linewidth=1.8)
    for peak in peaks[:6]:
        delta = peak["precursor_minus_peak_mz"]
        label = f"{peak['mz']:.4f}\nΔ={delta:.4f}" if delta is not None else f"{peak['mz']:.4f}"
        axis.annotate(
            label,
            (peak["mz"], peak["relative_intensity"]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=55,
        )
    axis.set_xlabel("Observed peak m/z (labels: peak m/z and precursor−peak Δm/z)")
    axis.set_ylabel("Relative intensity")
    axis.set_ylim(0, max(intensities, default=1.0) * 1.32)
    axis.set_title(f"{row['identifier']} | precursor m/z {row.get('precursor_mz')} | strongest peaks")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def choose_representatives(
    errors: list[dict[str, Any]], per_stratum: int
) -> list[dict[str, Any]]:
    actionable = [row for row in errors if row.get("projected_molecule_cosine") is not None]
    if not actionable:
        return []
    margins = np.asarray([row["score_margin_wrong_minus_gt"] for row in actionable])
    shifts = np.asarray([row["projection_similarity_shift"] for row in actionable])
    margin_low, margin_high = np.quantile(margins, [0.2, 0.8])
    shift_high = float(np.quantile(shifts, 0.9))
    strata = [
        (
            "stereoisomer_only",
            lambda row: bool(row.get("stereoisomer_only")),
            lambda row: -row["score_margin_wrong_minus_gt"],
        ),
        (
            "same_formula_constitutional",
            lambda row: row.get("structure_category") == "same_formula_constitutional",
            lambda row: -row["score_margin_wrong_minus_gt"],
        ),
        (
            "different_formula_near_mass",
            lambda row: row.get("structure_category") == "different_formula_near_mass",
            lambda row: (row["neutral_exact_mass_abs_delta_da"], -row["score_margin_wrong_minus_gt"]),
        ),
        (
            "projection_collision",
            lambda row: row["projection_similarity_shift"] >= shift_high,
            lambda row: (-row["projection_similarity_shift"], -row["projected_molecule_cosine"]),
        ),
        (
            "near_tie",
            lambda row: row["score_margin_wrong_minus_gt"] <= margin_low,
            lambda row: row["score_margin_wrong_minus_gt"],
        ),
        (
            "high_confidence_wrong",
            lambda row: row["score_margin_wrong_minus_gt"] >= margin_high,
            lambda row: -row["score_margin_wrong_minus_gt"],
        ),
    ]
    selected: list[dict[str, Any]] = []
    used_gt: set[str] = set()
    used_ids: set[str] = set()
    for label, predicate, sort_key in strata:
        candidates = sorted((row for row in actionable if predicate(row)), key=sort_key)
        taken = 0
        for row in candidates:
            gt_key = str(row["gt_group_key"])
            if gt_key in used_gt or row["identifier"] in used_ids:
                continue
            copy = dict(row)
            copy["representative_stratum"] = label
            selected.append(copy)
            used_gt.add(gt_key)
            used_ids.add(row["identifier"])
            taken += 1
            if taken >= per_stratum:
                break
    return selected


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    excluded = {"major_peaks", "gt_profile", "top1_profile", "gt_match", "wrong_match", "mcs_smarts"}
    keys = sorted(set().union(*(row.keys() for row in rows)) - excluded)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(all_rows: list[dict[str, Any]], errors: list[dict[str, Any]], out_dir: Path) -> None:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    order = ["rank_1", "rank_2_5", "rank_6_20", "rank_21_50", "rank_51_100", "rank_gt_100"]
    counts = Counter(row["rank_bucket"] for row in all_rows)
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    values = [counts[label] for label in order]
    axis.bar(range(len(order)), values, color=["#2e7d32"] + ["#c62828"] * 5)
    axis.set_xticks(range(len(order)), [label.replace("rank_", "") for label in order])
    axis.set_ylabel("Validation queries")
    axis.set_xlabel("Ground-truth rank")
    axis.set_title("Run K full-validation rank distribution")
    for idx, value in enumerate(values):
        axis.text(idx, value + max(values) * 0.012, f"{value:,}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "rank_distribution.png", dpi=180)
    plt.close(fig)

    actionable = [row for row in errors if row.get("projected_molecule_cosine") is not None]
    if actionable:
        fig, axis = plt.subplots(figsize=(6.4, 5.6))
        for same, color, label in ((False, "#ef5350", "different formula"), (True, "#1e88e5", "same formula")):
            subset = [row for row in actionable if bool(row.get("same_formula")) is same]
            axis.scatter(
                [row["raw_block_l2_concat_cosine"] for row in subset],
                [row["projected_molecule_cosine"] for row in subset],
                s=7,
                alpha=0.22,
                c=color,
                label=f"{label} (n={len(subset):,})",
                edgecolors="none",
            )
        axis.plot([-0.05, 1.02], [-0.05, 1.02], "--", color="black", linewidth=1)
        axis.set_xlim(-0.05, 1.02)
        axis.set_ylim(-0.05, 1.02)
        axis.set_xlabel("Raw four-block fingerprint cosine")
        axis.set_ylabel("Projected molecule cosine")
        axis.set_title("Molecular geometry among Run K top1 errors (GT vs wrong top1)")
        axis.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "raw_vs_projected_molecule_similarity.png", dpi=200)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(7.2, 4.8))
        same = [row["score_margin_wrong_minus_gt"] for row in actionable if row.get("same_formula")]
        different = [row["score_margin_wrong_minus_gt"] for row in actionable if not row.get("same_formula")]
        axis.hist(different, bins=60, alpha=0.65, label=f"different formula (n={len(different):,})")
        axis.hist(same, bins=60, alpha=0.65, label=f"same formula (n={len(same):,})")
        axis.set_xlabel("Run K logit margin: wrong top1 − GT")
        axis.set_ylabel("Errors")
        axis.set_title("Error confidence for rerank-actionable failures")
        axis.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "error_score_margin.png", dpi=180)
        plt.close(fig)


def summarize_numeric(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    values = np.asarray([float(row[key]) for row in rows if row.get(key) is not None], dtype=np.float64)
    if not len(values):
        return None
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q10": float(np.quantile(values, 0.1)),
        "q90": float(np.quantile(values, 0.9)),
    }


def representation_slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in rows if row.get("projected_molecule_cosine") is not None]
    if not rows:
        return {"count": 0}
    raw = np.asarray([row["raw_block_l2_concat_cosine"] for row in rows])
    projected = np.asarray([row["projected_molecule_cosine"] for row in rows])
    margin = np.asarray([row["score_margin_wrong_minus_gt"] for row in rows])
    return {
        "count": len(rows),
        "raw_cosine": summarize_numeric(rows, "raw_block_l2_concat_cosine"),
        "projected_molecule_cosine": summarize_numeric(rows, "projected_molecule_cosine"),
        "score_margin_wrong_minus_gt": summarize_numeric(rows, "score_margin_wrong_minus_gt"),
        "fraction_projected_cosine_above_0_8": float(np.mean(projected > 0.8)),
        "fraction_raw_cosine_above_0_8": float(np.mean(raw > 0.8)),
        "fraction_projection_similarity_increases": float(np.mean(projected > raw)),
        "pearson_margin_vs_raw_cosine": float(np.corrcoef(margin, raw)[0, 1]),
        "pearson_margin_vs_projected_cosine": float(np.corrcoef(margin, projected)[0, 1]),
    }


def peak_count_error_bins(rows: list[dict[str, Any]]) -> dict[str, Any]:
    peak_counts = np.asarray([int(row["num_peaks"]) for row in rows])
    q25, q50, q75 = [int(value) for value in np.quantile(peak_counts, [0.25, 0.5, 0.75])]
    ranges = ((0, q25), (q25 + 1, q50), (q50 + 1, q75), (q75 + 1, int(peak_counts.max())))
    output = {}
    for lower, upper in ranges:
        subset = [row for row in rows if lower <= int(row["num_peaks"]) <= upper]
        output[f"{lower}_{upper}"] = {
            "queries": len(subset),
            "top1_error_rate": float(np.mean([not row["is_top1_correct"] for row in subset])),
        }
    return output


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_jsonl(args.metadata, args.limit)
    topk = load_jsonl(args.topk_jsonl, args.limit)
    if len(metadata) != len(topk):
        raise ValueError(f"metadata/top-k row count mismatch: {len(metadata)} != {len(topk)}")
    metadata_by_id = {row["identifier"]: row for row in metadata}
    all_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for topk_row in topk:
        identifier = topk_row["identifier"]
        meta = metadata_by_id.get(identifier)
        if meta is None:
            raise KeyError(f"Missing validation metadata for {identifier}")
        rank = int(topk_row["stage1_gt_rank"])
        candidates = topk_row["candidates"]
        if not candidates:
            raise ValueError(f"No exported candidates for {identifier}")
        gt_group_key = str(meta.get("inchikey") or meta["canonical_isomeric_smiles"])
        query = {
            "identifier": identifier,
            "gt_group_key": gt_group_key,
            "rank": rank,
            "rank_bucket": rank_bucket(rank),
            "is_top1_correct": rank == 1,
            "formula": meta.get("formula"),
            "precursor_mz": finite(meta.get("precursor_mz")),
            "parent_mass": finite(meta.get("parent_mass")),
            "adduct": meta.get("adduct"),
            "instrument_type": meta.get("instrument_type"),
            "collision_energy": finite(meta.get("collision_energy")),
            "collision_energy_missing": finite(meta.get("collision_energy")) is None,
            "num_peaks": int(meta.get("num_peaks") or len(topk_row.get("mzs") or [])),
        }
        all_rows.append(query)
        if rank == 1:
            continue
        top1 = candidates[0]
        gt_candidate = next((candidate for candidate in candidates if candidate.get("is_gt")), None)
        gt_profile = mol_profile(meta["canonical_isomeric_smiles"])
        top1_profile = mol_profile(top1["smiles"])
        pair = pair_structure(gt_profile, top1_profile)
        gt_score = finite(gt_candidate.get("stage1_score")) if gt_candidate else None
        top1_score = float(top1["stage1_score"])
        row = dict(query)
        row.update(pair)
        row.update(
            {
                "gt_smiles": meta["canonical_isomeric_smiles"],
                "gt_formula": gt_profile.get("formula") or meta.get("formula"),
                "gt_neutral_exact_mass": gt_profile.get("exact_mass"),
                "top1_smiles": top1["smiles"],
                "top1_formula": top1_profile.get("formula"),
                "top1_neutral_exact_mass": top1_profile.get("exact_mass"),
                "top1_global_index": int(top1["global_index"]),
                "gt_global_index": int(gt_candidate["global_index"]) if gt_candidate else None,
                "top1_score": top1_score,
                "gt_score": gt_score,
                "score_margin_wrong_minus_gt": top1_score - gt_score if gt_score is not None else None,
                "gt_exported_in_topk": gt_candidate is not None,
                "major_peaks": major_peaks(
                    topk_row.get("mzs") or [],
                    topk_row.get("intensities") or [],
                    finite(meta.get("precursor_mz")),
                    args.top_peaks,
                ),
                "gt_profile": gt_profile,
                "top1_profile": top1_profile,
            }
        )
        errors.append(row)

    representation = {"skipped": True}
    if not args.skip_representation:
        representation = add_representation_geometry(
            errors,
            fp_manifest=args.fingerprint_manifest,
            r3_manifest=args.r3_manifest,
            composite_manifest=args.composite_fingerprint_manifest,
            checkpoint=args.checkpoint,
            state_key=args.state_key,
            device=resolve_device(args.device),
            batch_size=args.batch_size,
        )
        representation["skipped"] = False

    representatives = choose_representatives(errors, args.representatives_per_stratum)
    case_dir = args.out_dir / "figures/cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    enriched_representatives = []
    for index, row in enumerate(representatives, start=1):
        row.update(mcs_details(row["gt_smiles"], row["top1_smiles"], args.mcs_timeout))
        stem = f"case_{index:02d}_{row['representative_stratum']}_{row['identifier']}"
        structure_path = case_dir / f"{stem}_structure.svg"
        structure_png_path = case_dir / f"{stem}_structure.png"
        spectrum_path = case_dir / f"{stem}_peaks.png"
        draw_structure_pair(row, structure_path, structure_png_path)
        draw_spectrum(row, spectrum_path)
        row["structure_figure"] = str(structure_path.relative_to(args.out_dir))
        row["structure_png"] = str(structure_png_path.relative_to(args.out_dir))
        row["spectrum_figure"] = str(spectrum_path.relative_to(args.out_dir))
        enriched_representatives.append(row)

    rank_counts = categorical_counts(all_rows, "rank_bucket")
    query_rank_distribution = {key: value / len(all_rows) for key, value in rank_counts.items()}
    actionable = [row for row in errors if row.get("gt_global_index") is not None]
    same_formula_actionable = [row for row in actionable if row.get("same_formula")]
    different_formula_actionable = [row for row in actionable if not row.get("same_formula")]
    margin_values = [row["score_margin_wrong_minus_gt"] for row in actionable]
    summary = {
        "protocol": {
            "scope": [
                "molecule_candidate_structure",
                "run_k_representation_and_score",
                "observed_peaks_and_precursor_minus_peak_mz",
            ],
            "excluded": ["ODF attributes", "ODF scores", "query-conditioned fragment claims"],
            "metadata": str(args.metadata),
            "topk_jsonl": str(args.topk_jsonl),
            "checkpoint": str(args.checkpoint),
            "state_key": args.state_key,
            "fingerprint_manifest": str(args.fingerprint_manifest),
            "r3_manifest": str(args.r3_manifest),
            "composite_fingerprint_manifest": (
                str(args.composite_fingerprint_manifest)
                if args.composite_fingerprint_manifest
                else None
            ),
            "top_peaks_per_case": args.top_peaks,
            "gt_group_weight": "equal weight per validation GT InChIKey/connectivity group",
            "neutral_loss_language": "precursor-minus-peak m/z; putative coordinate, not an assigned neutral identity",
        },
        "queries": len(all_rows),
        "unique_gt_groups": len({row["gt_group_key"] for row in all_rows}),
        "top1_correct": sum(row["is_top1_correct"] for row in all_rows),
        "top1_errors": len(errors),
        "top1_recall": sum(row["is_top1_correct"] for row in all_rows) / len(all_rows),
        "rank_counts_query_weighted": rank_counts,
        "rank_distribution_query_weighted": query_rank_distribution,
        "rank_distribution_gt_group_weighted": grouped_distribution(all_rows, "gt_group_key", "rank_bucket"),
        "top1_recall_gt_group_weighted": grouped_distribution(
            all_rows, "gt_group_key", "is_top1_correct"
        ).get("True", 0.0),
        "error_structure_categories": categorical_counts(errors, "structure_category"),
        "error_same_formula": categorical_counts(errors, "same_formula"),
        "same_formula_fraction_by_error_rank_bucket": {
            bucket: float(np.mean([row["same_formula"] for row in errors if row["rank_bucket"] == bucket]))
            for bucket in sorted({row["rank_bucket"] for row in errors})
        },
        "rerank_actionable_errors_gt_in_exported_topk": sum(
            row["gt_exported_in_topk"] for row in errors
        ),
        "rank_beyond_exported_topk_errors": sum(not row["gt_exported_in_topk"] for row in errors),
        "oracle_candidate_ceiling": {
            f"rank_at_most_{cut}": {
                "queries": sum(row["rank"] <= cut for row in all_rows),
                "recall": float(np.mean([row["rank"] <= cut for row in all_rows])),
                "fraction_of_top1_errors_recoverable": float(
                    np.mean([row["rank"] <= cut for row in errors])
                ),
            }
            for cut in (2, 5, 20, 50, 100)
        },
        "representation": representation,
        "representation_slices": {
            "all_actionable_errors": representation_slice(actionable),
            "same_formula_actionable_errors": representation_slice(same_formula_actionable),
            "different_formula_actionable_errors": representation_slice(different_formula_actionable),
        },
        "score_margin_thresholds": {
            f"at_most_{str(threshold).replace('.', '_')}": float(
                np.mean(np.asarray(margin_values) <= threshold)
            )
            for threshold in (0.1, 0.5, 1.0, 2.0, 5.0)
        },
        "numeric_summaries": {
            key: summarize_numeric(errors, key)
            for key in (
                "neutral_exact_mass_abs_delta_da",
                "score_margin_wrong_minus_gt",
                "raw_block_l2_concat_cosine",
                "projected_molecule_cosine",
                "projection_similarity_shift",
                "morgan_r3_cosine",
                "atom_pair_cosine",
                "rdkit_path_cosine",
                "maccs_cosine",
            )
        },
        "slices": {
            "adduct": slice_error_rates(all_rows, "adduct"),
            "instrument_type": slice_error_rates(all_rows, "instrument_type"),
            "collision_energy_missing": slice_error_rates(all_rows, "collision_energy_missing"),
            "peak_count_quartiles": peak_count_error_bins(all_rows),
        },
        "representative_cases": len(enriched_representatives),
    }
    plot_summary(all_rows, errors, args.out_dir)
    write_jsonl(args.out_dir / "all_queries.jsonl", all_rows)
    write_jsonl(args.out_dir / "errors.jsonl", errors)
    write_csv(args.out_dir / "errors.csv", errors)
    write_jsonl(args.out_dir / "representative_cases.jsonl", enriched_representatives)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
