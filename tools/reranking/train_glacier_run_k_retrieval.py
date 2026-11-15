#!/usr/bin/env python3
"""Minimal Run K hard-decoy retrieval fine-tuning for the local GLACIER model.

This is a custom post-P0.3 development diagnostic.  It deliberately keeps the
Run K molecule side and mass-only top-20 candidate protocol fixed.  Stage1 is
used only to select deterministic hard decoys; Stage1 scores never enter the
training logits.

The executable has two phases:

``prepare``
    Materialize an immutable query-group artifact containing a GT and the first
    K valid, 2D-connectivity-distinct Run K hard decoys.  Invalid/disconnected
    structures, unsupported atoms, and duplicate 2D connectivities are removed.

``train``
    Start from the specified forward checkpoint and optimize native spectrum
    reconstruction, matched-MAGMa fragment supervision, and an optional raw
    entropy-energy ranking loss.  Setting ``--rank-weight 0`` is the matched
    reconstruction-only causal control.

No evaluation candidate is inserted by this tool.  The prepared training
artifact does contain the known GT by design; deployment evaluation must use
the original Run K candidate list unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
MS_PRED_SRC = ROOT / "external/ms-pred/src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MS_PRED_SRC) not in sys.path:
    sys.path.insert(0, str(MS_PRED_SRC))


DEFAULT_TOPK = (
    ROOT
    / "outputs/run_k_full_train/post_p03_all_row_seed20260621"
    / "top50_train20k/run_k_topk.jsonl"
)
DEFAULT_MAGMA_LABELS = (
    ROOT
    / "outputs/magma_peak_fragment_alignment"
    / "formula_group_train20k_full20k_seed20260710/labels.jsonl"
)
DEFAULT_CHECKPOINT = (
    ROOT / "external/ms-pred/results/glacier_msg/split_rnd1/version_1/best.ckpt"
)
DEFAULT_OUT_DIR = ROOT / "outputs/glacier_run_k_retrieval_finetune/minimal_v1"

SCHEMA_VERSION = 1
PROTOCOL = (
    "custom post-P0.3 ordinary-DreaMS Run K mass top20; no formula filter; "
    "training GT plus deterministic Stage1-ranked hard decoys; Stage1 score "
    "excluded from loss; 2D-connectivity-deduplicated first pilot"
)

# Must match external/ms-pred/src/ms_pred/common/chem_utils.py.  Keeping this
# lightweight constant lets `prepare` reject unsupported atoms without loading
# the full GLACIER/DGL stack.
SUPPORTED_ATOMIC_NUMBERS = frozenset(
    {
        1,   # H
        5,   # B
        6,   # C
        7,   # N
        8,   # O
        9,   # F
        11,  # Na
        14,  # Si
        15,  # P
        16,  # S
        17,  # Cl
        19,  # K
        26,  # Fe
        27,  # Co
        33,  # As
        34,  # Se
        35,  # Br
        53,  # I
    }
)
SUPPORTED_ADDUCTS = frozenset(
    {
        "[M+H]+",
        "[M+Na]+",
        "[M+K]+",
        "[M-H2O+H]+",
        "[M+H-H2O]+",
        "[M+H3N+H]+",
        "[M+NH4]+",
        "[M]+",
        "[M-H4O2+H]+",
        "[M+H-2H2O]+",
        "[M+H-H3N]+",
        "[M-H3N+H]+",
        "[M-NH3+H]+",
        "[M+H-NH3]+",
        "[M+H-CH2O2]+",
        "[M+CHO2]-",
        "[M-H]-",
        "[M+Cl]-",
        "[M-H2O-H]-",
        "[M-H-H2O]-",
        "[M-H-CO2]-",
    }
)
SUPPORTED_INSTRUMENTS = frozenset({"Orbitrap", "QTOF", "IT-FT", "Unknown"})


@dataclass(frozen=True)
class MoleculeRecord:
    canonical_isomeric_smiles: str
    connectivity: str
    heavy_atoms: int


@dataclass(frozen=True)
class CompactMagmaLabel:
    identifier: str
    canonical_isomeric_smiles: str
    connectivity: str
    num_atoms: int
    fragment_masks: tuple[tuple[int, ...], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


@lru_cache(maxsize=500_000)
def molecule_record(smiles: str, max_heavy_atoms: int = 100) -> MoleculeRecord:
    """Validate and canonicalize one model-supported, single-component molecule."""

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("empty_smiles")
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError("invalid_smiles")
    mol = Chem.RemoveHs(mol)
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("disconnected_structure")
    atomic_numbers = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
    if not atomic_numbers.issubset(SUPPORTED_ATOMIC_NUMBERS):
        raise ValueError("unsupported_atom")
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    if heavy_atoms <= 0:
        raise ValueError("no_heavy_atoms")
    if heavy_atoms > max_heavy_atoms:
        raise ValueError("too_many_heavy_atoms")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if not canonical:
        raise ValueError("canonicalization_failed")
    inchikey = Chem.MolToInchiKey(mol)
    if not inchikey or "-" not in inchikey:
        raise ValueError("inchikey_failed")
    return MoleculeRecord(
        canonical_isomeric_smiles=canonical,
        connectivity=inchikey.split("-", 1)[0],
        heavy_atoms=heavy_atoms,
    )


def fragment_masks_from_label(
    row: dict[str, Any], *, num_atoms: int
) -> tuple[tuple[int, ...], ...]:
    """Return unique matched-peak atom masks in deterministic label order."""

    masks: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for peak in row.get("peaks", []):
        for match in peak.get("matches", []):
            raw_indices = match.get("atom_indices", [])
            try:
                mask = tuple(sorted({int(index) for index in raw_indices}))
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid_fragment_atom_index") from exc
            if not mask:
                continue
            if mask[0] < 0 or mask[-1] >= num_atoms:
                raise ValueError("fragment_atom_index_out_of_range")
            if mask not in seen:
                seen.add(mask)
                masks.append(mask)
    return tuple(masks)


def load_compact_magma_labels(path: Path) -> dict[str, CompactMagmaLabel]:
    labels: dict[str, CompactMagmaLabel] = {}
    for row in iter_jsonl(path):
        identifier = str(row.get("identifier") or "")
        if not identifier:
            raise ValueError("MAGMa label missing identifier")
        if identifier in labels:
            raise ValueError(f"Duplicate MAGMa identifier: {identifier}")
        raw_smiles = str(
            row.get("magma_canonical_smiles")
            or row.get("canonical_isomeric_smiles")
            or ""
        )
        record = molecule_record(raw_smiles)
        num_atoms = int(row.get("num_atoms", record.heavy_atoms))
        if num_atoms != record.heavy_atoms:
            raise ValueError(
                f"MAGMa atom count mismatch for {identifier}: "
                f"{num_atoms} != {record.heavy_atoms}"
            )
        masks = fragment_masks_from_label(row, num_atoms=num_atoms)
        # Preserve the exact MAGMa string/atom order used to produce
        # ``atom_indices``.  Re-canonicalizing and then reusing the old indices
        # would silently attach fragment masks to the wrong atoms when RDKit's
        # serialization order changes.
        labels[identifier] = CompactMagmaLabel(
            identifier=identifier,
            canonical_isomeric_smiles=raw_smiles,
            connectivity=record.connectivity,
            num_atoms=num_atoms,
            fragment_masks=masks,
        )
    return labels


def _finite_collision_energy(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _candidate_failure_reason(candidate: dict[str, Any]) -> str | None:
    try:
        molecule_record(str(candidate.get("smiles") or ""))
    except ValueError as exc:
        return str(exc)
    return None


def select_hard_decoys(
    candidates: Sequence[dict[str, Any]],
    *,
    gt_connectivity: str,
    top_k: int,
    num_decoys: int,
    dedup_2d_connectivity: bool = True,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Select the first valid Stage1-ranked hard decoys under the data contract."""

    selected: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    seen_connectivities = {gt_connectivity}
    for candidate in candidates[:top_k]:
        if bool(candidate.get("is_gt")):
            continue
        failure = _candidate_failure_reason(candidate)
        if failure is not None:
            reasons[failure] += 1
            continue
        record = molecule_record(str(candidate["smiles"]))
        if dedup_2d_connectivity and record.connectivity == gt_connectivity:
            reasons["same_2d_as_gt"] += 1
            continue
        if dedup_2d_connectivity and record.connectivity in seen_connectivities:
            reasons["duplicate_2d_decoy"] += 1
            continue
        seen_connectivities.add(record.connectivity)
        selected.append(
            {
                "rank": int(candidate["rank"]),
                "global_index": int(candidate["global_index"]),
                "smiles": record.canonical_isomeric_smiles,
                "connectivity": record.connectivity,
            }
        )
        if len(selected) == num_decoys:
            break
    return selected, reasons


def prepare_examples(
    topk_path: Path,
    labels_path: Path,
    output_path: Path,
    *,
    top_k: int,
    num_decoys: int,
    max_input_rows: int | None,
    dedup_2d_connectivity: bool,
) -> dict[str, Any]:
    labels = load_compact_magma_labels(labels_path)
    counts: Counter[str] = Counter()
    candidate_filter_counts: Counter[str] = Counter()
    selected_connectivities: set[str] = set()
    gt_ranks: Counter[int] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = output_path.with_name(output_path.name + ".tmp")
    with staging_path.open("w", encoding="utf-8") as output:
        for row_index, row in enumerate(iter_jsonl(topk_path)):
            if max_input_rows is not None and row_index >= max_input_rows:
                break
            counts["input_rows"] += 1
            identifier = str(row.get("identifier") or "")
            label = labels.get(identifier)
            if label is None:
                counts["missing_magma_label"] += 1
                continue
            collision_energy = _finite_collision_energy(row.get("collision_energy"))
            if collision_energy is None:
                counts["missing_collision_energy"] += 1
                continue
            adduct = str(row.get("adduct") or "")
            if adduct not in SUPPORTED_ADDUCTS:
                counts["unsupported_adduct"] += 1
                continue
            instrument = str(row.get("instrument_type") or "Unknown")
            if instrument not in SUPPORTED_INSTRUMENTS:
                counts["unsupported_instrument"] += 1
                continue
            candidates = list(row.get("candidates") or [])[:top_k]
            gt_candidates = [candidate for candidate in candidates if candidate.get("is_gt")]
            if len(gt_candidates) != 1:
                counts["gt_not_exactly_once_in_topk"] += 1
                continue
            gt_candidate = gt_candidates[0]
            gt_failure = _candidate_failure_reason(gt_candidate)
            if gt_failure is not None:
                counts[f"gt_{gt_failure}"] += 1
                continue
            gt_record = molecule_record(str(gt_candidate["smiles"]))
            if gt_record.connectivity != label.connectivity:
                counts["gt_magma_connectivity_mismatch"] += 1
                continue
            if not label.fragment_masks:
                counts["no_matched_fragment_masks"] += 1
                continue
            mzs = row.get("mzs") or []
            intensities = row.get("intensities") or []
            if not mzs or len(mzs) != len(intensities):
                counts["invalid_observed_spectrum"] += 1
                continue
            try:
                spectrum = [
                    [float(mz), float(intensity)]
                    for mz, intensity in zip(mzs, intensities)
                    if math.isfinite(float(mz))
                    and math.isfinite(float(intensity))
                    and float(mz) > 0
                    and float(intensity) > 0
                ]
            except (TypeError, ValueError):
                counts["invalid_observed_spectrum"] += 1
                continue
            if not spectrum:
                counts["invalid_observed_spectrum"] += 1
                continue
            decoys, filter_counts = select_hard_decoys(
                candidates,
                gt_connectivity=gt_record.connectivity,
                top_k=top_k,
                num_decoys=num_decoys,
                dedup_2d_connectivity=dedup_2d_connectivity,
            )
            candidate_filter_counts.update(filter_counts)
            if len(decoys) != num_decoys:
                counts["insufficient_valid_decoys"] += 1
                continue
            gt_rank = int(gt_candidate["rank"])
            example = {
                "schema": SCHEMA_VERSION,
                "identifier": identifier,
                "source_row_index": row_index,
                "adduct": adduct,
                "instrument": instrument,
                "collision_energy": collision_energy,
                "observed_spectrum": spectrum,
                "gt": {
                    "rank": gt_rank,
                    "global_index": int(gt_candidate["global_index"]),
                    # Use the MAGMa-canonical atom order so fragment masks are exact.
                    "smiles": label.canonical_isomeric_smiles,
                    "run_k_smiles": gt_record.canonical_isomeric_smiles,
                    "connectivity": gt_record.connectivity,
                    "fragment_masks": [list(mask) for mask in label.fragment_masks],
                },
                "decoys": decoys,
            }
            output.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
            counts["selected_rows"] += 1
            selected_connectivities.add(gt_record.connectivity)
            gt_ranks[gt_rank] += 1

    os.replace(staging_path, output_path)

    output_sha256 = sha256_file(output_path)
    summary = {
        "schema": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "top_k": top_k,
        "num_decoys": num_decoys,
        "dedup_2d_connectivity": dedup_2d_connectivity,
        "sources": {
            "topk_jsonl": str(topk_path.resolve()),
            "topk_sha256": sha256_file(topk_path),
            "magma_labels_jsonl": str(labels_path.resolve()),
            "magma_labels_sha256": sha256_file(labels_path),
        },
        "output": {
            "examples_jsonl": str(output_path.resolve()),
            "examples_sha256": output_sha256,
        },
        "counts": dict(sorted(counts.items())),
        "candidate_filter_counts": dict(sorted(candidate_filter_counts.items())),
        "selected_gt_connectivities": len(selected_connectivities),
        "selected_gt_rank_histogram": {
            str(rank): count for rank, count in sorted(gt_ranks.items())
        },
        "max_input_rows": max_input_rows,
        "fragment_supervision": (
            "unique atom masks attached to observed peaks in the existing "
            "MAGMa matched-peak labels; not the unavailable native GLACIER full tree"
        ),
    }
    return summary


def listwise_entropy_rank_loss(
    gt_energy: torch.Tensor,
    decoy_energy: torch.Tensor,
    num_decoys: torch.Tensor,
    *,
    temperature: float = 0.05,
    margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy over raw candidate entropy energies (lower is better).

    This equals, per query,

        log(1 + sum_j exp((E_gt - E_j + margin) / temperature)).

    ``decoy_energy`` is flattened query-major.  Padding is masked and therefore
    cannot affect the objective.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if gt_energy.ndim != 1 or decoy_energy.ndim != 1 or num_decoys.ndim != 1:
        raise ValueError("energies and num_decoys must be 1D")
    if gt_energy.shape[0] != num_decoys.shape[0]:
        raise ValueError("gt_energy and num_decoys batch sizes differ")
    if int(num_decoys.sum().item()) != decoy_energy.shape[0]:
        raise ValueError("flattened decoy energy count does not match num_decoys")
    if torch.any(num_decoys <= 0):
        raise ValueError("every ranking group must contain at least one decoy")
    max_decoys = int(num_decoys.max().item())
    logits = gt_energy.new_full((gt_energy.shape[0], max_decoys + 1), float("-inf"))
    logits[:, 0] = -gt_energy / temperature
    start = 0
    for row, count_tensor in enumerate(num_decoys):
        count = int(count_tensor.item())
        logits[row, 1 : count + 1] = (
            -decoy_energy[start : start + count] + margin
        ) / temperature
        start += count
    targets = torch.zeros(gt_energy.shape[0], dtype=torch.long, device=gt_energy.device)
    per_query = F.cross_entropy(logits, targets, reduction="none")
    return per_query.mean(), per_query


def native_sinkhorn_rank_loss(
    gt_energy: torch.Tensor,
    decoy_energy: torch.Tensor,
    num_decoys: torch.Tensor,
    *,
    temperature: float,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Matched control reproducing the released GLACIER Sinkhorn top-1 loss."""

    import pygmtools as pygm

    max_decoys = int(num_decoys.max().item())
    energies = gt_energy.new_ones((gt_energy.shape[0], max_decoys + 1))
    energies[:, 0] = gt_energy
    start = 0
    for row, count_tensor in enumerate(num_decoys):
        count = int(count_tensor.item())
        energies[row, 1 : count + 1] = decoy_energy[start : start + count]
        start += count
    sorted_energies = torch.sort(energies, dim=-1).values.detach()
    ranking_dist = torch.abs(energies[:, :, None] - sorted_energies[:, None, :])
    lengths = num_decoys + 1
    top1_probability = pygm.sinkhorn(
        -ranking_dist,
        n1=lengths,
        n2=lengths,
        tau=temperature,
        backend="pytorch",
    )[:, 0, 0]
    per_query = torch.relu(-torch.log(top1_probability + threshold))
    return per_query.mean(), per_query


def load_prepared_examples(
    path: Path, max_examples: int | None, *, offset: int = 0
) -> list[dict[str, Any]]:
    if offset < 0:
        raise ValueError("example offset must be non-negative")
    examples: list[dict[str, Any]] = []
    for row_index, row in enumerate(iter_jsonl(path)):
        if row_index < offset:
            continue
        if int(row.get("schema", -1)) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported prepared-example schema in {path}")
        examples.append(row)
        if max_examples is not None and len(examples) >= max_examples:
            break
    if not examples:
        raise ValueError(f"No prepared examples loaded from {path}")
    return examples


def _fragment_mask_tensor(masks: Sequence[Sequence[int]], num_atoms: int) -> torch.Tensor:
    output = torch.zeros((len(masks), num_atoms), dtype=torch.bool)
    for row, atom_indices in enumerate(masks):
        output[row, torch.tensor(atom_indices, dtype=torch.long)] = True
    return output


class RunKHardDecoyDataset(torch.utils.data.Dataset):
    """Lazily featurize immutable prepared candidate groups."""

    def __init__(self, examples: Sequence[dict[str, Any]], model_hparams: dict[str, Any]):
        from ms_pred.glacier.dataset import TreeProcessor

        self.examples = list(examples)
        self.processor = TreeProcessor(
            pe_embed_k=int(model_hparams["pe_embed_k"]),
            root_encode="graphormer",
            embed_elem_group=bool(model_hparams["embed_elem_group"]),
            multi_hop_max_dist=int(model_hparams["multi_hop_max_dist"]),
        )

    def __len__(self) -> int:
        return len(self.examples)

    def _featurize_candidate(
        self,
        *,
        example: dict[str, Any],
        smiles: str,
        is_gt: bool,
    ) -> dict[str, Any]:
        import ms_pred.common as common

        molecule = molecule_record(smiles)
        # GT fragment indices refer to the exact MAGMa SMILES atom order.
        # Decoys have no atom-index supervision and can use the normalized
        # canonical serialization from the data contract.
        model_smiles = smiles if is_gt else molecule.canonical_isomeric_smiles
        tree = {
            "root_canonical_smiles": model_smiles,
            "adduct": example["adduct"],
            "collision_energy": float(example["collision_energy"]),
        }
        feature = self.processor.featurize_tree(
            tree, include_inten_targs=False, include_frag_targs=False
        )
        if is_gt:
            feature["frag_targs"] = _fragment_mask_tensor(
                example["gt"]["fragment_masks"], molecule.heavy_atoms
            )
            feature["inten_targs"] = np.asarray(
                example["observed_spectrum"], dtype=np.float32
            )
        theoretical_precursor = (
            common.mass_from_smi(example["gt"]["smiles"])
            + common.ion2mass[example["adduct"]]
        )
        feature.update(
            {
                "name": example["identifier"],
                "adduct": common.ion2onehot_pos[example["adduct"]],
                "instrument": common.instrument2onehot_pos[example["instrument"]],
                "precursor": float(theoretical_precursor),
            }
        )
        return feature

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        gt = self._featurize_candidate(
            example=example, smiles=example["gt"]["smiles"], is_gt=True
        )
        decoys = [
            self._featurize_candidate(
                example=example, smiles=decoy["smiles"], is_gt=False
            )
            for decoy in example["decoys"]
        ]
        return {"gt": gt, "decoys": decoys, "identifier": example["identifier"]}


def collate_candidate_groups(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from ms_pred.glacier.dataset import IntenDataset

    gt_batch = IntenDataset.collate_fn([item["gt"] for item in items])
    flat_decoys = [decoy for item in items for decoy in item["decoys"]]
    if not flat_decoys:
        raise ValueError("Every candidate group must contain a decoy")
    decoy_batch = IntenDataset.collate_fn(flat_decoys)
    return {
        "gt": gt_batch,
        "decoy": decoy_batch,
        "num_decoys": torch.tensor(
            [len(item["decoys"]) for item in items], dtype=torch.long
        ),
        "identifiers": [item["identifier"] for item in items],
    }


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def _forward_model(model: torch.nn.Module, batch: dict[str, Any], *, is_decoy: bool):
    return model.forward(
        graphormer_input=batch.get("graphormer_input"),
        num_atoms=batch["num_atoms"],
        adducts=batch["adducts"],
        collision_engs=batch["collision_engs"],
        root_form_vecs=batch["root_form_vecs"],
        masses=batch["masses"],
        adduct_mass_shifts=batch["adduct_mass_shifts"],
        atom_form_vecs=batch["atom_form_vecs"],
        adj_matrices=batch["adj_matrices"],
        atom_hs=batch["atom_hs"],
        total_hs=batch["total_hs"],
        instruments=batch["instruments"] if model.embed_instrument else None,
        include_magma=False,
        frag_targs=None,
        num_frag_targs=None,
        is_decoy=is_decoy,
    )


def _fragment_loss(model: torch.nn.Module, prediction: dict[str, Any], batch: dict[str, Any]):
    breakpoints = prediction["breakpoints_pred"]
    if model.enable_aux_loss:
        loss = torch.zeros((), device=batch["num_atoms"].device)
        for layer in range(model.frag_decoder_layers):
            layer_weight = 0.9 ** (model.frag_decoder_layers - 1 - layer)
            loss = loss + layer_weight * model.frag_loss(
                breakpoints["frag_logits"][layer],
                breakpoints["frag_card_logits"][layer],
                batch["frag_targs"],
                batch["num_frag_targs"],
                batch["num_atoms"],
                batch["adj_matrices"],
            )
        return loss
    return model.frag_loss(
        breakpoints["frag_logits"][-1],
        breakpoints["frag_card_logits"][-1],
        batch["frag_targs"],
        batch["num_frag_targs"],
        batch["num_atoms"],
        batch["adj_matrices"],
    )


def compute_training_losses(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    spec_weight: float,
    frag_weight: float,
    rank_weight: float,
    rank_loss_name: str,
    rank_temperature: float,
    rank_margin: float,
    native_threshold: float,
) -> dict[str, torch.Tensor]:
    gt_batch = batch["gt"]
    decoy_batch = batch["decoy"]
    num_decoys = batch["num_decoys"]
    gt_prediction = _forward_model(model, gt_batch, is_decoy=False)
    decoy_prediction = _forward_model(model, decoy_batch, is_decoy=True)
    gt_spectrum = gt_prediction["inten_pred_end_to_end"]["output"]
    decoy_spectrum = decoy_prediction["inten_pred_end_to_end"]["output"]
    spec_loss = model.inten_loss_fn(
        gt_spectrum,
        gt_batch["inten_targs"],
        parent_mass=gt_batch["precursor_mzs"],
    )["loss"].mean()
    frag_loss = _fragment_loss(model, gt_prediction, gt_batch)
    decoy_targets = gt_batch["inten_targs"].repeat_interleave(num_decoys, dim=0)
    gt_energy = model.entropy_loss_sparse(
        gt_spectrum,
        gt_batch["inten_targs"],
        parent_mass=gt_batch["precursor_mzs"],
    )["loss"]
    decoy_energy = model.entropy_loss_sparse(
        decoy_spectrum,
        decoy_targets,
        parent_mass=decoy_batch["precursor_mzs"],
    )["loss"]
    if rank_loss_name == "listwise_ce":
        rank_loss, rank_per_query = listwise_entropy_rank_loss(
            gt_energy,
            decoy_energy,
            num_decoys,
            temperature=rank_temperature,
            margin=rank_margin,
        )
    elif rank_loss_name == "native_sinkhorn":
        if rank_margin != 0:
            raise ValueError("rank_margin is only defined for listwise_ce")
        rank_loss, rank_per_query = native_sinkhorn_rank_loss(
            gt_energy,
            decoy_energy,
            num_decoys,
            temperature=rank_temperature,
            threshold=native_threshold,
        )
    else:
        raise ValueError(f"Unknown rank loss: {rank_loss_name}")
    total = spec_weight * spec_loss + frag_weight * frag_loss + rank_weight * rank_loss
    return {
        "loss": total,
        "spec_loss": spec_loss,
        "frag_loss": frag_loss,
        "rank_loss": rank_loss,
        "rank_per_query": rank_per_query,
        "gt_energy": gt_energy,
        "decoy_energy": decoy_energy,
    }


def connectivity_balanced_epoch_indices(
    examples: Sequence[dict[str, Any]], *, seed: int, epoch: int
) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[str(example["gt"]["connectivity"])].append(index)
    rng = random.Random(seed + epoch * 1_000_003)
    indices = [rng.choice(group) for _, group in sorted(groups.items())]
    rng.shuffle(indices)
    return indices


def query_weighted_epoch_indices(
    examples: Sequence[dict[str, Any]], *, seed: int, epoch: int
) -> list[int]:
    indices = list(range(len(examples)))
    random.Random(seed + epoch * 1_000_003).shuffle(indices)
    return indices


def _batch_indices(indices: Sequence[int], batch_size: int) -> Iterable[list[int]]:
    for start in range(0, len(indices), batch_size):
        yield list(indices[start : start + batch_size])


def _gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().norm().item()) ** 2
    return math.sqrt(squared)


def _minimum_energy_gaps(
    gt_energy: torch.Tensor, decoy_energy: torch.Tensor, num_decoys: torch.Tensor
) -> torch.Tensor:
    gaps = []
    start = 0
    for gt, count_tensor in zip(gt_energy, num_decoys):
        count = int(count_tensor.item())
        gaps.append(decoy_energy[start : start + count].min() - gt)
        start += count
    return torch.stack(gaps)


def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_hparams: dict[str, Any],
    global_step: int,
    protocol: dict[str, Any],
    last_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_hparams": model_hparams,
        "global_step": global_step,
        "protocol": protocol,
        "last_metrics": last_metrics,
    }


def _save_checkpoint_atomic(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint in-place only after torch.save finishes cleanly."""
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    from ms_pred.glacier.joint_model import JointModel

    _seed_everything(args.seed)
    examples = load_prepared_examples(
        args.examples_jsonl, args.max_examples, offset=args.example_offset
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_hparams = dict(checkpoint["hyper_parameters"])
    model = JointModel(**model_hparams)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = _resolve_device(args.device)
    model.to(device)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dataset = RunKHardDecoyDataset(examples, model_hparams)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "train_metrics.jsonl"
    protocol = {
        "schema": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "bounded smoke" if args.max_steps is not None else "training",
        "causal_arm": (
            "retrieval_aware" if args.rank_weight > 0 else "reconstruction_only_control"
        ),
        "rank_energy": "raw differentiable GLACIER sparse entropy distance",
        "rank_loss": args.rank_loss,
        "rank_temperature": args.rank_temperature,
        "rank_margin": args.rank_margin,
        "native_threshold": args.native_threshold,
        "loss": {
            "spec_weight": args.spec_weight,
            "frag_weight": args.frag_weight,
            "rank_weight": args.rank_weight,
        },
        "stage1_in_training_logits": False,
        "candidate_selection": "Stage1-ranked Run K mass-top20 hard decoys only",
        "formula_filter": False,
        "gt_insertion": "training groups only; forbidden at evaluation",
        "fragment_supervision": "existing MAGMa matched-peak masks, not native full tree",
        "sampling": args.sampling,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_algorithm": args.gradient_clip_algorithm,
            "gradient_clip_value": args.gradient_clip_value,
        },
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "checkpoint_every_steps": args.checkpoint_every,
        "log_every_steps": args.log_every,
        "examples_loaded": len(examples),
        "example_offset": args.example_offset,
        "connectivities_loaded": len(
            {str(example["gt"]["connectivity"]) for example in examples}
        ),
        "sources": {
            "examples_jsonl": str(args.examples_jsonl.resolve()),
            "examples_sha256": sha256_file(args.examples_jsonl),
            "initial_checkpoint": str(args.checkpoint.resolve()),
            "initial_checkpoint_sha256": sha256_file(args.checkpoint),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "device": str(device),
        "torch_version": torch.__version__,
        "python_executable": sys.executable,
        "assumptions": [
            "full checkpoint is fine-tuned; no layer-freezing ablation in minimal v1",
            "checkpoint-native cosine is L_spec",
            "learning rate is a new pilot hyperparameter, not a locked fact",
            "current checkpoint provenance remains unresolved; result cannot be a clean headline",
        ],
    }
    (args.out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    global_step = 0
    last_metrics: dict[str, Any] | None = None
    training_started = time.perf_counter()
    with metrics_path.open("w", encoding="utf-8") as metrics_handle:
        for epoch in range(args.epochs):
            if args.sampling == "connectivity":
                epoch_indices = connectivity_balanced_epoch_indices(
                    examples, seed=args.seed, epoch=epoch
                )
            else:
                epoch_indices = query_weighted_epoch_indices(
                    examples, seed=args.seed, epoch=epoch
                )
            for index_batch in _batch_indices(epoch_indices, args.batch_size):
                feature_items = [dataset[index] for index in index_batch]
                batch = move_to_device(collate_candidate_groups(feature_items), device)
                optimizer.zero_grad(set_to_none=True)
                losses = compute_training_losses(
                    model,
                    batch,
                    spec_weight=args.spec_weight,
                    frag_weight=args.frag_weight,
                    rank_weight=args.rank_weight,
                    rank_loss_name=args.rank_loss,
                    rank_temperature=args.rank_temperature,
                    rank_margin=args.rank_margin,
                    native_threshold=args.native_threshold,
                )
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError(f"Non-finite loss at step {global_step}")
                losses["loss"].backward()
                grad_norm_before_clip = _gradient_norm(model.parameters())
                if args.gradient_clip_algorithm == "value":
                    torch.nn.utils.clip_grad_value_(
                        model.parameters(), args.gradient_clip_value
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.gradient_clip_value
                    )
                grad_norm_after_clip = _gradient_norm(model.parameters())
                optimizer.step()
                gaps = _minimum_energy_gaps(
                    losses["gt_energy"], losses["decoy_energy"], batch["num_decoys"]
                )
                last_metrics = {
                    "step": global_step,
                    "epoch": epoch,
                    "identifiers": batch["identifiers"],
                    "loss": float(losses["loss"].detach().cpu()),
                    "spec_loss": float(losses["spec_loss"].detach().cpu()),
                    "frag_loss": float(losses["frag_loss"].detach().cpu()),
                    "rank_loss": float(losses["rank_loss"].detach().cpu()),
                    "gt_energy_mean": float(losses["gt_energy"].detach().mean().cpu()),
                    "decoy_energy_mean": float(
                        losses["decoy_energy"].detach().mean().cpu()
                    ),
                    "minimum_energy_gap_mean": float(gaps.detach().mean().cpu()),
                    "hard_group_top1_fraction": float((gaps > 0).float().mean().cpu()),
                    "grad_norm_before_clip": grad_norm_before_clip,
                    "grad_norm_after_clip": grad_norm_after_clip,
                    "gradient_clip_algorithm": args.gradient_clip_algorithm,
                    "gradient_clip_value": args.gradient_clip_value,
                    "learning_rate": args.learning_rate,
                    "process_max_rss_mib": (
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                    ),
                }
                if device.type == "cuda":
                    last_metrics.update(
                        {
                            "cuda_peak_allocated_mib": (
                                torch.cuda.max_memory_allocated(device) / (1024**2)
                            ),
                            "cuda_peak_reserved_mib": (
                                torch.cuda.max_memory_reserved(device) / (1024**2)
                            ),
                        }
                    )
                metrics_handle.write(
                    json.dumps(last_metrics, ensure_ascii=False, sort_keys=True) + "\n"
                )
                metrics_handle.flush()
                global_step += 1
                if global_step == 1 or global_step % args.log_every == 0:
                    print(json.dumps(last_metrics, ensure_ascii=False, sort_keys=True))
                if (
                    args.checkpoint_every > 0
                    and global_step % args.checkpoint_every == 0
                ):
                    _save_checkpoint_atomic(
                        _checkpoint_payload(
                            model=model,
                            optimizer=optimizer,
                            model_hparams=model_hparams,
                            global_step=global_step,
                            protocol=protocol,
                            last_metrics=last_metrics,
                        ),
                        args.out_dir / "latest.ckpt",
                    )
                if args.max_steps is not None and global_step >= args.max_steps:
                    break
            if args.max_steps is not None and global_step >= args.max_steps:
                break
    if global_step == 0 or last_metrics is None:
        raise RuntimeError("Training completed without an optimizer step")
    output_checkpoint = args.out_dir / "last.ckpt"
    _save_checkpoint_atomic(
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            model_hparams=model_hparams,
            global_step=global_step,
            protocol=protocol,
            last_metrics=last_metrics,
        ),
        output_checkpoint,
    )
    summary = {
        "schema": SCHEMA_VERSION,
        "steps_completed": global_step,
        "training_wall_seconds": time.perf_counter() - training_started,
        "last_metrics": last_metrics,
        "output_checkpoint": str(output_checkpoint.resolve()),
        "output_checkpoint_sha256": sha256_file(output_checkpoint),
        "metrics_jsonl": str(metrics_path.resolve()),
        "metrics_sha256": sha256_file(metrics_path),
        "large_training_started": False if args.max_steps is not None else None,
    }
    (args.out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "train"), required=True)
    parser.add_argument("--topk-jsonl", type=Path, default=DEFAULT_TOPK)
    parser.add_argument("--magma-labels-jsonl", type=Path, default=DEFAULT_MAGMA_LABELS)
    parser.add_argument("--examples-jsonl", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-decoys", type=int, default=5)
    parser.add_argument("--max-input-rows", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--example-offset",
        type=int,
        default=0,
        help="Bounded smoke/debug only: skip this many prepared examples.",
    )
    parser.add_argument(
        "--keep-duplicate-2d",
        action="store_true",
        help="Stereo-specific follow-up only: keep candidates sharing 2D connectivity.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1,
        help="Bounded by default. Pass a negative value to remove the step cap.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Atomically refresh latest.ckpt every N completed steps; 0 disables it.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print one metric record every N steps; metrics.jsonl remains per-step.",
    )
    parser.add_argument("--sampling", choices=("connectivity", "query"), default="connectivity")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--gradient-clip-algorithm",
        choices=("value", "norm"),
        default="value",
        help="Default matches the released GLACIER Lightning trainer.",
    )
    parser.add_argument("--gradient-clip-value", type=float, default=5.0)
    parser.add_argument("--spec-weight", type=float, default=1.0)
    parser.add_argument("--frag-weight", type=float, default=0.1)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument(
        "--rank-loss",
        choices=("listwise_ce", "native_sinkhorn"),
        default="listwise_ce",
    )
    parser.add_argument("--rank-temperature", type=float, default=0.05)
    parser.add_argument("--rank-margin", type=float, default=0.0)
    parser.add_argument("--native-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.top_k <= 1:
        parser.error("--top-k must exceed 1")
    if args.num_decoys <= 0 or args.num_decoys >= args.top_k:
        parser.error("--num-decoys must be in [1, top-k-1]")
    if args.batch_size <= 0 or args.epochs <= 0:
        parser.error("--batch-size and --epochs must be positive")
    if args.example_offset < 0:
        parser.error("--example-offset must be non-negative")
    if args.gradient_clip_value <= 0:
        parser.error("--gradient-clip-value must be positive")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be non-negative")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    if args.max_steps is not None and args.max_steps < 0:
        args.max_steps = None
    if args.examples_jsonl is None:
        args.examples_jsonl = args.out_dir / "examples.jsonl"
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        summary = prepare_examples(
            args.topk_jsonl,
            args.magma_labels_jsonl,
            args.examples_jsonl,
            top_k=args.top_k,
            num_decoys=args.num_decoys,
            max_input_rows=args.max_input_rows,
            dedup_2d_connectivity=not args.keep_duplicate_2d,
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.out_dir / "prepare_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        summary = run_training(args)
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
