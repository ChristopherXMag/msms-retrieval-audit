"""E1-F: explicit candidate-fragment residual on top of locked atom-node E1.

Scientific protocol
-------------------
* MassSpecGym v1.5 formula-group, mass candidates capped at 256.
* Frozen Run-K supplies the original top-50; GT is never inserted.
* The released raw atom-node E1 checkpoint is frozen and its selected alpha is
  used as the new baseline.
* Fragment enumeration is candidate-only and deterministic.  It uses connected
  components from one non-ring cut plus two cuts within the same ring, capped at
  a fixed number per parent candidate.  The query spectrum never changes the
  fragment set.
* A1 (composition control) and A2 (connectivity) use identical fragment masks,
  formula/exact-mass/size features, peak tokens, and per-peak Top-2-mean pooling.
  A2 alone receives frozen E1 parent-node mean/max pooling and cut-boundary
  features.

This is deliberately a bounded representation test.  Failure rejects this
finite cut set/tokenizer/scorer, not fragmentation representations in general.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.reranking.train_msalign_cross_attention_reranker import rank_metrics  # noqa: E402
from tools.reranking.train_run_k_flare_local_reranker import (  # noqa: E402
    ELEMENTS,
    LocalInteractionModel,
    PackedGraphStore,
    RerankDataset,
    TopKSplit,
    collect_local_residuals,
    json_safe,
    load_token_cache,
    paired_bootstrap,
    parse_float_grid,
)


DEFAULT_E1_DIR = REPO_ROOT / "outputs/run_k_flare_local_reranker/phase1_seed20260806"
DEFAULT_E1_CHECKPOINT = DEFAULT_E1_DIR / "raw_seed20260806.pt"
DEFAULT_TOPK_CACHE = DEFAULT_E1_DIR / "run_k_train20k_val_top50.pt"
DEFAULT_GRAPH_CACHE = DEFAULT_E1_DIR / "run_k_train20k_val_top50_graphs.pt"
DEFAULT_CHEM_CACHE = (
    REPO_ROOT
    / "outputs/msalign_molecule_ablation/chemberta_mtr_morgan_formula_group_20k_val_test_mass_cap256.pt"
)
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/run_k_e1_fragment_residual"

BOUNDARY_DIM = 7  # 5 bond-type counts + conjugated count + ring-bond count
MAX_FORMULA_ELEMENTS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-tokens", type=Path, default=Path("outputs/dreams_chemistry_adapter/train_sample20k_seed20260623.tokens.pt"))
    parser.add_argument("--val-tokens", type=Path, default=Path("outputs/dreams_chemistry_adapter/val.tokens.pt"))
    parser.add_argument("--topk-cache", type=Path, default=DEFAULT_TOPK_CACHE)
    parser.add_argument("--graph-cache", type=Path, default=DEFAULT_GRAPH_CACHE)
    parser.add_argument("--chem-cache", type=Path, default=DEFAULT_CHEM_CACHE)
    parser.add_argument("--e1-checkpoint", type=Path, default=DEFAULT_E1_CHECKPOINT)
    parser.add_argument("--e1-ranks", type=Path, default=DEFAULT_E1_DIR / "raw_seed20260806.val_ranks.pt")
    parser.add_argument("--fragment-cache", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mode", choices=["both", "composition", "connectivity", "cache-only"], default="both")
    parser.add_argument("--train-limit", type=int, default=4096)
    parser.add_argument("--val-limit", type=int, default=4096)
    parser.add_argument("--max-fragments", type=int, default=32)
    parser.add_argument("--fragment-top-k", type=int, default=2)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fragment-temperature", type=float, default=0.05)
    parser.add_argument("--fragment-aux-weight", type=float, default=0.1)
    parser.add_argument("--gamma-grid", default="0,0.25,0.5,1,2")
    parser.add_argument("--damage-target", type=float, default=0.08)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def _component_masks(mol: Chem.Mol, removed_bonds: tuple[int, ...]) -> list[tuple[int, ...]]:
    removed = set(removed_bonds)
    adjacency = [[] for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        if bond.GetIdx() in removed:
            continue
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[begin].append(end)
        adjacency[end].append(begin)
    unseen = set(range(mol.GetNumAtoms()))
    components: list[tuple[int, ...]] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        unseen.remove(root)
        component = []
        while stack:
            atom = stack.pop()
            component.append(atom)
            for neighbor in adjacency[atom]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return components


def enumerate_fragment_masks(mol: Chem.Mol, max_fragments: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return deterministic ``(atom mask, cut bonds)`` tokens for one parent."""
    atom_count = mol.GetNumAtoms()
    best_route: dict[tuple[int, ...], tuple[int, ...]] = {}

    def add(cuts: tuple[int, ...]) -> None:
        components = _component_masks(mol, cuts)
        if len(components) <= 1:
            return
        for component in components:
            if len(component) == atom_count:
                continue
            previous = best_route.get(component)
            if previous is None or (len(cuts), cuts) < (len(previous), previous):
                best_route[component] = cuts

    for bond in mol.GetBonds():
        if not bond.IsInRing():
            add((bond.GetIdx(),))

    # Two cuts are restricted to pairs sharing an RDKit SSSR ring.  This opens
    # cyclic systems without the quadratic all-bond-pair combinatorics.
    ring_pairs: set[tuple[int, int]] = set()
    for ring in Chem.GetSymmSSSR(mol):
        atoms = [int(value) for value in ring]
        ring_bonds = []
        for begin, end in zip(atoms, atoms[1:] + atoms[:1]):
            bond = mol.GetBondBetweenAtoms(begin, end)
            if bond is not None:
                ring_bonds.append(bond.GetIdx())
        for left_index, left in enumerate(ring_bonds):
            for right in ring_bonds[left_index + 1 :]:
                ring_pairs.add(tuple(sorted((left, right))))
    for cuts in sorted(ring_pairs):
        add(cuts)

    ordered = sorted(best_route.items(), key=lambda item: (len(item[0]), item[0], item[1]))
    if not ordered:
        return [(tuple(range(atom_count)), tuple())]

    one_cut = [item for item in ordered if len(item[1]) == 1]
    two_cut = [item for item in ordered if len(item[1]) == 2]
    # Reserve 1/4 of a capped set for ring-opening two-cut fragments whenever
    # available, then fill unused capacity from one-cut and finally two-cut.
    # Within each group, evenly spaced selection over the size-sorted list
    # avoids retaining only the smallest components.
    target_two = min(len(two_cut), max_fragments // 4)
    target_one = min(len(one_cut), max_fragments - target_two)
    remaining = max_fragments - target_one - target_two
    extra_one = min(len(one_cut) - target_one, remaining)
    target_one += extra_one
    remaining -= extra_one
    target_two += min(len(two_cut) - target_two, remaining)

    def evenly_select(items: list[tuple[tuple[int, ...], tuple[int, ...]]], count: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        if count <= 0:
            return []
        if count >= len(items):
            return items
        if count == 1:
            return [items[len(items) // 2]]
        indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
        return [items[index] for index in indices]

    return evenly_select(one_cut, target_one) + evenly_select(two_cut, target_two)


def fragment_formula_features(mol: Chem.Mol, atom_mask: tuple[int, ...], cuts: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    periodic_table = Chem.GetPeriodicTable()
    count_by_atomic_number: dict[int, int] = {}
    mass = 0.0
    for atom_index in atom_mask:
        atom = mol.GetAtomWithIdx(atom_index)
        atomic_number = atom.GetAtomicNum()
        count_by_atomic_number[atomic_number] = count_by_atomic_number.get(atomic_number, 0) + 1
        mass += float(periodic_table.GetMostCommonIsotopeMass(atomic_number))
        hydrogens = int(atom.GetTotalNumHs())
        count_by_atomic_number[1] = count_by_atomic_number.get(1, 0) + hydrogens
        mass += hydrogens * float(periodic_table.GetMostCommonIsotopeMass(1))
    if len(count_by_atomic_number) > MAX_FORMULA_ELEMENTS:
        raise ValueError(f"Fragment formula has {len(count_by_atomic_number)} element types")
    element_indices = np.zeros(MAX_FORMULA_ELEMENTS, dtype=np.int16)
    element_counts = np.zeros(MAX_FORMULA_ELEMENTS, dtype=np.int16)
    for slot, (atomic_number, count) in enumerate(sorted(count_by_atomic_number.items())):
        element_indices[slot] = atomic_number
        element_counts[slot] = count
    boundary = np.zeros(BOUNDARY_DIM, dtype=np.int8)
    for bond_index in cuts:
        bond = mol.GetBondWithIdx(bond_index)
        type_index = {
            Chem.BondType.SINGLE: 1,
            Chem.BondType.DOUBLE: 2,
            Chem.BondType.TRIPLE: 3,
            Chem.BondType.AROMATIC: 4,
        }.get(bond.GetBondType(), 0)
        boundary[type_index] += 1
        boundary[5] += int(bond.GetIsConjugated())
        boundary[6] += int(bond.IsInRing())
    return element_indices, element_counts, mass, boundary


def _tensor_digest(values: torch.Tensor) -> str:
    return hashlib.sha256(values.contiguous().numpy().tobytes()).hexdigest()


def build_fragment_cache(path: Path, global_indices: torch.Tensor, smiles: list[str], max_fragments: int) -> dict[str, Any]:
    fragment_offsets = [0]
    atom_offsets = [0]
    atom_indices: list[int] = []
    element_indices: list[np.ndarray] = []
    element_counts: list[np.ndarray] = []
    exact_masses: list[float] = []
    atom_counts: list[int] = []
    cut_counts: list[int] = []
    boundaries: list[np.ndarray] = []
    fallback_count = 0
    for global_index in tqdm(global_indices.tolist(), desc="Enumerating deterministic candidate fragments"):
        mol = Chem.MolFromSmiles(smiles[global_index])
        if mol is None:
            raise ValueError(f"RDKit failed for candidate {global_index}: {smiles[global_index]!r}")
        fragments = enumerate_fragment_masks(mol, max_fragments)
        fallback_count += int(len(fragments) == 1 and not fragments[0][1])
        for atom_mask, cuts in fragments:
            indices, counts, mass, boundary = fragment_formula_features(mol, atom_mask, cuts)
            atom_indices.extend(atom_mask)
            atom_offsets.append(len(atom_indices))
            element_indices.append(indices)
            element_counts.append(counts)
            exact_masses.append(mass)
            atom_counts.append(len(atom_mask))
            cut_counts.append(len(cuts))
            boundaries.append(boundary)
        fragment_offsets.append(len(element_indices))
    payload = {
        "metadata": {
            "schema": 2,
            "representation": "parent_masks_single_nonring_and_same_ring_double_cut_size_stratified_3to1",
            "candidate_only": True,
            "query_conditioned_selection": False,
            "max_fragments": max_fragments,
            "candidates": len(global_indices),
            "fragments": len(element_indices),
            "atom_memberships": len(atom_indices),
            "whole_parent_fallbacks": fallback_count,
            "global_indices_sha256": _tensor_digest(global_indices),
        },
        "global_indices": global_indices.to(torch.int32),
        "fragment_offsets": torch.tensor(fragment_offsets, dtype=torch.int64),
        "atom_offsets": torch.tensor(atom_offsets, dtype=torch.int64),
        "atom_indices": torch.tensor(atom_indices, dtype=torch.int16),
        "element_indices": torch.from_numpy(np.stack(element_indices)).to(torch.int16),
        "element_counts": torch.from_numpy(np.stack(element_counts)).to(torch.int16),
        "exact_mass": torch.tensor(exact_masses, dtype=torch.float32),
        "atom_count": torch.tensor(atom_counts, dtype=torch.int16),
        "cut_count": torch.tensor(cut_counts, dtype=torch.int8),
        "boundary": torch.from_numpy(np.stack(boundaries)).to(torch.int8),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


class PackedFragmentStore:
    def __init__(self, payload: dict[str, Any], total_candidates: int) -> None:
        self.metadata = payload["metadata"]
        self.global_indices = payload["global_indices"].long()
        self.fragment_offsets = payload["fragment_offsets"]
        self.atom_offsets = payload["atom_offsets"]
        self.atom_indices = payload["atom_indices"]
        self.element_indices = payload["element_indices"]
        self.element_counts = payload["element_counts"]
        self.exact_mass = payload["exact_mass"]
        self.atom_count = payload["atom_count"]
        self.cut_count = payload["cut_count"]
        self.boundary = payload["boundary"]
        self.max_fragments = int(self.metadata["max_fragments"])
        self.global_to_row = torch.full((total_candidates,), -1, dtype=torch.int32)
        self.global_to_row[self.global_indices] = torch.arange(len(self.global_indices), dtype=torch.int32)

    def gather(self, candidate_indices: torch.Tensor, graph_node_counts: torch.Tensor) -> dict[str, torch.Tensor]:
        unique_global, inverse = torch.unique(candidate_indices.long().cpu(), sorted=True, return_inverse=True)
        rows = self.global_to_row[unique_global].long()
        if bool((rows < 0).any()):
            raise KeyError(f"Fragment cache misses {unique_global[rows < 0][:5].tolist()}")
        if len(rows) != len(graph_node_counts):
            raise ValueError("Graph/fragment unique-candidate order mismatch")
        count = len(rows)
        width = self.max_fragments
        element_indices = torch.zeros((count, width, MAX_FORMULA_ELEMENTS), dtype=torch.long)
        element_counts = torch.zeros((count, width, MAX_FORMULA_ELEMENTS), dtype=torch.float32)
        scalar = torch.zeros((count, width, 3), dtype=torch.float32)
        boundary = torch.zeros((count, width, BOUNDARY_DIM), dtype=torch.float32)
        mask = torch.zeros((count, width), dtype=torch.bool)
        atom_node_indices: list[torch.Tensor] = []
        atom_fragment_indices: list[torch.Tensor] = []
        node_bases = torch.cumsum(graph_node_counts, dim=0) - graph_node_counts
        for output_row, cache_row in enumerate(rows.tolist()):
            start = int(self.fragment_offsets[cache_row])
            end = int(self.fragment_offsets[cache_row + 1])
            fragment_ids = torch.arange(start, end)
            length = len(fragment_ids)
            mask[output_row, :length] = True
            element_indices[output_row, :length] = self.element_indices[fragment_ids].long()
            element_counts[output_row, :length] = self.element_counts[fragment_ids].float()
            scalar[output_row, :length, 0] = self.exact_mass[fragment_ids] / 1000.0
            scalar[output_row, :length, 1] = self.atom_count[fragment_ids].float() / 100.0
            scalar[output_row, :length, 2] = self.cut_count[fragment_ids].float() / 2.0
            boundary[output_row, :length] = self.boundary[fragment_ids].float() / 2.0
            for local_fragment, fragment_id in enumerate(fragment_ids.tolist()):
                atom_start = int(self.atom_offsets[fragment_id])
                atom_end = int(self.atom_offsets[fragment_id + 1])
                local_atoms = self.atom_indices[atom_start:atom_end].long()
                atom_node_indices.append(local_atoms + node_bases[output_row])
                atom_fragment_indices.append(torch.full((len(local_atoms),), output_row * width + local_fragment, dtype=torch.long))
        return {
            "element_indices": element_indices,
            "element_counts": element_counts,
            "scalar": scalar,
            "boundary": boundary,
            "fragment_mask": mask,
            "atom_node_indices": torch.cat(atom_node_indices),
            "atom_fragment_indices": torch.cat(atom_fragment_indices),
            "inverse": inverse,
        }


def load_fragment_store(path: Path, global_indices: torch.Tensor, smiles: list[str], max_fragments: int) -> PackedFragmentStore:
    digest = _tensor_digest(global_indices)
    if path.exists():
        payload = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
        metadata = payload.get("metadata") or {}
        if int(metadata.get("schema", -1)) != 2 or metadata.get("global_indices_sha256") != digest or int(metadata.get("max_fragments", -1)) != max_fragments:
            raise ValueError(f"Stale/incompatible fragment cache: {path}")
    else:
        payload = build_fragment_cache(path, global_indices, smiles, max_fragments)
    return PackedFragmentStore(payload, len(smiles))


def subset_topk(split: TopKSplit, limit: int, seed: int) -> tuple[TopKSplit, torch.Tensor]:
    total = len(split.baseline_ranks)
    if limit <= 0 or limit >= total:
        selected = torch.arange(total)
    else:
        selected = torch.tensor(sorted(random.Random(seed).sample(range(total), limit)), dtype=torch.long)
    selected_lookup = torch.full((total,), -1, dtype=torch.long)
    selected_lookup[selected] = torch.arange(len(selected))
    keep = selected_lookup[split.order_indices.long()] >= 0
    remapped_orders = selected_lookup[split.order_indices[keep].long()].to(torch.int32)
    stats = dict(split.stats) | {
        "raw_subset_size": len(selected),
        "rerankable_usable": int(keep.sum()),
        "gt_not_in_top_k": int(len(selected) - keep.sum()),
        "subset_seed": seed,
    }
    output = TopKSplit(
        identifiers=[split.identifiers[index] for index in selected.tolist()],
        baseline_ranks=split.baseline_ranks[selected].clone(),
        order_indices=remapped_orders,
        spec_indices=split.spec_indices[keep].clone(),
        candidate_indices=split.candidate_indices[keep].clone(),
        candidate_mask=split.candidate_mask[keep].clone(),
        stage1_scores=split.stage1_scores[keep].clone(),
        gt_indices=split.gt_indices[keep].clone(),
        stage1_ranks=split.stage1_ranks[keep].clone(),
        stats=stats,
    )
    return output, selected


@torch.no_grad()
def encode_e1_features(e1: LocalInteractionModel, tokens: torch.Tensor, masks: torch.Tensor, spec_indices: torch.Tensor, graphs: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    peaks = F.normalize(e1.peak_projection(tokens[spec_indices.long(), 1:].float().to(device)), dim=-1, eps=1e-6)
    peak_mask = masks[spec_indices.long(), 1:].to(device)
    nodes = e1.atom_encoder(graphs["node_features"])
    edges = e1.bond_encoder(graphs["edge_features"])
    for layer in e1.gnn:
        nodes = layer(nodes, graphs["edge_index"], edges)
    nodes = F.normalize(nodes, dim=-1, eps=1e-6)
    return peaks, peak_mask, nodes


class FragmentResidualModel(nn.Module):
    def __init__(self, dim: int, dropout: float, connectivity: bool, fragment_top_k: int) -> None:
        super().__init__()
        self.connectivity = connectivity
        self.fragment_top_k = fragment_top_k
        self.element_embedding = nn.Embedding(119, 64, padding_idx=0)
        self.composition = nn.Sequential(
            nn.Linear(64 + 3, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim)
        )
        if connectivity:
            self.structure = nn.Sequential(
                nn.Linear(2 * dim + BOUNDARY_DIM, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim)
            )
            self.structure_gate = nn.Parameter(torch.tensor(0.1))
        else:
            self.structure = None
            self.register_parameter("structure_gate", None)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, peaks: torch.Tensor, peak_mask: torch.Tensor, fragments: dict[str, torch.Tensor], nodes: torch.Tensor, candidate_shape: tuple[int, int]) -> torch.Tensor:
        element_vectors = self.element_embedding(fragments["element_indices"])
        counts = fragments["element_counts"]
        formula = (element_vectors * torch.log1p(counts)[..., None]).sum(-2)
        formula = formula / (counts > 0).sum(-1).clamp_min(1)[..., None]
        tokens = self.composition(torch.cat((formula, fragments["scalar"]), dim=-1))
        if self.connectivity:
            if self.structure is None or self.structure_gate is None:
                raise RuntimeError("Missing structural branch")
            total_fragments = fragments["fragment_mask"].numel()
            dim = nodes.shape[-1]
            membership_nodes = nodes[fragments["atom_node_indices"]]
            membership_fragments = fragments["atom_fragment_indices"]
            pooled_sum = nodes.new_zeros((total_fragments, dim))
            pooled_sum.index_add_(0, membership_fragments, membership_nodes)
            counts = torch.bincount(membership_fragments, minlength=total_fragments).to(nodes.dtype).clamp_min(1).unsqueeze(-1)
            pooled_mean = pooled_sum / counts
            pooled_max = nodes.new_full((total_fragments, dim), -1e4)
            pooled_max.scatter_reduce_(0, membership_fragments[:, None].expand(-1, dim), membership_nodes, reduce="amax", include_self=True)
            pooled_max = torch.where(counts > 0, pooled_max, torch.zeros_like(pooled_max))
            shape = (*fragments["fragment_mask"].shape, dim)
            structure_input = torch.cat((pooled_mean.reshape(shape), pooled_max.reshape(shape), fragments["boundary"]), dim=-1)
            tokens = tokens + self.structure_gate * self.structure(structure_input)
        tokens = F.normalize(tokens, dim=-1, eps=1e-6)
        batch_size, candidate_count = candidate_shape
        candidate_tokens = tokens[fragments["inverse"]].reshape(batch_size, candidate_count, tokens.shape[1], -1)
        candidate_fragment_mask = fragments["fragment_mask"][fragments["inverse"]].reshape(batch_size, candidate_count, tokens.shape[1])
        return masked_topk_mean_similarity(peaks, peak_mask, candidate_tokens, candidate_fragment_mask, self.fragment_top_k)


def masked_topk_mean_similarity(peaks: torch.Tensor, peak_mask: torch.Tensor, candidate_tokens: torch.Tensor, candidate_fragment_mask: torch.Tensor, fragment_top_k: int) -> torch.Tensor:
    """Average each peak's top-k valid fragment cosine similarities."""
    similarity = torch.einsum("bpd,bkfd->bkpf", peaks, candidate_tokens)
    valid = peak_mask[:, None, :, None] & candidate_fragment_mask[:, :, None, :]
    similarity = similarity.masked_fill(~valid, -1e4)
    top_k = min(fragment_top_k, similarity.shape[-1])
    top_values = similarity.topk(top_k, dim=-1).values
    valid_counts = candidate_fragment_mask.sum(-1).clamp(min=1, max=top_k)
    top_valid = torch.arange(top_k, device=similarity.device)[None, None, None, :] < valid_counts[:, :, None, None]
    peak_scores = (top_values * top_valid).sum(-1) / valid_counts.to(similarity.dtype)[:, :, None]
    peak_scores = torch.where(peak_mask[:, None], peak_scores, torch.zeros_like(peak_scores))
    return peak_scores.sum(-1) / peak_mask.sum(-1).clamp_min(1)[:, None]


def move_tensors(payload: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in payload.items()}


def prepare_fragment_batch(split: TopKSplit, indices: torch.Tensor, tokens: torch.Tensor, masks: torch.Tensor, graph_store: PackedGraphStore, fragment_store: PackedFragmentStore, e1: LocalInteractionModel, device: torch.device) -> dict[str, Any]:
    candidate_mask = split.candidate_mask[indices]
    candidates = split.candidate_indices[indices].long()
    safe_candidates = candidates.clone()
    safe_candidates[~candidate_mask] = candidates[candidate_mask][0]
    graphs = move_tensors(graph_store.gather(safe_candidates.reshape(-1)), device)
    fragments = fragment_store.gather(safe_candidates.reshape(-1), graphs["node_counts"].cpu())
    fragments = move_tensors(fragments, device)
    spec_indices = split.spec_indices[indices].long()
    peaks, peak_mask, nodes = encode_e1_features(e1, tokens, masks, spec_indices, graphs, device)
    return {
        "peaks": peaks,
        "peak_mask": peak_mask,
        "nodes": nodes,
        "fragments": fragments,
        "candidate_shape": tuple(candidates.shape),
        "candidate_mask": candidate_mask.to(device),
        "base_scores": split.stage1_scores[indices].to(device),
        "labels": split.gt_indices[indices].long().to(device),
    }


def train_epoch(model: FragmentResidualModel, split: TopKSplit, tokens: torch.Tensor, masks: torch.Tensor, graph_store: PackedGraphStore, fragment_store: PackedFragmentStore, e1: LocalInteractionModel, optimizer: torch.optim.Optimizer, device: torch.device, batch_size: int, num_workers: int, temperature: float, aux_weight: float) -> dict[str, Any]:
    model.train()
    loader = DataLoader(RerankDataset(len(split.order_indices)), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    total_loss = 0.0
    examples = 0
    for indices in tqdm(loader, desc="Training E1-F residual"):
        batch = prepare_fragment_batch(split, indices.long(), tokens, masks, graph_store, fragment_store, e1, device)
        scores = model(batch["peaks"], batch["peak_mask"], batch["fragments"], batch["nodes"], batch["candidate_shape"])
        raw_logits = scores / temperature
        fragment_logits = raw_logits.masked_fill(~batch["candidate_mask"], float("-inf"))
        fused = (batch["base_scores"] + model.residual_scale * raw_logits).masked_fill(~batch["candidate_mask"], float("-inf"))
        loss = F.cross_entropy(fused, batch["labels"]) + aux_weight * F.cross_entropy(fragment_logits, batch["labels"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite E1-F loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            model.residual_scale.clamp_(min=0.0)
        total_loss += float(loss.detach().cpu()) * len(indices)
        examples += len(indices)
    return {"loss": total_loss / max(examples, 1), "examples": examples, "residual_scale": float(model.residual_scale.detach().cpu())}


@torch.no_grad()
def collect_residuals(model: FragmentResidualModel, split: TopKSplit, tokens: torch.Tensor, masks: torch.Tensor, graph_store: PackedGraphStore, fragment_store: PackedFragmentStore, e1: LocalInteractionModel, device: torch.device, batch_size: int, temperature: float) -> torch.Tensor:
    model.eval()
    output = torch.zeros_like(split.stage1_scores)
    loader = DataLoader(RerankDataset(len(split.order_indices)), batch_size=batch_size, shuffle=False)
    for indices in tqdm(loader, desc="Scoring E1-F residual"):
        indices = indices.long()
        batch = prepare_fragment_batch(split, indices, tokens, masks, graph_store, fragment_store, e1, device)
        scores = model(batch["peaks"], batch["peak_mask"], batch["fragments"], batch["nodes"], batch["candidate_shape"])
        output[indices] = (model.residual_scale * scores / temperature).cpu()
    return output


def evaluate(split: TopKSplit, residuals: torch.Tensor, gamma: float) -> dict[str, Any]:
    ranks = split.baseline_ranks.clone().long()
    rank1_total = 0
    rank1_damaged = 0
    for row in range(len(split.order_indices)):
        valid = split.candidate_mask[row]
        scores = split.stage1_scores[row, valid] + gamma * residuals[row, valid]
        order = torch.argsort(scores, descending=True)
        rank = int((order == int(split.gt_indices[row])).nonzero(as_tuple=False)[0]) + 1
        ranks[int(split.order_indices[row])] = rank
        if int(split.stage1_ranks[row]) == 1:
            rank1_total += 1
            rank1_damaged += int(rank > 1)
    return {**rank_metrics(ranks.tolist()), "gamma": gamma, "rank1_total": rank1_total, "rank1_damage_rate": rank1_damaged / max(rank1_total, 1), "ranks": ranks}


def select_policy(results: dict[float, dict[str, Any]], damage_target: float) -> float:
    eligible = [(gamma, result) for gamma, result in results.items() if result["rank1_damage_rate"] <= damage_target]
    if not eligible:
        eligible = list(results.items())
    return max(eligible, key=lambda item: (item[1]["mrr"], item[1]["recall@1"], item[1]["recall@20"], -item[1]["rank1_damage_rate"], -item[0]))[0]


def incremental_gate(bootstrap: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    r1 = bootstrap["recall@1"]
    mrr = bootstrap["mrr"]
    triggers = []
    if r1["delta_percentage_points"] >= 0.5 and r1["probability_positive"] >= 0.8:
        triggers.append("recall@1")
    if mrr["delta_percentage_points"] >= 0.3 and mrr["probability_positive"] >= 0.8:
        triggers.append("mrr")
    other_ok = ("recall@1" in triggers and mrr["delta_percentage_points"] >= -0.1) or ("mrr" in triggers and r1["delta_percentage_points"] >= -0.1)
    checks = {
        "r1_ge_0_5pp_or_mrr_ge_0_3pp_with_p_positive_ge_0_8": bool(triggers),
        "other_primary_not_below_minus_0_1pp": other_ok,
        "recall20_not_below_minus_0_2pp": bootstrap["recall@20"]["delta_percentage_points"] >= -0.2,
        "rank1_damage_at_most_8pct": selected["rank1_damage_rate"] <= 0.08,
    }
    return {"passed": all(checks.values()), "triggering_metrics": triggers, "checks": checks}


def make_e1_baseline(original: TopKSplit, residuals: torch.Tensor, alpha: float) -> tuple[TopKSplit, dict[str, Any]]:
    evaluated = evaluate(original, residuals, alpha)
    scores = original.stage1_scores + alpha * residuals
    rerankable_ranks = torch.empty_like(original.stage1_ranks)
    for row in range(len(original.order_indices)):
        valid_scores = scores[row, original.candidate_mask[row]]
        order = torch.argsort(valid_scores, descending=True)
        rerankable_ranks[row] = int((order == int(original.gt_indices[row])).nonzero(as_tuple=False)[0]) + 1
    return replace(original, baseline_ranks=evaluated["ranks"], stage1_scores=scores, stage1_ranks=rerankable_ranks), evaluated


def run_arm(name: str, args: argparse.Namespace, splits: dict[str, TopKSplit], tokens: dict[str, torch.Tensor], masks: dict[str, torch.Tensor], graph_store: PackedGraphStore, fragment_store: PackedFragmentStore, e1: LocalInteractionModel, device: torch.device, common: dict[str, Any]) -> dict[str, Any]:
    # Match initialization and DataLoader order across A1/A2.  A2 has extra
    # parameters, but the shared formula branch sees the same random stream.
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = FragmentResidualModel(args.dim, args.dropout, name == "connectivity", args.fragment_top_k).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gamma_grid = parse_float_grid(args.gamma_grid)
    baseline = rank_metrics(splits["val"].baseline_ranks.tolist())
    zero = torch.zeros_like(splits["val"].stage1_scores)
    if any(evaluate(splits["val"], zero, gamma)["ranks"].tolist() != splits["val"].baseline_ranks.tolist() for gamma in gamma_grid):
        raise RuntimeError("Epoch-0 E1 parity failed")
    history = [{"epoch": 0, "selected_gamma": 0.0, "val_selected": baseline, "exact_locked_e1_fallback": True}]
    best_mrr = float(baseline["mrr"])
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(model, splits["train"], tokens["train"], masks["train"], graph_store, fragment_store, e1, optimizer, device, args.batch_size, args.num_workers, args.fragment_temperature, args.fragment_aux_weight)
        residuals = collect_residuals(model, splits["val"], tokens["val"], masks["val"], graph_store, fragment_store, e1, device, args.eval_batch_size, args.fragment_temperature)
        grid = {gamma: evaluate(splits["val"], residuals, gamma) for gamma in gamma_grid}
        selected_gamma = select_policy(grid, args.damage_target)
        selected = grid[selected_gamma]
        entry = {"epoch": epoch, "train": training, "selected_gamma": selected_gamma, "val_selected": {key: value for key, value in selected.items() if key != "ranks"}}
        history.append(entry)
        print(json.dumps({"arm": name, **entry}, sort_keys=True), flush=True)
        if float(selected["mrr"]) > best_mrr:
            best_mrr = float(selected["mrr"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    residuals = collect_residuals(model, splits["val"], tokens["val"], masks["val"], graph_store, fragment_store, e1, device, args.eval_batch_size, args.fragment_temperature)
    grid = {gamma: evaluate(splits["val"], residuals, gamma) for gamma in gamma_grid}
    selected_gamma = select_policy(grid, args.damage_target)
    selected = grid[selected_gamma]
    bootstrap = paired_bootstrap(splits["val"].baseline_ranks, selected["ranks"], args.bootstrap_samples, seed + (10 if name == "composition" else 11))
    output = {
        **common,
        "arm": name,
        "scientific_role": "formula/exact-mass/size matched control" if name == "composition" else "composition plus parent-contextualized connectivity/boundary",
        "history": history,
        "best_epoch": best_epoch,
        "selected_gamma": selected_gamma,
        "val_locked_e1": baseline,
        "val_selected": {key: value for key, value in selected.items() if key != "ranks"},
        "paired_bootstrap_vs_e1": bootstrap,
        "incremental_gate": incremental_gate(bootstrap, selected),
    }
    stem = f"{name}_seed{args.seed}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "best_epoch": best_epoch, "selected_gamma": selected_gamma, "config": common, "arm": name}, args.out_dir / f"{stem}.pt")
    torch.save({"identifiers": splits["val"].identifiers, "locked_e1_ranks": splits["val"].baseline_ranks, "selected_ranks": selected["ranks"]}, args.out_dir / f"{stem}.val_ranks.pt")
    (args.out_dir / f"{stem}.json").write_text(json.dumps(json_safe(output), indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.max_fragments < 1 or args.fragment_top_k < 1 or args.fragment_top_k > args.max_fragments:
        raise ValueError("Invalid epochs/max-fragments/fragment-top-k")
    started = time.time()
    args.out_dir = args.out_dir.resolve()
    device = choose_device(args.device)
    train_tokens = load_token_cache(args.train_tokens)
    val_tokens = load_token_cache(args.val_tokens)
    full_payload = torch.load(str(args.topk_cache), map_location="cpu", weights_only=False, mmap=True)
    full_splits = {name: TopKSplit.from_payload(full_payload[name]) for name in ("train", "val")}
    splits = {}
    selected_indices = {}
    for name, limit, seed in (("train", args.train_limit, args.seed), ("val", args.val_limit, args.seed + 1)):
        splits[name], selected_indices[name] = subset_topk(full_splits[name], limit, seed)
    token_payloads = {"train": train_tokens["tokens"], "val": val_tokens["tokens"]}
    mask_payloads = {"train": train_tokens["mask"].bool(), "val": val_tokens["mask"].bool()}

    graph_payload = torch.load(str(args.graph_cache), map_location="cpu", weights_only=False, mmap=True)
    chem_payload = torch.load(str(args.chem_cache), map_location="cpu", weights_only=False, mmap=True)
    smiles = [str(value) for value in chem_payload["smiles"]]
    graph_store = PackedGraphStore(graph_payload, len(smiles))
    unique_candidates = torch.unique(torch.cat([split.candidate_indices[split.candidate_mask].long() for split in splits.values()]), sorted=True)
    fragment_path = args.fragment_cache or args.out_dir / f"fragments_train{args.train_limit}_val{args.val_limit}_max{args.max_fragments}_seed{args.seed}.pt"
    fragment_store = load_fragment_store(fragment_path, unique_candidates, smiles, args.max_fragments)
    print(json.dumps({"subset": {name: split.stats for name, split in splits.items()}, "fragment_cache": fragment_store.metadata}, indent=2), flush=True)
    if args.mode == "cache-only":
        return

    checkpoint = torch.load(str(args.e1_checkpoint), map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint["config"]["args"]
    e1 = LocalInteractionModel(int(checkpoint_args["local_dim"]), int(checkpoint_args["gnn_layers"]), float(checkpoint_args["dropout"]), False).to(device)
    e1.load_state_dict(checkpoint["model_state_dict"])
    e1.eval()
    for parameter in e1.parameters():
        parameter.requires_grad_(False)
    e1_alpha = float(checkpoint["selected_alpha"])
    original_splits = splits
    e1_splits = {}
    e1_metrics = {}
    for name in ("train", "val"):
        residuals = collect_local_residuals(e1, original_splits[name], token_payloads[name], mask_payloads[name], graph_store, None, device, args.eval_batch_size, float(checkpoint_args["local_temperature"]))
        e1_splits[name], evaluated = make_e1_baseline(original_splits[name], residuals, e1_alpha)
        e1_metrics[name] = {key: value for key, value in evaluated.items() if key != "ranks"}

    parity = None
    if args.e1_ranks.exists():
        locked = torch.load(str(args.e1_ranks), map_location="cpu", weights_only=False)["selected_ranks"]
        expected = locked[selected_indices["val"]]
        actual = e1_splits["val"].baseline_ranks
        parity = {"exact": bool(torch.equal(expected.long(), actual.long())), "mismatches": int((expected.long() != actual.long()).sum())}
        if not parity["exact"]:
            raise RuntimeError(f"Locked E1 rank parity failed: {parity}")

    common = {
        "protocol": "MassSpecGym v1.5 formula-group; cap256; original Run-K top50; no GT insertion; frozen Run-K/DreaMS/atom-E1; candidate-only deterministic finite-cut fragments; per-peak Top-2 mean",
        "interpretation_boundary": "Failure rejects only the finite one-cut/same-ring-two-cut masks and current tokenization/scorer, not fragmentation representations in general.",
        "args": json_safe(vars(args)),
        "locked_e1": {"checkpoint": str(args.e1_checkpoint), "selected_alpha": e1_alpha, "metrics": e1_metrics, "rank_parity": parity},
        "subset": {name: split.stats for name, split in e1_splits.items()},
        "fragment_cache": {"path": str(fragment_path), **fragment_store.metadata},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"common_seed{args.seed}.json").write_text(json.dumps(json_safe(common), indent=2, sort_keys=True) + "\n")
    outputs = {}
    arms = ["composition", "connectivity"] if args.mode == "both" else [args.mode]
    for arm in arms:
        outputs[arm] = run_arm(arm, args, e1_splits, token_payloads, mask_payloads, graph_store, fragment_store, e1, device, common)
    if set(outputs) == {"composition", "connectivity"}:
        composition_ranks = torch.load(str(args.out_dir / f"composition_seed{args.seed}.val_ranks.pt"), map_location="cpu", weights_only=False)["selected_ranks"]
        connectivity_ranks = torch.load(str(args.out_dir / f"connectivity_seed{args.seed}.val_ranks.pt"), map_location="cpu", weights_only=False)["selected_ranks"]
        attribution = paired_bootstrap(composition_ranks, connectivity_ranks, args.bootstrap_samples, args.seed + 99)
        report = {
            "composition_vs_locked_e1": outputs["composition"]["paired_bootstrap_vs_e1"],
            "connectivity_vs_locked_e1": outputs["connectivity"]["paired_bootstrap_vs_e1"],
            "connectivity_vs_composition": attribution,
            "connectivity_attribution_passed": attribution["mrr"]["delta_percentage_points"] >= 0.3 and attribution["mrr"]["probability_positive"] >= 0.8 and attribution["recall@1"]["delta_percentage_points"] >= -0.1 and attribution["recall@20"]["delta_percentage_points"] >= -0.2,
            "elapsed_seconds": time.time() - started,
        }
        (args.out_dir / f"comparison_seed{args.seed}.json").write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n")
        print(json.dumps(json_safe(report), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
