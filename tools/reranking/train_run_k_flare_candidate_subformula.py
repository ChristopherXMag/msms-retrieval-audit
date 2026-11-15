"""Deployable candidate-conditioned subformula extension for Run-K + E1.

This module is called by ``train_run_k_flare_local_reranker.py``.  It keeps the
validated raw E1 scorer frozen and trains only a candidate-formula conditioning
residual.  Query ground-truth formulae are never consumed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glmr.subformula import (
    ATOMIC_NUMBER_TO_INDEX,
    BoundedFormulaDecomposer,
    ELEMENTS,
)
from tools.reranking.train_run_k_flare_local_reranker import (
    LocalInteractionModel,
    PackedGraphStore,
    RerankDataset,
    TopKSplit,
    json_safe,
    load_peak_lists,
    paired_bootstrap,
    prepare_batch,
    rank_metrics,
    selected_dreams_peaks,
    sha256_file,
)


FORMULA_NORM = torch.tensor(
    [102.0, 59.0, 25.0, 13.0, 3.0, 6.0, 6.0, 17.0, 4.0, 4.0, 1.0, 1.0, 5.0, 2.0],
    dtype=torch.float32,
)
ISOTOPE_PATTERN = re.compile(r"\[\d+[A-Z]")


def _graph_formula_table(
    graph_store: PackedGraphStore, smiles: list[str]
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Derive formula groups from the already audited packed RDKit graphs."""
    graph_count = len(graph_store.global_indices)
    counts = torch.zeros((graph_count, len(ELEMENTS)), dtype=torch.int16)
    unsupported = torch.zeros(graph_count, dtype=torch.bool)
    offsets = graph_store.node_offsets.long()
    nodes = graph_store.node_features.long()
    chunk = 100_000
    for start in tqdm(range(0, graph_count, chunk), desc="Deriving candidate formulae"):
        end = min(start + chunk, graph_count)
        node_start, node_end = int(offsets[start]), int(offsets[end])
        node_counts = offsets[start + 1 : end + 1] - offsets[start:end]
        rows = torch.repeat_interleave(torch.arange(end - start), node_counts)
        local = torch.zeros((end - start, len(ELEMENTS)), dtype=torch.int32)
        atomic_numbers = nodes[node_start:node_end, 0]
        for atomic_number, element_index in ATOMIC_NUMBER_TO_INDEX.items():
            local[:, element_index].index_add_(0, rows, (atomic_numbers == atomic_number).int())
        local[:, 0].index_add_(0, rows, nodes[node_start:node_end, 4].int())
        bad_atoms = ~torch.isin(
            atomic_numbers,
            torch.tensor(sorted(ATOMIC_NUMBER_TO_INDEX), dtype=atomic_numbers.dtype),
        )
        bad_per_graph = torch.zeros(end - start, dtype=torch.int32)
        bad_per_graph.index_add_(0, rows, bad_atoms.int())
        unsupported[start:end] = bad_per_graph > 0
        unsupported[start:end] |= (local > 255).any(dim=1)
        counts[start:end] = local.clamp_max(255).to(torch.int16)
    isotope_count = 0
    for row, global_index in enumerate(graph_store.global_indices.tolist()):
        if ISOTOPE_PATTERN.search(smiles[int(global_index)]):
            unsupported[row] = True
            isotope_count += 1
    supported_rows = (~unsupported).nonzero(as_tuple=False).flatten()
    formula_counts, inverse = torch.unique(
        counts[supported_rows], dim=0, sorted=True, return_inverse=True
    )
    global_to_formula = torch.full((len(smiles),), -1, dtype=torch.int32)
    global_to_formula[graph_store.global_indices[supported_rows].long()] = inverse.to(torch.int32)
    return global_to_formula, formula_counts.to(torch.uint8), {
        "candidate_graphs": graph_count,
        "supported_candidate_graphs": int(len(supported_rows)),
        "unsupported_candidate_graphs": int(unsupported.sum()),
        "isotope_labelled_candidate_graphs": isotope_count,
        "unique_supported_formulae": int(len(formula_counts)),
    }


def _make_layout(
    split: TopKSplit,
    records: list[dict[str, Any]],
    masks: torch.Tensor,
    global_to_formula: torch.Tensor,
    peak_lists: dict[str, tuple[list[float], list[float]]],
) -> dict[str, Any]:
    rows, width = split.candidate_indices.shape
    candidate_group = torch.full((rows, width), -1, dtype=torch.int8)
    group_offsets = [0]
    group_formula_ids: list[int] = []
    group_query_rows: list[int] = []
    group_entry_offsets = [0]
    row_peaks: list[np.ndarray] = []
    row_precursor: list[float] = []
    row_adduct: list[str] = []
    parity_checks = 0
    unsupported_slots = 0
    valid_slots = 0
    for row in tqdm(range(rows), desc="Grouping top50 candidate formulae"):
        spec_index = int(split.spec_indices[row])
        record = records[spec_index]
        identifier = str(record["identifier"])
        mzs, intensities = peak_lists[identifier]
        peaks = selected_dreams_peaks(mzs, intensities)
        expected = int(masks[spec_index, 1:].sum())
        if len(peaks) != expected:
            raise ValueError(
                f"DreaMS peak/token parity failed for {identifier}: peaks={len(peaks)} tokens={expected}"
            )
        parity_checks += 1
        row_peaks.append(peaks)
        row_precursor.append(float(record["precursor_mz"]))
        row_adduct.append(str(record["adduct"]))
        local: dict[int, int] = {}
        for column in range(width):
            if not bool(split.candidate_mask[row, column]):
                continue
            valid_slots += 1
            global_index = int(split.candidate_indices[row, column])
            formula_id = int(global_to_formula[global_index])
            if formula_id < 0:
                unsupported_slots += 1
                continue
            if formula_id not in local:
                local[formula_id] = len(local)
                group_formula_ids.append(formula_id)
                group_query_rows.append(row)
                group_entry_offsets.append(group_entry_offsets[-1] + len(peaks))
            candidate_group[row, column] = local[formula_id]
        group_offsets.append(len(group_formula_ids))
    return {
        "candidate_group": candidate_group,
        "group_offsets": torch.tensor(group_offsets, dtype=torch.int64),
        "group_formula_ids": torch.tensor(group_formula_ids, dtype=torch.int32),
        "group_query_rows": torch.tensor(group_query_rows, dtype=torch.int32),
        "group_entry_offsets": torch.tensor(group_entry_offsets, dtype=torch.int64),
        "row_peaks": row_peaks,
        "row_precursor": row_precursor,
        "row_adduct": row_adduct,
        "stats": {
            "rerankable_queries": rows,
            "formula_groups": len(group_formula_ids),
            "packed_group_peak_entries": group_entry_offsets[-1],
            "peak_token_parity_checks": parity_checks,
            "valid_candidate_slots": valid_slots,
            "unsupported_candidate_slots": unsupported_slots,
            "unsupported_candidate_fraction": unsupported_slots / max(valid_slots, 1),
        },
    }


def _allocate_layout_features(layout: dict[str, Any], top_n: int) -> dict[str, np.ndarray]:
    entries = int(layout["group_entry_offsets"][-1])
    return {
        "counts": np.zeros((entries, top_n, len(ELEMENTS)), dtype=np.uint8),
        "ppm": np.full((entries, top_n), np.inf, dtype=np.float16),
        "route": np.zeros((entries, top_n), dtype=np.uint8),
        "ambiguity": np.zeros(entries, dtype=np.uint16),
        "direct_count": np.zeros(entries, dtype=np.uint16),
        "neutral_loss_count": np.zeros(entries, dtype=np.uint16),
        "group_compatibility": np.zeros(len(layout["group_formula_ids"]), dtype=np.float32),
    }


def _cache_expected_metadata(
    args: Any,
    graph_store: PackedGraphStore,
    splits: dict[str, TopKSplit],
) -> dict[str, Any]:
    graph_digest = hashlib.sha256(graph_store.global_indices.numpy().tobytes()).hexdigest()
    return {
        "schema": 1,
        "kind": "deployable_candidate_formula_conditioned_subformula",
        "ppm_tolerance": float(args.subformula_ppm),
        "top_n": int(args.subformula_top_n),
        "elements": list(ELEMENTS),
        "routes": {"direct": 1, "neutral_loss": 2, "both": 3},
        "graph_global_indices_sha256": graph_digest,
        "train_rows": len(splits["train"].order_indices),
        "val_rows": len(splits["val"].order_indices),
        "candidate_formula_source": "candidate RDKit graph/SMILES only; no query GT formula",
        "supported_adducts": ["[M+H]+", "[M+Na]+"],
    }


def _validate_cache(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    metadata = payload.get("metadata") or {}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Stale candidate-subformula cache metadata {key}: {metadata.get(key)!r} != {value!r}")
    for split_name in ("train", "val"):
        split = payload[split_name]
        entries = int(split["group_entry_offsets"][-1])
        top_n = int(expected["top_n"])
        if tuple(split["counts"].shape) != (entries, top_n, len(ELEMENTS)):
            raise ValueError(f"Malformed {split_name} candidate-subformula counts")
        if tuple(split["ppm"].shape) != (entries, top_n) or tuple(split["route"].shape) != (entries, top_n):
            raise ValueError(f"Malformed {split_name} candidate-subformula ppm/route")
        valid = split["route"] > 0
        if bool((split["counts"][valid].sum(dim=-1) == 0).any()):
            raise ValueError(f"{split_name} contains empty valid subformulae")
        if bool((split["ppm"][valid].float() > float(expected["ppm_tolerance"]) + 0.05).any()):
            raise ValueError(f"{split_name} contains out-of-tolerance subformulae")


def load_or_build_candidate_subformula_cache(
    path: Path,
    args: Any,
    splits: dict[str, TopKSplit],
    records_by_split: dict[str, list[dict[str, Any]]],
    masks_by_split: dict[str, torch.Tensor],
    token_metadata: dict[str, Any],
    graph_store: PackedGraphStore,
    smiles: list[str],
) -> dict[str, Any]:
    expected = _cache_expected_metadata(args, graph_store, splits)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        _validate_cache(payload, expected)
        return payload
    started = time.time()
    global_to_formula, formula_counts, formula_stats = _graph_formula_table(graph_store, smiles)
    identifiers = {
        str(row["identifier"])
        for records in records_by_split.values()
        for row in records
    }
    peak_lists = load_peak_lists(Path(str(token_metadata["v15_tsv"])), identifiers)
    layouts = {
        name: _make_layout(splits[name], records_by_split[name], masks_by_split[name], global_to_formula, peak_lists)
        for name in ("train", "val")
    }
    features = {name: _allocate_layout_features(layouts[name], args.subformula_top_n) for name in layouts}
    occurrences: dict[int, list[tuple[str, int]]] = {}
    for split_name, layout in layouts.items():
        for group_index, formula_id in enumerate(layout["group_formula_ids"].tolist()):
            occurrences.setdefault(int(formula_id), []).append((split_name, group_index))
    for formula_id in tqdm(sorted(occurrences), desc="Matching candidate subformulae"):
        decomposer = BoundedFormulaDecomposer(formula_counts[formula_id].numpy())
        for split_name, group_index in occurrences[formula_id]:
            layout = layouts[split_name]
            query_row = int(layout["group_query_rows"][group_index])
            result = decomposer.match(
                layout["row_peaks"][query_row],
                layout["row_precursor"][query_row],
                layout["row_adduct"][query_row],
                ppm_tolerance=args.subformula_ppm,
                top_n=args.subformula_top_n,
            )
            start = int(layout["group_entry_offsets"][group_index])
            end = int(layout["group_entry_offsets"][group_index + 1])
            target = features[split_name]
            target["counts"][start:end] = result.counts
            target["ppm"][start:end] = result.ppm.astype(np.float16)
            target["route"][start:end] = result.route
            target["ambiguity"][start:end] = result.ambiguity
            target["direct_count"][start:end] = result.direct_match_count
            target["neutral_loss_count"][start:end] = result.neutral_loss_match_count
            valid = result.route[:, 0] > 0
            quality = np.zeros(len(valid), dtype=np.float32)
            quality[valid] = np.maximum(0.0, 1.0 - result.ppm[valid, 0] / float(args.subformula_ppm))
            target["group_compatibility"][group_index] = float(quality.mean()) if len(quality) else 0.0
    payload: dict[str, Any] = {
        "metadata": {
            **expected,
            "formula_table": formula_stats,
            "build_seconds": time.time() - started,
            "layout_stats": {name: layouts[name]["stats"] for name in layouts},
        }
    }
    for split_name in ("train", "val"):
        layout = layouts[split_name]
        feature = features[split_name]
        route = feature["route"]
        union = (route > 0).any(axis=1)
        direct = ((route & 1) > 0).any(axis=1)
        neutral = ((route & 2) > 0).any(axis=1)
        payload["metadata"]["layout_stats"][split_name].update(
            {
                "direct_coverage": float(direct.mean()) if len(direct) else 0.0,
                "neutral_loss_coverage": float(neutral.mean()) if len(neutral) else 0.0,
                "union_coverage": float(union.mean()) if len(union) else 0.0,
                "mean_ambiguity": float(feature["ambiguity"].mean()) if len(union) else 0.0,
            }
        )
        payload[split_name] = {
            "candidate_group": layout["candidate_group"],
            "group_offsets": layout["group_offsets"],
            "group_formula_ids": layout["group_formula_ids"],
            "group_entry_offsets": layout["group_entry_offsets"],
            **{key: torch.from_numpy(value) for key, value in feature.items()},
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    check = torch.load(temporary, map_location="cpu", weights_only=False, mmap=True)
    _validate_cache(check, expected)
    os.replace(temporary, path)
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


class CandidateSubformulaStore:
    def __init__(self, payload: dict[str, torch.Tensor]) -> None:
        self.payload = payload

    def gather(
        self,
        indices: torch.Tensor,
        candidate_mask: torch.Tensor,
        peak_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        indices = indices.long().cpu()
        candidate_group = self.payload["candidate_group"][indices].long()
        group_base = self.payload["group_offsets"][indices].long().unsqueeze(1)
        global_group = group_base + candidate_group.clamp_min(0)
        entry_start = self.payload["group_entry_offsets"][global_group].long()
        peak_position = torch.arange(peak_mask.shape[1]).view(1, 1, -1)
        source = entry_start.unsqueeze(-1) + peak_position
        valid = (
            (candidate_group >= 0).unsqueeze(-1)
            & candidate_mask.cpu().unsqueeze(-1)
            & peak_mask.cpu().unsqueeze(1)
        )
        source = source.masked_fill(~valid, 0)
        output = {
            key: self.payload[key][source]
            for key in ("counts", "ppm", "route", "ambiguity")
        }
        output["valid"] = valid
        return output

    def compatibility(self, split: TopKSplit) -> torch.Tensor:
        groups = self.payload["candidate_group"].long()
        global_group = self.payload["group_offsets"][:-1].long().unsqueeze(1) + groups.clamp_min(0)
        values = self.payload["group_compatibility"][global_group].float()
        values[groups < 0] = 0.0
        values[~split.candidate_mask] = 0.0
        valid_count = split.candidate_mask.sum(dim=1, keepdim=True).clamp_min(1)
        mean = values.sum(dim=1, keepdim=True) / valid_count
        variance = (((values - mean) ** 2) * split.candidate_mask).sum(dim=1, keepdim=True) / valid_count
        z = (values - mean) / variance.sqrt().clamp_min(1e-6)
        z[~split.candidate_mask] = 0.0
        return z


class CandidateSubformulaModel(nn.Module):
    def __init__(self, base: LocalInteractionModel, dim: int, ppm_tolerance: float) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        self.ppm_tolerance = float(ppm_tolerance)
        self.formula_encoder = nn.Sequential(
            nn.Linear(len(ELEMENTS) + 1 + 3 + 1, 128),
            nn.GELU(),
            nn.Linear(128, dim),
        )
        self.hypothesis_attention = nn.Linear(dim, 1)
        self.formula_gate = nn.Parameter(torch.tensor(0.1))
        self.delta_scale = nn.Parameter(torch.zeros(()))

    def train(self, mode: bool = True) -> "CandidateSubformulaModel":
        super().train(mode)
        self.base.eval()
        return self

    @staticmethod
    def _maxsim(
        peaks: torch.Tensor,
        peak_mask: torch.Tensor,
        candidate_nodes: torch.Tensor,
        candidate_node_mask: torch.Tensor,
    ) -> torch.Tensor:
        if peaks.ndim == 3:
            similarity = torch.einsum("bpd,bkad->bkpa", peaks, candidate_nodes)
        else:
            similarity = torch.einsum("bkpd,bkad->bkpa", peaks, candidate_nodes)
        valid = peak_mask[:, None, :, None] & candidate_node_mask[:, :, None, :]
        similarity = similarity.masked_fill(~valid, -1e4)
        peak_to_atom = similarity.max(dim=-1).values
        atom_to_peak = similarity.max(dim=-2).values
        peak_to_atom = torch.where(peak_mask[:, None], peak_to_atom, torch.zeros_like(peak_to_atom))
        atom_to_peak = torch.where(candidate_node_mask, atom_to_peak, torch.zeros_like(atom_to_peak))
        peak_score = (peak_to_atom * peak_mask[:, None]).sum(-1) / peak_mask.sum(-1).clamp_min(1)[:, None]
        atom_score = (atom_to_peak * candidate_node_mask).sum(-1) / candidate_node_mask.sum(-1).clamp_min(1)
        return 0.5 * (peak_score + atom_score)

    def forward(
        self,
        peak_tokens: torch.Tensor,
        peak_mask: torch.Tensor,
        graphs: dict[str, torch.Tensor],
        candidate_shape: tuple[int, int],
        hypothesis: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            raw_peaks = self.base.peak_projection(peak_tokens)
            base_peaks = F.normalize(raw_peaks, dim=-1, eps=1e-6)
            nodes = self.base.atom_encoder(graphs["node_features"])
            edges = self.base.bond_encoder(graphs["edge_features"])
            for layer in self.base.gnn:
                nodes = layer(nodes, graphs["edge_index"], edges)
            nodes = F.normalize(nodes, dim=-1, eps=1e-6)
            max_nodes = int(graphs["node_counts"].max())
            padded = nodes.new_zeros((len(graphs["node_counts"]), max_nodes, nodes.shape[-1]))
            padded[graphs["node_batch"], graphs["node_local"]] = nodes
            node_mask = torch.arange(max_nodes, device=nodes.device).unsqueeze(0) < graphs["node_counts"].unsqueeze(1)
            batch_size, candidate_count = candidate_shape
            candidate_nodes = padded[graphs["inverse"]].reshape(batch_size, candidate_count, max_nodes, -1)
            candidate_node_mask = node_mask[graphs["inverse"]].reshape(batch_size, candidate_count, max_nodes)
            base_score = self._maxsim(base_peaks, peak_mask, candidate_nodes, candidate_node_mask)
        route = hypothesis["route"].long()
        valid_hypothesis = route > 0
        counts = hypothesis["counts"].float() / FORMULA_NORM.to(route.device)
        ppm = hypothesis["ppm"].float()
        quality = (1.0 - ppm / self.ppm_tolerance).clamp(0.0, 1.0)
        route_onehot = F.one_hot(route.clamp(1, 3) - 1, num_classes=3).float()
        ambiguity = torch.log1p(hypothesis["ambiguity"].float()).unsqueeze(-1)
        ambiguity = ambiguity.expand(*ambiguity.shape[:-1], counts.shape[-2]).unsqueeze(-1) / math.log(65536.0)
        encoded = self.formula_encoder(torch.cat([counts, quality.unsqueeze(-1), route_onehot, ambiguity], dim=-1))
        encoded = encoded * valid_hypothesis.unsqueeze(-1)
        attention = self.hypothesis_attention(encoded).squeeze(-1) + torch.log(quality.clamp_min(1e-4))
        attention = attention.masked_fill(~valid_hypothesis, -1e4)
        weights = torch.softmax(attention, dim=-1) * valid_hypothesis
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        formula = (weights.unsqueeze(-1) * encoded).sum(dim=-2)
        conditioned = F.normalize(raw_peaks[:, None] + self.formula_gate * formula, dim=-1, eps=1e-6)
        conditioned_score = self._maxsim(conditioned, peak_mask, candidate_nodes, candidate_node_mask)
        # Unsupported and completely unmatched candidates must be a literal E1
        # fallback, not merely numerically close after a redundant normalize.
        has_formula = valid_hypothesis.any(dim=(-1, -2))
        conditioned_score = torch.where(has_formula, conditioned_score, base_score)
        return base_score, conditioned_score


def _evaluate_components(
    split: TopKSplit,
    base_residual: torch.Tensor,
    delta_residual: torch.Tensor,
    alpha_e1: float,
    beta: float,
) -> dict[str, Any]:
    ranks = split.baseline_ranks.clone().long()
    rank1_total = 0
    rank1_damaged = 0
    for row in range(len(split.order_indices)):
        valid = split.candidate_mask[row]
        scores = (
            split.stage1_scores[row, valid]
            + alpha_e1 * base_residual[row, valid]
            + beta * delta_residual[row, valid]
        )
        order = torch.argsort(scores, descending=True)
        rank = int((order == int(split.gt_indices[row])).nonzero(as_tuple=False)[0]) + 1
        ranks[int(split.order_indices[row])] = rank
        if int(split.stage1_ranks[row]) == 1:
            rank1_total += 1
            rank1_damaged += int(rank > 1)
    return {
        **rank_metrics(ranks.tolist()),
        "beta": beta,
        "rank1_total": rank1_total,
        "rank1_damage_rate": rank1_damaged / max(rank1_total, 1),
        "ranks": ranks,
    }


def _select_beta(results: dict[float, dict[str, Any]], damage_target: float) -> float:
    eligible = [(beta, value) for beta, value in results.items() if value["rank1_damage_rate"] <= damage_target]
    if not eligible:
        eligible = list(results.items())
    return max(
        eligible,
        key=lambda item: (
            float(item[1]["mrr"]),
            float(item[1]["recall@1"]),
            float(item[1]["recall@20"]),
            -float(item[1]["rank1_damage_rate"]),
            -float(item[0]),
        ),
    )[0]


def _candidate_gate(bootstrap: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    triggering = [
        metric
        for metric in ("recall@1", "mrr")
        if bootstrap[metric]["delta_percentage_points"] >= 0.3
        and bootstrap[metric]["probability_positive"] >= 0.8
    ]
    other_ok = (
        ("recall@1" in triggering and bootstrap["mrr"]["delta_percentage_points"] >= -0.1)
        or ("mrr" in triggering and bootstrap["recall@1"]["delta_percentage_points"] >= -0.1)
    )
    checks = {
        "r1_or_mrr_gain_and_bootstrap": bool(triggering),
        "other_primary_not_below_minus_0_1pp": other_ok,
        "recall20_not_below_minus_0_2pp": bootstrap["recall@20"]["delta_percentage_points"] >= -0.2,
        "run_k_rank1_damage_at_most_8pct": selected["rank1_damage_rate"] <= 0.08,
    }
    return {"passed": all(checks.values()), "triggering_metrics": triggering, "checks": checks}


def _batch_hypotheses(
    store: CandidateSubformulaStore,
    split: TopKSplit,
    indices: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    candidate_mask = split.candidate_mask[indices]
    peak_mask = masks[split.spec_indices[indices].long(), 1:].bool()
    return {key: value.to(device, non_blocking=True) for key, value in store.gather(indices, candidate_mask, peak_mask).items()}


def _train_epoch(
    model: CandidateSubformulaModel,
    split: TopKSplit,
    tokens: torch.Tensor,
    masks: torch.Tensor,
    graph_store: PackedGraphStore,
    store: CandidateSubformulaStore,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    alpha_e1: float,
    local_temperature: float,
    local_aux_weight: float,
) -> dict[str, Any]:
    model.train()
    loader = DataLoader(RerankDataset(len(split.order_indices)), batch_size=batch_size, shuffle=True, num_workers=0)
    total_loss = 0.0
    examples = 0
    first_gradients = None
    for batch_number, raw_indices in enumerate(tqdm(loader, desc="Training candidate subformula")):
        indices = raw_indices.long()
        batch = prepare_batch(split, indices, tokens, masks, graph_store, device, None)
        hypothesis = _batch_hypotheses(store, split, indices, masks, device)
        base_score, conditioned_score = model(
            batch["tokens"], batch["peak_mask"], batch["graphs"], batch["candidate_shape"], hypothesis
        )
        base_residual = model.base.residual_scale.detach() * base_score / local_temperature
        delta = model.base.residual_scale.detach() * (conditioned_score - base_score) / local_temperature
        candidate_mask = batch["candidate_mask"]
        fused = (
            batch["stage1_scores"] + alpha_e1 * base_residual + model.delta_scale * delta
        ).masked_fill(~candidate_mask, float("-inf"))
        conditioned_logits = (conditioned_score / local_temperature).masked_fill(~candidate_mask, float("-inf"))
        loss = F.cross_entropy(fused, batch["labels"]) + local_aux_weight * F.cross_entropy(
            conditioned_logits, batch["labels"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        if batch_number == 0:
            gradients = {
                name: float(parameter.grad.detach().norm().cpu())
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None
            }
            first_gradients = gradients
        optimizer.step()
        with torch.no_grad():
            model.delta_scale.clamp_(min=0.0)
        total_loss += float(loss.detach().cpu()) * len(indices)
        examples += len(indices)
    return {
        "loss": total_loss / max(examples, 1),
        "examples": examples,
        "delta_scale": float(model.delta_scale.detach().cpu()),
        "first_batch_gradients": first_gradients,
    }


@torch.no_grad()
def _collect_components(
    model: CandidateSubformulaModel,
    split: TopKSplit,
    tokens: torch.Tensor,
    masks: torch.Tensor,
    graph_store: PackedGraphStore,
    store: CandidateSubformulaStore,
    device: torch.device,
    batch_size: int,
    local_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    base_output = torch.zeros_like(split.stage1_scores)
    delta_output = torch.zeros_like(split.stage1_scores)
    loader = DataLoader(RerankDataset(len(split.order_indices)), batch_size=batch_size, shuffle=False)
    for raw_indices in tqdm(loader, desc="Scoring candidate subformula"):
        indices = raw_indices.long()
        batch = prepare_batch(split, indices, tokens, masks, graph_store, device, None)
        hypothesis = _batch_hypotheses(store, split, indices, masks, device)
        base_score, conditioned_score = model(
            batch["tokens"], batch["peak_mask"], batch["graphs"], batch["candidate_shape"], hypothesis
        )
        scale = model.base.residual_scale.detach()
        base_output[indices] = (scale * base_score / local_temperature).cpu()
        delta_output[indices] = (model.delta_scale * scale * (conditioned_score - base_score) / local_temperature).cpu()
    return base_output, delta_output


def run_candidate_subformula(
    args: Any,
    splits: dict[str, TopKSplit],
    tokens_by_split: dict[str, torch.Tensor],
    masks_by_split: dict[str, torch.Tensor],
    records_by_split: dict[str, list[dict[str, Any]]],
    token_metadata: dict[str, Any],
    graph_store: PackedGraphStore,
    smiles: list[str],
    device: torch.device,
    common: dict[str, Any],
) -> dict[str, Any] | None:
    cache_path = args.candidate_subformula_cache or args.out_dir / "candidate_subformula_train20k_val_top50.pt"
    payload = load_or_build_candidate_subformula_cache(
        cache_path,
        args,
        splits,
        records_by_split,
        masks_by_split,
        token_metadata,
        graph_store,
        smiles,
    )
    if args.candidate_subformula_cache_only:
        print(json.dumps({"candidate_subformula_cache": str(cache_path), "metadata": payload["metadata"]}, indent=2))
        return None
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    checkpoint = torch.load(args.candidate_e1_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("arm") != "raw":
        raise ValueError("candidate-subformula must initialize from a raw E1 checkpoint")
    alpha_e1 = float(checkpoint["selected_alpha"])
    base = LocalInteractionModel(args.local_dim, args.gnn_layers, args.dropout, use_subformula=False)
    base.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = CandidateSubformulaModel(base, args.local_dim, args.subformula_ppm).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    stores = {name: CandidateSubformulaStore(payload[name]) for name in ("train", "val")}
    beta_grid = [float(value) for value in args.beta_grid.split(",") if value.strip()]
    if 0.0 not in beta_grid or any(value < 0 for value in beta_grid):
        raise ValueError("beta-grid must contain zero and be non-negative")
    base_residual, zero_delta = _collect_components(
        model,
        splits["val"],
        tokens_by_split["val"],
        masks_by_split["val"],
        graph_store,
        stores["val"],
        device,
        args.candidate_eval_batch_size,
        args.local_temperature,
    )
    e1 = _evaluate_components(splits["val"], base_residual, zero_delta, alpha_e1, 0.0)
    expected_rank_path = args.candidate_e1_checkpoint.with_suffix(".val_ranks.pt")
    if expected_rank_path.exists() and int(args.val_limit) == 0:
        expected_ranks = torch.load(expected_rank_path, map_location="cpu", weights_only=False)["selected_ranks"].long()
        if e1["ranks"].tolist() != expected_ranks.tolist():
            raise RuntimeError("Frozen E1 checkpoint/rank parity failed")
    history: list[dict[str, Any]] = [
        {"epoch": 0, "selected_beta": 0.0, "val_selected": {k: v for k, v in e1.items() if k != "ranks"}, "exact_e1_fallback": True}
    ]
    best_epoch = 0
    best_beta = 0.0
    best_mrr = float(e1["mrr"])
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("base.")
    }
    for epoch in range(1, args.epochs + 1):
        report = _train_epoch(
            model,
            splits["train"],
            tokens_by_split["train"],
            masks_by_split["train"],
            graph_store,
            stores["train"],
            optimizer,
            device,
            args.candidate_batch_size,
            alpha_e1,
            args.local_temperature,
            args.local_aux_weight,
        )
        base_residual, delta_residual = _collect_components(
            model,
            splits["val"],
            tokens_by_split["val"],
            masks_by_split["val"],
            graph_store,
            stores["val"],
            device,
            args.candidate_eval_batch_size,
            args.local_temperature,
        )
        results = {
            beta: _evaluate_components(splits["val"], base_residual, delta_residual, alpha_e1, beta)
            for beta in beta_grid
        }
        selected_beta = _select_beta(results, args.damage_target)
        selected = results[selected_beta]
        entry = {
            "epoch": epoch,
            "train": report,
            "selected_beta": selected_beta,
            "val_selected": {key: value for key, value in selected.items() if key != "ranks"},
        }
        history.append(entry)
        print(json.dumps(entry, sort_keys=True), flush=True)
        if float(selected["mrr"]) > best_mrr:
            best_epoch = epoch
            best_beta = selected_beta
            best_mrr = float(selected["mrr"])
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
                if not key.startswith("base.")
            }
    model.load_state_dict(best_state, strict=False)
    base_residual, delta_residual = _collect_components(
        model,
        splits["val"],
        tokens_by_split["val"],
        masks_by_split["val"],
        graph_store,
        stores["val"],
        device,
        args.candidate_eval_batch_size,
        args.local_temperature,
    )
    final_grid = {
        beta: _evaluate_components(splits["val"], base_residual, delta_residual, alpha_e1, beta)
        for beta in beta_grid
    }
    best_beta = _select_beta(final_grid, args.damage_target)
    selected = final_grid[best_beta]
    bootstrap = paired_bootstrap(e1["ranks"], selected["ranks"], args.bootstrap_samples, args.seed + 31)
    gate = _candidate_gate(bootstrap, selected)
    composition = stores["val"].compatibility(splits["val"])
    composition_grid = {
        beta: _evaluate_components(splits["val"], base_residual, composition, alpha_e1, beta)
        for beta in beta_grid
    }
    composition_beta = _select_beta(composition_grid, args.damage_target)
    composition_selected = composition_grid[composition_beta]
    composition_bootstrap = paired_bootstrap(
        e1["ranks"], composition_selected["ranks"], args.bootstrap_samples, args.seed + 32
    )
    output = {
        **common,
        "arm": "candidate_subformula",
        "scientific_role": "deployable candidate-formula-conditioned arm",
        "candidate_subformula_cache": {
            "path": str(cache_path),
            "sha256": sha256_file(cache_path),
            "metadata": payload["metadata"],
        },
        "e1_checkpoint": str(args.candidate_e1_checkpoint),
        "e1_checkpoint_sha256": sha256_file(args.candidate_e1_checkpoint),
        "alpha_e1": alpha_e1,
        "history": history,
        "best_epoch": best_epoch,
        "selected_beta": best_beta,
        "val_e1": {key: value for key, value in e1.items() if key != "ranks"},
        "val_selected": {key: value for key, value in selected.items() if key != "ranks"},
        "paired_bootstrap_vs_e1": bootstrap,
        "screening_gate_vs_e1": gate,
        "composition_only_control": {
            "selected_beta": composition_beta,
            "val_selected": {key: value for key, value in composition_selected.items() if key != "ranks"},
            "paired_bootstrap_vs_e1": composition_bootstrap,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"candidate_subformula_seed{args.seed}"
    torch.save(
        {
            "formula_state_dict": best_state,
            "base_e1_checkpoint": str(args.candidate_e1_checkpoint),
            "best_epoch": best_epoch,
            "selected_beta": best_beta,
            "alpha_e1": alpha_e1,
            "arm": "candidate_subformula",
        },
        args.out_dir / f"{stem}.pt",
    )
    (args.out_dir / f"{stem}.json").write_text(json.dumps(json_safe(output), indent=2, sort_keys=True) + "\n")
    torch.save(
        {
            "identifiers": splits["val"].identifiers,
            "run_k_ranks": splits["val"].baseline_ranks,
            "e1_ranks": e1["ranks"],
            "selected_ranks": selected["ranks"],
            "composition_only_ranks": composition_selected["ranks"],
        },
        args.out_dir / f"{stem}.val_ranks.pt",
    )
    return output
