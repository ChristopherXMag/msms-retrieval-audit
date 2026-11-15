#!/usr/bin/env python3
"""Evaluate the frozen official/data-safe MPFIR-v2 cap64 top-5 readout.

The script first serves the 1,024-query official-train safety screen.  Official
validation is accepted only with a passing, non-plumbing safety summary.  It
never accepts official-test inputs and never changes candidates or inserts GT.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors

from glmr.subformula import ion_shift
from tools.analysis.analyze_stage1_logit_ensemble import paired_bootstrap
from tools.reranking.audit_official_datasafe_pair_controls import candidate_order
from tools.reranking.train_official_datasafe_pair_reranker import PairReranker, score_features
from tools.reranking.train_official_datasafe_score_fusion import endpoint
from tools.reranking.train_official_magma_forward import digest, rows
from tools.reranking.train_official_mpfir_v2_alignment import DirectAlignment, memory_guard
from tools.reranking.train_run_k_e1_fragment_residual import (
    PackedFragmentStore,
    build_fragment_cache,
)
from tools.reranking.train_run_k_flare_local_reranker import (
    PackedGraphStore,
    build_graph_cache,
)


H_SHIFTS = (-2, -1, 0, 1, 2)
PPM_TOLERANCE = 20.0
MAX_ATOMS = 60
MAX_FRAGMENTS = 64
LOCAL_TAU = 0.1
LOCAL_WEIGHT = 0.25
STD_FLOOR = 0.02
TIE_SEED = 20260926


def file_tensor_digest(values: torch.Tensor) -> str:
    return hashlib.sha256(values.contiguous().numpy().tobytes()).hexdigest()


def load_tokens(path: Path, identifiers: list[str]) -> dict[str, Any]:
    payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    cached_identifiers = list(payload.get("identifiers") or [])
    if cached_identifiers[:len(identifiers)] != identifiers:
        raise ValueError("Token/input identifier order mismatch")
    metadata = payload.get("metadata") or {}
    expected = "d9b4e8b920ff367370c5a3fa154eaa951f90bafa58d5224a594424d9c54b1832"
    if metadata.get("dreams_ckpt_sha256") != expected:
        raise ValueError("D1 requires the data-safe DreaMS checkpoint")
    if tuple(payload["tokens"].shape[1:]) != (101, 1024):
        raise ValueError("Unexpected DreaMS token shape")
    if len(cached_identifiers) != len(identifiers):
        payload = dict(payload)
        payload["tokens"] = payload["tokens"][:len(identifiers)]
        payload["mask"] = payload["mask"][:len(identifiers)]
        payload["identifiers"] = cached_identifiers[:len(identifiers)]
    return payload


def candidate_status(smiles: str) -> tuple[str, float | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 1:
        return "candidate_rdkit_failure", None
    if len(Chem.GetMolFrags(mol)) != 1:
        return "candidate_disconnected_structure", None
    if mol.GetNumHeavyAtoms() > MAX_ATOMS:
        return "candidate_too_many_heavy_atoms", None
    return "supported", float(Descriptors.ExactMolWt(mol))


def build_registry(
    inputs: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str], torch.Tensor, dict[str, int]]:
    support = np.ones(len(inputs), dtype=bool)
    reasons: Counter[str] = Counter()
    by_global: dict[int, tuple[str, str, float | None]] = {}
    for index, query in enumerate(inputs):
        if query.get("adduct") not in ("[M+H]+", "[M+Na]+"):
            support[index] = False
            reasons["unsupported_adduct"] += 1
            continue
        query_reason = "supported"
        for global_row, smiles in zip(query["candidate_global_rows"], query["candidate_smiles"], strict=True):
            global_row = int(global_row)
            previous = by_global.get(global_row)
            if previous is not None:
                if previous[0] != str(smiles):
                    raise ValueError(f"Global candidate identity drift at row {global_row}")
                status, mass = previous[1], previous[2]
            else:
                status, mass = candidate_status(str(smiles))
            current = (str(smiles), status, mass)
            by_global[global_row] = current
            if status != "supported" and query_reason == "supported":
                query_reason = status
        if query_reason != "supported":
            support[index] = False
            reasons[query_reason] += 1
        else:
            reasons["supported"] += 1

    used_global = sorted({int(row) for i, query in enumerate(inputs) if support[i]
                          for row in query["candidate_global_rows"]})
    lookup = {value: index for index, value in enumerate(used_global)}
    compact = np.full((len(inputs), 5), -1, dtype=np.int32)
    for i, query in enumerate(inputs):
        if support[i]:
            compact[i] = [lookup[int(value)] for value in query["candidate_global_rows"]]
    smiles = [by_global[value][0] for value in used_global]
    parent_mass = torch.tensor([float(by_global[value][2]) for value in used_global], dtype=torch.float32)
    return support, compact, smiles, parent_mass, dict(reasons)


def load_or_build_stores(
    directory: Path, smiles: list[str]
) -> tuple[PackedGraphStore, PackedFragmentStore, dict[str, Any]]:
    indices = torch.arange(len(smiles), dtype=torch.long)
    graph_path = directory / "candidate_graphs.pt"
    fragment_path = directory / "candidate_fragments_cap64.pt"
    expected = file_tensor_digest(indices)
    if graph_path.exists():
        graph_payload = torch.load(str(graph_path), map_location="cpu", mmap=True, weights_only=False)
        if (graph_payload.get("metadata") or {}).get("global_indices_sha256") != expected:
            raise ValueError("Stale D1 graph cache")
    else:
        graph_payload = build_graph_cache(graph_path, indices, smiles)
    if fragment_path.exists():
        fragment_payload = torch.load(str(fragment_path), map_location="cpu", mmap=True, weights_only=False)
        metadata = fragment_payload.get("metadata") or {}
        if metadata.get("global_indices_sha256") != expected or metadata.get("max_fragments") != MAX_FRAGMENTS:
            raise ValueError("Stale D1 fragment cache")
    else:
        fragment_payload = build_fragment_cache(fragment_path, indices, smiles, MAX_FRAGMENTS)
    graph = PackedGraphStore(graph_payload, len(smiles))
    fragment = PackedFragmentStore(fragment_payload, len(smiles))
    metadata = {
        "candidates": len(smiles),
        "graph": graph_payload["metadata"],
        "fragment": fragment_payload["metadata"],
        "graph_sha256": digest(graph_path),
        "fragment_sha256": digest(fragment_path),
    }
    return graph, fragment, metadata


def move(payload: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in payload.items()}


def load_alignment(path: Path, device: torch.device) -> DirectAlignment:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    model = DirectAlignment(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def peak_arrays(inputs: list[dict[str, Any]], tokens: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    mz = torch.zeros((len(inputs), 100), dtype=torch.float32)
    intensity = torch.zeros((len(inputs), 100), dtype=torch.float32)
    for index, query in enumerate(inputs):
        observed = query["observed_spectrum"]
        valid = int(tokens["mask"][index, 1:].sum())
        if len(observed) != valid or len(observed) > 100:
            raise ValueError(f"Peak/token parity failed at {query['identifier']}: {len(observed)} != {valid}")
        if observed:
            values = torch.tensor(observed, dtype=torch.float32)
            mz[index, :len(observed)] = values[:, 0]
            intensity[index, :len(observed)] = values[:, 1]
    return mz, intensity


def masked_null_support(similarity: torch.Tensor, valid: torch.Tensor, tau: float = LOCAL_TAU) -> torch.Tensor:
    if similarity.shape != valid.shape:
        raise ValueError("Similarity/feasibility shape mismatch")
    count = valid.sum(-1)
    summed = (torch.exp(similarity / tau) * valid).sum(-1)
    mean = summed / count.clamp_min(1).to(similarity.dtype)
    return torch.where(count > 0, tau * torch.log1p(mean), torch.zeros_like(mean))


def candidate_relative_score(support: torch.Tensor, peak_mask: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
    median = support.median(dim=1, keepdim=True).values
    residual = (support - median).clamp_min(0)
    weights = intensity.clamp_min(0).sqrt() * peak_mask
    return (residual * weights[:, None]).sum(-1) / weights.sum(-1).clamp_min(1e-8)[:, None]


def fragment_tokens(
    model: DirectAlignment,
    graphs: dict[str, torch.Tensor],
    fragments: dict[str, torch.Tensor],
    candidate_shape: tuple[int, int],
) -> torch.Tensor:
    nodes = model.atom_encoder(graphs["node_features"])
    edges = model.bond_encoder(graphs["edge_features"])
    for layer in model.gnn:
        nodes = layer(nodes, graphs["edge_index"], edges)
    total = int(fragments["fragment_mask"].numel())
    pooled = nodes.new_zeros((total, nodes.shape[-1]))
    pooled.index_add_(0, fragments["atom_fragment_indices"], nodes[fragments["atom_node_indices"]])
    counts = torch.bincount(fragments["atom_fragment_indices"], minlength=total).to(nodes.dtype).clamp_min(1)
    pooled = pooled / counts[:, None]
    pooled = F.normalize(model.fragment_projection(pooled), dim=-1, eps=1e-6)
    width = fragments["fragment_mask"].shape[1]
    unique = pooled.reshape(len(fragments["fragment_mask"]), width, -1)
    return unique[fragments["inverse"]].reshape(*candidate_shape, width, -1)


def feasibility(
    peak_mz: torch.Tensor,
    precursor_mz: torch.Tensor,
    adduct_shift: torch.Tensor,
    fragment_mass: torch.Tensor,
    parent_mass: torch.Tensor,
    fragment_mask: torch.Tensor,
    peak_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    hydrogen = float(Chem.GetPeriodicTable().GetMostCommonIsotopeMass(1))
    shifts = peak_mz.new_tensor(H_SHIFTS) * hydrogen
    predicted_direct = fragment_mass[:, :, None, :, None] + adduct_shift[:, None, None, None, None] + shifts
    direct_signed = ((peak_mz[:, None, :, None, None] - predicted_direct)
                     / predicted_direct.abs().clamp_min(1e-6) * 1e6)
    direct_abs, direct_shift = direct_signed.abs().min(-1)
    direct_best = direct_signed.gather(-1, direct_shift[..., None]).squeeze(-1)
    observed_loss = precursor_mz[:, None] - peak_mz
    predicted_loss = parent_mass[:, :, None, None, None] - fragment_mass[:, :, None, :, None] + shifts
    loss_signed = ((observed_loss[:, None, :, None, None] - predicted_loss)
                   / predicted_loss.abs().clamp_min(1e-6) * 1e6)
    loss_abs, loss_shift = loss_signed.abs().min(-1)
    loss_best = loss_signed.gather(-1, loss_shift[..., None]).squeeze(-1)
    choose_direct = direct_abs <= loss_abs
    best_abs = torch.where(choose_direct, direct_abs, loss_abs)
    best_signed = torch.where(choose_direct, direct_best, loss_best)
    best_shift = torch.where(choose_direct, direct_shift, loss_shift)
    valid = (best_abs < PPM_TOLERANCE) & fragment_mask[:, :, None, :] & peak_mask[:, None, :, None]
    selected_shift = best_shift[valid].detach().cpu().numpy()
    selected_ppm = best_signed[valid].detach().cpu().numpy()
    audit = {
        "feasible_edges": int(valid.sum()),
        "direct_route_edges": int((choose_direct & valid).sum()),
        "neutral_loss_route_edges": int((~choose_direct & valid).sum()),
        "h_shift_counts": np.bincount(selected_shift, minlength=len(H_SHIFTS)).astype(np.int64),
        "signed_ppm_histogram": np.histogram(selected_ppm, bins=[-20, -10, -5, 0, 5, 10, 20])[0].astype(np.int64),
        "signed_ppm_sum": float(selected_ppm.sum()),
        "absolute_ppm_sum": float(np.abs(selected_ppm).sum()),
    }
    return valid, audit


def merge_audit(total: dict[str, Any], current: dict[str, Any]) -> None:
    for key in ("feasible_edges", "direct_route_edges", "neutral_loss_route_edges"):
        total[key] = total.get(key, 0) + current[key]
    for key in ("signed_ppm_sum", "absolute_ppm_sum"):
        total[key] = total.get(key, 0.0) + current[key]
    for key in ("h_shift_counts", "signed_ppm_histogram"):
        total[key] = np.asarray(total.get(key, np.zeros_like(current[key]))) + current[key]


@torch.no_grad()
def score_batch(
    model: DirectAlignment,
    query_indices: np.ndarray,
    candidate_indices: torch.Tensor,
    tokens: dict[str, Any],
    peak_mz: torch.Tensor,
    intensity: torch.Tensor,
    precursor_mz: torch.Tensor,
    adduct_shift: torch.Tensor,
    parent_mass_store: torch.Tensor,
    graph_store: PackedGraphStore,
    fragment_store: PackedFragmentStore,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    candidates = candidate_indices.long()
    graph_cpu = graph_store.gather(candidates.reshape(-1))
    fragment_cpu = fragment_store.gather(candidates.reshape(-1), graph_cpu["node_counts"])
    graphs = move(graph_cpu, device)
    fragments = move(fragment_cpu, device)
    candidate_shape = tuple(candidates.shape)
    fragment = fragment_tokens(model, graphs, fragments, candidate_shape)
    rows_t = torch.from_numpy(query_indices)
    peaks = F.normalize(model.peak_projection(tokens["tokens"][rows_t, 1:].float().to(device)), dim=-1, eps=1e-6)
    token_mask = tokens["mask"][rows_t, 1:].to(device)
    similarity = torch.einsum("bpd,bkfd->bkpf", peaks, fragment)
    inverse = fragments["inverse"]
    fragment_mask = fragments["fragment_mask"][inverse].reshape(*candidate_shape, MAX_FRAGMENTS)
    fragment_mass = (fragments["scalar"][inverse, :, 0] * 1000).reshape(*candidate_shape, MAX_FRAGMENTS)
    parent_mass = parent_mass_store[candidates].to(device)
    valid, audit = feasibility(
        peak_mz[rows_t].to(device), precursor_mz[rows_t].to(device), adduct_shift[rows_t].to(device),
        fragment_mass, parent_mass, fragment_mask, token_mask,
    )
    learned = candidate_relative_score(masked_null_support(similarity, valid), token_mask, intensity[rows_t].to(device))
    coverage = candidate_relative_score(masked_null_support(torch.ones_like(similarity), valid), token_mask,
                                        intensity[rows_t].to(device))
    feasible_peaks = valid.any(-1).sum(-1)
    return learned.cpu(), coverage.cpu(), audit | {"feasible_peaks": feasible_peaks.cpu()}


@torch.no_grad()
def load_pair_scores(
    partition: str,
    inputs: list[dict[str, Any]],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    source_path = ROOT / "outputs/stage2_official_datasafe/pair_reranker/formal_top5/summary.json"
    source = json.loads(source_path.read_text())
    cache_path = ROOT / source["args"][f"{partition}_top5"]
    cache = torch.load(str(cache_path), map_location="cpu", mmap=True, weights_only=False)
    projection_path = ROOT / source["args"]["candidate_projection"]
    molecule = torch.load(str(projection_path), map_location="cpu", mmap=True, weights_only=False)["embeddings"]
    models = []
    for entry in source["checkpoint_files"]:
        path = ROOT / entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError("Frozen pair checkpoint drift")
        model = PairReranker(1024, 11, 256, 64, .1).to(device)
        model.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=False)["model_state_dict"])
        models.append(model.eval())
    scores = np.empty((len(inputs), 5), dtype=np.float32)
    for start in range(0, len(inputs), 512):
        batch = inputs[start:start + 512]
        source_rows = torch.tensor([int(query["query_index"]) for query in batch])
        for local, query in enumerate(batch):
            row = int(source_rows[local])
            if cache["identifiers"][row] != query["identifier"]:
                raise ValueError("Frozen top5 identifier drift")
            if cache["candidate_global_rows"][row].tolist() != query["candidate_global_rows"]:
                raise ValueError("Frozen top5 candidate drift")
        spec = cache["spec_embeddings"][source_rows].float().to(device)
        candidates = molecule[cache["candidate_local_indices"][source_rows].long()].float().to(device)
        base = cache["ensemble_scores"][source_rows].float().to(device)
        relative = score_features(cache["stage1_scores"][source_rows].to(device), base)
        scores[start:start + len(batch)] = np.mean(
            np.stack([model(spec, candidates, relative, base).cpu().numpy() for model in models]), axis=0)
    audit = {
        "summary": str(source_path.relative_to(ROOT)),
        "summary_sha256": digest(source_path),
        "top5_sha256": digest(cache_path),
        "candidate_projection_sha256": digest(projection_path),
    }
    return scores, cache, audit


def local_rank(target: dict[str, Any], candidate_rows: list[int], scores: np.ndarray) -> int:
    position = int(target["gt_top5_position"])
    if position < 0:
        return int(target["ensemble_rank"])
    order = candidate_order(target["identifier"], np.asarray(candidate_rows), scores, TIE_SEED)
    return int(np.flatnonzero(order == position)[0]) + 1


def zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(1, keepdims=True)) / np.maximum(values.std(1, keepdims=True), STD_FLOOR)


def tie_invariance(inputs: list[dict[str, Any]], scores: np.ndarray) -> float:
    permutation = np.asarray([2, 4, 1, 0, 3])
    mismatches = 0
    for query, value in zip(inputs, scores, strict=True):
        candidates = np.asarray(query["candidate_global_rows"])
        first = candidates[candidate_order(query["identifier"], candidates, value, TIE_SEED)]
        second = candidates[permutation][candidate_order(
            query["identifier"], candidates[permutation], value[permutation], TIE_SEED)]
        mismatches += int(not np.array_equal(first, second))
    return float(mismatches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=("train", "val"), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--l0-dir", type=Path, default=ROOT / "outputs/mpfir_v2_official_20261002/l0_seed20261002")
    parser.add_argument("--calibration-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query-limit", type=int, default=0, help="Plumbing only")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if any("test" in path.name.lower() for path in (args.input_dir, args.tokens, args.output_dir)):
        raise ValueError("Official-test-like paths are forbidden")
    l0_summary = json.loads((args.l0_dir / "summary.json").read_text())
    if l0_summary.get("verdict") != "L0_PASS" or l0_summary.get("plumbing_only"):
        raise ValueError("A formal passing L0 result is required")
    if digest(args.l0_dir / "true/best.pt") != l0_summary["arms"]["true"]["checkpoint_sha256"]:
        raise ValueError("True L0 checkpoint drift")
    if digest(args.l0_dir / "shuffled/best.pt") != l0_summary["arms"]["shuffled"]["checkpoint_sha256"]:
        raise ValueError("Shuffled L0 checkpoint drift")
    if args.partition == "val":
        if args.calibration_summary is None:
            raise ValueError("Validation requires the train safety summary")
        safety = json.loads(args.calibration_summary.read_text())
        if safety.get("verdict") != "CALIBRATION_SAFETY_PASS" or safety.get("plumbing_only"):
            raise ValueError("Formal train safety did not pass")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    memory_guard(); started = time.time(); device = torch.device(args.device)
    source = json.loads((args.input_dir / "summary.json").read_text())
    if source.get("official_test_read") or source.get("partition") != args.partition:
        raise ValueError("Input partition/provenance mismatch")
    for filename, checksum in source["output_sha256"].items():
        if digest(args.input_dir / filename) != checksum:
            raise ValueError("Changed frozen D1 inputs")
    inputs = list(rows(args.input_dir / "inputs.jsonl"))
    targets = list(rows(args.input_dir / "metric_targets.jsonl"))
    if args.query_limit:
        inputs = inputs[:args.query_limit]; targets = targets[:args.query_limit]
    if [item["identifier"] for item in inputs] != [item["identifier"] for item in targets]:
        raise ValueError("Input/target order drift")
    tokens = load_tokens(args.tokens, [item["identifier"] for item in inputs])
    peak_mz, intensity = peak_arrays(inputs, tokens)
    support, compact, smiles, parent_mass, support_counts = build_registry(inputs)
    args.output_dir.mkdir(parents=True)
    graph_store, fragment_store, store_audit = load_or_build_stores(args.output_dir, smiles)
    precursor_mz = torch.tensor([float(item["precursor_mz"]) for item in inputs])
    adduct_shift = torch.tensor([ion_shift(item["adduct"]) if item["adduct"] in ("[M+H]+", "[M+Na]+") else 0.
                                 for item in inputs])
    true_model = load_alignment(args.l0_dir / "true/best.pt", device)
    shuffled_model = load_alignment(args.l0_dir / "shuffled/best.pt", device)
    pair_scores, pair_cache, pair_audit = load_pair_scores(args.partition, inputs, device)
    learned = np.zeros((len(inputs), 5), dtype=np.float32)
    shuffled = np.zeros_like(learned); coverage = np.zeros_like(learned)
    feasible_peaks = np.zeros((len(inputs), 5), dtype=np.int16)
    chemistry_audit: dict[str, Any] = {}
    permutation_error = 0.0; checked_permutation = False
    supported_indices = np.flatnonzero(support)
    for start in range(0, len(supported_indices), args.batch_size):
        memory_guard(); selected = supported_indices[start:start + args.batch_size]
        candidate_batch = torch.from_numpy(compact[selected])
        true_score, coverage_score, audit = score_batch(
            true_model, selected, candidate_batch, tokens, peak_mz, intensity, precursor_mz,
            adduct_shift, parent_mass, graph_store, fragment_store, device,
        )
        shuffled_score, shuffled_coverage, _ = score_batch(
            shuffled_model, selected, candidate_batch, tokens, peak_mz, intensity, precursor_mz,
            adduct_shift, parent_mass, graph_store, fragment_store, device,
        )
        if not torch.equal(coverage_score, shuffled_coverage):
            raise RuntimeError("Coverage control changed with encoder")
        learned[selected] = true_score.numpy(); shuffled[selected] = shuffled_score.numpy()
        coverage[selected] = coverage_score.numpy(); feasible_peaks[selected] = audit.pop("feasible_peaks").numpy()
        merge_audit(chemistry_audit, audit)
        if not checked_permutation:
            permutation = torch.tensor([2, 4, 1, 0, 3])
            perm_true, perm_coverage, _ = score_batch(
                true_model, selected, candidate_batch[:, permutation], tokens, peak_mz, intensity,
                precursor_mz, adduct_shift, parent_mass, graph_store, fragment_store, device,
            )
            permutation_error = max(float((perm_true[:, torch.argsort(permutation)] - true_score).abs().max()),
                                    float((perm_coverage[:, torch.argsort(permutation)] - coverage_score).abs().max()))
            if permutation_error > 2e-6:
                raise RuntimeError(f"Candidate permutation equivariance failed: {permutation_error}")
            checked_permutation = True
        if start % (args.batch_size * 64) == 0:
            print(json.dumps({"scored": min(start + args.batch_size, len(supported_indices)),
                              "supported": len(supported_indices), "elapsed_seconds": time.time() - started}), flush=True)

    baseline = np.asarray([local_rank(target, query["candidate_global_rows"], pair_scores[index])
                           for index, (query, target) in enumerate(zip(inputs, targets, strict=True))], dtype=np.int32)
    if args.partition == "val" and not args.query_limit:
        # pair_cache is the tensor cache; the canonical rank file is in the summary.
        pair_source = json.loads((ROOT / "outputs/stage2_official_datasafe/pair_reranker/formal_top5/summary.json").read_text())
        reference = np.load(ROOT / pair_source["rank_file"]["path"])["reranked_ranks"]
        expected = reference[[int(query["query_index"]) for query in inputs]]
        if not np.array_equal(baseline, expected):
            raise ValueError("Frozen pair validation rank parity failed")

    def compose(local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        combined = pair_scores.copy()
        combined[support] += (LOCAL_WEIGHT * zscore(local[support])).astype(np.float32)
        ranks = np.asarray([local_rank(target, query["candidate_global_rows"], combined[index])
                            for index, (query, target) in enumerate(zip(inputs, targets, strict=True))], dtype=np.int32)
        return combined, ranks

    learned_scores, learned_ranks = compose(learned)
    coverage_scores, coverage_ranks = compose(coverage)
    shuffled_scores, shuffled_ranks = compose(shuffled)
    groups = np.asarray([target["connectivity"] for target in targets])
    learned_endpoint = endpoint(baseline, learned_ranks, groups)
    coverage_endpoint = endpoint(baseline, coverage_ranks, groups)
    shuffled_endpoint = endpoint(baseline, shuffled_ranks, groups)
    learned_vs_coverage = endpoint(coverage_ranks, learned_ranks, groups)
    learned_vs_shuffled = endpoint(shuffled_ranks, learned_ranks, groups)
    identity_tie_mismatches = tie_invariance(inputs, learned_scores)
    if identity_tie_mismatches:
        raise RuntimeError("Identity-keyed tie invariance failed")
    plumbing = bool(args.query_limit)
    if args.partition == "train":
        gates = {
            "recall1_nonnegative": learned_endpoint["delta_query_weighted_pp"]["recall@1"] >= 0,
            "mrr_nonnegative": learned_endpoint["delta_query_weighted_pp"]["mrr"] >= 0,
            "damage_at_most_8pct": learned_endpoint["baseline_rank1_damage_rate"] <= .08,
        }
        bootstrap = None
        verdict = "PLUMBING_ONLY" if plumbing else ("CALIBRATION_SAFETY_PASS" if all(gates.values()) else "STOP_D1_FIXED_READOUT")
    else:
        bootstrap = paired_bootstrap(baseline, learned_ranks, groups, 10_000, 20261002)
        ci = bootstrap["connectivity_cluster_query_weighted"]
        gates = {
            "recall1_gain_at_least_0_5pp": learned_endpoint["delta_query_weighted_pp"]["recall@1"] >= .5,
            "mrr_gain_above_0_1pp": learned_endpoint["delta_query_weighted_pp"]["mrr"] > .1,
            "damage_at_most_8pct": learned_endpoint["baseline_rank1_damage_rate"] <= .08,
            "recall5_unchanged": learned_endpoint["delta_query_weighted_pp"]["recall@5"] == 0,
            "recall20_unchanged": learned_endpoint["delta_query_weighted_pp"]["recall@20"] == 0,
            "recall1_cluster_ci_low_positive": ci["recall@1"]["ci95_low_pp"] > 0,
            "mrr_cluster_ci_low_positive": ci["mrr"]["ci95_low_pp"] > 0,
            "mrr_not_below_coverage": learned_vs_coverage["delta_query_weighted_pp"]["mrr"] >= 0,
            "mrr_not_below_shuffled": learned_vs_shuffled["delta_query_weighted_pp"]["mrr"] >= 0,
        }
        verdict = "PLUMBING_ONLY" if plumbing else ("COMPONENT_GATE_PASS_CONTROLS_PENDING" if all(gates.values()) else "STOP_D1_FIXED_READOUT")

    chemistry_audit = {key: value.tolist() if isinstance(value, np.ndarray) else value
                       for key, value in chemistry_audit.items()}
    if chemistry_audit.get("feasible_edges"):
        chemistry_audit["mean_signed_ppm"] = chemistry_audit["signed_ppm_sum"] / chemistry_audit["feasible_edges"]
        chemistry_audit["mean_absolute_ppm"] = chemistry_audit["absolute_ppm_sum"] / chemistry_audit["feasible_edges"]
    baseline_gap = learned_endpoint["baseline_query_weighted"]["recall@20"] - learned_endpoint["baseline_query_weighted"]["recall@1"]
    report = {
        "verdict": verdict,
        "official_test_read": False,
        "partition": args.partition,
        "plumbing_only": plumbing,
        "protocol": "official/data-safe MPFIR-v2 D1 cap64 fixed top5 readout",
        "gates": gates,
        "fixed": {"max_atoms": MAX_ATOMS, "max_fragments": MAX_FRAGMENTS, "ppm_tolerance": PPM_TOLERANCE,
                  "h_shifts": list(H_SHIFTS), "tau": LOCAL_TAU, "local_weight": LOCAL_WEIGHT,
                  "std_floor": STD_FLOOR, "coverage_similarity": 1.0, "tie_seed": TIE_SEED},
        "endpoint": learned_endpoint,
        "coverage_control_endpoint": coverage_endpoint,
        "shuffled_control_endpoint": shuffled_endpoint,
        "learned_vs_coverage": learned_vs_coverage,
        "learned_vs_shuffled": learned_vs_shuffled,
        "paired_bootstrap": bootstrap,
        "gap_closure_fraction": learned_endpoint["delta_query_weighted_pp"]["recall@1"] / baseline_gap if baseline_gap else None,
        "supported_queries": int(support.sum()),
        "total_queries": len(inputs),
        "supported_fraction": float(support.mean()),
        "support_counts": support_counts,
        "fallback": "whole-query exact frozen-pair no-op",
        "chemistry_audit": chemistry_audit,
        "mean_feasible_peaks_per_supported_candidate": float(feasible_peaks[support].mean()) if support.any() else 0.,
        "candidate_permutation_max_abs_error": permutation_error,
        "identity_tie_mismatches": identity_tie_mismatches,
        "l0_summary_sha256": digest(args.l0_dir / "summary.json"),
        "true_checkpoint_sha256": digest(args.l0_dir / "true/best.pt"),
        "shuffled_checkpoint_sha256": digest(args.l0_dir / "shuffled/best.pt"),
        "input_summary_sha256": digest(args.input_dir / "summary.json"),
        "token_sha256": digest(args.tokens),
        "pair": pair_audit,
        "stores": store_audit,
        "rdkit": rdBase.rdkitVersion,
        "elapsed_seconds": time.time() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "cuda_peak_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.,
        "code_sha256": digest(Path(__file__)),
        "caveat": "train safety is not full-stack OOF; validation is development data",
    }
    np.savez_compressed(
        args.output_dir / "per_query.npz",
        identifiers=np.asarray([query["identifier"] for query in inputs]),
        connectivities=groups,
        candidate_global_rows=np.asarray([query["candidate_global_rows"] for query in inputs], dtype=np.int64),
        baseline_ranks=baseline,
        learned_ranks=learned_ranks,
        coverage_ranks=coverage_ranks,
        shuffled_ranks=shuffled_ranks,
        pair_scores=pair_scores,
        learned_local=learned,
        coverage_local=coverage,
        shuffled_local=shuffled,
        supported=support,
        feasible_peaks=feasible_peaks,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
