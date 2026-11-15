"""Run-K aligned-feature top-K reranker.

This is a controlled Stage-2 experiment.  The complete Run-K Stage1 model is
frozen.  For every spectrum/candidate pair the reranker sees only the already
aligned 1024-d global vectors plus their Stage1 score:

    [z_spec, z_mol, abs(z_spec-z_mol), z_spec*z_mol, stage1_score] -> MLP

Training uses listwise candidate CE inside the unmodified Stage1 top-K.  GT is
never inserted into top-K; queries whose GT is outside top-K remain in all
unconditional metrics at their original Stage1 rank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.reranking.train_msalign_cross_attention_reranker import (  # noqa: E402
    gt_rank_under_policy,
    parse_float_grid,
    policy_key,
    rank_metrics,
)
from tools.stage1.train_msalign_projection_smoke import (  # noqa: E402
    MoleculeEmbeddingStore,
    ProjectionModel,
    load_candidate_sidecar,
    load_jsonl,
)


DEFAULT_RUN_K = REPO_ROOT / "outputs/msalign_molecule_ablation/run_k_r3_allfp_blockl2_20k_cap256_5ep.pt"
DEFAULT_CHEMBERTA = (
    REPO_ROOT / "outputs/msalign_molecule_ablation/chemberta_mtr_morgan_formula_group_20k_val_test_mass_cap256.pt"
)
DEFAULT_SIDECAR = (
    REPO_ROOT / "outputs/msalign_mtr_probe/chemberta_mtr_formula_group_20k_val_test_mass_cap256.candidates.jsonl"
)
DEFAULT_FP = (
    REPO_ROOT / "outputs/msalign_molecule_ablation/sparse_fingerprints_formula_group_20k_val_test_cap256/manifest.json"
)
DEFAULT_R3 = (
    REPO_ROOT
    / "outputs/msalign_molecule_ablation/sparse_count_morgan_r3_formula_group_20k_val_test_cap256/manifest.json"
)
DEFAULT_PRIMARY_PROJECTIONS = (
    REPO_ROOT / "outputs/dreams_chemistry_adapter/run_k_projected_candidates_train20k_val_fp16.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen Run-K aligned-feature pairwise/listwise reranker")
    parser.add_argument(
        "--train-metadata",
        type=Path,
        default=Path("outputs/msalign_formula_group_split_v15/train_sample20k_seed20260623.metadata.jsonl"),
    )
    parser.add_argument("--val-metadata", type=Path, default=Path("outputs/msalign_formula_group_split_v15/val.metadata.jsonl"))
    parser.add_argument("--test-metadata", type=Path, default=Path("outputs/msalign_formula_group_split_v15/test.metadata.jsonl"))
    parser.add_argument(
        "--train-dreams",
        type=Path,
        default=Path("outputs/msalign_dreams_no_head_formula_group_masscand/train_sample20k_seed20260623.no_head.npy"),
    )
    parser.add_argument("--val-dreams", type=Path, default=Path("outputs/msalign_dreams_no_head_formula_group_masscand/val.no_head.npy"))
    parser.add_argument("--test-dreams", type=Path, default=Path("outputs/msalign_dreams_no_head_formula_group_masscand/test.no_head.npy"))
    parser.add_argument("--run-k-checkpoint", type=Path, default=DEFAULT_RUN_K)
    parser.add_argument("--chemberta-cache", type=Path, default=DEFAULT_CHEMBERTA)
    parser.add_argument("--candidate-sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--fingerprint-cache", type=Path, default=DEFAULT_FP)
    parser.add_argument("--additional-fingerprint-cache", type=Path, action="append", default=[DEFAULT_R3])
    parser.add_argument("--primary-projection-cache", type=Path, default=DEFAULT_PRIMARY_PROJECTIONS)
    parser.add_argument(
        "--supplement-projection-cache",
        type=Path,
        default=Path("outputs/run_k_pairwise_reranker/run_k_projected_candidates_test_supplement_fp16.pt"),
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--rank-output", type=Path, default=None)
    parser.add_argument("--candidate-pool-limit", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--stage1-batch-size", type=int, default=128)
    parser.add_argument("--projection-build-batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage1-weight-train", type=float, default=0.0)
    parser.add_argument("--alpha-grid", default="0,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--gate-grid", default="0,0.01,0.02,0.05")
    parser.add_argument("--damage-target", type=float, default=0.08)
    parser.add_argument("--mrr-tolerance", type=float, default=0.005)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--preload-projected-caches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def sample_records(records: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(records):
        return records
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), limit))
    return [records[index] for index in indices]


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_split_inputs(
    metadata_path: Path, dreams_path: Path, limit: int, seed: int
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    records = load_jsonl(metadata_path)
    dreams = torch.from_numpy(np.array(np.load(dreams_path, mmap_mode="r"), copy=True))
    if len(records) != len(dreams):
        raise ValueError(f"Metadata/DreaMS mismatch for {metadata_path}: {len(records)} vs {len(dreams)}")
    if limit <= 0 or limit >= len(records):
        return records, dreams
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), limit))
    return [records[index] for index in indices], dreams[indices].clone()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run_k(path: Path, device: torch.device) -> tuple[ProjectionModel, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    args = dict(payload.get("args") or {})
    state = payload.get("best_model_state_dict") or payload.get("model_state_dict")
    if state is None:
        raise ValueError(f"No model state in {path}")
    model = ProjectionModel(
        spec_dim=1024,
        mol_dim=6311,
        hidden_dim=int(args.get("hidden_dim", 2048)),
        proj_dim=int(args.get("proj_dim", 1024)),
        dropout=float(args.get("dropout", 0.2)),
        temperature=float(args.get("temperature", 0.07)),
        head_type=str(args.get("head_type", "mlp")),
        mol_block_dims=[2048, 2048, 2048, 167],
        mol_block_normalization="l2",
    )
    model.load_state_dict(state)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "best_epoch": payload.get("best_epoch"),
        "best_val": payload.get("best_val"),
        "args": json_safe(args),
    }


@dataclass
class Query:
    order_index: int
    identifier: str
    spec_index: int
    gt_global_index: int
    candidate_indices: list[int]


@dataclass
class QuerySet:
    queries: list[Query]
    stats: dict[str, int]


def build_queries(
    records: list[dict[str, Any]],
    sidecar: dict[str, dict[str, Any]],
    smiles_to_index: dict[str, int],
    spec_rows: int,
    candidate_pool_limit: int,
) -> QuerySet:
    if len(records) != spec_rows:
        raise ValueError(f"Metadata/DreaMS mismatch: {len(records)} vs {spec_rows}")
    stats = {
        "records_seen": len(records),
        "missing_sidecar": 0,
        "missing_gt_embedding": 0,
        "gt_missing_from_candidate_pool": 0,
        "candidate_pool_too_small": 0,
        "invalid_candidate_embedding": 0,
        "duplicate_candidates_removed": 0,
        "usable_records": 0,
    }
    queries: list[Query] = []
    for row_index, record in enumerate(records):
        identifier = str(record.get("identifier"))
        sidecar_row = sidecar.get(identifier)
        if sidecar_row is None:
            stats["missing_sidecar"] += 1
            continue
        gt = sidecar_row.get("normalized_gt_smiles")
        if gt is None or str(gt) not in smiles_to_index:
            stats["missing_gt_embedding"] += 1
            continue
        candidates = list(sidecar_row.get("normalized_candidates") or [])
        if candidate_pool_limit > 0:
            candidates = candidates[:candidate_pool_limit]
        if str(gt) not in candidates:
            stats["gt_missing_from_candidate_pool"] += 1
            continue
        indices: list[int] = []
        seen: set[int] = set()
        for smiles in candidates:
            global_index = smiles_to_index.get(str(smiles))
            if global_index is None:
                stats["invalid_candidate_embedding"] += 1
                continue
            if global_index in seen:
                stats["duplicate_candidates_removed"] += 1
                continue
            seen.add(global_index)
            indices.append(int(global_index))
        gt_global_index = int(smiles_to_index[str(gt)])
        if gt_global_index not in seen:
            stats["gt_missing_from_candidate_pool"] += 1
            continue
        if len(indices) < 2:
            stats["candidate_pool_too_small"] += 1
            continue
        queries.append(
            Query(
                order_index=len(queries),
                identifier=identifier,
                spec_index=row_index,
                gt_global_index=gt_global_index,
                candidate_indices=indices,
            )
        )
    stats["usable_records"] = len(queries)
    return QuerySet(queries, stats)


class CompositeProjectedStore:
    def __init__(self, payloads: list[dict[str, Any]], total_rows: int, preload: bool) -> None:
        self.embeddings: list[torch.Tensor] = []
        self.source = torch.full((total_rows,), -1, dtype=torch.int8)
        self.local = torch.full((total_rows,), -1, dtype=torch.int32)
        self.metadata: list[dict[str, Any]] = []
        dim = None
        for source_index, payload in enumerate(payloads):
            embeddings = payload["embeddings"]
            if preload:
                print(f"Preloading projected cache {source_index}: {tuple(embeddings.shape)}", flush=True)
                embeddings = embeddings.clone()
            global_indices = payload["global_indices"].long()
            if embeddings.shape[0] != global_indices.numel():
                raise ValueError("Projection cache embeddings/global_indices mismatch")
            if dim is None:
                dim = int(embeddings.shape[1])
            elif int(embeddings.shape[1]) != dim:
                raise ValueError("Projection cache dimensions differ")
            available = self.source[global_indices] < 0
            rows = global_indices[available]
            locals_ = torch.arange(global_indices.numel(), dtype=torch.int32)[available]
            self.source[rows] = source_index
            self.local[rows] = locals_
            self.embeddings.append(embeddings)
            self.metadata.append(dict(payload.get("metadata") or {}))
        self.dim = int(dim or 0)

    def missing(self, indices: list[int]) -> list[int]:
        tensor = torch.tensor(indices, dtype=torch.long)
        return tensor[self.source[tensor] < 0].tolist()

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        original_shape = tuple(indices.shape)
        flat = indices.reshape(-1).cpu().long()
        sources = self.source[flat]
        if bool((sources < 0).any()):
            raise KeyError(f"Projected rows missing: {flat[sources < 0][:5].tolist()}")
        output = torch.empty((flat.numel(), self.dim), dtype=torch.float32)
        for source_index, embeddings in enumerate(self.embeddings):
            mask = sources == source_index
            if bool(mask.any()):
                output[mask] = embeddings[self.local[flat[mask]].long()].float()
        return output.reshape(*original_shape, self.dim)


@torch.inference_mode()
def build_supplement(
    path: Path,
    missing: list[int],
    mol_store: MoleculeEmbeddingStore,
    projection: ProjectionModel,
    device: torch.device,
    batch_size: int,
    run_k_info: dict[str, Any],
) -> dict[str, Any]:
    global_indices = torch.tensor(sorted(missing), dtype=torch.long)
    embeddings = torch.empty((len(global_indices), 1024), dtype=torch.float16)
    for start in tqdm(range(0, len(global_indices), batch_size), desc="Building test projection supplement"):
        rows = global_indices[start : start + batch_size]
        raw = mol_store.gather(rows).to(device)
        embeddings[start : start + len(rows)] = projection.project_mol(raw).cpu().half()
    payload = {
        "global_indices": global_indices,
        "embeddings": embeddings,
        "metadata": {
            "representation": "frozen_run_k_projected_candidates_test_supplement",
            "rows": len(global_indices),
            "projection_dim": 1024,
            "dtype": str(embeddings.dtype),
            "run_k_checkpoint": run_k_info["path"],
            "run_k_checkpoint_sha256": run_k_info["sha256"],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


@dataclass
class RerankExample:
    order_index: int
    identifier: str
    spec_feat: torch.Tensor
    cand_feat: torch.Tensor
    stage1_scores: torch.Tensor
    gt_index: int
    stage1_rank: int


@dataclass
class SplitData:
    examples: list[RerankExample]
    identifiers: list[str]
    baseline_ranks: list[int]
    stats: dict[str, int]


@torch.inference_mode()
def build_split_data(
    query_set: QuerySet,
    dreams: torch.Tensor,
    projected_store: CompositeProjectedStore,
    projection: ProjectionModel,
    top_k: int,
    device: torch.device,
    batch_size: int,
) -> SplitData:
    examples: list[RerankExample] = []
    identifiers = [query.identifier for query in query_set.queries]
    baseline_ranks = [0] * len(query_set.queries)
    gt_not_in_top_k = 0
    temp = projection.temperature()
    for start in tqdm(range(0, len(query_set.queries), batch_size), desc="Building frozen Run-K topK"):
        batch = query_set.queries[start : start + batch_size]
        spec = dreams[[query.spec_index for query in batch]].float().to(device)
        spec_proj = projection.project_spec(spec)
        max_candidates = max(len(query.candidate_indices) for query in batch)
        padded: list[list[int]] = []
        masks: list[list[bool]] = []
        for query in batch:
            row = query.candidate_indices
            padded.append(row + [row[0]] * (max_candidates - len(row)))
            masks.append([True] * len(row) + [False] * (max_candidates - len(row)))
        index_tensor = torch.tensor(padded, dtype=torch.long)
        mol_proj = projected_store.gather(index_tensor).to(device)
        scores = torch.einsum("bd,bcd->bc", spec_proj, mol_proj) / temp
        mask_tensor = torch.tensor(masks, dtype=torch.bool, device=device)
        scores = scores.masked_fill(~mask_tensor, float("-inf"))
        order = torch.argsort(scores, dim=1, descending=True)
        for row_index, query in enumerate(batch):
            valid_count = len(query.candidate_indices)
            gt_pool_index = query.candidate_indices.index(query.gt_global_index)
            valid_order = order[row_index, :valid_count]
            rank = int((valid_order == gt_pool_index).nonzero(as_tuple=False)[0].item()) + 1
            baseline_ranks[query.order_index] = rank
            selected_order = valid_order[: min(top_k, valid_count)].detach().cpu().long()
            if gt_pool_index not in selected_order.tolist():
                gt_not_in_top_k += 1
                continue
            examples.append(
                RerankExample(
                    order_index=query.order_index,
                    identifier=query.identifier,
                    spec_feat=spec_proj[row_index].detach().cpu().half(),
                    cand_feat=mol_proj[row_index, selected_order.to(device)].detach().cpu().half(),
                    stage1_scores=scores[row_index, selected_order.to(device)].detach().cpu().float(),
                    gt_index=selected_order.tolist().index(gt_pool_index),
                    stage1_rank=rank,
                )
            )
    stats = dict(query_set.stats) | {
        "gt_not_in_top_k": gt_not_in_top_k,
        "rerankable_usable": len(examples),
    }
    return SplitData(examples, identifiers, baseline_ranks, stats)


class ExampleDataset(Dataset[RerankExample]):
    def __init__(self, examples: list[RerankExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> RerankExample:
        return self.examples[index]


def collate_examples(items: list[RerankExample]) -> dict[str, Any]:
    max_k = max(item.cand_feat.shape[0] for item in items)
    dim = items[0].cand_feat.shape[1]
    cand_feat = torch.zeros((len(items), max_k, dim), dtype=torch.float16)
    # Padding must stay finite while it is an MLP input.  The separate mask is
    # applied to final logits before CE/evaluation.
    stage1_scores = torch.zeros((len(items), max_k), dtype=torch.float32)
    mask = torch.zeros((len(items), max_k), dtype=torch.bool)
    for row, item in enumerate(items):
        count = item.cand_feat.shape[0]
        cand_feat[row, :count] = item.cand_feat
        stage1_scores[row, :count] = item.stage1_scores
        mask[row, :count] = True
    return {
        "order_index": torch.tensor([item.order_index for item in items], dtype=torch.long),
        "spec_feat": torch.stack([item.spec_feat for item in items]),
        "cand_feat": cand_feat,
        "stage1_scores": stage1_scores,
        "mask": mask,
        "gt_index": torch.tensor([item.gt_index for item in items], dtype=torch.long),
        "stage1_rank": torch.tensor([item.stage1_rank for item in items], dtype=torch.long),
    }


class PairwiseReranker(nn.Module):
    def __init__(self, proj_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(proj_dim * 4 + 1),
            nn.Linear(proj_dim * 4 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, spec: torch.Tensor, mol: torch.Tensor, stage1: torch.Tensor) -> torch.Tensor:
        expanded = spec.unsqueeze(1).expand(-1, mol.shape[1], -1)
        features = torch.cat(
            [expanded, mol, torch.abs(expanded - mol), expanded * mol, stage1.unsqueeze(-1)], dim=-1
        )
        return self.net(features).squeeze(-1)


def train_epoch(
    model: PairwiseReranker,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stage1_weight: float,
) -> dict[str, Any]:
    model.train()
    total_loss = 0.0
    total_items = 0
    first_grad = None
    for batch_index, batch in enumerate(tqdm(loader, desc="Training Run-K pairwise reranker")):
        spec = batch["spec_feat"].float().to(device)
        mol = batch["cand_feat"].float().to(device)
        stage1 = batch["stage1_scores"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["gt_index"].to(device)
        scores = model(spec, mol, stage1).masked_fill(~mask, float("-inf"))
        loss = F.cross_entropy(scores + stage1_weight * stage1, labels)
        if not bool(torch.isfinite(scores[mask]).all()) or not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite reranker score/loss on valid candidates")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if batch_index == 0:
            grads = [parameter.grad.detach().norm() for parameter in model.parameters() if parameter.grad is not None]
            first_grad = {
                "reranker_params_with_grad": len(grads),
                "reranker_grad_norm": float(torch.stack(grads).norm().cpu()) if grads else 0.0,
            }
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * spec.shape[0]
        total_items += spec.shape[0]
    return {
        "loss": total_loss / max(total_items, 1),
        "items": total_items,
        "first_batch_gradients": first_grad,
    }


@torch.no_grad()
def collect_scores(model: PairwiseReranker, data: SplitData, device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    model.eval()
    loader = DataLoader(ExampleDataset(data.examples), batch_size=batch_size, shuffle=False, collate_fn=collate_examples)
    collected: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="Scoring Run-K pairwise reranker"):
        spec = batch["spec_feat"].float().to(device)
        mol = batch["cand_feat"].float().to(device)
        stage1 = batch["stage1_scores"].to(device)
        mask = batch["mask"].to(device)
        verifier = model(spec, mol, stage1).masked_fill(~mask, float("-inf"))
        for row in range(spec.shape[0]):
            count = int(mask[row].sum().item())
            collected.append(
                {
                    "order_index": int(batch["order_index"][row]),
                    "stage1": stage1[row, :count].cpu(),
                    "verifier": verifier[row, :count].cpu(),
                    "label": int(batch["gt_index"][row]),
                    "stage1_rank": int(batch["stage1_rank"][row]),
                }
            )
    return collected


def policy_result(data: SplitData, collected: list[dict[str, Any]], alpha: float, gate: float) -> dict[str, Any]:
    ranks = list(data.baseline_ranks)
    rank1_total = 0
    rank1_damaged = 0
    rank21_50_total = 0
    rank21_50_to_top20 = 0
    rank2_50_total = 0
    rank2_50_to_top1 = 0
    for row in collected:
        rank = gt_rank_under_policy(row["stage1"], row["verifier"], row["label"], alpha, gate)
        ranks[row["order_index"]] = rank
        stage1_rank = row["stage1_rank"]
        if stage1_rank == 1:
            rank1_total += 1
            rank1_damaged += int(rank > 1)
        if 21 <= stage1_rank <= len(row["stage1"]):
            rank21_50_total += 1
            rank21_50_to_top20 += int(rank <= 20)
        if 2 <= stage1_rank <= len(row["stage1"]):
            rank2_50_total += 1
            rank2_50_to_top1 += int(rank == 1)
    return {
        **rank_metrics(ranks),
        "alpha": alpha,
        "gate": gate,
        "rank1_total": rank1_total,
        "rank1_damage_rate": rank1_damaged / max(rank1_total, 1),
        "rank21_50_total": rank21_50_total,
        "rank21_50_to_top20_rate": rank21_50_to_top20 / max(rank21_50_total, 1),
        "rank2_50_total": rank2_50_total,
        "rank2_50_to_top1_rate": rank2_50_to_top1 / max(rank2_50_total, 1),
        "ranks": ranks,
    }


def evaluate_grid(
    model: PairwiseReranker,
    data: SplitData,
    device: torch.device,
    batch_size: int,
    alpha_grid: list[float],
    gate_grid: list[float],
) -> dict[str, Any]:
    collected = collect_scores(model, data, device, batch_size)
    verifier = policy_result(data, collected, 0.0, 0.0)
    combined = {
        policy_key(alpha, gate): policy_result(data, collected, alpha, gate)
        for alpha in alpha_grid
        for gate in gate_grid
    }
    return {
        "stage1": rank_metrics(data.baseline_ranks),
        "verifier_only": {key: value for key, value in verifier.items() if key != "ranks"},
        "verifier_only_ranks": verifier["ranks"],
        "combined_by_policy": combined,
    }


def choose_policy(report: dict[str, Any], damage_target: float, mrr_tolerance: float) -> str:
    candidates = [
        (key, value)
        for key, value in report["combined_by_policy"].items()
        if float(value["rank1_damage_rate"]) <= damage_target
    ]
    if not candidates:
        candidates = list(report["combined_by_policy"].items())
    best_mrr = max(float(value["mrr"]) for _, value in candidates)
    near = [(key, value) for key, value in candidates if float(value["mrr"]) >= best_mrr - mrr_tolerance]
    return max(
        near,
        key=lambda item: (
            -float(item[1]["rank1_damage_rate"]),
            float(item[1]["mrr"]),
            float(item[1]["recall@20"]),
            float(item[1]["recall@1"]),
        ),
    )[0]


def strip_ranks(report: dict[str, Any]) -> dict[str, Any]:
    output = dict(report)
    output.pop("verifier_only_ranks", None)
    output["combined_by_policy"] = {
        key: {metric: value for metric, value in row.items() if metric != "ranks"}
        for key, row in report["combined_by_policy"].items()
    }
    return output


def contributions(ranks: list[int]) -> np.ndarray:
    values = np.asarray(ranks, dtype=np.float64)
    return np.stack(
        [(values <= 1) * 100, (values <= 5) * 100, (values <= 20) * 100, 100.0 / values], axis=1
    )


def paired_bootstrap(baseline: list[int], selected: list[int], samples: int, seed: int) -> dict[str, Any]:
    delta = contributions(selected) - contributions(baseline)
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    for start in range(0, samples, 256):
        count = min(256, samples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        chunks.append(delta[indices].mean(axis=1))
    boot = np.concatenate(chunks)
    names = ["recall@1", "recall@5", "recall@20", "mrr"]
    output = {
        name: {
            "delta_percentage_points": float(delta[:, index].mean()),
            "ci95": [float(np.quantile(boot[:, index], 0.025)), float(np.quantile(boot[:, index], 0.975))],
        }
        for index, name in enumerate(names)
    }
    output["pilot_gate"] = {
        "thresholds": {"recall@1_gain_pp": 0.5, "mrr_gain_pp": 0.3, "recall@20_min_pp": -0.2},
        "checks": {
            "recall@1": output["recall@1"]["delta_percentage_points"] >= 0.5,
            "mrr": output["mrr"]["delta_percentage_points"] >= 0.3,
            "recall@20": output["recall@20"]["delta_percentage_points"] >= -0.2,
        },
    }
    output["pilot_gate"]["passed"] = all(output["pilot_gate"]["checks"].values())
    return output


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.top_k < 2:
        raise ValueError("epochs must be positive and top-k >= 2")
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    train_records, train_dreams = load_split_inputs(
        args.train_metadata, args.train_dreams, args.train_limit, args.seed
    )
    val_records, val_dreams = load_split_inputs(args.val_metadata, args.val_dreams, args.val_limit, args.seed + 1)
    test_records, test_dreams = load_split_inputs(
        args.test_metadata, args.test_dreams, args.test_limit, args.seed + 2
    )

    sidecar = load_candidate_sidecar(args.candidate_sidecar)
    print("Loading molecule index/cache with mmap...", flush=True)
    chemberta = torch.load(args.chemberta_cache, map_location="cpu", weights_only=False, mmap=True)
    smiles_to_index = {str(key): int(value) for key, value in chemberta["smiles_to_index"].items()}
    train_queries = build_queries(train_records, sidecar, smiles_to_index, len(train_dreams), args.candidate_pool_limit)
    val_queries = build_queries(val_records, sidecar, smiles_to_index, len(val_dreams), args.candidate_pool_limit)
    test_queries = build_queries(test_records, sidecar, smiles_to_index, len(test_dreams), args.candidate_pool_limit)
    if any(query_set.stats["gt_missing_from_candidate_pool"] for query_set in (train_queries, val_queries, test_queries)):
        raise ValueError("GT is absent from an original cap256 candidate pool; refusing to insert it")

    projection, run_k_info = load_run_k(args.run_k_checkpoint, device)
    primary = torch.load(args.primary_projection_cache, map_location="cpu", weights_only=False, mmap=True)
    primary_hash = str((primary.get("metadata") or {}).get("run_k_checkpoint_sha256") or "")
    if primary_hash and primary_hash != run_k_info["sha256"]:
        raise ValueError("Primary projected cache uses another Run-K checkpoint")
    total_rows = len(smiles_to_index)
    primary_store = CompositeProjectedStore([primary], total_rows, preload=False)
    required = sorted(
        {
            index
            for query_set in (train_queries, val_queries, test_queries)
            for query in query_set.queries
            for index in query.candidate_indices
        }
    )
    missing = primary_store.missing(required)
    supplement = None
    if args.supplement_projection_cache.exists():
        supplement = torch.load(args.supplement_projection_cache, map_location="cpu", weights_only=False, mmap=True)
        supplement_hash = str((supplement.get("metadata") or {}).get("run_k_checkpoint_sha256") or "")
        if supplement_hash and supplement_hash != run_k_info["sha256"]:
            raise ValueError("Supplement projected cache uses another Run-K checkpoint")
    if supplement is None or CompositeProjectedStore([primary, supplement], total_rows, preload=False).missing(required):
        print(f"Building projection supplement for {len(missing):,} missing candidate rows", flush=True)
        mol_store = MoleculeEmbeddingStore(
            chemberta,
            "count_morgan_r3_atom_pair_path_maccs",
            args.fingerprint_cache,
            additional_fingerprint_caches=args.additional_fingerprint_cache,
            preload_sparse_features=False,
        )
        supplement = build_supplement(
            args.supplement_projection_cache,
            missing,
            mol_store,
            projection,
            device,
            args.projection_build_batch_size,
            run_k_info,
        )
    projected = CompositeProjectedStore([primary, supplement], total_rows, args.preload_projected_caches)
    remaining = projected.missing(required)
    if remaining:
        raise ValueError(f"Projected caches still miss {len(remaining)} required rows")

    train_data = build_split_data(
        train_queries, train_dreams, projected, projection, args.top_k, device, args.stage1_batch_size
    )
    val_data = build_split_data(val_queries, val_dreams, projected, projection, args.top_k, device, args.stage1_batch_size)
    test_data = build_split_data(test_queries, test_dreams, projected, projection, args.top_k, device, args.stage1_batch_size)
    build_stats = {"train": train_data.stats, "val": val_data.stats, "test": test_data.stats}
    baseline = {
        "train": rank_metrics(train_data.baseline_ranks),
        "val": rank_metrics(val_data.baseline_ranks),
        "test": rank_metrics(test_data.baseline_ranks),
    }
    print(json.dumps({"build_stats": build_stats, "stage1": baseline}, indent=2), flush=True)

    expected_val = (run_k_info.get("best_val") or {})
    expected_test_path = REPO_ROOT / "outputs/msalign_molecule_ablation/run_k_r3_allfp_blockl2_20k_cap256_5ep_test_best.json"
    expected_test = json.loads(expected_test_path.read_text())["metrics"] if expected_test_path.exists() else {}
    parity = {}
    for split, expected in (("val", expected_val), ("test", expected_test)):
        parity[split] = {
            metric: float(baseline[split][metric]) - float(expected[metric])
            for metric in ("recall@1", "recall@5", "recall@20", "mrr")
            if metric in expected
        }
    if args.val_limit == 0 and args.test_limit == 0:
        max_delta = max(abs(value) for row in parity.values() for value in row.values())
        if max_delta > 0.02:
            raise RuntimeError(f"Frozen Run-K baseline parity failed: {parity}")

    common_config = {
        "args": json_safe(vars(args)),
        "protocol": "MassSpecGym v1.5 formula-group; mass cap256; frozen Run-K; original Stage1 top50",
        "run_k": run_k_info,
        "projection_cache_metadata": projected.metadata,
        "build_stats": build_stats,
        "baseline": baseline,
        "baseline_parity_delta_pp": parity,
    }
    if args.cache_only:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(common_config, indent=2, sort_keys=True) + "\n")
        return

    alpha_grid = parse_float_grid(args.alpha_grid)
    gate_grid = parse_float_grid(args.gate_grid)
    if 1.0 not in alpha_grid:
        raise ValueError("alpha-grid must contain 1.0 so the unchanged Stage1 baseline is selectable")
    model = PairwiseReranker(1024, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        ExampleDataset(train_data.examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_examples,
    )

    history: list[dict[str, Any]] = []
    initial = evaluate_grid(model, val_data, device, args.eval_batch_size, alpha_grid, gate_grid)
    initial_policy = choose_policy(initial, args.damage_target, args.mrr_tolerance)
    best_epoch = 0
    best_policy = initial_policy
    best_mrr = float(initial["combined_by_policy"][initial_policy]["mrr"])
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history.append(
        {
            "epoch": 0,
            "selected_policy": initial_policy,
            "val_selected": {k: v for k, v in initial["combined_by_policy"][initial_policy].items() if k != "ranks"},
            "val_verifier_only": initial["verifier_only"],
        }
    )
    for epoch in range(1, args.epochs + 1):
        train_report = train_epoch(model, loader, optimizer, device, args.stage1_weight_train)
        val_report = evaluate_grid(model, val_data, device, args.eval_batch_size, alpha_grid, gate_grid)
        selected = choose_policy(val_report, args.damage_target, args.mrr_tolerance)
        selected_metrics = val_report["combined_by_policy"][selected]
        history.append(
            {
                "epoch": epoch,
                "train": train_report,
                "selected_policy": selected,
                "val_selected": {k: v for k, v in selected_metrics.items() if k != "ranks"},
                "val_verifier_only": val_report["verifier_only"],
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if float(selected_metrics["mrr"]) > best_mrr:
            best_epoch = epoch
            best_policy = selected
            best_mrr = float(selected_metrics["mrr"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_final = evaluate_grid(model, val_data, device, args.eval_batch_size, alpha_grid, gate_grid)
    best_policy = choose_policy(val_final, args.damage_target, args.mrr_tolerance)
    test_final = evaluate_grid(model, test_data, device, args.eval_batch_size, [float(best_policy.split("|")[0].split("=")[1])], [float(best_policy.split("|")[1].split("=")[1])])
    test_selected = test_final["combined_by_policy"][best_policy]
    selected_ranks = test_selected["ranks"]
    bootstrap = paired_bootstrap(test_data.baseline_ranks, selected_ranks, args.bootstrap_samples, args.seed + 100)

    rank_output = args.rank_output or args.out_json.with_name(f"{args.out_json.stem}.ranks.json")
    rank_payload = {
        "identifiers": test_data.identifiers,
        "stage1_ranks": test_data.baseline_ranks,
        "selected_ranks": selected_ranks,
        "selected_policy": best_policy,
    }
    rank_output.parent.mkdir(parents=True, exist_ok=True)
    rank_output.write_text(json.dumps(rank_payload, separators=(",", ":")) + "\n")

    payload = common_config | {
        "history": history,
        "best_epoch": best_epoch,
        "selected_policy": best_policy,
        "val_selected_policy_metrics": {
            key: value for key, value in val_final["combined_by_policy"][best_policy].items() if key != "ranks"
        },
        "test_selected_policy_metrics": {key: value for key, value in test_selected.items() if key != "ranks"},
        "val_unconditional": strip_ranks(val_final),
        "test_unconditional": strip_ranks(test_final),
        "paired_bootstrap": bootstrap,
        "rank_output": str(rank_output),
        "elapsed_seconds": time.time() - started,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "best_epoch": best_epoch,
                "selected_policy": best_policy,
                "config": common_config,
            },
            args.checkpoint,
        )
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "best_epoch": best_epoch,
                "selected_policy": best_policy,
                "val_stage1": baseline["val"],
                "val_selected": payload["val_selected_policy_metrics"],
                "test_stage1": baseline["test"],
                "test_selected": payload["test_selected_policy_metrics"],
                "paired_bootstrap": bootstrap,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
