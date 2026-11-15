from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # Lightweight metric/data imports do not require ChemBERTa.
    AutoModel = None
    AutoTokenizer = None

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.stage1.train_msalign_projection_smoke import (  # noqa: E402
    ProjectionModel,
    load_candidate_sidecar,
    load_jsonl,
    load_mol_embeddings,
)


DEFAULT_CHEMBERTA_CACHE = REPO_ROOT / "outputs/msalign_stage1_full_v15/chemberta_full_v15_mass_pool256.pt"
DEFAULT_MASS_SIDECAR = REPO_ROOT / "outputs/msalign_stage1_full_v15/chemberta_full_v15_mass_pool256.candidates.jsonl"
DEFAULT_NO_HEAD_CKPT = (
    REPO_ROOT
    / "outputs/msalign_dreams_no_head_formula_group_masscand"
    / "projection_no_head_formula_group_masscand_mean_mlp_h2048_neg127_lr1e-4_bs128_fixedtemp0.07_5ep.pt"
)


@dataclass
class RerankExample:
    identifier: str
    token_index: int
    candidate_smiles: list[str]
    stage1_scores: torch.Tensor
    gt_index: int
    stage1_rank: int


@dataclass
class SplitData:
    examples: list[RerankExample]
    nonrerankable_ranks: list[int]
    all_stage1_ranks: list[int]
    stats: dict[str, int]


class RerankDataset(Dataset[RerankExample]):
    def __init__(self, examples: list[RerankExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> RerankExample:
        return self.examples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MSAlign-style topK cross-attention reranker over DreaMS/ChemBERTa token states."
    )
    parser.add_argument("--train-metadata", type=Path, default=Path("outputs/msalign_formula_group_split_v15/train.metadata.jsonl"))
    parser.add_argument("--val-metadata", type=Path, default=Path("outputs/msalign_formula_group_split_v15/val.metadata.jsonl"))
    parser.add_argument("--test-metadata", type=Path, default=Path("outputs/msalign_formula_group_split_v15/test.metadata.jsonl"))
    parser.add_argument("--train-token-cache", type=Path, required=True)
    parser.add_argument("--val-token-cache", type=Path, required=True)
    parser.add_argument("--test-token-cache", type=Path, required=True)
    parser.add_argument("--chemberta-cache", type=Path, default=DEFAULT_CHEMBERTA_CACHE)
    parser.add_argument("--candidate-sidecar", type=Path, default=DEFAULT_MASS_SIDECAR)
    parser.add_argument("--projection-checkpoint", type=Path, default=DEFAULT_NO_HEAD_CKPT)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--chemberta-model-name", type=str, default="DeepChem/ChemBERTa-100M-MLM")
    parser.add_argument("--mol-pooling", choices=["mean", "cls", "mean_cls_concat"], default="mean")
    parser.add_argument("--candidate-pool-limit", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--val-samples", type=int, default=4096)
    parser.add_argument("--test-samples", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--stage1-score-batch-size", type=int, default=128)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--mol-max-length", type=int, default=256)
    parser.add_argument("--attn-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage1-weight-train", type=float, default=0.0)
    parser.add_argument("--alpha-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--gate-grid", default="0,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--damage-target", type=float, default=0.08)
    parser.add_argument("--mrr-tolerance", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument(
        "--warm-start-projection",
        action="store_true",
        help=(
            "Initialize the spec/mol token projections from the trained Stage1 "
            "projection heads so cross-attention starts in the aligned space. "
            "Requires --attn-dim == Stage1 hidden_dim (default 2048)."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()



def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(name)


def parse_float_grid(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def load_token_cache(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"tokens", "mask", "identifiers"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    identifiers = [str(x) for x in payload["identifiers"]]
    if "identifier_to_index" not in payload:
        payload["identifier_to_index"] = {identifier: idx for idx, identifier in enumerate(identifiers)}
    return payload


def checkpoint_args(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("args")
    return dict(raw) if isinstance(raw, dict) else {}


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_projection_model(
    checkpoint_path: Path,
    spec_dim: int,
    mol_dim: int,
    device: torch.device,
) -> tuple[ProjectionModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint_args(payload)
    state = payload.get("best_model_state_dict") or payload.get("model_state_dict") or payload.get("last_model_state_dict")
    if state is None:
        raise ValueError(f"No projection state dict found in {checkpoint_path}")
    if state and all(str(key).startswith("projection.") for key in state):
        state = {str(key).removeprefix("projection."): value for key, value in state.items()}
    model = ProjectionModel(
        spec_dim=spec_dim,
        mol_dim=mol_dim,
        hidden_dim=int(ckpt_args.get("hidden_dim", 2048)),
        proj_dim=int(ckpt_args.get("proj_dim", 1024)),
        dropout=float(ckpt_args.get("dropout", 0.2)),
        temperature=float(ckpt_args.get("temperature", 0.07)),
        head_type=str(ckpt_args.get("head_type", "mlp")),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    info = {
        "path": str(checkpoint_path),
        "best_epoch": payload.get("best_epoch"),
        "best_val": payload.get("best_val"),
        "args": json_safe(ckpt_args),
    }
    return model, info


def rank_metrics(ranks: list[int]) -> dict[str, float | int]:
    if not ranks:
        return {
            "valid_queries": 0,
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@20": 0.0,
            "recall@50": 0.0,
            "recall@100": 0.0,
            "mrr": 0.0,
            "median_rank": 0.0,
            "mean_rank": 0.0,
        }
    values = torch.tensor(ranks, dtype=torch.float32)
    return {
        "valid_queries": len(ranks),
        "recall@1": round(float((values <= 1).float().mean()) * 100, 4),
        "recall@5": round(float((values <= 5).float().mean()) * 100, 4),
        "recall@20": round(float((values <= 20).float().mean()) * 100, 4),
        "recall@50": round(float((values <= 50).float().mean()) * 100, 4),
        "recall@100": round(float((values <= 100).float().mean()) * 100, 4),
        "mrr": round(float((1.0 / values).mean()) * 100, 4),
        "median_rank": round(float(values.median()), 4),
        "mean_rank": round(float(values.mean()), 4),
    }


def normalized_pool_from_sidecar(record: dict[str, Any], sidecar: dict[str, dict[str, Any]], limit: int) -> tuple[str | None, list[str]]:
    identifier = str(record.get("identifier"))
    sidecar_record = sidecar.get(identifier)
    if sidecar_record is None:
        return None, []
    gt = sidecar_record.get("normalized_gt_smiles")
    if gt is None:
        return None, []
    pool = list(sidecar_record.get("normalized_candidates") or [])
    if limit > 0:
        pool = pool[:limit]
    if gt not in pool:
        pool.insert(0, gt)
    return str(gt), [str(smiles) for smiles in pool]


@torch.inference_mode()
def build_split_data(
    records: list[dict[str, Any]],
    token_cache: dict[str, Any],
    sidecar: dict[str, dict[str, Any]],
    smiles_to_index: dict[str, int],
    mol_embeddings: torch.Tensor,
    projection: ProjectionModel,
    top_k: int,
    candidate_pool_limit: int,
    device: torch.device,
    score_batch_size: int,
) -> SplitData:
    tokens: torch.Tensor = token_cache["tokens"]
    identifier_to_index: dict[str, int] = {str(k): int(v) for k, v in token_cache["identifier_to_index"].items()}
    raw_items: list[dict[str, Any]] = []
    stats = {
        "records_seen": len(records),
        "missing_token_cache": 0,
        "missing_sidecar": 0,
        "missing_gt_embedding": 0,
        "candidate_pool_too_small": 0,
        "unconditional_usable": 0,
        "gt_not_in_top_k": 0,
        "rerankable_usable": 0,
    }
    for record in records:
        identifier = str(record.get("identifier"))
        token_index = identifier_to_index.get(identifier)
        if token_index is None:
            stats["missing_token_cache"] += 1
            continue
        gt, pool = normalized_pool_from_sidecar(record, sidecar, candidate_pool_limit)
        if gt is None:
            stats["missing_sidecar"] += 1
            continue
        pool = [smiles for smiles in pool if smiles in smiles_to_index]
        if gt not in smiles_to_index:
            stats["missing_gt_embedding"] += 1
            continue
        if gt not in pool:
            pool.insert(0, gt)
        if len(pool) < 2:
            stats["candidate_pool_too_small"] += 1
            continue
        raw_items.append(
            {
                "identifier": identifier,
                "token_index": token_index,
                "gt": gt,
                "candidate_smiles": pool,
                "candidate_indices": [smiles_to_index[smiles] for smiles in pool],
            }
        )

    examples: list[RerankExample] = []
    nonrerankable_ranks: list[int] = []
    all_stage1_ranks: list[int] = []
    for start in tqdm(range(0, len(raw_items), score_batch_size), desc="Building Stage1 topK"):
        batch = raw_items[start : start + score_batch_size]
        spec = tokens[[item["token_index"] for item in batch], 0, :].float().to(device)
        spec_proj = projection.project_spec(spec)
        candidate_indices = [item["candidate_indices"] for item in batch]
        max_len = max(len(indices) for indices in candidate_indices)
        padded_indices = []
        mask = []
        for indices in candidate_indices:
            padded_indices.append(indices + [indices[0]] * (max_len - len(indices)))
            mask.append([True] * len(indices) + [False] * (max_len - len(indices)))
        index_tensor = torch.tensor(padded_indices, dtype=torch.long)
        mol = mol_embeddings[index_tensor.reshape(-1)].reshape(len(batch), max_len, -1).to(device)
        mol_proj = projection.project_mol(mol.reshape(-1, mol.shape[-1])).reshape(len(batch), max_len, -1)
        scores = torch.einsum("bd,bcd->bc", spec_proj, mol_proj) / projection.temperature()
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)
        scores = scores.masked_fill(~mask_tensor, float("-inf"))
        order = torch.argsort(scores, dim=1, descending=True)
        for row_idx, item in enumerate(batch):
            gt_pool_index = item["candidate_smiles"].index(item["gt"])
            rank = int((order[row_idx] == gt_pool_index).nonzero(as_tuple=False)[0].item()) + 1
            all_stage1_ranks.append(rank)
            stats["unconditional_usable"] += 1
            top_order = order[row_idx, :top_k].detach().cpu().tolist()
            top_order = [idx for idx in top_order if idx < len(item["candidate_smiles"])]
            if gt_pool_index not in top_order:
                stats["gt_not_in_top_k"] += 1
                nonrerankable_ranks.append(rank)
                continue
            examples.append(
                RerankExample(
                    identifier=item["identifier"],
                    token_index=int(item["token_index"]),
                    candidate_smiles=[item["candidate_smiles"][idx] for idx in top_order],
                    stage1_scores=scores[row_idx, top_order].detach().cpu().float(),
                    gt_index=top_order.index(gt_pool_index),
                    stage1_rank=rank,
                )
            )
            stats["rerankable_usable"] += 1
    return SplitData(examples=examples, nonrerankable_ranks=nonrerankable_ranks, all_stage1_ranks=all_stage1_ranks, stats=stats)


def collate_examples(items: list[RerankExample], tokens: torch.Tensor, masks: torch.Tensor) -> dict[str, Any]:
    token_indices = [item.token_index for item in items]
    return {
        "identifier": [item.identifier for item in items],
        "spec_hidden": tokens[token_indices].float(),
        "spec_mask": masks[token_indices].bool(),
        "candidate_smiles": [item.candidate_smiles for item in items],
        "stage1_scores": torch.stack([item.stage1_scores for item in items], dim=0),
        "gt_index": torch.tensor([item.gt_index for item in items], dtype=torch.long),
        "stage1_rank": torch.tensor([item.stage1_rank for item in items], dtype=torch.long),
    }


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


class CrossAttentionReranker(nn.Module):
    def __init__(
        self,
        spec_dim: int,
        mol_dim: int,
        attn_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        warm_start_projection: bool = False,
    ) -> None:
        super().__init__()
        if warm_start_projection:

            # Project DreaMS / ChemBERTa tokens into the trained Stage1 alignment
            # space using the SAME LayerNorm+Linear topology as the Stage1
            # projection heads, so weights can be copied in directly (see
            # load_warm_start). This is the key fix: the old GLMR verifier worked
            # because spec/mol tokens were already in one aligned space; here we
            # warm-start the token projections into that aligned space instead of
            # learning cross-modal alignment from a random init.
            self.spec_proj = nn.Sequential(nn.LayerNorm(spec_dim), nn.Linear(spec_dim, attn_dim))
            self.mol_proj = nn.Sequential(nn.LayerNorm(mol_dim), nn.Linear(mol_dim, attn_dim))
        else:
            self.spec_proj = nn.Sequential(nn.LayerNorm(spec_dim), nn.Linear(spec_dim, attn_dim), nn.GELU())
            self.mol_proj = nn.Sequential(nn.LayerNorm(mol_dim), nn.Linear(mol_dim, attn_dim), nn.GELU())
        self.warm_start_projection = warm_start_projection
        self.spec_norm = nn.LayerNorm(attn_dim)

        self.mol_norm = nn.LayerNorm(attn_dim)
        self.spec_to_mol = nn.MultiheadAttention(attn_dim, num_heads, dropout=dropout, batch_first=True)
        self.mol_to_spec = nn.MultiheadAttention(attn_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(
            nn.Linear(attn_dim * 4 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @torch.no_grad()
    def load_warm_start(self, projection: "ProjectionModel") -> None:
        """Copy Stage1 spec_head/mol_head LayerNorm + first Linear into token projections.

        Stage1 ProjectionHead (mlp) starts with LayerNorm(in) -> Linear(in, hidden).
        With warm_start_projection we built spec_proj/mol_proj as
        LayerNorm(in) -> Linear(in, attn_dim), so this copy requires
        attn_dim == Stage1 hidden_dim. This places DreaMS / ChemBERTa tokens into
        the same subspace the aligned Stage1 global vectors live in, so the
        cross-attention starts in an aligned space instead of a random one. This
        is the structural fix for why the old GLMR verifier worked (already-aligned
        tokens) but the v1.5 random-init reranker failed.
        """
        if not self.warm_start_projection:
            raise RuntimeError("load_warm_start requires warm_start_projection=True")

        def copy_first(dst: nn.Sequential, src_head: nn.Module) -> None:
            src = src_head.net  # Sequential starting LayerNorm(in), Linear(in, hidden), ...
            dst[0].load_state_dict(src[0].state_dict())
            if tuple(dst[1].weight.shape) != tuple(src[1].weight.shape):
                raise ValueError(
                    f"warm-start shape mismatch: reranker Linear {tuple(dst[1].weight.shape)} "
                    f"vs Stage1 Linear {tuple(src[1].weight.shape)}; set --attn-dim to Stage1 hidden_dim"
                )
            dst[1].load_state_dict(src[1].state_dict())

        copy_first(self.spec_proj, projection.spec_head)
        copy_first(self.mol_proj, projection.mol_head)

    def forward(
        self,
        spec_hidden: torch.Tensor,
        spec_mask: torch.Tensor,
        mol_hidden: torch.Tensor,
        mol_mask: torch.Tensor,
        stage1_scores: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_candidates, mol_len, _ = mol_hidden.shape

        spec_len = spec_hidden.size(1)
        spec = self.spec_proj(spec_hidden.float())
        mol = self.mol_proj(mol_hidden.float())

        spec_pair = spec.unsqueeze(1).expand(batch_size, num_candidates, spec_len, spec.size(-1))
        spec_pair = spec_pair.reshape(batch_size * num_candidates, spec_len, spec.size(-1))
        spec_pair_mask = spec_mask.unsqueeze(1).expand(batch_size, num_candidates, spec_len).reshape(
            batch_size * num_candidates, spec_len
        )
        mol_pair = mol.reshape(batch_size * num_candidates, mol_len, mol.size(-1))
        mol_pair_mask = mol_mask.reshape(batch_size * num_candidates, mol_len)

        spec_normed = self.spec_norm(spec_pair)
        mol_normed = self.mol_norm(mol_pair)
        spec_cross, _ = self.spec_to_mol(
            query=spec_normed,
            key=mol_normed,
            value=mol_normed,
            key_padding_mask=~mol_pair_mask,
            need_weights=False,
        )
        mol_cross, _ = self.mol_to_spec(
            query=mol_normed,
            key=spec_normed,
            value=spec_normed,
            key_padding_mask=~spec_pair_mask,
            need_weights=False,
        )
        spec_pooled = masked_mean(spec_cross, spec_pair_mask)
        mol_pooled = masked_mean(mol_cross, mol_pair_mask)
        features = torch.cat(
            [
                spec_pooled,
                mol_pooled,
                torch.abs(spec_pooled - mol_pooled),
                spec_pooled * mol_pooled,
                stage1_scores.reshape(-1, 1),
            ],
            dim=-1,
        )
        return self.net(self.dropout(features)).reshape(batch_size, num_candidates)


@torch.no_grad()
def encode_candidate_tokens(
    candidate_smiles: list[list[str]],
    tokenizer: Any,
    model: nn.Module,
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(candidate_smiles)
    num_candidates = len(candidate_smiles[0])
    flat_smiles = [smiles for row in candidate_smiles for smiles in row]
    encoded = tokenizer(
        flat_smiles,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output = model(**encoded)
    hidden = output.last_hidden_state
    mask = encoded["attention_mask"].bool()
    hidden = hidden.reshape(batch_size, num_candidates, hidden.size(1), hidden.size(2))
    mask = mask.reshape(batch_size, num_candidates, mask.size(1))
    return hidden, mask


def first_batch_grad_stats(reranker: nn.Module, chemberta_model: nn.Module) -> dict[str, Any]:
    def stats_for(parameters: list[nn.Parameter]) -> dict[str, float | int]:
        out = {"params_with_grad": 0, "nonzero_grad_params": 0, "grad_norm": 0.0}
        for parameter in parameters:
            if parameter.grad is None:
                continue
            norm = float(parameter.grad.detach().float().norm().cpu())
            out["params_with_grad"] += 1
            out["grad_norm"] += norm
            if norm > 0:
                out["nonzero_grad_params"] += 1
        out["grad_norm"] = round(float(out["grad_norm"]), 6)
        return out

    return {
        "reranker": stats_for(list(reranker.parameters())),
        "chemberta": stats_for(list(chemberta_model.parameters())),
    }


def train_epoch(
    reranker: CrossAttentionReranker,
    chemberta_model: nn.Module,
    tokenizer: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mol_max_length: int,
    stage1_weight: float,
    max_batches: int,
) -> dict[str, Any]:
    reranker.train()
    total_loss = 0.0
    total_items = 0
    grad_stats = None
    for batch_idx, batch in enumerate(tqdm(loader, desc="Training MSAlign reranker"), start=1):
        spec_hidden = batch["spec_hidden"].to(device)
        spec_mask = batch["spec_mask"].to(device)
        stage1_scores = batch["stage1_scores"].to(device)
        labels = batch["gt_index"].to(device)
        mol_hidden, mol_mask = encode_candidate_tokens(
            batch["candidate_smiles"], tokenizer, chemberta_model, device, mol_max_length
        )
        verifier_scores = reranker(spec_hidden, spec_mask, mol_hidden, mol_mask, stage1_scores)
        loss = F.cross_entropy(verifier_scores + stage1_weight * stage1_scores, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_stats is None:
            grad_stats = first_batch_grad_stats(reranker, chemberta_model)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * spec_hidden.size(0)
        total_items += spec_hidden.size(0)
        if max_batches > 0 and batch_idx >= max_batches:
            break
    return {"loss": total_loss / max(total_items, 1), "items": total_items, "first_batch_grad_stats": grad_stats}


@dataclass
class CollectedScores:
    stage1: list[torch.Tensor]
    verifier: list[torch.Tensor]
    labels: list[int]
    stage1_ranks: list[int]


@torch.no_grad()
def collect_scores(
    reranker: CrossAttentionReranker,
    chemberta_model: nn.Module,
    tokenizer: Any,
    loader: DataLoader,
    device: torch.device,
    mol_max_length: int,
) -> CollectedScores:
    reranker.eval()
    collected = CollectedScores(stage1=[], verifier=[], labels=[], stage1_ranks=[])
    for batch in tqdm(loader, desc="Evaluating MSAlign reranker"):
        spec_hidden = batch["spec_hidden"].to(device)
        spec_mask = batch["spec_mask"].to(device)
        stage1_scores = batch["stage1_scores"].to(device)
        mol_hidden, mol_mask = encode_candidate_tokens(
            batch["candidate_smiles"], tokenizer, chemberta_model, device, mol_max_length
        )
        verifier_scores = reranker(spec_hidden, spec_mask, mol_hidden, mol_mask, stage1_scores)
        for row_idx in range(spec_hidden.size(0)):
            collected.stage1.append(stage1_scores[row_idx].detach().cpu())
            collected.verifier.append(verifier_scores[row_idx].detach().cpu())
            collected.labels.append(int(batch["gt_index"][row_idx]))
            collected.stage1_ranks.append(int(batch["stage1_rank"][row_idx]))
    return collected


def gt_rank_under_policy(stage1: torch.Tensor, verifier: torch.Tensor, label: int, alpha: float, gate: float) -> int:
    fused = alpha * stage1 + (1.0 - alpha) * verifier
    order = torch.argsort(fused, descending=True).tolist()
    if gate > 0.0 and stage1.numel() >= 2:
        stage1_order = torch.argsort(stage1, descending=True).tolist()
        margin = float(stage1[stage1_order[0]] - stage1[stage1_order[1]])
        if margin >= gate and order[0] != stage1_order[0]:
            order = [stage1_order[0]] + [idx for idx in order if idx != stage1_order[0]]
    return order.index(label) + 1


def policy_key(alpha: float, gate: float) -> str:
    return f"alpha={alpha:g}|gate={gate:g}"


def policy_metrics(collected: CollectedScores, nonrerankable_ranks: list[int], alpha: float, gate: float) -> dict[str, Any]:
    ranks = list(nonrerankable_ranks)
    rank1_total = 0
    rank1_damaged = 0
    rank21_50_total = 0
    rank21_50_to_top20 = 0
    rank2_50_total = 0
    rank2_50_to_top1 = 0
    for stage1, verifier, label, stage1_rank in zip(
        collected.stage1, collected.verifier, collected.labels, collected.stage1_ranks
    ):
        rank = gt_rank_under_policy(stage1, verifier, label, alpha, gate)
        ranks.append(rank)
        if stage1_rank == 1:
            rank1_total += 1
            if rank > 1:
                rank1_damaged += 1
        if 21 <= stage1_rank <= stage1.numel():
            rank21_50_total += 1
            if rank <= 20:
                rank21_50_to_top20 += 1
        if 2 <= stage1_rank <= stage1.numel():
            rank2_50_total += 1
            if rank == 1:
                rank2_50_to_top1 += 1
    return {
        **rank_metrics(ranks),
        "alpha": alpha,
        "gate": gate,
        "rank1_total": rank1_total,
        "rank1_damage_rate": round(rank1_damaged / rank1_total, 6) if rank1_total else 0.0,
        "rank21_50_total": rank21_50_total,
        "rank21_50_to_top20_rate": round(rank21_50_to_top20 / rank21_50_total, 6) if rank21_50_total else 0.0,
        "rank2_50_total": rank2_50_total,
        "rank2_50_to_top1_rate": round(rank2_50_to_top1 / rank2_50_total, 6) if rank2_50_total else 0.0,
    }


def best_policy(report: dict[str, Any], damage_target: float, mrr_tolerance: float) -> str:
    combined = report["combined_by_policy"]
    feasible = [(key, value) for key, value in combined.items() if value["rank1_damage_rate"] <= damage_target]
    candidates = feasible if feasible else list(combined.items())
    best_mrr = max(float(value["mrr"]) for _, value in candidates)
    near_best = [(key, value) for key, value in candidates if float(value["mrr"]) >= best_mrr - mrr_tolerance]
    return max(
        near_best,
        key=lambda item: (
            -float(item[1]["rank1_damage_rate"]),
            float(item[1]["mrr"]),
            float(item[1]["recall@20"]),
            float(item[1]["recall@5"]),
            float(item[1]["recall@1"]),
        ),
    )[0]


@torch.no_grad()
def evaluate(
    reranker: CrossAttentionReranker,
    chemberta_model: nn.Module,
    tokenizer: Any,
    data: SplitData,
    tokens: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
    batch_size: int,
    mol_max_length: int,
    alpha_grid: list[float],
    gate_grid: list[float],
) -> dict[str, Any]:
    loader = DataLoader(
        RerankDataset(data.examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_examples(batch, tokens, masks),
    )
    collected = collect_scores(reranker, chemberta_model, tokenizer, loader, device, mol_max_length)
    stage1_ranks = list(data.nonrerankable_ranks) + list(collected.stage1_ranks)
    verifier_only_ranks = list(data.nonrerankable_ranks)
    for verifier, label in zip(collected.verifier, collected.labels):
        verifier_only_ranks.append(int((torch.argsort(verifier, descending=True) == label).nonzero(as_tuple=False)[0].item()) + 1)
    combined = {
        policy_key(alpha, gate): policy_metrics(collected, data.nonrerankable_ranks, alpha, gate)
        for alpha in alpha_grid
        for gate in gate_grid
    }
    return {
        "stage1": rank_metrics(stage1_ranks),
        "verifier_only": rank_metrics(verifier_only_ranks),
        "combined_by_policy": combined,
        "nonrerankable_count": len(data.nonrerankable_ranks),
        "rerankable_count": len(data.examples),
    }


def truncate_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return records[:limit] if limit > 0 else records


def cache_only_report(config: dict[str, Any], build_stats: dict[str, Any], train_data: SplitData, val_data: SplitData, test_data: SplitData) -> dict[str, Any]:
    return {
        "config": config,
        "build_stats": build_stats,
        "split_sizes": {
            "train_rerankable": len(train_data.examples),
            "val_rerankable": len(val_data.examples),
            "test_rerankable": len(test_data.examples),
            "train_unconditional": len(train_data.all_stage1_ranks),
            "val_unconditional": len(val_data.all_stage1_ranks),
            "test_unconditional": len(test_data.all_stage1_ranks),
        },
        "stage1": {
            "train": rank_metrics(train_data.all_stage1_ranks),
            "val": rank_metrics(val_data.all_stage1_ranks),
            "test": rank_metrics(test_data.all_stage1_ranks),
        },
    }


def main() -> None:
    args = parse_args()
    if AutoModel is None or AutoTokenizer is None:
        raise ImportError(
            "transformers is required to train the cross-attention reranker; "
            "install it in the active environment before running this entrypoint"
        )
    started = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    train_records = truncate_records(load_jsonl(args.train_metadata), args.train_samples)
    val_records = truncate_records(load_jsonl(args.val_metadata), args.val_samples)
    test_records = truncate_records(load_jsonl(args.test_metadata), args.test_samples)
    train_cache = load_token_cache(args.train_token_cache)
    val_cache = load_token_cache(args.val_token_cache)
    test_cache = load_token_cache(args.test_token_cache)
    sidecar = load_candidate_sidecar(args.candidate_sidecar)
    chemberta = torch.load(args.chemberta_cache, map_location="cpu", weights_only=False)
    smiles_to_index = chemberta["smiles_to_index"]
    mol_embeddings = load_mol_embeddings(chemberta, args.mol_pooling)
    projection, projection_info = load_projection_model(
        args.projection_checkpoint,
        spec_dim=int(train_cache["tokens"].shape[-1]),
        mol_dim=int(mol_embeddings.shape[-1]),
        device=device,
    )

    print("Building train split...", flush=True)
    train_data = build_split_data(
        train_records,
        train_cache,
        sidecar,
        smiles_to_index,
        mol_embeddings,
        projection,
        args.top_k,
        args.candidate_pool_limit,
        device,
        args.stage1_score_batch_size,
    )
    print("Building val split...", flush=True)
    val_data = build_split_data(
        val_records,
        val_cache,
        sidecar,
        smiles_to_index,
        mol_embeddings,
        projection,
        args.top_k,
        args.candidate_pool_limit,
        device,
        args.stage1_score_batch_size,
    )
    print("Building test split...", flush=True)
    test_data = build_split_data(
        test_records,
        test_cache,
        sidecar,
        smiles_to_index,
        mol_embeddings,
        projection,
        args.top_k,
        args.candidate_pool_limit,
        device,
        args.stage1_score_batch_size,
    )
    build_stats = {"train": train_data.stats, "val": val_data.stats, "test": test_data.stats}
    print("Build stats:", json.dumps(build_stats, indent=2, sort_keys=True), flush=True)
    base_config = {
        "train_metadata": str(args.train_metadata),
        "val_metadata": str(args.val_metadata),
        "test_metadata": str(args.test_metadata),
        "train_token_cache": str(args.train_token_cache),
        "val_token_cache": str(args.val_token_cache),
        "test_token_cache": str(args.test_token_cache),
        "chemberta_cache": str(args.chemberta_cache),
        "candidate_sidecar": str(args.candidate_sidecar),
        "projection_checkpoint": str(args.projection_checkpoint),
        "chemberta_model_name": args.chemberta_model_name,
        "top_k": args.top_k,
        "candidate_pool_limit": args.candidate_pool_limit,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "test_samples": args.test_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "mol_max_length": args.mol_max_length,
        "attn_dim": args.attn_dim,
        "num_heads": args.num_heads,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "stage1_weight_train": args.stage1_weight_train,
        "damage_target": args.damage_target,
        "mrr_tolerance": args.mrr_tolerance,
        "device": str(device),
    }
    if args.cache_only:
        payload = cache_only_report(base_config, build_stats, train_data, val_data, test_data)
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"out_json": str(args.out_json), "stage1": payload["stage1"]}, indent=2, sort_keys=True), flush=True)
        return
    if not train_data.examples or not val_data.examples:
        raise ValueError(f"Empty rerankable train/val examples: {build_stats}")

    tokenizer = AutoTokenizer.from_pretrained(args.chemberta_model_name)
    chemberta_model = AutoModel.from_pretrained(args.chemberta_model_name).to(device)
    chemberta_model.eval()
    for parameter in chemberta_model.parameters():
        parameter.requires_grad_(False)

    reranker = CrossAttentionReranker(
        spec_dim=int(train_cache["tokens"].shape[-1]),
        mol_dim=int(chemberta_model.config.hidden_size),
        attn_dim=args.attn_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        warm_start_projection=bool(args.warm_start_projection),
    ).to(device)
    if args.warm_start_projection:
        reranker.load_warm_start(projection)
        print(
            "Warm-started spec/mol token projections from Stage1 projection heads "
            f"(attn_dim={args.attn_dim}).",
            flush=True,
        )
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    alpha_grid = parse_float_grid(args.alpha_grid)
    gate_grid = parse_float_grid(args.gate_grid)
    train_loader = DataLoader(
        RerankDataset(train_data.examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_examples(batch, train_cache["tokens"], train_cache["mask"]),
    )

    history: list[dict[str, Any]] = []
    best_state = None
    best_val_mrr = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            reranker,
            chemberta_model,
            tokenizer,
            train_loader,
            optimizer,
            device,
            args.mol_max_length,
            args.stage1_weight_train,
            args.max_train_batches,
        )
        val_report = evaluate(
            reranker,
            chemberta_model,
            tokenizer,
            val_data,
            val_cache["tokens"],
            val_cache["mask"],
            device,
            args.eval_batch_size,
            args.mol_max_length,
            alpha_grid,
            gate_grid,
        )
        selected = best_policy(val_report, args.damage_target, args.mrr_tolerance)
        val_best = val_report["combined_by_policy"][selected]
        row = {"epoch": epoch, "train": train_metrics, "best_policy": selected, "val_best": val_best}
        history.append(row)
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)
        if float(val_best["mrr"]) > best_val_mrr:
            best_val_mrr = float(val_best["mrr"])
            best_state = {key: value.detach().cpu().clone() for key, value in reranker.state_dict().items()}

    if best_state is not None:
        reranker.load_state_dict(best_state)
    val_report = evaluate(
        reranker,
        chemberta_model,
        tokenizer,
        val_data,
        val_cache["tokens"],
        val_cache["mask"],
        device,
        args.eval_batch_size,
        args.mol_max_length,
        alpha_grid,
        gate_grid,
    )
    selected_policy = best_policy(val_report, args.damage_target, args.mrr_tolerance)
    test_report = evaluate(
        reranker,
        chemberta_model,
        tokenizer,
        test_data,
        test_cache["tokens"],
        test_cache["mask"],
        device,
        args.eval_batch_size,
        args.mol_max_length,
        alpha_grid,
        gate_grid,
    )
    config = base_config | {
        "projection_info": projection_info,
        "alpha_grid": alpha_grid,
        "gate_grid": gate_grid,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    payload = {
        "config": config,
        "build_stats": build_stats,
        "split_sizes": {
            "train_rerankable": len(train_data.examples),
            "val_rerankable": len(val_data.examples),
            "test_rerankable": len(test_data.examples),
            "train_unconditional": len(train_data.all_stage1_ranks),
            "val_unconditional": len(val_data.all_stage1_ranks),
            "test_unconditional": len(test_data.all_stage1_ranks),
        },
        "history": history,
        "selected_policy": selected_policy,
        "val_selected_policy_metrics": val_report["combined_by_policy"][selected_policy],
        "test_selected_policy_metrics": test_report["combined_by_policy"][selected_policy],
        "val_unconditional": val_report,
        "test_unconditional": test_report,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": reranker.state_dict(),
                "best_model_state_dict": best_state or reranker.state_dict(),
                "config": config,
                "selected_policy": selected_policy,
                "val_selected_policy_metrics": payload["val_selected_policy_metrics"],
                "test_selected_policy_metrics": payload["test_selected_policy_metrics"],
            },
            args.checkpoint,
        )
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "selected_policy": selected_policy,
                "val_selected": payload["val_selected_policy_metrics"],
                "test_selected": payload["test_selected_policy_metrics"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
