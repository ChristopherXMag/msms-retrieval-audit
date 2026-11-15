"""Train a frozen-DreaMS token readout adapter for MSAlign Stage1.

This is a Stage1 retriever experiment, not a topK reranker. It consumes cached
DreaMS backbone token states [precursor + top100 peaks], learns a small
Perceiver-style readout, and keeps the molecule side anchored to the existing
no_head ChemBERTa projection checkpoint.

Default first pass:
  DreaMS token cache frozen
  ChemBERTa embedding cache frozen
  mol_proj loaded from no_head checkpoint and frozen
  spec_proj loaded from no_head checkpoint and trainable
  h_spec = H[:, 0, :] + gate * adapter(H)
  candidate-based CE over each spectrum's mass candidate pool
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.stage1.train_msalign_projection_smoke import (  # noqa: E402
    ProjectionHead,
    add_morgan_hard_negative_order,
    append_inbatch_gt_indices,
    load_candidate_sidecar,
    load_jsonl,
    load_mol_embeddings,
    sample_candidate_indices,
)


DEFAULT_CHEMBERTA_CACHE = REPO_ROOT / "outputs/msalign_stage1_full_v15/chemberta_full_v15_mass_pool256.pt"
DEFAULT_MASS_SIDECAR = REPO_ROOT / "outputs/msalign_stage1_full_v15/chemberta_full_v15_mass_pool256.candidates.jsonl"
DEFAULT_NO_HEAD_CKPT = (
    REPO_ROOT
    / "outputs/msalign_dreams_no_head_formula_group_masscand"
    / "projection_no_head_formula_group_masscand_mean_mlp_h2048_neg127_lr1e-4_bs128_fixedtemp0.07_5ep.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a gated Perceiver readout over frozen DreaMS token states for MSAlign Stage1."
    )
    parser.add_argument("--train-metadata", type=Path, default=Path("outputs/msalign_formula_group_split_v15/train.metadata.jsonl"))
    parser.add_argument("--val-metadata", type=Path, default=Path("outputs/msalign_formula_group_split_v15/val.metadata.jsonl"))
    parser.add_argument("--test-metadata", type=Path, default=None)
    parser.add_argument("--train-token-cache", type=Path, required=True)
    parser.add_argument("--val-token-cache", type=Path, required=True)
    parser.add_argument("--test-token-cache", type=Path, default=None)
    parser.add_argument("--chemberta-cache", type=Path, default=DEFAULT_CHEMBERTA_CACHE)
    parser.add_argument("--candidate-sidecar", type=Path, default=DEFAULT_MASS_SIDECAR)
    parser.add_argument("--projection-checkpoint", type=Path, default=DEFAULT_NO_HEAD_CKPT)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mol-pooling", choices=["mean", "cls", "mean_cls_concat", "morgan", "mean_morgan_concat"], default="mean")
    parser.add_argument("--candidate-pool-limit", type=int, default=256)
    parser.add_argument("--candidate-negatives", type=int, default=127)
    parser.add_argument(
        "--negative-sampling",
        choices=["random", "morgan_tanimoto", "morgan_tanimoto_hybrid"],
        default="random",
    )
    parser.add_argument("--hard-negative-fraction", type=float, default=0.5)
    parser.add_argument("--loss-mode", choices=["candidate_ce", "candidate_ce_inbatch_gt"], default="candidate_ce")
    parser.add_argument("--train-samples", type=int, default=0, help="0 means all cache-covered rows.")
    parser.add_argument("--val-samples", type=int, default=0, help="0 means all cache-covered rows.")
    parser.add_argument("--test-samples", type=int, default=0, help="0 means all cache-covered rows.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--adapter-queries", type=int, default=4)
    parser.add_argument("--adapter-heads", type=int, default=8)
    parser.add_argument("--adapter-ffn-dim", type=int, default=2048)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    parser.add_argument("--adapter-gate-init", type=float, default=0.01)
    parser.add_argument("--latent-output", action="store_true", help="Return M spectrum latents and pool them before spec_proj.")
    parser.add_argument("--latent-pool", choices=["mean", "attention"], default="mean")
    parser.add_argument(
        "--score-mode",
        choices=["meanpool", "mean_sim", "max_spec", "logsumexp_spec"],
        default="meanpool",
        help=(
            "How to aggregate M spectrum latents into a candidate score (molecule N=1). "
            "meanpool: pool M latents to 1 vector then cos (current baseline). "
            "mean_sim: mean over per-latent cos. "
            "max_spec: max over per-latent cos (best-matching viewpoint). "
            "logsumexp_spec: soft max over per-latent cos (use --maxsim-temp). "
            "With M=1 all four are numerically near-identical (smoke self-check)."
        ),
    )
    parser.add_argument("--maxsim-temp", type=float, default=0.1, help="Temperature for logsumexp_spec; small->max_spec, large->mean_sim.")

    parser.add_argument("--export-latents", type=Path, default=None, help="Optional .pt path for best-checkpoint val/test spectrum latents.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-mol-proj", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-spec-proj", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze-temperature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(name)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_token_cache(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing = sorted({"tokens", "mask", "identifiers"} - set(payload))
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    identifiers = [str(x) for x in payload["identifiers"]]
    payload["identifiers"] = identifiers
    if "identifier_to_index" not in payload:
        payload["identifier_to_index"] = {identifier: idx for idx, identifier in enumerate(identifiers)}
    else:
        payload["identifier_to_index"] = {str(k): int(v) for k, v in payload["identifier_to_index"].items()}
    return payload


def checkpoint_args(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("args")
    return dict(raw) if isinstance(raw, dict) else {}


def load_projection_state(path: Path, spec_dim: int, mol_dim: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint_args(payload)
    state = payload.get("best_model_state_dict") or payload.get("model_state_dict") or payload.get("last_model_state_dict")
    if state is None:
        raise ValueError(f"No projection state dict found in {path}")
    if state and all(str(key).startswith("projection.") for key in state):
        state = {str(key).removeprefix("projection."): value for key, value in state.items()}
    info = {
        "path": str(path),
        "best_epoch": payload.get("best_epoch"),
        "best_val": payload.get("best_val"),
        "args": json_safe(args),
    }
    expected_spec = int(args.get("spec_dim", spec_dim))
    expected_mol = int(args.get("mol_dim", mol_dim))
    if expected_spec != spec_dim:
        # Older checkpoints do not store spec_dim; the state_dict shape check below is authoritative.
        expected_spec = spec_dim
    if expected_mol != mol_dim:
        expected_mol = mol_dim
    del expected_spec, expected_mol
    return state, info


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


def filter_to_cache(records: list[dict[str, Any]], token_cache: dict[str, Any]) -> list[dict[str, Any]]:
    available = set(token_cache["identifier_to_index"])
    return [record for record in records if str(record.get("identifier")) in available]


def sample_records(records: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(records):
        return records
    rng = random.Random(seed)
    indices = rng.sample(range(len(records)), limit)
    indices.sort()
    return [records[idx] for idx in indices]


class TokenCandidateDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[dict[str, Any]],
        token_cache: dict[str, Any],
        sidecar: dict[str, dict[str, Any]],
        smiles_to_index: dict[str, int],
        candidate_pool_limit: int,
    ) -> None:
        self.tokens: torch.Tensor = token_cache["tokens"]
        self.mask: torch.Tensor = token_cache["mask"]
        identifier_to_index: dict[str, int] = token_cache["identifier_to_index"]
        self.items: list[dict[str, Any]] = []
        stats = {
            "records_seen": len(records),
            "missing_token_cache": 0,
            "missing_sidecar": 0,
            "missing_gt_embedding": 0,
            "candidate_pool_too_small": 0,
            "usable_records": 0,
        }
        for record in records:
            identifier = str(record.get("identifier"))
            token_index = identifier_to_index.get(identifier)
            if token_index is None:
                stats["missing_token_cache"] += 1
                continue
            sidecar_record = sidecar.get(identifier)
            if sidecar_record is None:
                stats["missing_sidecar"] += 1
                continue
            gt = sidecar_record.get("normalized_gt_smiles")
            if gt is None or gt not in smiles_to_index:
                stats["missing_gt_embedding"] += 1
                continue
            pool = list(sidecar_record.get("normalized_candidates") or [])
            if candidate_pool_limit > 0:
                pool = pool[:candidate_pool_limit]
            if gt not in pool:
                pool.insert(0, gt)
            elif pool[0] != gt:
                pool = [gt] + [smiles for smiles in pool if smiles != gt]
            pool = [smiles for smiles in pool if smiles in smiles_to_index]
            if gt not in pool:
                pool.insert(0, gt)
            negatives = [smiles for smiles in pool if smiles != gt]
            if not negatives:
                stats["candidate_pool_too_small"] += 1
                continue
            self.items.append(
                {
                    "identifier": identifier,
                    "token_index": int(token_index),
                    "gt": str(gt),
                    "gt_identity": str(record.get("inchikey") or gt),
                    "candidates": [gt] + negatives,
                    "candidate_indices": [smiles_to_index[smiles] for smiles in [gt] + negatives],
                }
            )
        stats["usable_records"] = len(self.items)
        self.stats = stats

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        token_index = int(item["token_index"])
        return {
            **item,
            "tokens": self.tokens[token_index],
            "mask": self.mask[token_index],
        }


def collate_records(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return items


class GatedPerceiverReadout(nn.Module):
    def __init__(
        self,
        dim: int,
        num_queries: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        gate_init: float,
        latent_pool: str,
    ) -> None:
        super().__init__()
        if latent_pool not in {"mean", "attention"}:
            raise ValueError(f"Unsupported latent_pool: {latent_pool}")
        self.num_queries = int(num_queries)
        self.latent_pool = latent_pool
        self.input_norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.empty(num_queries, dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
        self.latent_pool_score = nn.Linear(dim, 1) if latent_pool == "attention" else None

    def query_states(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = tokens[:, 0, :].float()
        hidden = self.input_norm(tokens.float())
        key_padding_mask = ~mask.bool()
        queries = self.queries.unsqueeze(0).expand(tokens.size(0), -1, -1)
        attended, _ = self.attn(queries, hidden, hidden, key_padding_mask=key_padding_mask, need_weights=False)
        query_hidden = self.query_norm(queries + attended)
        query_hidden = self.out_norm(query_hidden + self.ffn(query_hidden))
        return h0, query_hidden

    def latents(self, tokens: torch.Tensor, mask: torch.Tensor, disable_adapter: bool = False) -> torch.Tensor:
        h0 = tokens[:, 0, :].float()
        if disable_adapter:
            return h0.unsqueeze(1).expand(-1, self.num_queries, -1)
        h0, query_hidden = self.query_states(tokens, mask)
        return h0.unsqueeze(1) + self.gate * query_hidden

    def pool_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.latent_pool == "mean":
            return latents.mean(dim=1)
        if self.latent_pool_score is None:
            raise RuntimeError("attention latent pooling was requested but latent_pool_score is missing")
        weights = torch.softmax(self.latent_pool_score(latents).squeeze(-1), dim=1)
        return torch.einsum("bm,bmd->bd", weights, latents)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, disable_adapter: bool = False) -> torch.Tensor:
        h0 = tokens[:, 0, :].float()
        if disable_adapter:
            return h0
        _, query_hidden = self.query_states(tokens, mask)
        delta = query_hidden.mean(dim=1)
        return h0 + self.gate * delta


class TokenAdapterProjectionModel(nn.Module):
    def __init__(
        self,
        spec_dim: int,
        mol_dim: int,
        hidden_dim: int,
        proj_dim: int,
        dropout: float,
        head_type: str,
        temperature: float,
        adapter_queries: int,
        adapter_heads: int,
        adapter_ffn_dim: int,
        adapter_dropout: float,
        adapter_gate_init: float,
        latent_output: bool,
        latent_pool: str,
    ) -> None:
        super().__init__()
        self.latent_output = bool(latent_output)
        self.latent_pool = latent_pool
        self.readout = GatedPerceiverReadout(
            dim=spec_dim,
            num_queries=adapter_queries,
            num_heads=adapter_heads,
            ffn_dim=adapter_ffn_dim,
            dropout=adapter_dropout,
            gate_init=adapter_gate_init,
            latent_pool=latent_pool,
        )
        self.spec_head = ProjectionHead(spec_dim, hidden_dim, proj_dim, dropout, head_type)
        self.mol_head = ProjectionHead(mol_dim, hidden_dim, proj_dim, dropout, head_type)
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())

    def spec_latents(self, tokens: torch.Tensor, mask: torch.Tensor, disable_adapter: bool = False) -> torch.Tensor:
        return self.readout.latents(tokens, mask, disable_adapter=disable_adapter)

    def spec_readout(self, tokens: torch.Tensor, mask: torch.Tensor, disable_adapter: bool = False) -> torch.Tensor:
        if self.latent_output:
            return self.readout.pool_latents(self.spec_latents(tokens, mask, disable_adapter=disable_adapter))
        return self.readout(tokens, mask, disable_adapter=disable_adapter)

    def project_spec_tokens(self, tokens: torch.Tensor, mask: torch.Tensor, disable_adapter: bool = False) -> torch.Tensor:
        return self.spec_head(self.spec_readout(tokens, mask, disable_adapter=disable_adapter))

    def project_spec_latents(self, tokens: torch.Tensor, mask: torch.Tensor, disable_adapter: bool = False) -> torch.Tensor:
        """Project each of the M spectrum latents through spec_head -> [B, M, proj].

        ProjectionHead already L2-normalizes its output, so each of the M
        projected latents is unit norm. This is the multi-vector counterpart of
        project_spec_tokens (which first pools M latents into one vector).
        """
        latents = self.spec_latents(tokens, mask, disable_adapter=disable_adapter)  # [B, M, spec_dim]
        b, m, d = latents.shape
        projected = self.spec_head(latents.reshape(b * m, d)).reshape(b, m, -1)  # [B, M, proj], unit norm per latent
        return projected

    def project_mol(self, mol: torch.Tensor) -> torch.Tensor:
        return self.mol_head(mol)

    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)

    def score_candidates(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        mol_proj: torch.Tensor,
        score_mode: str,
        maxsim_temp: float,
        disable_adapter: bool = False,
    ) -> torch.Tensor:
        """Return candidate logits [B, C] under one of four spec-latent aggregations.

        mol_proj: [B, C, proj] already L2-normalized molecule projections (N=1 per
        candidate). All four modes divide the final similarity by self.temperature()
        so they share the CE logit scale; logsumexp_spec additionally uses its own
        maxsim_temp to interpolate between mean_sim (large temp) and max_spec (small temp).

        With M=1 the four modes are numerically (near-)identical, which is used as a
        self-check in the smoke run.
        """
        temp = self.temperature()
        if score_mode == "meanpool":
            # pool M latents -> 1 vector -> spec_head (normalizes) -> cos with mol
            spec_proj = self.project_spec_tokens(tokens, mask, disable_adapter=disable_adapter)  # [B, proj]
            sims = torch.einsum("bd,bcd->bc", spec_proj, mol_proj)  # [B, C]
            return sims / temp
        # multi-vector modes: per-latent cos, then aggregate over M
        spec_latents = self.project_spec_latents(tokens, mask, disable_adapter=disable_adapter)  # [B, M, proj]
        # cos between each spectrum latent and each candidate: [B, M, C]
        sim_mc = torch.einsum("bmd,bcd->bmc", spec_latents, mol_proj)
        if score_mode == "mean_sim":
            agg = sim_mc.mean(dim=1)  # [B, C]
        elif score_mode == "max_spec":
            agg = sim_mc.max(dim=1).values  # [B, C]
        elif score_mode == "logsumexp_spec":
            tau = float(maxsim_temp)
            agg = tau * torch.logsumexp(sim_mc / tau, dim=1) - tau * torch.log(
                torch.tensor(float(sim_mc.size(1)), device=sim_mc.device)
            )
            # subtract log(M) so logsumexp_spec -> mean_sim as tau -> inf and -> max_spec as tau -> 0
        else:
            raise ValueError(f"Unsupported score_mode: {score_mode}")
        return agg / temp



def load_projection_into_model(model: TokenAdapterProjectionModel, state: dict[str, torch.Tensor]) -> None:
    own_state = model.state_dict()
    copied: list[str] = []
    for key in ("log_temperature",):
        if key in state and key in own_state:
            own_state[key].copy_(state[key])
            copied.append(key)
    for prefix in ("spec_head.", "mol_head."):
        for key, value in state.items():
            if key.startswith(prefix) and key in own_state:
                own_state[key].copy_(value)
                copied.append(key)
    if not any(key.startswith("spec_head.") for key in copied):
        raise ValueError("Projection checkpoint did not provide spec_head weights")
    if not any(key.startswith("mol_head.") for key in copied):
        raise ValueError("Projection checkpoint did not provide mol_head weights")


def set_trainable(model: TokenAdapterProjectionModel, freeze_mol_proj: bool, freeze_spec_proj: bool, freeze_temperature: bool) -> None:
    if freeze_mol_proj:
        for parameter in model.mol_head.parameters():
            parameter.requires_grad_(False)
    if freeze_spec_proj:
        for parameter in model.spec_head.parameters():
            parameter.requires_grad_(False)
    if freeze_temperature:
        model.log_temperature.requires_grad_(False)


def trainable_summary(model: nn.Module) -> dict[str, int]:
    total = 0
    trainable = 0
    by_prefix: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            prefix = name.split(".", 1)[0]
            by_prefix[prefix] = by_prefix.get(prefix, 0) + count
    return {"total": total, "trainable": trainable, **{f"trainable_{k}": v for k, v in sorted(by_prefix.items())}}


def sample_batch_candidate_indices(
    batch: list[dict[str, Any]],
    smiles_to_index: dict[str, int],
    candidate_negatives: int,
    rng: random.Random,
    negative_sampling: str,
    hard_negative_fraction: float,
    loss_mode: str,
) -> tuple[list[list[int]], list[list[bool]], dict[str, float]]:
    candidate_indices = [
        sample_candidate_indices(
            item,
            smiles_to_index,
            candidate_negatives,
            rng,
            negative_sampling,
            hard_negative_fraction,
        )
        for item in batch
    ]
    mask = [[True] * len(indices) for indices in candidate_indices]
    stats: dict[str, float] = {}
    if loss_mode == "candidate_ce_inbatch_gt":
        candidate_indices, mask, stats = append_inbatch_gt_indices(batch, candidate_indices, mask, smiles_to_index)
    elif loss_mode != "candidate_ce":
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")
    return candidate_indices, mask, stats


def make_padded_index_tensors(candidate_indices: list[list[int]], mask: list[list[bool]]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(indices) for indices in candidate_indices)
    padded_indices = [indices + [indices[0]] * (max_len - len(indices)) for indices in candidate_indices]
    padded_mask = [row + [False] * (max_len - len(row)) for row in mask]
    return torch.tensor(padded_indices, dtype=torch.long), torch.tensor(padded_mask, dtype=torch.bool)


def grad_stats(model: nn.Module) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for name, parameter in model.named_parameters():
        prefix = name.split(".", 1)[0]
        group = stats.setdefault(prefix, {"params_with_grad": 0, "nonzero_grad_params": 0, "grad_norm": 0.0})
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().float().norm().cpu())
        group["params_with_grad"] = int(group["params_with_grad"]) + 1
        group["grad_norm"] = float(group["grad_norm"]) + norm
        if norm > 0:
            group["nonzero_grad_params"] = int(group["nonzero_grad_params"]) + 1
    return {key: {"params_with_grad": int(value["params_with_grad"]), "nonzero_grad_params": int(value["nonzero_grad_params"]), "grad_norm": round(float(value["grad_norm"]), 6)} for key, value in stats.items()}


def train_one_epoch(
    model: TokenAdapterProjectionModel,
    loader: DataLoader,
    mol_embeddings: torch.Tensor,
    smiles_to_index: dict[str, int],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    candidate_negatives: int,
    loss_mode: str,
    negative_sampling: str,
    hard_negative_fraction: float,
    seed: int,
    max_train_batches: int,
    score_mode: str = "meanpool",
    maxsim_temp: float = 0.1,
) -> dict[str, Any]:
    model.train()
    rng = random.Random(seed)

    total_loss = 0.0
    total_items = 0
    total_valid_extra = 0.0
    total_masked_extra = 0.0
    first_batch_grad_stats: dict[str, Any] | None = None
    for batch_idx, batch in enumerate(tqdm(loader, desc="Training token adapter"), start=1):
        tokens = torch.stack([item["tokens"] for item in batch], dim=0).to(device)
        token_mask = torch.stack([item["mask"] for item in batch], dim=0).to(device)
        candidate_indices, candidate_mask, inbatch_stats = sample_batch_candidate_indices(
            batch,
            smiles_to_index,
            candidate_negatives,
            rng,
            negative_sampling,
            hard_negative_fraction,
            loss_mode,
        )
        total_valid_extra += inbatch_stats.get("inbatch_gt_valid_per_query", 0.0) * len(batch)
        total_masked_extra += inbatch_stats.get("inbatch_gt_masked_per_query", 0.0) * len(batch)
        index_tensor, mask_tensor = make_padded_index_tensors(candidate_indices, candidate_mask)
        mol = mol_embeddings[index_tensor.reshape(-1)].reshape(len(batch), index_tensor.shape[1], -1).to(device)
        mask_tensor = mask_tensor.to(device)

        mol_proj = model.project_mol(mol.reshape(-1, mol.shape[-1])).reshape(len(batch), index_tensor.shape[1], -1)
        logits = model.score_candidates(tokens, token_mask, mol_proj, score_mode, maxsim_temp)
        logits = logits.masked_fill(~mask_tensor, float("-inf"))
        labels = torch.zeros(len(batch), dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)


        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if first_batch_grad_stats is None:
            first_batch_grad_stats = grad_stats(model)
        optimizer.step()

        total_loss += float(loss.detach().cpu()) * len(batch)
        total_items += len(batch)
        if max_train_batches > 0 and batch_idx >= max_train_batches:
            break
    metrics: dict[str, Any] = {
        "loss": total_loss / max(total_items, 1),
        "items": total_items,
        "temperature": float(model.temperature().detach().cpu()),
        "adapter_gate": float(model.readout.gate.detach().cpu()),
    }
    if first_batch_grad_stats is not None:
        metrics["first_batch_grad_stats"] = first_batch_grad_stats
    if loss_mode == "candidate_ce_inbatch_gt":
        metrics["inbatch_gt_valid_per_query"] = total_valid_extra / max(total_items, 1)
        metrics["inbatch_gt_masked_per_query"] = total_masked_extra / max(total_items, 1)
    return metrics


@torch.inference_mode()
def evaluate(
    model: TokenAdapterProjectionModel,
    dataset: TokenCandidateDataset,
    mol_embeddings: torch.Tensor,
    device: torch.device,
    batch_size: int,
    disable_adapter: bool = False,
    score_mode: str = "meanpool",
    maxsim_temp: float = 0.1,
) -> dict[str, float | int]:
    model.eval()
    ranks: list[int] = []
    for start in tqdm(range(0, len(dataset), batch_size), desc="Evaluating token adapter"):
        batch = [dataset[idx] for idx in range(start, min(start + batch_size, len(dataset)))]
        tokens = torch.stack([item["tokens"] for item in batch], dim=0).to(device)
        token_mask = torch.stack([item["mask"] for item in batch], dim=0).to(device)
        candidate_indices = [item["candidate_indices"] for item in batch]
        candidate_mask = [[True] * len(indices) for indices in candidate_indices]
        index_tensor, mask_tensor = make_padded_index_tensors(candidate_indices, candidate_mask)
        mol = mol_embeddings[index_tensor.reshape(-1)].reshape(len(batch), index_tensor.shape[1], -1).to(device)
        mask_tensor = mask_tensor.to(device)
        mol_proj = model.project_mol(mol.reshape(-1, mol.shape[-1])).reshape(len(batch), index_tensor.shape[1], -1)
        scores = model.score_candidates(tokens, token_mask, mol_proj, score_mode, maxsim_temp, disable_adapter=disable_adapter)
        scores = scores.masked_fill(~mask_tensor, float("-inf"))

        order = torch.argsort(scores, dim=1, descending=True)
        batch_ranks = (order == 0).nonzero(as_tuple=False)[:, 1] + 1
        ranks.extend(int(rank) for rank in batch_ranks.cpu())
    return rank_metrics(ranks)


@torch.inference_mode()
def collect_latents(
    model: TokenAdapterProjectionModel,
    dataset: TokenCandidateDataset,
    device: torch.device,
    batch_size: int,
    disable_adapter: bool = False,
) -> dict[str, Any]:
    model.eval()
    latent_chunks: list[torch.Tensor] = []
    identifiers: list[str] = []
    for start in tqdm(range(0, len(dataset), batch_size), desc="Exporting spectrum latents"):
        batch = [dataset[idx] for idx in range(start, min(start + batch_size, len(dataset)))]
        tokens = torch.stack([item["tokens"] for item in batch], dim=0).to(device)
        token_mask = torch.stack([item["mask"] for item in batch], dim=0).to(device)
        latents = model.spec_latents(tokens, token_mask, disable_adapter=disable_adapter)
        latent_chunks.append(latents.detach().cpu().to(torch.float16))
        identifiers.extend(str(item["identifier"]) for item in batch)
    latents = torch.cat(latent_chunks, dim=0) if latent_chunks else torch.empty(0, 0, 0, dtype=torch.float16)
    return {
        "latents": latents,
        "identifiers": identifiers,
        "identifier_to_index": {identifier: idx for idx, identifier in enumerate(identifiers)},
    }


def export_latent_cache(
    path: Path,
    model: TokenAdapterProjectionModel,
    val_dataset: TokenCandidateDataset,
    test_dataset: TokenCandidateDataset | None,
    device: torch.device,
    batch_size: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metadata": {
            "representation": "dreams_stage1_spectrum_latents",
            "config": json_safe(config),
            "dtype": "float16",
        },
        "val": collect_latents(model, val_dataset, device, batch_size),
    }
    if test_dataset is not None:
        payload["test"] = collect_latents(model, test_dataset, device, batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "val_shape": list(payload["val"]["latents"].shape),
        "test_shape": list(payload["test"]["latents"].shape) if "test" in payload else None,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    train_cache = load_token_cache(args.train_token_cache)
    val_cache = load_token_cache(args.val_token_cache)
    test_cache = load_token_cache(args.test_token_cache) if args.test_token_cache is not None else None

    train_pool = filter_to_cache(load_jsonl(args.train_metadata), train_cache)
    val_pool = filter_to_cache(load_jsonl(args.val_metadata), val_cache)
    test_pool: list[dict[str, Any]] = []
    if args.test_metadata is not None and test_cache is not None:
        test_pool = filter_to_cache(load_jsonl(args.test_metadata), test_cache)
    train_records = sample_records(train_pool, args.train_samples, args.seed)
    val_records = sample_records(val_pool, args.val_samples, args.seed + 1)
    test_records = sample_records(test_pool, args.test_samples, args.seed + 2)
    print(
        f"Cache-covered records: train={len(train_pool)} val={len(val_pool)} test={len(test_pool)}; "
        f"sampled train={len(train_records)} val={len(val_records)} test={len(test_records)}",
        flush=True,
    )

    sidecar = load_candidate_sidecar(args.candidate_sidecar)
    print("Loading ChemBERTa cache (large)...", flush=True)
    chemberta = torch.load(args.chemberta_cache, map_location="cpu", weights_only=False)
    smiles_to_index = chemberta["smiles_to_index"]
    mol_embeddings = load_mol_embeddings(chemberta, args.mol_pooling)

    train_dataset = TokenCandidateDataset(train_records, train_cache, sidecar, smiles_to_index, args.candidate_pool_limit)
    val_dataset = TokenCandidateDataset(val_records, val_cache, sidecar, smiles_to_index, args.candidate_pool_limit)
    test_dataset = (
        TokenCandidateDataset(test_records, test_cache, sidecar, smiles_to_index, args.candidate_pool_limit)
        if test_cache is not None and test_records
        else None
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(f"Empty train/val dataset: train={train_dataset.stats}, val={val_dataset.stats}")
    if args.negative_sampling.startswith("morgan_tanimoto"):
        morgan_embeddings = chemberta.get("embeddings", {}).get("morgan")
        if morgan_embeddings is None:
            raise ValueError("--negative-sampling morgan_* requires embeddings['morgan'] in --chemberta-cache")
        add_morgan_hard_negative_order(train_dataset, morgan_embeddings)

    ckpt_payload = torch.load(args.projection_checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint_args(ckpt_payload)
    projection_state, projection_info = load_projection_state(
        args.projection_checkpoint,
        spec_dim=int(train_cache["tokens"].shape[-1]),
        mol_dim=int(mol_embeddings.shape[-1]),
    )
    model = TokenAdapterProjectionModel(
        spec_dim=int(train_cache["tokens"].shape[-1]),
        mol_dim=int(mol_embeddings.shape[-1]),
        hidden_dim=int(ckpt_args.get("hidden_dim", 2048)),
        proj_dim=int(ckpt_args.get("proj_dim", 1024)),
        dropout=float(ckpt_args.get("dropout", 0.2)),
        head_type=str(ckpt_args.get("head_type", "mlp")),
        temperature=float(ckpt_args.get("temperature", 0.07)),
        adapter_queries=args.adapter_queries,
        adapter_heads=args.adapter_heads,
        adapter_ffn_dim=args.adapter_ffn_dim,
        adapter_dropout=args.adapter_dropout,
        adapter_gate_init=args.adapter_gate_init,
        latent_output=args.latent_output,
        latent_pool=args.latent_pool,
    ).to(device)
    load_projection_into_model(model, projection_state)
    set_trainable(model, args.freeze_mol_proj, args.freeze_spec_proj, args.freeze_temperature)

    summary_common = {
        "args": json_safe(vars(args)),
        "projection_checkpoint": projection_info,
        "latent_output": bool(args.latent_output),
        "latent_pool": str(args.latent_pool),
        "adapter_queries": int(args.adapter_queries),
        "latent_shape": [int(args.adapter_queries), int(train_cache["tokens"].shape[-1])],
        "train_dataset": train_dataset.stats,
        "val_dataset": val_dataset.stats,
        "test_dataset": test_dataset.stats if test_dataset is not None else None,
        "token_cache_metadata": {
            "train": train_cache.get("metadata", {}),
            "val": val_cache.get("metadata", {}),
            "test": test_cache.get("metadata", {}) if test_cache is not None else None,
        },
        "trainable": trainable_summary(model),
    }
    print(json.dumps({"trainable": summary_common["trainable"]}, indent=2, sort_keys=True), flush=True)

    baseline_val = evaluate(model, val_dataset, mol_embeddings, device, args.eval_batch_size, disable_adapter=True)
    init_val = evaluate(model, val_dataset, mol_embeddings, device, args.eval_batch_size, disable_adapter=False)
    summary_common["baseline_disable_adapter_metrics"] = baseline_val
    history: list[dict[str, Any]] = [{"epoch": 0, "val_no_adapter_baseline": baseline_val, "val": init_val}]
    print(json.dumps(history[-1], indent=2, sort_keys=True), flush=True)
    if args.cache_only:
        payload = {
            **summary_common,
            "history": history,
            "best_epoch": 0,
            "best_val": init_val,
            "adapter_gate": float(model.readout.gate.detach().cpu()),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_records, drop_last=False)
    best_epoch = 0
    best_val = init_val
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            loader,
            mol_embeddings,
            smiles_to_index,
            optimizer,
            device,
            args.candidate_negatives,
            args.loss_mode,
            args.negative_sampling,
            args.hard_negative_fraction,
            args.seed + epoch,
            args.max_train_batches,
            args.score_mode,
            args.maxsim_temp,
        )
        val_metrics = evaluate(model, val_dataset, mol_embeddings, device, args.eval_batch_size, score_mode=args.score_mode, maxsim_temp=args.maxsim_temp)

        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        if (
            float(val_metrics["mrr"]),
            float(val_metrics["recall@1"]),
            float(val_metrics["recall@20"]),
        ) > (
            float(best_val["mrr"]),
            float(best_val["recall@1"]),
            float(best_val["recall@20"]),
        ):
            best_epoch = epoch
            best_val = val_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)

    model.load_state_dict(best_state)
    test_metrics = (
        evaluate(model, test_dataset, mol_embeddings, device, args.eval_batch_size, score_mode=args.score_mode, maxsim_temp=args.maxsim_temp)
        if test_dataset is not None
        else None
    )

    latent_export = (
        export_latent_cache(
            args.export_latents,
            model,
            val_dataset,
            test_dataset,
            device,
            args.eval_batch_size,
            {
                "latent_output": bool(args.latent_output),
                "latent_pool": str(args.latent_pool),
                "adapter_queries": int(args.adapter_queries),
                "projection_checkpoint": str(args.projection_checkpoint),
                "best_epoch": int(best_epoch),
            },
        )
        if args.export_latents is not None
        else None
    )
    payload = {
        **summary_common,
        "history": history,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "test_at_best": test_metrics,
        "adapter_gate": float(model.readout.gate.detach().cpu()),
        "latent_export": latent_export,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "best_model_state_dict": best_state,
                "best_epoch": best_epoch,
                "best_val": best_val,
                "test_at_best": test_metrics,
                "adapter_gate": float(model.readout.gate.detach().cpu()),
                "latent_export": latent_export,
                "args": json_safe(vars(args)),
                "projection_checkpoint": projection_info,
                "history": history,
            },
            args.checkpoint,
        )
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "best_epoch": best_epoch,
                "best_val": best_val,
                "test_at_best": test_metrics,
                "latent_export": latent_export,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
