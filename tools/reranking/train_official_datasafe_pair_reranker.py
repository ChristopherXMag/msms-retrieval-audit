#!/usr/bin/env python3
"""Train an official/data-safe top-5 nonlinear spectrum–candidate reranker.

The frozen Stage-1 mean-logit ensemble defines the top-5.  A shared residual
MLP sees the primary SCCF seed's aligned spectrum/molecule vectors plus
relative three-seed scores.  It never sees candidate slot, molecule row ID, or
query GT formula.  GT is never inserted and queries outside top-5 are no-ops.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analysis.analyze_stage1_logit_ensemble import paired_bootstrap, sha256_file  # noqa: E402
from tools.reranking.train_official_datasafe_score_fusion import (  # noqa: E402
    endpoint,
    stable_fold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-summary", type=Path, required=True)
    parser.add_argument("--candidate-projection", type=Path, required=True)
    parser.add_argument("--train-top5", type=Path, required=True)
    parser.add_argument("--val-top5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--bottleneck-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--calibration-folds", type=int, default=5)
    parser.add_argument("--calibration-fold", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20261001)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=(20261001, 20261002, 20261003))
    parser.add_argument("--damage-cap", type=float, default=0.08)
    parser.add_argument(
        "--training-balance",
        choices=("natural", "rank1_vs_challenger_equal"),
        default="natural",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--train-query-limit", type=int, default=0)
    parser.add_argument("--val-query-limit", type=int, default=0)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


@dataclass
class Split:
    identifiers: list[str]
    groups: np.ndarray
    spec: torch.Tensor
    candidate_local: torch.Tensor
    stage1_scores: torch.Tensor
    ensemble_scores: torch.Tensor
    labels: torch.Tensor
    baseline_ranks: np.ndarray
    audit: dict[str, Any]


def load_split(path: Path, limit: int) -> Split:
    payload = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    count = len(payload["identifiers"])
    if limit > 0:
        count = min(count, limit)
    result = Split(
        identifiers=list(payload["identifiers"][:count]),
        groups=np.asarray(payload["connectivities"][:count]),
        spec=payload["spec_embeddings"][:count],
        candidate_local=payload["candidate_local_indices"][:count].long(),
        stage1_scores=payload["stage1_scores"][:count],
        ensemble_scores=payload["ensemble_scores"][:count],
        labels=payload["labels"][:count].long(),
        baseline_ranks=payload["baseline_ranks"][:count].numpy().astype(np.int32),
        audit=dict(payload["audit"]),
    )
    reachable = result.labels.numpy() >= 0
    if not np.array_equal(reachable, result.baseline_ranks <= 5):
        raise ValueError(f"GT/top5 reachability parity failed for {path}")
    return result


def score_features(stage1_scores: torch.Tensor, ensemble_scores: torch.Tensor) -> torch.Tensor:
    values = stage1_scores.float()
    mean = values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    z = (values - mean) / std
    margins = z - z.max(dim=1, keepdim=True).values
    z_mean = z.mean(dim=2, keepdim=True)
    z_std = z.std(dim=2, keepdim=True, unbiased=False)
    z_span = z.max(dim=2, keepdim=True).values - z.min(dim=2, keepdim=True).values
    ensemble = ensemble_scores.float()
    ensemble_z = (ensemble - ensemble.mean(dim=1, keepdim=True)) / ensemble.std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)
    ensemble_margin = ensemble_z - ensemble_z.max(dim=1, keepdim=True).values
    return torch.cat(
        [z, margins, z_mean, z_std, z_span, ensemble_z.unsqueeze(-1), ensemble_margin.unsqueeze(-1)],
        dim=-1,
    )


class PairReranker(nn.Module):
    def __init__(
        self,
        projection_dim: int,
        score_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        input_dim = projection_dim * 4 + score_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        spec: torch.Tensor,
        molecule: torch.Tensor,
        relative_scores: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> torch.Tensor:
        expanded = spec.unsqueeze(1).expand_as(molecule)
        features = torch.cat(
            [expanded, molecule, torch.abs(expanded - molecule), expanded * molecule, relative_scores],
            dim=-1,
        )
        return base_scores + self.net(features).squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(indices: np.ndarray, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(indices.astype(np.int64)))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def forward_batch(
    model: PairReranker,
    split: Split,
    molecule_projection: torch.Tensor,
    query_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    spec = split.spec[query_indices].float().to(device)
    local = split.candidate_local[query_indices]
    molecule = molecule_projection[local].float().to(device)
    stage_scores = split.stage1_scores[query_indices].to(device)
    ensemble = split.ensemble_scores[query_indices].float().to(device)
    relative = score_features(stage_scores, ensemble)
    return model(spec, molecule, relative, ensemble)


def train_epoch(
    model: PairReranker,
    split: Split,
    molecule_projection: torch.Tensor,
    query_indices: np.ndarray,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    seed: int,
    device: torch.device,
    training_balance: str,
) -> float:
    model.train()
    total = 0.0
    count = 0
    if training_balance == "rank1_vs_challenger_equal":
        rank1_total = int(np.count_nonzero(split.baseline_ranks[query_indices] == 1))
        challenger_total = len(query_indices) - rank1_total
        if rank1_total == 0 or challenger_total == 0:
            raise ValueError("Balanced training requires both rank1 and rank2-5 examples")
        rank1_weight = len(query_indices) / (2.0 * rank1_total)
        challenger_weight = len(query_indices) / (2.0 * challenger_total)
    else:
        rank1_weight = challenger_weight = 1.0
    for (indices,) in loader(query_indices, batch_size, seed, True):
        logits = forward_batch(model, split, molecule_projection, indices, device)
        labels = split.labels[indices].to(device)
        losses = F.cross_entropy(logits, labels, reduction="none")
        if training_balance == "rank1_vs_challenger_equal":
            is_rank1 = torch.from_numpy(split.baseline_ranks[indices.numpy()] == 1).to(device)
            weights = torch.where(
                is_rank1,
                torch.as_tensor(rank1_weight, device=device),
                torch.as_tensor(challenger_weight, device=device),
            )
            loss = (losses * weights).mean()
        else:
            loss = losses.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * len(indices)
        count += len(indices)
    return total / max(count, 1)


@torch.no_grad()
def collect_scores(
    model: PairReranker,
    split: Split,
    molecule_projection: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output = np.empty((len(split.identifiers), 5), dtype=np.float32)
    indices = np.arange(len(split.identifiers), dtype=np.int64)
    for (batch,) in loader(indices, batch_size, 0, False):
        output[batch.numpy()] = forward_batch(
            model, split, molecule_projection, batch, device
        ).cpu().numpy()
    return output


def ranks_from_scores(split: Split, scores: np.ndarray) -> np.ndarray:
    ranks = split.baseline_ranks.copy()
    reachable = split.labels.numpy() >= 0
    order = np.argsort(-scores[reachable], axis=1, kind="stable")
    labels = split.labels.numpy()[reachable]
    ranks[reachable] = (np.argmax(order == labels[:, None], axis=1) + 1).astype(np.int32)
    return ranks


def make_model(split: Split, args: argparse.Namespace, device: torch.device) -> PairReranker:
    score_dim = int(score_features(split.stage1_scores[:1], split.ensemble_scores[:1]).shape[-1])
    return PairReranker(
        int(split.spec.shape[1]), score_dim, args.hidden_dim, args.bottleneck_dim, args.dropout
    ).to(device)


def select_epoch(
    train: Split,
    molecule_projection: torch.Tensor,
    model_seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]]]:
    folds = np.asarray(
        [stable_fold(str(group), args.calibration_folds, args.split_seed) for group in train.groups],
        dtype=np.int16,
    )
    calibration = folds == args.calibration_fold
    reachable = train.labels.numpy() >= 0
    training_queries = np.flatnonzero(reachable & ~calibration)
    seed_everything(model_seed)
    model = make_model(train, args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, Any]] = []
    history.append({
        "epoch": 0,
        "loss": None,
        "calibration": endpoint(
            train.baseline_ranks, train.baseline_ranks, train.groups, calibration
        ),
    })
    for epoch in range(1, args.max_epochs + 1):
        loss = train_epoch(
            model, train, molecule_projection, training_queries, optimizer,
            args.batch_size, model_seed + epoch, device, args.training_balance,
        )
        scores = collect_scores(model, train, molecule_projection, args.batch_size * 2, device)
        ranks = ranks_from_scores(train, scores)
        history.append({
            "epoch": epoch,
            "loss": loss,
            "calibration": endpoint(train.baseline_ranks, ranks, train.groups, calibration),
        })
    eligible = [
        row for row in history
        if float(row["calibration"]["baseline_rank1_damage_rate"]) <= args.damage_cap
    ] or [history[0]]
    best = max(
        eligible,
        key=lambda row: (
            float(row["calibration"]["reranked_query_weighted"]["recall@1"]),
            float(row["calibration"]["reranked_connectivity_balanced"]["recall@1"]),
            float(row["calibration"]["reranked_query_weighted"]["mrr"]),
            -int(row["epoch"]),
        ),
    )
    return int(best["epoch"]), history


def fit_full(
    train: Split,
    molecule_projection: torch.Tensor,
    epochs: int,
    model_seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> PairReranker:
    seed_everything(model_seed)
    model = make_model(train, args, device)
    if epochs == 0:
        return model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    queries = np.flatnonzero(train.labels.numpy() >= 0)
    for epoch in range(1, epochs + 1):
        train_epoch(
            model, train, molecule_projection, queries, optimizer,
            args.batch_size, model_seed + epoch, device, args.training_balance,
        )
    return model


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_summary = json.loads(args.cache_summary.read_text(encoding="utf-8"))
    projection_payload = torch.load(
        str(args.candidate_projection), map_location="cpu", weights_only=False, mmap=True
    )
    molecule_projection = projection_payload["embeddings"]
    train = load_split(args.train_top5, args.train_query_limit)
    val = load_split(args.val_top5, args.val_query_limit)
    overlap = len(set(train.groups.tolist()) & set(val.groups.tolist()))
    if overlap:
        raise ValueError(f"Train/validation connectivity overlap: {overlap}")
    selected_epochs: dict[str, int] = {}
    histories: dict[str, Any] = {}
    individual: dict[str, Any] = {}
    val_score_sets: list[np.ndarray] = []
    checkpoint_files: list[dict[str, Any]] = []
    for model_seed in args.model_seeds:
        selected, history = select_epoch(
            train, molecule_projection, model_seed, args, device
        )
        selected_epochs[str(model_seed)] = selected
        histories[str(model_seed)] = history
        model = fit_full(
            train, molecule_projection, selected, model_seed, args, device
        )
        scores = collect_scores(
            model, val, molecule_projection, args.batch_size * 2, device
        )
        val_score_sets.append(scores)
        ranks = ranks_from_scores(val, scores)
        individual[str(model_seed)] = endpoint(
            val.baseline_ranks, ranks, val.groups
        )
        checkpoint = args.output_dir / f"model_seed{model_seed}.pt"
        torch.save({
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "selected_epochs": selected,
            "model_seed": model_seed,
        }, checkpoint)
        checkpoint_files.append({"path": str(checkpoint), "sha256": sha256_file(checkpoint)})
    mean_scores = np.mean(np.stack(val_score_sets), axis=0)
    final_ranks = ranks_from_scores(val, mean_scores)
    final = endpoint(val.baseline_ranks, final_ranks, val.groups)
    bootstrap = paired_bootstrap(
        val.baseline_ranks, final_ranks, val.groups,
        args.bootstrap_resamples, args.bootstrap_seed,
    )
    if not np.array_equal(final_ranks <= 5, val.baseline_ranks <= 5):
        raise ValueError("Top5 membership changed")
    if not np.array_equal(final_ranks <= 20, val.baseline_ranks <= 20):
        raise ValueError("Top20 membership changed")
    ranks_path = args.output_dir / "val_ranks.npz"
    np.savez_compressed(
        ranks_path,
        identifiers=np.asarray(val.identifiers),
        connectivities=val.groups,
        baseline_ranks=val.baseline_ranks,
        reranked_ranks=final_ranks,
        model_seeds=np.asarray(args.model_seeds, dtype=np.int64),
    )
    report = {
        "format": "glmr_official_datasafe_top5_pair_reranker_v1",
        "status": "official_validation_stage2_baseline",
        "official_test_read": False,
        "protocol": {
            "top_k": 5,
            "top5_source": "frozen three-seed mean-logit SCCF ensemble",
            "pair_representation": "primary seed20260621 aligned spectrum/molecule vectors",
            "candidate_features": "[spec,molecule,abs difference,product] plus relative three-seed scores",
            "shared_candidate_mlp": True,
            "candidate_slot_feature": False,
            "candidate_row_id_feature": False,
            "query_gt_formula_feature": False,
            "gt_insertion": False,
            "outside_top5_policy": "exact frozen ensemble no-op",
            "train_negative_exclusion": "same held-out-connectivity denylist as frozen SCCF",
            "epoch_selection": "one connectivity-disjoint official-train calibration fold; <=8% rank1 damage",
            "training_balance": args.training_balance,
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "inputs": {
            "cache_summary": {"path": str(args.cache_summary), "sha256": sha256_file(args.cache_summary)},
            "candidate_projection": {"path": str(args.candidate_projection), "sha256": sha256_file(args.candidate_projection)},
            "train_top5": {"path": str(args.train_top5), "sha256": sha256_file(args.train_top5)},
            "val_top5": {"path": str(args.val_top5), "sha256": sha256_file(args.val_top5)},
            "projection_checkpoint": cache_summary["projection_checkpoint"],
        },
        "data": {
            "train_queries": len(train.identifiers),
            "train_top5_reachable": int(np.count_nonzero(train.labels.numpy() >= 0)),
            "train_connectivities": int(len(np.unique(train.groups))),
            "val_queries": len(val.identifiers),
            "val_top5_reachable": int(np.count_nonzero(val.labels.numpy() >= 0)),
            "val_connectivities": int(len(np.unique(val.groups))),
            "train_val_connectivity_overlap": overlap,
            "projection_dim": int(train.spec.shape[1]),
            "unique_candidate_projections": int(len(molecule_projection)),
        },
        "selected_epochs": selected_epochs,
        "internal_train_calibration_history": histories,
        "individual_validation_endpoints": individual,
        "mean_model_score_validation_endpoint": final,
        "paired_bootstrap_vs_frozen_ensemble": bootstrap,
        "checkpoint_files": checkpoint_files,
        "rank_file": {"path": str(ranks_path), "sha256": sha256_file(ranks_path)},
        "interpretation_guardrail": "This tests nonlinear reuse of frozen aligned vectors; it is not independent fragment/local chemical evidence.",
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "selected_epochs": selected_epochs,
        "baseline": final["baseline_query_weighted"],
        "reranked": final["reranked_query_weighted"],
        "delta": final["delta_query_weighted_pp"],
        "connectivity_delta": final["delta_connectivity_balanced_pp"],
        "rescues": final["rescues"],
        "damages": final["damages"],
    }, indent=2))


if __name__ == "__main__":
    main()
