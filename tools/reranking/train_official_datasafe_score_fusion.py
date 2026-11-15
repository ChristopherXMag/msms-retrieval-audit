#!/usr/bin/env python3
"""Train a split-safe top-5 score-fusion reranker on frozen SCCF logits.

The shared candidate MLP sees only relative scores from three frozen Stage-1
seeds.  Candidate identity, candidate slot, and query GT formula are never model
features.  GT is never inserted: queries outside the frozen ensemble top-5 are
exact no-ops and remain in every unconditional metric.
"""

from __future__ import annotations

import argparse
import hashlib
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

from tools.analysis.analyze_stage1_logit_ensemble import (  # noqa: E402
    contributions,
    grouped,
    metrics,
    paired_bootstrap,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-scores", type=Path, required=True)
    parser.add_argument("--val-scores", type=Path, required=True)
    parser.add_argument("--train-summary", type=Path, required=True)
    parser.add_argument("--val-summary", type=Path, required=True)
    parser.add_argument("--negative-exclusion-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--calibration-folds", type=int, default=5)
    parser.add_argument("--calibration-fold", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20261001)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=(20261001, 20261002, 20261003))
    parser.add_argument("--damage-cap", type=float, default=0.08)
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


def stable_fold(group: str, folds: int, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}\0{group}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % folds


@dataclass
class Prepared:
    identifiers: np.ndarray
    groups: np.ndarray
    baseline_ranks: np.ndarray
    query_indices: np.ndarray
    features: np.ndarray
    base_scores: np.ndarray
    labels: np.ndarray
    source_sha256: str
    audit: dict[str, int]


def prepare(
    path: Path,
    top_k: int,
    query_limit: int = 0,
    excluded_rows: np.ndarray | None = None,
) -> Prepared:
    arrays = np.load(path)
    identifiers = arrays["identifiers"]
    groups = arrays["connectivities"]
    indptr = arrays["indptr"]
    scores = arrays["scores"]
    ensemble = arrays["ensemble_scores"]
    candidate_rows = arrays["candidate_rows"]
    stored_ranks = arrays["ranks_ensemble_mean_logits"].astype(np.int32)
    if query_limit > 0:
        query_count = min(query_limit, len(identifiers))
        identifiers = identifiers[:query_count]
        groups = groups[:query_count]
        indptr = indptr[: query_count + 1]
        stored_ranks = stored_ranks[:query_count]
    else:
        query_count = len(identifiers)
    if scores.shape[0] != 3:
        raise ValueError(f"Expected exactly three Stage-1 score rows, got {scores.shape}")
    feature_rows: list[np.ndarray] = []
    base_rows: list[np.ndarray] = []
    labels: list[int] = []
    query_indices: list[int] = []
    computed_ranks = np.empty(query_count, dtype=np.int32)
    exclusion_mask: np.ndarray | None = None
    excluded_occurrences = 0
    affected_queries = 0
    if excluded_rows is not None:
        exclusion_mask = np.zeros(int(candidate_rows.max()) + 1, dtype=bool)
        valid_excluded = excluded_rows[excluded_rows < len(exclusion_mask)]
        exclusion_mask[valid_excluded.astype(np.int64)] = True
    for query in range(query_count):
        start, end = int(indptr[query]), int(indptr[query + 1])
        row_scores = np.asarray(scores[:, start:end], dtype=np.float32).T
        base = np.asarray(ensemble[start:end], dtype=np.float32)
        if exclusion_mask is not None:
            rows = np.asarray(candidate_rows[start:end], dtype=np.int64)
            keep = ~exclusion_mask[rows]
            keep[0] = True
            removed = int(np.count_nonzero(~keep))
            excluded_occurrences += removed
            affected_queries += int(removed > 0)
            row_scores = row_scores[keep]
            base = base[keep]
        order = np.argsort(-base, kind="stable")
        computed_ranks[query] = int(np.flatnonzero(order == 0)[0]) + 1
        selected = order[:top_k]
        gt = np.flatnonzero(selected == 0)
        if not len(gt):
            continue
        means = row_scores.mean(axis=0, keepdims=True)
        stds = row_scores.std(axis=0, keepdims=True)
        z = (row_scores - means) / np.maximum(stds, 1e-6)
        z_selected = z[selected]
        seed_margins = z_selected - z.max(axis=0, keepdims=True)
        ensemble_mean = base.mean()
        ensemble_std = max(float(base.std()), 1e-6)
        ensemble_z = (base[selected] - ensemble_mean) / ensemble_std
        ensemble_margin = ensemble_z - float(ensemble_z.max())
        disagreement = z_selected.std(axis=1)
        span = z_selected.max(axis=1) - z_selected.min(axis=1)
        feature = np.concatenate(
            [
                z_selected,
                seed_margins,
                z_selected.mean(axis=1, keepdims=True),
                disagreement[:, None],
                span[:, None],
                ensemble_z[:, None],
                ensemble_margin[:, None],
            ],
            axis=1,
        ).astype(np.float32)
        feature_rows.append(feature)
        base_rows.append(base[selected].astype(np.float32))
        labels.append(int(gt[0]))
        query_indices.append(query)
    if excluded_rows is None and not np.array_equal(computed_ranks, stored_ranks):
        mismatch = int(np.count_nonzero(computed_ranks != stored_ranks))
        raise ValueError(f"Frozen ensemble rank parity failed for {path}: {mismatch} rows")
    return Prepared(
        identifiers=np.asarray(identifiers),
        groups=np.asarray(groups),
        baseline_ranks=computed_ranks,
        query_indices=np.asarray(query_indices, dtype=np.int32),
        features=np.stack(feature_rows),
        base_scores=np.stack(base_rows),
        labels=np.asarray(labels, dtype=np.int64),
        source_sha256=sha256_file(path),
        audit={
            "excluded_negative_occurrences": excluded_occurrences,
            "candidate_rows_affected_by_exclusion": affected_queries,
            "top5_reachable_queries": len(query_indices),
        },
    )


class ScoreFusion(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor, base_scores: torch.Tensor) -> torch.Tensor:
        return base_scores + self.net(features).squeeze(-1)


def make_loader(data: Prepared, example_mask: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    indices = np.flatnonzero(example_mask)
    dataset = TensorDataset(
        torch.from_numpy(data.features[indices]),
        torch.from_numpy(data.base_scores[indices]),
        torch.from_numpy(data.labels[indices]),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def train_epoch(
    model: ScoreFusion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for features, base, labels in loader:
        features = features.to(device)
        base = base.to(device)
        labels = labels.to(device)
        logits = model(features, base)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * len(labels)
        count += len(labels)
    return total / max(count, 1)


@torch.no_grad()
def collect_scores(
    model: ScoreFusion,
    data: Prepared,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output = np.empty_like(data.base_scores)
    for start in range(0, len(data.features), batch_size):
        end = min(start + batch_size, len(data.features))
        features = torch.from_numpy(data.features[start:end]).to(device)
        base = torch.from_numpy(data.base_scores[start:end]).to(device)
        output[start:end] = model(features, base).cpu().numpy()
    return output


def ranks_from_scores(data: Prepared, scores: np.ndarray) -> np.ndarray:
    ranks = data.baseline_ranks.copy()
    order = np.argsort(-scores, axis=1, kind="stable")
    reranked = np.argmax(order == data.labels[:, None], axis=1) + 1
    ranks[data.query_indices] = reranked.astype(np.int32)
    return ranks


def endpoint(
    baseline: np.ndarray,
    ranks: np.ndarray,
    groups: np.ndarray,
    query_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if query_mask is None:
        query_mask = np.ones(len(ranks), dtype=bool)
    b = baseline[query_mask]
    r = ranks[query_mask]
    g = groups[query_mask]
    base_values = contributions(b)
    rank_values = contributions(r)
    base_group = grouped(base_values, g)[2]
    rank_group = grouped(rank_values, g)[2]
    rescue = int(np.count_nonzero((b > 1) & (r == 1)))
    damage = int(np.count_nonzero((b == 1) & (r > 1)))
    base_metrics = metrics(base_values)
    rank_metrics = metrics(rank_values)
    return {
        "queries": int(query_mask.sum()),
        "connectivities": int(len(np.unique(g))),
        "baseline_query_weighted": base_metrics,
        "reranked_query_weighted": rank_metrics,
        "delta_query_weighted_pp": {
            key: round(rank_metrics[key] - base_metrics[key], 6) for key in base_metrics
        },
        "baseline_connectivity_balanced": metrics(base_group),
        "reranked_connectivity_balanced": metrics(rank_group),
        "delta_connectivity_balanced_pp": {
            key: round(float((rank_group[:, i] - base_group[:, i]).mean() * 100.0), 6)
            for i, key in enumerate(("recall@1", "recall@5", "recall@20", "mrr"))
        },
        "rescues": rescue,
        "damages": damage,
        "net_correct": rescue - damage,
        "rescue_damage_ratio": round(rescue / max(damage, 1), 6),
        "baseline_rank1_damage_rate": round(damage / max(int(np.count_nonzero(b == 1)), 1), 8),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_epoch(
    train: Prepared,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]]]:
    query_folds = np.asarray(
        [stable_fold(str(group), args.calibration_folds, args.split_seed) for group in train.groups],
        dtype=np.int16,
    )
    calibration_query = query_folds == args.calibration_fold
    train_example = query_folds[train.query_indices] != args.calibration_fold
    seed_everything(seed)
    model = ScoreFusion(train.features.shape[-1], args.hidden_dim, args.dropout).to(device)
    loader = make_loader(train, train_example, args.batch_size, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, Any]] = []
    initial_ranks = ranks_from_scores(train, train.base_scores)
    initial = endpoint(train.baseline_ranks, initial_ranks, train.groups, calibration_query)
    history.append({"epoch": 0, "loss": None, "calibration": initial})
    for epoch in range(1, args.max_epochs + 1):
        loss = train_epoch(model, loader, optimizer, device)
        scores = collect_scores(model, train, device, args.batch_size * 2)
        ranks = ranks_from_scores(train, scores)
        history.append({
            "epoch": epoch,
            "loss": loss,
            "calibration": endpoint(train.baseline_ranks, ranks, train.groups, calibration_query),
        })
    eligible = [
        row for row in history
        if float(row["calibration"]["baseline_rank1_damage_rate"]) <= args.damage_cap
    ]
    if not eligible:
        eligible = [history[0]]
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
    train: Prepared,
    epochs: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> ScoreFusion:
    seed_everything(seed)
    model = ScoreFusion(train.features.shape[-1], args.hidden_dim, args.dropout).to(device)
    if epochs == 0:
        return model
    loader = make_loader(
        train,
        np.ones(len(train.query_indices), dtype=bool),
        args.batch_size,
        seed,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(epochs):
        train_epoch(model, loader, optimizer, device)
    return model


def main() -> None:
    args = parse_args()
    if args.top_k != 5:
        raise ValueError("This frozen first baseline is explicitly top-k=5")
    if not 0 <= args.calibration_fold < args.calibration_folds:
        raise ValueError("Invalid calibration fold")
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exclusion_manifest = json.loads(
        args.negative_exclusion_manifest.read_text(encoding="utf-8")
    )
    exclusion_path = args.negative_exclusion_manifest.parent / str(
        exclusion_manifest["indices"]
    )
    excluded_rows = np.load(exclusion_path)
    train = prepare(
        args.train_scores,
        args.top_k,
        args.train_query_limit,
        excluded_rows=excluded_rows,
    )
    val = prepare(args.val_scores, args.top_k, args.val_query_limit)
    if train.features.shape[-1] != val.features.shape[-1]:
        raise ValueError("Train/validation feature dimensions differ")
    histories: dict[str, Any] = {}
    selected_epochs: dict[str, int] = {}
    val_scores_by_seed: list[np.ndarray] = []
    individual: dict[str, Any] = {}
    checkpoint_files: list[dict[str, Any]] = []
    for model_seed in args.model_seeds:
        epoch, history = select_epoch(train, model_seed, args, device)
        selected_epochs[str(model_seed)] = epoch
        histories[str(model_seed)] = history
        model = fit_full(train, epoch, model_seed, args, device)
        val_scores = collect_scores(model, val, device, args.batch_size * 2)
        val_scores_by_seed.append(val_scores)
        ranks = ranks_from_scores(val, val_scores)
        individual[str(model_seed)] = endpoint(val.baseline_ranks, ranks, val.groups)
        checkpoint = args.output_dir / f"model_seed{model_seed}.pt"
        torch.save({
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "selected_epochs": epoch,
            "model_seed": model_seed,
            "feature_names": [
                "seed_z_0", "seed_z_1", "seed_z_2",
                "seed_margin_0", "seed_margin_1", "seed_margin_2",
                "seed_z_mean", "seed_z_std", "seed_z_span",
                "ensemble_z", "ensemble_margin",
            ],
        }, checkpoint)
        checkpoint_files.append({"path": str(checkpoint), "sha256": sha256_file(checkpoint)})
    mean_scores = np.mean(np.stack(val_scores_by_seed), axis=0)
    final_ranks = ranks_from_scores(val, mean_scores)
    final = endpoint(val.baseline_ranks, final_ranks, val.groups)
    bootstrap = paired_bootstrap(
        val.baseline_ranks,
        final_ranks,
        val.groups,
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    if not np.array_equal(final_ranks <= 5, val.baseline_ranks <= 5):
        raise ValueError("Top-5 membership changed; top-5 reranking contract violated")
    if not np.array_equal(final_ranks <= 20, val.baseline_ranks <= 20):
        raise ValueError("Top-20 membership changed; top-5 reranking contract violated")
    rank_path = args.output_dir / "val_ranks.npz"
    np.savez_compressed(
        rank_path,
        identifiers=val.identifiers,
        connectivities=val.groups,
        baseline_ranks=val.baseline_ranks,
        reranked_ranks=final_ranks,
        model_seeds=np.asarray(args.model_seeds, dtype=np.int64),
    )
    train_summary = json.loads(args.train_summary.read_text(encoding="utf-8"))
    val_summary = json.loads(args.val_summary.read_text(encoding="utf-8"))
    report = {
        "format": "glmr_official_datasafe_top5_score_fusion_v1",
        "status": "official_validation_stage2_baseline",
        "official_test_read": False,
        "protocol": {
            "stage1": "unweighted mean-logit ensemble of three frozen official/data-safe SCCF checkpoints",
            "stage2": "shared candidate MLP residual over three-seed relative score/disagreement features",
            "top_k": args.top_k,
            "gt_insertion": False,
            "outside_top5_policy": "exact frozen ensemble no-op",
            "candidate_slot_feature": False,
            "query_gt_formula_feature": False,
            "epoch_selection": "one connectivity-disjoint official-train calibration fold; <=8% rank1 damage",
            "final_training": "all official-train queries whose GT naturally occurs in frozen ensemble top5",
            "train_negative_exclusion": "same held-out-connectivity denylist as frozen Stage-1; GT always retained",
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "inputs": {
            "train_scores": {"path": str(args.train_scores), "sha256": train.source_sha256},
            "val_scores": {"path": str(args.val_scores), "sha256": val.source_sha256},
            "train_score_summary": {"path": str(args.train_summary), "sha256": sha256_file(args.train_summary)},
            "val_score_summary": {"path": str(args.val_summary), "sha256": sha256_file(args.val_summary)},
            "negative_exclusion_manifest": {"path": str(args.negative_exclusion_manifest), "sha256": sha256_file(args.negative_exclusion_manifest)},
            "negative_exclusion_indices": {"path": str(exclusion_path), "sha256": sha256_file(exclusion_path)},
            "stage1_checkpoints": val_summary["checkpoints"],
        },
        "data": {
            "train_queries": len(train.identifiers),
            "train_top5_reachable": len(train.query_indices),
            "train_connectivities": int(len(np.unique(train.groups))),
            "val_queries": len(val.identifiers),
            "val_top5_reachable": len(val.query_indices),
            "val_connectivities": int(len(np.unique(val.groups))),
            "train_val_connectivity_overlap": int(len(set(train.groups.tolist()) & set(val.groups.tolist()))),
            "feature_dim": int(train.features.shape[-1]),
            "train_candidate_exclusion_audit": train.audit,
            "val_candidate_audit": val.audit,
        },
        "selected_epochs": selected_epochs,
        "internal_train_calibration_history": histories,
        "individual_validation_endpoints": individual,
        "mean_model_score_validation_endpoint": final,
        "paired_bootstrap_vs_frozen_ensemble": bootstrap,
        "checkpoint_files": checkpoint_files,
        "rank_file": {"path": str(rank_path), "sha256": sha256_file(rank_path)},
        "interpretation_guardrail": "This baseline tests whether cross-seed score geometry is exploitable; it introduces no new fragment/local chemical evidence.",
    }
    if report["data"]["train_val_connectivity_overlap"] != 0:
        raise ValueError("Official train/validation connectivity overlap is non-zero")
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
