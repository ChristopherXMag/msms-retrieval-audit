#!/usr/bin/env python3
"""Build frozen top-5 aligned-vector caches for official/data-safe Stage 2.

Top-5 membership is determined by the already frozen three-seed mean-logit
ensemble.  Training pools apply the held-out-connectivity negative denylist;
validation pools are unchanged.  No GT is inserted and no official-test input
is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glmr.smiles import profile_smiles  # noqa: E402
from tools.stage1.export_stage1_candidate_scores import sha256_file  # noqa: E402
from tools.stage1.train_msalign_projection_smoke import (  # noqa: E402
    MoleculeEmbeddingStore,
    build_projection_model,
    checkpoint_model_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-key", default="model_state_dict")
    parser.add_argument("--fingerprint-cache", type=Path, required=True)
    parser.add_argument("--train-scores", type=Path, required=True)
    parser.add_argument("--val-scores", type=Path, required=True)
    parser.add_argument("--train-dreams", type=Path, required=True)
    parser.add_argument("--val-dreams", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--val-metadata", type=Path, required=True)
    parser.add_argument("--negative-exclusion-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def topk_from_scores(
    path: Path,
    top_k: int,
    exclusion_mask: np.ndarray | None,
) -> dict[str, Any]:
    arrays = np.load(path)
    identifiers = arrays["identifiers"]
    connectivities = arrays["connectivities"]
    indptr = arrays["indptr"]
    candidate_rows = arrays["candidate_rows"]
    scores = arrays["scores"]
    ensemble = arrays["ensemble_scores"]
    stored_ranks = arrays["ranks_ensemble_mean_logits"]
    selected_rows = np.empty((len(identifiers), top_k), dtype=np.int32)
    selected_scores = np.empty((len(identifiers), top_k, 3), dtype=np.float32)
    selected_ensemble = np.empty((len(identifiers), top_k), dtype=np.float32)
    labels = np.full(len(identifiers), -1, dtype=np.int8)
    ranks = np.empty(len(identifiers), dtype=np.int16)
    excluded_occurrences = 0
    affected_queries = 0
    for query in range(len(identifiers)):
        start, end = int(indptr[query]), int(indptr[query + 1])
        rows = np.asarray(candidate_rows[start:end], dtype=np.int32)
        row_scores = np.asarray(scores[:, start:end], dtype=np.float32).T
        row_ensemble = np.asarray(ensemble[start:end], dtype=np.float32)
        if exclusion_mask is not None:
            keep = ~exclusion_mask[rows]
            keep[0] = True
            removed = int(np.count_nonzero(~keep))
            excluded_occurrences += removed
            affected_queries += int(removed > 0)
            rows = rows[keep]
            row_scores = row_scores[keep]
            row_ensemble = row_ensemble[keep]
        if len(rows) < top_k:
            raise ValueError(f"Query {identifiers[query]} has fewer than top_k candidates")
        order = np.argsort(-row_ensemble, kind="stable")
        ranks[query] = int(np.flatnonzero(order == 0)[0]) + 1
        selected = order[:top_k]
        selected_rows[query] = rows[selected]
        selected_scores[query] = row_scores[selected]
        selected_ensemble[query] = row_ensemble[selected]
        hit = np.flatnonzero(selected == 0)
        if len(hit):
            labels[query] = int(hit[0])
    if exclusion_mask is None and not np.array_equal(ranks.astype(np.int32), stored_ranks):
        raise ValueError("Validation mean-logit rank parity failed")
    return {
        "identifiers": identifiers.tolist(),
        "connectivities": connectivities.tolist(),
        "candidate_rows": selected_rows,
        "stage1_scores": selected_scores,
        "ensemble_scores": selected_ensemble,
        "labels": labels,
        "baseline_ranks": ranks,
        "audit": {
            "queries": len(identifiers),
            "top5_reachable": int(np.count_nonzero(labels >= 0)),
            "excluded_negative_occurrences": excluded_occurrences,
            "candidate_rows_affected_by_exclusion": affected_queries,
        },
    }


@torch.inference_mode()
def project_specs(
    dreams_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    projection_dim: int,
) -> torch.Tensor:
    dreams = np.load(dreams_path, mmap_mode="r")
    output = torch.empty((len(dreams), projection_dim), dtype=torch.float16)
    for start in tqdm(range(0, len(dreams), batch_size), desc=f"Projecting {dreams_path.stem}"):
        end = min(start + batch_size, len(dreams))
        values = torch.from_numpy(np.array(dreams[start:end], copy=True)).to(device)
        output[start:end] = model.project_spec(values).cpu().half()
    return output


@torch.inference_mode()
def project_molecules(
    rows: np.ndarray,
    store: MoleculeEmbeddingStore,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    projection_dim: int,
) -> torch.Tensor:
    output = torch.empty((len(rows), projection_dim), dtype=torch.float16)
    for start in tqdm(range(0, len(rows), batch_size), desc="Projecting top5 molecules"):
        end = min(start + batch_size, len(rows))
        logical = torch.from_numpy(rows[start:end].astype(np.int64, copy=False))
        identity = torch.arange(len(logical), dtype=torch.long)
        ordered, original_to_ordered = store.canonicalize_unique_indices(logical, identity)
        projected = model.project_mol(store.gather(ordered).to(device)).cpu().half()
        output[start:end] = projected[original_to_ordered]
    return output


def main() -> None:
    args = parse_args()
    if args.top_k != 5:
        raise ValueError("This cache contract is frozen at top-k=5")
    device = choose_device(args.device)
    train_metadata = load_jsonl(args.train_metadata)
    val_metadata = load_jsonl(args.val_metadata)
    exclusion = json.loads(args.negative_exclusion_manifest.read_text(encoding="utf-8"))
    exclusion_path = args.negative_exclusion_manifest.parent / str(exclusion["indices"])
    excluded_rows = np.load(exclusion_path)
    molecule_rows = int(exclusion["molecule_rows"])
    exclusion_mask = np.zeros(molecule_rows, dtype=bool)
    exclusion_mask[excluded_rows.astype(np.int64)] = True
    train = topk_from_scores(args.train_scores, args.top_k, exclusion_mask)
    val = topk_from_scores(args.val_scores, args.top_k, None)
    if len(train_metadata) != train["audit"]["queries"] or len(val_metadata) != val["audit"]["queries"]:
        raise ValueError("Metadata and score-cache query counts differ")
    if train["identifiers"] != [str(row["identifier"]) for row in train_metadata]:
        raise ValueError("Training metadata identifier order differs from score cache")
    if val["identifiers"] != [str(row["identifier"]) for row in val_metadata]:
        raise ValueError("Validation metadata identifier order differs from score cache")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = dict(checkpoint.get("args") or {})
    blocks = list(config.get("mol_blocks") or [])
    store = MoleculeEmbeddingStore(
        None,
        str(config.get("mol_pooling", "mean")),
        args.fingerprint_cache,
        additional_fingerprint_caches=[],
        preload_sparse_features=True,
        mol_blocks=blocks,
    )
    train_dreams = np.load(args.train_dreams, mmap_mode="r")
    model = build_projection_model(
        spec_dim=int(train_dreams.shape[1]),
        mol_dim=store.dim,
        mol_block_dims=store.block_dims,
        mol_block_names=store.block_names,
        config=checkpoint_model_config(checkpoint),
    ).to(device)
    state = checkpoint.get(args.state_key)
    if state is None:
        raise KeyError(f"Checkpoint lacks {args.state_key}")
    model.load_state_dict(state)
    model.eval()
    projection_dim = int(checkpoint_model_config(checkpoint).get("proj_dim", 1024))

    union_rows = np.unique(
        np.concatenate([train["candidate_rows"].reshape(-1), val["candidate_rows"].reshape(-1)])
    ).astype(np.int32)
    local_map = np.full(store.size, -1, dtype=np.int32)
    local_map[union_rows] = np.arange(len(union_rows), dtype=np.int32)
    train_local = local_map[train["candidate_rows"]]
    val_local = local_map[val["candidate_rows"]]
    molecule_projection = project_molecules(
        union_rows, store, model, device, args.batch_size, projection_dim
    )
    train_spec = project_specs(
        args.train_dreams, model, device, args.batch_size, projection_dim
    )
    val_spec = project_specs(
        args.val_dreams, model, device, args.batch_size, projection_dim
    )

    # Recompute every selected seed-0 logit to bind vector and score caches.
    temperature = float(model.temperature().detach().cpu())
    max_diff = 0.0
    for split, spec, local in ((train, train_spec, train_local), (val, val_spec, val_local)):
        for start in range(0, len(spec), args.batch_size):
            end = min(start + args.batch_size, len(spec))
            s = spec[start:end].float()
            m = molecule_projection[torch.from_numpy(local[start:end]).long()].float()
            logits = torch.einsum("bd,bkd->bk", s, m).numpy() / temperature
            expected = split["stage1_scores"][start:end, :, 0]
            max_diff = max(max_diff, float(np.max(np.abs(logits - expected))))
    if max_diff > 2e-3:
        raise ValueError(f"Projection/logit parity failed: max abs diff {max_diff}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    molecule_path = args.output_dir / "candidate_projection.pt"
    torch.save({
        "global_rows": torch.from_numpy(union_rows.astype(np.int64)),
        "embeddings": molecule_projection,
    }, molecule_path)
    split_paths: dict[str, Path] = {}
    for name, split, spec, local in (
        ("train", train, train_spec, train_local),
        ("val", val, val_spec, val_local),
    ):
        path = args.output_dir / f"{name}_top5.pt"
        torch.save({
            "identifiers": split["identifiers"],
            "connectivities": split["connectivities"],
            "spec_embeddings": spec,
            "candidate_local_indices": torch.from_numpy(local.astype(np.int32)),
            "candidate_global_rows": torch.from_numpy(split["candidate_rows"].astype(np.int32)),
            "stage1_scores": torch.from_numpy(split["stage1_scores"]),
            "ensemble_scores": torch.from_numpy(split["ensemble_scores"]),
            "labels": torch.from_numpy(split["labels"].astype(np.int64)),
            "baseline_ranks": torch.from_numpy(split["baseline_ranks"].astype(np.int32)),
            "audit": split["audit"],
        }, path)
        split_paths[name] = path
    report = {
        "format": "glmr_official_datasafe_top5_projection_cache_v1",
        "official_test_read": False,
        "top_k": args.top_k,
        "selection": "frozen unweighted mean-logit ensemble",
        "train_negative_exclusion": "held-out connectivity denylist; GT retained",
        "projection_checkpoint": {"path": str(args.checkpoint), "sha256": sha256_file(args.checkpoint)},
        "state_key": args.state_key,
        "molecule_blocks": store.block_names,
        "fingerprint_cache": {"path": str(args.fingerprint_cache), "sha256": sha256_file(args.fingerprint_cache)},
        "train_scores": {"path": str(args.train_scores), "sha256": sha256_file(args.train_scores)},
        "val_scores": {"path": str(args.val_scores), "sha256": sha256_file(args.val_scores)},
        "train_dreams": {"path": str(args.train_dreams), "sha256": sha256_file(args.train_dreams)},
        "val_dreams": {"path": str(args.val_dreams), "sha256": sha256_file(args.val_dreams)},
        "negative_exclusion_manifest": {"path": str(args.negative_exclusion_manifest), "sha256": sha256_file(args.negative_exclusion_manifest)},
        "negative_exclusion_indices": {"path": str(exclusion_path), "sha256": sha256_file(exclusion_path)},
        "train_audit": train["audit"],
        "val_audit": val["audit"],
        "unique_top5_candidate_rows": len(union_rows),
        "projection_dim": int(molecule_projection.shape[1]),
        "projection_score_max_abs_diff": max_diff,
        "artifacts": {
            "candidate_projection": {"path": str(molecule_path), "sha256": sha256_file(molecule_path)},
            "train_top5": {"path": str(split_paths["train"]), "sha256": sha256_file(split_paths["train"])},
            "val_top5": {"path": str(split_paths["val"]), "sha256": sha256_file(split_paths["val"])},
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "train_audit": train["audit"],
        "val_audit": val["audit"],
        "unique_top5_candidate_rows": len(union_rows),
        "projection_score_max_abs_diff": max_diff,
    }, indent=2))


if __name__ == "__main__":
    main()
