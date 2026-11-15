#!/usr/bin/env python3
"""Export exact validation candidate logits for one Stage 1 arm and its seed ensemble."""

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

from glmr.candidate_indices import CandidateIndexStore  # noqa: E402
from glmr.smiles import profile_smiles  # noqa: E402
from tools.stage1.train_msalign_projection_smoke import (  # noqa: E402
    MoleculeEmbeddingStore,
    ProjectionDataset,
    build_projection_model,
    checkpoint_model_config,
    choose_device,
    load_dreams,
    load_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dreams", type=Path, required=True)
    parser.add_argument("--candidate-index-manifest", type=Path, required=True)
    parser.add_argument("--fingerprint-cache", type=Path, default=None)
    parser.add_argument("--chemberta-cache", type=Path, default=None)
    parser.add_argument("--no-preload-sparse-features", dest="preload_sparse_features", action="store_false", default=True, help="Read identical fingerprint rows on demand; avoid preloading the full cache")
    parser.add_argument("--state-key", default="model_state_dict")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--tie-seeds", default="20260926,20260927,20260928")
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tie_key(identifier: str, candidate_row: int, seed: int) -> int:
    payload = f"{seed}\0{identifier}\0{candidate_row}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def tie_safe_rank_for_target(
    identifier: str,
    candidate_rows: np.ndarray,
    scores: np.ndarray,
    target_offset: int,
    seed: int,
) -> int:
    target_score = scores[target_offset]
    greater = int(np.count_nonzero(scores > target_score))
    target_key = tie_key(identifier, int(candidate_rows[target_offset]), seed)
    tied_before = 0
    for offset in np.flatnonzero(scores == target_score):
        if int(offset) == target_offset:
            continue
        if tie_key(identifier, int(candidate_rows[int(offset)]), seed) < target_key:
            tied_before += 1
    return 1 + greater + tied_before


def tie_safe_ranks(
    identifiers: list[str],
    indptr: np.ndarray,
    candidate_rows: np.ndarray,
    scores: np.ndarray,
    seed: int,
) -> np.ndarray:
    ranks = np.empty(len(identifiers), dtype=np.int32)
    for query, identifier in enumerate(identifiers):
        start = int(indptr[query])
        end = int(indptr[query + 1])
        row_scores = scores[start:end]
        ranks[query] = tie_safe_rank_for_target(
            identifier,
            candidate_rows[start:end],
            row_scores,
            0,
            seed,
        )
    return ranks


def shuffled_tie_safe_ranks(
    identifiers: list[str],
    indptr: np.ndarray,
    candidate_rows: np.ndarray,
    scores: np.ndarray,
    tie_seed: int,
    shuffle_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(shuffle_seed)
    ranks = np.empty(len(identifiers), dtype=np.int32)
    for query, identifier in enumerate(identifiers):
        start = int(indptr[query])
        end = int(indptr[query + 1])
        rows = candidate_rows[start:end]
        row_scores = scores[start:end]
        permutation = rng.permutation(len(rows))
        target_offset = int(np.flatnonzero(permutation == 0)[0])
        ranks[query] = tie_safe_rank_for_target(
            identifier,
            rows[permutation],
            row_scores[permutation],
            target_offset,
            tie_seed,
        )
    return ranks


def metric_summary(ranks: np.ndarray) -> dict[str, float | int]:
    values = ranks.astype(np.float64)
    return {
        "queries": len(ranks),
        "recall@1": round(float(np.mean(values <= 1) * 100), 6),
        "recall@5": round(float(np.mean(values <= 5) * 100), 6),
        "recall@20": round(float(np.mean(values <= 20) * 100), 6),
        "mrr": round(float(np.mean(1.0 / values) * 100), 6),
        "mean_rank": round(float(np.mean(values)), 6),
    }


def resolve_checkpoint_blocks(stored: dict[str, Any]) -> list[str] | None:
    raw = stored.get("mol_blocks")
    if raw is None:
        return None
    return [str(value) for value in raw]


def main() -> None:
    args = parse_args()
    tie_seeds = [int(value) for value in args.tie_seeds.split(",") if value.strip()]
    if len(tie_seeds) != 3:
        raise ValueError("Exactly three tie seeds are required")
    device = choose_device(args.device)
    records = load_jsonl(args.metadata)
    dreams = load_dreams(args.dreams)
    candidate_store = CandidateIndexStore(args.candidate_index_manifest)
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False) for path in args.checkpoint
    ]
    first_args = dict(payloads[0].get("args") or {})
    chemberta = (
        torch.load(args.chemberta_cache, map_location="cpu", weights_only=False, mmap=True)
        if args.chemberta_cache is not None
        else None
    )
    blocks = resolve_checkpoint_blocks(first_args)
    if not args.preload_sparse_features:
        from tools.reranking.train_official_mpfir_oof_retriever import memory_guard
        memory_guard()
    mol_store = MoleculeEmbeddingStore(
        chemberta,
        str(first_args.get("mol_pooling", "mean")),
        args.fingerprint_cache,
        additional_fingerprint_caches=[],
        preload_sparse_features=args.preload_sparse_features,
        mol_blocks=blocks,
    )
    if mol_store.size != candidate_store.molecule_rows:
        raise ValueError("Molecule features and candidate index have different row counts")
    dataset = ProjectionDataset(
        records,
        dreams,
        {},
        chemberta["smiles_to_index"] if chemberta is not None else {},
        int(first_args.get("candidate_pool_limit", 256)),
        candidate_index_store=candidate_store,
    )
    if len(dataset) != len(records):
        raise ValueError(f"Validation query drift: {len(dataset)} != {len(records)}")

    models = []
    for checkpoint_path, payload in zip(args.checkpoint, payloads, strict=True):
        stored = dict(payload.get("args") or {})
        if resolve_checkpoint_blocks(stored) != blocks:
            raise ValueError("All ensemble checkpoints must use identical ordered molecule blocks")
        model = build_projection_model(
            spec_dim=int(dreams.shape[1]),
            mol_dim=mol_store.dim,
            mol_block_dims=mol_store.block_dims,
            mol_block_names=mol_store.block_names,
            config=checkpoint_model_config(payload),
        ).to(device)
        state = payload.get(args.state_key)
        if state is None:
            raise KeyError(f"{checkpoint_path} lacks {args.state_key}")
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    candidate_counts = np.asarray(
        [len(dataset[index]["candidate_indices"]) for index in range(len(dataset))],
        dtype=np.int32,
    )
    indptr = np.empty(len(dataset) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(candidate_counts, out=indptr[1:])
    total = int(indptr[-1])
    candidate_rows = np.empty(total, dtype=np.int32)
    all_scores = np.empty((len(models), total), dtype=np.float32)

    cursor = 0
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(dataset), args.eval_batch_size),
            desc="Exporting candidate logits",
        ):
            if not args.preload_sparse_features:
                memory_guard()
            items = [
                dataset[index]
                for index in range(start, min(start + args.eval_batch_size, len(dataset)))
            ]
            specs = torch.stack([item["spec"] for item in items]).to(device)
            rows = [item["candidate_indices"] for item in items]
            max_len = max(len(values) for values in rows)
            padded = [values + [values[0]] * (max_len - len(values)) for values in rows]
            index_tensor = torch.tensor(padded, dtype=torch.long)
            unique, inverse = torch.unique(index_tensor.reshape(-1), sorted=True, return_inverse=True)
            unique, inverse = mol_store.canonicalize_unique_indices(unique, inverse)
            molecule_values = mol_store.gather(unique).to(device)
            batch_scores = []
            for model in models:
                spec_projection = model.project_spec(specs)
                molecule_projection = model.project_mol(molecule_values)
                projected = molecule_projection[inverse.to(device)].reshape(
                    len(items), max_len, -1
                )
                logits = (
                    torch.einsum("bd,bcd->bc", spec_projection, projected)
                    / model.temperature()
                )
                batch_scores.append(logits.detach().cpu().numpy().astype(np.float32))
            for local, values in enumerate(rows):
                count = len(values)
                candidate_rows[cursor : cursor + count] = np.asarray(values, dtype=np.int32)
                for model_index, scores in enumerate(batch_scores):
                    all_scores[model_index, cursor : cursor + count] = scores[local, :count]
                cursor += count
    if cursor != total:
        raise RuntimeError(f"Score export cursor mismatch: {cursor} != {total}")

    identifiers = [str(item["identifier"]) for item in dataset.items]
    record_by_identifier = {str(record["identifier"]): record for record in records}
    connectivities = []
    for identifier in identifiers:
        record = record_by_identifier[identifier]
        profile = profile_smiles(
            record.get("canonical_isomeric_smiles") or record.get("raw_smiles")
        )
        if not profile.valid or not profile.inchi_key_first14:
            raise ValueError(f"Invalid validation GT identity for {identifier}")
        connectivities.append(profile.inchi_key_first14)
    ensemble_scores = all_scores.mean(axis=0, dtype=np.float64).astype(np.float32)

    rank_payload: dict[str, Any] = {}
    score_sets = [
        (f"seed_{index}", all_scores[index]) for index in range(len(args.checkpoint))
    ] + [("ensemble_mean_logits", ensemble_scores)]
    rank_arrays: dict[str, np.ndarray] = {}
    for label, scores in score_sets:
        per_tie_seed = []
        for seed in tie_seeds:
            ranks = tie_safe_ranks(identifiers, indptr, candidate_rows, scores, seed)
            per_tie_seed.append({"tie_seed": seed, "metrics": metric_summary(ranks)})
        primary_ranks = tie_safe_ranks(
            identifiers, indptr, candidate_rows, scores, tie_seeds[0]
        )
        rank_arrays[label] = primary_ranks
        shuffle_parity = {}
        for shuffle_seed in tie_seeds:
            shuffled = shuffled_tie_safe_ranks(
                identifiers,
                indptr,
                candidate_rows,
                scores,
                tie_seeds[0],
                shuffle_seed,
            )
            shuffle_parity[str(shuffle_seed)] = bool(np.array_equal(primary_ranks, shuffled))
        if not all(shuffle_parity.values()):
            raise ValueError(f"Candidate-order shuffle parity failed for {label}")
        rank_payload[label] = {
            "tie_safe": per_tie_seed,
            "candidate_order_shuffle_parity": shuffle_parity,
            "exact_tie_queries": int(
                sum(
                    np.count_nonzero(
                        scores[int(indptr[q]) : int(indptr[q + 1])]
                        == scores[int(indptr[q])]
                    )
                    > 1
                    for q in range(len(identifiers))
                )
            ),
        }

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_npz,
        identifiers=np.asarray(identifiers),
        connectivities=np.asarray(connectivities),
        indptr=indptr,
        candidate_counts=candidate_counts,
        candidate_rows=candidate_rows,
        scores=all_scores,
        ensemble_scores=ensemble_scores,
        **{f"ranks_{key}": value for key, value in rank_arrays.items()},
    )
    summary = {
        "format": "glmr_stage1_candidate_scores_v1",
        "metadata": str(args.metadata),
        "metadata_sha256": sha256_file(args.metadata),
        "dreams": str(args.dreams),
        "dreams_sha256": sha256_file(args.dreams),
        "candidate_index_manifest": str(args.candidate_index_manifest),
        "candidate_index_manifest_sha256": sha256_file(args.candidate_index_manifest),
        "fingerprint_cache": str(args.fingerprint_cache) if args.fingerprint_cache else None,
        "chemberta_cache": str(args.chemberta_cache) if args.chemberta_cache else None,
        "checkpoints": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in args.checkpoint
        ],
        "state_key": args.state_key,
        "molecule_blocks": mol_store.block_names,
        "preload_sparse_features": args.preload_sparse_features,
        "query_count": len(identifiers),
        "candidate_occurrences": total,
        "candidate_count_min": int(candidate_counts.min()),
        "candidate_count_max": int(candidate_counts.max()),
        "dataset": dataset.stats,
        "tie_seeds": tie_seeds,
        "rank_metrics": rank_payload,
        "npz": str(args.out_npz),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
