"""Minimal Run-K + FLARE-style local-interaction validation experiment.

Protocol
--------
* MassSpecGym v1.5 formula-group split, precursor-mass candidates capped at 256.
* Frozen Run-K supplies the original top-50 candidate list and global scores.
* Ground truth is never inserted into top-50. Queries outside top-50 keep their
  original Run-K rank in unconditional metrics and do not enter reranker CE.
* Frozen DreaMS fragment tokens (the synthetic precursor token is excluded)
  interact with trainable candidate atom-node embeddings through bidirectional
  MaxSim.
* ``raw`` is the protocol-matched scientific arm. ``subformula`` uses the
  query's GT molecular formula/adduct and is explicitly a formula-known upper
  bound, not a protocol-matched replacement.

The implementation intentionally uses only RDKit and PyTorch. It does not
install or depend on DGL/dgllife and does not unfreeze DreaMS or Run-K.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from train_spec_encoder.train_run_k_pairwise_reranker import (  # noqa: E402
    CompositeProjectedStore,
    build_queries,
    json_safe,
    load_run_k,
)
from tools.reranking.train_msalign_cross_attention_reranker import rank_metrics  # noqa: E402
from tools.stage1.train_msalign_projection_smoke import load_candidate_sidecar, load_jsonl  # noqa: E402


DEFAULT_RUN_K = REPO_ROOT / "outputs/msalign_molecule_ablation/run_k_r3_allfp_blockl2_20k_cap256_5ep.pt"
DEFAULT_CHEMBERTA = (
    REPO_ROOT
    / "outputs/msalign_molecule_ablation/chemberta_mtr_morgan_formula_group_20k_val_test_mass_cap256.pt"
)
DEFAULT_SIDECAR = (
    REPO_ROOT
    / "outputs/msalign_mtr_probe/chemberta_mtr_formula_group_20k_val_test_mass_cap256.candidates.jsonl"
)
DEFAULT_PROJECTIONS = (
    REPO_ROOT / "outputs/dreams_chemistry_adapter/run_k_projected_candidates_train20k_val_fp16.pt"
)
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/run_k_flare_local_reranker"
DEFAULT_E1_CHECKPOINT = DEFAULT_OUT_DIR / "phase1_seed20260806/raw_seed20260806.pt"

ELEMENTS = ("H", "C", "O", "N", "P", "S", "Cl", "F", "Br", "I", "B", "As", "Si", "Se")
ELEMENT_TO_INDEX = {element: index for index, element in enumerate(ELEMENTS)}
FORMULA_PATTERN = re.compile(r"([A-Z][a-z]*)([0-9]*)")
FORMULA_NORM = np.asarray(
    [102.0, 59.0, 25.0, 13.0, 3.0, 6.0, 6.0, 17.0, 4.0, 4.0, 1.0, 1.0, 5.0, 2.0],
    dtype=np.float32,
)
ELECTRON_MASS = 0.00054858


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-metadata",
        type=Path,
        default=Path("outputs/msalign_formula_group_split_v15/train_sample20k_seed20260623.metadata.jsonl"),
    )
    parser.add_argument(
        "--val-metadata", type=Path, default=Path("outputs/msalign_formula_group_split_v15/val.metadata.jsonl")
    )
    parser.add_argument(
        "--train-tokens",
        type=Path,
        default=Path("outputs/dreams_chemistry_adapter/train_sample20k_seed20260623.tokens.pt"),
    )
    parser.add_argument(
        "--val-tokens", type=Path, default=Path("outputs/dreams_chemistry_adapter/val.tokens.pt")
    )
    parser.add_argument("--run-k-checkpoint", type=Path, default=DEFAULT_RUN_K)
    parser.add_argument("--chemberta-cache", type=Path, default=DEFAULT_CHEMBERTA)
    parser.add_argument("--candidate-sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--projection-cache", type=Path, default=DEFAULT_PROJECTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--topk-cache", type=Path, default=None)
    parser.add_argument("--graph-cache", type=Path, default=None)
    parser.add_argument("--subformula-cache", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["raw", "subformula", "both", "candidate_subformula", "cache-only"],
        default="both",
    )
    parser.add_argument("--candidate-pool-limit", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--stage1-batch-size", type=int, default=128)
    parser.add_argument("--local-dim", type=int, default=256)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--local-temperature", type=float, default=0.05)
    parser.add_argument("--local-aux-weight", type=float, default=0.1)
    parser.add_argument("--alpha-grid", default="0,0.25,0.5,1,2")
    parser.add_argument("--damage-target", type=float, default=0.08)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--subformula-ppm", type=float, default=20.0)
    parser.add_argument("--subformula-top-n", type=int, default=4)
    parser.add_argument("--candidate-subformula-cache", type=Path, default=None)
    parser.add_argument("--candidate-subformula-cache-only", action="store_true")
    parser.add_argument("--candidate-e1-checkpoint", type=Path, default=DEFAULT_E1_CHECKPOINT)
    parser.add_argument("--candidate-batch-size", type=int, default=8)
    parser.add_argument("--candidate-eval-batch-size", type=int, default=16)
    parser.add_argument("--beta-grid", default="0,0.25,0.5,1,2")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_float_grid(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed or any(item < 0 for item in parsed):
        raise ValueError("alpha-grid must contain non-negative values")
    if 0.0 not in parsed:
        raise ValueError("alpha-grid must include 0 for the exact Run-K fallback")
    return parsed


def load_token_cache(path: Path) -> dict[str, Any]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    required = {"tokens", "mask", "identifiers", "identifier_to_index", "metadata"}
    if not required.issubset(payload):
        raise ValueError(f"Malformed DreaMS token cache {path}: missing {sorted(required - set(payload))}")
    if payload["tokens"].ndim != 3 or payload["tokens"].shape[1:] != (101, 1024):
        raise ValueError(f"Unexpected token shape in {path}: {tuple(payload['tokens'].shape)}")
    if tuple(payload["mask"].shape) != tuple(payload["tokens"].shape[:2]):
        raise ValueError(f"Token/mask mismatch in {path}")
    return payload


def subset_records_and_tokens(
    records: list[dict[str, Any]], token_cache: dict[str, Any], limit: int, seed: int
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    cache_ids = [str(value) for value in token_cache["identifiers"]]
    record_ids = [str(row.get("identifier")) for row in records]
    if cache_ids != record_ids:
        raise ValueError("Metadata and DreaMS token-cache identifier order differ")
    tokens = token_cache["tokens"]
    masks = token_cache["mask"].bool()
    if limit <= 0 or limit >= len(records):
        return records, tokens, masks
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), limit))
    tensor_indices = torch.tensor(indices, dtype=torch.long)
    return [records[index] for index in indices], tokens[tensor_indices].clone(), masks[tensor_indices].clone()


@dataclass
class TopKSplit:
    identifiers: list[str]
    baseline_ranks: torch.Tensor
    order_indices: torch.Tensor
    spec_indices: torch.Tensor
    candidate_indices: torch.Tensor
    candidate_mask: torch.Tensor
    stage1_scores: torch.Tensor
    gt_indices: torch.Tensor
    stage1_ranks: torch.Tensor
    stats: dict[str, int]

    def to_payload(self) -> dict[str, Any]:
        return {
            "identifiers": self.identifiers,
            "baseline_ranks": self.baseline_ranks,
            "order_indices": self.order_indices,
            "spec_indices": self.spec_indices,
            "candidate_indices": self.candidate_indices,
            "candidate_mask": self.candidate_mask,
            "stage1_scores": self.stage1_scores,
            "gt_indices": self.gt_indices,
            "stage1_ranks": self.stage1_ranks,
            "stats": self.stats,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TopKSplit":
        return cls(**payload)


@torch.inference_mode()
def build_topk_split(
    query_set: Any,
    tokens: torch.Tensor,
    projected: CompositeProjectedStore,
    projection: nn.Module,
    top_k: int,
    device: torch.device,
    batch_size: int,
) -> TopKSplit:
    identifiers = [query.identifier for query in query_set.queries]
    baseline_ranks = torch.zeros(len(query_set.queries), dtype=torch.int16)
    examples: list[dict[str, Any]] = []
    gt_not_in_top_k = 0
    temperature = projection.temperature()
    for start in tqdm(range(0, len(query_set.queries), batch_size), desc="Building frozen Run-K top50"):
        batch = query_set.queries[start : start + batch_size]
        spec = tokens[[query.spec_index for query in batch], 0].float().to(device)
        spec_proj = projection.project_spec(spec)
        max_candidates = max(len(query.candidate_indices) for query in batch)
        padded = [
            query.candidate_indices + [query.candidate_indices[0]] * (max_candidates - len(query.candidate_indices))
            for query in batch
        ]
        valid_mask = torch.tensor(
            [[True] * len(query.candidate_indices) + [False] * (max_candidates - len(query.candidate_indices)) for query in batch],
            dtype=torch.bool,
            device=device,
        )
        index_tensor = torch.tensor(padded, dtype=torch.long)
        mol_proj = projected.gather(index_tensor).to(device)
        scores = torch.einsum("bd,bcd->bc", spec_proj, mol_proj) / temperature
        scores = scores.masked_fill(~valid_mask, float("-inf"))
        orders = torch.argsort(scores, dim=1, descending=True)
        for row_index, query in enumerate(batch):
            valid_count = len(query.candidate_indices)
            gt_pool_index = query.candidate_indices.index(query.gt_global_index)
            valid_order = orders[row_index, :valid_count]
            rank = int((valid_order == gt_pool_index).nonzero(as_tuple=False)[0].item()) + 1
            baseline_ranks[query.order_index] = rank
            selected = valid_order[: min(top_k, valid_count)].detach().cpu().long()
            selected_list = selected.tolist()
            if gt_pool_index not in selected_list:
                gt_not_in_top_k += 1
                continue
            selected_device = selected.to(device)
            candidates = torch.tensor(query.candidate_indices, dtype=torch.long)[selected]
            examples.append(
                {
                    "order_index": query.order_index,
                    "spec_index": query.spec_index,
                    "candidates": candidates,
                    "scores": scores[row_index, selected_device].detach().cpu().float(),
                    "gt_index": selected_list.index(gt_pool_index),
                    "stage1_rank": rank,
                }
            )
    count = len(examples)
    candidate_indices = torch.full((count, top_k), -1, dtype=torch.int32)
    candidate_mask = torch.zeros((count, top_k), dtype=torch.bool)
    stage1_scores = torch.zeros((count, top_k), dtype=torch.float32)
    for row_index, example in enumerate(examples):
        width = len(example["candidates"])
        candidate_indices[row_index, :width] = example["candidates"].to(torch.int32)
        candidate_mask[row_index, :width] = True
        stage1_scores[row_index, :width] = example["scores"]
    stats = dict(query_set.stats) | {"gt_not_in_top_k": gt_not_in_top_k, "rerankable_usable": count}
    return TopKSplit(
        identifiers=identifiers,
        baseline_ranks=baseline_ranks,
        order_indices=torch.tensor([example["order_index"] for example in examples], dtype=torch.int32),
        spec_indices=torch.tensor([example["spec_index"] for example in examples], dtype=torch.int32),
        candidate_indices=candidate_indices,
        candidate_mask=candidate_mask,
        stage1_scores=stage1_scores,
        gt_indices=torch.tensor([example["gt_index"] for example in examples], dtype=torch.int16),
        stage1_ranks=torch.tensor([example["stage1_rank"] for example in examples], dtype=torch.int16),
        stats=stats,
    )


def topk_cache_metadata(args: argparse.Namespace, run_k_hash: str) -> dict[str, Any]:
    return {
        "schema": 1,
        "protocol": "MassSpecGym v1.5 formula-group; mass cap256; frozen Run-K; original top50; no GT insertion",
        "run_k_sha256": run_k_hash,
        "candidate_sidecar_sha256": sha256_file(args.candidate_sidecar),
        "train_metadata_sha256": sha256_file(args.train_metadata),
        "val_metadata_sha256": sha256_file(args.val_metadata),
        "candidate_pool_limit": args.candidate_pool_limit,
        "top_k": args.top_k,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "seed": args.seed,
    }


def topk_cache_metadata_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Allow full-split top-K caches to be reused across training seeds.

    Seed only changes which records are sampled when a split limit is active;
    it has no effect on a full train/validation top-K construction.
    """
    actual = dict(actual)
    expected = dict(expected)
    if int(expected.get("train_limit", 0)) == 0 and int(expected.get("val_limit", 0)) == 0:
        actual.pop("seed", None)
        expected.pop("seed", None)
    return actual == expected


def load_or_build_topk(
    args: argparse.Namespace,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, TopKSplit], dict[str, Any], list[str]]:
    topk_path = args.topk_cache or args.out_dir / "run_k_train20k_val_top50.pt"
    projection, run_k_info = load_run_k(args.run_k_checkpoint, device)
    expected_metadata = topk_cache_metadata(args, run_k_info["sha256"])
    chemberta = torch.load(str(args.chemberta_cache), map_location="cpu", weights_only=False, mmap=True)
    smiles = [str(value) for value in chemberta["smiles"]]
    if topk_path.exists():
        payload = torch.load(str(topk_path), map_location="cpu", weights_only=False, mmap=True)
        if not topk_cache_metadata_matches(payload.get("metadata") or {}, expected_metadata):
            raise ValueError(f"Stale/incompatible top-K cache: {topk_path}")
        return (
            {name: TopKSplit.from_payload(payload[name]) for name in ("train", "val")},
            run_k_info,
            smiles,
        )

    sidecar = load_candidate_sidecar(args.candidate_sidecar)
    smiles_to_index = {str(key): int(value) for key, value in chemberta["smiles_to_index"].items()}
    train_queries = build_queries(
        train_records, sidecar, smiles_to_index, len(train_tokens), args.candidate_pool_limit
    )
    val_queries = build_queries(val_records, sidecar, smiles_to_index, len(val_tokens), args.candidate_pool_limit)
    if train_queries.stats["gt_missing_from_candidate_pool"] or val_queries.stats["gt_missing_from_candidate_pool"]:
        raise ValueError("GT absent from original cap256 pool; refusing to insert it")
    projected_payload = torch.load(str(args.projection_cache), map_location="cpu", weights_only=False, mmap=True)
    cache_hash = str((projected_payload.get("metadata") or {}).get("run_k_checkpoint_sha256") or "")
    if cache_hash and cache_hash != run_k_info["sha256"]:
        raise ValueError("Projected candidate cache belongs to another Run-K checkpoint")
    projected = CompositeProjectedStore([projected_payload], len(smiles), preload=False)
    required = sorted(
        {
            index
            for query_set in (train_queries, val_queries)
            for query in query_set.queries
            for index in query.candidate_indices
        }
    )
    missing = projected.missing(required)
    if missing:
        raise ValueError(f"Primary train/val projection cache misses {len(missing)} candidates")
    splits = {
        "train": build_topk_split(
            train_queries, train_tokens, projected, projection, args.top_k, device, args.stage1_batch_size
        ),
        "val": build_topk_split(
            val_queries, val_tokens, projected, projection, args.top_k, device, args.stage1_batch_size
        ),
    }
    topk_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"metadata": expected_metadata, **{name: split.to_payload() for name, split in splits.items()}},
        topk_path,
    )
    return splits, run_k_info, smiles


def atom_features(atom: Chem.Atom) -> list[int]:
    hybridization_map = {
        Chem.HybridizationType.S: 1,
        Chem.HybridizationType.SP: 2,
        Chem.HybridizationType.SP2: 3,
        Chem.HybridizationType.SP3: 4,
        Chem.HybridizationType.SP3D: 5,
        Chem.HybridizationType.SP3D2: 6,
    }
    chiral_map = {
        Chem.ChiralType.CHI_UNSPECIFIED: 0,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
        Chem.ChiralType.CHI_OTHER: 3,
    }
    return [
        min(int(atom.GetAtomicNum()), 118),
        min(int(atom.GetTotalDegree()), 10),
        min(max(int(atom.GetFormalCharge()), -5), 5) + 5,
        hybridization_map.get(atom.GetHybridization(), 0),
        min(int(atom.GetTotalNumHs()), 8),
        chiral_map.get(atom.GetChiralTag(), 3),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
    ]


def bond_features(bond: Chem.Bond) -> list[int]:
    bond_type_map = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    stereo_map = {
        Chem.BondStereo.STEREONONE: 0,
        Chem.BondStereo.STEREOANY: 1,
        Chem.BondStereo.STEREOZ: 2,
        Chem.BondStereo.STEREOE: 3,
        Chem.BondStereo.STEREOCIS: 4,
        Chem.BondStereo.STEREOTRANS: 5,
    }
    return [
        bond_type_map.get(bond.GetBondType(), 0),
        stereo_map.get(bond.GetStereo(), 1),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]


def unique_topk_candidates(splits: dict[str, TopKSplit]) -> torch.Tensor:
    values = torch.cat(
        [split.candidate_indices[split.candidate_mask].long() for split in splits.values()], dim=0
    )
    return torch.unique(values, sorted=True)


def build_graph_cache(path: Path, global_indices: torch.Tensor, smiles: list[str]) -> dict[str, Any]:
    node_counts = torch.empty(len(global_indices), dtype=torch.int32)
    edge_counts = torch.empty(len(global_indices), dtype=torch.int32)
    for row, global_index in enumerate(tqdm(global_indices.tolist(), desc="Auditing candidate graph sizes")):
        mol = Chem.MolFromSmiles(smiles[global_index])
        if mol is None or mol.GetNumAtoms() < 1:
            raise ValueError(f"RDKit failed for candidate index {global_index}: {smiles[global_index]!r}")
        node_counts[row] = mol.GetNumAtoms()
        edge_counts[row] = 2 * mol.GetNumBonds()
    node_offsets = torch.zeros(len(global_indices) + 1, dtype=torch.int64)
    edge_offsets = torch.zeros(len(global_indices) + 1, dtype=torch.int64)
    node_offsets[1:] = torch.cumsum(node_counts.long(), dim=0)
    edge_offsets[1:] = torch.cumsum(edge_counts.long(), dim=0)
    nodes = torch.empty((int(node_offsets[-1]), 8), dtype=torch.int16)
    edge_index = torch.empty((2, int(edge_offsets[-1])), dtype=torch.int32)
    edges = torch.empty((int(edge_offsets[-1]), 4), dtype=torch.int8)
    for row, global_index in enumerate(tqdm(global_indices.tolist(), desc="Packing candidate graphs")):
        mol = Chem.MolFromSmiles(smiles[global_index])
        assert mol is not None
        node_start = int(node_offsets[row])
        edge_start = int(edge_offsets[row])
        for atom_index, atom in enumerate(mol.GetAtoms()):
            nodes[node_start + atom_index] = torch.tensor(atom_features(atom), dtype=torch.int16)
        cursor = edge_start
        for bond in mol.GetBonds():
            features = torch.tensor(bond_features(bond), dtype=torch.int8)
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index[:, cursor] = torch.tensor([begin, end], dtype=torch.int32)
            edges[cursor] = features
            cursor += 1
            edge_index[:, cursor] = torch.tensor([end, begin], dtype=torch.int32)
            edges[cursor] = features
            cursor += 1
    payload = {
        "metadata": {
            "schema": 1,
            "representation": "packed_rdkit_atom_bond_categorical_graphs",
            "graphs": len(global_indices),
            "nodes": int(node_offsets[-1]),
            "directed_edges": int(edge_offsets[-1]),
            "global_indices_sha256": hashlib.sha256(global_indices.numpy().tobytes()).hexdigest(),
        },
        "global_indices": global_indices.to(torch.int32),
        "node_offsets": node_offsets,
        "edge_offsets": edge_offsets,
        "node_features": nodes,
        "edge_index": edge_index,
        "edge_features": edges,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_or_build_graph_cache(
    path: Path, global_indices: torch.Tensor, smiles: list[str]
) -> "PackedGraphStore":
    expected_digest = hashlib.sha256(global_indices.numpy().tobytes()).hexdigest()
    if path.exists():
        payload = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
        if (payload.get("metadata") or {}).get("global_indices_sha256") != expected_digest:
            raise ValueError(f"Stale/incompatible graph cache: {path}")
    else:
        payload = build_graph_cache(path, global_indices, smiles)
    return PackedGraphStore(payload, len(smiles))


class PackedGraphStore:
    def __init__(self, payload: dict[str, Any], total_candidates: int) -> None:
        self.global_indices = payload["global_indices"].long()
        self.node_offsets = payload["node_offsets"]
        self.edge_offsets = payload["edge_offsets"]
        self.node_features = payload["node_features"]
        self.edge_index = payload["edge_index"]
        self.edge_features = payload["edge_features"]
        self.global_to_row = torch.full((total_candidates,), -1, dtype=torch.int32)
        self.global_to_row[self.global_indices] = torch.arange(len(self.global_indices), dtype=torch.int32)

    def gather(self, candidate_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """Vectorized gather of unique graphs, returning one packed graph batch."""
        unique_global, inverse = torch.unique(candidate_indices.long().cpu(), sorted=True, return_inverse=True)
        graph_rows = self.global_to_row[unique_global].long()
        if bool((graph_rows < 0).any()):
            raise KeyError(f"Graph cache misses candidate indices {unique_global[graph_rows < 0][:5].tolist()}")
        node_starts = self.node_offsets[graph_rows]
        node_counts = self.node_offsets[graph_rows + 1] - node_starts
        node_bases = torch.cumsum(node_counts, dim=0) - node_counts
        total_nodes = int(node_counts.sum())
        node_batch = torch.repeat_interleave(torch.arange(len(graph_rows)), node_counts)
        node_local = torch.arange(total_nodes) - torch.repeat_interleave(node_bases, node_counts)
        node_source = torch.repeat_interleave(node_starts, node_counts) + node_local

        edge_starts = self.edge_offsets[graph_rows]
        edge_counts = self.edge_offsets[graph_rows + 1] - edge_starts
        total_edges = int(edge_counts.sum())
        edge_batch = torch.repeat_interleave(torch.arange(len(graph_rows)), edge_counts)
        edge_bases = torch.cumsum(edge_counts, dim=0) - edge_counts
        edge_local = torch.arange(total_edges) - torch.repeat_interleave(edge_bases, edge_counts)
        edge_source = torch.repeat_interleave(edge_starts, edge_counts) + edge_local
        gathered_edges = self.edge_index[:, edge_source].long()
        if total_edges:
            gathered_edges = gathered_edges + node_bases[edge_batch].unsqueeze(0)
        return {
            "node_features": self.node_features[node_source].long(),
            "edge_index": gathered_edges,
            "edge_features": self.edge_features[edge_source].long(),
            "node_batch": node_batch.long(),
            "node_local": node_local.long(),
            "node_counts": node_counts.long(),
            "inverse": inverse.long(),
        }


def parse_formula(formula: str) -> np.ndarray:
    # MassSpecGym contains a very small number of intrinsically charged
    # molecules whose formula ends in a bare charge sign (for example
    # ``C12H10OP+``). The sign is charge metadata, not an element; subformula
    # enumeration only needs elemental composition. Keep this deliberately
    # narrow so more complex formula syntax still fails loudly.
    composition = formula[:-1] if formula.endswith(("+", "-")) else formula
    counts = np.zeros(len(ELEMENTS), dtype=np.int16)
    matched = ""
    for element, raw_count in FORMULA_PATTERN.findall(composition):
        matched += f"{element}{raw_count}"
        if element not in ELEMENT_TO_INDEX:
            raise ValueError(f"Unsupported element {element!r} in formula {formula!r}")
        counts[ELEMENT_TO_INDEX[element]] += int(raw_count or 1)
    if matched != composition:
        raise ValueError(f"Could not fully parse formula {formula!r}")
    return counts


def selected_dreams_peaks(mzs: list[float], intensities: list[float], n_highest: int = 100) -> np.ndarray:
    mz_array = np.asarray(mzs, dtype=np.float64)
    intensity_array = np.asarray(intensities, dtype=np.float64)
    if len(mz_array) != len(intensity_array):
        raise ValueError("m/z and intensity lengths differ")
    if len(mz_array) > n_highest:
        selected = np.sort(np.argsort(intensity_array)[-n_highest:])
        mz_array = mz_array[selected]
    return mz_array


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()] if value else []


def load_peak_lists(tsv_path: Path, identifiers: set[str]) -> dict[str, tuple[list[float], list[float]]]:
    output: dict[str, tuple[list[float], list[float]]] = {}
    with tsv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            identifier = str(row.get("identifier"))
            if identifier in identifiers:
                output[identifier] = (parse_float_list(row.get("mzs", "")), parse_float_list(row.get("intensities", "")))
    missing = identifiers - set(output)
    if missing:
        raise ValueError(f"v1.5 TSV misses {len(missing)} requested spectra; first={next(iter(missing))}")
    return output


def ion_shift(adduct: str, periodic_table: Any) -> float:
    if adduct == "[M+H]+":
        return float(periodic_table.GetMostCommonIsotopeMass(1)) - ELECTRON_MASS
    if adduct == "[M+Na]+":
        return float(periodic_table.GetMostCommonIsotopeMass(11)) - ELECTRON_MASS
    raise ValueError(f"Unsupported formula-known adduct: {adduct!r}")


def assign_formula_group(
    parent_counts: np.ndarray,
    observed_mzs: np.ndarray,
    adduct: str,
    ppm_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign nearest parent-subformula composition to each observed peak.

    All subset masses are represented by a mixed-radix flat index. Only the
    selected indices are decoded to 14-element formula vectors, avoiding a
    potentially very large ``num_subsets x 14`` matrix.
    """
    periodic_table = Chem.GetPeriodicTable()
    masses = np.asarray(
        [periodic_table.GetMostCommonIsotopeMass(periodic_table.GetAtomicNumber(element)) for element in ELEMENTS],
        dtype=np.float64,
    )
    radices = parent_counts.astype(np.int64) + 1
    total = int(np.prod(radices, dtype=np.int64))
    flat = np.arange(total, dtype=np.int64)
    working = flat.copy()
    subset_masses = np.zeros(total, dtype=np.float64)
    for radix, mass in zip(radices, masses):
        digit = working % radix
        subset_masses += digit * mass
        working //= radix
    subset_masses += ion_shift(adduct, periodic_table)
    order = np.argsort(subset_masses)
    sorted_masses = subset_masses[order]
    positions = np.searchsorted(sorted_masses, observed_mzs)
    right = np.clip(positions, 0, total - 1)
    left = np.clip(positions - 1, 0, total - 1)
    choose_right = np.abs(sorted_masses[right] - observed_mzs) < np.abs(sorted_masses[left] - observed_mzs)
    nearest_pos = np.where(choose_right, right, left)
    nearest_mass = sorted_masses[nearest_pos]
    nearest_flat = order[nearest_pos]
    ppm = np.abs(nearest_mass - observed_mzs) / np.maximum(observed_mzs, 1e-12) * 1e6
    assigned = ppm < ppm_threshold
    features = np.zeros((len(observed_mzs), len(ELEMENTS)), dtype=np.float32)
    selected_flat = nearest_flat[assigned].copy()
    for element_index, radix in enumerate(radices):
        features[assigned, element_index] = selected_flat % radix
        selected_flat //= radix
    return features, assigned


def build_subformula_cache(
    path: Path,
    records_by_split: dict[str, list[dict[str, Any]]],
    masks_by_split: dict[str, torch.Tensor],
    token_metadata: dict[str, Any],
    ppm_threshold: float,
) -> dict[str, Any]:
    tsv_path = Path(str(token_metadata["v15_tsv"]))
    all_identifiers = {str(row["identifier"]) for records in records_by_split.values() for row in records}
    peak_lists = load_peak_lists(tsv_path, all_identifiers)
    output: dict[str, torch.Tensor] = {}
    split_stats: dict[str, Any] = {}
    for split_name, records in records_by_split.items():
        features = torch.zeros((len(records), 100, len(ELEMENTS) + 1), dtype=torch.float16)
        grouped: dict[tuple[str, str], list[int]] = {}
        for index, row in enumerate(records):
            grouped.setdefault((str(row["formula"]), str(row["adduct"])), []).append(index)
        assigned_total = 0
        peak_total = 0
        for (formula, adduct), row_indices in tqdm(grouped.items(), desc=f"Assigning {split_name} GT subformulae"):
            parent_counts = parse_formula(formula)
            mz_chunks: list[np.ndarray] = []
            lengths: list[int] = []
            for row_index in row_indices:
                identifier = str(records[row_index]["identifier"])
                mzs, intensities = peak_lists[identifier]
                selected = selected_dreams_peaks(mzs, intensities)
                expected = int(masks_by_split[split_name][row_index, 1:].sum())
                if len(selected) != expected:
                    raise ValueError(
                        f"DreaMS peak/token parity failed for {identifier}: selected={len(selected)} mask={expected}"
                    )
                mz_chunks.append(selected)
                lengths.append(len(selected))
            observed = np.concatenate(mz_chunks) if mz_chunks else np.empty(0, dtype=np.float64)
            formula_features, assigned = assign_formula_group(parent_counts, observed, adduct, ppm_threshold)
            cursor = 0
            for row_index, length in zip(row_indices, lengths):
                normalized = formula_features[cursor : cursor + length] / FORMULA_NORM
                features[row_index, :length, : len(ELEMENTS)] = torch.from_numpy(normalized).half()
                features[row_index, :length, -1] = torch.from_numpy(assigned[cursor : cursor + length]).half()
                cursor += length
            assigned_total += int(assigned.sum())
            peak_total += len(assigned)
        output[split_name] = features
        split_stats[split_name] = {
            "peaks": peak_total,
            "assigned": assigned_total,
            "assignment_rate": assigned_total / max(peak_total, 1),
            "spectra": len(records),
            "unique_formula_adduct": len(grouped),
        }
    payload = {
        "metadata": {
            "schema": 1,
            "label": "GT-formula-known upper bound; query formula/adduct shared across all candidates",
            "elements": list(ELEMENTS),
            "formula_norm": FORMULA_NORM.tolist(),
            "ppm_threshold": ppm_threshold,
            "v15_tsv": str(tsv_path),
            "stats": split_stats,
        },
        **output,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_or_build_subformula_cache(
    path: Path,
    records_by_split: dict[str, list[dict[str, Any]]],
    masks_by_split: dict[str, torch.Tensor],
    token_metadata: dict[str, Any],
    ppm_threshold: float,
) -> dict[str, Any]:
    if path.exists():
        payload = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
        metadata = payload.get("metadata") or {}
        expected_shapes = {
            split: (len(records), 100, len(ELEMENTS) + 1) for split, records in records_by_split.items()
        }
        if float(metadata.get("ppm_threshold", -1)) != ppm_threshold or any(
            tuple(payload[split].shape) != shape for split, shape in expected_shapes.items()
        ):
            raise ValueError(f"Stale/incompatible subformula cache: {path}")
        return payload
    return build_subformula_cache(path, records_by_split, masks_by_split, token_metadata, ppm_threshold)


class AtomFeatureEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(119, 32),
                nn.Embedding(11, 8),
                nn.Embedding(11, 8),
                nn.Embedding(7, 8),
                nn.Embedding(9, 8),
                nn.Embedding(4, 8),
                nn.Embedding(2, 4),
                nn.Embedding(2, 4),
            ]
        )
        self.projection = nn.Sequential(nn.Linear(80, dim), nn.LayerNorm(dim), nn.GELU())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.cat([embedding(features[:, index]) for index, embedding in enumerate(self.embeddings)], dim=-1))


class BondFeatureEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(5, 16), nn.Embedding(6, 8), nn.Embedding(2, 4), nn.Embedding(2, 4)]
        )
        self.projection = nn.Sequential(nn.Linear(32, dim), nn.GELU())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.cat([embedding(features[:, index]) for index, embedding in enumerate(self.embeddings)], dim=-1))


class EdgeAwareGINLayer(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.update = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        aggregate = torch.zeros_like(nodes)
        if edge_index.shape[1]:
            src, dst = edge_index
            messages = self.message(nodes[src] + edges)
            # LayerNorm keeps ``nodes`` in fp32 under autocast while the MLP
            # message path may be bf16. Accumulate in the node dtype.
            aggregate.index_add_(0, dst, messages.to(nodes.dtype))
            degree = torch.bincount(dst, minlength=len(nodes)).to(nodes.dtype).clamp_min(1).sqrt().unsqueeze(-1)
            aggregate = aggregate / degree
        return self.norm(nodes + self.update(nodes + aggregate))


class LocalInteractionModel(nn.Module):
    def __init__(self, dim: int, layers: int, dropout: float, use_subformula: bool) -> None:
        super().__init__()
        self.use_subformula = use_subformula
        self.peak_projection = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, dim)
        )
        self.atom_encoder = AtomFeatureEncoder(dim)
        self.bond_encoder = BondFeatureEncoder(dim)
        self.gnn = nn.ModuleList([EdgeAwareGINLayer(dim, dropout) for _ in range(layers)])
        if use_subformula:
            self.formula_projection = nn.Sequential(
                nn.Linear(len(ELEMENTS) + 1, 128), nn.GELU(), nn.Linear(128, dim)
            )
            self.formula_gate = nn.Parameter(torch.tensor(0.1))
        else:
            self.formula_projection = None
            self.register_parameter("formula_gate", None)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        peak_tokens: torch.Tensor,
        peak_mask: torch.Tensor,
        graphs: dict[str, torch.Tensor],
        candidate_shape: tuple[int, int],
        formula_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        peaks = self.peak_projection(peak_tokens)
        if self.use_subformula:
            if formula_features is None or self.formula_projection is None or self.formula_gate is None:
                raise ValueError("subformula model requires formula features")
            peaks = peaks + self.formula_gate * self.formula_projection(formula_features)
        peaks = F.normalize(peaks, dim=-1, eps=1e-6)

        nodes = self.atom_encoder(graphs["node_features"])
        edges = self.bond_encoder(graphs["edge_features"])
        for layer in self.gnn:
            nodes = layer(nodes, graphs["edge_index"], edges)
        nodes = F.normalize(nodes, dim=-1, eps=1e-6)
        graph_count = len(graphs["node_counts"])
        max_nodes = int(graphs["node_counts"].max())
        padded_nodes = nodes.new_zeros((graph_count, max_nodes, nodes.shape[-1]))
        padded_nodes[graphs["node_batch"], graphs["node_local"]] = nodes
        node_mask = torch.arange(max_nodes, device=nodes.device).unsqueeze(0) < graphs["node_counts"].unsqueeze(1)
        batch_size, candidate_count = candidate_shape
        candidate_nodes = padded_nodes[graphs["inverse"]].reshape(batch_size, candidate_count, max_nodes, -1)
        candidate_node_mask = node_mask[graphs["inverse"]].reshape(batch_size, candidate_count, max_nodes)
        similarity = torch.einsum("bpd,bkad->bkpa", peaks, candidate_nodes)
        valid = peak_mask[:, None, :, None] & candidate_node_mask[:, :, None, :]
        # Cosine similarity is in [-1, 1]. A finite sentinel avoids NaN/Inf
        # gradients from max-reduction over an all-padding slice while being
        # exactly irrelevant to every valid maximum.
        similarity = similarity.masked_fill(~valid, -1e4)
        peak_to_atom = similarity.max(dim=-1).values
        atom_to_peak = similarity.max(dim=-2).values
        peak_to_atom = torch.where(peak_mask[:, None], peak_to_atom, torch.zeros_like(peak_to_atom))
        atom_to_peak = torch.where(candidate_node_mask, atom_to_peak, torch.zeros_like(atom_to_peak))
        peak_score = (peak_to_atom * peak_mask[:, None]).sum(-1) / peak_mask.sum(-1).clamp_min(1)[:, None]
        atom_score = (atom_to_peak * candidate_node_mask).sum(-1) / candidate_node_mask.sum(-1).clamp_min(1)
        return 0.5 * (peak_score + atom_score)


class RerankDataset(Dataset[int]):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


def move_graphs(graphs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in graphs.items()}


def prepare_batch(
    split: TopKSplit,
    indices: torch.Tensor,
    tokens: torch.Tensor,
    masks: torch.Tensor,
    graph_store: PackedGraphStore,
    device: torch.device,
    formula_features: torch.Tensor | None,
) -> dict[str, Any]:
    candidate_mask = split.candidate_mask[indices]
    candidates = split.candidate_indices[indices].long()
    safe_candidates = candidates.clone()
    safe_candidates[~candidate_mask] = candidates[candidate_mask][0]
    graphs = graph_store.gather(safe_candidates.reshape(-1))
    spec_indices = split.spec_indices[indices].long()
    batch = {
        "tokens": tokens[spec_indices, 1:].float().to(device),
        "peak_mask": masks[spec_indices, 1:].to(device),
        "graphs": move_graphs(graphs, device),
        "candidate_mask": candidate_mask.to(device),
        "stage1_scores": split.stage1_scores[indices].to(device),
        "labels": split.gt_indices[indices].long().to(device),
        "candidate_shape": tuple(candidates.shape),
    }
    if formula_features is not None:
        batch["formula_features"] = formula_features[spec_indices].float().to(device)
    else:
        batch["formula_features"] = None
    return batch


def train_epoch(
    model: LocalInteractionModel,
    split: TopKSplit,
    tokens: torch.Tensor,
    masks: torch.Tensor,
    graph_store: PackedGraphStore,
    formula_features: torch.Tensor | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    local_temperature: float,
    local_aux_weight: float,
) -> dict[str, Any]:
    model.train()
    generator = torch.Generator().manual_seed(torch.initial_seed())
    loader = DataLoader(
        RerankDataset(len(split.order_indices)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    total_loss = 0.0
    total_examples = 0
    first_gradients = None
    for batch_number, raw_indices in enumerate(tqdm(loader, desc="Training local MaxSim")):
        indices = raw_indices.long()
        batch = prepare_batch(split, indices, tokens, masks, graph_store, device, formula_features)
        # Keep this path in fp32. bf16 autocast was unstable for the sparse
        # hard-MaxSim/L2-normalization gradients on some full-data batches.
        local_scores = model(
            batch["tokens"],
            batch["peak_mask"],
            batch["graphs"],
            batch["candidate_shape"],
            batch["formula_features"],
        )
        raw_local_logits = local_scores / local_temperature
        candidate_mask = batch["candidate_mask"]
        local_logits = raw_local_logits.masked_fill(~candidate_mask, float("-inf"))
        # Multiply the calibration scalar before inserting -inf padding.
        # Otherwise d/d(scale) encounters the undefined 0 * -inf product on
        # masked candidates even though their cross-entropy gradient is zero.
        fused_logits = (batch["stage1_scores"] + model.residual_scale * raw_local_logits).masked_fill(
            ~candidate_mask, float("-inf")
        )
        combined_loss = F.cross_entropy(fused_logits, batch["labels"])
        local_loss = F.cross_entropy(local_logits, batch["labels"])
        loss = combined_loss + local_aux_weight * local_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite local-interaction loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        bad_gradients = []
        for parameter_name, parameter in model.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                bad_gradients.append(
                    {
                        "name": parameter_name,
                        "nan": int(torch.isnan(parameter.grad).sum().item()),
                        "inf": int(torch.isinf(parameter.grad).sum().item()),
                    }
                )
        if bad_gradients:
            raise FloatingPointError(
                "Non-finite local-interaction gradients: "
                + json.dumps(
                    {
                        "batch": batch_number,
                        "bad": bad_gradients,
                        "local_score_min": float(local_scores.detach().min().cpu()),
                        "local_score_max": float(local_scores.detach().max().cpu()),
                        "residual_scale": float(model.residual_scale.detach().cpu()),
                    }
                )
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if batch_number == 0:
            grad_norms = [parameter.grad.detach().norm() for parameter in model.parameters() if parameter.grad is not None]
            first_gradients = {
                "parameters_with_grad": len(grad_norms),
                "total_grad_norm": float(torch.stack(grad_norms).norm().cpu()) if grad_norms else 0.0,
                "residual_scale_grad": float(model.residual_scale.grad.detach().cpu()) if model.residual_scale.grad is not None else None,
            }
        optimizer.step()
        # The local term is a positive-evidence residual by definition. This
        # projected scalar keeps epoch 0 exactly at Run-K while preventing a
        # learned sign flip from turning MaxSim into an anti-similarity score.
        with torch.no_grad():
            model.residual_scale.clamp_(min=0.0)
        total_loss += float(loss.detach().cpu()) * len(indices)
        total_examples += len(indices)
    return {
        "loss": total_loss / max(total_examples, 1),
        "examples": total_examples,
        "residual_scale": float(model.residual_scale.detach().cpu()),
        "first_batch_gradients": first_gradients,
    }


@torch.no_grad()
def collect_local_residuals(
    model: LocalInteractionModel,
    split: TopKSplit,
    tokens: torch.Tensor,
    masks: torch.Tensor,
    graph_store: PackedGraphStore,
    formula_features: torch.Tensor | None,
    device: torch.device,
    batch_size: int,
    local_temperature: float,
) -> torch.Tensor:
    model.eval()
    output = torch.zeros_like(split.stage1_scores)
    loader = DataLoader(RerankDataset(len(split.order_indices)), batch_size=batch_size, shuffle=False)
    for raw_indices in tqdm(loader, desc="Scoring local MaxSim"):
        indices = raw_indices.long()
        batch = prepare_batch(split, indices, tokens, masks, graph_store, device, formula_features)
        local = model(
            batch["tokens"],
            batch["peak_mask"],
            batch["graphs"],
            batch["candidate_shape"],
            batch["formula_features"],
        )
        output[indices] = (model.residual_scale * local / local_temperature).detach().cpu()
    return output


def evaluate_policy(split: TopKSplit, residuals: torch.Tensor, alpha: float) -> dict[str, Any]:
    ranks = split.baseline_ranks.clone().long()
    rank1_total = 0
    rank1_damaged = 0
    for row in range(len(split.order_indices)):
        valid = split.candidate_mask[row]
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        scores = split.stage1_scores[row, valid] + alpha * residuals[row, valid]
        order = torch.argsort(scores, descending=True)
        ranked_indices = valid_indices[order]
        matches = (ranked_indices == int(split.gt_indices[row])).nonzero(as_tuple=False)
        if not len(matches):
            raise ValueError(f"GT candidate is masked out in rerankable row {row}")
        rank = int(matches[0].item()) + 1
        ranks[int(split.order_indices[row])] = rank
        if int(split.stage1_ranks[row]) == 1:
            rank1_total += 1
            rank1_damaged += int(rank > 1)
    return {
        **rank_metrics(ranks.tolist()),
        "alpha": alpha,
        "rank1_total": rank1_total,
        "rank1_damage_rate": rank1_damaged / max(rank1_total, 1),
        "ranks": ranks,
    }


def select_policy(results: dict[float, dict[str, Any]], damage_target: float) -> float:
    eligible = [(alpha, result) for alpha, result in results.items() if result["rank1_damage_rate"] <= damage_target]
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


def contributions(ranks: torch.Tensor) -> np.ndarray:
    values = ranks.cpu().numpy().astype(np.float64)
    return np.stack(
        [(values <= 1) * 100.0, (values <= 5) * 100.0, (values <= 20) * 100.0, 100.0 / values],
        axis=1,
    )


def paired_bootstrap(
    baseline_ranks: torch.Tensor, selected_ranks: torch.Tensor, samples: int, seed: int
) -> dict[str, Any]:
    delta = contributions(selected_ranks) - contributions(baseline_ranks)
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    for start in range(0, samples, 128):
        count = min(128, samples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        chunks.append(delta[indices].mean(axis=1))
    boot = np.concatenate(chunks)
    names = ("recall@1", "recall@5", "recall@20", "mrr")
    return {
        name: {
            "delta_percentage_points": float(delta[:, column].mean()),
            "ci95": [float(np.quantile(boot[:, column], 0.025)), float(np.quantile(boot[:, column], 0.975))],
            "probability_positive": float((boot[:, column] > 0).mean()),
        }
        for column, name in enumerate(names)
    }


def screening_gate(bootstrap: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    r1 = bootstrap["recall@1"]
    mrr = bootstrap["mrr"]
    triggering = [
        name
        for name, result in (("recall@1", r1), ("mrr", mrr))
        if result["delta_percentage_points"] >= 0.3 and result["probability_positive"] >= 0.8
    ]
    other_ok = False
    if "recall@1" in triggering:
        other_ok = mrr["delta_percentage_points"] >= -0.1
    if "mrr" in triggering:
        other_ok = other_ok or r1["delta_percentage_points"] >= -0.1
    checks = {
        "r1_or_mrr_gain_and_bootstrap": bool(triggering),
        "other_primary_not_below_minus_0_1pp": other_ok,
        "recall20_not_below_minus_0_2pp": bootstrap["recall@20"]["delta_percentage_points"] >= -0.2,
        "rank1_damage_at_most_8pct": selected["rank1_damage_rate"] <= 0.08,
    }
    return {"passed": all(checks.values()), "triggering_metrics": triggering, "checks": checks}


def run_arm(
    name: str,
    args: argparse.Namespace,
    splits: dict[str, TopKSplit],
    tokens_by_split: dict[str, torch.Tensor],
    masks_by_split: dict[str, torch.Tensor],
    graph_store: PackedGraphStore,
    formula_by_split: dict[str, torch.Tensor] | None,
    device: torch.device,
    common: dict[str, Any],
) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    use_subformula = name == "subformula"
    model = LocalInteractionModel(args.local_dim, args.gnn_layers, args.dropout, use_subformula).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    alpha_grid = parse_float_grid(args.alpha_grid)
    baseline = rank_metrics(splits["val"].baseline_ranks.tolist())

    history: list[dict[str, Any]] = []
    epoch0_residuals = torch.zeros_like(splits["val"].stage1_scores)
    epoch0_results = {alpha: evaluate_policy(splits["val"], epoch0_residuals, alpha) for alpha in alpha_grid}
    if any(result["ranks"].tolist() != splits["val"].baseline_ranks.tolist() for result in epoch0_results.values()):
        raise RuntimeError("Epoch-0 zero-residual parity failed")
    best_epoch = 0
    best_alpha = 0.0
    best_mrr = float(baseline["mrr"])
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history.append({"epoch": 0, "selected_alpha": 0.0, "val_selected": baseline, "exact_run_k_fallback": True})

    for epoch in range(1, args.epochs + 1):
        train_report = train_epoch(
            model,
            splits["train"],
            tokens_by_split["train"],
            masks_by_split["train"],
            graph_store,
            formula_by_split["train"] if formula_by_split is not None else None,
            optimizer,
            device,
            args.batch_size,
            args.num_workers,
            args.local_temperature,
            args.local_aux_weight,
        )
        residuals = collect_local_residuals(
            model,
            splits["val"],
            tokens_by_split["val"],
            masks_by_split["val"],
            graph_store,
            formula_by_split["val"] if formula_by_split is not None else None,
            device,
            args.eval_batch_size,
            args.local_temperature,
        )
        results = {alpha: evaluate_policy(splits["val"], residuals, alpha) for alpha in alpha_grid}
        selected_alpha = select_policy(results, args.damage_target)
        selected = results[selected_alpha]
        entry = {
            "epoch": epoch,
            "train": train_report,
            "selected_alpha": selected_alpha,
            "val_selected": {key: value for key, value in selected.items() if key != "ranks"},
        }
        history.append(entry)
        print(json.dumps({"arm": name, **entry}, sort_keys=True), flush=True)
        if float(selected["mrr"]) > best_mrr:
            best_epoch = epoch
            best_alpha = selected_alpha
            best_mrr = float(selected["mrr"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    residuals = collect_local_residuals(
        model,
        splits["val"],
        tokens_by_split["val"],
        masks_by_split["val"],
        graph_store,
        formula_by_split["val"] if formula_by_split is not None else None,
        device,
        args.eval_batch_size,
        args.local_temperature,
    )
    final_grid = {alpha: evaluate_policy(splits["val"], residuals, alpha) for alpha in alpha_grid}
    best_alpha = select_policy(final_grid, args.damage_target)
    selected = final_grid[best_alpha]
    bootstrap = paired_bootstrap(
        splits["val"].baseline_ranks, selected["ranks"], args.bootstrap_samples, args.seed + (1 if name == "raw" else 2)
    )
    gate = screening_gate(bootstrap, selected)
    output = {
        **common,
        "arm": name,
        "scientific_role": "protocol-matched primary experiment" if name == "raw" else "GT-formula-known upper bound",
        "history": history,
        "best_epoch": best_epoch,
        "selected_alpha": best_alpha,
        "val_run_k": baseline,
        "val_selected": {key: value for key, value in selected.items() if key != "ranks"},
        "paired_bootstrap": bootstrap,
        "screening_gate": gate,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_seed{args.seed}"
    torch.save(
        {
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "selected_alpha": best_alpha,
            "config": common,
            "arm": name,
        },
        args.out_dir / f"{stem}.pt",
    )
    (args.out_dir / f"{stem}.json").write_text(json.dumps(json_safe(output), indent=2, sort_keys=True) + "\n")
    torch.save(
        {
            "identifiers": splits["val"].identifiers,
            "run_k_ranks": splits["val"].baseline_ranks,
            "selected_ranks": selected["ranks"],
        },
        args.out_dir / f"{stem}.val_ranks.pt",
    )
    return output


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.top_k < 2 or args.local_temperature <= 0:
        raise ValueError("epochs/top-k/local-temperature are invalid")
    if args.subformula_top_n < 1 or args.candidate_batch_size < 1 or args.candidate_eval_batch_size < 1:
        raise ValueError("candidate-subformula top-n/batch sizes must be positive")
    if args.candidate_subformula_cache_only and args.mode != "candidate_subformula":
        raise ValueError("--candidate-subformula-cache-only requires --mode candidate_subformula")
    started = time.time()
    args.out_dir = args.out_dir.resolve()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    train_records = load_jsonl(args.train_metadata)
    val_records = load_jsonl(args.val_metadata)
    train_cache = load_token_cache(args.train_tokens)
    val_cache = load_token_cache(args.val_tokens)
    train_records, train_tokens, train_masks = subset_records_and_tokens(
        train_records, train_cache, args.train_limit, args.seed
    )
    val_records, val_tokens, val_masks = subset_records_and_tokens(
        val_records, val_cache, args.val_limit, args.seed + 1
    )
    tokens_by_split = {"train": train_tokens, "val": val_tokens}
    masks_by_split = {"train": train_masks, "val": val_masks}
    records_by_split = {"train": train_records, "val": val_records}

    splits, run_k_info, smiles = load_or_build_topk(
        args, train_records, val_records, train_tokens, val_tokens, device
    )
    baseline = {name: rank_metrics(split.baseline_ranks.tolist()) for name, split in splits.items()}
    expected_val = run_k_info.get("best_val") or {}
    parity = {
        metric: float(baseline["val"][metric]) - float(expected_val[metric])
        for metric in ("recall@1", "recall@5", "recall@20", "mrr")
        if metric in expected_val
    }
    if args.val_limit == 0 and parity and max(abs(value) for value in parity.values()) > 0.02:
        raise RuntimeError(f"Frozen Run-K validation parity failed: {parity}")
    graph_indices = unique_topk_candidates(splits)
    graph_path = args.graph_cache or args.out_dir / "run_k_train20k_val_top50_graphs.pt"
    graph_store = load_or_build_graph_cache(graph_path, graph_indices, smiles)

    formula_payload = None
    if args.mode in {"subformula", "both"}:
        formula_path = args.subformula_cache or args.out_dir / "gt_formula_known_train20k_val_peak_features.pt"
        formula_payload = load_or_build_subformula_cache(
            formula_path,
            records_by_split,
            masks_by_split,
            train_cache["metadata"],
            args.subformula_ppm,
        )
    common = {
        "protocol": "MassSpecGym v1.5 formula-group; mass cap256; frozen DreaMS and Run-K; original top50; no GT insertion; precursor token excluded",
        "args": json_safe(vars(args)),
        "run_k": run_k_info,
        "baseline": baseline,
        "baseline_parity_delta_pp": parity,
        "topk_stats": {name: split.stats for name, split in splits.items()},
        "graph_cache": {
            "path": str(graph_path),
            "graphs": len(graph_store.global_indices),
            "nodes": len(graph_store.node_features),
            "directed_edges": graph_store.edge_index.shape[1],
        },
        "subformula": formula_payload["metadata"] if formula_payload is not None else None,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "phase1_common.json").write_text(json.dumps(json_safe(common), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"baseline": baseline, "parity": parity, "topk_stats": common["topk_stats"]}, indent=2), flush=True)
    if args.mode == "cache-only":
        return

    if args.mode == "candidate_subformula":
        from tools.reranking.train_run_k_flare_candidate_subformula import run_candidate_subformula

        output = run_candidate_subformula(
            args,
            splits,
            tokens_by_split,
            masks_by_split,
            records_by_split,
            train_cache["metadata"],
            graph_store,
            smiles,
            device,
            common,
        )
        if output is not None:
            print(json.dumps(json_safe(output), indent=2, sort_keys=True), flush=True)
        return

    outputs: dict[str, Any] = {}
    if args.mode in {"raw", "both"}:
        outputs["raw"] = run_arm(
            "raw", args, splits, tokens_by_split, masks_by_split, graph_store, None, device, common
        )
    if args.mode in {"subformula", "both"}:
        assert formula_payload is not None
        outputs["subformula"] = run_arm(
            "subformula",
            args,
            splits,
            tokens_by_split,
            masks_by_split,
            graph_store,
            {"train": formula_payload["train"], "val": formula_payload["val"]},
            device,
            common,
        )
    summary_path = args.out_dir / f"phase1_seed{args.seed}_summary.json"
    previous_summary: dict[str, Any] = {}
    if summary_path.exists():
        previous_summary = json.loads(summary_path.read_text())
    arm_summaries = dict(previous_summary.get("arms", {}))
    arm_summaries.update({
            name: {
                "scientific_role": output["scientific_role"],
                "best_epoch": output["best_epoch"],
                "selected_alpha": output["selected_alpha"],
                "val_run_k": output["val_run_k"],
                "val_selected": output["val_selected"],
                "paired_bootstrap": output["paired_bootstrap"],
                "screening_gate": output["screening_gate"],
            }
            for name, output in outputs.items()
        })
    summary = {
        "elapsed_seconds": float(previous_summary.get("elapsed_seconds", 0.0)) + time.time() - started,
        "arms": arm_summaries,
        "decision_rule": (
            "Advance raw if its gate passes. Only if raw fails and GT-formula upper bound passes, "
            "advance the upper bound as a mechanism diagnostic. Otherwise stop."
        ),
    }
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
