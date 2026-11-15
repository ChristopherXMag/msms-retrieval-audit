"""Train a residual chemistry adapter on frozen DreaMS token states.

This entry point implements the matched Stage A/B experiment:

Stage A:
  frozen DreaMS tokens + frozen Run-K projection + residual token adapter
  + global candidate CE

Stage B:
  Stage A + masked multi-positive MAGMa peak-fragment InfoNCE

The Run-K molecule representation and both projection heads stay frozen.  The
only trainable module is a residual transformer over cached DreaMS tokens, so
the MAGMa ablation changes one loss term rather than both modalities at once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.spectrum.train_msalign_dreams_token_adapter import (  # noqa: E402
    load_token_cache,
    make_padded_index_tensors,
)
from tools.stage1.train_msalign_projection_smoke import (  # noqa: E402
    MoleculeEmbeddingStore,
    ProjectionModel,
    load_candidate_sidecar,
    load_jsonl,
    sample_candidate_indices,
)


DEFAULT_RUN_K = REPO_ROOT / "outputs/msalign_molecule_ablation/run_k_r3_allfp_blockl2_20k_cap256_5ep.pt"
DEFAULT_CHEMBERTA = (
    REPO_ROOT / "outputs/msalign_molecule_ablation/chemberta_mtr_morgan_formula_group_20k_val_test_mass_cap256.pt"
)
DEFAULT_FP = REPO_ROOT / "outputs/msalign_molecule_ablation/sparse_fingerprints_formula_group_20k_val_test_cap256/manifest.json"
DEFAULT_R3 = (
    REPO_ROOT
    / "outputs/msalign_molecule_ablation/sparse_count_morgan_r3_formula_group_20k_val_test_cap256/manifest.json"
)
DEFAULT_SIDECAR = (
    REPO_ROOT / "outputs/msalign_mtr_probe/chemberta_mtr_formula_group_20k_val_test_mass_cap256.candidates.jsonl"
)
DEFAULT_MAGMA = (
    REPO_ROOT
    / "outputs/magma_peak_fragment_alignment/formula_group_train20k_full20k_seed20260710/labels.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched Stage A/B residual chemistry adapter over frozen DreaMS token states."
    )
    parser.add_argument(
        "--train-metadata",
        type=Path,
        default=Path("outputs/msalign_formula_group_split_v15/train_sample20k_seed20260623.metadata.jsonl"),
    )
    parser.add_argument(
        "--val-metadata",
        type=Path,
        default=Path("outputs/msalign_formula_group_split_v15/val.metadata.jsonl"),
    )
    parser.add_argument("--train-token-cache", type=Path, required=True)
    parser.add_argument("--val-token-cache", type=Path, required=True)
    parser.add_argument("--magma-labels", type=Path, default=DEFAULT_MAGMA)
    parser.add_argument("--fragment-cache", type=Path, required=True)
    parser.add_argument("--run-k-checkpoint", type=Path, default=DEFAULT_RUN_K)
    parser.add_argument("--chemberta-cache", type=Path, default=DEFAULT_CHEMBERTA)
    parser.add_argument("--fingerprint-cache", type=Path, default=DEFAULT_FP)
    parser.add_argument("--additional-fingerprint-cache", type=Path, action="append", default=[DEFAULT_R3])
    parser.add_argument("--candidate-sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument(
        "--candidate-projection-cache",
        type=Path,
        default=Path("outputs/dreams_chemistry_adapter/run_k_projected_candidates_train20k_val.pt"),
    )
    parser.add_argument("--candidate-projection-build-batch-size", type=int, default=2048)
    parser.add_argument("--candidate-projection-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--preload-sparse-features",
        action="store_true",
        help="Sequentially preload sparse fingerprint arrays into RAM before building a large projection cache.",
    )
    parser.add_argument("--candidate-cache-parity-queries", type=int, default=0)
    parser.add_argument("--candidate-projection-cache-only", action="store_true")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--rank-output", type=Path, default=None)

    parser.add_argument("--candidate-pool-limit", type=int, default=256)
    parser.add_argument("--candidate-negatives", type=int, default=127)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-train-batches", type=int, default=0)

    parser.add_argument("--adapter-layers", type=int, default=1)
    parser.add_argument("--adapter-heads", type=int, default=8)
    parser.add_argument("--adapter-ffn-dim", type=int, default=2048)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    parser.add_argument("--adapter-gate-init", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument(
        "--magma-weight",
        type=float,
        default=0.0,
        help="0 is Stage A; a positive value enables Stage B while preserving all other settings.",
    )
    parser.add_argument("--magma-temperature", type=float, default=0.07)
    parser.add_argument("--magma-holdout-fraction", type=float, default=0.1)
    parser.add_argument("--magma-intensity-weight", choices=["none", "sqrt"], default="none")
    parser.add_argument("--calibrate-magma-weight", action="store_true")
    parser.add_argument(
        "--magma-weight-candidates",
        type=float,
        nargs="+",
        default=[0.01, 0.03, 0.1],
    )
    parser.add_argument("--target-local-global-grad-ratio", type=float, default=0.5)
    parser.add_argument(
        "--fragment-cache-scope",
        choices=["used", "all_labels"],
        default="all_labels",
        help="Use 'used' only for small smoke runs; formal A/B should cache all label fragments once.",
    )
    parser.add_argument("--fragment-build-batch-size", type=int, default=256)
    parser.add_argument("--fragment-cache-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260715)
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


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def int_indices_sha256(indices: list[int]) -> str:
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def sample_records(records: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(records):
        return records
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), limit))
    return [records[index] for index in indices]


def load_magma_labels(path: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = str(row["identifier"])
            if identifier in labels:
                raise ValueError(f"Duplicate MAGMa identifier: {identifier}")
            labels[identifier] = row
    return labels


def fragment_smiles_from_labels(labels: list[dict[str, Any]]) -> list[str]:
    values: set[str] = set()
    for row in labels:
        for peak in row.get("peaks") or []:
            if not peak.get("assigned"):
                continue
            for match in peak.get("matches") or []:
                smiles = match.get("fragment_smiles")
                if smiles:
                    values.add(str(smiles))
    return sorted(values)


class ChemistryTokenDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[dict[str, Any]],
        token_cache: dict[str, Any],
        sidecar: dict[str, dict[str, Any]],
        smiles_to_index: dict[str, int],
        magma_labels: dict[str, dict[str, Any]],
        candidate_pool_limit: int,
        magma_holdout: set[str],
    ) -> None:
        self.tokens: torch.Tensor = token_cache["tokens"]
        self.mask: torch.Tensor = token_cache["mask"]
        token_index = token_cache["identifier_to_index"]
        self.items: list[dict[str, Any]] = []
        stats = {
            "records_seen": len(records),
            "missing_token_cache": 0,
            "missing_sidecar": 0,
            "missing_gt_embedding": 0,
            "candidate_pool_too_small": 0,
            "with_magma_labels": 0,
            "magma_holdout": 0,
            "usable_records": 0,
        }
        for record in records:
            identifier = str(record.get("identifier"))
            cache_index = token_index.get(identifier)
            if cache_index is None:
                stats["missing_token_cache"] += 1
                continue
            sidecar_row = sidecar.get(identifier)
            if sidecar_row is None:
                stats["missing_sidecar"] += 1
                continue
            gt = sidecar_row.get("normalized_gt_smiles")
            if gt is None or gt not in smiles_to_index:
                stats["missing_gt_embedding"] += 1
                continue
            candidates = list(sidecar_row.get("normalized_candidates") or [])
            if candidate_pool_limit > 0:
                candidates = candidates[:candidate_pool_limit]
            candidates = [str(gt)] + [str(value) for value in candidates if value != gt]
            candidates = [value for value in candidates if value in smiles_to_index]
            if len(candidates) < 2:
                stats["candidate_pool_too_small"] += 1
                continue
            magma = magma_labels.get(identifier)
            if magma is not None:
                stats["with_magma_labels"] += 1
            is_holdout = identifier in magma_holdout
            stats["magma_holdout"] += int(is_holdout)
            self.items.append(
                {
                    "identifier": identifier,
                    "token_index": int(cache_index),
                    "gt": str(gt),
                    "candidates": candidates,
                    "candidate_indices": [smiles_to_index[value] for value in candidates],
                    "magma": magma,
                    "magma_holdout": is_holdout,
                }
            )
        stats["usable_records"] = len(self.items)
        self.stats = stats

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        token_index = int(item["token_index"])
        return {**item, "tokens": self.tokens[token_index], "mask": self.mask[token_index]}


def collate_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return items


class ResidualChemistryAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        layers: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
        gate_init: float,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(dim)
        block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, disable: bool = False) -> torch.Tensor:
        tokens = tokens.float()
        if disable:
            return tokens
        hidden = self.input_norm(tokens)
        delta = self.encoder(hidden, src_key_padding_mask=~mask.bool())
        delta = self.output_norm(delta)
        return tokens + self.gate * delta


class ChemistryAlignmentModel(nn.Module):
    def __init__(self, projection: ProjectionModel, adapter: ResidualChemistryAdapter) -> None:
        super().__init__()
        self.projection = projection
        self.adapter = adapter

    def adapted_tokens(self, tokens: torch.Tensor, mask: torch.Tensor, disable_adapter: bool = False) -> torch.Tensor:
        return self.adapter(tokens, mask, disable=disable_adapter)

    def project_global(
        self, tokens: torch.Tensor, mask: torch.Tensor, disable_adapter: bool = False
    ) -> torch.Tensor:
        adapted = self.adapted_tokens(tokens, mask, disable_adapter=disable_adapter)
        return self.projection.project_spec(adapted[:, 0, :])

    def project_peaks_from_adapted(self, adapted: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        batch_indices = torch.arange(adapted.shape[0], device=adapted.device)
        peaks = adapted[batch_indices, token_indices]
        return self.projection.project_spec(peaks)


class ProjectedMoleculeStore:
    """Memory-mapped frozen Run-K candidate projections keyed by global row."""

    def __init__(self, payload: dict[str, Any], total_rows: int) -> None:
        self.embeddings: torch.Tensor = payload["embeddings"]
        self.global_indices: torch.Tensor = payload["global_indices"].long()
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != self.global_indices.numel():
            raise ValueError("Projected candidate cache embeddings/global_indices mismatch")
        if self.global_indices.numel() and not bool(torch.all(self.global_indices[1:] > self.global_indices[:-1])):
            raise ValueError("Projected candidate cache global_indices must be sorted and unique")
        self.global_to_local = torch.full((total_rows,), -1, dtype=torch.int32)
        self.global_to_local[self.global_indices] = torch.arange(
            self.global_indices.numel(), dtype=torch.int32
        )
        self.metadata = dict(payload.get("metadata") or {})

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        original_shape = tuple(indices.shape)
        flat = indices.reshape(-1).cpu()
        local = self.global_to_local[flat].long()
        if bool((local < 0).any()):
            missing = flat[local < 0][:5].tolist()
            raise KeyError(f"Projected candidate rows are absent from cache: {missing}")
        return self.embeddings[local].float().reshape(*original_shape, self.dim)


def load_run_k_model(path: Path, device: torch.device, adapter_args: argparse.Namespace) -> ChemistryAlignmentModel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    ckpt_args = dict(payload.get("args") or {})
    state = payload.get("best_model_state_dict") or payload.get("model_state_dict")
    if state is None:
        raise ValueError(f"No model state in {path}")
    projection = ProjectionModel(
        spec_dim=1024,
        mol_dim=6311,
        hidden_dim=int(ckpt_args.get("hidden_dim", 2048)),
        proj_dim=int(ckpt_args.get("proj_dim", 1024)),
        dropout=float(ckpt_args.get("dropout", 0.2)),
        temperature=float(ckpt_args.get("temperature", 0.07)),
        head_type=str(ckpt_args.get("head_type", "mlp")),
        mol_block_dims=[2048, 2048, 2048, 167],
        mol_block_normalization="l2",
    )
    projection.load_state_dict(state)
    for parameter in projection.parameters():
        parameter.requires_grad_(False)
    adapter = ResidualChemistryAdapter(
        dim=1024,
        layers=adapter_args.adapter_layers,
        heads=adapter_args.adapter_heads,
        ffn_dim=adapter_args.adapter_ffn_dim,
        dropout=adapter_args.adapter_dropout,
        gate_init=adapter_args.adapter_gate_init,
    )
    return ChemistryAlignmentModel(projection, adapter).to(device)


def required_candidate_indices(datasets: list[ChemistryTokenDataset]) -> list[int]:
    required: set[int] = set()
    for dataset in datasets:
        for item in dataset.items:
            required.update(int(index) for index in item["candidate_indices"])
    return sorted(required)


@torch.inference_mode()
def build_projected_candidate_cache(
    path: Path,
    required_indices: list[int],
    mol_store: MoleculeEmbeddingStore,
    model: ChemistryAlignmentModel,
    device: torch.device,
    batch_size: int,
    run_k_checkpoint: Path,
    dtype: str,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("--candidate-projection-build-batch-size must be positive")
    global_indices = torch.tensor(required_indices, dtype=torch.long)
    output_dtype = torch.float16 if dtype == "float16" else torch.float32
    embeddings = torch.empty((len(required_indices), 1024), dtype=output_dtype)
    model.eval()
    model.projection.eval()
    for start in tqdm(
        range(0, len(required_indices), batch_size), desc="Building frozen Run-K candidate projections"
    ):
        rows = global_indices[start : start + batch_size]
        raw = mol_store.gather(rows).to(device)
        embeddings[start : start + len(rows)] = model.projection.project_mol(raw).cpu().to(output_dtype)
    smiles_sha256 = None
    for metadata in mol_store.fingerprint_cache_metadata:
        candidate_digest = metadata.get("smiles_sha256")
        if candidate_digest:
            smiles_sha256 = str(candidate_digest)
            break
    payload = {
        "global_indices": global_indices,
        "embeddings": embeddings,
        "metadata": {
            "representation": "frozen_run_k_projected_candidates",
            "rows": len(required_indices),
            "total_source_rows": mol_store.size,
            "projection_dim": int(embeddings.shape[1]),
            "dtype": str(embeddings.dtype),
            "required_indices_sha256": int_indices_sha256(required_indices),
            "smiles_sha256": smiles_sha256,
            "run_k_checkpoint": str(run_k_checkpoint.resolve()),
            "run_k_checkpoint_sha256": file_sha256(run_k_checkpoint),
            "feature_blocks": list(mol_store.block_names),
            "block_dims": list(mol_store.block_dims),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_or_build_projected_candidate_cache(
    path: Path,
    required_indices: list[int],
    mol_store: MoleculeEmbeddingStore,
    model: ChemistryAlignmentModel,
    device: torch.device,
    batch_size: int,
    run_k_checkpoint: Path,
    dtype: str,
) -> ProjectedMoleculeStore:
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        metadata = dict(payload.get("metadata") or {})
        expected_checkpoint = file_sha256(run_k_checkpoint)
        if metadata.get("run_k_checkpoint_sha256") != expected_checkpoint:
            raise ValueError(f"Projected candidate cache uses a different Run-K checkpoint: {path}")
        store = ProjectedMoleculeStore(payload, mol_store.size)
        required = torch.tensor(required_indices, dtype=torch.long)
        if bool((store.global_to_local[required] < 0).any()):
            missing = int((store.global_to_local[required] < 0).sum())
            raise ValueError(f"Projected candidate cache misses {missing} required rows: {path}")
        return store
    payload = build_projected_candidate_cache(
        path,
        required_indices,
        mol_store,
        model,
        device,
        batch_size,
        run_k_checkpoint,
        dtype,
    )
    return ProjectedMoleculeStore(payload, mol_store.size)


def gather_projected_molecules(
    model: ChemistryAlignmentModel,
    store: MoleculeEmbeddingStore | ProjectedMoleculeStore,
    indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    values = store.gather(indices).to(device)
    if isinstance(store, ProjectedMoleculeStore):
        return values
    return model.projection.project_mol(values)


def run_k_fingerprint_batch(smiles_values: list[str]) -> tuple[torch.Tensor, list[bool]]:
    generators = {
        "morgan": rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048, includeChirality=True),
        "atom_pair": rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048, includeChirality=True),
        "path": rdFingerprintGenerator.GetRDKitFPGenerator(
            fpSize=2048,
            minPath=1,
            maxPath=7,
            useHs=True,
            branchedPaths=True,
            useBondOrder=True,
            countSimulation=False,
        ),
    }
    features = np.zeros((len(smiles_values), 6311), dtype=np.float32)
    valid: list[bool] = []
    offsets = {"morgan": 0, "atom_pair": 2048, "path": 4096}
    for row, smiles in enumerate(smiles_values):
        mol = Chem.MolFromSmiles(smiles)
        valid.append(mol is not None)
        if mol is None:
            continue
        for name, generator in generators.items():
            values = generator.GetCountFingerprint(mol).GetNonzeroElements()
            offset = offsets[name]
            for column, count in values.items():
                features[row, offset + int(column)] = math.log1p(int(count))
        on_bits = MACCSkeys.GenMACCSKeys(mol).GetOnBits()
        features[row, 6144 + np.asarray(list(on_bits), dtype=np.int64)] = 1.0
    return torch.from_numpy(features), valid


@torch.inference_mode()
def build_fragment_cache(
    path: Path,
    smiles_values: list[str],
    model: ChemistryAlignmentModel,
    device: torch.device,
    batch_size: int,
    source_labels: Path,
    run_k_checkpoint: Path,
) -> dict[str, Any]:
    unique = sorted(set(smiles_values))
    kept_smiles: list[str] = []
    chunks: list[torch.Tensor] = []
    invalid = 0
    model.eval()
    for start in tqdm(range(0, len(unique), batch_size), desc="Building Run-K fragment targets"):
        values = unique[start : start + batch_size]
        features, valid = run_k_fingerprint_batch(values)
        valid_indices = [index for index, flag in enumerate(valid) if flag]
        invalid += len(values) - len(valid_indices)
        if not valid_indices:
            continue
        selected = features[torch.tensor(valid_indices, dtype=torch.long)].to(device)
        projected = model.projection.project_mol(selected).cpu().half()
        chunks.append(projected)
        kept_smiles.extend(values[index] for index in valid_indices)
    embeddings = torch.cat(chunks, dim=0) if chunks else torch.empty((0, 1024), dtype=torch.float16)
    payload = {
        "smiles": kept_smiles,
        "smiles_to_index": {value: index for index, value in enumerate(kept_smiles)},
        "embeddings": embeddings,
        "metadata": {
            "representation": "run_k_projected_fragment_fingerprint",
            "run_k_checkpoint": str(run_k_checkpoint),
            "source_labels": str(source_labels),
            "requested_fragments": len(unique),
            "valid_fragments": len(kept_smiles),
            "invalid_fragments": invalid,
            "feature_blocks": ["count_morgan_r3", "count_atom_pair", "rdkit_path_count", "maccs"],
            "block_dims": [2048, 2048, 2048, 167],
            "block_normalization": "l2",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_or_build_fragment_cache(
    path: Path,
    required_smiles: list[str],
    model: ChemistryAlignmentModel,
    device: torch.device,
    batch_size: int,
    source_labels: Path,
    run_k_checkpoint: Path,
) -> dict[str, Any]:
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        missing = sorted(set(required_smiles) - set(payload["smiles_to_index"]))
        if missing:
            raise ValueError(
                f"Fragment cache {path} misses {len(missing)} required fragments; "
                "use a new path or rebuild with --fragment-cache-scope all_labels"
            )
        return payload
    return build_fragment_cache(
        path,
        required_smiles,
        model,
        device,
        batch_size,
        source_labels,
        run_k_checkpoint,
    )


def make_holdout(labels: dict[str, dict[str, Any]], fraction: float, seed: int) -> set[str]:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("--magma-holdout-fraction must be in [0, 1)")
    identifiers = sorted(labels)
    rng = random.Random(seed)
    rng.shuffle(identifiers)
    count = int(round(len(identifiers) * fraction))
    return set(identifiers[:count])


def magma_peak_rows(
    batch: list[dict[str, Any]],
    fragment_to_index: dict[str, int],
    supervision: str,
) -> tuple[list[tuple[int, int, list[int], float]], dict[str, int]]:
    rows: list[tuple[int, int, list[int], float]] = []
    stats = {"labeled_queries": 0, "assigned_peaks": 0, "missing_fragment_targets": 0}
    for batch_index, item in enumerate(batch):
        magma = item.get("magma")
        if magma is None:
            continue
        is_holdout = bool(item.get("magma_holdout"))
        if supervision == "train" and is_holdout:
            continue
        if supervision == "holdout" and not is_holdout:
            continue
        stats["labeled_queries"] += 1
        token_count = int(item["mask"].sum())
        for peak in magma.get("peaks") or []:
            if not peak.get("assigned"):
                continue
            token_index = int(peak["token_index"])
            if token_index <= 0 or token_index >= token_count:
                continue
            positives = sorted(
                {
                    fragment_to_index[str(match["fragment_smiles"])]
                    for match in peak.get("matches") or []
                    if str(match.get("fragment_smiles")) in fragment_to_index
                }
            )
            if not positives:
                stats["missing_fragment_targets"] += 1
                continue
            intensity = float(peak.get("relative_intensity", peak.get("intensity", 1.0)))
            rows.append((batch_index, token_index, positives, intensity))
            stats["assigned_peaks"] += 1
    return rows, stats


def magma_loss(
    model: ChemistryAlignmentModel,
    adapted: torch.Tensor,
    batch: list[dict[str, Any]],
    fragment_embeddings: torch.Tensor,
    fragment_to_index: dict[str, int],
    temperature: float,
    intensity_weight: str,
    supervision: str,
) -> tuple[torch.Tensor | None, dict[str, float | int], list[dict[str, Any]]]:
    rows, counts = magma_peak_rows(batch, fragment_to_index, supervision)
    if not rows:
        return None, counts, []
    unique_fragments = sorted({index for _, _, positives, _ in rows for index in positives})
    if len(unique_fragments) < 2:
        return None, counts | {"unique_fragments": len(unique_fragments)}, []
    local_index = {global_index: index for index, global_index in enumerate(unique_fragments)}
    batch_indices = torch.tensor([row[0] for row in rows], dtype=torch.long, device=adapted.device)
    token_indices = torch.tensor([row[1] for row in rows], dtype=torch.long, device=adapted.device)
    peak_states = adapted[batch_indices, token_indices]
    peak_proj = model.projection.project_spec(peak_states)
    fragment_proj = fragment_embeddings[torch.tensor(unique_fragments, dtype=torch.long)].to(
        device=adapted.device, dtype=torch.float32
    )
    logits = torch.matmul(peak_proj, fragment_proj.T) / float(temperature)
    positive_mask = torch.zeros_like(logits, dtype=torch.bool)
    for row_index, (_, _, positives, _) in enumerate(rows):
        positive_mask[row_index, [local_index[index] for index in positives]] = True
    positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
    losses = torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)
    if intensity_weight == "sqrt":
        weights = torch.tensor([math.sqrt(max(row[3], 0.0)) for row in rows], device=losses.device)
        loss = (losses * weights).sum() / weights.sum().clamp_min(1e-8)
    else:
        loss = losses.mean()

    order = torch.argsort(logits.detach(), dim=1, descending=True)
    ordered_positive = positive_mask.gather(1, order)
    best_ranks = ordered_positive.float().argmax(dim=1) + 1
    top1 = float((best_ranks == 1).float().mean().cpu())
    mrr = float((1.0 / best_ranks.float()).mean().cpu())
    metrics: dict[str, float | int] = counts | {
        "peaks": len(rows),
        "unique_fragments": len(unique_fragments),
        "top1": top1,
        "mrr": mrr,
        "mean_best_positive_rank": float(best_ranks.float().mean().cpu()),
    }
    rank_records = []
    losses_cpu = losses.detach().float().cpu()
    ranks_cpu = best_ranks.detach().cpu()
    for row_index, (batch_index, token_index, positives, intensity) in enumerate(rows):
        rank_records.append(
            {
                "identifier": str(batch[batch_index]["identifier"]),
                "token_index": int(token_index),
                "rank": int(ranks_cpu[row_index]),
                "loss": float(losses_cpu[row_index]),
                "positive_fragments": len(positives),
                "comparison_fragments": len(unique_fragments),
                "intensity": float(intensity),
            }
        )
    return loss, metrics, rank_records


def candidate_batch_indices(
    batch: list[dict[str, Any]], candidate_negatives: int, rng: random.Random
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [
        sample_candidate_indices(item, {}, candidate_negatives, rng, "random", 0.0)
        for item in batch
    ]
    masks = [[True] * len(row) for row in rows]
    return make_padded_index_tensors(rows, masks)


def train_one_epoch(
    model: ChemistryAlignmentModel,
    loader: DataLoader,
    mol_store: MoleculeEmbeddingStore | ProjectedMoleculeStore,
    fragment_embeddings: torch.Tensor,
    fragment_to_index: dict[str, int],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, Any]:
    model.train()
    model.projection.eval()
    rng = random.Random(args.seed + epoch)
    totals = {"loss": 0.0, "global_loss": 0.0, "magma_loss": 0.0, "items": 0, "magma_peaks": 0}
    first_grad: dict[str, float] | None = None
    for batch_index, batch in enumerate(tqdm(loader, desc=f"Training chemistry adapter epoch {epoch}"), start=1):
        tokens = torch.stack([item["tokens"] for item in batch]).to(device)
        masks = torch.stack([item["mask"] for item in batch]).to(device)
        candidate_indices, candidate_mask = candidate_batch_indices(batch, args.candidate_negatives, rng)
        flat_unique, inverse = torch.unique(candidate_indices.reshape(-1), sorted=True, return_inverse=True)
        with torch.no_grad():
            mol_proj_unique = gather_projected_molecules(model, mol_store, flat_unique, device)
        mol_proj = mol_proj_unique[inverse.to(device)].reshape(len(batch), candidate_indices.shape[1], -1)
        adapted = model.adapted_tokens(tokens, masks)
        spec_proj = model.projection.project_spec(adapted[:, 0, :])
        logits = torch.einsum("bd,bcd->bc", spec_proj, mol_proj) / model.projection.temperature()
        logits = logits.masked_fill(~candidate_mask.to(device), float("-inf"))
        global_loss = F.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=device))
        if args.magma_weight > 0:
            local_loss, local_stats, _ = magma_loss(
                model,
                adapted,
                batch,
                fragment_embeddings,
                fragment_to_index,
                args.magma_temperature,
                args.magma_intensity_weight,
                "train",
            )
        else:
            local_loss, local_stats = None, {}
        loss = global_loss
        if local_loss is not None and args.magma_weight > 0:
            loss = loss + float(args.magma_weight) * local_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if first_grad is None:
            first_grad = {
                "adapter_grad_norm": round(
                    sum(
                        float(parameter.grad.detach().float().norm().cpu())
                        for parameter in model.adapter.parameters()
                        if parameter.grad is not None
                    ),
                    6,
                ),
                "projection_params_with_grad": sum(
                    int(parameter.grad is not None) for parameter in model.projection.parameters()
                ),
            }
        optimizer.step()

        count = len(batch)
        totals["loss"] += float(loss.detach().cpu()) * count
        totals["global_loss"] += float(global_loss.detach().cpu()) * count
        if local_loss is not None:
            totals["magma_loss"] += float(local_loss.detach().cpu()) * count
        totals["magma_peaks"] += int(local_stats.get("peaks", 0))
        totals["items"] += count
        if args.max_train_batches > 0 and batch_index >= args.max_train_batches:
            break
    items = max(int(totals["items"]), 1)
    return {
        "loss": totals["loss"] / items,
        "global_loss": totals["global_loss"] / items,
        "magma_loss": totals["magma_loss"] / items,
        "magma_peaks": int(totals["magma_peaks"]),
        "items": int(totals["items"]),
        "adapter_gate": float(model.adapter.gate.detach().cpu()),
        "first_batch_grad_stats": first_grad,
    }


def rank_metrics(ranks: list[int]) -> dict[str, float | int]:
    if not ranks:
        return {"valid_queries": 0, "recall@1": 0.0, "recall@5": 0.0, "recall@20": 0.0, "mrr": 0.0}
    values = torch.tensor(ranks, dtype=torch.float32)
    result: dict[str, float | int] = {"valid_queries": len(ranks)}
    for cutoff in (1, 5, 20, 50, 100):
        result[f"recall@{cutoff}"] = round(float((values <= cutoff).float().mean()) * 100, 4)
    result["mrr"] = round(float((1.0 / values).mean()) * 100, 4)
    result["mean_rank"] = round(float(values.mean()), 4)
    return result


@torch.inference_mode()
def evaluate_global(
    model: ChemistryAlignmentModel,
    dataset: ChemistryTokenDataset,
    mol_store: MoleculeEmbeddingStore | ProjectedMoleculeStore,
    device: torch.device,
    batch_size: int,
    disable_adapter: bool,
    dataset_indices: list[int] | None = None,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    model.eval()
    ranks: list[int] = []
    rank_records: list[dict[str, Any]] = []
    selected = dataset_indices if dataset_indices is not None else list(range(len(dataset)))
    for start in tqdm(range(0, len(selected), batch_size), desc="Evaluating global candidate ranks"):
        indices = selected[start : start + batch_size]
        batch = [dataset[index] for index in indices]
        tokens = torch.stack([item["tokens"] for item in batch]).to(device)
        masks = torch.stack([item["mask"] for item in batch]).to(device)
        candidate_indices = [item["candidate_indices"] for item in batch]
        index_tensor, candidate_mask = make_padded_index_tensors(
            candidate_indices, [[True] * len(row) for row in candidate_indices]
        )
        unique, inverse = torch.unique(index_tensor.reshape(-1), sorted=True, return_inverse=True)
        mol_proj_unique = gather_projected_molecules(model, mol_store, unique, device)
        mol_proj = mol_proj_unique[inverse.to(device)].reshape(len(batch), index_tensor.shape[1], -1)
        spec_proj = model.project_global(tokens, masks, disable_adapter=disable_adapter)
        scores = torch.einsum("bd,bcd->bc", spec_proj, mol_proj) / model.projection.temperature()
        scores = scores.masked_fill(~candidate_mask.to(device), float("-inf"))
        order = torch.argsort(scores, dim=1, descending=True)
        batch_ranks = [int(value) for value in ((order == 0).nonzero(as_tuple=False)[:, 1] + 1).cpu()]
        ranks.extend(batch_ranks)
        rank_records.extend(
            {
                "identifier": str(item["identifier"]),
                "rank": rank,
                "candidate_count": len(item["candidate_indices"]),
            }
            for item, rank in zip(batch, batch_ranks, strict=True)
        )
    return rank_metrics(ranks), rank_records


@torch.inference_mode()
def evaluate_local_holdout(
    model: ChemistryAlignmentModel,
    dataset: ChemistryTokenDataset,
    fragment_embeddings: torch.Tensor,
    fragment_to_index: dict[str, int],
    device: torch.device,
    batch_size: int,
    args: argparse.Namespace,
    disable_adapter: bool,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    model.eval()
    holdout_indices = [index for index, item in enumerate(dataset.items) if item.get("magma_holdout")]
    totals = {"loss": 0.0, "peaks": 0, "top1_hits": 0.0, "rr_sum": 0.0, "rank_sum": 0.0}
    rank_records: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(holdout_indices), batch_size), desc="Evaluating MAGMa holdout"):
        indices = holdout_indices[start : start + batch_size]
        batch = [dataset[index] for index in indices]
        tokens = torch.stack([item["tokens"] for item in batch]).to(device)
        masks = torch.stack([item["mask"] for item in batch]).to(device)
        adapted = model.adapted_tokens(tokens, masks, disable_adapter=disable_adapter)
        loss, metrics, batch_rank_records = magma_loss(
            model,
            adapted,
            batch,
            fragment_embeddings,
            fragment_to_index,
            args.magma_temperature,
            args.magma_intensity_weight,
            "holdout",
        )
        peaks = int(metrics.get("peaks", 0))
        if loss is None or peaks == 0:
            continue
        totals["loss"] += float(loss.cpu()) * peaks
        totals["peaks"] += peaks
        totals["top1_hits"] += float(metrics["top1"]) * peaks
        totals["rr_sum"] += float(metrics["mrr"]) * peaks
        totals["rank_sum"] += float(metrics["mean_best_positive_rank"]) * peaks
        rank_records.extend(batch_rank_records)
    peaks = int(totals["peaks"])
    return {
        "holdout_queries": len(holdout_indices),
        "peaks": peaks,
        "loss": totals["loss"] / max(peaks, 1),
        "top1": totals["top1_hits"] / max(peaks, 1),
        "mrr": totals["rr_sum"] / max(peaks, 1),
        "mean_best_positive_rank": totals["rank_sum"] / max(peaks, 1),
    }, rank_records


def grad_l2_norm(grads: tuple[torch.Tensor | None, ...]) -> float:
    squared = 0.0
    for grad in grads:
        if grad is not None:
            squared += float(grad.detach().float().square().sum().cpu())
    return math.sqrt(squared)


def calibrate_magma_weight(
    model: ChemistryAlignmentModel,
    dataset: ChemistryTokenDataset,
    mol_store: MoleculeEmbeddingStore | ProjectedMoleculeStore,
    fragment_embeddings: torch.Tensor,
    fragment_to_index: dict[str, int],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Choose lambda on one fixed batch, then restore RNG so A/B stay matched."""
    eligible = [index for index, item in enumerate(dataset.items) if not item.get("magma_holdout")]
    if len(eligible) < args.batch_size:
        raise ValueError("Not enough non-holdout MAGMa queries for gradient calibration")
    rng = random.Random(args.seed + 991)
    chosen_indices = sorted(rng.sample(eligible, args.batch_size))
    batch = [dataset[index] for index in chosen_indices]
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    try:
        model.train()
        model.projection.eval()
        tokens = torch.stack([item["tokens"] for item in batch]).to(device)
        masks = torch.stack([item["mask"] for item in batch]).to(device)
        candidate_indices, candidate_mask = candidate_batch_indices(
            batch, args.candidate_negatives, random.Random(args.seed + 1)
        )
        flat_unique, inverse = torch.unique(candidate_indices.reshape(-1), sorted=True, return_inverse=True)
        with torch.no_grad():
            mol_proj_unique = gather_projected_molecules(model, mol_store, flat_unique, device)
        mol_proj = mol_proj_unique[inverse.to(device)].reshape(len(batch), candidate_indices.shape[1], -1)
        adapted = model.adapted_tokens(tokens, masks)
        spec_proj = model.projection.project_spec(adapted[:, 0, :])
        logits = torch.einsum("bd,bcd->bc", spec_proj, mol_proj) / model.projection.temperature()
        logits = logits.masked_fill(~candidate_mask.to(device), float("-inf"))
        global_loss = F.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=device))
        local_loss, local_stats, _ = magma_loss(
            model,
            adapted,
            batch,
            fragment_embeddings,
            fragment_to_index,
            args.magma_temperature,
            args.magma_intensity_weight,
            "train",
        )
        if local_loss is None:
            raise ValueError("Fixed calibration batch produced no usable MAGMa local loss")
        parameters = tuple(model.adapter.parameters())
        global_grads = torch.autograd.grad(global_loss, parameters, retain_graph=True, allow_unused=True)
        local_grads = torch.autograd.grad(local_loss, parameters, allow_unused=True)
        global_norm = grad_l2_norm(global_grads)
        local_norm = grad_l2_norm(local_grads)
        if global_norm <= 0 or local_norm <= 0:
            raise ValueError(f"Invalid calibration gradient norms: global={global_norm}, local={local_norm}")
        candidates = sorted(set(float(value) for value in args.magma_weight_candidates))
        if not candidates or candidates[0] <= 0:
            raise ValueError("--magma-weight-candidates must contain only positive values")
        ratios = {str(value): value * local_norm / global_norm for value in candidates}
        chosen = min(
            candidates,
            key=lambda value: (abs(ratios[str(value)] - args.target_local_global_grad_ratio), value),
        )
        return {
            "batch_identifiers": [str(item["identifier"]) for item in batch],
            "global_loss": float(global_loss.detach().cpu()),
            "local_loss": float(local_loss.detach().cpu()),
            "global_grad_norm": global_norm,
            "local_grad_norm": local_norm,
            "candidate_weighted_grad_ratios": ratios,
            "target_local_global_grad_ratio": float(args.target_local_global_grad_ratio),
            "chosen_magma_weight": chosen,
            "local_stats": local_stats,
        }
    finally:
        model.zero_grad(set_to_none=True)
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def candidate_cache_parity(
    model: ChemistryAlignmentModel,
    dataset: ChemistryTokenDataset,
    raw_store: MoleculeEmbeddingStore,
    projected_store: ProjectedMoleculeStore,
    device: torch.device,
    batch_size: int,
    query_count: int,
    seed: int,
) -> dict[str, Any]:
    count = min(query_count, len(dataset))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(dataset)), count))
    raw_metrics, raw_records = evaluate_global(
        model, dataset, raw_store, device, batch_size, True, dataset_indices=indices
    )
    cached_metrics, cached_records = evaluate_global(
        model, dataset, projected_store, device, batch_size, True, dataset_indices=indices
    )
    raw_ranks = {row["identifier"]: int(row["rank"]) for row in raw_records}
    cached_ranks = {row["identifier"]: int(row["rank"]) for row in cached_records}
    keys = sorted(set(raw_ranks) & set(cached_ranks))
    exact = sum(raw_ranks[key] == cached_ranks[key] for key in keys) / max(len(keys), 1)
    metric_deltas = {
        key: float(cached_metrics[key]) - float(raw_metrics[key])
        for key in ("recall@1", "recall@5", "recall@20", "mrr")
    }
    passed = exact >= 0.999 and max(abs(value) for value in metric_deltas.values()) <= 0.02
    return {
        "queries": len(keys),
        "rank_exact_fraction": exact,
        "raw_metrics": raw_metrics,
        "cached_metrics": cached_metrics,
        "metric_deltas_percentage_points": metric_deltas,
        "thresholds": {"rank_exact_fraction": 0.999, "max_metric_delta_percentage_points": 0.02},
        "passed": passed,
    }


def main() -> None:
    args = parse_args()
    if args.magma_weight < 0:
        raise ValueError("--magma-weight must be non-negative")
    if args.calibrate_magma_weight and args.magma_weight <= 0:
        raise ValueError("--calibrate-magma-weight requires a positive --magma-weight to select Stage B")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = choose_device(args.device)
    started = time.time()

    train_records = sample_records(load_jsonl(args.train_metadata), args.train_limit, args.seed)
    val_records = sample_records(load_jsonl(args.val_metadata), args.val_limit, args.seed + 1)
    train_tokens = load_token_cache(args.train_token_cache)
    val_tokens = load_token_cache(args.val_token_cache)
    magma_labels = load_magma_labels(args.magma_labels)
    holdout = make_holdout(magma_labels, args.magma_holdout_fraction, args.seed)
    sidecar = load_candidate_sidecar(args.candidate_sidecar)
    chemberta = torch.load(args.chemberta_cache, map_location="cpu", weights_only=False, mmap=True)
    smiles_to_index = {str(key): int(value) for key, value in chemberta["smiles_to_index"].items()}
    mol_store = MoleculeEmbeddingStore(
        chemberta,
        "count_morgan_r3_atom_pair_path_maccs",
        args.fingerprint_cache,
        additional_fingerprint_caches=args.additional_fingerprint_cache,
        preload_sparse_features=args.preload_sparse_features,
    )
    model = load_run_k_model(args.run_k_checkpoint, device, args)

    train_dataset = ChemistryTokenDataset(
        train_records,
        train_tokens,
        sidecar,
        smiles_to_index,
        magma_labels,
        args.candidate_pool_limit,
        holdout,
    )
    val_dataset = ChemistryTokenDataset(
        val_records,
        val_tokens,
        sidecar,
        smiles_to_index,
        {},
        args.candidate_pool_limit,
        set(),
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(f"Empty train/val dataset: train={train_dataset.stats}, val={val_dataset.stats}")

    required_candidates = required_candidate_indices([train_dataset, val_dataset])
    projected_mol_store = load_or_build_projected_candidate_cache(
        args.candidate_projection_cache,
        required_candidates,
        mol_store,
        model,
        device,
        args.candidate_projection_build_batch_size,
        args.run_k_checkpoint,
        args.candidate_projection_dtype,
    )
    parity = None
    if args.candidate_cache_parity_queries > 0:
        parity = candidate_cache_parity(
            model,
            val_dataset,
            mol_store,
            projected_mol_store,
            device,
            args.eval_batch_size,
            args.candidate_cache_parity_queries,
            args.seed + 17,
        )
    if args.candidate_projection_cache_only:
        payload = {
            "args": json_safe(vars(args)),
            "train_dataset": train_dataset.stats,
            "val_dataset": val_dataset.stats,
            "candidate_projection_cache": projected_mol_store.metadata,
            "candidate_cache_parity": parity,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        if parity is not None and not parity["passed"]:
            raise RuntimeError(
                "Candidate projection cache parity failed; rebuild a new cache with "
                "--candidate-projection-dtype float32"
            )
        return
    if parity is not None and not parity["passed"]:
        raise RuntimeError(
            "Candidate projection cache parity failed; rebuild a new cache with "
            "--candidate-projection-dtype float32"
        )

    used_labels = [item["magma"] for item in train_dataset.items if item.get("magma") is not None]
    cache_labels = list(magma_labels.values()) if args.fragment_cache_scope == "all_labels" else used_labels
    required_fragments = fragment_smiles_from_labels(cache_labels)
    fragment_cache = load_or_build_fragment_cache(
        args.fragment_cache,
        required_fragments,
        model,
        device,
        args.fragment_build_batch_size,
        args.magma_labels,
        args.run_k_checkpoint,
    )
    fragment_embeddings = fragment_cache["embeddings"]
    fragment_to_index = {str(key): int(value) for key, value in fragment_cache["smiles_to_index"].items()}
    if args.fragment_cache_only:
        print(json.dumps(fragment_cache["metadata"], indent=2, sort_keys=True))
        return

    calibration = None
    requested_magma_weight = float(args.magma_weight)
    if args.calibrate_magma_weight:
        calibration = calibrate_magma_weight(
            model,
            train_dataset,
            projected_mol_store,
            fragment_embeddings,
            fragment_to_index,
            device,
            args,
        )
        args.magma_weight = float(calibration["chosen_magma_weight"])
        print(json.dumps({"magma_gradient_calibration": calibration}, indent=2, sort_keys=True), flush=True)

    baseline_global, baseline_global_ranks = evaluate_global(
        model, val_dataset, projected_mol_store, device, args.eval_batch_size, disable_adapter=True
    )
    baseline_local, baseline_local_ranks = evaluate_local_holdout(
        model,
        train_dataset,
        fragment_embeddings,
        fragment_to_index,
        device,
        args.eval_batch_size,
        args,
        disable_adapter=True,
    )
    init_global, init_global_ranks = evaluate_global(
        model, val_dataset, projected_mol_store, device, args.eval_batch_size, False
    )
    init_local, init_local_ranks = evaluate_local_holdout(
        model,
        train_dataset,
        fragment_embeddings,
        fragment_to_index,
        device,
        args.eval_batch_size,
        args,
        disable_adapter=False,
    )
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "global_no_adapter": baseline_global,
            "local_no_adapter": baseline_local,
            "global": init_global,
            "local": init_local,
        }
    ]
    rank_history: dict[str, Any] = {
        "global_no_adapter": baseline_global_ranks,
        "local_no_adapter": baseline_local_ranks,
        "epochs": {"0": {"global": init_global_ranks, "local": init_local_ranks}},
    }
    print(json.dumps(history[-1], indent=2, sort_keys=True), flush=True)

    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_items,
        drop_last=False,
    )
    best_epoch = 0
    best_global = init_global
    best_state = {key: value.detach().cpu().clone() for key, value in model.adapter.state_dict().items()}
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            loader,
            projected_mol_store,
            fragment_embeddings,
            fragment_to_index,
            optimizer,
            device,
            args,
            epoch,
        )
        global_metrics, global_rank_records = evaluate_global(
            model, val_dataset, projected_mol_store, device, args.eval_batch_size, False
        )
        local_metrics, local_rank_records = evaluate_local_holdout(
            model,
            train_dataset,
            fragment_embeddings,
            fragment_to_index,
            device,
            args.eval_batch_size,
            args,
            disable_adapter=False,
        )
        row = {"epoch": epoch, "train": train_metrics, "global": global_metrics, "local": local_metrics}
        history.append(row)
        rank_history["epochs"][str(epoch)] = {
            "global": global_rank_records,
            "local": local_rank_records,
        }
        if (
            float(global_metrics["mrr"]),
            float(global_metrics["recall@1"]),
            float(global_metrics["recall@20"]),
        ) > (
            float(best_global["mrr"]),
            float(best_global["recall@1"]),
            float(best_global["recall@20"]),
        ):
            best_epoch = epoch
            best_global = global_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.adapter.state_dict().items()}
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)

    payload = {
        "protocol": {
            "dataset": "MassSpecGym v1.5",
            "split": "formula-group",
            "candidate_protocol": f"mass cap{args.candidate_pool_limit}",
            "spectrum": "frozen DreaMS backbone token cache",
            "molecule_target": "Run K r3+AtomPair+path+MACCS block-L2",
            "trainable": "residual chemistry adapter only",
            "stage": "B" if args.magma_weight > 0 else "A",
        },
        "args": json_safe(vars(args)),
        "train_dataset": train_dataset.stats,
        "val_dataset": val_dataset.stats,
        "fragment_cache_metadata": fragment_cache.get("metadata", {}),
        "candidate_projection_cache_metadata": projected_mol_store.metadata,
        "candidate_cache_parity": parity,
        "requested_magma_weight": requested_magma_weight,
        "magma_gradient_calibration": calibration,
        "baseline_global": baseline_global,
        "baseline_local": baseline_local,
        "history": history,
        "best_epoch": best_epoch,
        "best_global": best_global,
        "adapter_gate": float(model.adapter.gate.detach().cpu()),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rank_output = args.rank_output or args.out_json.with_name(f"{args.out_json.stem}.ranks.json")
    rank_output.parent.mkdir(parents=True, exist_ok=True)
    rank_payload = {
        "source_summary": str(args.out_json),
        "stage": payload["protocol"]["stage"],
        "seed": args.seed,
        "best_epoch": best_epoch,
        **rank_history,
    }
    rank_output.write_text(json.dumps(rank_payload, separators=(",", ":")) + "\n", encoding="utf-8")
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "adapter_state_dict": model.adapter.state_dict(),
                "best_adapter_state_dict": best_state,
                "best_epoch": best_epoch,
                "best_global": best_global,
                "args": json_safe(vars(args)),
                "history": history,
                "run_k_checkpoint": str(args.run_k_checkpoint),
            },
            args.checkpoint,
        )
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "rank_output": str(rank_output),
                "best_epoch": best_epoch,
                "best_global": best_global,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
