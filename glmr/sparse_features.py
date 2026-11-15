from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class SparseFeatureMatrix:
    """Memory-mapped CSR-like rows with on-demand dense batch materialization."""

    def __init__(self, root: Path, metadata: dict[str, Any]) -> None:
        self.root = root
        self.dim = int(metadata["dim"])
        self.rows = int(metadata["rows"])
        self.transform = str(metadata.get("transform") or "none")
        self.indptr = np.load(root / metadata["indptr"], mmap_mode="r")
        self.indices = np.memmap(
            root / metadata["indices"],
            mode="r",
            dtype=np.dtype(metadata["index_dtype"]),
        )
        self.values = np.memmap(
            root / metadata["values"],
            mode="r",
            dtype=np.dtype(metadata["value_dtype"]),
        )
        if len(self.indptr) != self.rows + 1:
            raise ValueError(f"Invalid indptr length: {len(self.indptr)} != {self.rows + 1}")
        if int(self.indptr[-1]) != len(self.indices) or len(self.indices) != len(self.values):
            raise ValueError("Sparse feature indices/values/indptr lengths do not agree")

    def preload(self) -> None:
        """Replace memory maps with sequentially loaded RAM arrays for random-row training."""
        self.indptr = np.array(self.indptr, copy=True)
        self.indices = np.array(self.indices, copy=True)
        self.values = np.array(self.values, copy=True)

    def gather(self, flat_indices: torch.Tensor) -> torch.Tensor:
        flat = flat_indices.detach().reshape(-1).cpu().numpy().astype(np.int64, copy=False)
        if flat.size == 0:
            return torch.empty((0, self.dim), dtype=torch.float32)
        if flat.min() < 0 or flat.max() >= self.rows:
            raise IndexError(f"Sparse feature row outside [0, {self.rows}): {flat.min()}..{flat.max()}")

        starts = np.asarray(self.indptr[flat], dtype=np.int64)
        ends = np.asarray(self.indptr[flat + 1], dtype=np.int64)
        lengths = ends - starts
        total = int(lengths.sum())
        dense = np.zeros((len(flat), self.dim), dtype=np.float32)
        if total == 0:
            return torch.from_numpy(dense)

        row_ids = np.repeat(np.arange(len(flat), dtype=np.int64), lengths)
        row_offsets = np.repeat(np.cumsum(lengths, dtype=np.int64) - lengths, lengths)
        source = np.repeat(starts, lengths) + np.arange(total, dtype=np.int64) - row_offsets
        columns = np.asarray(self.indices[source], dtype=np.int64)
        values = np.asarray(self.values[source], dtype=np.float32)
        if self.transform == "log1p":
            np.log1p(values, out=values)
        elif self.transform != "none":
            raise ValueError(f"Unsupported sparse feature transform: {self.transform}")
        dense[row_ids, columns] = values
        return torch.from_numpy(dense)


class SparseFeatureCache:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path
        self.root = manifest_path.parent
        self.metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.rows = int(self.metadata["rows"])
        self.smiles_sha256 = self.metadata.get("smiles_sha256")
        self.features = {
            name: SparseFeatureMatrix(self.root, feature_metadata)
            for name, feature_metadata in self.metadata["features"].items()
        }
        self.row_sources: np.memmap | None = None
        self.row_indices: np.memmap | None = None
        projection_order = self.metadata.get("projection_order")
        if projection_order is not None:
            self.row_sources = np.memmap(
                self.root / projection_order["sources"],
                mode="r",
                dtype=np.dtype(projection_order["source_dtype"]),
            )
            self.row_indices = np.memmap(
                self.root / projection_order["indices"],
                mode="r",
                dtype=np.dtype(projection_order["index_dtype"]),
            )
            if len(self.row_sources) != self.rows or len(self.row_indices) != self.rows:
                raise ValueError("Sparse projection-order maps do not match manifest rows")

    def canonicalize_unique_indices(
        self, unique_indices: torch.Tensor, inverse: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.row_sources is None or self.row_indices is None or not len(unique_indices):
            return unique_indices, inverse
        logical = (
            unique_indices.detach().reshape(-1).cpu().numpy().astype(np.int64, copy=False)
        )
        permutation_np = np.lexsort((self.row_indices[logical], self.row_sources[logical]))
        permutation = torch.from_numpy(permutation_np.astype(np.int64, copy=False))
        old_to_new = torch.empty(len(permutation), dtype=torch.long)
        old_to_new[permutation] = torch.arange(len(permutation), dtype=torch.long)
        return unique_indices[permutation], old_to_new[inverse]

    def gather(self, name: str, flat_indices: torch.Tensor) -> torch.Tensor:
        try:
            feature = self.features[name]
        except KeyError as exc:
            raise KeyError(f"Sparse feature {name!r} not found in {self.manifest_path}") from exc
        return feature.gather(flat_indices)

    def dim(self, name: str) -> int:
        try:
            return self.features[name].dim
        except KeyError as exc:
            raise KeyError(f"Sparse feature {name!r} not found in {self.manifest_path}") from exc

    def preload(self, names: list[str] | tuple[str, ...]) -> None:
        for name in dict.fromkeys(names):
            try:
                self.features[name].preload()
            except KeyError as exc:
                raise KeyError(f"Sparse feature {name!r} not found in {self.manifest_path}") from exc


class CompositeSparseFeatureCache:
    """Concatenate existing sparse caches without copying their large CSR payloads."""

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.metadata = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.metadata.get("format") != "glmr_sparse_features_composite_v1":
            raise ValueError(f"Unsupported composite sparse format: {self.metadata.get('format')!r}")
        self.rows = int(self.metadata["rows"])
        self.smiles_sha256 = self.metadata.get("smiles_sha256")
        self.segments: list[dict[str, Any]] = []
        expected_start = 0
        feature_dims: dict[str, int] = {}
        for raw_segment in self.metadata["segments"]:
            start = int(raw_segment["row_start"])
            rows = int(raw_segment["rows"])
            segment_digest = str(raw_segment.get("smiles_sha256") or "")
            if start != expected_start or rows <= 0:
                raise ValueError(
                    f"Composite sparse segments must be positive and contiguous: {start}, {rows}, expected {expected_start}"
                )
            sources: dict[str, SparseFeatureCache] = {}
            for raw_path in raw_segment["manifests"]:
                path = Path(raw_path)
                if not path.is_absolute():
                    path = (self.root / path).resolve()
                cache = SparseFeatureCache(path)
                if cache.rows != rows:
                    raise ValueError(f"Composite segment row mismatch in {path}: {cache.rows} != {rows}")
                if segment_digest and cache.smiles_sha256 != segment_digest:
                    raise ValueError(f"Composite segment SMILES digest mismatch in {path}")
                for name, matrix in cache.features.items():
                    if name in sources:
                        raise ValueError(f"Duplicate feature {name!r} in composite segment at row {start}")
                    if name in feature_dims and feature_dims[name] != matrix.dim:
                        raise ValueError(f"Composite feature dimension mismatch for {name!r}")
                    feature_dims[name] = matrix.dim
                    sources[name] = cache
            self.segments.append({"start": start, "end": start + rows, "sources": sources})
            expected_start += rows
        if expected_start != self.rows:
            raise ValueError(f"Composite sparse rows mismatch: {expected_start} != {self.rows}")
        common = set(self.segments[0]["sources"])
        for segment in self.segments[1:]:
            common.intersection_update(segment["sources"])
        requested = set(self.metadata.get("features") or common)
        if not requested.issubset(common):
            raise ValueError(f"Composite features missing from one or more segments: {sorted(requested - common)}")
        self.features = {name: feature_dims[name] for name in sorted(requested)}

    def gather(self, name: str, flat_indices: torch.Tensor) -> torch.Tensor:
        if name not in self.features:
            raise KeyError(f"Sparse feature {name!r} not found in {self.manifest_path}")
        original = flat_indices.detach().reshape(-1).cpu().long()
        if original.numel() == 0:
            return torch.empty((0, self.dim(name)), dtype=torch.float32)
        if int(original.min()) < 0 or int(original.max()) >= self.rows:
            raise IndexError(
                f"Composite sparse row outside [0, {self.rows}): {int(original.min())}..{int(original.max())}"
            )
        output = torch.empty((len(original), self.dim(name)), dtype=torch.float32)
        for segment in self.segments:
            mask = (original >= segment["start"]) & (original < segment["end"])
            if not bool(mask.any()):
                continue
            local = original[mask] - int(segment["start"])
            source: SparseFeatureCache = segment["sources"][name]
            output[mask] = source.gather(name, local)
        return output

    def dim(self, name: str) -> int:
        try:
            return int(self.features[name])
        except KeyError as exc:
            raise KeyError(f"Sparse feature {name!r} not found in {self.manifest_path}") from exc

    def preload(self, names: list[str] | tuple[str, ...]) -> None:
        for segment in self.segments:
            caches = {id(cache): cache for cache in segment["sources"].values()}
            for cache in caches.values():
                selected = [name for name in names if name in cache.features]
                if selected:
                    cache.preload(selected)


class IndexedSparseFeatureCache:
    """Expose an arbitrary logical row order over one or more sparse caches.

    The indirection keeps the canonical full-v1.5 molecule row order without
    copying the large CSR payload already available in the legacy cache.
    """

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.metadata = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.metadata.get("format") != "glmr_sparse_features_indexed_v1":
            raise ValueError(f"Unsupported indexed sparse format: {self.metadata.get('format')!r}")
        self.rows = int(self.metadata["rows"])
        self.smiles_sha256 = self.metadata.get("smiles_sha256")
        row_map = self.metadata["row_map"]
        self.row_sources = np.memmap(
            self.root / row_map["sources"],
            mode="r",
            dtype=np.dtype(row_map["source_dtype"]),
        )
        self.row_indices = np.memmap(
            self.root / row_map["indices"],
            mode="r",
            dtype=np.dtype(row_map["index_dtype"]),
        )
        if len(self.row_sources) != self.rows or len(self.row_indices) != self.rows:
            raise ValueError("Indexed sparse row-map lengths do not match manifest rows")

        self.sources: list[dict[str, SparseFeatureCache]] = []
        feature_dims: dict[str, int] = {}
        for source_id, raw_source in enumerate(self.metadata["sources"]):
            if int(raw_source.get("source_id", source_id)) != source_id:
                raise ValueError("Indexed sparse source ids must be contiguous and zero-based")
            expected_rows = int(raw_source["rows"])
            expected_digest = str(raw_source.get("smiles_sha256") or "")
            feature_sources: dict[str, SparseFeatureCache] = {}
            for raw_path in raw_source["manifests"]:
                path = Path(raw_path)
                if not path.is_absolute():
                    path = (self.root / path).resolve()
                cache = SparseFeatureCache(path)
                if cache.rows != expected_rows:
                    raise ValueError(f"Indexed source row mismatch in {path}: {cache.rows} != {expected_rows}")
                if expected_digest and cache.smiles_sha256 != expected_digest:
                    raise ValueError(f"Indexed source SMILES digest mismatch in {path}")
                for name, matrix in cache.features.items():
                    if name in feature_sources:
                        raise ValueError(f"Duplicate feature {name!r} in indexed source {source_id}")
                    if name in feature_dims and feature_dims[name] != matrix.dim:
                        raise ValueError(f"Indexed feature dimension mismatch for {name!r}")
                    feature_dims[name] = matrix.dim
                    feature_sources[name] = cache
            self.sources.append(feature_sources)

        if self.rows:
            minimum_source = int(self.row_sources.min())
            maximum_source = int(self.row_sources.max())
            if minimum_source < 0 or maximum_source >= len(self.sources):
                raise ValueError(
                    f"Indexed sparse source id outside [0, {len(self.sources)}): "
                    f"{minimum_source}..{maximum_source}"
                )
            for source_id, raw_source in enumerate(self.metadata["sources"]):
                mask = self.row_sources == source_id
                if bool(mask.any()) and int(self.row_indices[mask].max()) >= int(raw_source["rows"]):
                    raise ValueError(f"Indexed sparse local row exceeds source {source_id} bounds")

        common = set(self.sources[0])
        for source in self.sources[1:]:
            common.intersection_update(source)
        requested = set(self.metadata.get("features") or common)
        if not requested.issubset(common):
            raise ValueError(f"Indexed features missing from one or more sources: {sorted(requested - common)}")
        self.features = {name: feature_dims[name] for name in sorted(requested)}

    def gather(self, name: str, flat_indices: torch.Tensor) -> torch.Tensor:
        if name not in self.features:
            raise KeyError(f"Sparse feature {name!r} not found in {self.manifest_path}")
        logical = flat_indices.detach().reshape(-1).cpu().numpy().astype(np.int64, copy=False)
        if logical.size == 0:
            return torch.empty((0, self.dim(name)), dtype=torch.float32)
        if logical.min() < 0 or logical.max() >= self.rows:
            raise IndexError(
                f"Indexed sparse row outside [0, {self.rows}): {logical.min()}..{logical.max()}"
            )
        source_ids = np.asarray(self.row_sources[logical], dtype=np.int64)
        local_rows = np.asarray(self.row_indices[logical], dtype=np.int64)
        output = torch.empty((len(logical), self.dim(name)), dtype=torch.float32)
        for source_id, source in enumerate(self.sources):
            positions = np.flatnonzero(source_ids == source_id)
            if not len(positions):
                continue
            cache = source[name]
            gathered = cache.gather(name, torch.from_numpy(local_rows[positions]))
            output[torch.from_numpy(positions)] = gathered
        return output

    def canonicalize_unique_indices(
        self,
        unique_indices: torch.Tensor,
        inverse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Order unique logical rows by their physical cache location.

        Candidate evaluation projects each unique molecule once.  Keeping that
        projection batch in physical-cache order preserves the numerical path
        of the legacy cache for reused rows, even when the public/logical row
        order is the canonical full-v1.5 order.  ``inverse`` is remapped so the
        caller still reconstructs the original candidate layout.
        """
        logical = unique_indices.detach().reshape(-1).cpu().numpy().astype(np.int64, copy=False)
        if logical.size == 0:
            return unique_indices, inverse
        source_ids = np.asarray(self.row_sources[logical], dtype=np.int64)
        local_rows = np.asarray(self.row_indices[logical], dtype=np.int64)
        permutation_np = np.lexsort((local_rows, source_ids)).astype(np.int64, copy=False)
        permutation = torch.from_numpy(permutation_np)
        old_to_new = torch.empty(len(permutation), dtype=torch.long)
        old_to_new[permutation] = torch.arange(len(permutation), dtype=torch.long)
        return unique_indices[permutation], old_to_new[inverse]

    def dim(self, name: str) -> int:
        try:
            return int(self.features[name])
        except KeyError as exc:
            raise KeyError(f"Sparse feature {name!r} not found in {self.manifest_path}") from exc

    def preload(self, names: list[str] | tuple[str, ...]) -> None:
        for source in self.sources:
            caches = {id(cache): cache for cache in source.values()}
            for cache in caches.values():
                selected = [name for name in names if name in cache.features]
                if selected:
                    cache.preload(selected)


SparseFeatureBackend = SparseFeatureCache | CompositeSparseFeatureCache | IndexedSparseFeatureCache


def load_sparse_feature_cache(manifest_path: Path) -> SparseFeatureBackend:
    metadata = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if metadata.get("format") == "glmr_sparse_features_composite_v1":
        return CompositeSparseFeatureCache(Path(manifest_path))
    if metadata.get("format") == "glmr_sparse_features_indexed_v1":
        return IndexedSparseFeatureCache(Path(manifest_path))
    return SparseFeatureCache(Path(manifest_path))
