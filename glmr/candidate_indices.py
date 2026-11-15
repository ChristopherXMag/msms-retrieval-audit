from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CANDIDATE_INDEX_FORMAT = "glmr_candidate_indices_v1"
NEGATIVE_EXCLUSION_FORMAT = "glmr_candidate_negative_exclusion_v1"


def identifier_digest(identifiers: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for identifier in identifiers:
        digest.update(str(identifier).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class CandidateIndexStore:
    """Memory-mapped candidate row indices keyed by MassSpecGym identifier."""

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.metadata: dict[str, Any] = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.metadata.get("format") != CANDIDATE_INDEX_FORMAT:
            raise ValueError(
                f"Unsupported candidate-index format: {self.metadata.get('format')!r}"
            )
        self.records = int(self.metadata["records"])
        self.molecule_rows = int(self.metadata["molecule_rows"])
        self.molecule_smiles_sha256 = str(self.metadata.get("molecule_smiles_sha256") or "")
        self.indptr = np.load(self.root / self.metadata["indptr"], mmap_mode="r")
        self.indices = np.memmap(
            self.root / self.metadata["indices"],
            mode="r",
            dtype=np.dtype(self.metadata["index_dtype"]),
        )
        identifiers_path = self.root / self.metadata["identifiers"]
        with identifiers_path.open(encoding="utf-8") as handle:
            self.identifiers = [line.rstrip("\n") for line in handle if line.rstrip("\n")]
        if len(self.identifiers) != self.records:
            raise ValueError(
                f"Candidate identifiers length mismatch: {len(self.identifiers)} != {self.records}"
            )
        if len(self.indptr) != self.records + 1:
            raise ValueError(f"Candidate indptr length mismatch: {len(self.indptr)} != {self.records + 1}")
        if int(self.indptr[-1]) != len(self.indices):
            raise ValueError(f"Candidate nnz mismatch: {int(self.indptr[-1])} != {len(self.indices)}")
        expected_digest = str(self.metadata.get("identifier_sha256") or "")
        if expected_digest and identifier_digest(self.identifiers) != expected_digest:
            raise ValueError("Candidate identifier digest mismatch")
        if len(set(self.identifiers)) != len(self.identifiers):
            raise ValueError("Candidate index contains duplicate identifiers")
        self.identifier_to_row = {identifier: row for row, identifier in enumerate(self.identifiers)}

    def get(self, identifier: str, limit: int = 0) -> list[int] | None:
        row = self.row_for(identifier)
        if row is None:
            return None
        return self.get_row(row, limit)

    def row_for(self, identifier: str) -> int | None:
        return self.identifier_to_row.get(str(identifier))

    def row_length(self, row: int, limit: int = 0) -> int:
        if row < 0 or row >= self.records:
            raise IndexError(f"Candidate record row outside [0, {self.records}): {row}")
        start = int(self.indptr[row])
        end = int(self.indptr[row + 1])
        if limit > 0:
            end = min(end, start + int(limit))
        return end - start

    def get_row(self, row: int, limit: int = 0) -> list[int]:
        if row < 0 or row >= self.records:
            raise IndexError(f"Candidate record row outside [0, {self.records}): {row}")
        start = int(self.indptr[row])
        end = start + self.row_length(row, limit)
        values = np.asarray(self.indices[start:end], dtype=np.int64)
        if values.size == 0:
            return []
        if values.min() < 0 or values.max() >= self.molecule_rows:
            raise IndexError(
                f"Candidate rows outside [0, {self.molecule_rows}): {values.min()}..{values.max()}"
            )
        return [int(value) for value in values]


class CandidateNegativeExclusion:
    """Boolean molecule-row mask for split-safe candidate-negative sampling."""

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.metadata: dict[str, Any] = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        if self.metadata.get("format") != NEGATIVE_EXCLUSION_FORMAT:
            raise ValueError(
                "Unsupported candidate-negative exclusion format: "
                f"{self.metadata.get('format')!r}"
            )
        self.molecule_rows = int(self.metadata["molecule_rows"])
        self.molecule_smiles_sha256 = str(
            self.metadata.get("molecule_smiles_sha256") or ""
        )
        indices = np.load(self.root / str(self.metadata["indices"]), mmap_mode="r")
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Negative-exclusion indices must be a one-dimensional integer array")
        if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= self.molecule_rows):
            raise IndexError(
                "Negative-exclusion row outside molecule universe: "
                f"{int(indices.min())}..{int(indices.max())} vs {self.molecule_rows}"
            )
        if len(indices) > 1 and bool(np.any(indices[1:] <= indices[:-1])):
            raise ValueError("Negative-exclusion indices must be strictly increasing and unique")
        expected = int(self.metadata.get("excluded_rows", len(indices)))
        if expected != len(indices):
            raise ValueError(f"Negative-exclusion row count mismatch: {len(indices)} != {expected}")
        self.indices = indices
        self.mask = np.zeros(self.molecule_rows, dtype=np.bool_)
        self.mask[np.asarray(indices, dtype=np.int64)] = True

    def validate_candidate_store(self, store: CandidateIndexStore) -> None:
        if self.molecule_rows != store.molecule_rows:
            raise ValueError(
                "Negative-exclusion and candidate-index molecule row counts differ: "
                f"{self.molecule_rows} != {store.molecule_rows}"
            )
        if (
            self.molecule_smiles_sha256
            and store.molecule_smiles_sha256
            and self.molecule_smiles_sha256 != store.molecule_smiles_sha256
        ):
            raise ValueError("Negative-exclusion and candidate-index SMILES digests differ")

    def filter_candidates(self, values: list[int]) -> tuple[list[int], int]:
        if not values:
            return [], 0
        gt = int(values[0])
        # The denylist applies only when a molecule is sampled as a negative.
        # A current row's positive must remain available even when the same
        # connectivity occurs in a held-out split.
        kept = [gt]
        excluded = 0
        for value in values[1:]:
            if bool(self.mask[int(value)]):
                excluded += 1
            else:
                kept.append(int(value))
        return kept, excluded
