from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from tqdm import tqdm


FEATURES = {
    "count_morgan": {"dim": 2048, "transform": "log1p"},
    "count_morgan_r3": {"dim": 2048, "transform": "log1p"},
    "count_atom_pair": {"dim": 2048, "transform": "log1p"},
    "rdkit_path_count": {"dim": 2048, "transform": "log1p"},
    "maccs": {"dim": 167, "transform": "none"},
}
DEFAULT_FEATURES = ("count_morgan", "count_atom_pair", "rdkit_path_count", "maccs")
GENERATOR_METADATA = {
    "count_morgan": {"radius": 2, "fp_size": 2048, "include_chirality": True},
    "count_morgan_r3": {"radius": 3, "fp_size": 2048, "include_chirality": True},
    "count_atom_pair": {"fp_size": 2048, "include_chirality": True},
    "rdkit_path_count": {"fp_size": 2048, "min_path": 1, "max_path": 7, "use_hs": True},
    "maccs": {"bits": 167},
}
_GENERATORS: dict[str, Any] = {}
_ACTIVE_FEATURES: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build resumable sparse count/path fingerprint features in an existing molecule-cache row order."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-cache", type=Path)
    source.add_argument(
        "--smiles-jsonl",
        type=Path,
        help="One canonical SMILES per line, used when no dense molecule cache is needed.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--chirality-mode",
        choices=["chiral", "nonchiral"],
        default="chiral",
        help="Whether Morgan/AtomPair generators encode stereochemistry. Defaults to the historical chiral setting.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Debug row limit; 0 means all molecules.")
    parser.add_argument(
        "--features",
        nargs="+",
        choices=sorted(FEATURES),
        default=list(DEFAULT_FEATURES),
        help="Feature blocks to generate. Defaults preserve the original four-block cache.",
    )
    return parser.parse_args()


def generator_metadata(feature_names: tuple[str, ...], chirality_mode: str) -> dict[str, dict[str, Any]]:
    if chirality_mode not in {"chiral", "nonchiral"}:
        raise ValueError(f"Unsupported chirality mode: {chirality_mode!r}")
    include_chirality = chirality_mode == "chiral"
    metadata: dict[str, dict[str, Any]] = {}
    for name in feature_names:
        config = dict(GENERATOR_METADATA[name])
        if "include_chirality" in config:
            config["include_chirality"] = include_chirality
        metadata[name] = config
    return metadata


def init_worker(feature_names: tuple[str, ...], chirality_mode: str = "chiral") -> None:
    global _ACTIVE_FEATURES, _GENERATORS
    _ACTIVE_FEATURES = feature_names
    _GENERATORS = {}
    metadata = generator_metadata(feature_names, chirality_mode)
    for name in feature_names:
        if name in ("count_morgan", "count_morgan_r3"):
            _GENERATORS[name] = rdFingerprintGenerator.GetMorganGenerator(
                radius=int(metadata[name]["radius"]),
                fpSize=FEATURES[name]["dim"],
                includeChirality=bool(metadata[name]["include_chirality"]),
            )
        elif name == "count_atom_pair":
            _GENERATORS[name] = rdFingerprintGenerator.GetAtomPairGenerator(
                fpSize=FEATURES[name]["dim"],
                includeChirality=bool(metadata[name]["include_chirality"]),
            )
        elif name == "rdkit_path_count":
            _GENERATORS[name] = rdFingerprintGenerator.GetRDKitFPGenerator(
                fpSize=FEATURES[name]["dim"],
                minPath=1,
                maxPath=7,
                useHs=True,
                branchedPaths=True,
                useBondOrder=True,
                countSimulation=False,
            )


def sparse_count(fp: Any) -> tuple[np.ndarray, np.ndarray]:
    elements = fp.GetNonzeroElements()
    if not elements:
        return np.empty(0, dtype=np.uint16), np.empty(0, dtype=np.uint16)
    ordered = sorted(elements.items())
    indices = np.fromiter((item[0] for item in ordered), dtype=np.uint16, count=len(ordered))
    values_int = [int(item[1]) for item in ordered]
    if max(values_int) > np.iinfo(np.uint16).max:
        raise OverflowError(f"Fingerprint count exceeds uint16: {max(values_int)}")
    values = np.asarray(values_int, dtype=np.uint16)
    return indices, values


def fingerprint_chunk(payload: tuple[int, list[str]]) -> dict[str, Any]:
    start, smiles_values = payload
    lengths = {name: np.zeros(len(smiles_values), dtype=np.uint32) for name in _ACTIVE_FEATURES}
    index_parts: dict[str, list[np.ndarray]] = {name: [] for name in _ACTIVE_FEATURES}
    value_parts: dict[str, list[np.ndarray]] = {name: [] for name in _ACTIVE_FEATURES}
    invalid = 0
    for row, smiles in enumerate(smiles_values):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid += 1
            continue
        for name in _ACTIVE_FEATURES:
            if name == "maccs":
                maccs = MACCSkeys.GenMACCSKeys(mol)
                indices = np.asarray(list(maccs.GetOnBits()), dtype=np.uint16)
                values = np.ones(len(indices), dtype=np.uint16)
            else:
                indices, values = sparse_count(_GENERATORS[name].GetCountFingerprint(mol))
            lengths[name][row] = len(indices)
            index_parts[name].append(indices)
            value_parts[name].append(values)

    result: dict[str, Any] = {"start": start, "rows": len(smiles_values), "invalid": invalid, "features": {}}
    for name in _ACTIVE_FEATURES:
        result["features"][name] = {
            "lengths": lengths[name],
            "indices": np.concatenate(index_parts[name]) if index_parts[name] else np.empty(0, dtype=np.uint16),
            "values": np.concatenate(value_parts[name]) if value_parts[name] else np.empty(0, dtype=np.uint16),
        }
    return result


def smiles_digest(smiles: list[str]) -> str:
    digest = hashlib.sha256()
    for value in smiles:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_resume_configuration(
    progress: dict[str, Any],
    *,
    rows: int,
    digest: str,
    feature_names: tuple[str, ...],
    chirality_mode: str,
    actual_generator_metadata: dict[str, dict[str, Any]],
) -> None:
    if int(progress.get("rows", rows)) != rows or progress.get("smiles_sha256", digest) != digest:
        raise ValueError("Resume cache row order does not match --base-cache")
    resumed_features = tuple(progress.get("features") or progress.get("nnz", {}).keys())
    if resumed_features != feature_names:
        raise ValueError(f"Resume feature set does not match: {resumed_features} != {feature_names}")
    if progress.get("chirality_mode") != chirality_mode:
        raise ValueError(
            "Resume chirality mode does not match: "
            f"{progress.get('chirality_mode')!r} != {chirality_mode!r}"
        )
    if progress.get("generator") != actual_generator_metadata:
        raise ValueError("Resume fingerprint generator configuration does not match requested configuration")
    if progress.get("rdkit_version") != rdBase.rdkitVersion:
        raise ValueError(
            "Resume RDKit version does not match: "
            f"{progress.get('rdkit_version')!r} != {rdBase.rdkitVersion!r}"
        )
    if progress.get("rdkit_build") != rdBase.rdkitBuild:
        raise ValueError(
            "Resume RDKit build does not match: "
            f"{progress.get('rdkit_build')!r} != {rdBase.rdkitBuild!r}"
        )


def main() -> None:
    args = parse_args()
    feature_names = tuple(dict.fromkeys(args.features))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.base_cache is not None:
        print(f"Loading molecule row order from {args.base_cache}", flush=True)
        base = torch.load(args.base_cache, map_location="cpu", weights_only=False, mmap=True)
        smiles = list(base["smiles"])
        source_description = str(args.base_cache)
    else:
        print(f"Loading molecule row order from {args.smiles_jsonl}", flush=True)
        with args.smiles_jsonl.open(encoding="utf-8") as handle:
            smiles = [line.rstrip("\n") for line in handle if line.rstrip("\n")]
        source_description = str(args.smiles_jsonl)
    if args.limit > 0:
        smiles = smiles[: args.limit]
    rows = len(smiles)
    digest = smiles_digest(smiles)
    actual_generator_metadata = generator_metadata(feature_names, args.chirality_mode)
    progress_path = args.out_dir / "progress.json"

    progress = {
        "completed_rows": 0,
        "invalid_smiles": 0,
        "features": list(feature_names),
        "nnz": {name: 0 for name in feature_names},
        "chirality_mode": args.chirality_mode,
        "generator": actual_generator_metadata,
        "rdkit_version": rdBase.rdkitVersion,
        "rdkit_build": rdBase.rdkitBuild,
    }
    if args.resume:
        if not progress_path.exists():
            raise FileNotFoundError(f"--resume requested but no progress file exists: {progress_path}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        validate_resume_configuration(
            progress,
            rows=rows,
            digest=digest,
            feature_names=feature_names,
            chirality_mode=args.chirality_mode,
            actual_generator_metadata=actual_generator_metadata,
        )
    elif progress_path.exists():
        raise FileExistsError(f"Output already contains progress; use --resume or a new --out-dir: {progress_path}")

    completed = int(progress["completed_rows"])
    indptrs: dict[str, np.memmap] = {}
    handles: dict[str, tuple[Any, Any]] = {}
    for name in feature_names:
        indptr_path = args.out_dir / f"{name}.indptr.npy"
        if completed:
            indptr = np.load(indptr_path, mmap_mode="r+")
        else:
            indptr = np.lib.format.open_memmap(indptr_path, mode="w+", dtype=np.int64, shape=(rows + 1,))
            indptr[0] = 0
        indptrs[name] = indptr
        index_path = args.out_dir / f"{name}.indices.bin"
        value_path = args.out_dir / f"{name}.values.bin"
        expected_bytes = int(progress["nnz"][name]) * np.dtype(np.uint16).itemsize
        if completed and (index_path.stat().st_size != expected_bytes or value_path.stat().st_size != expected_bytes):
            raise ValueError(f"Resume file size mismatch for {name}")
        handles[name] = (index_path.open("ab"), value_path.open("ab"))

    tasks = [
        (start, smiles[start : min(start + args.chunk_size, rows)])
        for start in range(completed, rows, args.chunk_size)
    ]
    context = mp.get_context("spawn")
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=init_worker,
            initargs=(feature_names, args.chirality_mode),
        ) as executor:
            iterator = executor.map(fingerprint_chunk, tasks, chunksize=1)
            for result in tqdm(iterator, total=len(tasks), desc="Building sparse fingerprints"):
                start = int(result["start"])
                if start != completed:
                    raise ValueError(f"Out-of-order chunk: expected {completed}, got {start}")
                chunk_rows = int(result["rows"])
                for name in feature_names:
                    feature = result["features"][name]
                    feature["indices"].tofile(handles[name][0])
                    feature["values"].tofile(handles[name][1])
                    lengths = feature["lengths"].astype(np.int64, copy=False)
                    base_nnz = int(progress["nnz"][name])
                    indptrs[name][start + 1 : start + chunk_rows + 1] = base_nnz + np.cumsum(lengths)
                    progress["nnz"][name] = base_nnz + int(lengths.sum())
                    handles[name][0].flush()
                    handles[name][1].flush()
                    indptrs[name].flush()
                completed += chunk_rows
                progress.update(
                    {
                        "base_cache": source_description,
                        "rows": rows,
                        "smiles_sha256": digest,
                        "completed_rows": completed,
                        "invalid_smiles": int(progress["invalid_smiles"]) + int(result["invalid"]),
                    }
                )
                write_progress(progress_path, progress)
    finally:
        for index_handle, value_handle in handles.values():
            index_handle.close()
            value_handle.close()

    manifest = {
        "format": "glmr_sparse_features_v1",
        "base_cache": source_description,
        "rows": rows,
        "smiles_sha256": digest,
        "invalid_smiles": int(progress["invalid_smiles"]),
        "features": {
            name: {
                "rows": rows,
                "dim": config["dim"],
                "transform": config["transform"],
                "indptr": f"{name}.indptr.npy",
                "indices": f"{name}.indices.bin",
                "values": f"{name}.values.bin",
                "index_dtype": "uint16",
                "value_dtype": "uint16",
                "nnz": int(progress["nnz"][name]),
            }
            for name, config in FEATURES.items()
            if name in feature_names
        },
        "chirality_mode": args.chirality_mode,
        "generator": actual_generator_metadata,
        "rdkit_version": rdBase.rdkitVersion,
        "rdkit_build": rdBase.rdkitBuild,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **progress}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
