from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
import time
import types
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, RDLogger
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
MS_PRED_SRC = REPO_ROOT / "external" / "ms-pred" / "src"
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / "outputs" / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "outputs" / ".cache" / "matplotlib"))

def _load_ms_pred_magma() -> tuple[Any, Any]:
    """Load MAGMa without importing ms-pred's optional torch-scatter stack."""

    if not MS_PRED_SRC.exists():
        raise FileNotFoundError(f"Missing ms-pred source tree: {MS_PRED_SRC}")
    package_paths = {
        "ms_pred": MS_PRED_SRC / "ms_pred",
        "ms_pred.common": MS_PRED_SRC / "ms_pred" / "common",
        "ms_pred.magma": MS_PRED_SRC / "ms_pred" / "magma",
    }
    for name, path in package_paths.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        module.__package__ = name
        sys.modules[name] = module

    def load(name: str, path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    chem_utils = load(
        "ms_pred.common.chem_utils",
        MS_PRED_SRC / "ms_pred" / "common" / "chem_utils.py",
    )
    sys.modules["ms_pred.common"].chem_utils = chem_utils
    fragmentation_module = load(
        "ms_pred.magma.fragmentation",
        MS_PRED_SRC / "ms_pred" / "magma" / "fragmentation.py",
    )
    sys.modules["ms_pred.magma"].fragmentation = fragmentation_module
    return chem_utils, fragmentation_module


common, fragmentation = _load_ms_pred_magma()


DEFAULT_METADATA = (
    REPO_ROOT
    / "outputs"
    / "msalign_formula_group_split_v15"
    / "train_sample20k_seed20260623.metadata.jsonl"
)
DEFAULT_V15_TSV = REPO_ROOT / "data" / "MassSpecGym1.5.tsv"

_WORKER_CONFIG: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and audit deterministic MAGMa peak-to-fragment pseudo-labels "
            "for a fixed MassSpecGym v1.5 metadata sample."
        )
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--v15-tsv", type=Path, default=DEFAULT_V15_TSV)
    parser.add_argument(
        "--topk-export",
        type=Path,
        default=None,
        help=(
            "Optional split-scoped top-K JSONL supplying mz/intensity/adduct. "
            "When set, the historical combined v1.5 TSV is not read."
        ),
    )
    parser.add_argument("--protocol-label", default="formula_group_train20k")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--ppm-tolerance", type=float, default=20.0)
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=int(fragmentation.FRAGMENT_ENGINE_PARAMS["max_tree_depth"]),
    )
    parser.add_argument(
        "--max-broken-bonds",
        type=int,
        default=int(fragmentation.FRAGMENT_ENGINE_PARAMS["max_broken_bonds"]),
        help=(
            "Pass-through ms-pred FragmentEngine parameter. In the current "
            "implementation this bounds hydrogen-shift buckets, not tree depth."
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def parse_float_list(value: str | None) -> list[float]:
    if not value:
        return []
    return [float(item) for item in str(value).split(",") if item]


def load_v15_rows(path: Path, identifiers: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            identifier = str(row.get("identifier"))
            if identifier not in identifiers:
                continue
            rows[identifier] = {
                "identifier": identifier,
                "mzs": parse_float_list(row.get("mzs")),
                "intensities": parse_float_list(row.get("intensities")),
                "smiles": row.get("smiles"),
                "adduct": row.get("adduct"),
                "precursor_mz": float(row["precursor_mz"]) if row.get("precursor_mz") else None,
            }
            if len(rows) == len(identifiers):
                break
    missing = identifiers - set(rows)
    if missing:
        raise ValueError(f"Missing {len(missing)} identifiers from {path}; first={sorted(missing)[0]!r}")
    return rows


def load_topk_rows(path: Path, identifiers: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            identifier = str(row.get("identifier"))
            if identifier not in identifiers:
                continue
            if identifier in rows:
                raise ValueError(f"Duplicate identifier in {path}: {identifier!r}")
            rows[identifier] = {
                "identifier": identifier,
                "mzs": [float(value) for value in row.get("mzs") or []],
                "intensities": [float(value) for value in row.get("intensities") or []],
                "smiles": None,
                "adduct": row.get("adduct"),
                "precursor_mz": row.get("precursor_mz"),
            }
    missing = identifiers - set(rows)
    if missing:
        raise ValueError(f"Missing {len(missing)} identifiers from {path}; first={sorted(missing)[0]!r}")
    return rows


def select_dreams_peaks(
    mzs: list[float], intensities: list[float], n_highest_peaks: int
) -> list[dict[str, float | int]]:
    if len(mzs) != len(intensities):
        raise ValueError(f"m/z and intensity length mismatch: {len(mzs)} vs {len(intensities)}")
    if not mzs:
        return []
    mz_ar = np.asarray(mzs, dtype=np.float64)
    inten_ar = np.asarray(intensities, dtype=np.float64)
    if n_highest_peaks > 0 and len(mz_ar) > n_highest_peaks:
        # DreaMS keeps the highest-intensity peaks but restores their original order.
        selected = np.sort(np.argsort(inten_ar)[-n_highest_peaks:])
    else:
        selected = np.arange(len(mz_ar))
    max_intensity = float(inten_ar[selected].max(initial=0.0))
    return [
        {
            "source_peak_index": int(source_idx),
            "token_index": int(token_idx + 1),  # token 0 is the precursor token
            "mz": float(mz_ar[source_idx]),
            "intensity": float(inten_ar[source_idx]),
            "relative_intensity": (
                float(inten_ar[source_idx] / max_intensity) if max_intensity > 0 else 0.0
            ),
        }
        for token_idx, source_idx in enumerate(selected)
    ]


def fragment_smiles(engine: fragmentation.FragmentEngine, frag: int) -> str:
    atom_indices, _ = engine.get_present_atoms(int(frag))
    if not atom_indices:
        return ""
    return Chem.MolFragmentToSmiles(
        engine.mol,
        atomsToUse=atom_indices,
        canonical=True,
        isomericSmiles=True,
        kekuleSmiles=True,
    )


def init_worker(config: dict[str, Any]) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config
    RDLogger.DisableLog("rdApp.*")


def process_record(payload: tuple[int, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    sample_index, metadata, raw = payload
    started = time.perf_counter()
    identifier = str(metadata["identifier"])
    try:
        adduct = str(metadata.get("adduct") or raw.get("adduct") or "")
        if adduct not in common.ion2mass:
            raise ValueError(f"Unsupported adduct: {adduct!r}")
        smiles = str(metadata.get("canonical_isomeric_smiles") or raw.get("smiles") or "")
        if not smiles:
            raise ValueError("Missing SMILES")

        peaks = select_dreams_peaks(
            raw["mzs"], raw["intensities"], int(_WORKER_CONFIG["n_highest_peaks"])
        )
        if not peaks:
            raise ValueError("No spectrum peaks")

        engine = fragmentation.FragmentEngine(
            mol_str=smiles,
            max_tree_depth=int(_WORKER_CONFIG["max_tree_depth"]),
            max_broken_bonds=int(_WORKER_CONFIG["max_broken_bonds"]),
        )
        engine.generate_fragments()
        frag_hashes, _frag_ints, shift_indices, masses, scores = engine.get_frag_masses()
        if len(masses) == 0:
            raise ValueError("MAGMa generated no fragment masses")

        mz_ar = np.asarray([peak["mz"] for peak in peaks], dtype=np.float64)
        adjusted_m = mz_ar - float(common.ion2mass[adduct])
        if common.is_positive_adduct(adduct):
            adjusted_e_m = mz_ar + float(common.ELECTRON_MASS)
        else:
            adjusted_e_m = mz_ar - float(common.ELECTRON_MASS)
        masses = masses.astype(np.float64, copy=False)
        ppm_diffs = np.minimum(
            np.abs(masses[None, :] - adjusted_m[:, None]) / masses[None, :] * 1e6,
            np.abs(masses[None, :] - adjusted_e_m[:, None]) / masses[None, :] * 1e6,
        )

        min_ppms = ppm_diffs.min(axis=1)
        tolerance = float(_WORKER_CONFIG["ppm_tolerance"])
        frag_smiles_cache: dict[int, str] = {}
        peak_labels: list[dict[str, Any]] = []
        for peak_idx, peak in enumerate(peaks):
            min_ppm = float(min_ppms[peak_idx])
            assigned = min_ppm < tolerance
            matches: list[dict[str, Any]] = []
            if assigned:
                tied = np.flatnonzero(ppm_diffs[peak_idx] == min_ppms[peak_idx])
                min_score = float(scores[tied].min())
                tied = tied[scores[tied] == min_score]
                seen: set[tuple[str, int]] = set()
                for mass_idx in tied.tolist():
                    frag_hash = str(frag_hashes[mass_idx])
                    shift_index = int(shift_indices[mass_idx])
                    dedup_key = (frag_hash, shift_index)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    entry = engine.frag_to_entry[frag_hash]
                    # ms-pred converts the parallel fragment-int list to NumPy.
                    # At exactly 64 atoms that array can become float64 and lose
                    # bitmask precision, so use the authoritative Python int.
                    frag_int = int(entry["frag"])
                    atom_indices, _ = engine.get_present_atoms(frag_int)
                    if frag_int not in frag_smiles_cache:
                        frag_smiles_cache[frag_int] = fragment_smiles(engine, frag_int)
                    h_shift = int(engine.shift_buckets[shift_index])
                    matches.append(
                        {
                            "frag_hash": frag_hash,
                            "atom_indices": [int(value) for value in atom_indices],
                            "fragment_smiles": frag_smiles_cache[frag_int],
                            "base_formula": str(entry["form"]),
                            "h_shift": h_shift,
                            "h_shift_index": shift_index,
                            "fragment_mass": float(masses[mass_idx]),
                            "break_count": int(entry["max_broken"]),
                            "tree_depth": int(entry["tree_depth"]),
                            "break_score": float(scores[mass_idx]),
                        }
                    )
            peak_labels.append(
                {
                    **peak,
                    "assigned": assigned,
                    "min_ppm": min_ppm,
                    "matches": matches,
                }
            )

        elapsed = time.perf_counter() - started
        return {
            "ok": True,
            "sample_index": sample_index,
            "record": {
                "identifier": identifier,
                "sample_index": sample_index,
                "source_row_index": int(metadata.get("row_index", -1)),
                "source_metadata": str(_WORKER_CONFIG["metadata"]),
                "dataset": "MassSpecGym v1.5",
                "protocol": str(_WORKER_CONFIG["protocol_label"]),
                "canonical_isomeric_smiles": smiles,
                "magma_canonical_smiles": str(engine.smiles),
                "adduct": adduct,
                "precursor_mz": raw.get("precursor_mz") or metadata.get("precursor_mz"),
                "num_original_peaks": len(raw["mzs"]),
                "num_selected_peaks": len(peaks),
                "num_atoms": int(engine.natoms),
                "num_generated_fragments": len(engine.frag_to_entry),
                "num_generated_mass_variants": len(masses),
                "runtime_seconds": elapsed,
                "peaks": peak_labels,
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "sample_index": sample_index,
            "failure": {
                "identifier": identifier,
                "sample_index": sample_index,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "runtime_seconds": time.perf_counter() - started,
            },
        }


def quantiles(values: list[float | int]) -> dict[str, float | int]:
    if not values:
        return {}
    ar = np.asarray(values, dtype=np.float64)
    return {
        "min": float(ar.min()),
        "p25": float(np.quantile(ar, 0.25)),
        "median": float(np.quantile(ar, 0.5)),
        "mean": float(ar.mean()),
        "p75": float(np.quantile(ar, 0.75)),
        "p90": float(np.quantile(ar, 0.9)),
        "p95": float(np.quantile(ar, 0.95)),
        "max": float(ar.max()),
    }


def summarize(
    records: list[dict[str, Any]], failures: list[dict[str, Any]], args: argparse.Namespace, wall: float
) -> dict[str, Any]:
    selected_peaks = 0
    assigned_peaks = 0
    total_intensity = 0.0
    assigned_intensity = 0.0
    matches_per_assigned: list[int] = []
    assigned_ppms: list[float] = []
    assigned_breaks: list[int] = []
    min_break_per_peak: list[int] = []
    coverage_per_spectrum: list[float] = []
    top10_total = 0
    top10_assigned = 0
    intensity_bands: dict[str, list[int]] = {
        "relative_intensity_ge_0.5": [0, 0],
        "relative_intensity_0.1_to_0.5": [0, 0],
        "relative_intensity_lt_0.1": [0, 0],
    }
    adduct_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    for record in records:
        peaks = record["peaks"]
        num_assigned = sum(bool(peak["assigned"]) for peak in peaks)
        coverage_per_spectrum.append(num_assigned / len(peaks) if peaks else 0.0)
        adduct = str(record["adduct"])
        adduct_stats[adduct][0] += 1
        adduct_stats[adduct][1] += len(peaks)
        adduct_stats[adduct][2] += num_assigned

        top10_indices = set(
            sorted(range(len(peaks)), key=lambda idx: float(peaks[idx]["intensity"]), reverse=True)[:10]
        )
        for peak_idx, peak in enumerate(peaks):
            selected_peaks += 1
            intensity = float(peak["relative_intensity"])
            total_intensity += intensity
            if intensity >= 0.5:
                band = "relative_intensity_ge_0.5"
            elif intensity >= 0.1:
                band = "relative_intensity_0.1_to_0.5"
            else:
                band = "relative_intensity_lt_0.1"
            intensity_bands[band][0] += 1
            if peak_idx in top10_indices:
                top10_total += 1
            if not peak["assigned"]:
                continue
            assigned_peaks += 1
            assigned_intensity += intensity
            intensity_bands[band][1] += 1
            if peak_idx in top10_indices:
                top10_assigned += 1
            matches = peak["matches"]
            matches_per_assigned.append(len(matches))
            assigned_ppms.append(float(peak["min_ppm"]))
            break_counts = [int(match["break_count"]) for match in matches]
            assigned_breaks.extend(break_counts)
            if break_counts:
                min_break_per_peak.append(min(break_counts))

    ambiguity_counts = Counter(matches_per_assigned)
    break_counts = Counter(assigned_breaks)
    min_break_counts = Counter(min_break_per_peak)
    runtime_values = [float(record["runtime_seconds"]) for record in records]
    atom_values = [int(record["num_atoms"]) for record in records]
    generated_fragments = [int(record["num_generated_fragments"]) for record in records]
    generated_variants = [int(record["num_generated_mass_variants"]) for record in records]

    return {
        "dataset": "MassSpecGym v1.5 only",
        "split_protocol": args.protocol_label,
        "source_metadata": str(args.metadata),
        "source_v15_tsv": None if args.topk_export is not None else str(args.v15_tsv),
        "source_topk_export": None if args.topk_export is None else str(args.topk_export),
        "sample_size_requested": args.sample_size,
        "sample_seed": args.seed,
        "spectra_succeeded": len(records),
        "spectra_failed": len(failures),
        "failure_types": dict(sorted(Counter(item["error_type"] for item in failures).items())),
        "magma": {
            "ppm_tolerance": args.ppm_tolerance,
            "max_tree_depth": args.max_tree_depth,
            "max_broken_bonds": args.max_broken_bonds,
            "n_highest_peaks": args.n_highest_peaks,
            "peak_order": "DreaMS-compatible highest-intensity selection with original order restored",
        },
        "peak_coverage": {
            "selected_peaks": selected_peaks,
            "assigned_peaks": assigned_peaks,
            "coverage": assigned_peaks / selected_peaks if selected_peaks else 0.0,
            "intensity_weighted_coverage": (
                assigned_intensity / total_intensity if total_intensity > 0 else 0.0
            ),
            "top10_intensity_peaks": top10_total,
            "top10_assigned": top10_assigned,
            "top10_coverage": top10_assigned / top10_total if top10_total else 0.0,
            "per_spectrum_coverage": quantiles(coverage_per_spectrum),
            "spectra_with_zero_assigned_peaks": sum(value == 0 for value in coverage_per_spectrum),
            "spectra_with_at_least_half_peaks_assigned": sum(
                value >= 0.5 for value in coverage_per_spectrum
            ),
            "by_relative_intensity": {
                key: {
                    "peaks": values[0],
                    "assigned": values[1],
                    "coverage": values[1] / values[0] if values[0] else 0.0,
                }
                for key, values in intensity_bands.items()
            },
            "by_adduct": {
                key: {
                    "spectra": values[0],
                    "peaks": values[1],
                    "assigned": values[2],
                    "coverage": values[2] / values[1] if values[1] else 0.0,
                }
                for key, values in sorted(adduct_stats.items())
            },
        },
        "ambiguity": {
            "matches_per_assigned_peak": quantiles(matches_per_assigned),
            "match_count_histogram": {
                str(key): value for key, value in sorted(ambiguity_counts.items())
            },
            "assigned_peaks_with_multiple_matches": sum(value > 1 for value in matches_per_assigned),
            "multiple_match_fraction": (
                sum(value > 1 for value in matches_per_assigned) / len(matches_per_assigned)
                if matches_per_assigned
                else 0.0
            ),
        },
        "fragmentation": {
            "break_count_histogram_all_matches": {
                str(key): value for key, value in sorted(break_counts.items())
            },
            "min_break_count_histogram_per_assigned_peak": {
                str(key): value for key, value in sorted(min_break_counts.items())
            },
            "assigned_peak_min_ppm": quantiles(assigned_ppms),
            "num_atoms_per_molecule": quantiles(atom_values),
            "generated_fragments_per_molecule": quantiles(generated_fragments),
            "generated_mass_variants_per_molecule": quantiles(generated_variants),
        },
        "runtime": {
            "workers": args.workers,
            "wall_seconds": wall,
            "spectra_per_second": len(records) / wall if wall > 0 else 0.0,
            "per_spectrum_seconds": quantiles(runtime_values),
            "estimated_20k_wall_hours_at_observed_throughput": (
                20000 / (len(records) / wall) / 3600 if records and wall > 0 else None
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_records = load_jsonl(args.metadata)
    if args.sample_size > len(source_records):
        raise ValueError(
            f"Requested {args.sample_size} records but metadata contains only {len(source_records)}"
        )
    rng = random.Random(args.seed)
    sample_positions = sorted(rng.sample(range(len(source_records)), args.sample_size))
    sampled_metadata = [source_records[position] for position in sample_positions]
    for sample_index, (source_position, record) in enumerate(zip(sample_positions, sampled_metadata)):
        record["magma_audit_sample_index"] = sample_index
        record["magma_audit_source_position"] = source_position

    identifiers = {str(record["identifier"]) for record in sampled_metadata}
    raw_rows = (
        load_topk_rows(args.topk_export, identifiers)
        if args.topk_export is not None
        else load_v15_rows(args.v15_tsv, identifiers)
    )
    payloads = [
        (sample_index, metadata, raw_rows[str(metadata["identifier"])])
        for sample_index, metadata in enumerate(sampled_metadata)
    ]
    write_jsonl(args.out_dir / "sample.metadata.jsonl", sampled_metadata)

    config = {
        "metadata": str(args.metadata),
        "n_highest_peaks": args.n_highest_peaks,
        "protocol_label": args.protocol_label,
        "ppm_tolerance": args.ppm_tolerance,
        "max_tree_depth": args.max_tree_depth,
        "max_broken_bonds": args.max_broken_bonds,
    }
    started = time.perf_counter()
    if args.workers <= 1:
        init_worker(config)
        results = [process_record(payload) for payload in tqdm(payloads, desc="MAGMa labels")]
    else:
        with Pool(processes=args.workers, initializer=init_worker, initargs=(config,)) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(process_record, payloads, chunksize=1),
                    total=len(payloads),
                    desc="MAGMa labels",
                )
            )
    wall = time.perf_counter() - started

    results.sort(key=lambda item: int(item["sample_index"]))
    labels = [item["record"] for item in results if item["ok"]]
    failures = [item["failure"] for item in results if not item["ok"]]
    write_jsonl(args.out_dir / "labels.jsonl", labels)
    write_jsonl(args.out_dir / "failures.jsonl", failures)
    summary = summarize(labels, failures, args, wall)
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
