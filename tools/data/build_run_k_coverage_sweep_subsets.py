#!/usr/bin/env python3
"""Build nested training subsets for the Run K molecule-coverage sweep.

The train-proximity audit found that Run K's molecule-level accuracy rises from
3.6% to 71.7% with similarity to the nearest training positive, independently of
molecule size and of candidate-pool crowding. That is correlational: molecules
far from the 8,671 training positives could be failing because coverage is thin,
or because they are intrinsically harder in a way that also makes them unusual.

Subsetting the existing 20k sample downward turns the correlation into a causal
test without needing any new cache. Two arm families are emitted:

  narrow   nested molecule subsets, keeping every spectrum of the kept
           molecules. Molecule count and row count fall together.
  wide     the same row budget as the matched narrow arm, but spectra are taken
           round-robin across all molecules so diversity is maximised. This
           isolates unique-molecule count from row count and gradient steps,
           which the narrow family alone cannot separate.

Arms are nested within a seed: the molecule permutation is drawn once, so the
1/8 arm's molecules are a subset of the 1/4 arm's, and so on. Metadata rows and
DreaMS rows are emitted in the same order because ProjectionDataset pairs
records[i] with dreams[i] positionally.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = (
    ROOT / "outputs/msalign_formula_group_split_v15/train_sample20k_seed20260623.metadata.jsonl"
)
DEFAULT_DREAMS = (
    ROOT / "outputs/msalign_dreams_no_head_formula_group_masscand/train_sample20k_seed20260623.no_head.npy"
)
DEFAULT_OUT = ROOT / "outputs/run_k_coverage_sweep/subsets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--train-dreams", type=Path, default=DEFAULT_DREAMS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--denominators",
        type=int,
        nargs="+",
        default=[8, 4, 2],
        help="Molecule-count fractions to emit; the full arm is the existing 20k sample.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--skip-wide",
        action="store_true",
        help="Emit only the nested narrow arms.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def round_robin_rows(
    molecule_rows: dict[str, list[int]], order: list[str], budget: int, rng: np.random.Generator
) -> list[int]:
    """Take spectra one per molecule per pass, maximising molecule diversity."""
    shuffled = {key: list(rng.permutation(molecule_rows[key])) for key in order}
    selected: list[int] = []
    depth = 0
    max_depth = max(len(rows) for rows in molecule_rows.values())
    while len(selected) < budget and depth < max_depth:
        for key in order:
            if len(selected) >= budget:
                break
            rows = shuffled[key]
            if depth < len(rows):
                selected.append(int(rows[depth]))
        depth += 1
    return selected


def write_arm(
    name: str,
    rows: list[int],
    records: list[dict[str, Any]],
    dreams: np.ndarray,
    out_dir: Path,
) -> dict[str, Any]:
    rows = sorted(rows)
    metadata_path = out_dir / f"{name}.metadata.jsonl"
    dreams_path = out_dir / f"{name}.no_head.npy"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(records[row], ensure_ascii=False) + "\n")
    np.save(dreams_path, dreams[rows])
    molecules = {str(records[row].get("inchikey")) for row in rows}
    return {
        "arm": name,
        "metadata": str(metadata_path),
        "dreams": str(dreams_path),
        "rows": len(rows),
        "unique_molecules": len(molecules),
        "spectra_per_molecule": round(len(rows) / max(len(molecules), 1), 4),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.train_metadata)
    dreams = np.load(args.train_dreams)
    if len(records) != dreams.shape[0]:
        raise SystemExit(f"Row mismatch: {len(records)} records vs {dreams.shape[0]} embeddings")

    molecule_rows: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        key = record.get("inchikey")
        if not key:
            raise SystemExit(f"Record {index} has no inchikey; cannot define molecule coverage")
        molecule_rows[str(key)].append(index)
    all_molecules = sorted(molecule_rows)

    arms: list[dict[str, Any]] = []
    for seed in args.seeds:
        rng = np.random.default_rng(20260811 + seed)
        order = [str(key) for key in rng.permutation(all_molecules)]
        for denominator in sorted(args.denominators, reverse=True):
            count = max(1, len(order) // denominator)
            chosen = order[:count]
            rows = [row for key in chosen for row in molecule_rows[key]]
            arms.append(
                write_arm(f"narrow_m{count}_seed{seed}", rows, records, dreams, args.out_dir)
                | {"family": "narrow", "seed": seed, "denominator": denominator}
            )
            if not args.skip_wide:
                wide_rows = round_robin_rows(molecule_rows, order, len(rows), rng)
                arms.append(
                    write_arm(f"wide_n{len(rows)}_seed{seed}", wide_rows, records, dreams, args.out_dir)
                    | {
                        "family": "wide",
                        "seed": seed,
                        "denominator": denominator,
                        "row_matched_to": f"narrow_m{count}_seed{seed}",
                    }
                )

    manifest = {
        "source_metadata": str(args.train_metadata),
        "source_dreams": str(args.train_dreams),
        "source_rows": len(records),
        "source_unique_molecules": len(all_molecules),
        "nesting": "molecule permutation drawn once per seed; smaller arms are subsets of larger",
        "arms": arms,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"source: {len(records)} rows, {len(all_molecules)} unique molecules")
    for arm in arms:
        print(
            f"  {arm['arm']:28s} rows={arm['rows']:6d}  molecules={arm['unique_molecules']:5d}"
            f"  spectra/mol={arm['spectra_per_molecule']:.2f}"
        )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
