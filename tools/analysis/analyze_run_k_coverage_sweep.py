#!/usr/bin/env python3
"""Read out the Run K coverage sweep against training proximity.

Overall recall@1 cannot answer the question the sweep was built for. Coverage
could raise accuracy either by densifying regions the model already handles or
by extending its reach into chemistry it previously missed entirely, and those
imply different next steps. Two stratifications separate them:

  fixed axis   every arm is scored on deciles of proximity to the full 8,671
               training molecules. Bins hold the same val molecules across arms,
               so an arm's gain can be attributed to a difficulty level.
  self axis    every arm is scored on fixed similarity bands relative to its own
               training molecules. If accuracy is governed by proximity, all arms
               collapse onto one curve here and coverage acts only by moving val
               molecules into higher bands. A vertical shift instead means
               coverage buys something beyond proximity.

Proximity is the cosine to the nearest training positive in the raw Run K
four-block space, which does not depend on any trained checkpoint and is
therefore comparable across arms. Metrics are molecule-weighted because
correctness is strongly clustered by molecule (ICC(1) = 0.756), which makes
query-weighted recall track how often each molecule was measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audits.audit_train_proximity_vs_reachability import (  # noqa: E402
    block_normalize,
    featurize,
    load_jsonl,
    unique_molecules,
)

DEFAULT_SUBSETS = ROOT / "outputs/run_k_coverage_sweep/subsets/manifest.json"
DEFAULT_RANKS = ROOT / "outputs/run_k_coverage_sweep/ranks"
DEFAULT_VAL = ROOT / "outputs/msalign_formula_group_split_v15/val.metadata.jsonl"
DEFAULT_FULL_TRAIN = (
    ROOT / "outputs/msalign_formula_group_split_v15/train_sample20k_seed20260623.metadata.jsonl"
)
DEFAULT_OUT = ROOT / "outputs/run_k_coverage_sweep/coverage_sweep_analysis.json"

SELF_BANDS = (0.70, 0.78, 0.84, 0.90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-manifest", type=Path, default=DEFAULT_SUBSETS)
    parser.add_argument("--ranks-dir", type=Path, default=DEFAULT_RANKS)
    parser.add_argument("--val-metadata", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--full-train-metadata", type=Path, default=DEFAULT_FULL_TRAIN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deciles", type=int, default=10)
    parser.add_argument("--min-cell", type=int, default=15)
    return parser.parse_args()


def molecule_accuracy(ranks_path: Path) -> dict[str, float]:
    hits: dict[str, list[int]] = defaultdict(list)
    for row in load_jsonl(ranks_path):
        hits[row["gt_group_key"]].append(int(row["rank"]) == 1)
    return {key: float(np.mean(value)) for key, value in hits.items()}


def binned(
    values: np.ndarray, accuracy: np.ndarray, assignment: np.ndarray, bins: int, min_cell: int
) -> list[dict[str, Any]]:
    rows = []
    for index in range(bins):
        mask = assignment == index
        if mask.sum() < min_cell:
            rows.append({"bin": index, "molecules": int(mask.sum()), "molecule_weighted_r1": None})
            continue
        rows.append(
            {
                "bin": index,
                "molecules": int(mask.sum()),
                "mean_proximity": round(float(values[mask].mean()), 4),
                "molecule_weighted_r1": round(100 * float(accuracy[mask].mean()), 3),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    manifest = json.loads(args.subset_manifest.read_text(encoding="utf-8"))

    val_molecules = unique_molecules(args.val_metadata)
    full_train = unique_molecules(args.full_train_metadata)
    train_keys = sorted(full_train)
    train_position = {key: index for index, key in enumerate(train_keys)}

    val_keys = sorted(val_molecules)
    with torch.inference_mode():
        val_matrix = block_normalize(featurize([val_molecules[key] for key in val_keys])).to(device)
        train_matrix = block_normalize(featurize([full_train[key] for key in train_keys])).to(device)
        full_similarity = val_matrix @ train_matrix.T

    reference_proximity = full_similarity.max(dim=-1).values.cpu().numpy()
    decile_edges = np.quantile(reference_proximity, np.linspace(0.0, 1.0, args.deciles + 1))[1:-1]
    fixed_assignment = np.digitize(reference_proximity, decile_edges)

    arms = [
        {
            "arm": arm["arm"],
            "family": arm["family"],
            "seed": arm["seed"],
            "rows": arm["rows"],
            "unique_molecules": arm["unique_molecules"],
            "metadata": Path(arm["metadata"]),
            "row_matched_to": arm.get("row_matched_to"),
        }
        for arm in manifest["arms"]
    ]
    arms.append(
        {
            "arm": f"full_m{manifest['source_unique_molecules']}",
            "family": "narrow",
            "seed": -1,
            "rows": manifest["source_rows"],
            "unique_molecules": manifest["source_unique_molecules"],
            "metadata": args.full_train_metadata,
            "row_matched_to": None,
        }
    )

    results = []
    for arm in arms:
        ranks_path = args.ranks_dir / f"{arm['arm']}.ranks.jsonl"
        if not ranks_path.exists():
            continue
        accuracy_map = molecule_accuracy(ranks_path)
        present = np.array([key in accuracy_map for key in val_keys])
        accuracy = np.array([accuracy_map.get(key, 0.0) for key in val_keys])

        arm_molecules = sorted(unique_molecules(arm["metadata"]))
        columns = torch.tensor(
            [train_position[key] for key in arm_molecules if key in train_position],
            dtype=torch.long,
            device=device,
        )
        self_proximity = full_similarity[:, columns].max(dim=-1).values.cpu().numpy()
        self_assignment = np.digitize(self_proximity, np.asarray(SELF_BANDS))

        query_hits, query_total = 0, 0
        for row in load_jsonl(ranks_path):
            query_total += 1
            query_hits += int(row["rank"]) == 1

        results.append(
            {
                "arm": arm["arm"],
                "family": arm["family"],
                "seed": arm["seed"],
                "rows": arm["rows"],
                "unique_molecules": arm["unique_molecules"],
                "row_matched_to": arm["row_matched_to"],
                "query_weighted_r1": round(100 * query_hits / max(query_total, 1), 3),
                "molecule_weighted_r1": round(100 * float(accuracy[present].mean()), 3),
                "mean_self_proximity": round(float(self_proximity.mean()), 4),
                "by_fixed_proximity_decile": binned(
                    reference_proximity[present],
                    accuracy[present],
                    fixed_assignment[present],
                    args.deciles,
                    args.min_cell,
                ),
                "by_self_proximity_band": binned(
                    self_proximity[present],
                    accuracy[present],
                    self_assignment[present],
                    len(SELF_BANDS) + 1,
                    args.min_cell,
                ),
            }
        )

    summary = {
        "protocol": {
            "proximity": "cosine to nearest training positive, raw Run K four-block space",
            "fixed_axis": "deciles of proximity to the full 8,671-molecule training set",
            "self_axis_edges": list(SELF_BANDS),
            "metric": "molecule-weighted recall@1 over val connectivity groups",
            "val_molecules": len(val_keys),
        },
        "fixed_decile_edges": [round(float(edge), 4) for edge in decile_edges],
        "runs": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"analyzed {len(results)} arms -> {args.out}")


if __name__ == "__main__":
    main()
