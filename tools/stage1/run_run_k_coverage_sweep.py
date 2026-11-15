#!/usr/bin/env python3
"""Drive the Run K molecule-coverage sweep over the nested training subsets.

Every arm reuses the exact Run K recipe (count Morgan r3 + Atom Pair + RDKit path
+ MACCS, block L2, mass cap256, candidate CE, random negatives, frozen
temperature) and differs only in which training rows it sees.

Epochs are scaled so each arm takes roughly the same number of gradient steps as
the reference 20k run, and the trainer's own best-epoch-on-val selection guards
the small arms against the overfitting that the extra passes would otherwise
cause. Without this the smaller arms would also get proportionally less
optimisation, and undertraining would be indistinguishable from thin coverage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
SPLIT = ROOT / "outputs/msalign_formula_group_split_v15"
ABLATION = ROOT / "outputs/msalign_molecule_ablation"

REFERENCE_ROWS = 20000
REFERENCE_EPOCHS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        default=ROOT / "outputs/run_k_coverage_sweep/subsets/manifest.json",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/run_k_coverage_sweep/runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Filter arms by seed.")
    parser.add_argument("--families", nargs="+", default=None, help="Filter arms by family.")
    parser.add_argument(
        "--include-full",
        action="store_true",
        help="Also train the untouched 20k sample under the same driver.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--train-seed", type=int, default=20260621)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def epochs_for(rows: int, batch_size: int, max_epochs: int) -> int:
    reference_steps = (REFERENCE_ROWS / batch_size) * REFERENCE_EPOCHS
    steps_per_epoch = max(rows / batch_size, 1.0)
    return int(min(max_epochs, max(REFERENCE_EPOCHS, round(reference_steps / steps_per_epoch))))


def build_command(
    arm: str, metadata: Path, dreams: Path, epochs: int, out_dir: Path, args: argparse.Namespace
) -> list[str]:
    return [
        PYTHON,
        str(ROOT / "tools/stage1/train_msalign_projection_smoke.py"),
        "--train-metadata", str(metadata),
        "--train-dreams", str(dreams),
        "--val-metadata", str(SPLIT / "val.metadata.jsonl"),
        "--val-dreams", str(ROOT / "outputs/msalign_dreams_no_head_formula_group_masscand/val.no_head.npy"),
        "--chemberta-cache", str(ABLATION / "chemberta_mtr_morgan_formula_group_20k_val_test_mass_cap256.pt"),
        "--fingerprint-cache", str(ABLATION / "sparse_fingerprints_formula_group_20k_val_test_cap256/manifest.json"),
        "--additional-fingerprint-cache", str(ABLATION / "sparse_count_morgan_r3_formula_group_20k_val_test_cap256/manifest.json"),
        "--candidate-sidecar", str(ROOT / "outputs/msalign_mtr_probe/chemberta_mtr_formula_group_20k_val_test_mass_cap256.candidates.jsonl"),
        "--candidates-json", str(ROOT / "data/data/molecules/MassSpecGym1.5_retrieval_candidates_mass.json"),
        "--mol-pooling", "count_morgan_r3_atom_pair_path_maccs",
        "--mol-block-normalization", "l2",
        "--candidate-pool-limit", "256",
        "--candidate-negatives", "127",
        "--negative-sampling", "random",
        "--loss-mode", "candidate_ce",
        "--epochs", str(epochs),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", "64",
        "--lr", "1e-4",
        "--weight-decay", "1e-4",
        "--head-type", "mlp",
        "--proj-dim", "1024",
        "--hidden-dim", "2048",
        "--dropout", "0.2",
        "--temperature", "0.07",
        "--freeze-temperature",
        "--seed", str(args.train_seed),
        "--materialize-dataset-features",
        "--materialize-chunk-size", "4096",
        "--device", args.device,
        "--checkpoint", str(out_dir / f"{arm}.pt"),
        "--out-json", str(out_dir / f"{arm}_summary.json"),
    ]


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.subset_manifest.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plan = []
    for arm in manifest["arms"]:
        if args.seeds is not None and arm["seed"] not in args.seeds:
            continue
        if args.families is not None and arm["family"] not in args.families:
            continue
        plan.append(
            {
                "arm": arm["arm"],
                "family": arm["family"],
                "seed": arm["seed"],
                "rows": arm["rows"],
                "unique_molecules": arm["unique_molecules"],
                "metadata": Path(arm["metadata"]),
                "dreams": Path(arm["dreams"]),
            }
        )
    if args.include_full:
        plan.append(
            {
                "arm": f"full_m{manifest['source_unique_molecules']}",
                "family": "narrow",
                "seed": -1,
                "rows": manifest["source_rows"],
                "unique_molecules": manifest["source_unique_molecules"],
                "metadata": Path(manifest["source_metadata"]),
                "dreams": Path(manifest["source_dreams"]),
            }
        )

    plan.sort(key=lambda item: item["rows"])
    records = []
    for index, item in enumerate(plan, start=1):
        epochs = epochs_for(item["rows"], args.batch_size, args.max_epochs)
        command = build_command(
            item["arm"], item["metadata"], item["dreams"], epochs, args.out_dir, args
        )
        label = (
            f"[{index}/{len(plan)}] {item['arm']:28s} rows={item['rows']:6d}"
            f" molecules={item['unique_molecules']:5d} epochs={epochs:3d}"
        )
        print(label, flush=True)
        if args.dry_run:
            continue
        log_path = args.out_dir / f"{item['arm']}.log"
        started = time.time()
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        elapsed = time.time() - started
        if result.returncode != 0:
            raise SystemExit(f"{item['arm']} failed with code {result.returncode}; see {log_path}")
        summary = json.loads((args.out_dir / f"{item['arm']}_summary.json").read_text(encoding="utf-8"))
        recall = summary["best_val"]["recall@1"]
        print(f"    -> recall@1={recall:7.4f}  best_epoch={summary['best_epoch']}  {elapsed:.0f}s", flush=True)
        records.append({**{k: str(v) for k, v in item.items()}, "epochs": epochs, "recall@1": recall})

    if not args.dry_run:
        (args.out_dir / "sweep_index.json").write_text(
            json.dumps({"batch_size": args.batch_size, "runs": records}, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
