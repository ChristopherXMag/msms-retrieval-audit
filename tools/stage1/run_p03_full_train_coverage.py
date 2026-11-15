#!/usr/bin/env python3
"""Run one locked P0.3 coverage-scaling arm and its val/test evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
TRAINER = ROOT / "tools/stage1/train_msalign_projection_smoke.py"
EVALUATOR = ROOT / "tools/stage1/evaluate_msalign_projection_checkpoint.py"
OUTPUT_ROOT = ROOT / "outputs/run_k_full_train/experiments"

TRAIN_METADATA_FULL = ROOT / "outputs/msalign_formula_group_split_v15/train.metadata.jsonl"
TRAIN_DREAMS_FULL = ROOT / "outputs/msalign_dreams_no_head_formula_group_masscand/train.no_head.npy"
TRAIN_METADATA_20K = ROOT / "outputs/msalign_formula_group_split_v15/train_sample20k_seed20260623.metadata.jsonl"
TRAIN_DREAMS_20K = ROOT / "outputs/msalign_dreams_no_head_formula_group_masscand/train_sample20k_seed20260623.no_head.npy"
VAL_METADATA = ROOT / "outputs/msalign_formula_group_split_v15/val.metadata.jsonl"
VAL_DREAMS = ROOT / "outputs/msalign_dreams_no_head_formula_group_masscand/val.no_head.npy"
TEST_METADATA = ROOT / "outputs/msalign_formula_group_split_v15/test.metadata.jsonl"
TEST_DREAMS = ROOT / "outputs/msalign_dreams_no_head_formula_group_masscand/test.no_head.npy"
SPARSE_MANIFEST = ROOT / "outputs/run_k_full_train/full_sparse/manifest.json"
CANDIDATE_MANIFEST = (
    ROOT / "outputs/run_k_full_train/candidate_index_full_v15_mass_cap256_full_order/manifest.json"
)

SEEDS = (20260621, 20260622, 20260623)
ARM_CONFIG = {
    "control": {"epochs": 5, "sampling": "row_shuffle", "rows": 20_000, "groups": 8_671},
    "balanced": {"epochs": 4, "sampling": "molecule_balanced", "rows": 184_847, "groups": 22_934},
    "all_row": {"epochs": 1, "sampling": "row_shuffle", "rows": 184_847, "groups": 22_934},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARM_CONFIG), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--phase", choices=("train", "val", "test"), required=True)
    parser.add_argument(
        "--candidate-negatives",
        type=int,
        choices=(127, 255),
        default=127,
        help="127 is the locked P0.3 baseline; 255 uses every non-GT candidate in cap256.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def count_metadata(path: Path) -> tuple[int, int]:
    rows = 0
    groups: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rows += 1
            key = str(record.get("inchikey") or "").split("-", 1)[0]
            groups.add(key)
    return rows, len(groups)


def validate_assets(arm: str) -> None:
    required = [
        VAL_METADATA,
        VAL_DREAMS,
        TEST_METADATA,
        TEST_DREAMS,
        SPARSE_MANIFEST,
        CANDIDATE_MANIFEST,
    ]
    train_metadata = TRAIN_METADATA_20K if arm == "control" else TRAIN_METADATA_FULL
    train_dreams = TRAIN_DREAMS_20K if arm == "control" else TRAIN_DREAMS_FULL
    required.extend((train_metadata, train_dreams))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"P0.3 assets are missing: {missing}")

    config = ARM_CONFIG[arm]
    observed = count_metadata(train_metadata)
    expected = (int(config["rows"]), int(config["groups"]))
    if observed != expected:
        raise ValueError(f"{arm} train row/connectivity-group mismatch: {observed} != {expected}")
    dreams = np.load(train_dreams, mmap_mode="r")
    if tuple(dreams.shape) != (expected[0], 1024):
        raise ValueError(f"{arm} DreaMS shape mismatch: {dreams.shape} != {(expected[0], 1024)}")

    sparse = json.loads(SPARSE_MANIFEST.read_text(encoding="utf-8"))
    candidates = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    if int(sparse["rows"]) != 7_220_331 or int(candidates["molecule_rows"]) != 7_220_331:
        raise ValueError("P0.3 molecule row count is not the locked 7,220,331")
    if sparse.get("format") != "glmr_sparse_features_indexed_v1":
        raise ValueError("P0.3 requires the lossless indexed sparse backend")
    if sparse.get("row_order") != "full_v1.5":
        raise ValueError("P0.3 molecule rows are not in the locked full-v1.5 logical order")
    if sparse["smiles_sha256"] != candidates["molecule_smiles_sha256"]:
        raise ValueError("Sparse and candidate-index row digests differ")


def paths(arm: str, seed: int, candidate_negatives: int = 127) -> dict[str, Path]:
    arm_dir = arm if candidate_negatives == 127 else f"{arm}_neg{candidate_negatives}"
    run_dir = OUTPUT_ROOT / arm_dir / f"seed{seed}"
    return {
        "run_dir": run_dir,
        "summary": run_dir / "train_summary.json",
        "checkpoint": run_dir / "checkpoint.pt",
        "val": run_dir / "val_ranks.json",
        "test": run_dir / "test_ranks.json",
    }


def train_command(
    arm: str,
    seed: int,
    device: str,
    output: dict[str, Path],
    candidate_negatives: int = 127,
) -> list[str]:
    config = ARM_CONFIG[arm]
    train_metadata = TRAIN_METADATA_20K if arm == "control" else TRAIN_METADATA_FULL
    train_dreams = TRAIN_DREAMS_20K if arm == "control" else TRAIN_DREAMS_FULL
    return [
        str(PYTHON),
        str(TRAINER),
        "--train-metadata", str(train_metadata),
        "--val-metadata", str(VAL_METADATA),
        "--train-dreams", str(train_dreams),
        "--val-dreams", str(VAL_DREAMS),
        "--fingerprint-cache", str(SPARSE_MANIFEST),
        "--candidate-index-manifest", str(CANDIDATE_MANIFEST),
        "--mol-pooling", "count_morgan_r3_atom_pair_path_maccs",
        "--mol-block-normalization", "l2",
        "--mol-head-type", "shared_mlp",
        "--candidate-pool-limit", "256",
        "--candidate-negatives", str(candidate_negatives),
        "--negative-sampling", "random",
        "--loss-mode", "candidate_ce",
        "--epochs", str(config["epochs"]),
        "--batch-size", "128",
        "--eval-batch-size", "64",
        "--lr", "0.0001",
        "--weight-decay", "0.0001",
        "--head-type", "mlp",
        "--hidden-dim", "2048",
        "--proj-dim", "1024",
        "--dropout", "0.2",
        "--temperature", "0.07",
        "--freeze-temperature",
        "--seed", str(seed),
        "--train-sampling", str(config["sampling"]),
        "--device", device,
        "--out-json", str(output["summary"]),
        "--checkpoint", str(output["checkpoint"]),
    ]


def evaluation_command(
    arm: str, phase: str, device: str, output: dict[str, Path]
) -> list[str]:
    metadata = VAL_METADATA if phase == "val" else TEST_METADATA
    dreams = VAL_DREAMS if phase == "val" else TEST_DREAMS
    # The all-row protocol has exactly one epoch and is evaluated at its final state.
    state = "model_state_dict" if arm == "all_row" else "best_model_state_dict"
    return [
        str(PYTHON),
        str(EVALUATOR),
        "--metadata", str(metadata),
        "--dreams", str(dreams),
        "--fingerprint-cache", str(SPARSE_MANIFEST),
        "--checkpoint", str(output["checkpoint"]),
        "--candidate-index-manifest", str(CANDIDATE_MANIFEST),
        "--candidate-pool-limit", "256",
        "--state-key", state,
        "--eval-batch-size", "64",
        "--device", device,
        "--out-json", str(output[phase]),
    ]


def main() -> None:
    args = parse_args()
    validate_assets(args.arm)
    output = paths(args.arm, args.seed, args.candidate_negatives)
    output["run_dir"].mkdir(parents=True, exist_ok=True)
    if args.phase == "train":
        command = train_command(
            args.arm, args.seed, args.device, output, args.candidate_negatives
        )
        expected_output = output["checkpoint"]
    else:
        if not output["checkpoint"].exists():
            raise FileNotFoundError(f"Train checkpoint is missing: {output['checkpoint']}")
        command = evaluation_command(args.arm, args.phase, args.device, output)
        expected_output = output[args.phase]
    if expected_output.exists() and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite completed P0.3 output: {expected_output}")
    print(
        json.dumps(
            {
                "arm": args.arm,
                "seed": args.seed,
                "candidate_negatives": args.candidate_negatives,
                "phase": args.phase,
                "command": command,
            },
            indent=2,
        )
    )
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
