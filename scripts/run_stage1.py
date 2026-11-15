#!/usr/bin/env python3
"""Run only the frozen SCCF official train/validation protocol."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/official_v15_sccf.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--phase", choices=("train", "val", "both"), default="both")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def execute(command: list[str], expected: Path, dry_run: bool) -> None:
    print(json.dumps({"command": command, "expected_output": str(expected.relative_to(ROOT))}, indent=2))
    if dry_run:
        return
    if expected.exists():
        raise FileExistsError(f"Refusing to overwrite frozen output: {expected}")
    expected.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    if not expected.exists():
        raise RuntimeError(f"Expected output was not produced: {expected}")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("official_test_read") is not False:
        raise ValueError("The public runner requires official_test_read=false")
    seeds = [int(value) for value in config["seeds"]]
    if args.seed is not None:
        if args.seed not in seeds:
            raise ValueError(f"Seed {args.seed} is outside the frozen seed set {seeds}")
        seeds = [args.seed]
    paths = {name: resolve(value) for name, value in config["paths"].items()}
    protocol = config["protocol"]
    phases = ("train", "val") if args.phase == "both" else (args.phase,)
    for seed in seeds:
        run_dir = paths["run_root"] / f"seed{seed}"
        checkpoint = run_dir / "checkpoint.pt"
        summary = run_dir / "train_summary.json"
        ranks = run_dir / "val_ranks.json"
        for phase in phases:
            if phase == "train":
                command = [
                    sys.executable,
                    str(ROOT / "tools/stage1/train_msalign_projection_smoke.py"),
                    "--train-metadata", str(paths["train_metadata"]),
                    "--val-metadata", str(paths["val_metadata"]),
                    "--train-dreams", str(paths["train_dreams"]),
                    "--val-dreams", str(paths["val_dreams"]),
                    "--fingerprint-cache", str(paths["fingerprint_manifest"]),
                    "--candidate-index-manifest", str(paths["candidate_manifest"]),
                    "--negative-exclusion-manifest", str(paths["negative_exclusion_manifest"]),
                    "--candidate-pool-limit", str(protocol["candidate_pool_limit"]),
                    "--candidate-negatives", str(protocol["candidate_negatives"]),
                    "--negative-sampling", str(protocol["negative_sampling"]),
                    "--loss-mode", "candidate_ce",
                    "--epochs", str(protocol["epochs"]),
                    "--batch-size", str(protocol["batch_size"]),
                    "--eval-batch-size", str(protocol["eval_batch_size"]),
                    "--lr", str(protocol["learning_rate"]),
                    "--weight-decay", str(protocol["weight_decay"]),
                    "--head-type", "mlp",
                    "--hidden-dim", str(protocol["hidden_dim"]),
                    "--proj-dim", str(protocol["projection_dim"]),
                    "--dropout", str(protocol["dropout"]),
                    "--temperature", str(protocol["temperature"]),
                    "--freeze-temperature",
                    "--seed", str(seed),
                    "--train-sampling", str(protocol["train_sampling"]),
                    "--mol-head-type", "shared_mlp",
                    "--mol-block-normalization", str(protocol["block_normalization"]),
                    "--mol-blocks", ",".join(protocol["blocks"]),
                    "--preload-sparse-features",
                    "--device", args.device,
                    "--out-json", str(summary),
                    "--checkpoint", str(checkpoint),
                ]
                execute(command, checkpoint, args.dry_run)
            else:
                if not checkpoint.exists() and not args.dry_run:
                    raise FileNotFoundError(checkpoint)
                command = [
                    sys.executable,
                    str(ROOT / "tools/stage1/evaluate_msalign_projection_checkpoint.py"),
                    "--metadata", str(paths["val_metadata"]),
                    "--dreams", str(paths["val_dreams"]),
                    "--fingerprint-cache", str(paths["fingerprint_manifest"]),
                    "--candidate-index-manifest", str(paths["candidate_manifest"]),
                    "--candidate-pool-limit", str(protocol["candidate_pool_limit"]),
                    "--checkpoint", str(checkpoint),
                    "--state-key", "model_state_dict",
                    "--eval-batch-size", str(protocol["eval_batch_size"]),
                    "--preload-sparse-features",
                    "--device", args.device,
                    "--out-json", str(ranks),
                ]
                execute(command, ranks, args.dry_run)


if __name__ == "__main__":
    main()
