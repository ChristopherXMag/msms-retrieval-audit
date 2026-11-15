#!/usr/bin/env python3
"""Run frozen SCCF constant, wrong-spectrum, random, and candidate-only controls."""

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
    parser.add_argument("--control", choices=("spectrum", "candidate-only", "all"), default="all")
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
        raise FileExistsError(f"Refusing to overwrite control output: {expected}")
    expected.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    if not expected.exists():
        raise RuntimeError(f"Expected output was not produced: {expected}")


def spectrum_command(config: dict, paths: dict[str, Path], seed: int, scope: str) -> tuple[list[str], Path]:
    run_dir = paths["run_root"] / f"seed{seed}"
    controls_dir = run_dir / "blind_controls"
    stem = "constant_spectrum_full_val" if scope == "full_val_constant" else "spectrum_permutation_exact_mass"
    output = controls_dir / f"{stem}.json"
    control = config["controls"]
    protocol = config["protocol"]
    command = [
        sys.executable,
        str(ROOT / "tools/audits/audit_spectrum_permutation.py"),
        "--metadata", str(paths["val_metadata"]),
        "--dreams", str(paths["val_dreams"]),
        "--train-dreams-for-mean", str(paths["train_dreams"]),
        "--checkpoint", str(run_dir / "checkpoint.pt"),
        "--fingerprint-cache", str(paths["fingerprint_manifest"]),
        "--candidate-index-manifest", str(paths["candidate_manifest"]),
        "--reference-ranks", str(run_dir / "val_ranks.json"),
        "--candidate-pool-limit", str(protocol["candidate_pool_limit"]),
        "--evaluation-scope", scope,
        "--permutation-seeds", ",".join(str(value) for value in control["permutation_seeds"]),
        "--candidate-order-seed", str(control["candidate_order_seed"]),
        "--bootstrap-resamples", str(control["bootstrap_resamples"]),
        "--bootstrap-seed", str(control["bootstrap_seed"]),
        "--state-key", "best_model_state_dict",
        "--eval-batch-size", str(protocol["eval_batch_size"]),
        "--device", config.get("device_override", "auto"),
        "--out-json", str(output),
        "--out-npz", str(controls_dir / f"{stem}.per_query.npz"),
    ]
    return command, output


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("official_test_read") is not False:
        raise ValueError("The public control runner requires official_test_read=false")
    config["device_override"] = args.device
    paths = {name: resolve(value) for name, value in config["paths"].items()}
    seeds = [int(value) for value in config["seeds"]]
    if args.seed is not None:
        if args.seed not in seeds:
            raise ValueError(f"Seed {args.seed} is outside the frozen seed set {seeds}")
        seeds = [args.seed]
    if args.control in ("spectrum", "all"):
        for seed in seeds:
            for scope in ("full_val_constant", "exact_mass_permutation"):
                command, output = spectrum_command(config, paths, seed, scope)
                execute(command, output, args.dry_run)
    if args.control in ("candidate-only", "all"):
        control = config["controls"]
        output_dir = paths["run_root"] / "candidate_only"
        command = [
            sys.executable,
            str(ROOT / "tools/audits/audit_candidate_shortcuts.py"),
            "--train-metadata", str(paths["train_metadata"]),
            "--val-metadata", str(paths["val_metadata"]),
            "--candidate-index-manifest", str(paths["candidate_manifest"]),
            "--smiles", str(paths["molecule_smiles"]),
            "--train-negative-exclusion-manifest", str(paths["negative_exclusion_manifest"]),
            "--candidate-pool-limit", str(config["protocol"]["candidate_pool_limit"]),
            "--train-pools", str(control["candidate_only_train_pools"]),
            "--workers", str(control["candidate_only_workers"]),
            "--seed", str(control["bootstrap_seed"]),
            "--out-dir", str(output_dir),
        ]
        execute(command, output_dir / "audit.json", args.dry_run)


if __name__ == "__main__":
    main()
