#!/usr/bin/env python3
"""Run the ordered official/data-safe Stage 1 submission matrix.

The ordering is deliberate: single-representation controls first, then the
final SCCF bundle, and only then matched attribution experiments.
Official test is intentionally not exposed by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
TRAINER = ROOT / "tools/stage1/train_msalign_projection_smoke.py"
EVALUATOR = ROOT / "tools/stage1/evaluate_msalign_projection_checkpoint.py"
OUTPUT_ROOT = ROOT / "outputs/stage1_submission_0926"
ASSET_ROOT = OUTPUT_ROOT / "assets"

TRAIN_METADATA = ROOT / "outputs/msalign_stage1_full_v15/train_full.metadata.jsonl"
VAL_METADATA = ROOT / "outputs/msalign_stage1_full_v15/val_full.metadata.jsonl"
TRAIN_DREAMS = ROOT / "outputs/run_k_official_datasafe_seed20260621/train.no_head.npy"
VAL_DREAMS = ROOT / "outputs/run_k_official_datasafe_seed20260621/val.no_head.npy"
CANDIDATE_MANIFEST = (
    ROOT / "outputs/run_k_full_train/candidate_index_full_v15_mass_cap256_full_order/manifest.json"
)
CANDIDATE_AUDIT = (
    ROOT / "outputs/run_k_full_train/candidate_index_full_v15_mass_cap256_full_order/audit_full.json"
)
R2_MANIFEST = ASSET_ROOT / "full_sparse_direct/manifest_historical_projection_order.json"
R3_MANIFEST = R2_MANIFEST
NEGATIVE_EXCLUSION = ASSET_ROOT / "heldout_connectivity_exclusion/manifest.json"
CHEMBERTA_CACHE = ASSET_ROOT / "chemberta_77m_mtr_full_v15.pt"

SEEDS = (20260621, 20260622, 20260623)
OFFICIAL_CANDIDATE_SHA256 = "9536a330cba24c399271364889119f90124579400dba14e9f78630c75c13c73a"
LOCKED_RDKIT_VERSION = "2025.09.6"
OFFICIAL_CANDIDATES = ROOT / "data/data/molecules/MassSpecGym1.5_retrieval_candidates_mass.json"


@dataclass(frozen=True)
class Arm:
    stage: str
    blocks: tuple[str, ...] = ()
    manifest: str | None = None
    normalization: str = "block_l2"
    chemberta: bool = False
    seeds: tuple[int, ...] = (20260621,)


ARMS: dict[str, Arm] = {
    # Controls are always completed before any combined fingerprint claim.
    "morgan_r2_only": Arm("controls", ("count_morgan",), "r2", seeds=SEEDS),
    "morgan_r3_only": Arm("controls", ("count_morgan_r3",), "r3", seeds=SEEDS),
    "atom_pair_only": Arm("controls", ("count_atom_pair",), "r3"),
    "path_only": Arm("controls", ("rdkit_path_count",), "r3"),
    "maccs_only": Arm("controls", ("maccs",), "r3"),
    "chemberta_mtr_mean": Arm("controls", normalization="none", chemberta=True, seeds=SEEDS),
    # Attribution is deliberately downstream of the frozen final bundle.
    "full_r3_without_atom_pair": Arm("attribution", ("count_morgan_r3", "rdkit_path_count", "maccs"), "r3"),
    "full_r3_without_path": Arm(
        "attribution",
        ("count_morgan_r3", "count_atom_pair", "maccs"),
        "r3",
    ),
    "full_r3_without_maccs": Arm(
        "attribution",
        ("count_morgan_r3", "count_atom_pair", "rdkit_path_count"),
        "r3",
    ),
    "full_r2_block_l2": Arm(
        "attribution",
        ("count_morgan", "count_atom_pair", "rdkit_path_count", "maccs"),
        "r2",
        seeds=SEEDS,
    ),
    "full_r3_global_l2": Arm(
        "attribution",
        ("count_morgan_r3", "count_atom_pair", "rdkit_path_count", "maccs"),
        "r3",
        normalization="global_l2",
    ),
    "full_r3_block_l2": Arm(
        "final",
        ("count_morgan_r3", "count_atom_pair", "rdkit_path_count", "maccs"),
        "r3",
        seeds=SEEDS,
    ),
}
STAGE_ORDER = ("controls", "final", "attribution")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--stage", choices=(*STAGE_ORDER, "all"))
    selection.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--seed", type=int, choices=SEEDS, default=None)
    parser.add_argument("--phase", choices=("train", "val", "both"), default="both")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument(
        "--allow-out-of-order",
        action="store_true",
        help="Development only. Never use this flag for the formal matrix.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run_paths(arm_name: str, seed: int) -> dict[str, Path]:
    run_dir = OUTPUT_ROOT / "runs" / arm_name / f"seed{seed}"
    return {
        "run_dir": run_dir,
        "summary": run_dir / "train_summary.json",
        "checkpoint": run_dir / "checkpoint.pt",
        "val": run_dir / "val_ranks.json",
        "log": run_dir / "command.json",
    }


def fingerprint_manifest(arm: Arm) -> Path:
    if arm.manifest == "r2":
        return R2_MANIFEST
    if arm.manifest == "r3":
        return R3_MANIFEST
    raise ValueError(f"Arm has no sparse manifest: {arm}")


def validate_assets(selected: list[str]) -> dict[str, Any]:
    required = [
        TRAIN_METADATA,
        VAL_METADATA,
        TRAIN_DREAMS,
        VAL_DREAMS,
        CANDIDATE_MANIFEST,
        CANDIDATE_AUDIT,
        NEGATIVE_EXCLUSION,
        OFFICIAL_CANDIDATES,
    ]
    if any(not ARMS[name].chemberta for name in selected):
        required.extend((R2_MANIFEST, R3_MANIFEST))
    if any(ARMS[name].chemberta for name in selected):
        required.append(CHEMBERTA_CACHE)
    missing = [str(path) for path in dict.fromkeys(required) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Stage 1 submission assets are missing: {missing}")
    if sha256_file(OFFICIAL_CANDIDATES) != OFFICIAL_CANDIDATE_SHA256:
        raise ValueError("Official MassSpecGym v1.5 candidate JSON digest changed")
    sparse_manifest_sha256 = None
    sparse_rdkit_version = None
    if any(not ARMS[name].chemberta for name in selected):
        sparse = json.loads(R3_MANIFEST.read_text(encoding="utf-8"))
        sparse_rdkit_version = sparse.get("rdkit_version")
        if sparse_rdkit_version != LOCKED_RDKIT_VERSION:
            raise ValueError(
                f"Sparse cache RDKit version differs: {sparse_rdkit_version!r} != "
                f"{LOCKED_RDKIT_VERSION!r}"
            )
        expected_features = {
            "count_morgan", "count_morgan_r3", "count_atom_pair", "rdkit_path_count", "maccs"
        }
        if set(sparse.get("features", {})) != expected_features:
            raise ValueError("Sparse cache does not contain the locked five feature blocks")
        sparse_manifest_sha256 = sha256_file(R3_MANIFEST)
    candidate = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    audit = json.loads(CANDIDATE_AUDIT.read_text(encoding="utf-8"))
    exclusion = json.loads(NEGATIVE_EXCLUSION.read_text(encoding="utf-8"))
    if int(candidate["candidate_pool_limit"]) != 256:
        raise ValueError("Candidate index is not the locked official cap256 store")
    if int(candidate["records"]) != 231104 or not audit.get("candidate_order_and_indices"):
        raise ValueError("Candidate index integrity audit is incomplete")
    if int(exclusion["molecule_rows"]) != int(candidate["molecule_rows"]):
        raise ValueError("Negative exclusion and candidate molecule universes differ")
    if exclusion["molecule_smiles_sha256"] != candidate["molecule_smiles_sha256"]:
        raise ValueError("Negative exclusion and candidate row digests differ")
    if exclusion.get("missing_connectivities"):
        raise ValueError("Held-out connectivity exclusion has unresolved identities")
    return {
        "official_candidate_sha256": OFFICIAL_CANDIDATE_SHA256,
        "candidate_manifest_sha256": sha256_file(CANDIDATE_MANIFEST),
        "negative_exclusion_sha256": sha256_file(NEGATIVE_EXCLUSION),
        "train_metadata_sha256": sha256_file(TRAIN_METADATA),
        "val_metadata_sha256": sha256_file(VAL_METADATA),
        "train_dreams_sha256": sha256_file(TRAIN_DREAMS),
        "val_dreams_sha256": sha256_file(VAL_DREAMS),
        "sparse_manifest_sha256": sparse_manifest_sha256,
        "sparse_rdkit_version": sparse_rdkit_version,
    }


def prerequisite_outputs(name: str) -> tuple[str, list[Path]]:
    stage = ARMS[name].stage
    if name == "full_r2_block_l2":
        description = "the two completed full-r3 SCCF review seeds"
        return description, [
            run_paths("full_r3_block_l2", seed)["val"]
            for seed in SEEDS[:2]
        ]
    if stage == "final":
        prerequisite_names = [name for name, arm in ARMS.items() if arm.stage == "controls"]
        description = "single-representation controls"
    elif stage == "attribution":
        prerequisite_names = ["full_r3_block_l2"]
        description = "the frozen final SCCF seed set"
    else:
        return "", []
    paths = [
        run_paths(name, seed)["val"]
        for name in prerequisite_names
        for seed in ARMS[name].seeds
    ]
    return description, paths


def enforce_order(name: str, dry_run: bool, allow_out_of_order: bool) -> None:
    if allow_out_of_order or dry_run:
        return
    description, required = prerequisite_outputs(name)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            f"{name} is gated on completed {description}. Missing: "
            + ", ".join(missing)
        )


def resource_snapshot(device: str) -> dict[str, Any]:
    memory = shutil.disk_usage(ROOT)
    snapshot: dict[str, Any] = {
        "disk_free_bytes": memory.free,
        "disk_total_bytes": memory.total,
    }
    free_output = subprocess.run(
        ["free", "-b"], check=True, capture_output=True, text=True
    ).stdout
    memory_line = next(line for line in free_output.splitlines() if line.startswith("Mem:"))
    fields = memory_line.split()
    available = int(fields[-1])
    snapshot["host_available_bytes"] = available
    if available < 64 * 1024**3:
        raise RuntimeError(f"Host available memory below 64 GiB gate: {available}")
    if memory.free < 25 * 1024**3:
        raise RuntimeError(f"Disk free space below 25 GiB gate: {memory.free}")
    if device == "cuda":
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        snapshot["gpu"] = query
        torch_check = subprocess.run(
            [
                str(PYTHON),
                "-c",
                "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))",
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "0"},
        ).stdout.strip()
        snapshot["torch_gpu"] = torch_check
    return snapshot


def train_command(name: str, seed: int, device: str, output: dict[str, Path]) -> list[str]:
    arm = ARMS[name]
    command = [
        str(PYTHON),
        str(TRAINER),
        "--train-metadata", str(TRAIN_METADATA),
        "--val-metadata", str(VAL_METADATA),
        "--train-dreams", str(TRAIN_DREAMS),
        "--val-dreams", str(VAL_DREAMS),
        "--candidate-index-manifest", str(CANDIDATE_MANIFEST),
        "--negative-exclusion-manifest", str(NEGATIVE_EXCLUSION),
        "--candidate-pool-limit", "256",
        "--candidate-negatives", "127",
        "--negative-sampling", "random",
        "--loss-mode", "candidate_ce",
        "--epochs", "1",
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
        "--train-sampling", "row_shuffle",
        "--mol-head-type", "shared_mlp",
        "--mol-block-normalization", arm.normalization,
        "--device", device,
        "--out-json", str(output["summary"]),
        "--checkpoint", str(output["checkpoint"]),
    ]
    if arm.chemberta:
        command.extend(("--chemberta-cache", str(CHEMBERTA_CACHE), "--mol-pooling", "mean"))
    else:
        command.extend(
            (
                "--fingerprint-cache",
                str(fingerprint_manifest(arm)),
                "--mol-pooling",
                "mean",
                "--mol-blocks",
                ",".join(arm.blocks),
                "--preload-sparse-features",
            )
        )
    return command


def validation_command(name: str, device: str, output: dict[str, Path]) -> list[str]:
    arm = ARMS[name]
    command = [
        str(PYTHON),
        str(EVALUATOR),
        "--metadata", str(VAL_METADATA),
        "--dreams", str(VAL_DREAMS),
        "--checkpoint", str(output["checkpoint"]),
        "--candidate-index-manifest", str(CANDIDATE_MANIFEST),
        "--candidate-pool-limit", "256",
        "--state-key", "model_state_dict",
        "--eval-batch-size", "64",
        "--device", device,
        "--out-json", str(output["val"]),
    ]
    if arm.chemberta:
        command.extend(("--chemberta-cache", str(CHEMBERTA_CACHE)))
    else:
        command.extend(("--fingerprint-cache", str(fingerprint_manifest(arm)), "--preload-sparse-features"))
    return command


def selected_arms(args: argparse.Namespace) -> list[str]:
    if args.arm:
        names = [args.arm]
    else:
        stages = STAGE_ORDER if args.stage == "all" else (args.stage,)
        names = [name for stage in stages for name, arm in ARMS.items() if arm.stage == stage]
    return names


def selected_seeds(name: str, requested_seed: int | None) -> tuple[int, ...]:
    seeds = ARMS[name].seeds
    if requested_seed is None:
        return seeds
    if requested_seed not in seeds:
        raise ValueError(f"{name} is preregistered only for seeds {seeds}, not {requested_seed}")
    return (requested_seed,)


def execute(command: list[str], output: Path, args: argparse.Namespace) -> None:
    if output.exists():
        if args.skip_complete:
            print(json.dumps({"status": "skip_complete", "output": str(output)}), flush=True)
            return
        if not args.dry_run:
            raise FileExistsError(f"Refusing to overwrite formal output: {output}")
    print(json.dumps({"command": command, "expected_output": str(output)}, indent=2), flush=True)
    if args.dry_run:
        return
    snapshot = resource_snapshot(args.device)
    print(json.dumps({"resource_snapshot": snapshot}, indent=2), flush=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    if args.device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(command, cwd=ROOT, check=True, env=env)
    if not output.exists():
        raise RuntimeError(f"Command completed without expected output: {output}")


def main() -> None:
    args = parse_args()
    names = selected_arms(args)
    digests = validate_assets(names)
    plan = {
        "protocol": "official-v1.5/data-safe-DreaMS/cap256/negative127/candidate-CE/1epoch",
        "official_test": "not read or evaluated by this runner",
        "fixed_additive_order": [
            "count_morgan_r3",
            "count_atom_pair",
            "rdkit_path_count",
            "maccs",
        ],
        "selected_arms": names,
        "digests": digests,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    plan_path = OUTPUT_ROOT / "run_plan.snapshot.json"
    if not args.dry_run:
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)

    phases = ("train", "val") if args.phase == "both" else (args.phase,)
    for name in names:
        enforce_order(name, args.dry_run, args.allow_out_of_order)
        for seed in selected_seeds(name, args.seed):
            output = run_paths(name, seed)
            output["run_dir"].mkdir(parents=True, exist_ok=True)
            if not args.dry_run:
                output["log"].write_text(
                    json.dumps(
                        {
                            "arm": name,
                            "stage": ARMS[name].stage,
                            "seed": seed,
                            "blocks": ARMS[name].blocks,
                            "normalization": ARMS[name].normalization,
                            "digests": digests,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            for phase in phases:
                if phase == "train":
                    command = train_command(name, seed, args.device, output)
                    expected = output["checkpoint"]
                else:
                    if not output["checkpoint"].exists() and not args.dry_run:
                        raise FileNotFoundError(output["checkpoint"])
                    command = validation_command(name, args.device, output)
                    expected = output["val"]
                execute(command, expected, args)


if __name__ == "__main__":
    main()
