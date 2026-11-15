#!/usr/bin/env python3
"""Prepare connectivity-isolated MPFIR ranking pilot and exact fold denylists.

Reads official TRAIN labels/metadata and an existing identity-only held-out
denylist. Never opens validation/test spectra, ranks, or labels. The lexical
element-count screen is only an optimization: retained molecules are matched
by recomputed RDKit InChI connectivity, including stereo/tautomer aliases.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from rdkit import Chem, RDLogger, rdBase
from tools.reranking.prepare_official_magma_local_targets import peak_targets
from tools.reranking.train_official_magma_forward import rows, digest, key
from glmr.candidate_indices import CandidateIndexStore, CandidateNegativeExclusion

RDLogger.DisableLog("rdApp.*")
ATOM = re.compile(r"\[([^\]]+)\]|Cl|Br|[BCNOPSFIbcnops]")
BRACKET = re.compile(r"^\d*([A-Z][a-z]?|se|as|[bcnops]|\*)")
_TARGETS: set[str] = set()
_SIGNATURES: set[tuple] = set()


def heavy_element_signature(smiles: str) -> tuple | None:
    """Conservative heavy-element multiset; None disables the fast screen."""
    if re.search(r"[A-Za-z*]", ATOM.sub("", smiles)):
        return None
    counts = Counter()
    for match in ATOM.finditer(smiles):
        if match.group(1) is not None:
            found = BRACKET.match(match.group(1))
            if found is None or found.group(1) == "*":
                return None
            symbol = found.group(1).capitalize()
        else:
            symbol = match.group(0).capitalize()
        if symbol != "H":
            counts[symbol] += 1
    if "*" in smiles or not counts:
        return None
    return tuple(sorted(counts.items()))


def connectivity(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid molecular identity: {smiles}")
    result = Chem.MolToInchiKey(mol)[:14]
    if len(result) != 14:
        raise ValueError("Missing recomputed connectivity")
    return result


def part_for(group: str, seed: int) -> str:
    value = int(key(seed, group)[:8], 16) % 10
    return "fit" if value < 6 else "calibration" if value == 6 else "evaluation"


def inner_fold(group: str, seed: int) -> int:
    return int(key(seed + 1, group)[:8], 16) % 2


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_rows(path: Path, values) -> None:
    with path.open("w") as handle:
        for value in values:
            handle.write(json.dumps(value, separators=(",", ":")) + "\n")


def init_scan(targets, signatures):
    global _TARGETS, _SIGNATURES
    _TARGETS, _SIGNATURES = set(targets), set(signatures)
    RDLogger.DisableLog("rdApp.*")


def scan_chunk(payload):
    start, values = payload
    found = []
    checked = 0
    for offset, smiles in enumerate(values):
        signature = heavy_element_signature(smiles)
        if signature is not None and signature not in _SIGNATURES:
            continue
        checked += 1
        group = connectivity(smiles)
        if group in _TARGETS:
            found.append((start + offset, group))
    return found, len(values), checked


def chunks(path, width=4096):
    batch, start = [], 0
    with path.open() as handle:
        for line in handle:
            smiles = line.rstrip("\n")
            if not smiles:
                raise ValueError("Empty molecule row")
            batch.append(smiles)
            if len(batch) == width:
                yield start, batch
                start += len(batch)
                batch = []
    if batch:
        yield start, batch


def prepare(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    source = ROOT / "outputs/mpfir_fragment_cap_audit/magma_official_train20k"
    official_path = ROOT / "outputs/msalign_stage1_full_v15/train_full.metadata.jsonl"
    official = list(rows(official_path))
    if any(row["fold"] != "train" for row in official):
        raise ValueError("Only official train is accepted")
    by_id = {row["identifier"]: row for row in official}
    if len(by_id) != len(official):
        raise ValueError("Duplicate official-train identifier")
    grouped = defaultdict(list)
    exclusions = Counter()
    for label in rows(source / "labels.jsonl"):
        meta = by_id[label["identifier"]]
        try:
            item = peak_targets(label, meta)
        except ValueError as exc:
            exclusions[str(exc)] += 1
            continue
        grouped[item["connectivity"]].append(item)
    pools = defaultdict(list)
    for group, members in grouped.items():
        chosen = min(members, key=lambda row: key(args.seed, row["identifier"]))
        chosen["pilot_partition"] = part_for(group, args.seed)
        chosen["inner_fold"] = inner_fold(group, args.seed)
        pools[chosen["pilot_partition"]].append(chosen)
    selected = []
    for name, limit in (("fit", args.fit), ("calibration", args.calibration),
                        ("evaluation", args.evaluation)):
        pool = sorted(pools[name], key=lambda row: key(args.seed, row["connectivity"]))
        if len(pool) < limit:
            raise ValueError(f"Insufficient {name} connectivities: {len(pool)} < {limit}")
        selected.extend(pool[:limit])
    groups = [row["connectivity"] for row in selected]
    if len(set(groups)) != len(groups):
        raise ValueError("Pilot must have one query per connectivity")
    write_rows(args.out / "labels.jsonl", selected)
    write_rows(args.out / "query_metadata.jsonl", [by_id[row["identifier"]] for row in selected])
    identity_cache = {}
    for row in official:
        smiles = row["canonical_isomeric_smiles"]
        if smiles not in identity_cache:
            identity_cache[smiles] = connectivity(smiles)
    official_groups = np.asarray([identity_cache[row["canonical_isomeric_smiles"]] for row in official])
    np.save(args.out / "official_train_connectivities.npy", official_groups)
    holdout = {row["connectivity"] for row in selected if row["pilot_partition"] != "fit"}
    target_sets = {"endpoint": holdout}
    for fold in range(2):
        target_sets[f"mining{fold}"] = holdout | {
            row["connectivity"] for row in selected
            if row["pilot_partition"] == "fit" and row["inner_fold"] == fold}
    signatures = {heavy_element_signature(row["smiles"]) for row in selected}
    if None in signatures:
        raise ValueError("Target with unsupported lexical element screen")
    write_json(args.out / "scan_targets.json", {
        "sets": {name: sorted(values) for name, values in target_sets.items()},
        "signatures": [list(signature) for signature in sorted(signatures)]})
    report = dict(seed=args.seed, official_test_read=False,
        official_validation_read=False, sampling="one hash-selected query per recomputed connectivity; label-available official-train cohort",
        partitions={name: sum(row["pilot_partition"] == name for row in selected)
                    for name in ("fit", "calibration", "evaluation")},
        inner_fold_counts={str(fold): sum(row["pilot_partition"] == "fit" and row["inner_fold"] == fold for row in selected)
                           for fold in range(2)},
        label_exclusions=dict(exclusions), eligible_connectivities=len(grouped),
        identity_rdkit=rdBase.rdkitVersion,
        inputs={str(path.relative_to(ROOT)): digest(path) for path in
                (official_path, source / "labels.jsonl", source / "sample.metadata.jsonl")},
        output_sha256={name: digest(args.out / name) for name in
                       ("labels.jsonl", "query_metadata.jsonl", "official_train_connectivities.npy", "scan_targets.json")})
    write_json(args.out / "prepare_summary.json", report)
    print(json.dumps(report), flush=True)


def scan(args):
    started = time.time()
    target_info = json.loads((args.out / "scan_targets.json").read_text())
    target_sets = {name: set(values) for name, values in target_info["sets"].items()}
    targets = set().union(*target_sets.values())
    signatures = {heavy_element_signature(row["smiles"]) for row in rows(args.out / "labels.jsonl")}
    recorded = {tuple(tuple(pair) for pair in value) for value in target_info["signatures"]}
    if None in signatures:
        raise ValueError("Unsupported target element signature")
    if signatures != recorded:
        target_info["signatures"] = [list(signature) for signature in sorted(signatures)]
        write_json(args.out / "scan_targets.json",target_info)
        summary=json.loads((args.out/"prepare_summary.json").read_text())
        summary["output_sha256"]["scan_targets.json"]=digest(args.out/"scan_targets.json")
        write_json(args.out/"prepare_summary.json",summary)
        print(json.dumps(dict(refreshed_stale_target_signature_cache=True)),flush=True)
    candidate = CandidateIndexStore(ROOT / "outputs/run_k_full_train/candidate_index_full_v15_mass_cap256_full_order/manifest.json")
    existing = CandidateNegativeExclusion(ROOT / "outputs/stage1_submission_0926/assets/heldout_connectivity_exclusion/manifest.json")
    existing.validate_candidate_store(candidate)
    smiles_path = ROOT / "outputs/run_k_full_train/full_sparse/full_smiles.txt"
    if digest(smiles_path) != candidate.molecule_smiles_sha256:
        raise ValueError("Molecule row digest mismatch")
    found, scanned, checked = [], 0, 0
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context,
                             initializer=init_scan, initargs=(targets, signatures)) as executor:
        pending = deque()
        for payload in chunks(smiles_path):
            pending.append(executor.submit(scan_chunk, payload))
            if len(pending) >= 4:
                result, count, parsed = pending.popleft().result()
                found.extend(result); scanned += count; checked += parsed
                if scanned % (4096 * 64) == 0:
                    print(json.dumps(dict(scanned=scanned, rdkit_parsed=checked,
                                          matches=len(found), seconds=round(time.time()-started, 1))), flush=True)
        while pending:
            result, count, parsed = pending.popleft().result()
            found.extend(result); scanned += count; checked += parsed
    if scanned != candidate.molecule_rows:
        raise ValueError("Incomplete molecule-universe scan")
    observed = {group for _, group in found}
    write_rows(args.out / "scan_matches.jsonl", [dict(molecule_row=index,connectivity=group) for index,group in found])
    if targets - observed:
        write_json(args.out/"scan_missing.json",dict(missing=sorted(targets-observed)))
        raise ValueError("Pilot identities missing from full molecule scan")
    for name, values in target_sets.items():
        folder = args.out / f"denylist_{name}"
        folder.mkdir()
        indices = np.asarray(sorted(set(existing.indices.tolist()) |
                                    {index for index, group in found if group in values}), dtype=np.int32)
        np.save(folder / "excluded_rows.npy", indices)
        write_json(folder / "manifest.json", dict(format="glmr_candidate_negative_exclusion_v1",
            molecule_rows=candidate.molecule_rows, molecule_smiles_sha256=candidate.molecule_smiles_sha256,
            indices="excluded_rows.npy", excluded_rows=len(indices),
            identity_definition="recomputed RDKit InChI connectivity block",
            existing_official_denylist_sha256=digest(existing.manifest_path),
            additional_connectivities=sorted(values), scan_rows=scanned,
            screen="conservative lexical heavy-element multiset; exact RDKit confirmation",
            indices_sha256=digest(folder / "excluded_rows.npy")))
    write_json(args.out / "scan_summary.json", dict(scanned=scanned, rdkit_parsed=checked,
        matches=len(found), target_connectivities=len(targets), missing=[], workers=2,
        seconds=time.time()-started, official_test_read=False,
        code_sha256=digest(Path(__file__))))
    print(json.dumps(json.loads((args.out / "scan_summary.json").read_text())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "scan"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fit", type=int, default=2048)
    parser.add_argument("--calibration", type=int, default=512)
    parser.add_argument("--evaluation", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
    else:
        scan(args)


if __name__ == "__main__":
    main()
