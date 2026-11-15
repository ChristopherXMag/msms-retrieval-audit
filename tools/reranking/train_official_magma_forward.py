#!/usr/bin/env python3
"""Bounded, random-init MAGMa curriculum pilot on official training only.

Reuses the pinned upstream GLACIER model and losses, not its pretrained weights.
This is forward-model pretraining, NOT a retrieval result or a DreaMS replacement.
All matched atom masks are weak set targets; unassigned peaks remain spectrum
reconstruction targets, never negative fragment labels. No test input is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
from collections import Counter, OrderedDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from rdkit import Chem
import torch
from torch.utils.data import Dataset, DataLoader

from glmr.glacier_runtime import load_glacier
from tools.reranking.train_glacier_run_k_retrieval import (
    fragment_masks_from_label, molecule_record, move_to_device, _forward_model,
)

PIN = "cd616f629a82040b950fbf8f0a983cc70ce1cf3b"


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def key(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def partition(seed, connectivity):
    return "calibration" if int(key(seed, connectivity)[:8], 16) % 5 == 0 else "fit"


def compact_label(label, meta, max_atoms, max_breakpoints):
    """Keep exact label atom order, every ambiguity and every selected peak."""
    if meta["fold"] != "train":
        raise ValueError("not_official_train")
    if label["adduct"] != meta["adduct"]:
        raise ValueError("adduct_mismatch")
    if meta["adduct"] not in ("[M+H]+", "[M+Na]+"):
        raise ValueError("unsupported_adduct")
    if meta.get("instrument_type") not in ("Orbitrap", "QTOF"):
        raise ValueError("unsupported_instrument")
    ce = meta.get("collision_energy")
    if ce is None or not math.isfinite(float(ce)) or float(ce) < 0:
        raise ValueError("missing_or_non_scalar_ce")
    smi = label["magma_canonical_smiles"]
    record = molecule_record(smi, max_heavy_atoms=max_atoms)
    reference = molecule_record(meta["canonical_isomeric_smiles"], max_heavy_atoms=max_atoms)
    if record.canonical_isomeric_smiles != reference.canonical_isomeric_smiles:
        raise ValueError("structure_mismatch")
    mol = Chem.RemoveHs(Chem.MolFromSmiles(smi))
    n = mol.GetNumAtoms()
    if n != label["num_atoms"]:
        raise ValueError("atom_count_mismatch")
    masks = fragment_masks_from_label(label, num_atoms=n)
    if not masks:
        raise ValueError("no_matched_fragment_masks")
    # Native model predicts boundary cardinalities 0..3. Do not silently truncate
    # either alternatives or targets beyond its architectural support.
    boundaries = set()
    for mask in masks:
        inside = set(mask)
        boundary = frozenset(a.GetIdx() for i in mask for a in mol.GetAtomWithIdx(i).GetNeighbors()
                             if a.GetIdx() not in inside)
        if len(boundary) > 3:
            raise ValueError("boundary_cardinality_above_three")
        boundaries.add(boundary)
    if len(boundaries) > max_breakpoints:
        raise ValueError("boundary_target_count_above_slots")
    peaks = [[float(p["mz"]), float(p["relative_intensity"])] for p in label["peaks"]]
    spec = np.asarray(peaks, dtype=np.float32)
    if not len(spec) or not np.isfinite(spec).all() or np.any(spec[:, 0] <= 0) or np.any(spec[:, 1] < 0) or not np.any(spec[:, 1] > 0):
        raise ValueError("invalid_spectrum")
    return dict(identifier=label["identifier"], connectivity=record.connectivity,
                smiles=smi, num_atoms=n, fragment_masks=masks, observed_spectrum=peaks,
                adduct=meta["adduct"], instrument=meta["instrument_type"],
                collision_energy=float(ce), precursor_mz=float(meta["precursor_mz"]),
                matched_peaks=sum(bool(p.get("matches")) for p in label["peaks"]),
                ambiguous_peaks=sum(len(p.get("matches", [])) > 1 for p in label["peaks"]),
                boundary_patterns=len(boundaries))


def prepare(args):
    out = args.out
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    meta = {r["identifier"]: r for r in rows(args.metadata)}
    train_ids = {r["identifier"] for r in rows(args.official_train) if r["fold"] == "train"}
    val = list(rows(args.official_val))
    val_ids = {r["identifier"] for r in val}
    val_connectivities = {molecule_record(r["canonical_isomeric_smiles"], 1000).connectivity for r in val}
    pool = {"fit": [], "calibration": []}
    excluded = Counter()
    seen = set()
    for label in rows(args.labels):
        ident = label["identifier"]
        if ident in seen:
            raise ValueError(f"Duplicate label: {ident}")
        seen.add(ident)
        if ident not in train_ids or ident in val_ids:
            raise ValueError(f"Label outside official train: {ident}")
        try:
            item = compact_label(label, meta[ident], args.max_atoms, args.slots)
        except ValueError as exc:
            excluded[str(exc)] += 1
            continue
        if item["connectivity"] in val_connectivities:
            raise ValueError(f"Train/val connectivity leak: {ident}")
        pool[partition(args.seed, item["connectivity"])].append(item)
    selected = {}
    for part, count in (("fit", args.fit_rows), ("calibration", args.calibration_rows)):
        selected[part] = sorted(pool[part], key=lambda r: key(args.seed, r["identifier"]))[:count]
        if len(selected[part]) != count:
            raise ValueError(f"Insufficient eligible {part}: {len(selected[part])}/{count}")
    assert not ({r["connectivity"] for r in selected["fit"]} &
                {r["connectivity"] for r in selected["calibration"]})
    upstream_pin = subprocess.check_output(["git", "-C", str(args.ms_pred), "rev-parse", "HEAD"], text=True).strip()
    if upstream_pin != PIN:
        raise ValueError(f"Unreviewed upstream source: {upstream_pin}")
    out.mkdir(parents=True)
    for part, items in selected.items():
        with (out / f"{part}.jsonl").open("w") as handle:
            for item in items:
                handle.write(json.dumps(item) + "\n")
    manifest = dict(schema=1, protocol="official-train-only MAGMa forward pilot; no test; random init",
                    created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    source_pin=upstream_pin, seed=args.seed, fit_rows=args.fit_rows,
                    calibration_rows=args.calibration_rows, max_atoms=args.max_atoms,
                    max_breakpoints=args.slots, eligible={k:len(v) for k,v in pool.items()},
                    excluded=dict(excluded), source_sha256={str(p.relative_to(ROOT)):digest(p)
                        for p in (args.labels, args.metadata, args.official_train, args.official_val)},
                    implementation_sha256={str(p.relative_to(ROOT)):digest(p) for p in (
                        Path(__file__), ROOT/"glmr/glacier_runtime.py",
                        ROOT/"tools/generation/run_glacier_seeded_predict.py",
                        args.ms_pred/"src/ms_pred/glacier/joint_model.py",
                        args.ms_pred/"src/ms_pred/glacier/dataset.py")},
                    data_sha256={part:digest(out/f"{part}.jsonl") for part in selected},
                    guarantees=["connectivity-disjoint fit/calibration", "all alternatives retained",
                                "unmatched peaks retained for reconstruction", "no test file read"],
                    curriculum=dict(warmup_steps=128, decay_steps=256, decay_rate=0.5),
                    budgets=dict(epochs=4, batch_size=2, calibration_batch_size=1, workers=0, cache=512),
                    inference_contract="eval-safe functional SDPA; isolated batch1",
                    checkpoint_selection="minimum free-running true-spectrum cosine loss on train calibration; epoch 0 included",
                    forward_continuation_gate="decay best loss at least 0.01 below epoch 0 and not worse than constant best; reconstruction only, not retrieval success",
                    retrieval_gate="not evaluated here; later freeze top5 fusion on train, compare against frozen pair, damage <=8%, full-system blind controls required")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


class ForwardDataset(Dataset):
    def __init__(self, examples, processor, common, cache_size=512):
        self.examples, self.processor, self.common = examples, processor, common
        self.cache_size, self.cache = cache_size, OrderedDict()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        row = self.examples[index]
        feat = self.processor.featurize_tree(dict(root_canonical_smiles=row["smiles"],
            adduct=row["adduct"], collision_energy=row["collision_energy"]),
            include_inten_targs=False, include_frag_targs=False)
        if len(feat["masses"]) != row["num_atoms"]:
            raise ValueError("Feature/label atom count mismatch")
        mask = torch.zeros((len(row["fragment_masks"]), row["num_atoms"]), dtype=torch.bool)
        for i, indices in enumerate(row["fragment_masks"]):
            mask[i, list(indices)] = True
        feat.update(name=row["identifier"], frag_targs=mask,
                    inten_targs=np.asarray(row["observed_spectrum"], dtype=np.float32),
                    adduct=self.common.ion2onehot_pos[row["adduct"]],
                    instrument=self.common.instrument2onehot_pos[row["instrument"]],
                    precursor=row["precursor_mz"])
        if self.cache_size:
            self.cache[index] = feat
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return feat


def resource_snapshot():
    snap = dict(rss_peak_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30)
    base = Path("/sys/fs/cgroup")
    if (base / "memory.max").exists():
        limit = (base / "memory.max").read_text().strip()
        if limit != "max":
            current = int((base / "memory.current").read_text())
            snap.update(cgroup_used_gib=current / 2**30, cgroup_free_gib=(int(limit)-current)/2**30)
            if snap["cgroup_free_gib"] < 32:
                raise MemoryError(f"Host cgroup memory safety stop: {snap}")
    return snap


def train(args):
    manifest = json.loads((args.data / "manifest.json").read_text())
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite run: {args.out}")
    for part, expected in manifest["data_sha256"].items():
        if digest(args.data / f"{part}.jsonl") != expected:
            raise ValueError(f"Changed frozen data: {part}")
    for path, expected in manifest.get("implementation_sha256", {}).items():
        if digest(ROOT/path) != expected:
            raise ValueError(f"Changed frozen implementation: {path}; prepare a new manifest")
    pin = subprocess.check_output(["git", "-C", str(args.ms_pred), "rev-parse", "HEAD"], text=True).strip()
    if pin != manifest["source_pin"]:
        raise ValueError("Upstream source changed")
    random.seed(manifest["seed"]); np.random.seed(manifest["seed"]); torch.manual_seed(manifest["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; verify host CUDA outside sandbox")
    common, Processor, NativeDataset, Model = load_glacier(args.ms_pred)
    proc = Processor(root_encode="graphormer", pe_embed_k=0, embed_elem_group=True, multi_hop_max_dist=10)
    datasets = {p:ForwardDataset(list(rows(args.data/f"{p}.jsonl")), proc, common) for p in ("fit", "calibration")}
    first = datasets["fit"][0]
    kwargs = dict(hidden_size=256, graphormer_layers=5, graphormer_dropout=0.1,
        frag_decoder_layers=4, frag_encoder_layers=0, frag_dropout=0.1,
        inten_decoder_layers=4, inten_encoder_layers=3, inten_dropout=0.1,
        node_feats=first["graphormer_input"]["x"].shape[-1],
        edge_feats=first["graphormer_input"]["attn_edge_type"].shape[-1],
        multi_hop_max_dist=10, num_edge_dis=10, max_breakpoints=manifest["max_breakpoints"],
        embed_adduct=True, embed_collision=True, embed_elem_group=True, embed_instrument=True,
        encode_forms=True, enable_aux_loss=True, max_broken_bonds=6, linsat_tau=15,
        sk_tau=0.05, ppm_tol=20, num_bins=150000, inten_loss_fn="cosine",
        lr=4e-4, lr_decay_rate=0.9, warmup=128, frag_weight=0.1, inten_weight=1,
        magma_warmup_steps=128, magma_decay_steps=256,
        magma_decay_rate=0.5 if args.arm=="decay" else 1.0)
    model = Model(**kwargs).cuda()
    logs = {}
    # Native loss math unchanged. Capture metrics without requiring a PL Trainer.
    model.log = lambda name, value, **unused: logs.update({name:float(value.detach()) if torch.is_tensor(value) else float(value)})
    optim_conf = model.configure_optimizers()
    optimizer = optim_conf["optimizer"]
    scheduler = optim_conf["lr_scheduler"]["scheduler"]
    gen = torch.Generator().manual_seed(manifest["seed"])
    loaders = {p:DataLoader(ds, batch_size=args.batch_size if p=="fit" else 1, shuffle=(p=="fit"),
        num_workers=0, collate_fn=NativeDataset.collate_fn, generator=gen if p=="fit" else None)
        for p, ds in datasets.items()}
    args.out.mkdir(parents=True)
    config = dict(arm=args.arm, epochs=args.epochs, batch_size=args.batch_size,
                  calibration_batch_size=1, eval_functional_attention_dropout=0.0,
                  max_steps=args.max_steps, model_kwargs=kwargs, manifest_sha256=digest(args.data/"manifest.json"),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  no_pretrained_weights=True, official_test_read=False,
                  initial_state_sha256=hashlib.sha256(b"".join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest())
    (args.out/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    best, best_epoch, total_steps = float("inf"), 0, 0
    start = time.time()
    metrics_path = args.out/"metrics.jsonl"
    def emit(record):
        record.update(elapsed_seconds=time.time()-start, **resource_snapshot())
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record)+"\n")
        print(json.dumps(record), flush=True)
    def save(path, epoch):
        temp = path.with_suffix(".tmp")
        torch.save(dict(state_dict=model.state_dict(), hyper_parameters=dict(model.hparams),
                        epoch=epoch, optimizer_steps=total_steps, magma_step=model.step,
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), config=config), temp)
        temp.replace(path)
    for epoch in range(args.epochs+1):
        if epoch:
            model.train()
            sums, n = Counter(), 0
            for batch in loaders["fit"]:
                batch = move_to_device(batch, torch.device("cuda"))
                optimizer.zero_grad(set_to_none=True)
                logs.clear()
                loss = model.training_step(batch, total_steps)["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {total_steps}")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); total_steps += 1
                b = len(batch["names"]); n += b
                for k,v in logs.items(): sums[k] += v*b
                if total_steps % 16 == 0 or total_steps == 1:
                    emit(dict(kind="step", epoch=epoch, step=total_steps, magma_weight=model.magma_weight_scheduler(),
                              grad_norm=float(grad), **logs))
                if args.max_steps and total_steps >= args.max_steps:
                    break
            emit(dict(kind="train_epoch", epoch=epoch, step=total_steps,
                      magma_weight=model.magma_weight_scheduler(), **{k:v/n for k,v in sums.items()}))
        model.eval()
        loss_sum, count = 0.0, 0
        losses, identifiers = [], []
        with torch.no_grad():
            for i,batch in enumerate(loaders["calibration"]):
                batch = move_to_device(batch, torch.device("cuda"))
                prediction = _forward_model(model, batch, is_decoy=False)
                per_query = model.inten_loss_fn(prediction["inten_pred_end_to_end"]["output"],
                    batch["inten_targs"], parent_mass=batch["precursor_mzs"], use_hun=True)["loss"]
                val_loss = per_query.mean()
                if not torch.isfinite(val_loss):
                    raise FloatingPointError("Nonfinite free-running calibration loss")
                b = len(batch["names"]); loss_sum += float(val_loss)*b; count += b
                losses.extend(per_query.cpu().tolist()); identifiers.extend(batch["names"])
        mean_loss = loss_sum/count
        np.savez_compressed(args.out/f"calibration_epoch{epoch}.npz",
                            identifiers=np.asarray(identifiers), loss=np.asarray(losses))
        emit(dict(kind="calibration", epoch=epoch, step=total_steps,
                  true_spectrum_cosine_loss=mean_loss, examples=count, include_magma=False))
        if mean_loss < best:
            best, best_epoch = mean_loss, epoch
            save(args.out/"best.ckpt", epoch)
        save(args.out/"last.ckpt", epoch)
        (args.out/"summary.json").write_text(json.dumps(dict(status="complete" if epoch==args.epochs or (args.max_steps and total_steps>=args.max_steps) else "running",
            last_epoch=epoch, best_epoch=best_epoch, best_calibration_loss=best,
            optimizer_steps=total_steps, magma_weight=model.magma_weight_scheduler(),
            elapsed_seconds=time.time()-start, **resource_snapshot()), indent=2)+"\n")
        if args.max_steps and total_steps >= args.max_steps:
            break


def pilot(args):
    """Serial child jobs: same seed/data/budget, no simultaneous GPU workers."""
    args.out.mkdir(exist_ok=True, parents=True)
    state_path = args.out/"supervisor.json"
    if state_path.exists():
        raise FileExistsError(state_path)
    state = dict(status="running", supervisor_pid=os.getpid(), arms=[])
    for arm in ("decay", "constant"):
        # Recheck actual GPU load immediately before each subprocess launch.
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv"], text=True)
        print(gpu, flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "train", "--data", str(args.data),
               "--out", str(args.out/arm), "--arm", arm, "--ms-pred", str(args.ms_pred)]
        with (args.out/f"{arm}.log").open("x") as log:
            child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            state.update(active_arm=arm, child_pid=child.pid)
            state_path.write_text(json.dumps(state, indent=2)+"\n")
            code = child.wait()
        state["arms"].append(dict(arm=arm, returncode=code))
        if code:
            state.update(status="failed", active_arm=None, child_pid=None)
            state_path.write_text(json.dumps(state, indent=2)+"\n")
            raise RuntimeError(f"{arm} failed with exit {code}; matched control not launched")
    state.update(status="complete", active_arm=None, child_pid=None)
    state_path.write_text(json.dumps(state, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--ms-pred", type=Path, default=ROOT/"external/ms-pred")
    p = sub.add_parser("prepare", parents=[shared])
    p.add_argument("--labels", type=Path, default=ROOT/"outputs/mpfir_fragment_cap_audit/magma_official_train20k/labels.jsonl")
    p.add_argument("--metadata", type=Path, default=ROOT/"outputs/mpfir_fragment_cap_audit/magma_official_train20k/sample.metadata.jsonl")
    p.add_argument("--official-train", type=Path, default=ROOT/"outputs/msalign_stage1_full_v15/train_full.metadata.jsonl")
    p.add_argument("--official-val", type=Path, default=ROOT/"outputs/msalign_stage1_full_v15/val_full.metadata.jsonl")
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--fit-rows", type=int, default=2048)
    p.add_argument("--calibration-rows", type=int, default=512)
    p.add_argument("--max-atoms", type=int, default=60)
    p.add_argument("--slots", type=int, default=200)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("train", parents=[shared])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", choices=("decay", "constant"), required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=0, help="plumbing-only smoke step limit")
    p = sub.add_parser("pilot", parents=[shared])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("DGLBACKEND", "pytorch")
    if args.command == "prepare": prepare(args)
    elif args.command == "train": train(args)
    else: pilot(args)


if __name__ == "__main__":
    main()
