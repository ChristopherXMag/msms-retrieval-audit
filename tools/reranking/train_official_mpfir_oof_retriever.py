#!/usr/bin/env python3
"""Fit fresh SCCF fold projections and mine natural official-train top5 lists.

Frozen data-safe DreaMS / radius-2 SCCF, one full complementary-train epoch,
GT + 127 random mass-pool negatives. No released checkpoint initialization,
validation selection, or official-test input. This is a single-seed OOF mining
retriever, not the publication three-seed ensemble/pair endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from glmr.candidate_indices import CandidateIndexStore, CandidateNegativeExclusion
from tools.reranking.train_official_magma_forward import rows, digest
from tools.reranking.train_official_mpfir_v2_alignment import seed_all
from tools.reranking.audit_official_datasafe_pair_controls import candidate_order
from tools.stage1.train_msalign_projection_smoke import MoleculeEmbeddingStore, build_projection_model

BLOCKS = ["count_morgan", "count_atom_pair", "rdkit_path_count", "maccs"]
CONFIG = dict(hidden_dim=2048, proj_dim=1024, dropout=0.2, temperature=0.07,
              head_type="mlp", mol_block_normalization="block_l2", mol_head_type="shared_mlp")


def memory_state():
    base = Path("/sys/fs/cgroup")
    statistics={line.split()[0]:int(line.split()[1]) for line in (base/"memory.stat").read_text().splitlines()}
    maximum=(base/"memory.max").read_text().strip()
    current=int((base/"memory.current").read_text())
    # Standard cgroup available-memory estimate: inactive file pages are
    # reclaimable. Never count anonymous pages or swap as available memory.
    available=int(maximum)-current+statistics.get("inactive_file",0) if maximum!="max" else None
    return dict(rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                cgroup_gib=current / 2**30,cgroup_anon_gib=statistics.get("anon",0)/2**30,
                cgroup_available_gib=available/2**30 if available is not None else None,
                cuda_peak_gib=torch.cuda.max_memory_allocated() / 2**30)


def memory_guard():
    state=memory_state()
    if state["cgroup_available_gib"] is not None and state["cgroup_available_gib"]<32:
        raise MemoryError(f"Container available memory below 32 GiB: {state}")
    host={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    if host['MemAvailable']<32*2**30 or host.get('SwapTotal',0)-host.get('SwapFree',0)>0:
        raise MemoryError("Host memory/swap safety gate failed")


def tensor_digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


@torch.no_grad()
def rank_queries(model, metadata, query_rows, official_lookup, spectra, candidates, store,
                 endpoint_denylist, model_name, directory, device):
    """GT is an evaluation target only; selection is a permuted natural cap256."""
    model.eval()
    output = []
    for start in range(0, len(query_rows), 32):
        indices = query_rows[start:start+32]
        records = [metadata[i] for i in indices]
        full = [candidates.get(row["identifier"], 256) for row in records]
        # Local TRAIN lists obey the outer calibration/evaluation negative
        # denylist. Held-out evaluation lists retain the unmodified official pool.
        full = [endpoint_denylist.filter_candidates(pool)[0] if row["pilot_partition"] == "fit" else pool
                for row, pool in zip(records, full, strict=True)]
        permutations = [np.random.default_rng(20261010 + int(i)).permutation(len(pool))
                        for i, pool in zip(indices, full, strict=True)]
        shuffled = [np.asarray(pool)[order] for pool, order in zip(full, permutations, strict=True)]
        width = max(map(len, full))
        padded = np.asarray([list(pool) + [pool[0]]*(width-len(pool)) for pool in shuffled])
        mask = np.arange(width)[None] < np.asarray(list(map(len, full)))[:, None]
        row_indices = [official_lookup[row["identifier"]] for row in records]
        spec = torch.from_numpy(np.asarray(spectra[row_indices]).copy()).to(device)
        index_tensor = torch.from_numpy(padded).long()
        unique, inverse = torch.unique(index_tensor.flatten(), sorted=True, return_inverse=True)
        unique, inverse = store.canonicalize_unique_indices(unique, inverse)
        molecule = model.project_mol(store.gather(unique).to(device))
        molecule = molecule[inverse.to(device)].reshape(len(indices), width, -1)
        spectrum = model.project_spec(spec)
        scores = (torch.einsum("bd,bcd->bc", spectrum, molecule) / model.temperature()).cpu().numpy()
        scores[~mask] = -np.inf
        for j, (index, record) in enumerate(zip(indices, records, strict=True)):
            pool = shuffled[j]
            values = scores[j, :len(pool)]
            order = candidate_order(record["identifier"], pool, values, 20260926)
            gt_global = int(full[j][0])
            ranked = pool[order]
            rank = int(np.flatnonzero(ranked == gt_global)[0]) + 1
            top5 = ranked[:5].astype(np.int32)
            if len(top5) != 5:
                raise ValueError("Natural candidate pool has fewer than five entries")
            position = np.flatnonzero(top5 == gt_global)
            output.append(dict(query_row=index, identifier=record["identifier"],
                connectivity=record["connectivity"], partition=record["pilot_partition"],
                mining_model=model_name, candidate_global_rows=top5.tolist(),
                base_scores=values[order[:5]].tolist(), baseline_rank=rank,
                gt_position=int(position[0]) if len(position) else -1,
                candidate_count=len(pool), gt_global_row=gt_global))
        if start % 256 == 0:
            print(json.dumps(dict(stage="mining", model=model_name, queries=min(start+32,len(query_rows)),
                                  **memory_state())), flush=True)
    path = directory / "top5.jsonl"
    with path.open("w") as handle:
        for record in output:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    ranks = np.asarray([row["baseline_rank"] for row in output])
    return dict(queries=len(output), top5_reachable=int((ranks<=5).sum()),
                recall1_pct=100*float((ranks==1).mean()), recall5_pct=100*float((ranks<=5).mean()),
                recall20_pct=100*float((ranks<=20).mean()), mrr_pct=100*float((1/ranks).mean()),
                top5_sha256=digest(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", choices=("mining0", "mining1", "endpoint"), required=True)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--skip-mining", action="store_true",
                        help="Save the fresh fit; defer ranking to a matched multi-seed exporter")
    args = parser.parse_args()
    args.data=args.data.resolve()
    args.data.relative_to(ROOT/"outputs")
    started = time.time()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Fresh host CUDA check required")
    directory = args.data / (args.model + (f"_smoke{args.smoke_steps}" if args.smoke_steps else ""))
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir()
    metadata_path = ROOT / "outputs/msalign_stage1_full_v15/train_full.metadata.jsonl"
    official = list(rows(metadata_path))
    if any(row["fold"] != "train" for row in official):
        raise ValueError("Non-train input")
    lookup = {row["identifier"]: index for index, row in enumerate(official)}
    group_path = args.data / "official_train_connectivities.npy"
    groups = np.load(group_path)
    if len(groups) != len(official):
        raise ValueError("Official identity row drift")
    manifest = args.data / f"denylist_{args.model}/manifest.json"
    denylist = CandidateNegativeExclusion(manifest)
    excluded = set(denylist.metadata["additional_connectivities"])
    fit_indices = np.flatnonzero(~np.isin(groups, list(excluded)))
    if set(groups[fit_indices]) & excluded:
        raise ValueError("OOF GT leakage")
    candidate_path = ROOT / "outputs/run_k_full_train/candidate_index_full_v15_mass_cap256_full_order/manifest.json"
    candidates = CandidateIndexStore(candidate_path)
    denylist.validate_candidate_store(candidates)
    endpoint_denylist = CandidateNegativeExclusion(args.data / "denylist_endpoint/manifest.json")
    spectrum_path = ROOT / "outputs/run_k_official_datasafe_seed20260621/train.no_head.npy"
    provenance = json.loads(spectrum_path.with_suffix(".summary.json").read_text())
    if provenance["dreams_ckpt_sha256"] != "d9b4e8b920ff367370c5a3fa154eaa951f90bafa58d5224a594424d9c54b1832":
        raise ValueError("Data-safe DreaMS provenance mismatch")
    spectra = np.load(spectrum_path, mmap_mode="r")
    if spectra.shape != (len(official), 1024):
        raise ValueError("Global spectrum row mismatch")
    fingerprint = ROOT / "outputs/stage1_submission_0926/assets/full_sparse_direct/manifest_historical_projection_order.json"
    store = MoleculeEmbeddingStore(None, "mean", fingerprint, mol_blocks=BLOCKS)
    if store.smiles_sha256 != candidates.molecule_smiles_sha256:
        raise ValueError("SCCF/candidate row drift")
    seed_all(args.seed)
    model = build_projection_model(spec_dim=1024, mol_dim=store.dim,
        mol_block_dims=store.block_dims, mol_block_names=BLOCKS, config=CONFIG).to(device)
    model.log_temperature.requires_grad_(False)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                 lr=1e-4, weight_decay=1e-4)
    order = np.random.default_rng(args.seed).permutation(fit_indices)
    rng = random.Random(args.seed+1)
    total, count, negative_count = 0.0, 0, 0
    model.train()
    for step, start in enumerate(range(0, len(order), 128)):
        if args.smoke_steps and step >= args.smoke_steps:
            break
        if step % 50 == 0:
            memory_guard()
        batch = order[start:start+128]
        pools = [denylist.filter_candidates(candidates.get(official[int(i)]["identifier"], 256))[0]
                 for i in batch]
        if any(len(pool)<2 for pool in pools):
            raise ValueError("Fold exclusion left no negative")
        # Match the existing Stage1 sampler: all negatives when the pool is
        # small, otherwise 127 distinct negatives. Padding never enters CE.
        chosen = [[pool[0]] + (rng.sample(pool[1:], 127) if len(pool)>127 else pool[1:])
                  for pool in pools]
        width=max(map(len,chosen))
        candidate_rows = torch.tensor([pool+[pool[0]]*(width-len(pool)) for pool in chosen], dtype=torch.long)
        candidate_mask=torch.arange(width)[None,:]<torch.tensor(list(map(len,chosen)))[:,None]
        if bool(denylist.mask[candidate_rows[:, 1:].numpy()][candidate_mask[:,1:].numpy()].any()):
            raise ValueError("OOF negative leakage")
        spec = torch.from_numpy(np.asarray(spectra[batch]).copy()).to(device)
        mol = store.gather(candidate_rows).to(device)
        spec = model.project_spec(spec)
        mol = model.project_mol(mol.reshape(-1, store.dim)).reshape(len(batch), width, -1)
        logits = torch.einsum("bd,bcd->bc", spec, mol) / model.temperature()
        logits=logits.masked_fill(~candidate_mask.to(device),float("-inf"))
        loss = F.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(batch); count += len(batch)
        negative_count += sum(256-len(pool) for pool in pools)
        if step % 50 == 0:
            print(json.dumps(dict(stage="fit", model=args.model, step=step,
                rows=count, planned_rows=len(order), mean_loss=total/count,
                elapsed_seconds=round(time.time()-started,1), **memory_state())), flush=True)
    checkpoint = directory / "checkpoint.pt"
    torch.save(dict(model_state_dict=model.state_dict(), model_config=CONFIG,
                    seed=args.seed, initialization="fresh random", epochs=1,
                    excluded_connectivities=sorted(excluded)), checkpoint)
    queries = list(rows(args.data / "labels.jsonl"))
    if args.model == "endpoint":
        query_indices = [i for i, row in enumerate(queries) if row["pilot_partition"] != "fit"]
    else:
        fold = int(args.model[-1])
        query_indices = [i for i, row in enumerate(queries)
                         if row["pilot_partition"] == "fit" and row["inner_fold"] == fold]
    if args.smoke_steps:
        query_indices = query_indices[:32]
    if any(queries[i]["connectivity"] not in excluded for i in query_indices):
        raise ValueError("Mining query seen by the upstream retriever")
    if args.skip_mining:
        audit = {"status": "deferred_to_matched_multiseed_export", "queries": 0}
    else:
        audit = rank_queries(model, queries, query_indices, lookup, spectra, candidates, store,
                             endpoint_denylist, args.model, directory, device)
    summary = dict(model=args.model, seed=args.seed, epochs=1, fresh_initialization=True,
        training_rows=count, complementary_train_rows=len(order), training_loss=total/count,
        train_rows_sha256=tensor_digest(order), excluded_connectivities=len(excluded),
        negative_excluded_occurrences=negative_count, oof_gt_overlap=0, negative_overlap=0,
        architecture=CONFIG, molecule_blocks=BLOCKS, workers=0, feature_preload=False,
        official_test_read=False, official_validation_read=False,
        plumbing_only=bool(args.smoke_steps), mining=audit,
        elapsed_seconds=time.time()-started, resources=memory_state(),
        inputs={str(path.relative_to(ROOT)):digest(path) for path in
                (metadata_path, group_path, manifest, fingerprint, candidate_path, spectrum_path)},
        checkpoint_sha256=digest(checkpoint), code_sha256=digest(Path(__file__)))
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
