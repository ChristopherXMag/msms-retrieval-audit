#!/usr/bin/env python3
"""Frozen official-validation end-to-end SCCF ensemble/pair spectrum controls.

No fitting or threshold selection. Recipient candidate pools stay fixed, but
shortlists and all spectrum-dependent scores are recomputed for each control.
The test partition is never accepted. Use --query-limit only for plumbing.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.analyze_stage1_logit_ensemble import paired_bootstrap, sha256_file
from tools.reranking.train_official_datasafe_pair_reranker import PairReranker, score_features
from tools.reranking.train_official_datasafe_score_fusion import endpoint
from tools.stage1.export_stage1_candidate_scores import tie_key
from tools.stage1.train_msalign_projection_smoke import (
    MoleculeEmbeddingStore, build_projection_model, checkpoint_model_config,
)

TIE_SEEDS = (20260926, 20260927, 20260928)
DONOR_SEEDS = (20260820, 20260821, 20260822)


def memory_guard():
    base = Path("/sys/fs/cgroup")
    if (base/"memory.max").exists():
        limit = (base/"memory.max").read_text().strip()
        if limit != "max":
            free = int(limit)-int((base/"memory.current").read_text())
            if free < 32*2**30:
                raise MemoryError(f"62-GiB allocation safety stop: free {free/2**30:.2f} GiB")


def candidate_order(identifier, rows, scores, seed):
    """Order by score then identity-keyed ties; independent of input slots."""
    keys = np.asarray([tie_key(identifier, int(row), seed) for row in rows], dtype=np.uint64)
    return np.lexsort((keys, -scores))


def uniform_tie_contributions(scores, target):
    """Exact expectations, not GT-first credit or finite tie-seed sampling."""
    value=scores[target]
    above=int(np.count_nonzero(scores>value))
    tied=int(np.count_nonzero(scores==value))
    ranks=np.arange(above+1,above+tied+1,dtype=np.float64)
    return np.asarray([(ranks<=1).mean(),(ranks<=5).mean(),(ranks<=20).mean(),(1/ranks).mean()])


def ordered_ranks(identifier, rows, stage_scores, pair_scores, selected, seed):
    order = candidate_order(identifier, rows, stage_scores, seed)
    baseline = int(np.flatnonzero(order == 0)[0]) + 1
    hit = np.flatnonzero(selected == 0)
    if not len(hit):
        return baseline, baseline
    local_order = candidate_order(identifier, rows[selected], pair_scores, seed)
    return baseline, int(np.flatnonzero(local_order == hit[0])[0]) + 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "outputs/stage2_official_datasafe"
    p.add_argument("--score-summary", type=Path, default=base / "stage1_candidate_scores/val_three_seed_logits.summary.json")
    p.add_argument("--pair-summary", type=Path, default=base / "pair_reranker/formal_top5/summary.json")
    p.add_argument("--donor-npz", type=Path, default=ROOT / "outputs/stage1_submission_0926/runs/full_r2_block_l2/seed20260621/blind_controls/spectrum_permutation_exact_mass.per_query.npz")
    p.add_argument("--train-dreams", type=Path, default=ROOT / "outputs/run_k_official_datasafe_seed20260621/train.no_head.npy")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--query-limit", type=int, default=0)
    p.add_argument("--projection-batch-size", type=int, default=2048)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--bootstrap-resamples", type=int, default=10000)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return p.parse_args()


def checked_path(entry):
    path = ROOT / entry["path"]
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"Frozen input digest mismatch: {path}")
    return path


@torch.inference_mode()
def project_molecules_float(rows, store, model, device, batch_size):
    output = torch.empty((len(rows), model.project_spec(torch.zeros(1, 1024, device=device)).shape[1]))
    for start in tqdm(range(0, len(rows), batch_size), desc="Project validation candidate union"):
        memory_guard()
        logical = torch.from_numpy(rows[start:start + batch_size].astype(np.int64))
        ordered, inverse = store.canonicalize_unique_indices(logical, torch.arange(len(logical)))
        output[start:start + len(logical)] = model.project_mol(store.gather(ordered).to(device)).cpu()[inverse]
    return output


@torch.inference_mode()
def evaluate_mode(name, query_rows, source_rows, constant, models, projections,
                  spec_real, dreams, source_scores, arrays, local_map,
                  pair_models, cached_real_spec, device, batch_size,
                  cached_real_molecule=None, cached_real_top5=None):
    results = {seed: (np.empty(len(query_rows), np.int32), np.empty(len(query_rows), np.int32)) for seed in TIE_SEEDS}
    tie_queries = 0
    score_max_diff = 0.0
    order_parity = True
    expected_ensemble=np.empty((len(query_rows),4),np.float64)
    expected_pair=np.empty_like(expected_ensemble)
    boundary_ties=0
    for start in tqdm(range(0, len(query_rows), batch_size), desc=name):
        memory_guard()
        current = query_rows[start:start + batch_size]
        count = len(current)
        rows = [arrays["candidate_rows"][arrays["indptr"][q]:arrays["indptr"][q + 1]] for q in current]
        width = max(map(len, rows))
        padded = np.stack([np.pad(r, (0, width - len(r)), mode="edge") for r in rows])
        local = torch.from_numpy(local_map[padded]).long()
        if constant is None:
            spectra = [s[torch.from_numpy(source_rows[start:start + count]).long()].to(device) for s in spec_real]
        else:
            spectra = [model.project_spec(torch.from_numpy(constant).to(device).unsqueeze(0)).expand(count, -1) for model in models]
        per_seed = []
        for model, mol, spec in zip(models, projections, spectra, strict=True):
            scores = torch.einsum("bd,bkd->bk", spec, mol[local].to(device)) / model.temperature()
            per_seed.append(scores.cpu().numpy())
        score_sets = np.stack(per_seed, axis=-1)
        if name == "real":
            for i, q in enumerate(current):
                a, b = arrays["indptr"][q:q + 2]
                exact = source_scores[:, a:b].T
                score_max_diff = max(score_max_diff, float(np.max(np.abs(score_sets[i, :len(rows[i])] - exact))))
                score_sets[i, :len(rows[i])] = exact
        # Exported SCCF ensemble uses float64 accumulation then float32 storage.
        # Float32-only mean shifts up to one ulp and can flip near-tied pair ranks.
        ensemble = score_sets.astype(np.float64).mean(axis=-1).astype(np.float32)
        for tie_seed in TIE_SEEDS:
            selected = np.stack([candidate_order(str(arrays["identifiers"][q]), rows[i], ensemble[i, :len(rows[i])], tie_seed)[:5] for i, q in enumerate(current)])
            top_local = local.gather(1, torch.from_numpy(selected))
            mol_pair = projections[0][top_local].half().float().to(device)
            spec_pair = spectra[0].half().float()
            if name == "real":
                spec_pair = cached_real_spec[torch.from_numpy(current)].float().to(device)
                if cached_real_molecule is not None:
                    frozen = cached_real_top5
                    actual_rows = np.take_along_axis(padded, selected, axis=1)
                    if not np.array_equal(actual_rows, frozen["candidate_global_rows"][current].numpy()):
                        raise ValueError("Frozen real top5 membership/order drift")
                    mol_pair = cached_real_molecule[frozen["candidate_local_indices"][current].long()].float().to(device)
            top_scores = np.take_along_axis(score_sets, selected[:, :, None], axis=1)
            base = torch.from_numpy(np.take_along_axis(ensemble, selected, axis=1)).to(device)
            features = score_features(torch.from_numpy(top_scores).to(device), base)
            learned = np.mean(np.stack([m(spec_pair, mol_pair, features, base).cpu().numpy()
                                        for m in pair_models]), axis=0)
            if tie_seed == TIE_SEEDS[0]:
                # Actually permute every query's local candidate inputs and
                # recompute relative features, not just the first batch.
                perms = np.stack([np.random.default_rng(20261002+int(q)).permutation(5) for q in current])
                p = torch.from_numpy(perms).to(device)
                shuffled_mol = mol_pair.gather(1, p[:, :, None].expand(-1, -1, mol_pair.shape[-1]))
                shuffled_base = base.gather(1, p)
                shuffled_raw = torch.from_numpy(np.take_along_axis(top_scores, perms[:, :, None], axis=1)).to(device)
                shuffled_features = score_features(shuffled_raw, shuffled_base)
                shuffled_learned = np.mean(np.stack([m(spec_pair, shuffled_mol, shuffled_features, shuffled_base).cpu().numpy()
                                                     for m in pair_models]), axis=0)
                if not np.allclose(shuffled_learned, np.take_along_axis(learned, perms, axis=1), atol=5e-5, rtol=0):
                    raise ValueError("Shared scorer/feature candidate permutation failed")
            for i, q in enumerate(current):
                identifier = str(arrays["identifiers"][q])
                baseline, reranked = ordered_ranks(identifier, rows[i], ensemble[i, :len(rows[i])], learned[i], selected[i], tie_seed)
                results[tie_seed][0][start + i] = baseline
                results[tie_seed][1][start + i] = reranked
                if tie_seed == TIE_SEEDS[0]:
                    stage_order=candidate_order(identifier,rows[i],ensemble[i,:len(rows[i])],tie_seed)
                    boundary_ties+=int(ensemble[i,stage_order[4]]==ensemble[i,stage_order[5]])
                    expected_ensemble[start+i]=uniform_tie_contributions(ensemble[i,:len(rows[i])],0)
                    target_local=np.flatnonzero(selected[i]==0)
                    expected_pair[start+i]=expected_ensemble[start+i]
                    if len(target_local):
                        expected_pair[start+i]=uniform_tie_contributions(learned[i],int(target_local[0]))
                    target = ensemble[i, 0]
                    tie_queries += int(np.count_nonzero(ensemble[i, :len(rows[i])] == target) > 1)
                    perm = np.random.default_rng(20261001 + int(q)).permutation(len(rows[i]))
                    original = candidate_order(identifier, rows[i], ensemble[i, :len(rows[i])], tie_seed)
                    shuffled = candidate_order(identifier, rows[i][perm], ensemble[i, :len(rows[i])][perm], tie_seed)
                    order_parity &= bool(np.array_equal(rows[i][original], rows[i][perm][shuffled]))
    if not order_parity or score_max_diff > 5e-5:
        raise ValueError(f"Frozen-score/order parity failed: {score_max_diff}, {order_parity}")
    from tools.analysis.analyze_stage1_logit_ensemble import metrics,grouped
    audit={"stage1_gt_exact_tie_queries":tie_queries,"candidate_order_parity":order_parity,
           "stage1_top5_boundary_tie_queries":boundary_ties,
           "pair_uniform_scope":"local score ties conditional on primary identity-keyed shortlist; full shortlist tie variation reported across three tie seeds",
           "real_logit_max_abs_diff":score_max_diff,"uniform_tie_query":
           {"ensemble":metrics(expected_ensemble),"pair":metrics(expected_pair)},
           "uniform_tie_connectivity_balanced":
           {"ensemble":metrics(grouped(expected_ensemble,arrays["connectivities"][query_rows])[2]),
            "pair":metrics(grouped(expected_pair,arrays["connectivities"][query_rows])[2])}}
    return results,audit


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite control output: {args.output_dir}")
    if args.query_limit < 0 or args.eval_batch_size < 1 or args.projection_batch_size < 1:
        raise ValueError("Invalid bounded settings")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    source = json.loads(args.score_summary.read_text())
    pair_summary = json.loads(args.pair_summary.read_text())
    cache_summary = json.loads(checked_path(pair_summary["inputs"]["cache_summary"]).read_text())
    for field in ("val_top5", "candidate_projection"):
        checked_path(pair_summary["inputs"][field])
    checked_path(cache_summary["fingerprint_cache"])
    checked_path(cache_summary["val_scores"])
    checked_path(cache_summary["train_dreams"])
    checked_path(cache_summary["val_dreams"])
    # Original natural-loss artifact predates the optional balance switch.
    if pair_summary["official_test_read"] or pair_summary["args"].get("training_balance", "natural") != "natural":
        raise ValueError("Expected frozen validation-only natural pair")
    for path in (source["metadata"], source["dreams"], source["npz"], str(args.train_dreams)):
        if "test" in Path(path).name.lower():
            raise ValueError("No test inputs permitted")
    arrays = dict(np.load(ROOT / source["npz"]))
    records = [json.loads(line) for line in (ROOT / source["metadata"]).read_text().splitlines() if line]
    if [str(r["identifier"]) for r in records] != arrays["identifiers"].tolist():
        raise ValueError("Recipient metadata order drift")
    dreams = np.load(ROOT / source["dreams"], mmap_mode="r")
    train = np.load(args.train_dreams, mmap_mode="r")
    train_mean = np.zeros(train.shape[1], np.float64)
    for start in range(0, len(train), 4096):
        train_mean += train[start:start + 4096].sum(0, dtype=np.float64)
    train_mean = (train_mean / len(train)).astype(np.float32)
    donor = dict(np.load(args.donor_npz))
    recipients = donor["recipient_rows"].astype(np.int64)
    groups = arrays["connectivities"]
    for seed in DONOR_SEEDS:
        donors = donor[f"seed{seed}_donor_rows"].astype(np.int64)
        if set(donors) != set(recipients) or len(np.unique(donors)) != len(donors) or np.any(groups[donors] == groups[recipients]):
            raise ValueError("Invalid closed connectivity derangement")
        for q, d in zip(recipients, donors, strict=True):
            if records[q]["adduct"] != records[d]["adduct"] or round(float(records[q]["parent_mass"]), 6) != round(float(records[d]["parent_mass"]), 6):
                raise ValueError("Donor mass/adduct mismatch")
    count = min(args.query_limit or len(records), len(records))
    query_rows = np.arange(count)
    selected_candidates = np.unique(arrays["candidate_rows"][:arrays["indptr"][count]])
    local_map = np.full(int(arrays["candidate_rows"].max()) + 1, -1, np.int32)
    local_map[selected_candidates] = np.arange(len(selected_candidates))
    models, projections, spec_real = [], [], []
    for entry in source["checkpoints"]:
        payload = torch.load(checked_path(entry), weights_only=False, map_location="cpu")
        # Never preload the 7.2M-row CSR payload (~30 GiB) for a small control
        # smoke or a 62-GiB server. Batch materialization uses existing mmap.
        store = MoleculeEmbeddingStore(None, "mean", ROOT / source["fingerprint_cache"], additional_fingerprint_caches=[], preload_sparse_features=False, mol_blocks=source["molecule_blocks"])
        model = build_projection_model(spec_dim=dreams.shape[1], mol_dim=store.dim, mol_block_dims=store.block_dims, mol_block_names=store.block_names, config=checkpoint_model_config(payload)).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        models.append(model)
        projections.append(project_molecules_float(selected_candidates, store, model, device, args.projection_batch_size))
        with torch.inference_mode():
            spec_real.append(torch.cat([model.project_spec(torch.from_numpy(np.array(dreams[start:start + 2048], copy=True)).to(device)).cpu() for start in range(0, len(dreams), 2048)]))
        del store, payload
    frozen_val = torch.load(str(ROOT / pair_summary["args"]["val_top5"]), weights_only=False, mmap=True, map_location="cpu")
    frozen_molecule = torch.load(str(checked_path(pair_summary["inputs"]["candidate_projection"])),
                                weights_only=False, mmap=True, map_location="cpu")["embeddings"]
    pair_models = []
    for entry in pair_summary["checkpoint_files"]:
        payload = torch.load(checked_path(entry), weights_only=False, map_location="cpu")
        cfg = pair_summary["args"]
        model = PairReranker(1024, 11, cfg["hidden_dim"], cfg["bottleneck_dim"], cfg["dropout"]).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        pair_models.append(model)
    # Preserve real-vector cache numerics; independent controls use identical fp16 storage.
    args.output_dir.mkdir(parents=True)
    modes = [("real", query_rows, query_rows, None), ("train_mean", query_rows, query_rows, train_mean), ("zero", query_rows, query_rows, np.zeros_like(train_mean))]
    mask = recipients < count
    for seed in DONOR_SEEDS:
        if np.any(mask):
            modes.append((f"wrong_seed{seed}", recipients[mask], donor[f"seed{seed}_donor_rows"][mask], None))
    report = {"format": "official_datasafe_complete_pair_controls_v1", "official_test_read": False, "plumbing_only": bool(args.query_limit), "protocol": {"training": "none; frozen checkpoints only", "shortlist": "recomputed per input control; fixed recipient full candidate pool", "tie_seeds": TIE_SEEDS, "donor_seeds": DONOR_SEEDS, "candidate_slot_features": False, "upstream_stage1_seeds": [20260621, 20260622, 20260623], "scorer_seeds": cfg["model_seeds"], "no_threshold_selection": True}, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "inputs": {}, "modes": {}}
    paths = [args.score_summary, args.pair_summary, args.donor_npz, args.train_dreams, ROOT / source["npz"], ROOT / source["dreams"], ROOT / source["metadata"], ROOT / source["fingerprint_cache"], ROOT / pair_summary["args"]["val_top5"], ROOT / pair_summary["args"]["candidate_projection"], Path(__file__)]
    report["inputs"] = {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}
    saved = {}
    outputs = {}
    for name, queries, sources, constant in modes:
        ranks, audit = evaluate_mode(name, queries, sources, constant, models, projections, spec_real, dreams, arrays["scores"], arrays, local_map, pair_models, frozen_val["spec_embeddings"], device, args.eval_batch_size,
                                     frozen_molecule, frozen_val)
        outputs[name] = (queries, ranks)
        per_tie = {str(seed): endpoint(b, r, groups[queries]) for seed, (b, r) in ranks.items()}
        if name == "real":
            original = np.load(ROOT / pair_summary["rank_file"]["path"] if isinstance(pair_summary["rank_file"], dict) else ROOT / pair_summary["rank_file"])
            for b, r in ranks.values():
                if not np.array_equal(b, original["baseline_ranks"][:count]) or not np.array_equal(r, original["reranked_ranks"][:count]):
                    raise ValueError("Frozen complete-system real rank parity failed")
        primary = ranks[TIE_SEEDS[0]]
        report["modes"][name] = {"audit": audit, "per_tie_seed": per_tie, "incremental_pair_bootstrap": paired_bootstrap(primary[0], primary[1], groups[queries], args.bootstrap_resamples, 20261001)}
        saved[f"{name}_query_rows"] = queries
        for seed, (b, r) in ranks.items():
            saved[f"{name}_tie{seed}_ensemble_ranks"] = b
            saved[f"{name}_tie{seed}_pair_ranks"] = r
        print(name, json.dumps(per_tie[str(TIE_SEEDS[0])]["reranked_query_weighted"]), flush=True)
        (args.output_dir / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, (queries, ranks) in outputs.items():
        if name == "real":
            continue
        real = outputs["real"][1][TIE_SEEDS[0]][1][queries]
        report["modes"][name]["complete_system_real_minus_control_bootstrap"] = paired_bootstrap(ranks[TIE_SEEDS[0]][1], real, groups[queries], args.bootstrap_resamples, 20261001)
    np.savez_compressed(args.output_dir / "per_query_ranks.npz", identifiers=arrays["identifiers"][:count], connectivities=groups[:count], **saved)
    report["rank_sha256"] = sha256_file(args.output_dir / "per_query_ranks.npz")
    report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    report["host_peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
