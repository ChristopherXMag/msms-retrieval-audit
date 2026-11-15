#!/usr/bin/env python3
"""Export train-only peak-specific masked multi-positive MAGMa targets.

Preparation only: no model fitting, inference, retrieval or test inputs. Unlike
the forward fragment-set dataset, this keeps the peak-to-mask relation and mass
variants. Unknown CE/instrument is retained, not imputed as an encoder feature.
"""
from __future__ import annotations

import argparse
from collections import Counter
import heapq
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from rdkit import Chem,rdBase
from tools.reranking.train_official_magma_forward import rows,digest,key,partition
from tools.reranking.train_glacier_run_k_retrieval import molecule_record


def peak_targets(label,meta,max_atoms=60):
    if meta["fold"]!="train":raise ValueError("not_official_train")
    if label["adduct"]!=meta["adduct"]:raise ValueError("adduct_mismatch")
    if meta["adduct"] not in ("[M+H]+","[M+Na]+"):raise ValueError("unsupported_adduct")
    smiles=label["magma_canonical_smiles"]
    record=molecule_record(smiles,max_atoms)
    reference=molecule_record(meta["canonical_isomeric_smiles"],max_atoms)
    if record.canonical_isomeric_smiles!=reference.canonical_isomeric_smiles:
        raise ValueError("structure_mismatch")
    n=Chem.RemoveHs(Chem.MolFromSmiles(smiles)).GetNumAtoms()
    if n!=label["num_atoms"]:raise ValueError("atom_count_mismatch")
    source_peaks=label.get("peaks",[])
    if not 0<len(source_peaks)<=100:raise ValueError("invalid_top100_peak_count")
    masks=[];lookup={};peaks=[]
    for peak in source_peaks:
        mz=float(peak["mz"]);intensity=float(peak["relative_intensity"])
        if not math.isfinite(mz) or mz<=0 or not math.isfinite(intensity) or intensity<0:
            raise ValueError("invalid_spectrum")
        positives=[];variants=[]
        for match in peak.get("matches",[]):
            indices=match.get("atom_indices",[])
            if not indices or any(type(i) is not int or not 0<=i<n for i in indices):
                raise ValueError("invalid_fragment_atom_index")
            mask=tuple(sorted(set(indices)))
            if mask not in lookup:
                lookup[mask]=len(masks);masks.append(list(mask))
            index=lookup[mask]
            if index not in positives:positives.append(index)
            # Distinct H shifts / mass variants remain separate explanations,
            # even if their atom masks are equal. Do not select one as truth.
            variant=dict(match);variant["mask_index"]=index
            variants.append(variant)
        if "assigned" in peak and bool(peak["assigned"])!=bool(positives):
            raise ValueError("assigned_match_disagreement")
        peaks.append(dict(mz=mz,relative_intensity=intensity,
            source_peak_index=peak.get("source_peak_index"),token_index=peak.get("token_index"),
            positive_mask_indices=positives,fragment_supervision_mask=bool(positives),
            explanations=variants))
    if not masks:raise ValueError("no_matched_fragment_masks")
    if not any(p["relative_intensity"]>0 for p in peaks):raise ValueError("invalid_spectrum")
    return dict(identifier=label["identifier"],connectivity=record.connectivity,
        smiles=smiles,num_atoms=n,adduct=meta["adduct"],precursor_mz=meta["precursor_mz"],
        # Retained provenance only; not declared model inputs in this exporter.
        collision_energy=meta.get("collision_energy"),instrument=meta.get("instrument_type"),
        fragment_masks=masks,peaks=peaks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    base=ROOT/"outputs/mpfir_fragment_cap_audit/magma_official_train20k"
    p.add_argument("--labels",type=Path,default=base/"labels.jsonl")
    p.add_argument("--metadata",type=Path,default=base/"sample.metadata.jsonl")
    p.add_argument("--official-train",type=Path,default=ROOT/"outputs/msalign_stage1_full_v15/train_full.metadata.jsonl")
    p.add_argument("--official-val",type=Path,default=ROOT/"outputs/msalign_stage1_full_v15/val_full.metadata.jsonl")
    p.add_argument("--seed",type=int,default=20261001)
    p.add_argument("--fit-rows",type=int,default=2048)
    p.add_argument("--calibration-rows",type=int,default=512)
    p.add_argument("--max-atoms",type=int,default=60)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    for field in ("labels","metadata","official_train","official_val","out"):
        path=getattr(args,field).resolve()
        path.relative_to(ROOT)  # Artifacts intentionally store repo-relative paths.
        setattr(args,field,path)
    if args.out.exists():raise FileExistsError(args.out)
    if min(args.fit_rows,args.calibration_rows,args.max_atoms)<=0:raise ValueError("Invalid bounds")
    inputs={name:getattr(args,name.replace("-","_")) for name in ("labels","metadata","official-train","official-val")}
    if any("test" in path.name.lower() for path in inputs.values()):raise ValueError("No test inputs")
    metadata={r["identifier"]:r for r in rows(args.metadata)}
    official={r["identifier"]:r["canonical_isomeric_smiles"] for r in rows(args.official_train) if r["fold"]=="train"}
    validation=list(rows(args.official_val))
    val_ids={r["identifier"] for r in validation}
    val_conn={molecule_record(r["canonical_isomeric_smiles"],1000).connectivity for r in validation}
    pools={"fit":[],"calibration":[]};limits={"fit":args.fit_rows,"calibration":args.calibration_rows}
    excluded=Counter();eligible=Counter();seen=set();all_counts=Counter()
    for label in rows(args.labels):
        ident=label["identifier"]
        if ident in seen:raise ValueError("duplicate_identifier")
        seen.add(ident)
        if ident not in official or ident in val_ids:raise ValueError("label_outside_official_train")
        meta=metadata[ident]
        if meta["canonical_isomeric_smiles"]!=official[ident]:raise ValueError("metadata_identity_drift")
        try:item=peak_targets(label,meta,args.max_atoms)
        except ValueError as exc:excluded[str(exc)]+=1;continue
        if item["connectivity"] in val_conn:raise ValueError("train_val_connectivity_leak")
        part=partition(args.seed,item["connectivity"]);eligible[part]+=1
        all_counts["peaks"]+=len(item["peaks"])
        all_counts["matched_peaks"]+=sum(x["fragment_supervision_mask"] for x in item["peaks"])
        all_counts["ambiguous_mask_peaks"]+=sum(len(x["positive_mask_indices"])>1 for x in item["peaks"])
        all_counts["missing_ce_queries"]+=int(item["collision_energy"] is None)
        rank=int(key(args.seed,ident),16)
        heap=pools[part];entry=(-rank,ident,item)
        if len(heap)<limits[part]:heapq.heappush(heap,entry)
        elif rank<-heap[0][0]:heapq.heapreplace(heap,entry)
    if any(len(pools[part])!=limits[part] for part in pools):raise ValueError("insufficient_eligible_rows")
    selected={part:[e[2] for e in sorted(heap,key=lambda e:-e[0])] for part,heap in pools.items()}
    if {r["connectivity"] for r in selected["fit"]}&{r["connectivity"] for r in selected["calibration"]}:
        raise ValueError("fit_calibration_connectivity_overlap")
    args.out.mkdir(parents=True)
    artifacts={}
    for part,items in selected.items():
        path=args.out/f"{part}.jsonl"
        with path.open("x") as handle:
            for item in items:handle.write(json.dumps(item,separators=(",",":"))+"\n")
        artifacts[part]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path),queries=len(items),
            connectivities=len({r["connectivity"] for r in items}),
            missing_ce_queries=sum(r["collision_energy"] is None for r in items))
    report=dict(protocol="official-train peak-specific masked multi-positive MAGMa targets; preparation only",
        official_test_read=False,training_executed=False,retrieval_evidence=False,
        rdkit_version=rdBase.rdkitVersion,seed=args.seed,max_atoms=args.max_atoms,
        contract=dict(unmatched_peaks="retained; fragment supervision masked, never negatives",
            alternative_masks="all retained; only identical atom masks deduplicated",
            mass_variants="all explanation records retained",atom_order="exact magma_canonical_smiles",
            metadata="CE/instrument retained as provenance only; not required, not imputed",
            sampling="query-weighted hash selection; connectivity-disjoint 80/20",
            validation_use="identifier/connectivity exclusion audit only",
            upstream_exposure="not full-stack OOF; frozen Stage1 saw official train"),
        eligible=dict(eligible),excluded=dict(excluded),eligible_peak_counts=dict(all_counts),
        inputs={name:dict(path=str(path.relative_to(ROOT)),sha256=digest(path)) for name,path in inputs.items()},
        code_sha256=digest(Path(__file__)),
        implementation_sha256={path:digest(ROOT/path) for path in (
            "tools/reranking/prepare_official_magma_local_targets.py",
            "tools/reranking/train_official_magma_forward.py",
            "tools/reranking/train_glacier_run_k_retrieval.py")},artifacts=artifacts)
    (args.out/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
