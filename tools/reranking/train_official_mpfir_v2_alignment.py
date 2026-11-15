#!/usr/bin/env python3
"""Train the frozen MPFIR-v2 L0 direct peak--fragment semantic pilot.

Official train only. DreaMS tokens are frozen. Peak-specific MAGMa alternatives
are multi-positive targets; unmatched peaks are masked. The matched control
cyclically shifts target fragment-mask indices within each molecule. This script
does not evaluate retrieval or accept official-test inputs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import resource
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.reranking.train_official_magma_forward import rows,digest
from tools.reranking.train_run_k_flare_local_reranker import (
    AtomFeatureEncoder,BondFeatureEncoder,EdgeAwareGINLayer,
    PackedGraphStore,build_graph_cache,
)


def memory_guard(min_free_gib=32):
    base=Path("/sys/fs/cgroup")
    if (base/"memory.max").exists():
        maximum=(base/"memory.max").read_text().strip()
        if maximum!="max":
            free=int(maximum)-int((base/"memory.current").read_text())
            if free<min_free_gib*2**30:
                raise MemoryError(f"62-GiB allocation safety stop: free {free/2**30:.2f} GiB")


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def load_tokens(path):
    payload=torch.load(str(path),map_location="cpu",weights_only=False,mmap=True)
    required={"tokens","mask","identifiers","metadata"}
    if not required<=set(payload):raise ValueError(f"Malformed token cache: {path}")
    if tuple(payload["tokens"].shape[1:])!=(101,1024):raise ValueError("Unexpected token shape")
    if tuple(payload["mask"].shape)!=tuple(payload["tokens"].shape[:2]):raise ValueError("Token/mask shape mismatch")
    return payload


def cyclic_shift_positive_mask(mask):
    """Shift each discriminative row by the smallest amount that changes it."""
    if mask.ndim!=2 or mask.dtype!=torch.bool:raise ValueError("Expected boolean [peaks,fragments]")
    output=mask.clone(); width=mask.shape[1]
    for row in range(mask.shape[0]):
        for shift in range(1,width):
            candidate=torch.roll(mask[row],shifts=shift,dims=0)
            if not torch.equal(candidate,mask[row]):
                output[row]=candidate;break
    return output


def multi_positive_metrics(logits,positive_mask,intensities,temperature=.07):
    if logits.shape!=positive_mask.shape or logits.ndim!=2:raise ValueError("Logit/target shape mismatch")
    eligible=positive_mask.any(1)&(~positive_mask).any(1)
    if not bool(eligible.any()):return None
    logits=logits[eligible]/temperature; positives=positive_mask[eligible]
    weights=intensities[eligible].clamp_min(0).sqrt()
    losses=torch.logsumexp(logits,1)-torch.logsumexp(logits.masked_fill(~positives,float("-inf")),1)
    loss=(losses*weights).sum()/weights.sum().clamp_min(1e-8)
    order=torch.argsort(logits.detach(),dim=1,descending=True)
    hit=positives.gather(1,order)
    ranks=hit.float().argmax(1)+1
    return loss,weights.detach(),ranks.detach(),eligible


def null_normalized_support(similarity,tau=.1):
    """Count-normalized positive support with an explicit fixed zero NULL."""
    if similarity.numel()==0:return similarity.new_zeros(())
    return tau*torch.log1p(torch.exp(similarity/tau).mean())


class DirectAlignment(nn.Module):
    def __init__(self,dim=256,layers=3,dropout=.1):
        super().__init__()
        self.peak_projection=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,512),nn.GELU(),nn.Dropout(dropout),nn.Linear(512,dim))
        self.atom_encoder=AtomFeatureEncoder(dim);self.bond_encoder=BondFeatureEncoder(dim)
        self.gnn=nn.ModuleList([EdgeAwareGINLayer(dim,dropout) for _ in range(layers)])
        self.fragment_projection=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim))

    def encode_nodes(self,graphs):
        nodes=self.atom_encoder(graphs["node_features"]);edges=self.bond_encoder(graphs["edge_features"])
        for layer in self.gnn:nodes=layer(nodes,graphs["edge_index"],edges)
        width=int(graphs["node_counts"].max());padded=nodes.new_zeros((len(graphs["node_counts"]),width,nodes.shape[-1]))
        padded[graphs["node_batch"],graphs["node_local"]]=nodes
        return padded

    def query_logits(self,peak_states,node_states,row,shuffle_targets=False):
        fragments=row["fragment_masks"]
        fragment=torch.stack([node_states[torch.tensor(mask,device=node_states.device)].mean(0) for mask in fragments])
        fragment=F.normalize(self.fragment_projection(fragment),dim=-1,eps=1e-6)
        annotated=[peak for peak in row["peaks"] if peak["fragment_supervision_mask"]]
        token_indices=torch.tensor([int(peak["token_index"]) for peak in annotated],device=peak_states.device)
        peaks=F.normalize(self.peak_projection(peak_states[token_indices]),dim=-1,eps=1e-6)
        target=torch.zeros((len(annotated),len(fragments)),dtype=torch.bool,device=peak_states.device)
        for i,peak in enumerate(annotated):target[i,torch.tensor(peak["positive_mask_indices"],device=peak_states.device)]=True
        true_target=target
        if shuffle_targets:target=cyclic_shift_positive_mask(target)
        intensity=torch.tensor([float(peak["relative_intensity"]) for peak in annotated],device=peak_states.device)
        return peaks@fragment.T,target,true_target,intensity


def move_graphs(graphs,device):return {k:v.to(device) for k,v in graphs.items()}


def validate_and_filter(label_path,token_path,full_metadata,full_nohead,threshold=.99999):
    labels=list(rows(label_path));tokens=load_tokens(token_path)
    if tokens["identifiers"]!=[r["identifier"] for r in labels]:raise ValueError("Label/token identifier order mismatch")
    for i,row in enumerate(labels):
        if int(tokens["mask"][i].sum())-1!=len(row["peaks"]):raise ValueError("Peak/token count mismatch")
        if [p["token_index"] for p in row["peaks"]]!=list(range(1,len(row["peaks"])+1)):raise ValueError("Peak token_index mismatch")
        if any(not mask or min(mask)<0 or max(mask)>=row["num_atoms"] for mask in row["fragment_masks"]):raise ValueError("Fragment atom index mismatch")
    metadata=list(rows(full_metadata));lookup={r["identifier"]:i for i,r in enumerate(metadata)}
    if len(lookup)!=len(metadata):raise ValueError("Duplicate official-train identifier")
    reference=np.load(full_nohead,mmap_mode="r");indices=np.asarray([lookup[r["identifier"]] for r in labels])
    current=tokens["tokens"][:,0].float().numpy();expected=np.asarray(reference[indices])
    cosine=np.sum(current*expected,1)/(np.linalg.norm(current,axis=1)*np.linalg.norm(expected,axis=1))
    keep=np.flatnonzero(cosine>=threshold).tolist();excluded=np.flatnonzero(cosine<threshold).tolist()
    if len(keep)<.995*len(labels):raise ValueError("Too many HDF/Stage1 spectrum provenance mismatches")
    filtered=[labels[i] for i in keep]
    return filtered,tokens,keep,dict(total=len(labels),kept=len(keep),threshold=threshold,
        excluded=[dict(identifier=labels[i]["identifier"],cosine=float(cosine[i])) for i in excluded],
        cosine_min=float(cosine.min()),cosine_mean=float(cosine.mean()),max_abs=float(np.max(np.abs(current-expected))))


def build_or_load_graphs(path,records):
    indices=torch.arange(len(records),dtype=torch.long);smiles=[r["smiles"] for r in records]
    expected=hashlib.sha256(indices.numpy().tobytes()).hexdigest()
    smiles_sha=hashlib.sha256("\n".join(smiles).encode()).hexdigest()
    if path.exists():
        payload=torch.load(str(path),map_location="cpu",weights_only=False,mmap=True)
        meta=payload.get("metadata",{})
        if meta.get("global_indices_sha256")!=expected or meta.get("smiles_sha256")!=smiles_sha:raise ValueError("Stale graph cache")
    else:
        payload=build_graph_cache(path,indices,smiles);payload["metadata"]["smiles_sha256"]=smiles_sha;torch.save(payload,path)
    return PackedGraphStore(payload,len(records)),dict(payload["metadata"])


def batches(count,batch_size,seed,shuffle,max_batches=0):
    order=list(range(count))
    if shuffle:random.Random(seed).shuffle(order)
    output=[order[i:i+batch_size] for i in range(0,count,batch_size)]
    return output[:max_batches] if max_batches else output


def run_epoch(model,records,tokens,token_rows,graph_store,record_offset,device,batch_size,temperature,
              train=False,optimizer=None,shuffle_targets=False,seed=0,max_batches=0):
    model.train(train);total_weight=0.;total_loss=0.;ranks=[];queries=0;peaks=0
    gradient_norms=[];optimizer_steps=0
    previous_grad_state=torch.is_grad_enabled();torch.set_grad_enabled(train)
    try:
        for group in batches(len(records),batch_size,seed,train,max_batches):
            memory_guard(); combined=torch.tensor([record_offset+i for i in group])
            graphs=move_graphs(graph_store.gather(combined),device);nodes=model.encode_nodes(graphs)
            losses=[];weights=[]
            for local,i in enumerate(group):
                graph_row=int(graphs["inverse"][local]);token_row=token_rows[i]
                peak_states=tokens["tokens"][token_row].float().to(device)
                logits,target,true_target,intensity=model.query_logits(peak_states,nodes[graph_row],records[i],shuffle_targets if train else False)
                result=multi_positive_metrics(logits,target,intensity,temperature)
                # Metrics always use unmodified labels, including for the control.
                truth=multi_positive_metrics(logits,true_target,intensity,temperature)
                if result is None or truth is None:continue
                losses.append(result[0]);weights.append(float(result[1].sum()))
                ranks.extend(truth[2].cpu().tolist());peaks+=len(truth[2]);queries+=1
            if not losses:continue
            denom=sum(weights);batch_loss=sum(loss*w for loss,w in zip(losses,weights,strict=True))/denom
            if train:
                assert optimizer is not None;optimizer.zero_grad(set_to_none=True);batch_loss.backward()
                squared=sum(float(parameter.grad.detach().float().square().sum().cpu())
                    for parameter in model.parameters() if parameter.grad is not None)
                gradient_norms.append(math.sqrt(squared))
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();optimizer_steps+=1
            total_loss+=float(batch_loss.detach().cpu())*denom;total_weight+=denom
        if not ranks or not total_weight:raise ValueError("No discriminative peak-fragment rows")
        ranks_np=np.asarray(ranks)
        result=dict(loss=total_loss/total_weight,top1=float(np.mean(ranks_np==1)),mrr=float(np.mean(1/ranks_np)),
            mean_rank=float(np.mean(ranks_np)),queries=queries,peaks=peaks)
        if train:result.update(optimizer_steps=optimizer_steps,max_unclipped_gradient_norm=max(gradient_norms),
            mean_unclipped_gradient_norm=float(np.mean(gradient_norms)))
        return result
    finally:
        torch.set_grad_enabled(previous_grad_state)


def train_arm(name,initial,model_kwargs,fit,calib,fit_tokens,calib_tokens,fit_rows,calib_rows,
              graph_store,device,args,out):
    seed_all(args.seed);model=DirectAlignment(**model_kwargs).to(device);model.load_state_dict(initial)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    history=[]
    metric=run_epoch(model,calib,calib_tokens,calib_rows,graph_store,len(fit),device,args.eval_batch_size,
        args.temperature,seed=args.seed);history.append(dict(epoch=0,calibration=metric,parameter_max_abs_delta=0.))
    best=(metric["loss"],0,copy.deepcopy(model.state_dict()))
    for epoch in range(1,args.epochs+1):
        train_metric=run_epoch(model,fit,fit_tokens,fit_rows,graph_store,0,device,args.batch_size,args.temperature,
            train=True,optimizer=optimizer,shuffle_targets=name=="shuffled",seed=args.seed+epoch,max_batches=args.max_batches)
        calibration=run_epoch(model,calib,calib_tokens,calib_rows,graph_store,len(fit),device,args.eval_batch_size,
            args.temperature,seed=args.seed)
        state=model.state_dict();parameter_delta=max(float((value.detach().cpu()-initial[key]).abs().max())
            for key,value in state.items())
        history.append(dict(epoch=epoch,train=train_metric,calibration=calibration,
            parameter_max_abs_delta=parameter_delta))
        if calibration["loss"]<best[0]:best=(calibration["loss"],epoch,copy.deepcopy(model.state_dict()))
        print(name,epoch,json.dumps(history[-1]),flush=True)
    arm=out/name;arm.mkdir(parents=True)
    torch.save(dict(state_dict=best[2],epoch=best[1],model_kwargs=model_kwargs,seed=args.seed),arm/"best.pt")
    summary=dict(arm=name,best_epoch=best[1],best_calibration_loss=best[0],history=history,
        checkpoint_sha256=digest(arm/"best.pt"))
    (arm/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    base=ROOT/"outputs/magma_official_local_targets_20261001_v2"
    p.add_argument("--fit-labels",type=Path,default=base/"fit.jsonl");p.add_argument("--calibration-labels",type=Path,default=base/"calibration.jsonl")
    p.add_argument("--fit-tokens",type=Path,default=base/"fit.datasafe.tokens.v2.pt");p.add_argument("--calibration-tokens",type=Path,default=base/"calibration.datasafe.tokens.v3.pt")
    p.add_argument("--full-train-metadata",type=Path,default=ROOT/"outputs/msalign_stage1_full_v15/train_full.metadata.jsonl")
    p.add_argument("--full-train-nohead",type=Path,default=ROOT/"outputs/run_k_official_datasafe_seed20260621/train.no_head.npy")
    p.add_argument("--out",type=Path,required=True);p.add_argument("--fit-limit",type=int,default=0);p.add_argument("--calibration-limit",type=int,default=0)
    p.add_argument("--epochs",type=int,default=3);p.add_argument("--max-batches",type=int,default=0)
    p.add_argument("--batch-size",type=int,default=8);p.add_argument("--eval-batch-size",type=int,default=16)
    p.add_argument("--dim",type=int,default=256);p.add_argument("--layers",type=int,default=3);p.add_argument("--dropout",type=float,default=.1)
    p.add_argument("--lr",type=float,default=2e-4);p.add_argument("--weight-decay",type=float,default=1e-4);p.add_argument("--temperature",type=float,default=.07)
    p.add_argument("--seed",type=int,default=20261002);p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    for path in (args.fit_labels,args.calibration_labels,args.fit_tokens,args.calibration_tokens,args.full_train_metadata,args.full_train_nohead):
        if "test" in path.name.lower():raise ValueError("Official test input is forbidden")
    if args.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    memory_guard();device=torch.device(args.device);started=time.time()
    fit,fit_tokens,fit_rows,fit_audit=validate_and_filter(args.fit_labels,args.fit_tokens,args.full_train_metadata,args.full_train_nohead)
    calib,calib_tokens,calib_rows,calib_audit=validate_and_filter(args.calibration_labels,args.calibration_tokens,args.full_train_metadata,args.full_train_nohead)
    if args.fit_limit:fit=fit[:args.fit_limit];fit_rows=fit_rows[:args.fit_limit]
    if args.calibration_limit:calib=calib[:args.calibration_limit];calib_rows=calib_rows[:args.calibration_limit]
    if {r["connectivity"] for r in fit}&{r["connectivity"] for r in calib}:raise ValueError("Fit/calibration connectivity overlap")
    args.out.mkdir(parents=True);combined=fit+calib
    graph_store,graph_meta=build_or_load_graphs(args.out/"graphs.pt",combined)
    for i,row in enumerate(combined):
        graph=graph_store.gather(torch.tensor([i]));
        if int(graph["node_counts"][0])!=row["num_atoms"]:raise ValueError("Graph/label atom order count mismatch")
    model_kwargs=dict(dim=args.dim,layers=args.layers,dropout=args.dropout);seed_all(args.seed)
    initial_model=DirectAlignment(**model_kwargs);initial=copy.deepcopy(initial_model.state_dict())
    initial_sha=hashlib.sha256(b"".join(v.detach().cpu().contiguous().numpy().tobytes() for v in initial.values())).hexdigest()
    true=train_arm("true",initial,model_kwargs,fit,calib,fit_tokens,calib_tokens,fit_rows,calib_rows,graph_store,device,args,args.out)
    control=train_arm("shuffled",initial,model_kwargs,fit,calib,fit_tokens,calib_tokens,fit_rows,calib_rows,graph_store,device,args,args.out)
    initial_metric=true["history"][0]["calibration"];selected=true["history"][true["best_epoch"]]["calibration"]
    gates=dict(loss_improvement=initial_metric["loss"]-selected["loss"]>=.02,
        mrr_improvement=selected["mrr"]-initial_metric["mrr"]>=.02,
        beats_shuffled_loss=control["best_calibration_loss"]-selected["loss"]>=.01)
    plumbing=bool(args.fit_limit or args.calibration_limit or args.max_batches or args.epochs!=3)
    report=dict(protocol="official/data-safe MPFIR-v2 L0 direct semantics",official_test_read=False,
        retrieval_evidence=False,plumbing_only=plumbing,verdict="PLUMBING_ONLY" if plumbing else ("L0_PASS" if all(gates.values()) else "STOP_L0_CONFIG"),
        gates=gates,initial_state_sha256=initial_sha,fit_audit=fit_audit,calibration_audit=calib_audit,
        selected_queries=dict(fit=len(fit),calibration=len(calib)),graph_cache=graph_meta,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        inputs={str(path.relative_to(ROOT)):digest(path) for path in (args.fit_labels,args.calibration_labels,args.fit_tokens,args.calibration_tokens,args.full_train_metadata,args.full_train_nohead)},
        arms=dict(true=true,shuffled=control),elapsed_seconds=time.time()-started,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        cuda_peak_gib=torch.cuda.max_memory_allocated()/2**30 if device.type=="cuda" else 0)
    (args.out/"summary.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({k:v for k,v in report.items() if k!="arms"}),flush=True)


if __name__=="__main__":main()
