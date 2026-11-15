#!/usr/bin/env python3
"""Matched L0-only versus deployed-candidate-listwise+L0 MPFIR pilot.

Requires fresh connectivity-isolated SCCF OOF lists. All model selection stays
inside official train. The frozen publication ensemble/pair is not loaded.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import resource
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from glmr.subformula import ion_shift
from tools.reranking.train_official_magma_forward import rows, digest
from tools.reranking.train_official_mpfir_v2_alignment import (
    DirectAlignment, seed_all, load_tokens, build_or_load_graphs,
    multi_positive_metrics,
)
from tools.reranking.eval_official_mpfir_v2_top5 import (
    build_registry, load_or_build_stores, fragment_tokens, move, feasibility,
    masked_null_support, candidate_relative_score, zscore, tie_invariance,
)
from tools.reranking.audit_official_datasafe_pair_controls import candidate_order
from tools.analysis.analyze_stage1_logit_ensemble import comparison, paired_bootstrap
from tools.reranking.train_official_datasafe_score_fusion import endpoint
from tools.reranking.prepare_official_mpfir_cross_candidate import connectivity
from tools.reranking.train_official_mpfir_oof_retriever import memory_guard


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def standardize(values):
    return (values-values.mean(1, keepdim=True)) / values.std(1, unbiased=False, keepdim=True).clamp_min(0.02)


def combine_scores(base, coverage, local=None):
    score = base + 0.25*standardize(coverage)
    if local is not None:
        score = score + 0.25*standardize(local)
    return score


def ranking_loss(scores, targets):
    keep = targets >= 0
    return F.cross_entropy(scores[keep], targets[keep]) if bool(keep.any()) else scores.sum()*0


def rank_scores(inputs, targets, scores):
    ranks = np.asarray([target["baseline_rank"] for target in targets], dtype=np.int32)
    for i, (query, target) in enumerate(zip(inputs, targets, strict=True)):
        position = target["gt_position"]
        if position >= 0:
            order = candidate_order(query["identifier"], np.asarray(query["candidate_global_rows"]),
                                    scores[i], 20260926)
            ranks[i] = int(np.flatnonzero(order == position)[0])+1
    return ranks


def prepare(args):
    started = time.time()
    folder = args.data / "features"
    if folder.exists():
        if not args.resume_prepare or (folder / "summary.json").exists():
            raise FileExistsError(folder)
    else:
        folder.mkdir()
    records = list(rows(args.data / "labels.jsonl"))
    tokens = load_tokens(args.data / "tokens.pt")
    if tokens["identifiers"] != [row["identifier"] for row in records]:
        raise ValueError("Token/label identifier drift")
    expected_digest = "d9b4e8b920ff367370c5a3fa154eaa951f90bafa58d5224a594424d9c54b1832"
    if tokens["metadata"]["dreams_ckpt_sha256"] != expected_digest:
        raise ValueError("Non-data-safe token cache")
    official = list(rows(ROOT / "outputs/msalign_stage1_full_v15/train_full.metadata.jsonl"))
    lookup = {row["identifier"]: i for i, row in enumerate(official)}
    nohead = np.load(ROOT / "outputs/run_k_official_datasafe_seed20260621/train.no_head.npy", mmap_mode="r")
    ref = np.asarray(nohead[[lookup[row["identifier"]] for row in records]])
    current = tokens["tokens"][:, 0].float().numpy()
    cosine = np.sum(ref*current, 1)/(np.linalg.norm(ref, axis=1)*np.linalg.norm(current, axis=1))
    kept = np.flatnonzero(cosine>=0.99999)
    rejected = np.flatnonzero(cosine<0.99999)
    if len(rejected) > 0.005*len(records):
        raise ValueError("Too many label/spectrum-source mismatches")
    original = records
    records = [original[i] for i in kept]
    all_lists = []
    for name in ("mining0", "mining1", "endpoint"):
        summary = json.loads((args.data / name / "summary.json").read_text())
        if summary["plumbing_only"] or summary["oof_gt_overlap"] or summary["negative_overlap"]:
            raise ValueError("Formal OOF prerequisite failed")
        all_lists.extend(rows(args.data / name / "top5.jsonl"))
    by_id = {row["identifier"]: row for row in all_lists}
    if len(by_id) != len(original) or len(all_lists) != len(by_id):
        raise ValueError("Incomplete/duplicate OOF mining")
    targets = [by_id[row["identifier"]] for row in records]
    wanted = {int(i) for row in targets for i in row["candidate_global_rows"]}
    smiles_map = {}
    with (ROOT / "outputs/run_k_full_train/full_sparse/full_smiles.txt").open() as handle:
        for i, line in enumerate(handle):
            if i in wanted:
                smiles_map[i] = line.rstrip("\n")
    if set(smiles_map) != wanted:
        raise ValueError("Candidate SMILES row missing")
    inputs = []
    for row, target, token_row in zip(records, targets, kept, strict=True):
        if len(row["peaks"]) != int(tokens["mask"][int(token_row), 1:].sum()):
            raise ValueError("Source peak/token count drift")
        if [p["token_index"] for p in row["peaks"]] != list(range(1,len(row["peaks"])+1)):
            raise ValueError("Source peak/token order drift")
        inputs.append(dict(identifier=row["identifier"], adduct=row["adduct"],
            precursor_mz=row["precursor_mz"],
            observed_spectrum=[[p["mz"], p["relative_intensity"]] for p in row["peaks"]],
            candidate_global_rows=target["candidate_global_rows"],
            candidate_smiles=[smiles_map[i] for i in target["candidate_global_rows"]]))
    support, compact, smiles, masses, reasons = build_registry(inputs)
    # Isotope-labelled candidates violate the historical parent-mask mass
    # convention. Apply the same whole-query fallback to every arm.
    isotope = {i for i, smi in smiles_map.items()
               if any(a.GetIsotope() for a in Chem.MolFromSmiles(smi).GetAtoms())}
    for i, query in enumerate(inputs):
        if support[i] and isotope.intersection(query["candidate_global_rows"]):
            support[i] = False; reasons["supported"] -= 1
            reasons["candidate_isotope_mass_unsupported"] = reasons.get("candidate_isotope_mass_unsupported", 0)+1
    candidate_folder = folder / "candidates"
    candidate_folder.mkdir(exist_ok=args.resume_prepare)
    graph, fragment, registry_meta = load_or_build_stores(candidate_folder, smiles)
    native_graph, native_meta = build_or_load_graphs(folder / "native_graphs.pt", records)
    valid = torch.zeros((len(records),5,100,64), dtype=torch.bool)
    peak_mask = tokens["mask"][torch.from_numpy(kept), 1:]
    mz = torch.zeros((len(records),100)); intensity = torch.zeros_like(mz)
    for i, query in enumerate(inputs):
        value = torch.tensor(query["observed_spectrum"])
        mz[i,:len(value)] = value[:,0]; intensity[i,:len(value)] = value[:,1]
    supported = np.flatnonzero(support)
    for start in range(0, len(supported), 32):
        selected = supported[start:start+32]
        ids = torch.from_numpy(compact[selected]).long()
        graphs = graph.gather(ids.flatten())
        fragments = fragment.gather(ids.flatten(), graphs["node_counts"])
        inverse = fragments["inverse"]
        mass = (fragments["scalar"][inverse,:,0]*1000).reshape(len(selected),5,64)
        fmask = fragments["fragment_mask"][inverse].reshape(len(selected),5,64)
        feasible, _ = feasibility(mz[selected], torch.tensor([inputs[i]["precursor_mz"] for i in selected]),
            torch.tensor([ion_shift(inputs[i]["adduct"]) for i in selected]), mass,
            masses[ids], fmask, peak_mask[selected])
        valid[selected] = feasible
    coverage = candidate_relative_score(masked_null_support(torch.ones(valid.shape),valid),peak_mask,intensity)
    same_formula = np.zeros((len(records),5), dtype=bool)
    candidate_formula = {i:rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smi)) for i,smi in smiles_map.items()}
    candidate_groups = {i:connectivity(smi) for i,smi in smiles_map.items()}
    for i, (record, query) in enumerate(zip(records, inputs, strict=True)):
        gt_formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(record["smiles"]))
        same_formula[i] = [candidate_formula[index] == gt_formula for index in query["candidate_global_rows"]]
    payload = dict(records=records, inputs=inputs, targets=targets,
        token_rows=torch.from_numpy(kept), support=torch.from_numpy(support),
        compact=torch.from_numpy(compact), valid=valid, intensity=intensity,
        coverage=coverage, feasible_peaks=valid.any(-1).sum(-1),
        same_formula=torch.from_numpy(same_formula),
        candidate_connectivities=[[candidate_groups[i] for i in q["candidate_global_rows"]] for q in inputs],
        base=torch.tensor([row["base_scores"] for row in targets]),
        labels=torch.tensor([row["gt_position"] for row in targets]),
        registry_smiles=smiles)
    torch.save(payload, folder / "data.pt")
    report = dict(queries=len(records), partitions=dict(Counter(r["pilot_partition"] for r in records)),
        support_counts=reasons, supported_queries=int(support.sum()),
        reachable_supported_fit=int(sum(r["pilot_partition"]=="fit" and support[i] and targets[i]["gt_position"]>=0
                                        for i,r in enumerate(records))),
        parity_exclusions=[dict(identifier=original[i]["identifier"], cosine=float(cosine[i])) for i in rejected],
        registry=registry_meta, native_graph=native_meta, rdkit=rdBase.rdkitVersion,
        formal_oof=True, official_test_read=False, official_validation_read=False,
        elapsed_seconds=time.time()-started, data_sha256=digest(folder/"data.pt"))
    write_json(folder / "summary.json", report)
    print(json.dumps(report), flush=True)


class PilotData:
    def __init__(self, path):
        self.root = path
        self.payload = torch.load(str(path/"features/data.pt"),map_location="cpu",weights_only=False,mmap=True)
        p = self.payload
        self.tokens = load_tokens(path/"tokens.pt")
        self.graph, self.fragment, _ = load_or_build_stores(path/"features/candidates",p["registry_smiles"])
        self.native_graph,_ = build_or_load_graphs(path/"features/native_graphs.pt",p["records"])
        self.parts = np.asarray([r["pilot_partition"] for r in p["records"]])
        self.groups = np.asarray([r["connectivity"] for r in p["records"]])
        for a,b in (("fit","calibration"),("fit","evaluation"),("calibration","evaluation")):
            if set(self.groups[self.parts==a]) & set(self.groups[self.parts==b]):
                raise ValueError("Local connectivity leakage")

    def local(self, model, indices, device, chemistry=None):
        p = self.payload
        selected = torch.as_tensor(indices,dtype=torch.long)
        supported = p["support"][selected]
        result = torch.zeros((len(selected),5),device=device)
        if not bool(supported.any()):
            return result
        active = selected[supported]
        ids = p["compact"][active].long()
        graphs_cpu = self.graph.gather(ids.flatten())
        fragments_cpu = self.fragment.gather(ids.flatten(),graphs_cpu["node_counts"])
        fragments = fragment_tokens(model,move(graphs_cpu,device),move(fragments_cpu,device),tuple(ids.shape))
        source = active if chemistry is None else chemistry["donor_rows"][active]
        token_rows = p["token_rows"][source]
        peaks = F.normalize(model.peak_projection(self.tokens["tokens"][token_rows,1:].float().to(device)),dim=-1,eps=1e-6)
        similarities = torch.einsum("bpd,bkfd->bkpf",peaks,fragments)
        valid = (p["valid"] if chemistry is None else chemistry["valid"])[active].to(device)
        score = candidate_relative_score(masked_null_support(similarities,valid),
            self.tokens["mask"][token_rows,1:].to(device),p["intensity"][source].to(device))
        result[supported.to(device)] = score
        return result

    def auxiliary(self, model, indices, device):
        p=self.payload
        selected=torch.as_tensor(indices,dtype=torch.long)
        graph=self.native_graph.gather(selected)
        nodes=model.encode_nodes(move(graph,device))
        inverse=graph["inverse"]
        losses=[]
        for j,index in enumerate(selected.tolist()):
            tokens=self.tokens["tokens"][p["token_rows"][index]].float().to(device)
            logits,target,_,intensity=model.query_logits(tokens,nodes[int(inverse[j])],p["records"][index])
            loss=multi_positive_metrics(logits,target,intensity)
            if loss is not None:
                losses.append(loss[0])
        if not losses:
            return nodes.sum()*0
        return torch.stack(losses).mean()


@torch.no_grad()
def collect(model,data,indices,device,batch=8,chemistry=None):
    model.eval()
    output=[]
    for start in range(0,len(indices),batch):
        output.append(data.local(model,indices[start:start+batch],device,chemistry).cpu())
    return torch.cat(output)


def calibration_metrics(data,indices,local):
    p=data.payload
    base=p["base"][indices];coverage=p["coverage"][indices]
    baseline=combine_scores(base,coverage).numpy()
    current=combine_scores(base,coverage,local).numpy()
    inputs=[p["inputs"][i] for i in indices];targets=[p["targets"][i] for i in indices]
    before=rank_scores(inputs,targets,baseline);after=rank_scores(inputs,targets,current)
    return dict(r1=float(np.mean(after==1)*100),mrr=float(np.mean(1/after)*100),
        delta_r1=float((np.mean(after==1)-np.mean(before==1))*100),
        delta_mrr=float((np.mean(1/after)-np.mean(1/before))*100),
        damage=float(np.sum((before==1)&(after>1))/max(np.sum(before==1),1)))


def train_arm(args,name,initial,data,device):
    started=time.time();p=data.payload
    seed_all(20261010)
    model=DirectAlignment();model.load_state_dict(initial);model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    fit=np.flatnonzero(data.parts=="fit");calibration=np.flatnonzero(data.parts=="calibration")
    folder=args.data/(name+("_smoke" if args.smoke_batches else ""))
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir()
    if args.smoke_batches:
        fit=fit[:64];calibration=calibration[:32]
    history=[];selected_epoch=0;best_mrr=-float("inf");best_state=copy.deepcopy(initial)
    for epoch in range(1,4):
        order=np.random.default_rng(20261010+epoch).permutation(fit)
        model.train();losses=[];rank_losses=[];aux_losses=[];rank_peak_grad=[];rank_graph_grad=[]
        for step,start in enumerate(range(0,len(order),8)):
            if args.smoke_batches and step>=args.smoke_batches:break
            if step%32==0:memory_guard()
            indices=order[start:start+8]
            local=data.local(model,indices,device)
            labels=p["labels"][indices].clone();labels[~p["support"][indices]]=-1
            logits=combine_scores(p["base"][indices].to(device),p["coverage"][indices].to(device),local)
            candidate_loss=ranking_loss(logits,labels.to(device))
            aux=data.auxiliary(model,indices,device)
            loss=aux if name=="l0_only" else candidate_loss+0.1*aux
            if name=="listwise_l0" and step<4 and candidate_loss.requires_grad:
                params=[model.peak_projection[-1].weight,model.fragment_projection[-1].weight]
                gradients=torch.autograd.grad(candidate_loss,params,retain_graph=True,allow_unused=True)
                rank_peak_grad.append(0.0 if gradients[0] is None else float(gradients[0].norm()))
                rank_graph_grad.append(0.0 if gradients[1] is None else float(gradients[1].norm()))
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            losses.append(float(loss.detach()));rank_losses.append(float(candidate_loss.detach()));aux_losses.append(float(aux.detach()))
            if step%32==0:
                print(json.dumps(dict(arm=name,epoch=epoch,step=step,loss=float(np.mean(losses)),
                    elapsed_seconds=round(time.time()-started,1),rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
        local=collect(model,data,calibration,device)
        metric=calibration_metrics(data,calibration,local)
        eligible=metric["delta_r1"]>=0 and metric["delta_mrr"]>0 and metric["damage"]<=0.08
        if eligible and metric["mrr"]>best_mrr:
            selected_epoch=epoch;best_mrr=metric["mrr"];best_state=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),ranking_loss=float(np.mean(rank_losses)),
            auxiliary_loss=float(np.mean(aux_losses)),calibration=metric,eligible=eligible,
            ranking_peak_gradient_norms=rank_peak_grad,ranking_fragment_gradient_norms=rank_graph_grad))
        print(json.dumps(dict(arm=name,epoch=epoch,calibration=metric,selected_epoch=selected_epoch)),flush=True)
    if name=="listwise_l0" and not (max(history[0]["ranking_peak_gradient_norms"],default=0)>0 and
                                    max(history[0]["ranking_fragment_gradient_norms"],default=0)>0):
        raise ValueError("Candidate loss failed to reach both local encoders")
    torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},epoch=3,
                    model_kwargs=dict(dim=256,layers=3,dropout=.1)),folder/"epoch3.pt")
    torch.save(dict(state_dict=best_state,selected_epoch=selected_epoch,model_kwargs=dict(dim=256,layers=3,dropout=.1)),folder/"best.pt")
    report=dict(arm=name,selected_epoch=selected_epoch,history=history,official_test_read=False,
        official_validation_read=False,plumbing_only=bool(args.smoke_batches),seed=20261010,
        seconds=time.time()-started,checkpoint_sha256=digest(folder/"best.pt"),code_sha256=digest(Path(__file__)))
    write_json(folder/"summary.json",report)
    return report


def win_interval(before,after,seed=20261010):
    if not len(before):return dict(queries=0)
    delta=np.asarray(after)-np.asarray(before)
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,10000,128):
        draw=rng.integers(0,len(delta),size=(min(128,10000-start),len(delta)))
        samples.extend(delta[draw].mean(1))
    return dict(queries=len(delta),before_win_pct=100*float(np.mean(before)),
        after_win_pct=100*float(np.mean(after)),delta_pp=100*float(delta.mean()),
        ci95_pp=(np.quantile(samples,[.025,.975])*100).tolist())


def equal_coverage_diagnostic(data,indices,locals_by_arm):
    p=data.payload
    pairs=[]
    for row in indices:
        gt=int(p["labels"][row])
        if gt<0 or not bool(p["support"][row]):continue
        # Comparison is strictly across different 2-D structures; stereo
        # aliases are reported by the inventory but not called hard isomers.
        decoys=[j for j in range(5) if j!=gt and
                p["candidate_connectivities"][row][j]!=data.groups[row]]
        if not decoys:continue
        wrong=max(decoys,key=lambda j:float(p["coverage"][row,j]))
        gt_peaks=int(p["feasible_peaks"][row,gt]);wrong_peaks=int(p["feasible_peaks"][row,wrong])
        pairs.append(dict(row=row,gt=gt,wrong=wrong,
            equal_count=gt_peaks==wrong_peaks and gt_peaks>0,
            equal_score=abs(float(p["coverage"][row,gt]-p["coverage"][row,wrong]))<=1e-6 and gt_peaks>0 and wrong_peaks>0,
            same_formula=bool(p["same_formula"][row,wrong])))
    mapping={row:i for i,row in enumerate(indices)}
    def wins(values,selected):
        result=[]
        for pair in selected:
            current=values[mapping[pair["row"]]]
            margin=float(current[pair["gt"]]-current[pair["wrong"]])
            result.append(1.0 if margin>1e-6 else 0.0 if margin< -1e-6 else .5)
        return np.asarray(result)
    report={"reachable_supported_different_connectivity_pairs":len(pairs),
            "decoy_selection":"strongest wrong coverage scorer, independent of learned outputs; metric only"}
    cov=p["coverage"][indices].numpy()
    for name in ("equal_count","equal_score"):
        for same_formula in (False,True):
            selected=[pair for pair in pairs if pair[name] and (not same_formula or pair["same_formula"])]
            label=name+("_same_formula" if same_formula else "")
            before=wins(cov,selected)
            arm_wins={arm:wins(values,selected) for arm,values in locals_by_arm.items()}
            report[label]=dict(queries=len(selected),against_coverage={arm:win_interval(before,value)
                                        for arm,value in arm_wins.items()},
                joint_vs_l0=win_interval(arm_wins["l0_only"],arm_wins["listwise_l0"]))
    return report


def wrong_spectrum_channel(data,indices,seed=20261010):
    from collections import defaultdict
    p=data.payload;blocks=defaultdict(list)
    for index in indices:
        query=p["inputs"][index]
        blocks[(query["adduct"],float(query["precursor_mz"]))].append(index)
    donor=torch.arange(len(data.parts));eligible=[];rng=np.random.default_rng(seed)
    for members in blocks.values():
        if len(members)<2:continue
        order=np.asarray(members)[rng.permutation(len(members))]
        donor[torch.as_tensor(order)]=torch.as_tensor(np.roll(order,1));eligible.extend(order.tolist())
    valid=p["valid"].clone()
    supported=[i for i in eligible if bool(p["support"][i])]
    for start in range(0,len(supported),32):
        active=supported[start:start+32];source=donor[active]
        ids=p["compact"][active].long();graph=data.graph.gather(ids.flatten())
        fragment=data.fragment.gather(ids.flatten(),graph["node_counts"]);inverse=fragment["inverse"]
        mass=(fragment["scalar"][inverse,:,0]*1000).reshape(len(active),5,64)
        fmask=fragment["fragment_mask"][inverse].reshape(len(active),5,64)
        mz=torch.zeros((len(active),100))
        for j,index in enumerate(source.tolist()):
            observed=p["inputs"][index]["observed_spectrum"]
            mz[j,:len(observed)]=torch.tensor(observed)[:,0]
        from rdkit.Chem import Descriptors
        parent=torch.tensor([[Descriptors.ExactMolWt(Chem.MolFromSmiles(smi))
                             for smi in p["inputs"][i]["candidate_smiles"]] for i in active])
        valid[active],_=feasibility(mz,torch.tensor([p["inputs"][i]["precursor_mz"] for i in active]),
            torch.tensor([ion_shift(p["inputs"][i]["adduct"]) for i in active]),mass,parent,fmask,
            data.tokens["mask"][p["token_rows"][source],1:])
    for row in eligible:
        other=int(donor[row])
        if row==other or data.groups[row]==data.groups[other]:
            raise ValueError("Wrong-spectrum donor leakage")
    return dict(donor_rows=donor,valid=valid),np.asarray(sorted(eligible),dtype=np.int64)


def evaluate(args,data,device):
    started=time.time();p=data.payload
    directory=args.data/"evaluation"
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir()
    indices=np.flatnonzero(data.parts=="evaluation")
    groups=data.groups[indices];inputs=[p["inputs"][i] for i in indices]
    targets=[p["targets"][i] for i in indices]
    base=p["base"][indices];coverage=p["coverage"][indices]
    scores={"base":base.numpy(),"coverage":combine_scores(base,coverage).numpy(),
            "double_coverage":(base+.5*standardize(coverage)).numpy()}
    models={};selected={};locals_by_arm={};raw_epoch3={};permutation_errors={}
    for arm in ("l0_only","listwise_l0"):
        summary=json.loads((args.data/arm/"summary.json").read_text())
        if summary["plumbing_only"]:raise ValueError("Plumbing checkpoints cannot enter formal evaluation")
        checkpoint=torch.load(args.data/arm/"best.pt",map_location="cpu",weights_only=False)
        selected[arm]=int(checkpoint["selected_epoch"])
        model=DirectAlignment();model.load_state_dict(checkpoint["state_dict"]);model.to(device).eval()
        models[arm]=model
        local=collect(model,data,indices,device) if selected[arm]>0 else torch.zeros((len(indices),5))
        locals_by_arm[arm]=local.numpy()
        scores[arm]=combine_scores(base,coverage,local if selected[arm]>0 else None).numpy()
        check=indices[:32];reference=collect(model,data,check,device)
        permutation=torch.tensor([2,4,1,0,3]);old_compact=p["compact"];old_valid=p["valid"]
        try:
            p["compact"]=old_compact[:,permutation];p["valid"]=old_valid[:,permutation]
            permuted=collect(model,data,check,device)
        finally:
            p["compact"]=old_compact;p["valid"]=old_valid
        error=float((reference-permuted[:,torch.argsort(permutation)]).abs().max())
        if error>1e-5:raise ValueError("Candidate permutation equivariance failed")
        permutation_errors[arm]=error
        final=torch.load(args.data/arm/"epoch3.pt",map_location="cpu",weights_only=False)
        raw_model=DirectAlignment();raw_model.load_state_dict(final["state_dict"]);raw_model.to(device).eval()
        raw_epoch3[arm]=collect(raw_model,data,indices,device).numpy()
        del raw_model
    ranks={name:rank_scores(inputs,targets,score) for name,score in scores.items()}
    tie_errors={name:tie_invariance(inputs,score) for name,score in scores.items()}
    if any(tie_errors.values()):raise ValueError("Identity-keyed ties changed under permutation")
    comparisons={name:comparison(ranks["coverage"],ranks[name],groups,10000,20261010)
                 for name in ("double_coverage","l0_only","listwise_l0")}
    comparisons["joint_vs_l0"]=comparison(ranks["l0_only"],ranks["listwise_l0"],groups,10000,20261010)
    diagnostic=equal_coverage_diagnostic(data,indices,locals_by_arm)
    raw_diagnostic=equal_coverage_diagnostic(data,indices,raw_epoch3)
    chemistry,eligible=wrong_spectrum_channel(data,indices)
    controls={"eligible_queries":len(eligible),"donor_seed":20261010,
              "scope":"isolated local channel; real upstream/coverage and fixed recipient candidates; exact precursor/adduct/different connectivity"}
    if len(eligible):
        local=collect(models["listwise_l0"],data,eligible,device,chemistry=chemistry)
        if selected["listwise_l0"]==0:local.zero_()
        wrong_scores=combine_scores(p["base"][eligible],p["coverage"][eligible],local).numpy()
        wrong_ranks=rank_scores([p["inputs"][i] for i in eligible],[p["targets"][i] for i in eligible],wrong_scores)
        real_lookup={row:i for i,row in enumerate(indices)}
        real_ranks=ranks["listwise_l0"][[real_lookup[i] for i in eligible]]
        controls["real_minus_wrong"]=comparison(wrong_ranks,real_ranks,data.groups[eligible],10000,20261010)
        np.savez_compressed(directory/"wrong_spectrum.npz",recipient_rows=eligible,
            donor_rows=chemistry["donor_rows"][eligible].numpy(),real_ranks=real_ranks,wrong_ranks=wrong_ranks)
    joint=comparisons["listwise_l0"];vs_l0=comparisons["joint_vs_l0"]
    diag=diagnostic["equal_score"]["joint_vs_l0"]
    gates=dict(selected_nonzero=selected["listwise_l0"]>0,
        r1_nonnegative_vs_coverage=joint["delta_query_weighted_pp"]["recall@1"]>=0,
        mrr_positive_vs_coverage=joint["delta_query_weighted_pp"]["mrr"]>0,
        r1_nonnegative_vs_l0=vs_l0["delta_query_weighted_pp"]["recall@1"]>=0,
        mrr_positive_vs_l0=vs_l0["delta_query_weighted_pp"]["mrr"]>0,
        mrr_not_below_double_coverage=comparisons["double_coverage"]["delta_query_weighted_pp"]["mrr"]<=joint["delta_query_weighted_pp"]["mrr"],
        damage_bounded=joint["rank1_transition"]["baseline_rank1_damage_rate_pct"]<=8,
        equal_coverage_positive=diag.get("delta_pp",0)>0,
        shortlist_invariant=all(np.array_equal(ranks["base"]<=k,ranks[name]<=k)
                                for name in ranks for k in (5,20)))
    if not all(gates.values()):verdict="STOP_THIS_CONFIGURATION"
    else:
        cluster=joint["paired_bootstrap"]["connectivity_cluster_query_weighted"]
        verdict="POSITIVE_BOUNDED_PILOT" if cluster["mrr"]["ci95_low_pp"]>0 and diag["ci95_pp"][0]>0 else "INCONCLUSIVE_DO_NOT_SCALE"
    same=p["same_formula"][indices].clone();labels=p["labels"][indices]
    same[torch.arange(len(indices))[labels>=0],labels[labels>=0]]=False
    np.savez_compressed(directory/"per_query.npz",identifiers=np.asarray([q["identifier"] for q in inputs]),
        connectivities=groups,query_rows=indices,labels=labels.numpy(),coverage=coverage.numpy(),
        support=p["support"][indices].numpy(),feasible_peaks=p["feasible_peaks"][indices].numpy(),
        **{f"ranks_{name}":rank for name,rank in ranks.items()},
        **{f"local_{name}":value for name,value in locals_by_arm.items()},
        **{f"epoch3_local_{name}":value for name,value in raw_epoch3.items()})
    report=dict(status=verdict,gates=gates,selected_epochs=selected,queries=len(indices),
        connectivities=len(set(groups)),supported_queries=int(p["support"][indices].sum()),
        reachable_queries=int((labels>=0).sum()),same_formula_competitor_queries=int(same.any(1).sum()),
        metrics={name:endpoint(ranks["coverage"],rank,groups) for name,rank in ranks.items()},
        comparisons=comparisons,equal_coverage=diagnostic,epoch3_mechanism_diagnostic=raw_diagnostic,
        spectrum_controls=controls,
        permutation_errors=permutation_errors,identity_tie_errors=tie_errors,
        official_test_read=False,official_validation_read=False,
        interpretation="independent official-train label-cohort, single-seed OOF SCCF; not publication ensemble+pair gain",
        seconds=time.time()-started,code_sha256=digest(Path(__file__)),
        per_query_sha256=digest(directory/"per_query.npz"))
    write_json(directory/"summary.json",report)
    print(json.dumps(dict(status=verdict,gates=gates,metrics=report["metrics"],equal_coverage=diagnostic)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("prepare","train","evaluate"),required=True)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--smoke-batches",type=int,default=0)
    parser.add_argument("--resume-prepare",action="store_true",help="Reuse verified caches after incomplete preparation only")
    args=parser.parse_args()
    if args.resume_prepare and args.stage!="prepare":raise ValueError("Resume is preparation-only")
    args.data=args.data.resolve()
    args.data.relative_to(ROOT/"outputs")
    if args.stage=="prepare":prepare(args);return
    data=PilotData(args.data);device=torch.device("cuda")
    if args.stage=="train":
        seed_all(20261010);model=DirectAlignment();initial=copy.deepcopy(model.state_dict())
        for name in ("l0_only","listwise_l0"):
            train_arm(args,name,initial,data,device)
    else:
        evaluate(args,data,device)


if __name__=="__main__":main()
