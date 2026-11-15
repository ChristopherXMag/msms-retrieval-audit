#!/usr/bin/env python3
"""Train/control EmbCos under frozen official train/val E2; no test input accepted."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from glmr.embcos import EmbCos,decode_packed,sample_candidates,cosine_schedule,uniform_tie_metrics,summarize_metrics,sha256_file,write_json


def save_checkpoint(path,payload):
    temp=path.with_suffix('.staging.pt');torch.save(payload,temp);temp.replace(path)


class Assets:
    def __init__(self,cache,contract,device,train_limit=0,val_limit=0):
        self.cache=cache;self.device=device
        self.manifest=json.loads((cache/'manifest.json').read_text())
        assert self.manifest['status']=='COMPLETE','Plumbing cache cannot support formal evidence'
        assert self.manifest['contract_sha256']==sha256_file(contract)
        assert self.manifest['scripts']['module']==sha256_file(ROOT/'glmr/embcos.py')
        self.rows={f:dict(np.load(cache/(f+'.rows.npz'))) for f in ('train','val')}
        for fold,limit in [('train',train_limit),('val',val_limit)]:
            if limit:self.rows[fold]={k:v[:limit] for k,v in self.rows[fold].items()}
        self.packed_cpu=np.load(cache/'morgan4096.packed.npy',mmap_mode='r')
        self.packed=torch.from_numpy(np.asarray(self.packed_cpu)).to(device)
        self.spectra={}
        for fold in ('train','val'):
            data=np.load(cache/(fold+'.bins.npy'),mmap_mode='r')
            size=len(self.rows[fold]['identifiers'])
            self.spectra[fold]=torch.from_numpy(np.asarray(data[:size])).to(device)
        self.exclusion=np.zeros(self.manifest['molecule_rows'],bool)
        self.exclusion[np.load(cache/'negative_exclusion.npy')]=True
        # Feature-equivalent candidates share one projected vector, without deleting identities.
        candidates=self.rows['val']['permuted_candidates'];valid=candidates>=0
        global_rows=np.unique(candidates[valid]);bits=self.packed_cpu[global_rows]
        _,representative,inverse=np.unique(bits.view('V512').reshape(-1),return_index=True,return_inverse=True)
        self.eval_representatives=torch.from_numpy(global_rows[representative].astype(np.int64)).to(device)
        lookup=np.full(self.manifest['molecule_rows'],-1,np.int32);lookup[global_rows]=inverse
        mapped=np.zeros(candidates.shape,np.int64);mapped[valid]=lookup[candidates[valid]]
        self.eval_indices=torch.from_numpy(mapped).to(device)
        self.eval_mask=torch.from_numpy(valid).to(device)
        self.gt_slots=torch.from_numpy(self.rows['val']['gt_slots'].astype(np.int64)).to(device)
        self.evaluation_cache={'unique_candidate_rows':len(global_rows),'unique_fingerprint_vectors':len(representative),
            'input_equivalence':'exact packed bits; scoring shares vectors but retains every identity, candidate count, GT slot and tie'}
        mean_sum=torch.zeros(10050,dtype=torch.float64,device=device)
        for start in range(0,len(self.spectra['train']),4096):
            mean_sum+=self.spectra['train'][start:start+4096].double().sum(0)
        self.train_mean=(mean_sum/len(self.spectra['train'])).float()
        self.validate_inputs(cache)
    def validate_inputs(self,cache):
        for fold in ('train','val'):
            assert sha256_file(cache/(fold+'.rows.npz'))==self.manifest['files'][fold]['rows_sha256']
            assert self.rows[fold]['candidates'].max()<len(self.packed_cpu)
        assert set(self.rows['train']['connectivities']).isdisjoint(self.rows['val']['connectivities'])
        assert sha256_file(cache/'morgan4096.packed.npy')==self.manifest['packed_sha256']
        assert sha256_file(cache/'negative_exclusion.npy')==self.manifest['negative_exclusion_sha256']
        for fold in ('train','val'):
            assert sha256_file(cache/(fold+'.bins.npy'))==self.manifest['spectra'][fold]['sha256']
        self.input_digests={'cache_manifest':sha256_file(cache/'manifest.json'),
            'packed':self.manifest['packed_sha256'],'train_bins':self.manifest['spectra']['train']['sha256'],
            'val_bins':self.manifest['spectra']['val']['sha256']}
    @torch.inference_mode()
    def molecules_for_evaluation(self,model):
        chunks=[]
        for start in range(0,len(self.eval_representatives),4096):
            fp=decode_packed(self.packed[self.eval_representatives[start:start+4096]])
            chunks.append(model.encode_mol(fp))
        return torch.cat(chunks,dim=0)


@torch.inference_mode()
def evaluate(model,assets,projected=None,recipient=None,donor=None,constant=None):
    model.eval();device=assets.device
    if projected is None:projected=assets.molecules_for_evaluation(model)
    n=len(assets.rows['val']['identifiers'])
    if recipient is None:recipient=np.arange(n,dtype=np.int64)
    if donor is None:donor=recipient
    recipients=np.asarray(recipient,np.int64);donors=np.asarray(donor,np.int64)
    scores=np.full((len(recipients),256),-np.inf,np.float32);ranks=[];greater=[];equal=[]
    for start in range(0,len(recipients),64):
        rr=torch.as_tensor(recipients[start:start+64],device=device);dd=torch.as_tensor(donors[start:start+64],device=device)
        if constant is None:spec=assets.spectra['val'][dd]
        else:spec=constant.unsqueeze(0).expand(len(rr),-1)
        q=model.encode_ms(spec);mol=projected[assets.eval_indices[rr]]
        # Released EmbCos validation scores cosine, not divided by learned temperature.
        values=torch.einsum('bd,bkd->bk',q,mol).masked_fill(~assets.eval_mask[rr],float('-inf'))
        slots=assets.gt_slots[rr];gt=values.gather(1,slots[:,None]);mask=assets.eval_mask[rr]
        g=((values>gt)&mask).sum(1);e=((values==gt)&mask).sum(1)
        order=torch.argsort(values,dim=1,descending=True,stable=True)
        sampled=(order==slots[:,None]).nonzero(as_tuple=False)[:,1]+1
        greater.extend(g.cpu().tolist());equal.extend(e.cpu().tolist());ranks.extend(sampled.cpu().tolist())
        scores[start:start+len(rr)]=values.cpu().numpy()
    g=np.asarray(greater,np.int32);e=np.asarray(equal,np.int32);vectors=uniform_tie_metrics(g,e)
    groups=assets.rows['val']['connectivities'][recipients]
    result={'recipient_rows':recipients,'spectrum_rows':donors,'identifiers':assets.rows['val']['identifiers'][recipients],
        'connectivities':groups,'candidate_counts':assets.rows['val']['lengths'][recipients],
        'gt_slots':assets.rows['val']['gt_slots'][recipients],'greater':g,'equal':e,
        'ranks':np.asarray(ranks,np.int32),'rank_best':g+1,'rank_worst':g+e,'expected_metrics':vectors,'scores':scores}
    metrics=summarize_metrics(vectors,groups)
    metrics.update({'candidate_recall':1.,'gt_exact_tie_queries':int(np.sum(e>1)),'max_gt_tie_size':int(e.max()),
        'scores':'float32 cosine; physical candidate order from cache; uniform exact tie expectation primary'})
    return metrics,result


def checkpoint_payload(model,optimizer,scheduler,args,global_step,epoch,order,cursor,negative_gen,shuffle_gen,history,best,assets,loss_sum,examples):
    return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
        'seed':args.seed,'contract_sha256':sha256_file(args.config),'cache_manifest_sha256':assets.input_digests['cache_manifest'],
        'runner_sha256':sha256_file(Path(__file__)),'module_sha256':sha256_file(ROOT/'glmr/embcos.py'),
        'global_step':global_step,'epoch':epoch,'order':order,'cursor':cursor,
        'negative_rng':negative_gen.get_state(),'shuffle_rng':shuffle_gen.get_state(),'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all() if assets.device.type=='cuda' else [],
        'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'history':history,'best':best,
        'epoch_loss_sum':loss_sum,'epoch_examples':examples}


def train(args,c,assets,out):
    started=time.time();torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    if assets.device.type=='cuda':torch.cuda.manual_seed_all(args.seed)
    model=EmbCos(c['e2']).to(assets.device);cfg=c['e2']['training']
    ceiling=args.smoke_steps if args.smoke else cfg['n_max_steps']
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=LambdaLR(optimizer,cosine_schedule(cfg['n_warmup_steps'],cfg['n_max_steps']))
    negative_gen=torch.Generator().manual_seed(args.seed);shuffle_gen=torch.Generator().manual_seed(args.seed)
    step=0;epoch=0;cursor=0;order=None;history=[];loss_sum=0.;examples=0;best={'recall@1':-1.,'global_step':None}
    if args.resume:
        p=torch.load(out/'checkpoint_last.pt',map_location=assets.device,weights_only=False)
        assert p['seed']==args.seed and p['contract_sha256']==sha256_file(args.config)
        assert p['cache_manifest_sha256']==assets.input_digests['cache_manifest']
        assert p['runner_sha256']==sha256_file(Path(__file__)) and p['module_sha256']==sha256_file(ROOT/'glmr/embcos.py')
        model.load_state_dict(p['model']);optimizer.load_state_dict(p['optimizer']);scheduler.load_state_dict(p['scheduler'])
        step=p['global_step'];epoch=p['epoch'];cursor=p['cursor'];order=p['order'].cpu() if p['order'] is not None else None
        history=p['history'];best=p['best'];loss_sum=p['epoch_loss_sum'];examples=p['epoch_examples'];negative_gen.set_state(p['negative_rng'].cpu());shuffle_gen.set_state(p['shuffle_rng'].cpu())
        torch.set_rng_state(p['torch_rng'].cpu());random.setstate(p['python_rng']);np.random.set_state(p['numpy_rng'])
        if p['cuda_rng']:torch.cuda.set_rng_state_all([state.cpu() for state in p['cuda_rng']])
    elif (out/'checkpoint_last.pt').exists():raise FileExistsError('Refusing to overwrite an existing run')
    if step==0:
        metrics,ranks=evaluate(model,assets);write_json(out/'epoch0_val.json',metrics);np.savez(out/'epoch0_val.npz',**ranks)
    n=len(assets.rows['train']['identifiers']);period=time.time();period_step=step
    batch_size=cfg['batch_size']
    while step<ceiling:
        if order is None:
            order=torch.randperm(n,generator=shuffle_gen);cursor=0;epoch+=1;loss_sum=0.;examples=0
        model.train()
        while cursor<n and step<ceiling:
            batch=order[cursor:cursor+batch_size];indices=[];masks=[]
            for row in batch.tolist():
                length=assets.rows['train']['lengths'][row]
                candidates=assets.rows['train']['candidates'][row,:length].tolist()
                chosen,mask=sample_candidates(candidates,assets.exclusion,cfg['k_candidates'],negative_gen)
                indices.append(chosen);masks.append(mask)
            selected=torch.stack(indices).to(assets.device);mask=torch.stack(masks).to(assets.device)
            molecule=decode_packed(assets.packed[selected]);molecule.masked_fill_(~mask.unsqueeze(-1),0.)
            spectrum=assets.spectra['train'][batch.to(assets.device)]
            logits=model(spectrum,molecule,mask);labels=torch.zeros(len(batch),dtype=torch.long,device=assets.device)
            loss=F.cross_entropy(logits,labels)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite train loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_val']);optimizer.step();scheduler.step()
            step+=1;cursor+=len(batch);loss_sum+=float(loss.detach())*len(batch);examples+=len(batch)
            if step%100==0:
                if assets.device.type=='cuda':torch.cuda.synchronize()
                print(json.dumps({'step':step,'epoch':epoch,'candidate_ce':loss_sum/examples,'temperature':float(model.temperature.detach()),
                    'lr':scheduler.get_last_lr()[0],'steps_per_second':(step-period_step)/(time.time()-period),
                    'cuda_peak_mib':torch.cuda.max_memory_allocated()/1024**2 if assets.device.type=='cuda' else 0,
                    'cgroup_current_mib':int(Path('/sys/fs/cgroup/memory.current').read_text())/1024**2}),flush=True)
                period=time.time();period_step=step
            if step%1000==0:
                payload=checkpoint_payload(model,optimizer,scheduler,args,step,epoch,order,cursor,negative_gen,shuffle_gen,history,best,assets,loss_sum,examples)
                save_checkpoint(out/'checkpoint_last.pt',payload)
        metrics,ranks=evaluate(model,assets)
        entry={'epoch':epoch,'global_step':step,'training_ce':loss_sum/max(1,examples),'validation':metrics}
        history.append(entry);print('VALIDATION',json.dumps(entry),flush=True)
        if metrics['recall@1']>best['recall@1']:
            best={'recall@1':metrics['recall@1'],'global_step':step,'epoch':epoch,'metrics':metrics}
            payload=checkpoint_payload(model,optimizer,scheduler,args,step,epoch,order,cursor,negative_gen,shuffle_gen,history,best,assets,loss_sum,examples)
            save_checkpoint(out/'checkpoint.pt',payload);np.savez(out/'val_ranks.npz',**ranks)
            write_json(out/'selected_val.json',best)
        if cursor>=n:order=None;cursor=0
        payload=checkpoint_payload(model,optimizer,scheduler,args,step,epoch,order,cursor,negative_gen,shuffle_gen,history,best,assets,loss_sum,examples)
        save_checkpoint(out/'checkpoint_last.pt',payload)
        write_json(out/'history.json',{'entries':history});loss_sum=0.;examples=0
    summary={'status':'PLUMBING_ONLY' if args.smoke else 'TRAINING_COMPLETE','seed':args.seed,'best':best,'history':history,
        'global_step':step,'ceiling':ceiling,'train_rows':n,'val_rows':len(assets.rows['val']['identifiers']),
        'parameters':sum(p.numel() for p in model.parameters()),'elapsed_seconds':time.time()-started,
        'peak_cuda_mib':torch.cuda.max_memory_allocated()/1024**2 if assets.device.type=='cuda' else 0,
        'contract_sha256':sha256_file(args.config),'inputs':assets.input_digests,'eval_cache':assets.evaluation_cache,
        'source':{'runner':sha256_file(Path(__file__)),'module':sha256_file(ROOT/'glmr/embcos.py')},
        'checkpoint_sha256':sha256_file(out/'checkpoint.pt'),'val_ranks_sha256':sha256_file(out/'val_ranks.npz'),
        'environment':{'torch':torch.__version__,'numpy':np.__version__,'device':str(assets.device),
            'gpu':torch.cuda.get_device_name(0) if assets.device.type=='cuda' else None,
            'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'workers':0},
        'resumed':args.resume,'elapsed_scope':'current process wall time; all prior segments retained in logs',
        'test_evaluated':False,'temperature_scoring':'train logits divide by learned temperature; validation uses cosine as release'}
    write_json(out/'train_summary.json',summary);print('TRAINING COMPLETE',json.dumps(best),flush=True)


def controls(args,c,assets,out):
    summary=json.loads((out/'train_summary.json').read_text());assert summary['status']=='TRAINING_COMPLETE'
    assert summary['source']['runner']==sha256_file(Path(__file__)) and summary['source']['module']==sha256_file(ROOT/'glmr/embcos.py')
    model=EmbCos(c['e2']).to(assets.device);p=torch.load(out/'checkpoint.pt',map_location=assets.device,weights_only=False)
    assert p['cache_manifest_sha256']==assets.input_digests['cache_manifest'];model.load_state_dict(p['model']);model.eval()
    projection=assets.molecules_for_evaluation(model)
    results={};control_dir=out/'controls';control_dir.mkdir(exist_ok=True)
    metrics,real=evaluate(model,assets,projected=projection)
    expected=dict(np.load(out/'val_ranks.npz'))
    for key in ('greater','equal','scores','gt_slots','ranks'):assert np.array_equal(real[key],expected[key]),('Frozen replay mismatch',key)
    np.savez(control_dir/'real.npz',**real);results['real']=metrics
    for name,constant in [('zero',torch.zeros(10050,device=assets.device)),('train_mean',assets.train_mean)]:
        metric,data=evaluate(model,assets,projected=projection,constant=constant)
        np.savez(control_dir/(name+'.npz'),**data);results[name]=metric
    with np.load(ROOT/c['assets']['wrong_spectrum_assignments']) as donor:
        recipient=donor['recipient_rows'].astype(np.int64)
        for seed in c['common']['controls']['donor_seeds']:
            assigned=donor[f'seed{seed}_donor_rows'].astype(np.int64)
            assert set(assigned)==set(recipient) and np.all(assigned!=recipient)
            assert np.all(assets.rows['val']['connectivities'][assigned]!=assets.rows['val']['connectivities'][recipient])
            metrics,data=evaluate(model,assets,projected=projection,recipient=recipient,donor=assigned)
            name=f'wrong_seed{seed}';np.savez(control_dir/(name+'.npz'),**data);results[name]=metrics
    write_json(control_dir/'summary.json',{'status':'COMPLETE','frozen_replay':'exact scores/counts/ranks passed','metrics':results,
        'checkpoint_sha256':summary['checkpoint_sha256'],'donor_sha256':sha256_file(ROOT/c['assets']['wrong_spectrum_assignments']),
        'spectrum_replacement':'candidate memberships/order fixed; no metadata in EmbCos','test_evaluated':False})
    print('CONTROLS COMPLETE',json.dumps(results),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=('train','controls'),default='train')
    p.add_argument('--config',type=Path,default=ROOT/'docs/msalign/configs/external_matched_e0_20261014.json')
    p.add_argument('--cache',type=Path,default=ROOT/'outputs/embcos_official_e2_20261014/cache')
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--seed',type=int,choices=(20260621,20260622,20260623),default=20260621)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda');p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-steps',type=int,default=200)
    p.add_argument('--train-limit',type=int,default=0);p.add_argument('--val-limit',type=int,default=0)
    args=p.parse_args()
    if not args.smoke and (args.train_limit or args.val_limit):raise ValueError('Reduced inputs allowed only for plumbing smoke')
    if args.action=='controls' and args.smoke:raise ValueError('Controls require the formal frozen full-val checkpoint')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Host CUDA unavailable')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.use_deterministic_algorithms(True)
    c=json.loads(args.config.read_text());device=torch.device(args.device);out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    assets=Assets(args.cache.resolve(),args.config,device,args.train_limit,args.val_limit)
    print('ASSETS READY',json.dumps(assets.evaluation_cache),'GPU allocated',torch.cuda.memory_allocated()/1024**2 if device.type=='cuda' else 0,flush=True)
    write_json(out/'run_contract.json',{'contract':c,'contract_sha256':sha256_file(args.config),'seed':args.seed,
        'cache':str(args.cache),'inputs':assets.input_digests,'smoke':args.smoke,'test_evaluated':False})
    if args.action=='train':train(args,c,assets,out)
    else:controls(args,c,assets,out)


if __name__=='__main__':main()
