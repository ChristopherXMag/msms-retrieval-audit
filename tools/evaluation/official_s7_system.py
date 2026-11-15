"""Frozen SCCF/D2/D3 inference with explicit inputs and block-wise uniform ties."""
from __future__ import annotations
import argparse,gc,json,sys,time,shutil
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.evaluation.official_s7_common import *
from tools.reranking import audit_official_d3_full_system as old
from tools.evaluation.official_s7_numerics import pair_scores
from tools.evaluation.official_s7_numerics import scalar_features
top=old.top;ctl=old.ctl;D2=old.D2;D3=old.D3;S2=old.S2

class System:
    def __init__(self,source,inputs,projection_dir,means,reference=False):
        self.source=read(source);self.pool=load(ROOT/self.source['npz'])
        self.ids=self.pool['identifiers'];self.groups=self.pool['connectivities'];self.n=len(self.ids)
        self.meta=top.records(ROOT/self.source['metadata']);self.inputs=top.records(inputs)
        assert [q['identifier'] for q in self.inputs]==self.ids.tolist()==[q['identifier'] for q in self.meta]
        self.means=load(means);self.names=read(D3/'features/summary.json')['feature_names']
        self.dreams=np.load(ROOT/self.source['dreams'],mmap_mode='r')
        self.device=torch.device('cuda');self.reference=reference;self.out=None
        rows=np.unique(self.pool['candidate_rows']);self.local=np.full(int(rows.max())+1,-1,np.int32);self.local[rows]=np.arange(len(rows))
        self.smiles=top.read_smiles(rows)
        for i,q in enumerate(self.meta):
            target=int(self.pool['candidate_rows'][self.pool['indptr'][i]])
            assert self.smiles[target]==q['canonical_isomeric_smiles'],'GT source/index identity mismatch; never insert GT'
        cache=Path(projection_dir);cache.mkdir(parents=True,exist_ok=True)
        self.models=[];self.projections=[];self.real_spec=[]
        store=ctl.MoleculeEmbeddingStore(None,'mean',old.FP,additional_fingerprint_caches=[],preload_sparse_features=False,mol_blocks=self.source['molecule_blocks'])
        cached_rows=cache/'candidate_rows.npy'
        if cached_rows.exists():assert np.array_equal(np.load(cached_rows),rows)
        else:np.save(cached_rows,rows)
        for j,entry in enumerate(self.source['checkpoints']):
            checkpoint=torch.load(ROOT/entry['path'],map_location='cpu',weights_only=False)
            model=ctl.build_projection_model(spec_dim=1024,mol_dim=store.dim,mol_block_dims=store.block_dims,mol_block_names=store.block_names,config=ctl.checkpoint_model_config(checkpoint)).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict']);model.eval();self.models.append(model)
            path=cache/f'seed{SEEDS[j]}.npy'
            if not path.exists():
                assert shutil.disk_usage(cache).free>(len(rows)*1024*4+12*2**30),'Projection cache would violate disk reserve'
                temporary=path.with_name(path.stem+'.staging.npy')
                a=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(len(rows),1024))
                for st in range(0,len(rows),256):
                    top.d2.memory_guard();ix=torch.from_numpy(rows[st:st+256].astype(np.int64))
                    ordered,inverse=store.canonicalize_unique_indices(ix,torch.arange(len(ix)))
                    with torch.inference_mode():a[st:st+len(ix)]=model.project_mol(store.gather(ordered).to(self.device)).cpu().numpy()[inverse]
                a.flush();del a;temporary.replace(path)
            a=np.load(path,mmap_mode='r');assert a.shape==(len(rows),1024) and a.dtype==np.float32
            self.projections.append(a)
            with torch.inference_mode():self.real_spec.append(torch.cat([model.project_spec(torch.from_numpy(np.array(self.dreams[st:st+2048],copy=True)).to(self.device)).cpu() for st in range(0,self.n,2048)]))
        del store,checkpoint;gc.collect()
        self.pair5=[];self.pair20=[];pair=read(S2/'pair_reranker/formal_top5/summary.json')
        for entries,dest in [(pair['checkpoint_files'],self.pair5),([dict(path=str((D3/f'formal/pair/seed{s}.pt').relative_to(ROOT))) for s in (20261001,20261002,20261003)],self.pair20)]:
            for item in entries:
                state=torch.load(ROOT/item['path'],map_location='cpu',weights_only=False)
                model=top.pair.PairReranker(1024,11,256,64,.1).to(self.device);model.load_state_dict(state['model_state_dict']);dest.append(model.eval())
        self.trees={k:{arm:[lgb.Booster(model_file=str(parent/f'trees/{arm}_seed{s}.txt')) for s in top.SEEDS] for arm in ('scalar','detail')} for k,parent in [(5,D2),(20,D3)]}
        self.ev=old.BoundedEvidence();self.max_logit_error=0.
        self.native=None
        if self.source.get('native_real_vectors'):
            from tools.evaluation.official_s7_inputs import NativeVectors
            self.native=NativeVectors(ROOT/self.source['native_real_vectors'])
            assert np.array_equal(self.native.identifiers,self.ids)
        if reference:
            self.ref2=load(D2/'formal_validation/scores_ranks.npz');self.ref3=load(D3/'formal/report/scores_ranks.npz')
            self.cached20=load(D3/'formal/features/features.npz');self.cached5=load(D2/'group_ablation/features_real.npz')
            self.frozen5=torch.load(str(ROOT/pair['args']['val_top5']),map_location='cpu',weights_only=False,mmap=True)
            self.molecule5=torch.load(str(ROOT/pair['args']['candidate_projection']),map_location='cpu',weights_only=False,mmap=True)['embeddings']
            self.frozen20_spec=np.load(D3/'formal/export/spec.npy',mmap_mode='r');self.frozen20_mol=np.load(D3/'formal/export/vectors.npy',mmap_mode='r')
            self.frozen20_local=np.load(D3/'formal/export/candidate_local.npy',mmap_mode='r');self.val_start=read(D3/'formal/export/summary.json')['train_queries']
            self.route_ref=load(ROOT/'outputs/lightgbm_top20_support_fallback_20261017/per_query.npz')

    def tree(self,k,arm,features,support,anchor):
        width=10 if arm=='scalar' else 36;features=features[:,:,:width];values=[]
        for model in self.trees[k][arm]:
            residual=model.predict(features.reshape(-1,width),num_threads=2).reshape(features.shape[:2]);residual[~support]=0;values.append(residual)
        return anchor+np.mean(np.stack(values),axis=0)

    def evidence(self,inputs,base,family):
        if family!=5 or base.shape[1]==5:return old.build_evidence(self.ev,inputs,base)
        # Natural pools shorter than five retain every candidate, without padding identities.
        z=self.ev.build(inputs,base,coverage_override=np.zeros_like(base));valid=torch.from_numpy(z['valid']).to(self.device)
        intensity=torch.zeros((len(inputs),100),device=self.device);present=torch.zeros((len(inputs),100),dtype=torch.bool,device=self.device)
        for i,q in enumerate(inputs):
            count=len(q['observed_spectrum'])
            if count:intensity[i,:count]=torch.tensor([p[1] for p in q['observed_spectrum']],device=self.device);present[i,:count]=True
        chunks=[]
        for st in range(0,len(inputs),16):
            v,it,pm=valid[st:st+16],intensity[st:st+16],present[st:st+16];n=len(v)
            def pad(t):return torch.cat([t,t[-1:].expand((16-n,)+t.shape[1:])],0) if n<16 else t
            v,it,pm=[pad(t) for t in (v,it,pm)]
            with torch.inference_mode():chunks.append(top.d2.candidate_relative_score(top.d2.masked_null_support(torch.ones(v.shape,device=self.device),v),pm,it).cpu().numpy()[:n])
        return self.ev.build(inputs,base,coverage_override=np.concatenate(chunks))

    def evaluate(self,name,queries,donor_rows=None,tie_seed=20260926):
        queries=np.asarray(queries,np.int64);n=len(queries)
        ranks={k:np.empty(n,np.int32) for k in ('stage1','original_d3','selected')}
        values={k:np.empty((n,4)) for k in ranks}
        data={k:np.full((n,20),np.nan,dtype) for k,dtype in [('stage1_scores',np.float32),('pair20',np.float32),('coverage20',np.float32),('detail20',np.float64),('selected_scores',np.float64)]}
        data.update(candidate_rows=np.full((n,20),-1,np.int32),routed_order=np.full((n,20),-1,np.int32),
            support20=np.zeros(n,bool),support5=np.zeros(n,bool),route5=np.zeros(n,bool),source_rows=np.empty(n,np.int64),
            gt_local=np.empty(n,np.int64),pair5_rows=np.full((n,5),-1,np.int32),pair5_scores=np.full((n,5),np.nan,np.float32))
        alternate=np.zeros((n,2),bool);boundary=np.zeros((n,2),bool);ties=Counter()
        for start in range(0,n,32):
            index=np.arange(start,min(start+32,n));current=queries[index]
            widths=np.minimum(20,self.pool['candidate_counts'][current])
            for k in np.unique(widths):
                pos=index[widths==k];current=queries[pos];count=len(current);k=int(k)
                source=current if donor_rows is None else np.asarray(donor_rows,np.int64)[pos]
                pools=[self.pool['candidate_rows'][self.pool['indptr'][q]:self.pool['indptr'][q+1]] for q in current]
                width=max(map(len,pools));padded=np.stack([np.pad(p,(0,width-len(p)),mode='edge') for p in pools]);local=self.local[padded]
                if name in ('zero','train_mean'):
                    raw=np.zeros(1024,np.float32) if name=='zero' else self.means['embedding']
                    with torch.inference_mode():spectra=[m.project_spec(torch.from_numpy(raw).to(self.device)[None]).expand(count,-1) for m in self.models]
                else:spectra=[a[source].to(self.device) for a in self.real_spec]
                scores=[]
                with torch.inference_mode():
                    for model,mol,spec in zip(self.models,self.projections,spectra):
                        vectors=torch.from_numpy(np.array(mol[local],copy=True)).to(self.device)
                        scores.append((torch.einsum('bd,bkd->bk',spec,vectors)/model.temperature()).cpu().numpy())
                scores=np.stack(scores,-1)
                if name=='real':
                    for i,q in enumerate(current):
                        a,b=self.pool['indptr'][q:q+2];frozen=self.pool['scores'][:,a:b].T
                        self.max_logit_error=max(self.max_logit_error,float(np.abs(scores[i,:len(pools[i])]-frozen).max()))
                        scores[i,:len(pools[i])]=frozen
                ensemble=scores.astype(np.float64).mean(-1).astype(np.float32)
                selected=[];bases=[];seed_scores=[];inputs=[];full_orders=[]
                for i,q in enumerate(current):
                    full=ensemble[i,:len(pools[i])];identifier=str(self.ids[q])
                    perm=np.random.default_rng(20261017+int(q)).permutation(len(full))
                    order=perm[ctl.candidate_order(identifier,pools[i][perm],full[perm],tie_seed)]
                    assert np.array_equal(order,ctl.candidate_order(identifier,pools[i],full,tie_seed))
                    for j,seed in enumerate(TIE_SEEDS[1:]):
                        other=ctl.candidate_order(identifier,pools[i],full,seed)
                        alternate[pos[i],j]=not np.array_equal(pools[i][other[:k]],pools[i][order[:k]])
                    for j,cut in enumerate((5,20)):
                        boundary[pos[i],j]=len(full)>cut and full[order[cut-1]]==full[order[cut]]
                    selected.append(order[:k]);full_orders.append(order);bases.append(full[order[:k]]);seed_scores.append(scores[i,order[:k]])
                    qin=dict(self.inputs[q]);qin['candidate_global_rows']=pools[i][order[:k]].tolist();qin['candidate_smiles']=[self.smiles[int(row)] for row in pools[i][order[:k]]]
                    qin['observed_spectrum']=[] if name=='zero' else self.means['peaks'].tolist() if name=='train_mean' else self.inputs[int(source[i])]['observed_spectrum'];inputs.append(qin)
                    values['stage1'][pos[i]]=ctl.uniform_tie_contributions(full,0);ranks['stage1'][pos[i]]=int(np.flatnonzero(order==0)[0])+1
                    ties['stage1']+=int((full==full[0]).sum()>1)
                selected=np.stack(selected);base=np.asarray(bases,np.float32);ss=np.asarray(seed_scores,np.float32)
                candidates=np.take_along_axis(padded,selected,axis=1)
                mol=np.array(self.projections[0][self.local[candidates]],copy=True).astype(np.float16).astype(np.float32)
                spec=spectra[0].cpu().numpy().astype(np.float16).astype(np.float32)
                if name=='real' and self.reference:
                    assert np.array_equal(candidates,self.ref3['candidate_rows'][current])
                    ix=self.val_start+current;mol=np.array(self.frozen20_mol[self.frozen20_local[ix]],copy=True).astype(np.float32);spec=np.array(self.frozen20_spec[ix],copy=True).astype(np.float32)
                if name=='real' and self.native is not None:
                    spec,mol=self.native.get(20,current,candidates)
                extent=min(512,self.n-(int(current[0])//512)*512) if name=='real' else None
                pair20=pair_scores(self.pair20,spec,mol,ss,base,extent,physical_width=20)
                if name=='real' and self.reference:
                    built=dict(features=self.cached20['features'][current],coverage=self.cached20['coverage'][current],support=self.cached20['support'][current])
                elif name=='zero':built=dict(features=np.zeros((count,k,36),np.float32),coverage=np.zeros((count,k),np.float32),support=np.zeros(count,bool))
                else:built=self.evidence(inputs,base,20)
                supported=np.asarray([old.support(q,q['candidate_smiles']) for q in inputs])
                support5=np.asarray([old.support(q,q['candidate_smiles'][:5]) for q in inputs]);assert np.array_equal(supported,built['support'])
                route=old.route_to_d2(support5,supported)
                anchor=top.d2.deployment_anchor(pair20,built['coverage'])
                detail=self.tree(20,'detail',built['features'],supported,anchor);final=detail.copy();p5=np.full((count,min(5,k)),np.nan,np.float32)
                fallback=np.flatnonzero(route)
                if len(fallback):
                    in5=[dict(inputs[i],candidate_smiles=inputs[i]['candidate_smiles'][:5],candidate_global_rows=inputs[i]['candidate_global_rows'][:5]) for i in fallback]
                    mol5=mol[fallback,:5].copy();spec5=spec[fallback].copy()
                    if name=='real' and self.reference:
                        mol5=self.molecule5[self.frozen5['candidate_local_indices'][current[fallback]].long()].float().numpy();spec5=self.frozen5['spec_embeddings'][current[fallback]].float().numpy()
                    if name=='real' and self.native is not None:
                        spec5,mol5=self.native.get(5,current[fallback],candidates[fallback,:5])
                    ps=old.pair_score(self.pair5,spec5,mol5,ss[fallback,:5],base[fallback,:5],extent)
                    b5=dict(features=self.cached5['features'][current[fallback]],coverage=self.cached5['coverage'][current[fallback]],support=support5[fallback]) if name=='real' and self.reference else self.evidence(in5,base[fallback,:5],5)
                    assert b5['support'].all()
                    d5=self.tree(5,'detail',b5['features'],b5['support'],top.d2.deployment_anchor(ps,b5['coverage']))
                    final[fallback,:5]=d5;final[fallback,5:]=np.nan;p5[fallback]=ps
                    if name=='real' and self.reference:
                        assert np.array_equal(ps,self.ref2['scores_pair'][current[fallback]])
                        assert np.array_equal(d5,self.ref2['scores_detail'][current[fallback]])
                for i,q in enumerate(current):
                    j=pos[i];full=ensemble[i,:len(pools[i])];identifier=str(self.ids[q])
                    o20=ctl.candidate_order(identifier,candidates[i],detail[i],tie_seed)
                    label=np.flatnonzero(selected[i]==0);data['gt_local'][j]=int(label[0]) if len(label) else -1
                    r20=ranks['stage1'][j] if not len(label) else int(np.flatnonzero(o20==label[0])[0])+1
                    ranks['original_d3'][j]=r20;values['original_d3'][j]=expected_reranked(full,selected[i],detail[i])
                    if len(label):ties['original_d3']+=int((detail[i]==detail[i,label[0]]).sum()>1)
                    order=candidates[i,o20]
                    if route[i]:
                        cut=min(5,k);o5=ctl.candidate_order(identifier,candidates[i,:cut],final[i,:cut],tie_seed)
                        order=np.r_[candidates[i,:cut][o5],candidates[i,cut:]]
                        values['selected'][j]=expected_reranked(full,selected[i,:cut],final[i,:cut])
                        ranks['selected'][j]=ranks['stage1'][j] if not len(label) or label[0]>=cut else int(np.flatnonzero(o5==label[0])[0])+1
                        if len(label) and label[0]<cut:ties['selected']+=int((final[i,:cut]==final[i,label[0]]).sum()>1)
                    else:
                        ranks['selected'][j]=r20;values['selected'][j]=values['original_d3'][j]
                        if len(label):ties['selected']+=int((detail[i]==detail[i,label[0]]).sum()>1)
                    data['routed_order'][j,:k]=order
                data['candidate_rows'][pos,:k]=candidates;data['stage1_scores'][pos,:k]=base;data['pair20'][pos,:k]=pair20;data['coverage20'][pos,:k]=built['coverage']
                data['detail20'][pos,:k]=detail;data['selected_scores'][pos,:k]=final;data['support20'][pos]=supported;data['support5'][pos]=support5;data['route5'][pos]=route
                data['source_rows'][pos]=source;data['pair5_rows'][pos,:min(5,k)]=candidates[:,:5];data['pair5_scores'][pos,:min(5,k)]=p5
                if name=='real' and self.reference:
                    assert np.array_equal(pair20,self.ref3['scores_pair'][current])
                    assert np.array_equal(detail,self.ref3['scores_detail'][current])
                    assert np.array_equal(ranks['selected'][pos],self.route_ref['ranks_routed_d3'][current])
            if start%512==0:print('SYSTEM_MODE',name,tie_seed,start,'/',n,flush=True)
        assert self.max_logit_error<=5e-5
        result=dict(queries=queries,groups=self.groups[queries],identifiers=self.ids[queries],**data,
            **{'ranks_'+k:v for k,v in ranks.items()},**{'values_'+k:v for k,v in values.items()},
            alternate_order=alternate,boundary_ties=boundary,candidate_counts=self.pool['candidate_counts'][queries])
        assert np.array_equal(ranks['selected']<=20,ranks['stage1']<=20)
        info=dict(queries=n,support20=int(data['support20'].sum()),support5=int(data['support5'].sum()),fallback=int(data['route5'].sum()),
            gt_exact_ties=dict(ties),boundary_ties5=int(boundary[:,0].sum()),boundary_ties20=int(boundary[:,1].sum()),
            alternate_order_queries=alternate.sum(0).tolist(),max_real_stage1_logit_error=self.max_logit_error)
        return result,info

    def real_comparators(self,real):
        n=self.n;output={key:np.empty((n,4)) for key in ('top5_pair','top5_coverage','top5_doubled_coverage','top5_scalar','top5_detail','top20_pair','top20_coverage','top20_doubled_coverage','top20_scalar','top20_detail')}
        for start in range(0,n,512):
            end=min(start+512,n);current=np.arange(start,end);k5=5
            # Validation has full K; shorter future pools are processed individually.
            if np.any(self.pool['candidate_counts'][current]<20):
                for q in current:self._comparators_block(np.array([q]),real,output)
            else:self._comparators_block(current,real,output)
        return output

    def _comparators_block(self,current,real,output):
        k=min(20,int(self.pool['candidate_counts'][current].min()));cut=min(5,k)
        candidates=real['candidate_rows'][current,:k];base=real['stage1_scores'][current,:k]
        scores=[];selected=[];inputs=[]
        for i,q in enumerate(current):
            a,b=self.pool['indptr'][q:q+2];pool=self.pool['candidate_rows'][a:b];full=self.pool['ensemble_scores'][a:b]
            order=ctl.candidate_order(str(self.ids[q]),pool,full,20260926);selected.append(order[:k])
            scores.append(self.pool['scores'][:,a:b][:,order[:k]].T)
            qin=dict(self.inputs[q],candidate_global_rows=candidates[i].tolist(),candidate_smiles=[self.smiles[int(row)] for row in candidates[i]]);inputs.append(qin)
        ss=np.stack(scores);selected=np.stack(selected)
        spec=self.real_spec[0][current].numpy().astype(np.float16).astype(np.float32)
        mol=np.array(self.projections[0][self.local[candidates[:,:cut]]],copy=True).astype(np.float16).astype(np.float32)
        if self.reference:
            mol=self.molecule5[self.frozen5['candidate_local_indices'][current].long()].float().numpy();spec=self.frozen5['spec_embeddings'][current].float().numpy()
        if self.native is not None:spec,mol=self.native.get(5,current,candidates[:,:cut])
        extent=min(512,self.n-(int(current[0])//512)*512)
        pair5=old.pair_score(self.pair5,spec,mol,ss[:,:cut],base[:,:cut],extent)
        in5=[dict(q,candidate_smiles=q['candidate_smiles'][:cut],candidate_global_rows=q['candidate_global_rows'][:cut]) for q in inputs]
        b5=dict(features=self.cached5['features'][current],coverage=self.cached5['coverage'][current],support=real['support5'][current]) if self.reference else self.evidence(in5,base[:,:cut],5)
        b20=dict(features=self.cached20['features'][current],coverage=self.cached20['coverage'][current],support=self.cached20['support'][current]) if self.reference else dict(features=scalar_features(base,real['coverage20'][current,:k]),coverage=real['coverage20'][current,:k],support=real['support20'][current])
        for width,pair,built,actual in [(5,pair5,b5,cut),(20,real['pair20'][current,:k],b20,k)]:
            anchor=top.d2.deployment_anchor(pair,built['coverage'])
            arms=dict(pair=pair,coverage=anchor,doubled_coverage=top.d2.deployment_anchor(pair,built['coverage'],2.),
                scalar=self.tree(width,'scalar',built['features'],built['support'],anchor),detail=real['detail20'][current,:k] if width==20 and not self.reference else self.tree(width,'detail',built['features'],built['support'],anchor))
            for name,values in arms.items():
                for i,q in enumerate(current):
                    a,b=self.pool['indptr'][q:q+2];full=self.pool['ensemble_scores'][a:b]
                    output[f'top{width}_{name}'][q]=expected_reranked(full,selected[i,:actual],values[i])
            if self.reference:
                ref=self.ref2 if width==5 else self.ref3
                for arm,v in arms.items():assert np.array_equal(v,ref['scores_'+arm][current]),(width,arm)
