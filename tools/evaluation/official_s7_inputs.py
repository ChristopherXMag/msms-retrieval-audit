"""Explicit split input adapters. No main/implicit test discovery or training."""
from __future__ import annotations
import json,sqlite3,shutil
from pathlib import Path
import numpy as np
from tools.evaluation.official_s7_common import *

def metadata_arrays(records):
    import pandas as pd
    from glmr.msalign_external import normalize_adduct
    manifest=read(E1/'assets/manifest.json');vocab={a:i for i,a in enumerate(manifest['adduct_vocabulary'])}
    energy=pd.to_numeric(pd.Series([q.get('collision_energy') for q in records]),errors='coerce').to_numpy(dtype=np.float32)
    assert not np.isinf(energy).any()
    return dict(collision_energy=energy,adduct=np.asarray([vocab.get(normalize_adduct(q.get('adduct')),manifest['unknown_adduct_index']) for q in records],np.int64))

def raw_inputs(records,hdf5):
    import h5py
    from tools.reranking.run_official_lightgbm_transfer import select_dreams_peaks
    result=[]
    with h5py.File(hdf5,'r') as f:
        assert len(f['TITLE'])==len(records)
        for i,q in enumerate(records):
            title=f['TITLE'][i];title=title.decode() if isinstance(title,bytes) else str(title);assert title==q['identifier']
            observed=f['spectrum'][i];valid=(observed[0]>0)&np.isfinite(observed).all(axis=0)&(observed[1]>=0)
            peaks=select_dreams_peaks(observed[0,valid].tolist(),observed[1,valid].tolist(),100)
            result.append(dict(identifier=q['identifier'],query_index=i,adduct=q['adduct'],collision_energy=q.get('collision_energy'),
                instrument=q.get('instrument_type'),precursor_mz=q['precursor_mz'],observed_spectrum=[[p['mz'],p['relative_intensity']] for p in peaks]))
    return result

def bins(records,mgf,out):
    from tools.data.cache_embcos_official import spectra_cache
    return spectra_cache(records,Path(mgf),Path(out))

class NativeVectors:
    def __init__(self,path):
        path=Path(path);manifest=read(path/'manifest.json');assert manifest['status']=='COMPLETE'
        self.identifiers=np.load(path/'identifiers.npy');self.spec=np.load(path/'spec.npy',mmap_mode='r');self.tables={}
        for k in (5,20):self.tables[k]=(np.load(path/f'rows{k}.npy'),np.load(path/f'mol{k}.npy',mmap_mode='r'))
    def get(self,width,queries,candidates):
        rows,values=self.tables[width];ix=np.searchsorted(rows,candidates)
        assert np.all(ix<len(rows)) and np.array_equal(rows[ix],candidates)
        return np.array(self.spec[queries],copy=True).astype(np.float32),np.array(values[ix],copy=True).astype(np.float32)

def native_vectors(source,out,allow_gpu=False):
    """Reuse immutable real-vector rows; only missing rows use each original builder."""
    import torch
    from tools.reranking import audit_official_d3_full_system as old
    from tools.reranking.build_official_datasafe_top5_projection_cache import project_molecules,project_specs
    source=read(source) if isinstance(source,(Path,str)) else source;pool=load(ROOT/source['npz']);out=Path(out)
    assert not out.exists();out.mkdir(parents=True)
    requested={5:set(),20:set()}
    for i,identifier in enumerate(pool['identifiers']):
        a,b=pool['indptr'][i:i+2];rows=pool['candidate_rows'][a:b];scores=pool['ensemble_scores'][a:b]
        for seed in TIE_SEEDS:
            order=old.ctl.candidate_order(str(identifier),rows,scores,seed)
            for width in requested:requested[width].update(rows[order[:width]].tolist())
    ps=read(old.S2/'pair_reranker/formal_top5/summary.json')
    cache=torch.load(str(ROOT/ps['args']['candidate_projection']),map_location='cpu',weights_only=False,mmap=True)
    known={5:(cache['global_rows'].numpy(),cache['embeddings'].numpy()),20:(np.load(old.D3/'formal/export/vector_rows.npy'),np.load(old.D3/'formal/export/vectors.npy',mmap_mode='r'))}
    model=store=None;reports={}
    for width,wanted in requested.items():
        rows=np.asarray(sorted(wanted),np.int32);oldrows,oldvalues=known[width];order=np.argsort(oldrows);sortedrows=oldrows[order]
        ix=np.searchsorted(sortedrows,rows);present=(ix<len(sortedrows));present[present]&=sortedrows[ix[present]]==rows[present]
        assert shutil.disk_usage(out).free>len(rows)*2048+12*2**30
        array=np.lib.format.open_memmap(out/f'mol{width}.npy',mode='w+',shape=(len(rows),1024),dtype=np.float16)
        for st in range(0,len(rows),4096):
            pos=np.arange(st,min(st+4096,len(rows)));pos=pos[present[pos]];array[pos]=oldvalues[order[ix[pos]]]
        missing=np.flatnonzero(~present)
        if len(missing):
            assert allow_gpu,'New real vectors require an authorized GPU stage'
            if model is None:store=old.ctl.MoleculeEmbeddingStore(None,'mean',old.FP,additional_fingerprint_caches=[],preload_sparse_features=False,mol_blocks=source['molecule_blocks']);model=old.top.primary_model(store,'cuda')
            if width==5:array[missing]=project_molecules(rows[missing],store,model,torch.device('cuda'),4096,1024).numpy()
            else:
                with torch.inference_mode():
                    for st in range(0,len(missing),256):
                        old.top.d2.memory_guard();pos=missing[st:st+256]
                        array[pos]=model.project_mol(store.gather(torch.from_numpy(rows[pos].astype(np.int64))).to('cuda')).cpu().half().numpy()
        array.flush();del array;np.save(out/f'rows{width}.npy',rows)
        reports[str(width)]=dict(rows=len(rows),reused=int(present.sum()),new=len(missing),batch=4096 if width==5 else 256,canonicalize=width==5)
    frozen=torch.load(str(ROOT/ps['args']['val_top5']),map_location='cpu',weights_only=False,mmap=True)
    # Reusing a query vector also requires an identical complete source cache, never an ID-only shortcut.
    parent=read(old.SOURCE);reuse=np.array_equal(pool['identifiers'],np.asarray(frozen['identifiers'])) and source['dreams']==parent['dreams']
    if reuse:spec=frozen['spec_embeddings'].numpy()
    else:
        assert allow_gpu
        if model is None:store=old.ctl.MoleculeEmbeddingStore(None,'mean',old.FP,additional_fingerprint_caches=[],preload_sparse_features=False,mol_blocks=source['molecule_blocks']);model=old.top.primary_model(store,'cuda')
        spec=project_specs(ROOT/source['dreams'],model,torch.device('cuda'),4096,1024).numpy()
    np.save(out/'spec.npy',spec);np.save(out/'identifiers.npy',pool['identifiers'])
    for p in out.glob('*.npy'):
        a=np.load(p,mmap_mode='r')
        if a.dtype.kind=='f':assert np.isfinite(a).all()
    write(out/'manifest.json',dict(status='COMPLETE',molecules=reports,spectrum_reused=reuse,spectrum_batch=4096,
        source=source,files={p.name:sha(p) for p in out.glob('*.npy')}))
    del model,store
    return out

def extend_e1(rows,out):
    """Train-frozen tokenizer/encoder, read-only old token aliases, no full-cache copy."""
    import torch,transformers,tokenizers
    from transformers import AutoModel,AutoTokenizer
    from tools.data.cache_moldeberta_official import token_key,encode_inputs
    from tools.reranking.run_official_lightgbm_transfer import memory_guard
    assert transformers.__version__=='4.53.3' and tokenizers.__version__=='0.21.4'
    out=Path(out);assert not out.exists();out.mkdir(parents=True)
    base=E1/'moldeberta_cache';lookup=np.array(np.load(base/'global_to_cache.npy'),copy=True)
    old=np.load(base/'embeddings.npy',mmap_mode='r');base_n=len(old);needed=np.unique(rows['candidates'][rows['candidates']>=0]);missing=needed[lookup[needed]<0]
    assert shutil.disk_usage(out).free>len(missing)*768*4+12*2**30
    strings=[];wanted=set(missing.tolist())
    with (ROOT/read(CONFIG)['assets']['smiles']).open() as f:
        for i,line in enumerate(f):
            if i in wanted:strings.append(line.rstrip('\n'))
    assert len(strings)==len(missing)
    modeldir=ROOT/'external/MolDeBERTa/base_123M_mtr_7d06a5f39098837a6f33ed095e2fa7706e30f8b9'
    acquisition=read(modeldir/'acquisition_manifest.json')
    for name,h in acquisition['files'].items():assert sha(modeldir/name)==h
    array=np.lib.format.open_memmap(out/'embeddings.staging.npy',mode='w+',shape=(len(missing),768),dtype=np.float32)
    aliases=np.empty(len(missing),np.int32);connection=sqlite3.connect(f'file:{base}/input_equivalence.sqlite?mode=ro',uri=True)
    local={};model=tokenizer=None
    if len(missing):
        tokenizer=AutoTokenizer.from_pretrained(modeldir,local_files_only=True);model=AutoModel.from_pretrained(modeldir,local_files_only=True).to('cuda').eval()
    for st in range(0,len(missing),4096):
        memory_guard();encoded=tokenizer(strings[st:st+4096],padding=False,truncation=True,max_length=128);positions=[]
        for j,ids in enumerate(encoded['input_ids']):
            key=token_key(ids);hit=connection.execute('SELECT first_row FROM representations WHERE tokens=?',(key,)).fetchone()
            ref=hit[0] if hit is not None else local.get(key,base_n+st+j)
            aliases[st+j]=ref
            if ref==base_n+st+j:positions.append(j);local[key]=ref
        encode_inputs(model,tokenizer,encoded,positions,128,torch.device('cuda'),array,st)
        for j in range(len(encoded['input_ids'])):
            ref=aliases[st+j]
            if ref<base_n:array[st+j]=old[ref]
            elif ref!=base_n+st+j:array[st+j]=array[ref-base_n]
        array.flush();assert np.isfinite(array[st:st+4096]).all()
    connection.close();array.flush();del array; (out/'embeddings.staging.npy').replace(out/'embeddings.npy')
    lookup[missing]=base_n+np.arange(len(missing),dtype=np.int32)
    np.save(out/'global_rows.npy',missing);np.save(out/'global_to_cache.npy',lookup);np.save(out/'equivalent_input_rows.npy',aliases)
    write(out/'manifest.json',dict(status='COMPLETE',new_rows=len(missing),old_rows=base_n,base_sql_read_only=True,
        batch128_chunk4096_maxlength128_firsttoken=True,model_acquisition_sha256=sha(modeldir/'acquisition_manifest.json'),
        files={p.name:sha(p) for p in out.glob('*.npy')}))
    del model,tokenizer
    return out
