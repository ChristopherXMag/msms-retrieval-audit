#!/usr/bin/env python3
"""Resumable float32 MolDeBERTa cache for frozen official train/val candidates."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import transformers
import tokenizers
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from glmr.embcos import sha256_file,write_json
CONFIG=ROOT/'docs/msalign/configs/external_matched_e0_20261014.json'
UUID='GPU-334b8f1d-7fd8-0f86-a959-90d282b47f47'


def resource_check(gpu=False):
    memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available=int(memory['MemAvailable'].split()[0])*1024
    swap=int(memory['SwapTotal'].split()[0])-int(memory['SwapFree'].split()[0])
    current=int(Path('/sys/fs/cgroup/memory.current').read_text())
    limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
    assert available>32*1024**3 and swap==0 and current<52*1024**3,'Memory safety gate'
    result={'host_available_bytes':available,'swap_used_kib':swap,'cgroup_current_bytes':current,'cgroup_limit_bytes':limit}
    if gpu:
        assert available>64*1024**3 and limit-current>16*1024**3
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        line=next(line for line in raw.splitlines() if UUID in line)
        assert int(line.split(',')[2].strip())<512,'GPU is already claimed'
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==UUID
        assert torch.cuda.is_available()
        result.update(gpu=line,torch_device=torch.cuda.get_device_name(0))
    return result


def prepare(out,config,limit):
    contract=json.loads(config.read_text());src=ROOT/'outputs/embcos_official_e2_20261014/cache'
    manifest=json.loads((src/'manifest.json').read_text())
    assert manifest['status']=='COMPLETE' and manifest['contract_sha256']==sha256_file(config)
    n=manifest['molecule_rows'];used=np.zeros(n,bool);counts={}
    digests={'contract':sha256_file(config),'candidate_manifest':sha256_file(ROOT/contract['assets']['candidate_manifest']),
        'full_smiles':sha256_file(ROOT/contract['assets']['smiles'])}
    assert digests['candidate_manifest']==manifest['candidate_manifest_sha256'] and digests['full_smiles']==manifest['molecule_smiles_sha256']
    for fold in ('train','val'):
        path=src/(fold+'.rows.npz');digest=sha256_file(path)
        assert digest==manifest['files'][fold]['rows_sha256'];digests[fold+'_candidate_rows']=digest
        fold_used=np.zeros(n,bool)
        with np.load(path) as z:
            candidates=z['candidates']
            for start in range(0,len(candidates),4096):
                block=candidates[start:start+4096];fold_used[block[block>=0]]=True
        counts[fold]=int(fold_used.sum());used|=fold_used
    rows=np.flatnonzero(used).astype(np.int32)
    if limit:rows=rows[:limit]
    if (out/'prepared.json').exists():
        p=json.loads((out/'prepared.json').read_text())
        assert p['inputs']==digests and p['rows']==len(rows) and p['limit']==limit
        for name,digest in p['files'].items():assert sha256_file(out/name)==digest,name
        return p
    np.save(out/'global_rows.npy',rows)
    lookup=np.full(n,-1,np.int32);lookup[rows]=np.arange(len(rows),dtype=np.int32)
    np.save(out/'global_to_cache.npy',lookup)
    target=out/'smiles.txt';stage=out/'smiles.staging.txt';cursor=0
    with stage.open('w') as writer,(ROOT/contract['assets']['smiles']).open() as reader:
        for i,line in enumerate(reader):
            if i==int(rows[cursor]):
                value=line.rstrip('\n');assert value
                writer.write(value+'\n');cursor+=1
                if cursor==len(rows):break
    assert cursor==len(rows);stage.replace(target)
    p={'status':'PREPARED','rows':len(rows),'full_universe':n,'referenced_by_fold':counts,'limit':limit,
        'inputs':digests,'files':{name:sha256_file(out/name) for name in ('global_rows.npy','global_to_cache.npy','smiles.txt')},
        'scope':'Train/val candidate union only; no pool alteration, candidate deletion or test query dataset'}
    write_json(out/'prepared.json',p);return p


def token_key(ids):
    values=np.asarray(ids,dtype=np.int64)
    assert len(values)>0 and values.min()>=0 and values.max()<65536
    return values.astype('<u2').tobytes()


def register_inputs(connection,sequences,start):
    refs=np.empty(len(sequences),np.int64);unique=[]
    for j,ids in enumerate(sequences):
        key=token_key(ids);row=connection.execute('SELECT first_row FROM representations WHERE tokens=?',(key,)).fetchone()
        if row is None:
            index=start+j;connection.execute('INSERT INTO representations(tokens,first_row) VALUES(?,?)',(key,index));unique.append(j)
        else:index=row[0]
        refs[j]=index
    return refs,unique


@torch.inference_mode()
def encode_inputs(model,tokenizer,encoded,positions,batch_size,device,array,start):
    for offset in range(0,len(positions),batch_size):
        selection=positions[offset:offset+batch_size]
        samples=[{key:encoded[key][i] for key in encoded} for i in selection]
        batch=tokenizer.pad(samples,padding=True,return_tensors='pt',verbose=False)
        values=model(**{key:value.to(device) for key,value in batch.items()}).last_hidden_state[:,0,:]
        assert values.shape==(len(selection),768) and torch.isfinite(values).all()
        array[start+np.asarray(selection)]=values.cpu().numpy().astype(np.float32,copy=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True);parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--model-dir',type=Path,default=ROOT/'external/MolDeBERTa/base_123M_mtr_7d06a5f39098837a6f33ed095e2fa7706e30f8b9')
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--limit',type=int,default=0,help='Plumbing-only prefix; not evidence')
    parser.add_argument('--batch-size',type=int,default=128);parser.add_argument('--chunk-size',type=int,default=4096)
    parser.add_argument('--max-chunks',type=int,default=0,help='Bounded interruption/soak; leaves resumable prefix')
    args=parser.parse_args();assert args.limit>=0 and args.batch_size>0 and args.chunk_size>0
    assert transformers.__version__=='4.53.3' and tokenizers.__version__=='0.21.4'
    out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'writer.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text());assert manifest['status'] in ('COMPLETE','PLUMBING_ONLY')
        print('CACHE ALREADY COMPLETE',flush=True);return
    resource_check();prepared=prepare(out,args.config,args.limit)
    if args.prepare_only:print('PREPARED',json.dumps(prepared),flush=True);return
    preflight=resource_check(gpu=True);write_json(out/'preflight.json',preflight)
    source=json.loads((args.model_dir/'acquisition_manifest.json').read_text())
    c=json.loads(args.config.read_text())['e1']['molecule']
    assert source['revision']==c['revision'] and source['repo_id']==c['model_id'] and source['status']=='VERIFIED'
    for name,digest in source['files'].items():assert sha256_file(args.model_dir/name)==digest,name
    n=prepared['rows'];progress_path=out/'progress.json';done=0
    assert __import__('shutil').disk_usage(out).free>n*768*4+2*1024**3 or args.resume
    identity={'script_sha256':sha256_file(Path(__file__)),'prepared_sha256':sha256_file(out/'prepared.json'),
        'acquisition_sha256':sha256_file(args.model_dir/'acquisition_manifest.json'),
        'batch_size':args.batch_size,'chunk_size':args.chunk_size,'dtype':'float32','pooling':'first-token','max_length':128}
    progress={'rows_done':0,'rows_total':n,'unique_model_inputs':0,'truncated_rows':0,'max_original_tokens':0,'elapsed_seconds':0.,'identity':identity}
    if progress_path.exists():
        assert args.resume,'Existing prefix: use --resume'
        progress=json.loads(progress_path.read_text());assert progress['identity']==identity and progress['rows_total']==n
        done=progress['rows_done']
    elif args.resume:raise FileNotFoundError('No saved progress to resume')
    connection=sqlite3.connect(out/'input_equivalence.sqlite')
    connection.execute('PRAGMA journal_mode=WAL');connection.execute('PRAGMA synchronous=FULL');connection.execute('PRAGMA cache_size=-65536')
    connection.execute('CREATE TABLE IF NOT EXISTS representations(tokens BLOB PRIMARY KEY,first_row INTEGER NOT NULL) WITHOUT ROWID')
    connection.execute('DELETE FROM representations WHERE first_row>=?',(done,));connection.commit()
    stage=out/'embeddings.staging.npy';target=out/'embeddings.npy'
    array_path=target if done==n and target.exists() else stage
    mode='r+' if array_path.exists() and progress_path.exists() else 'w+'
    array=np.lib.format.open_memmap(array_path,mode=mode,dtype=np.float32,shape=(n,768))
    lengths=np.lib.format.open_memmap(out/'original_token_lengths.npy',mode='r+' if done else 'w+',dtype=np.uint32,shape=(n,))
    aliases=np.lib.format.open_memmap(out/'equivalent_input_rows.npy',mode='r+' if done else 'w+',dtype=np.int32,shape=(n,))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True,use_fast=True)
    assert tokenizer.is_fast
    model,loading=AutoModel.from_pretrained(args.model_dir,local_files_only=True,torch_dtype=torch.float32,output_loading_info=True)
    assert not loading['missing_keys'] and not loading['mismatched_keys'],loading
    assert model.config.hidden_size==768 and model.config.max_position_embeddings==128
    model.requires_grad_(False);model.eval();model.to('cuda')
    write_json(out/'loading.json',{'info':loading,'parameters':sum(p.numel() for p in model.parameters()),'frozen':True,
        'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'torch':torch.__version__,'cuda_device':torch.cuda.get_device_name(0)})
    started=time.time();prior_elapsed=progress['elapsed_seconds'];chunks=0;last_log=started
    with (out/'smiles.txt').open() as reader:
        for _ in range(done):assert reader.readline()
        while done<n:
            count=min(args.chunk_size,n-done);strings=[reader.readline().rstrip('\n') for _ in range(count)];assert all(strings)
            original=tokenizer(strings,padding=False,truncation=False,return_attention_mask=False,verbose=False)['input_ids']
            lens=np.asarray([len(ids) for ids in original],np.uint32);del original
            encoded=tokenizer(strings,padding=False,truncation=True,max_length=128,return_attention_mask=True,verbose=False)
            assert set(encoded)<=set(('input_ids','attention_mask','token_type_ids'))
            assert 'token_type_ids' not in encoded or all(not any(ids) for ids in encoded['token_type_ids'])
            assert all(len(ids)<=128 for ids in encoded['input_ids']) and all(all(mask) for mask in encoded['attention_mask'])
            refs,unique=register_inputs(connection,encoded['input_ids'],done)
            encode_inputs(model,tokenizer,encoded,unique,args.batch_size,'cuda',array,done)
            array[done:done+count]=array[refs];lengths[done:done+count]=lens;aliases[done:done+count]=refs
            array.flush();lengths.flush();aliases.flush();connection.commit()
            done+=count;chunks+=1
            progress.update(rows_done=done,unique_model_inputs=progress['unique_model_inputs']+len(unique),
                truncated_rows=progress['truncated_rows']+int((lens>128).sum()),
                max_original_tokens=max(progress['max_original_tokens'],int(lens.max())),elapsed_seconds=prior_elapsed+time.time()-started,
                peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2,resources=resource_check())
            write_json(progress_path,progress)
            if time.time()-last_log>=20 or done==n:
                print('PROGRESS',json.dumps({key:progress[key] for key in ('rows_done','rows_total','unique_model_inputs','truncated_rows','elapsed_seconds','peak_cuda_mib')}),flush=True)
                last_log=time.time()
            if args.max_chunks and chunks>=args.max_chunks and done<n:
                print('BOUNDED STOP; resumable prefix saved',done,flush=True);return
    array.flush();del array;del lengths;del aliases;connection.execute('PRAGMA wal_checkpoint(TRUNCATE)');connection.close()
    # Full all-row read-integrity and equivalence validation before atomic completion.
    array=np.load(array_path,mmap_mode='r');refs=np.load(out/'equivalent_input_rows.npy',mmap_mode='r')
    for start in range(0,n,4096):
        block=array[start:start+4096];indices=refs[start:start+4096]
        assert np.isfinite(block).all() and np.all(indices<=np.arange(start,start+len(block))) and np.all(indices>=0)
        assert np.array_equal(block,array[indices]),'Equivalent token inputs must have identical cached vectors'
    del array;del refs
    if array_path==stage:stage.replace(target)
    result={'status':'PLUMBING_ONLY' if args.limit else 'COMPLETE','rows':n,'shape':[n,768],'identity':identity,
        'model':source,'inputs':prepared,'progress':progress,'test_evaluated':False,
        'equivalence':'Exact effective token sequences share the first cached float32 vector; candidate identities/order/counts unchanged',
        'normalization':'none; no L2 before projection','precision':'float32 model and cache; TF32 disabled',
        'files':{name:sha256_file(out/name) for name in ('embeddings.npy','global_rows.npy','global_to_cache.npy','smiles.txt','original_token_lengths.npy','equivalent_input_rows.npy')},
        'scope':'Frozen train/val referenced candidate features only; no test spectra/labels or test evaluation'}
    write_json(out/'manifest.json',result);print('CACHE COMPLETE',json.dumps({'rows':n,'status':result['status'],'elapsed_seconds':progress['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
