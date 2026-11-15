#!/usr/bin/env python3
"""Build matched EmbCos spectrum and packed Morgan caches; only train/val queries."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from rdkit import Chem,rdBase
from rdkit.Chem import rdFingerprintGenerator
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from glmr.embcos import sha256_file,write_json,bin_spectrum
from glmr.candidate_indices import CandidateIndexStore,CandidateNegativeExclusion
GENERATOR=None


def init_worker():
    global GENERATOR
    GENERATOR=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=4096,includeChirality=False)


def fingerprint_chunk(job):
    start,strings=job;packed=np.empty((len(strings),512),np.uint8)
    for i,s in enumerate(strings):
        m=Chem.MolFromSmiles(s)
        if m is None:raise ValueError(f'Invalid candidate SMILES at global row {start+i}')
        packed[i]=np.packbits(GENERATOR.GetFingerprintAsNumPy(m),bitorder='little')
    return start,packed


def read_records(path,fold,limit):
    with path.open() as f:rows=[json.loads(line) for line in f if line.strip()]
    assert all(r['fold']==fold for r in rows)
    if limit:rows=rows[:limit]
    return rows


def mgf_spectra(path):
    title=None;peaks=[]
    with path.open() as f:
        for line in f:
            s=line.strip()
            if s=='BEGIN IONS':title=None;peaks=[]
            elif s=='END IONS':
                if title is None:raise ValueError('MGF TITLE missing')
                yield title,np.asarray(peaks,np.float64).reshape(-1,2)
            elif s.startswith('TITLE='):title=s.partition('=')[2]
            elif s and '=' not in s:
                pieces=s.split()
                if len(pieces)==2:peaks.append((float(pieces[0]),float(pieces[1])))


def spectra_cache(records,mgf,path):
    ids={r['identifier']:i for i,r in enumerate(records)};seen=set();temporary=path.with_suffix('.staging.npy')
    array=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(len(records),10050))
    empty=0;capped=0;maximum_input_peaks=0
    for identifier,peaks in mgf_spectra(mgf):
        if identifier not in ids:continue
        if identifier in seen:raise ValueError('Duplicate MGF ID')
        vector,is_empty=bin_spectrum(peaks);array[ids[identifier]]=vector;seen.add(identifier)
        empty+=int(is_empty);capped+=len(peaks)>100;maximum_input_peaks=max(maximum_input_peaks,len(peaks))
        if len(seen)%20000==0:print('Binned',path.name,len(seen),'/',len(records),flush=True)
        if len(seen)==len(records):break
    assert seen==set(ids),'MGF coverage mismatch';array.flush();del array
    check=np.load(temporary,mmap_mode='r')
    for start in range(0,len(check),4096):
        block=check[start:start+4096];assert np.isfinite(block).all() and np.all((block>=0)&(block<=1.000001))
    del check;temporary.replace(path)
    return {'rows':len(records),'empty_guard_rows':empty,'rows_capped_to100':int(capped),'max_input_peaks':maximum_input_peaks,'sha256':sha256_file(path)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'docs/msalign/configs/external_matched_e0_20261014.json')
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=(0,1,2),default=0)
    p.add_argument('--molecule-limit',type=int,default=0,help='Plumbing-only first rows; never evidence')
    p.add_argument('--train-limit',type=int,default=0);p.add_argument('--val-limit',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();assert rdBase.rdkitVersion=='2025.09.6'
    c=json.loads(args.config.read_text());out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Cache already complete')
    started=time.time();paths=c['assets'];store=CandidateIndexStore(ROOT/paths['candidate_manifest'])
    exclusion=CandidateNegativeExclusion(ROOT/paths['negative_exclusion']);exclusion.validate_candidate_store(store)
    records={fold:read_records(ROOT/paths[fold+'_metadata'],fold,getattr(args,fold+'_limit')) for fold in ('train','val')}
    spectra={};files={}
    for fold,rows in records.items():
        path=out/(fold+'.bins.npy')
        if path.exists() and args.resume:
            assert np.load(path,mmap_mode='r').shape==(len(rows),10050)
            spectra[fold]=json.loads((out/(fold+'.spectra.json')).read_text());assert spectra[fold]['sha256']==sha256_file(path)
        else:
            spectra[fold]=spectra_cache(rows,ROOT/paths[fold+'_mgf'],path);write_json(out/(fold+'.spectra.json'),spectra[fold])
        # Original candidate rows retained for training. A separate physical eval permutation is persisted.
        candidates=np.full((len(rows),256),-1,np.int32);lengths=np.empty(len(rows),np.int32)
        gt_slots=np.zeros(len(rows),np.int32);permuted=candidates.copy();rng=np.random.default_rng(20261014)
        ids=[];smiles=[];groups=[];memo={}
        for i,r in enumerate(rows):
            values=store.get(r['identifier'],256);assert values and len(values)==len(set(values))
            lengths[i]=len(values);candidates[i,:len(values)]=values
            order=rng.permutation(len(values));permuted[i,:len(values)]=np.asarray(values)[order];gt_slots[i]=int(np.flatnonzero(order==0)[0])
            s=r['canonical_isomeric_smiles']
            if s not in memo:memo[s]=Chem.MolToInchiKey(Chem.MolFromSmiles(s)).split('-')[0]
            ids.append(r['identifier']);smiles.append(s);groups.append(memo[s])
        np.savez(out/(fold+'.rows.npz'),candidates=candidates,permuted_candidates=permuted,lengths=lengths,gt_slots=gt_slots,
            identifiers=np.asarray(ids),smiles=np.asarray(smiles),connectivities=np.asarray(groups))
        files[fold]={'metadata_sha256':sha256_file(ROOT/paths[fold+'_metadata']),'mgf_sha256':sha256_file(ROOT/paths[fold+'_mgf']),
            'rows_sha256':sha256_file(out/(fold+'.rows.npz'))}
    n=store.molecule_rows
    if args.molecule_limit:n=min(n,args.molecule_limit)
    partial=out/'morgan4096.staging.npy';progress=out/'morgan.progress.json';done=0
    if args.resume and progress.exists():
        x=json.loads(progress.read_text());done=int(x['rows_done']);assert x['total_rows']==n
        packed=np.lib.format.open_memmap(partial,mode='r+');assert packed.shape==(n,512)
    else:
        assert not partial.exists();packed=np.lib.format.open_memmap(partial,mode='w+',dtype=np.uint8,shape=(n,512))
    def jobs():
        strings=[];start=0
        with (ROOT/paths['smiles']).open() as f:
            for i,line in enumerate(f):
                if i<done:continue
                if i>=n:break
                if not strings:start=i
                strings.append(line.rstrip('\n'))
                if len(strings)==1024:yield start,strings;strings=[]
            if strings:yield start,strings
    def accept(result):
        nonlocal done
        start,data=result;assert start==done;packed[start:start+len(data)]=data;done+=len(data)
        packed.flush();write_json(progress,{'rows_done':done,'total_rows':n,'elapsed_seconds':time.time()-started})
        if done%32768==0 or done==n:print('Morgan',done,'/',n,'elapsed',round(time.time()-started,1),flush=True)
    init_worker()
    if args.workers:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
            iterator=iter(jobs());pending=[]
            for _ in range(2*args.workers):
                job=next(iterator,None)
                if job is not None:pending.append(pool.submit(fingerprint_chunk,job))
            while pending:
                accept(pending.pop(0).result());job=next(iterator,None)
                if job is not None:pending.append(pool.submit(fingerprint_chunk,job))
    else:
        for job in jobs():accept(fingerprint_chunk(job))
    assert done==n;packed.flush();del packed
    check=np.load(partial,mmap_mode='r');assert check.shape==(n,512) and check.dtype==np.uint8
    # Read the entire staging file to verify clean serialization before move.
    fingerprint_sha=sha256_file(partial);del check;target=out/'morgan4096.packed.npy';partial.replace(target)
    assert sha256_file(ROOT/paths['smiles'])==store.molecule_smiles_sha256
    np.save(out/'negative_exclusion.npy',np.asarray(exclusion.indices,dtype=np.int64))
    manifest={'format':'glmr_embcos_cache_v1','status':'PLUMBING_ONLY' if args.train_limit or args.val_limit or args.molecule_limit else 'COMPLETE',
        'contract_sha256':sha256_file(args.config),'source_commit':c['source']['commit'],'rdkit':rdBase.rdkitVersion,'molecule_rows':n,
        'molecule_smiles_sha256':store.molecule_smiles_sha256,'fingerprint':'binary_morgan_r2_4096_chirality_false','bitorder':'little',
        'packed_sha256':fingerprint_sha,'spectra':spectra,'files':files,'negative_exclusion_sha256':sha256_file(out/'negative_exclusion.npy'),
        'candidate_manifest_sha256':sha256_file(ROOT/paths['candidate_manifest']),'negative_manifest_sha256':sha256_file(ROOT/paths['negative_exclusion']),
        'scripts':{'cache':sha256_file(Path(__file__)),'module':sha256_file(ROOT/'glmr/embcos.py')},'workers':args.workers,
        'elapsed_seconds':time.time()-started,'scope':'train/val query spectra only; frozen unlabeled full molecule universe and inherited identity-only denylist reused; no test dataset created',
        'test_evaluated':False,'empty_guard':'zero vector; rows counted','source_peak_precision':'frozen split MGF, not substituted by later downloaded TSV'}
    write_json(out/'manifest.json',manifest);print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
