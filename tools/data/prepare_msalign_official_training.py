#!/usr/bin/env python3
"""Audit completed E1 features and build train-only metadata vocabulary; no test."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from glmr.embcos import sha256_file,write_json
from glmr.msalign_external import normalize_adduct
OUT=ROOT/'outputs/msalign_official_e1_20261014'


def main():
    cpath=ROOT/'docs/msalign/configs/external_matched_e0_20261014.json';c=json.loads(cpath.read_text())
    state=json.loads((OUT/'cache_pipeline_state.json').read_text());assert state['status']=='COMPLETE'
    mol=OUT/'moldeberta_cache';m=json.loads((mol/'manifest.json').read_text());assert m['status']=='COMPLETE' and m['rows']==6491031
    assert m['inputs']['inputs']['contract']==sha256_file(cpath)
    assert sha256_file(mol/'manifest.json')==state['moldeberta_manifest']
    for name,digest in m['files'].items():assert sha256_file(mol/name)==digest,name
    array=np.load(mol/'embeddings.npy',mmap_mode='r');assert array.dtype==np.float32 and array.shape==(6491031,768)
    aliases=np.load(mol/'equivalent_input_rows.npy',mmap_mode='r')
    for start in range(0,len(array),4096):
        block=array[start:start+4096];refs=aliases[start:start+4096]
        assert np.isfinite(block).all() and np.all(refs<=np.arange(start,start+len(block))) and np.all(refs>=0)
        assert np.array_equal(block,array[refs])
    records={};digests={};spectra={}
    source=ROOT/'outputs/embcos_official_e2_20261014/cache';rows_manifest=json.loads((source/'manifest.json').read_text())
    lookup=np.load(mol/'global_to_cache.npy',mmap_mode='r')
    for fold,n in [('train',194119),('val',19429)]:
        path=ROOT/c['assets'][fold+'_metadata'];digest=sha256_file(path)
        assert digest==rows_manifest['files'][fold]['metadata_sha256']
        with path.open() as f:records[fold]=[json.loads(line) for line in f if line.strip()]
        assert len(records[fold])==n and all(r['fold']==fold for r in records[fold])
        rp=source/(fold+'.rows.npz');assert sha256_file(rp)==rows_manifest['files'][fold]['rows_sha256']
        with np.load(rp) as z:
            assert np.array_equal(z['identifiers'],np.asarray([r['identifier'] for r in records[fold]]))
            assert np.array_equal(z['smiles'],np.asarray([r['canonical_isomeric_smiles'] for r in records[fold]]))
            assert np.all(lookup[z['candidates'][z['candidates']>=0]]>=0)
        sp=OUT/f'dreams_contrastive/{fold}.summary.json';s=json.loads(sp.read_text())
        assert sha256_file(sp)==state['spectrum_summaries'][fold]
        assert s['representation']=='contrastive' and s['dreams_ckpt_sha256']==c['assets']['datasafe_checkpoint_sha256']
        assert s['metadata_sha256']==digest and s['shape']==[n,1024] and s['parsed_spectrum_rows']==n
        data=OUT/f'dreams_contrastive/{fold}.npy';assert sha256_file(data)==s['cache_sha256']
        x=np.load(data,mmap_mode='r');assert x.shape==(n,1024) and x.dtype==np.float32 and np.isfinite(x).all()
        spectra[fold]=s;digests[fold]=digest
    vocab=sorted({normalize_adduct(r['adduct']) for r in records['train'] if isinstance(r.get('adduct'),str)})
    lookup_adduct={value:i for i,value in enumerate(vocab)};unknown=len(vocab)
    signs=[1 if a.endswith('+') else -1 if a.endswith('-') else 0 for a in vocab]+[0]
    assert all(signs[i]!=0 for i in range(len(vocab)))
    assets=OUT/'assets';assets.mkdir(exist_ok=True);stats={};files={}
    for fold,rows in records.items():
        energy=pd.to_numeric(pd.Series([r.get('collision_energy') for r in rows]),errors='coerce').to_numpy(dtype=np.float32)
        assert not np.isinf(energy).any()
        adduct=np.asarray([lookup_adduct.get(normalize_adduct(r.get('adduct')),unknown) for r in rows],np.int64)
        file=assets/(fold+'.metadata.npz');np.savez(file,collision_energy=energy,adduct=adduct)
        files[fold]=sha256_file(file);stats[fold]={'rows':len(rows),'missing_collision_energy':int(np.isnan(energy).sum()),'unknown_adduct':int((adduct==unknown).sum())}
    manifest={'status':'COMPLETE','contract_sha256':sha256_file(cpath),'n_adducts':len(vocab)+1,'unknown_adduct_index':unknown,'adduct_signs':signs,'adduct_vocabulary':vocab,
        'stats':stats,'metadata_source_sha256':digests,'metadata_arrays_sha256':files,
        'mol_manifest_sha256':sha256_file(mol/'manifest.json'),'pipeline_state_sha256':sha256_file(OUT/'cache_pipeline_state.json'),
        'spectra':spectra,'molecule_cache_files':m['files'],'candidate_rows':rows_manifest['files'],
        'mean_scope':'full official-train spectral inputs only; metadata recipient fixed during replacement',
        'test_evaluated':False,'adduct_vocabulary_scope':'official train only; val unknown retained','energy_parse':'released pd.to_numeric(errors=coerce) -> float32'}
    write_json(assets/'manifest.json',manifest);print('E1 TRAINING ASSETS VERIFIED',json.dumps({'vocabulary':vocab,'stats':stats}),flush=True)


if __name__=='__main__':main()
