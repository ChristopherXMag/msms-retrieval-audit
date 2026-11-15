#!/usr/bin/env python3
"""Train-only inventory/cache for local spectral-relation supervision; no GPU/test."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from rdkit import Chem,rdBase
from rdkit.Chem import rdMolDescriptors
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from glmr.embcos import sha256_file,write_json
from tools.data.fetch_msgym_v15_raw_train import REVISION
SOURCE=ROOT/f'data/source_massspecgym_v15_{REVISION}/acquisition_manifest.json'
TRAIN_ROWS=ROOT/'outputs/embcos_official_e2_20261014/cache/train.rows.npz'
META=ROOT/'outputs/msalign_stage1_full_v15/train_full.metadata.jsonl'


def resources():
    current=int(Path('/sys/fs/cgroup/memory.current').read_text())
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    assert current<52*1024**3 and available>64*1024**3,(current,available)
    return {'cgroup_gib':current/1024**3,'host_available_gib':available/1024**3,
        'process_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}


def freeze(out):
    assert not out.exists()
    source=json.loads(SOURCE.read_text());train=ROOT/source['train_file']
    paths=[Path(__file__),SOURCE,train,TRAIN_ROWS,META,ROOT/'glmr/embcos.py']
    contract={'status':'FROZEN_BEFORE_INVENTORY','date':'2026-10-15','seed':20261015,
        'source_revision':REVISION,'train_rows':194119,'rdkit':'2025.09.6',
        'scope':'official train only; no val/test records, ranking outcomes or model inference',
        'molecule_identity':'recomputed RDKit canonical-isomeric SMILES; cross-molecule pairs require different recomputed connectivities',
        'strict_condition':['formula','adduct','instrument_type','simulation_challenge','exact finite published numeric CE'],
        'CE_sensitivity':'fixed 5-unit bins, descriptive only; missing/unknown CE never a known-condition match',
        'limitations':['no fine library/acquisition identifiers','CE units and instrument model not published','nonidentical spectra are replicate proxies, not proven independent acquisitions'],
        'peak_filter':{'min_mz':0,'precursor_exclusion':'retain mz < precursor_mz - 1.5 Da','top_peaks':100,'intensity_transform':'sqrt then L2','aux_min_retained_peaks':5},
        'dedup':'hash sorted float32 mz rounded6 and sqrt/L2 intensity rounded6; raw published peak-string hash also retained',
        'data_gate':{'min_distinct_connectivities_in_repeat_supported_cross_pairs':500,'min_repeat_supported_cross_pairs':1000,
                     'repeat_support':'at least two distinct processed spectra in each exact canonical molecule / strict condition'},
        'gate_scope':'sufficient material for a bounded relational pilot only; teacher repeatability and held-out ranking benefit remain to be checked',
        'test_evaluated':False,'training_performed':False,
        'input_source_sha256':{str(p.relative_to(ROOT)):sha256_file(p) for p in paths}}
    assert contract['input_source_sha256'][str(train.relative_to(ROOT))]==source['train_file_sha256']
    out.mkdir(parents=True);write_json(out/'inventory_contract.json',contract)
    print('FROZEN',sha256_file(out/'inventory_contract.json'),flush=True)


def processed_peaks(mz_text,int_text,precursor):
    mz=np.fromstring(mz_text,sep=',',dtype=np.float64);intensity=np.fromstring(int_text,sep=',',dtype=np.float64)
    assert len(mz)==len(intensity)
    valid=np.isfinite(mz)&np.isfinite(intensity)&(mz>0)&(intensity>0)&(mz<precursor-1.5)
    mz=mz[valid];intensity=intensity[valid]
    keep=np.lexsort((mz,-intensity))[:100];mz=mz[keep];intensity=intensity[keep]
    order=np.argsort(mz,kind='stable');mz=mz[order];v=np.sqrt(intensity[order])
    if len(v):v/=np.linalg.norm(v)
    mz=mz.astype(np.float32);v=v.astype(np.float32)
    digest=hashlib.sha256(np.round(mz,6).tobytes()+np.round(v,6).tobytes()).hexdigest()
    return mz,v,digest


def group_statistics(groups,records):
    count=Counter();edges=set();repeat_edges=set();all_nodes=set();repeat_nodes=set();group_rows=[]
    for key,molecules in sorted(groups.items()):
        # Within a condition, retain an exact-isomer identity; collapse connectivity
        # only for counting cross-connectivity pairs, never for training GT rows.
        representatives={};reliable={}
        for smi,indices in sorted(molecules.items()):
            conn=records[indices[0]]['connectivity']
            valid=[i for i in indices if records[i]['n_retained_peaks']>=5]
            unique={records[i]['processed_spectrum_sha256'] for i in valid}
            count['molecule_condition_cells']+=1
            count['cells_two_raw_rows']+=len(indices)>=2
            count['cells_two_distinct_processed_spectra']+=len(unique)>=2
            count['cells_four_distinct_processed_spectra']+=len(unique)>=4
            if valid and conn not in representatives:representatives[conn]=smi
            if len(unique)>=2 and conn not in reliable:reliable[conn]=smi
        n=len(representatives);nr=len(reliable)
        count['condition_groups']+=1;count['groups_two_connectivities']+=n>=2;count['groups_three_connectivities']+=n>=3
        count['groups_two_repeat_supported_connectivities']+=nr>=2;count['groups_three_repeat_supported_connectivities']+=nr>=3
        count['cross_connectivity_pair_condition_instances']+=n*(n-1)//2
        count['repeat_supported_pair_condition_instances']+=nr*(nr-1)//2
        count['repeat_supported_triplet_condition_instances']+=nr*(nr-1)*(nr-2)//6
        if n>=2:
            for a,b in itertools.combinations(sorted(representatives),2):edges.add((a,b))
            all_nodes.update(representatives)
        if nr>=2:
            for a,b in itertools.combinations(sorted(reliable),2):repeat_edges.add((a,b))
            repeat_nodes.update(reliable)
        if n>=2:
            group_rows.append({'condition':list(key),'molecules':[
                {'canonical_isomeric_smiles':smi,'connectivity':conn,'rows':molecules[smi],
                 'distinct_processed_spectra':len({records[i]['processed_spectrum_sha256'] for i in molecules[smi] if records[i]['n_retained_peaks']>=5})}
                for conn,smi in sorted(representatives.items())],
                'repeat_supported_molecules':[{'canonical_isomeric_smiles':smi,'connectivity':conn,'rows':molecules[smi]} for conn,smi in sorted(reliable.items())]})
    result={**dict(count),'unique_cross_connectivity_pairs':len(edges),'connectivities_in_cross_pairs':len(all_nodes),
        'unique_repeat_supported_pairs':len(repeat_edges),'connectivities_in_repeat_supported_pairs':len(repeat_nodes)}
    return result,group_rows


def run(out):
    started=time.time();c=json.loads((out/'inventory_contract.json').read_text())
    for p,h in c['input_source_sha256'].items():assert sha256_file(ROOT/p)==h,p
    assert rdBase.rdkitVersion==c['rdkit']
    assert not (out/'inventory.json').exists()
    source=json.loads(SOURCE.read_text());n=c['train_rows'];resources()
    with np.load(TRAIN_ROWS,allow_pickle=False) as z:
        ids=z['identifiers'];smiles=z['smiles'];connectivities=z['connectivities']
    assert len(ids)==n
    stage_mz=out/'train.local_mz.staging.npy';stage_int=out/'train.local_intensity.staging.npy'
    mz_array=np.lib.format.open_memmap(stage_mz,mode='w+',shape=(n,100),dtype=np.float32)
    int_array=np.lib.format.open_memmap(stage_int,mode='w+',shape=(n,100),dtype=np.float32)
    mz_array[:]=0;int_array[:]=0
    records=[];memo={};strict=defaultdict(lambda:defaultdict(list));loose=defaultdict(lambda:defaultdict(list));missing=Counter();overall=Counter()
    csv.field_size_limit(100_000_000)
    with (ROOT/source['train_file']).open(newline='') as f:
        reader=csv.DictReader(f,delimiter='\t')
        for i,row in enumerate(reader):
            assert row['fold']=='train' and row['identifier']==ids[i]
            smi=row['smiles']
            if smi not in memo:
                mol=Chem.MolFromSmiles(smi);assert mol is not None
                memo[smi]=(Chem.MolToSmiles(mol,isomericSmiles=True),Chem.MolToInchiKey(mol).split('-')[0],rdMolDescriptors.CalcMolFormula(mol))
            canonical,conn,formula=memo[smi]
            assert canonical==smiles[i] and conn==connectivities[i]
            assert formula==row['formula'],(formula,row['formula'])
            ce_raw=row['collision_energy']
            try:ce=float(ce_raw)
            except ValueError:ce=float('nan')
            ce=None if not np.isfinite(ce) else ce
            instrument=row['instrument_type'].strip()
            if instrument.lower() in ('','nan','none','unknown','n/a'):instrument=None
            precursor=float(row['precursor_mz'])
            mz,ints,peak_hash=processed_peaks(row['mzs'],row['intensities'],precursor)
            mz_array[i,:len(mz)]=mz;int_array[i,:len(ints)]=ints
            record={'row':i,'identifier':row['identifier'],'canonical_isomeric_smiles':canonical,'connectivity':conn,
                'formula':formula,'adduct':row['adduct'],'instrument_type':instrument,'collision_energy':ce,'collision_energy_raw':ce_raw,
                'simulation_challenge':row['simulation_challenge'],'precursor_mz':precursor,
                'n_retained_peaks':len(mz),'processed_spectrum_sha256':peak_hash,
                'raw_peak_strings_sha256':hashlib.sha256((row['mzs']+'|'+row['intensities']).encode()).hexdigest()}
            records.append(record)
            overall['rows']+=1;overall['aux_at_least_five_nonprecursor_peaks']+=len(mz)>=5
            overall['missing_CE']+=ce is None;overall['missing_instrument']+=instrument is None
            missing[(row['adduct'],str(instrument),str(ce is None),row['simulation_challenge'])]+=1
            if ce is not None and instrument is not None:
                base=(formula,row['adduct'],instrument,row['simulation_challenge'])
                strict[(*base,ce)][canonical].append(i)
                loose[(*base,float(np.floor(ce/5)*5))][canonical].append(i)
            if (i+1)%20000==0:print('INVENTORY',i+1,resources(),flush=True)
    assert len(records)==n
    mz_array.flush();int_array.flush();del mz_array,int_array
    for stage,final in [(stage_mz,out/'train.local_mz.npy'),(stage_int,out/'train.local_intensity.npy')]:
        arr=np.load(stage,mmap_mode='r');assert arr.shape==(n,100)
        for start in range(0,n,4096):assert np.isfinite(arr[start:start+4096]).all()
        del arr;stage.replace(final)
    exact,groups=group_statistics(strict,records);coarse,_=group_statistics(loose,records)
    with (out/'train.rows.jsonl').open('w') as f:
        for record in records:f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    with (out/'strict.groups.jsonl').open('w') as f:
        for group in groups:f.write(json.dumps(group,ensure_ascii=False,allow_nan=False)+'\n')
    gate=c['data_gate'];enough=exact['connectivities_in_repeat_supported_pairs']>=gate['min_distinct_connectivities_in_repeat_supported_cross_pairs'] and exact['unique_repeat_supported_pairs']>=gate['min_repeat_supported_cross_pairs']
    result={'status':'COMPLETE','date':'2026-10-15','data_gate':'PASS_FOR_TEACHER_REPEATABILITY_CHECK' if enough else 'INSUFFICIENT_REPEAT_SUPPORTED_PAIRS',
        'overall':dict(overall),'unique_canonical_molecules':len(set(smiles)),'unique_connectivities':len(set(connectivities)),
        'strict_known_condition':exact,'five_unit_CE_bin_sensitivity':coarse,
        'input_cohorts':[{'adduct':k[0],'instrument':k[1],'CE_missing':k[2],'simulation_challenge':k[3],'rows':v} for k,v in sorted(missing.items())],
        'limitations':c['limitations'],'peak_preprocessing':c['peak_filter'],'elapsed_seconds':time.time()-started,'resources':resources(),
        'contract_sha256':sha256_file(out/'inventory_contract.json'),'source_sha256':source['train_file_sha256'],
        'test_evaluated':False,'validation_read':False,'training_performed':False}
    write_json(out/'inventory.json',result)
    paths=['inventory_contract.json','inventory.json','train.rows.jsonl','strict.groups.jsonl','train.local_mz.npy','train.local_intensity.npy']
    write_json(out/'inventory_complete.json',{'status':'COMPLETE','files_sha256':{p:sha256_file(out/p) for p in paths},'test_evaluated':False,'validation_read':False})
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('freeze','run'))
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/local_spectral_geometry_official_20261015')
    args=parser.parse_args()
    if args.action=='freeze':freeze(args.out)
    else:run(args.out)


if __name__=='__main__':main()
