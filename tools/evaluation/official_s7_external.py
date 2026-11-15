"""Inference-only E1/E2 assets; uses the frozen numerical evaluators unchanged."""
from __future__ import annotations
import argparse,gc,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.evaluation.official_s7_common import *
from tools.stage1 import run_msalign_official as ms
from tools.stage1 import run_embcos_official as emb

class JointMolecules:
    def __init__(self,base,extension):self.base=base;self.extension=extension;self.shape=(len(base)+len(extension),768)
    def __getitem__(self,indices):
        ix=np.asarray(indices);result=np.empty(ix.shape+(768,),np.float32);mask=ix<len(self.base)
        result[mask]=self.base[ix[mask]];result[~mask]=self.extension[ix[~mask]-len(self.base)]
        return result

class Assets:
    def __init__(self,method,rows,spectra,metadata=None,extension=None,device='cuda'):
        self.method=method;self.device=torch.device(device)
        # 'val' is the legacy evaluator's internal slot, not the scientific split.
        self.rows={'val':rows};self.spectra={'val':torch.from_numpy(np.array(np.load(spectra,mmap_mode='r'))).to(self.device)}
        candidates=rows['permuted_candidates'];valid=candidates>=0
        if method=='e1':
            self.manifest=read(E1/'assets/manifest.json')
            self.input_dimensions=dict(d_ms=1024,d_mol=768,**{k:self.manifest[k] for k in ('n_adducts','unknown_adduct_index','adduct_signs')})
            self.molecules=np.load(E1/'moldeberta_cache/embeddings.npy',mmap_mode='r')
            self.lookup=np.load(E1/'moldeberta_cache/global_to_cache.npy',mmap_mode='r')
            self.alias=np.load(E1/'moldeberta_cache/equivalent_input_rows.npy',mmap_mode='r')
            if extension is not None:
                extension=Path(extension);manifest=read(extension/'manifest.json');assert manifest['status']=='COMPLETE'
                self.lookup=np.load(extension/'global_to_cache.npy',mmap_mode='r')
                refs=np.load(extension/'equivalent_input_rows.npy',mmap_mode='r')
                self.alias=np.r_[self.alias,refs]
                self.molecules=JointMolecules(self.molecules,np.load(extension/'embeddings.npy',mmap_mode='r'))
            assert np.all(self.lookup[candidates[valid]]>=0)
            representatives=self.alias[self.lookup[candidates[valid]]];unique=np.unique(representatives)
            self.eval_representatives=unique
            mapped=np.zeros(candidates.shape,np.int64);mapped[valid]=np.searchsorted(unique,representatives)
            self.metadata={'val':{k:torch.from_numpy(v.copy()).to(self.device) for k,v in load(metadata).items()}}
        else:
            self.packed_cpu=np.load(E2/'cache/morgan4096.packed.npy',mmap_mode='r')
            self.packed=torch.from_numpy(np.asarray(self.packed_cpu)).to(self.device)
            global_rows=np.unique(candidates[valid]);bits=self.packed_cpu[global_rows]
            _,representative,inverse=np.unique(bits.view('V512').reshape(-1),return_index=True,return_inverse=True)
            self.eval_representatives=torch.from_numpy(global_rows[representative].astype(np.int64)).to(self.device)
            lookup=np.full(len(self.packed_cpu),-1,np.int32);lookup[global_rows]=inverse
            mapped=np.zeros(candidates.shape,np.int64);mapped[valid]=lookup[candidates[valid]]
        self.eval_indices=torch.from_numpy(mapped).to(self.device);self.eval_mask=torch.from_numpy(valid).to(self.device)
        self.gt_slots=torch.from_numpy(rows['gt_slots'].astype(np.int64)).to(self.device)
    def get_metadata(self,fold,rows):return {key:value[rows] for key,value in self.metadata[fold].items()}
    def molecules_for_evaluation(self,model):
        return (ms.Assets if self.method=='e1' else emb.Assets).molecules_for_evaluation(self,model)

def infer(method,rows_path,spectrum_path,out,metadata_path=None,extension=None,reference=False):
    start=time.time();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=load(rows_path);assets=Assets(method,rows,spectrum_path,metadata_path,extension)
    config=read(CONFIG);results=[];reports={}
    for seed in SEEDS:
        model=(ms.MSAlign(config['e1'],assets.input_dimensions) if method=='e1' else emb.EmbCos(config['e2'])).to(assets.device)
        parent=E1 if method=='e1' else E2
        checkpoint=parent/f'seed{seed}/checkpoint.pt'
        model.load_state_dict(torch.load(checkpoint,map_location=assets.device,weights_only=False)['model']);model.eval()
        summary,result=(ms.evaluate if method=='e1' else emb.evaluate)(model,assets)
        if reference:compare_npz(result,parent/f'seed{seed}/val_ranks.npz')
        np.savez_compressed(out/f'seed{seed}.npz',**result);results.append(result['expected_metrics'])
        reports[str(seed)]=dict(metrics=summary,checkpoint_sha256=sha(checkpoint),exact_validation_replay=bool(reference))
        del model;gc.collect();torch.cuda.empty_cache()
        print('EXTERNAL',method,seed,'PASS',flush=True)
    values=np.mean(np.stack(results),axis=0)
    np.savez_compressed(out/'seed_mean_per_query.npz',identifiers=rows['identifiers'],connectivities=rows['connectivities'],expected_metrics=values)
    write(out/'completion.json',dict(status='PASS',source_sha256=sha(Path(__file__)),method=method,seeds=reports,elapsed_seconds=time.time()-start,
        split='val' if reference else 'test',official_test_read=not reference,
        inputs={str(p):sha(p) for p in [Path(rows_path),Path(spectrum_path),*([Path(metadata_path)] if metadata_path else [])]},
        files_sha256={p.name:sha(p) for p in out.glob('*.npz')}))
    return values

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--val-replay',action='store_true',required=True);a=p.parse_args()
    install_no_test_guard()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    out=PREP/'external_replay';assert not (out/'completion.json').exists()
    gpu_preflight(out,'external_replay')
    rows=E2/'cache/val.rows.npz';results={}
    for method,spectrum,metadata in [('e1',E1/'dreams_contrastive/val.npy',E1/'assets/val.metadata.npz'),('e2',E2/'cache/val.bins.npy',None)]:
        results[method]=infer(method,rows,spectrum,out/method,metadata_path=metadata,reference=True)
        gc.collect();torch.cuda.empty_cache()
    expected=load(S6/'validation_per_query.npz')
    for method,key in [('e1','msalign_seed_mean'),('e2','embcos_seed_mean')]:assert exact(results[method],expected[key])
    write(out/'completion.json',dict(status='PASS',official_test_read=False,exact_seed_scores_ranks_ties_and_means=True,source_sha256=sha(Path(__file__))))
if __name__=='__main__':main()
