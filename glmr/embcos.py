"""Matched EmbCos architecture and tie-safe metrics.

MLP and model equations port the MIT-licensed MSAlign source at
c2ef425b68749052a17b9cdd5f5f11709322654e (Copyright2026 KrzakalaPaul).
This module performs no dataset loading and has no test evaluation entrypoint.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


class AlignmentMLP(nn.Module):
    def __init__(self,d_in,d_hidden,d_shared,n_hidden_layers,dropout,layernorm=True):
        super().__init__();layers=[];width=d_in
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(width,d_hidden))
            if layernorm:layers.append(nn.LayerNorm(d_hidden))
            layers.extend((nn.GELU(),nn.Dropout(dropout)));width=d_hidden
        layers.append(nn.Linear(width,d_shared));self.layers=nn.Sequential(*layers)
    def forward(self,x):return self.layers(x)


class EmbCos(nn.Module):
    def __init__(self,config: dict):
        super().__init__();c=config['model']
        self.fc_ms=AlignmentMLP(config['spectrum']['dimension'],c['d_hidden'],c['d_shared'],c['n_hidden_layers_ms'],c['dropout'],c['layernorm'])
        self.fc_mol=AlignmentMLP(config['molecule']['dimension'],c['d_hidden'],c['d_shared'],c['n_hidden_layers_mol'],c['dropout'],c['layernorm'])
        self.log_epsilon=nn.Parameter(torch.log(torch.tensor(c['temperature_init'])))
    @property
    def temperature(self):return self.log_epsilon.exp()
    def encode_ms(self,x):return F.normalize(self.fc_ms(x),p=2,dim=-1)
    def encode_mol(self,x):return F.normalize(self.fc_mol(x),p=2,dim=-1)
    def forward(self,x,candidates,mask):
        q=self.encode_ms(x);m=self.encode_mol(candidates)
        return torch.einsum('bd,bkd->bk',q,m).div(self.temperature).masked_fill(~mask,float('-inf'))


def cosine_schedule(warmup: int, ceiling: int):
    def schedule(step):
        if step<warmup:return step/max(1,warmup)
        progress=(step-warmup)/max(1,ceiling-warmup)
        return .5*(1+np.cos(np.pi*min(progress,1.)))
    return schedule


def bin_spectrum(peaks: np.ndarray) -> tuple[np.ndarray,bool]:
    """Released top100/float32/bin/log transform with the frozen empty guard."""
    peaks=np.asarray(peaks,dtype=np.float64).reshape(-1,2)
    if not np.isfinite(peaks).all() or np.any(peaks<0):raise ValueError('Invalid observed peaks')
    order=np.argsort(peaks[:,1])[::-1][:100]
    padded=np.zeros((100,2),np.float32);padded[:len(order)]=peaks[order]
    mz,intensity=padded.T
    indices=np.floor(mz/.1).astype(int);valid=mz<=1005.
    indices=np.clip(indices[valid],0,10049)
    bins=np.zeros(10050);np.add.at(bins,indices,intensity[valid])
    maximum=bins.max()
    if maximum<=0:return np.zeros(10050,np.float32),True
    return (np.log10(bins/maximum*999+1)/3).astype(np.float32),False


def decode_packed(packed: torch.Tensor) -> torch.Tensor:
    shifts=torch.arange(8,dtype=torch.uint8,device=packed.device)
    return ((packed.unsqueeze(-1)>>shifts)&1).flatten(-2).float()


def sample_candidates(candidates: list[int], exclusion: np.ndarray, k: int, generator: torch.Generator):
    if not candidates:raise ValueError('Missing training candidates')
    negatives=torch.tensor([c for c in candidates[1:] if not exclusion[c]],dtype=torch.long)
    negative_indices=torch.randperm(len(negatives),generator=generator)[:k-1]
    chosen=torch.cat((torch.tensor([candidates[0]],dtype=torch.long),negatives[negative_indices]))
    mask=torch.zeros(k,dtype=torch.bool);mask[:len(chosen)]=True
    result=torch.zeros(k,dtype=torch.long);result[:len(chosen)]=chosen
    return result,mask


def uniform_tie_metrics(greater: np.ndarray,equal: np.ndarray) -> np.ndarray:
    greater=np.asarray(greater,np.int64);equal=np.asarray(equal,np.int64)
    if np.any(greater<0) or np.any(equal<1):raise ValueError('Invalid tie counts')
    values=[np.clip((k-greater)/equal,0.,1.) for k in (1,5,20)]
    harmonic=np.concatenate(([0.],np.cumsum(1./np.arange(1,int((greater+equal).max())+1))))
    values.append((harmonic[greater+equal]-harmonic[greater])/equal)
    return np.stack(values,axis=-1)


def summarize_metrics(vectors: np.ndarray,connectivities: np.ndarray) -> dict:
    _,groups=np.unique(connectivities,return_inverse=True)
    counts=np.bincount(groups);balanced=np.stack([np.bincount(groups,weights=vectors[:,j])/counts for j in range(4)],axis=1)
    names=('recall@1','recall@5','recall@20','mrr')
    return {'valid_queries':len(vectors),'connectivity_groups':len(counts),
        **{name:float(vectors[:,j].mean()*100) for j,name in enumerate(names)},
        'connectivity_balanced':{name:float(balanced[:,j].mean()*100) for j,name in enumerate(names)}}
