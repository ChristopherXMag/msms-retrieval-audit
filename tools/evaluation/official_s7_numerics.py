"""Preserve the original pair20 physical width for naturally short candidate pools."""
import numpy as np
import torch
from tools.reranking import audit_official_d3_full_system as old

def pair_scores(models,spec,mol,seed_scores,base,extent=None,physical_width=20):
    k=base.shape[1];assert 0<k<=physical_width
    if k==physical_width:return old.pair_score(models,spec,mol,seed_scores,base,extent)
    device=next(models[0].parameters()).device;n=len(base);target=extent or 512;assert n<=target
    s=torch.as_tensor(spec,dtype=torch.float32,device=device);m=torch.as_tensor(mol,dtype=torch.float32,device=device)
    b=torch.as_tensor(base,device=device);ss=torch.as_tensor(seed_scores,device=device)
    def padq(t):return torch.cat([t,t[-1:].expand((target-n,)+t.shape[1:])],0) if n<target else t
    s,m,b,ss=map(padq,(s,m,b,ss))
    # The original forward_masked computes relative features over real candidates,
    # then executes the MLP at fixed physical width20 and masks unused outputs.
    features=old.top.pair.score_features(ss,b)
    m=torch.cat([m,m[:,-1:].expand(-1,physical_width-k,-1)],1)
    b=torch.cat([b,b[:,-1:].expand(-1,physical_width-k)],1)
    features=torch.cat([features,torch.zeros((target,physical_width-k,11),device=device)],1)
    with torch.inference_mode():result=np.mean(np.stack([model(s,m,features,b).cpu().numpy() for model in models]),axis=0)
    return result[:n,:k]


def scalar_features(base,coverage):
    """Original ten ordered scalar columns, without rebuilding chemistry details."""
    d2=old.top.d2;scalar,_=d2.trees.score_features(base,d2.add_coverage(base,coverage));k=np.shape(base)[1]
    if k!=5:
        for col,values in ((3,base),(8,d2.add_coverage(base,coverage))):
            values=np.asarray(values,np.float64);exponent=np.exp(values-values.max(1,keepdims=True))
            scalar[:,:,col]=(exponent/(k*d2.trees.symmetric_mean(exponent))).astype(np.float32)
    return scalar
