"""Independent NumPy-only metrics for archived, frozen predictions."""
from __future__ import annotations
import numpy as np
METRICS=('recall@1','recall@5','recall@20','mrr')


def uniform_ties(greater,equal):
    g=np.asarray(greater,dtype=np.int64);e=np.asarray(equal,dtype=np.int64)
    if g.shape!=e.shape or np.any(g<0) or np.any(e<1):
        raise ValueError('Invalid greater/equal tie counts')
    if not g.size:
        return np.empty(g.shape+(4,),dtype=np.float64)
    harmonic=np.r_[0.,np.cumsum(1./np.arange(1,int(np.max(g+e))+1))]
    return np.stack([np.clip((k-g)/e,0.,1.) for k in (1,5,20)]+[(harmonic[g+e]-harmonic[g])/e],axis=-1)


def grouped(values,identities):
    values=np.asarray(values,dtype=np.float64)
    identities=np.asarray(identities)
    if values.ndim!=2 or values.shape!=(len(identities),4) or not len(values):
        raise ValueError('Expected nonempty N-by-4 contributions and N group identities')
    if not np.isfinite(values).all() or np.any(values<0) or np.any(values>1):
        raise ValueError('Contributions must be finite fractions between zero and one')
    unique,inverse,counts=np.unique(identities,return_inverse=True,return_counts=True)
    sums=np.zeros((len(unique),4),dtype=np.float64)
    np.add.at(sums,inverse,values)
    return counts,sums


def summarize(values,identities):
    values=np.asarray(values,dtype=np.float64)
    counts,sums=grouped(values,identities)
    return {'query_weighted':dict(zip(METRICS,(100*values.mean(axis=0)).tolist())),
            'connectivity_balanced':dict(zip(METRICS,(100*(sums/counts[:,None]).mean(axis=0)).tolist()))}


def paired_cluster_interval(after,before,identities,draws=10000,seed=20261017):
    """Whole-connectivity percentile intervals, paired within each draw.

    Groups, rather than spectra, are resampled. Query-weighted estimates retain
    their actual sampled spectrum denominators; balanced estimates weight each
    resampled group equally. Batches are bounded independently of draw count.
    """
    if draws<1:
        raise ValueError('draws must be positive')
    counts,a=grouped(after,identities);other,b=grouped(before,identities)
    if not np.array_equal(counts,other):
        raise ValueError('Group populations differ')
    sums=a-b;means=sums/counts[:,None]
    rng=np.random.default_rng(seed)
    query=np.empty((draws,4));balanced=np.empty_like(query)
    for start in range(0,draws,8):
        ix=rng.integers(0,len(counts),size=(min(8,draws-start),len(counts)))
        query[start:start+len(ix)]=sums[ix].sum(axis=1)/counts[ix].sum(axis=1)[:,None]
        balanced[start:start+len(ix)]=means[ix].mean(axis=1)
    def cells(samples):
        intervals=100*np.quantile(samples,[.025,.975],axis=0)
        return {name:intervals[:,j].tolist() for j,name in enumerate(METRICS)}
    return {'draws':draws,'seed':seed,'resampling_unit':'connectivity',
            'query_weighted_ci95_pp':cells(query),'connectivity_balanced_ci95_pp':cells(balanced)}
