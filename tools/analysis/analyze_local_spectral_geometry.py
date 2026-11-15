#!/usr/bin/env python3
"""Frozen CPU-only descriptive geometry on existing official-train teacher edges."""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from numba import njit
from scipy.stats import spearmanr
import torch
from rdkit import Chem, rdBase
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from glmr.embcos import sha256_file, write_json
from glmr.msalign_external import MLP
from glmr.sparse_features import SparseFeatureMatrix
from tools.data.build_msalign_sparse_fingerprint_cache import init_worker, fingerprint_chunk, generator_metadata
from tools.data.inventory_official_local_spectral_pairs import resources, TRAIN_ROWS

BASE = ROOT / 'outputs/local_spectral_geometry_official_20261015'
E1 = ROOT / 'outputs/msalign_official_e1_20261014'
SPARSE = ROOT / 'outputs/stage1_submission_0926/assets/full_sparse_direct/manifest.json'
CONFIG = ROOT / 'docs/msalign/configs/external_matched_e0_20261014.json'
BLOCKS = ('count_morgan', 'count_atom_pair', 'rdkit_path_count', 'maccs')
SPACES = ('morgan_r2', 'sccf_raw', 'moldeberta_raw', 'e1_seed20260621', 'e1_seed20260622', 'e1_seed20260623')
SEEDS = (20260621, 20260622, 20260623)
BOOT = 5000
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / 'outputs/.numba_cache'))


def read_jsonl(p):
    return [json.loads(s) for s in p.open()]


def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


@njit(cache=True)
def curve_readouts(x, y, w, order, qpoint=.1, qend=.25):
    """Fractional distance ties; exact integral of prefix event rate, normalized by qend."""
    total = w.sum()
    if total <= 0:
        return np.nan, np.nan
    lp, le = total * qpoint, total * qend
    used = 0.; events = 0.; area = 0.; point = np.nan
    k = 0
    while k < len(order):
        end = k + 1
        while end < len(order) and x[order[end]] == x[order[k]]:
            end += 1
        mass = 0.; ev = 0.
        for z in range(k, end):
            i = order[z]; mass += w[i]; ev += w[i] * y[i]
        if mass > 0:
            rate = ev / mass
            stop = used + mass
            if np.isnan(point) and lp <= stop:
                point = (events + rate * (lp - used)) / lp
            if used < le:
                hi = min(stop, le)
                area += rate * (hi - used)
                if used > 0:
                    area += (events - rate * used) * np.log(hi / used)
            used = stop; events += ev
        k = end
    return point, area / le


@njit(cache=True)
def midranks(x, w, order):
    ranks = np.zeros(len(x)); cumulative = 0.; k = 0
    while k < len(order):
        end = k + 1
        while end < len(order) and x[order[end]] == x[order[k]]:
            end += 1
        mass = 0.
        for z in range(k, end): mass += w[order[z]]
        rank = cumulative + .5 * mass
        for z in range(k, end): ranks[order[z]] = rank
        cumulative += mass; k = end
    return ranks


@njit(cache=True)
def weighted_spearman(x, y, w, ox, oy):
    total = w.sum()
    if total <= 0: return np.nan
    rx = midranks(x, w, ox); ry = midranks(y, w, oy)
    mx = (w * rx).sum() / total; my = (w * ry).sum() / total
    vx = (w * (rx-mx)**2).sum(); vy = (w * (ry-my)**2).sum()
    if vx <= 0 or vy <= 0: return np.nan
    return (w * (rx-mx) * (ry-my)).sum() / np.sqrt(vx * vy)


def semantic_checks():
    x = np.array([0., 0., 1., 2.]); y = np.array([1., 0., 0., 1.]); w = np.ones(4)
    order = np.argsort(x, kind='stable')
    a = curve_readouts(x, y, w, order)
    assert np.allclose(a, [.5, .5]), a
    perm = np.array([3, 1, 0, 2]); xp=x[perm]; yp=y[perm]
    assert np.allclose(curve_readouts(xp,yp,w,np.argsort(xp,kind='stable')),a)
    assert np.allclose(curve_readouts(7*x,y,w,order),a)
    xx=np.repeat(x,2); yy=np.repeat(y,2); ww=np.full(8,.5)
    assert np.allclose(curve_readouts(xx,yy,ww,np.argsort(xx,kind='stable')),a)
    rng=np.random.default_rng(11)
    x=rng.integers(0,7,30).astype(float); y=rng.integers(0,9,30).astype(float)
    ox=np.argsort(x,kind='stable'); oy=np.argsort(y,kind='stable')
    assert abs(weighted_spearman(x,y,np.ones(30),ox,oy)-spearmanr(x,y).statistic)<1e-12
    w=rng.integers(1,5,30).astype(float)
    assert abs(weighted_spearman(x,y,w,ox,oy)-spearmanr(np.repeat(x,w.astype(int)),np.repeat(y,w.astype(int))).statistic)<1e-12
    # Check exact integration against dense numerical quadrature away from q=0.
    x=np.arange(5,dtype=float); y=np.array([1.,0.,1.,0.,0.]); w=np.array([.1,.2,.3,.4,1.])
    _,auc=curve_readouts(x,y,w,np.arange(5))
    qs=(np.arange(20000)+.5)/20000*.25
    rates=np.array([curve_readouts(x,y,w,np.arange(5),q,.25)[0] for q in qs])
    assert abs(auc-rates.mean())<1e-7,(auc,rates.mean())
    return {'status':'PASS','checks':['fractional ties','row permutation','positive distance rescaling','duplicate-condition mass invariance','weighted rank vs scipy','integer-weight rank vs explicit replication','analytic curve integral vs numerical quadrature']}


def freeze(base):
    out=base/'geometry'
    assert not out.exists(), 'Refuse to overwrite an existing frozen geometry analysis'
    checks=semantic_checks()
    inputs=[base/'teacher/complete.json',base/'teacher/pairs.jsonl',base/'teacher/triplets.jsonl',
            base/'inventory_complete.json',base/'integrity_and_sensitivity.json',base/'teacher/contract.json',
            TRAIN_ROWS,E1/'moldeberta_cache/manifest.json',E1/'moldeberta_cache/global_to_cache.npy',
            SPARSE,CONFIG,ROOT/'glmr/msalign_external.py',ROOT/'glmr/sparse_features.py',
            ROOT/'tools/data/build_msalign_sparse_fingerprint_cache.py',ROOT/'glmr/embcos.py',
            ROOT/'tools/data/inventory_official_local_spectral_pairs.py',Path(__file__)]
    inputs += [E1/f'seed{s}/checkpoint.pt' for s in SEEDS]
    teacher_contract=json.loads((base/'teacher/contract.json').read_text())
    # Existing complete full-cache digest is checked once at freeze; extraction rechecks selected-row bytes.
    fp=E1/'moldeberta_cache/embeddings.npy'
    assert sha256_file(fp)==teacher_contract['input_source_sha256'][str(fp.relative_to(ROOT))]
    pair=read_jsonl(base/'teacher/pairs.jsonl');trip=read_jsonl(base/'teacher/triplets.jsonl')
    assert len(pair)==2365 and len({(p['a'],p['b']) for p in pair})==926
    g0=np.array([t['gap_view0'] for t in trip]);g1=np.array([t['gap_view1'] for t in trip])
    main=(np.abs(g0)>=.05)&(np.sign(g0)==np.sign(g1))
    assert main.sum()==1849
    c={'status':'FROZEN_BEFORE_FOUR_SPACE_GEOMETRY_OUTCOMES','date':'2026-10-15','bootstrap_seed':20261015,
       'scope':'Existing official-TRAIN teacher only; CPU, no optimizer/model fitting or official val/test query/cache/rank reads',
       'spaces':list(SPACES),'main_contrast':'raw SCCF minus raw chiral Morgan-r2; report all three readouts, no selective endpoint reporting',
       'fingerprints':{'rdkit':'2025.09.6','blocks':list(BLOCKS),'generator':generator_metadata(BLOCKS,'chiral'),
                       'normalization':'log1p count blocks, each block L2, concatenate; all spaces cosine',
                       'morgan':'same chiral radius2/2048 count block as official SCCF, not previously inspected nonchiral teacher reference'},
       'e1_scope':'Frozen official-val-selected molecule towers; train examples reused; paired-DreaMS candidate CE, not this raw spectral-cosine target. Task-related/in-sample diagnostic, not generalization or a symmetric trained-SCCF comparison.',
       'previous_exposure':['teacher all-cohort raw MolDeBERTa and NONCHIRAL Morgan correlations','teacher repeatability/CE and internal learnability summaries',
                            'triplet counts corrected before this freeze; main1849/81formulas, strict1709/81',
                            'user-fixed both-view cliff label audit:1966/2365 instances; no four-space geometry results inspected'],
       'pair_scope':{'instances':2365,'unique_exact_isomer_pairs':926,'formula_clusters':355},
       'continuous_spectral_distance':'1 - existing mean_four_cross_cosine',
       'cliff_label':'((within_a+within_b)/2 - view0 >=0.05) AND ((within_a+within_b)/2 - view1 >=0.05)',
       'curve':{'main_readouts':['cliff fraction in closest10% pair mass','prefix-cliff-rate integral0-25% divided by0.25'],
                'tie_policy':'exact distance tie blocks included fractionally; weights define quantiles; bootstrap recomputes quantile mass',
                'grid':'q=0 limit of first positive-mass distance block, then0.001..0.25; continuous analytic AUC, not grid approximation',
                'direction':'lower cliff readouts = fewer repeat-supported large spectral differences among equally sized local-neighbour sets; not sufficient for retrieval quality'},
       'triplets':{'main':'abs(gap_view0)>=0.05 and same nonzero signs across views;1849 instances',
                   'strict_sensitivity':'also abs(gap_view1)>=0.05;1709 instances',
                   'weights':'each exact anchored triple (anchor,unordered alternatives) total1 over eligible conditions',
                   'distance_tie':'exact equal distances contribute0.5; explicit tie rate; target sign fixed by spectral views'},
       'weights':{'main':'each unordered molecular pair total1 over retained condition instances',
                  'pair_aggregate':'926 pairs; average continuous distance and condition cliff indicator; paired curves identical by construction, not independent validation',
                  'formula_equal':'each formula total1, each pair within formula equal; analogous anchored-triple weights',
                  'subsets':'recompute pair/triple weights after fixed robust/strict selection'},
       'bootstrap':{'replicates':BOOT,'unit':'whole eligible molecular formula','paired':'same cluster multiplicities for every space within each panel',
                    'interval':'percentile95%; conditional on selected train cohort/checkpoints; no training/selection uncertainty, no multiplicity correction; exploratory'},
       'robust':'existing1090instances/211formulas: no cross-view processed-hash overlap AND both within-repeat cosines<0.995; retain triplet only when both anchor edges pass',
       'auxiliary':{'rank':'weighted Spearman using mass midranks, continuous spectral distance; aggregate sensitivity uses pair-average distance',
                    'sali':'spectral distance/(1-cosine) within each space only; nonpositive/zero-distance collision denominator excluded and counted; weighted quantiles, no epsilon or cross-space mean comparison',
                    'collision':'exact equality of normalized feature vectors; exact positive numerical distance retained; no data-selected closeness threshold'},
       'noise_ceiling':'none claimed; repeatability is a reference, no division by0.86 or another unestablished upper bound',
       'input_source_sha256':{str(p.relative_to(ROOT)):sha256_file(p) for p in inputs},
       'moldeberta_parent_cache_sha256':teacher_contract['input_source_sha256'][str(fp.relative_to(ROOT))],
       'official_validation_read':False,'official_test_read':False,'training_performed':False}
    out.mkdir();write_json(out/'contract.json',c);write_json(out/'semantic_checks.json',checks)
    print('GEOMETRY CONTRACT FROZEN',sha256_file(out/'contract.json'),flush=True)


def unit(x):
    x=np.asarray(x,np.float64)
    norm=np.linalg.norm(x,axis=1,keepdims=True)
    assert np.all(norm>0) and np.isfinite(x).all()
    return x/norm


def extract_features(base,out,c,pairs,triplets):
    nodes=sorted({p[k] for p in pairs for k in ('a','b')}); index={s:i for i,s in enumerate(nodes)}
    assert len(nodes)==991 and all(t[k] in index for t in triplets for k in ('anchor','b','c'))
    with np.load(TRAIN_ROWS) as z:
        smiles=z['smiles'];positive=z['candidates'][:,0]
        ids=z['identifiers']
        gt={}
        for s,i in zip(smiles,positive):
            if s in index:
                if s in gt:assert gt[s]==int(i)
                gt[s]=int(i)
    assert len(gt)==len(nodes)
    rows=np.array([gt[s] for s in nodes],np.int64)
    lmmap=np.load(E1/'moldeberta_cache/embeddings.npy',mmap_mode='r')
    lookup=np.load(E1/'moldeberta_cache/global_to_cache.npy',mmap_mode='r')
    cr=np.asarray(lookup[rows],np.int64);assert np.all(cr>=0)
    lm=np.array(lmmap[cr],copy=True);del lmmap,lookup
    assert lm.shape==(991,768)
    init_worker(BLOCKS,'chiral');chunk=fingerprint_chunk((0,nodes));assert chunk['invalid']==0
    manifest=json.loads(SPARSE.read_text());assert manifest['rdkit_version']==rdBase.rdkitVersion==c['fingerprints']['rdkit']
    dense={};parity={};trows=torch.from_numpy(rows)
    for name in BLOCKS:
        f=chunk['features'][name];indptr=np.r_[0,np.cumsum(f['lengths'],dtype=np.int64)]
        dim=manifest['features'][name]['dim'];raw=np.zeros((len(nodes),dim),np.float32)
        for i in range(len(nodes)):
            lo,hi=indptr[i:i+2];raw[i,f['indices'][lo:hi]]=f['values'][lo:hi]
        if name!='maccs':raw=torch.log1p(torch.from_numpy(raw)).numpy()
        matrix=SparseFeatureMatrix(SPARSE.parent,manifest['features'][name])
        cached=matrix.gather(trows).numpy()
        maxerr=float(np.max(np.abs(raw-cached)));assert maxerr<1e-6,(name,maxerr)
        dense[name]=unit(raw);parity[name]={'all_train_nodes':len(nodes),'max_abs_error':maxerr}
        del matrix,cached,raw
    features={'morgan_r2':dense['count_morgan'],'sccf_raw':unit(np.concatenate([dense[n] for n in BLOCKS],axis=1)),
              'moldeberta_raw':unit(lm)}
    config=json.loads(CONFIG.read_text())['e1'];m=config['model'];steps={}
    torch.set_num_threads(1)
    with torch.inference_mode():
        for seed in SEEDS:
            ckpt=torch.load(str(E1/f'seed{seed}/checkpoint.pt'),map_location='cpu',weights_only=False,mmap=True)
            assert ckpt['seed']==seed and ckpt['module_sha256']==sha256_file(ROOT/'glmr/msalign_external.py')
            tower=MLP(768,int(m['d_hidden']),int(m['d_shared']),int(m['n_hidden_layers_mol']),float(m['dropout']))
            tower.load_state_dict({k[4:]:v for k,v in ckpt['model'].items() if k.startswith('mol.')},strict=True)
            tower.eval();assert not tower.training
            chunks=[]
            for start in range(0,len(lm),128):chunks.append(tower(torch.from_numpy(lm[start:start+128])).numpy())
            features[f'e1_seed{seed}']=unit(np.concatenate(chunks));steps[str(seed)]=int(ckpt['global_step'])
            del tower,ckpt,chunks;gc.collect()
    # Check previous raw-LM similarities on every teacher edge, without changing the earlier script.
    ai=np.array([index[p['a']] for p in pairs]);bi=np.array([index[p['b']] for p in pairs])
    lmc=np.sum(features['moldeberta_raw'][ai]*features['moldeberta_raw'][bi],axis=1)
    lmerr=float(np.max(np.abs(lmc-np.array([p['moldeberta_raw_cosine'] for p in pairs]))));assert lmerr<1e-10
    distances=[];collisions=[]
    for name in SPACES:
        f=features[name];eq=np.all(f[ai]==f[bi],axis=1)
        dist=np.clip(1-np.sum(f[ai]*f[bi],axis=1),0,2);dist[eq]=0.
        # Avoid silently treating tiny numerical dot error as an actual collision.
        assert np.all(dist[~eq]>0),(name,'nonidentical vectors with nonpositive computed distance')
        distances.append(dist);collisions.append(eq)
    np.savez(out/'node_features.npz',smiles=np.array(nodes),train_global_rows=rows,moldeberta_cache_rows=cr,**features)
    write_json(out/'feature_audit.json',{'status':'PASS','nodes':len(nodes),'rdkit':rdBase.rdkitVersion,
       'fingerprint_cache_parity':parity,'raw_moldeberta_teacher_max_abs_error':lmerr,'e1_selected_steps':steps,
       'selected_raw_lm_rows_sha256':array_hash(lm),'global_train_mirror_ids_sha256':array_hash(ids),
       'features_sha256':{name:array_hash(features[name]) for name in SPACES},
       'scope':'Only991 selected train molecule rows; parent LM cache fully checked at freeze; existing sparse cache reads selected rows only; no spectrum or val/test evaluation arrays opened'})
    return features,index,np.array(distances).T,np.array(collisions).T


def group_ids(keys):
    unique=sorted(set(keys));lookup={k:i for i,k in enumerate(unique)}
    return np.array([lookup[k] for k in keys],np.int64),unique


def equal_group_weights(group,formula,formula_equal=False):
    w=1/np.bincount(group)[group]
    if formula_equal:
        ng=np.bincount(formula,weights=w)
        w=w/ng[formula]
    return w


def weighted_quantile(values,w,qs):
    order=np.argsort(values,kind='stable');x=values[order];ww=w[order]
    positions=(np.cumsum(ww)-.5*ww)/ww.sum()
    return np.interp(qs,positions,x).tolist()


def ci(draw):
    finite=np.asarray(draw)[np.isfinite(draw)]
    return np.quantile(finite,[.025,.975]).tolist() if len(finite) else None


def analyze_pairs(label,dist,y,cliff,pair_ids,formulas,out,formula_equal=False,aggregate=False):
    localpair,unique_pairs=group_ids(list(pair_ids));fi,uf=group_ids(list(formulas))
    w=equal_group_weights(localpair,fi,formula_equal)
    if aggregate:
        ix=np.array([np.flatnonzero(localpair==i)[0] for i in range(len(unique_pairs))])
        count=np.bincount(localpair)
        y=np.bincount(localpair,weights=y)/count
        cliff=np.bincount(localpair,weights=cliff)/count
        dist=dist[ix];formulas=np.asarray(formulas)[ix];fi,uf=group_ids(list(formulas))
        localpair=np.arange(len(ix));w=equal_group_weights(localpair,fi,formula_equal)
    G=len(uf);rng=np.random.default_rng(20261015)
    mult=np.array([np.bincount(rng.integers(0,G,G),minlength=G) for _ in range(BOOT)],np.int32)
    orders=np.array([np.argsort(dist[:,j],kind='stable') for j in range(len(SPACES))]);oy=np.argsort(y,kind='stable')
    metrics=np.empty((BOOT,len(SPACES),3));points=np.empty((len(SPACES),3));curves={};sali={};collision_stats={}
    qgrid=np.r_[0.,np.linspace(.001,.25,250)]
    for j,name in enumerate(SPACES):
        x=np.ascontiguousarray(dist[:,j]);order=orders[j]
        points[j,:2]=curve_readouts(x,cliff,w,order)
        points[j,2]=weighted_spearman(x,y,w,order,oy)
        curve=[]
        for q in qgrid:
            if q==0:
                pos=x[w>0].min();mask=(x==pos);rate=float(np.average(cliff[mask],weights=w[mask]))
            else:rate=float(curve_readouts(x,cliff,w,order,q,.25)[0])
            curve.append(rate)
        curves[name]=curve
        valid=x>0;ratio=y[valid]/x[valid]
        sali[name]={'definition':'(1-mean_four_cross_cosine)/(1-molecule_cosine), within-space only',
                    'weighted_quantiles_q0_q25_q50_q75_q90_q100':weighted_quantile(ratio,w[valid],[0,.25,.5,.75,.9,1]) if valid.any() else None,
                    'nonpositive_distance_instances_excluded':int((~valid).sum()),'excluded_weight_fraction':float(w[~valid].sum()/w.sum())}
        collision_stats[name]={'zero_distance_rows':int((~valid).sum()),'zero_distance_pairs':len(set(np.asarray(pair_ids)[~valid])) if not aggregate else int((~valid).sum()),
                               'weight_fraction':float(w[~valid].sum()/w.sum()),
                               'cliff_fraction_in_zero_distance':float(np.average(cliff[~valid],weights=w[~valid])) if (~valid).any() else None}
    for b in range(BOOT):
        wb=w*mult[b,fi]
        for j in range(len(SPACES)):
            x=np.ascontiguousarray(dist[:,j]);metrics[b,j,:2]=curve_readouts(x,cliff,wb,orders[j])
            metrics[b,j,2]=weighted_spearman(x,y,wb,orders[j],oy)
    names=['cliff_nearest10','cliff_auc0_25_normalized','distance_spectral_distance_spearman']
    result={'rows':len(y),'unique_pairs':len(unique_pairs),'formula_clusters':G,'weighting':'formula_equal' if formula_equal else 'pair_equal',
            'aggregate_conditions':aggregate,'overall_cliff_fraction':float(np.average(cliff,weights=w)),
            'metrics':{name:{m:{'point':float(points[j,k]),'ci95':ci(metrics[:,j,k])} for k,m in enumerate(names)} for j,name in enumerate(SPACES)},
            'main_sccf_minus_morgan':{m:{'point':float(points[1,k]-points[0,k]),'ci95':ci(metrics[:,1,k]-metrics[:,0,k])} for k,m in enumerate(names)},
            'sali_within_space_only':sali,'zero_distance':collision_stats}
    np.savez(out/f'{label}.bootstrap.npz',draws=metrics,spaces=np.array(SPACES),metric_names=np.array(names),formula_names=np.array(uf),cluster_multiplicities=mult)
    write_json(out/f'{label}.curves.json',{'q':qgrid.tolist(),'curves':curves})
    write_json(out/f'{label}.json',result)
    print('PAIR PANEL COMPLETE',label,json.dumps(result['main_sccf_minus_morgan']),flush=True)
    return result


def analyze_triples(label,triplets,features,index,out,formula_equal=False):
    assert triplets,label
    anchor=np.array([index[t['anchor']] for t in triplets]);bb=np.array([index[t['b']] for t in triplets]);cc=np.array([index[t['c']] for t in triplets])
    target=np.sign([t['gap_view0'] for t in triplets])
    tid,ut=group_ids([(t['anchor'],*sorted([t['b'],t['c']])) for t in triplets]);fi,uf=group_ids([t['formula'] for t in triplets])
    w=equal_group_weights(tid,fi,formula_equal);scores=[];ties=[]
    for name in SPACES:
        f=features[name]
        # spectral gap>0 means b is closer than c; compare molecular cosine in same direction.
        gap=np.sum(f[anchor]*f[bb],axis=1)-np.sum(f[anchor]*f[cc],axis=1)
        tie=gap==0;s=(np.sign(gap)==target).astype(float);s[tie]=.5;scores.append(s);ties.append(tie)
    scores=np.array(scores).T;ties=np.array(ties).T
    G=len(uf);rng=np.random.default_rng(20261015)
    mult=np.array([np.bincount(rng.integers(0,G,G),minlength=G) for _ in range(BOOT)],np.int32)
    mass=np.bincount(fi,weights=w);weighted=np.stack([np.bincount(fi,weights=w*scores[:,j],minlength=G) for j in range(len(SPACES))],axis=1)
    draws=(mult@weighted)/(mult@mass)[:,None]
    point=np.average(scores,axis=0,weights=w)
    result={'instances':len(triplets),'unique_anchored_triples':len(ut),'formula_clusters':G,
            'weighting':'formula_equal' if formula_equal else 'anchored_triple_equal',
            'metrics':{name:{'agreement':float(point[j]),'ci95':ci(draws[:,j]),'exact_tie_fraction':float(np.average(ties[:,j],weights=w))} for j,name in enumerate(SPACES)},
            'main_sccf_minus_morgan':{'point':float(point[1]-point[0]),'ci95':ci(draws[:,1]-draws[:,0])}}
    np.savez(out/f'{label}.bootstrap.npz',draws=draws,spaces=np.array(SPACES),formula_names=np.array(uf),cluster_multiplicities=mult,
             per_instance_agreement=scores,per_instance_tie=ties,weights=w)
    write_json(out/f'{label}.json',result);print('TRIPLET PANEL COMPLETE',label,json.dumps(result['main_sccf_minus_morgan']),flush=True)
    return result


def figures(out,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Morgan-r2','SCCF','MolDeBERTa','E1 seed21','E1 seed22','E1 seed23']
    colors=['#3073a5','#c66427','#43855e','#8664a0','#aa82ba','#bfa4cb']
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,panel,title in zip(axes,['pairs_main','pairs_robust'],['All2365 condition instances','Robust1090 condition instances']):
        curves=json.loads((out/f'{panel}.curves.json').read_text());q=np.array(curves['q'])*100
        for j,name in enumerate(SPACES):ax.plot(q,np.array(curves['curves'][name])*100,label=labels[j],color=colors[j],ls='--' if j>=3 else '-')
        ax.axvline(10,color='grey',lw=.7,ls=':');ax.set(xlabel='Closest molecular-pair mass (%)',ylabel='Two-view spectral cliff fraction (%)',title=title,xlim=(0,25),ylim=(0,100))
    axes[0].legend(fontsize=8);fig.savefig(out/'cliff_curves.png',dpi=180);fig.savefig(out/'cliff_curves.pdf');plt.close(fig)
    result=summary['triplet_panels']['triplets_main']['metrics'];fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for j,name in enumerate(SPACES):
        r=result[name];lo,hi=r['ci95'];ax.errorbar(j,r['agreement']*100,yerr=np.array([[r['agreement']-lo],[hi-r['agreement']]])*100,fmt='o',color=colors[j],capsize=3)
    ax.axhline(50,color='grey',ls=':',lw=.7);ax.set_xticks(range(len(labels)),labels,rotation=20,ha='right');ax.set(ylabel='Anchored ordinal agreement (%)',title='Train-only spectral-sign-consistent triples; formula-cluster95% CI')
    fig.savefig(out/'triplet_agreement.png',dpi=180);fig.savefig(out/'triplet_agreement.pdf');plt.close(fig)


def run(base):
    started=time.time();out=base/'geometry';assert not (out/'summary.json').exists()
    c=json.loads((out/'contract.json').read_text())
    for p,h in c['input_source_sha256'].items():assert sha256_file(ROOT/p)==h,p
    for rel in ['teacher/complete.json','inventory_complete.json']:
        path=base/rel;d=json.loads(path.read_text())
        for p,h in d['files_sha256'].items():assert sha256_file(path.parent/p)==h,(rel,p)
    resources();checks=semantic_checks();pairs=read_jsonl(base/'teacher/pairs.jsonl');triplets=read_jsonl(base/'teacher/triplets.jsonl')
    features,index,dist,collision=extract_features(base,out,c,pairs,triplets)
    keys=[(p['a'],p['b']) for p in pairs];pid,up=group_ids(keys)
    labels=np.array(['|'.join(k) for k in keys]);formula=np.array([p['formula'] for p in pairs])
    within=np.array([.5*(p['within_a_cosine']+p['within_b_cosine']) for p in pairs]);v0=np.array([p['view0_cosine'] for p in pairs]);v1=np.array([p['view1_cosine'] for p in pairs])
    spectral=1-np.array([p['mean_four_cross_cosine'] for p in pairs]);cliff=((within-v0>=.05)&(within-v1>=.05)).astype(float)
    rows=read_jsonl(base/'train.rows.jsonl')
    robust=[]
    for p in pairs:
        a,b=p['rows_a'],p['rows_b'];view0={rows[a[0]]['processed_spectrum_sha256'],rows[b[0]]['processed_spectrum_sha256']};view1={rows[a[1]]['processed_spectrum_sha256'],rows[b[1]]['processed_spectrum_sha256']}
        robust.append(not(view0&view1) and p['within_a_cosine']<.995 and p['within_b_cosine']<.995)
    robust=np.array(robust);assert robust.sum()==1090 and len(set(formula[robust]))==211
    np.savez(out/'pair_metrics.npz',distances=dist,spaces=np.array(SPACES),spectral_distance=spectral,view0_cosine=v0,view1_cosine=v1,
             mean_within_cosine=within,cliff_label=cliff,robust=robust,pair_ids=pid,pair_names=np.array(['|'.join(k) for k in up]),formula=formula,exact_vector_collision=collision)
    pair_panels={}
    for subset,mask in [('main',np.ones(len(pairs),bool)),('robust',robust)]:
        for suffix,fe,ag in [('',False,False),('_aggregate',False,True),('_formula_equal',True,False)]:
            name='pairs_'+subset+suffix
            pair_panels[name]=analyze_pairs(name,dist[mask],spectral[mask],cliff[mask],labels[mask],formula[mask],out,fe,ag)
        a=pair_panels['pairs_'+subset];b=pair_panels['pairs_'+subset+'_aggregate']
        for space in SPACES:
            for metric in ['cliff_nearest10','cliff_auc0_25_normalized']:
                assert abs(a['metrics'][space][metric]['point']-b['metrics'][space][metric]['point'])<1e-12
    edge={(p['a'],p['b'],tuple(p['condition'])):bool(robust[i]) for i,p in enumerate(pairs)}
    def robust_triple(t):
        return all(edge[(*sorted([t['anchor'],t[k]]),tuple(t['condition']))] for k in ('b','c'))
    selected=[t for t in triplets if abs(t['gap_view0'])>=.05 and np.sign(t['gap_view0'])==np.sign(t['gap_view1'])]
    assert len(selected)==1849
    triple_panels={}
    for subset,ts in [('main',selected),('strict',[t for t in selected if abs(t['gap_view1'])>=.05]),
                      ('robust',[t for t in selected if robust_triple(t)]),('robust_strict',[t for t in selected if abs(t['gap_view1'])>=.05 and robust_triple(t)])]:
        name='triplets_'+subset;triple_panels[name]=analyze_triples(name,ts,features,index,out)
        if subset in ('main','robust'):
            name='triplets_'+subset+'_formula_equal';triple_panels[name]=analyze_triples(name,ts,features,index,out,True)
    summary={'status':'COMPLETE_DESCRIPTIVE_ONLY','date':'2026-10-15','contract_sha256':sha256_file(out/'contract.json'),
             'pair_panels':pair_panels,'triplet_panels':triple_panels,'semantic_checks':checks,
             'resources':resources(),'elapsed_seconds_before_figures':time.time()-started,'main_contrast':'raw SCCF minus matched raw chiral Morgan-r2',
             'interpretation':'Selected same-formula TRAIN subset only; related/supervised E1 spaces are in-sample/val-selected, no trained SCCF comparator, not encoder/retrieval superiority. Bootstrap conditions on selected cohort/checkpoints and omits model-selection uncertainty; no multiplicity correction.',
             'official_validation_read':False,'official_test_read':False,'training_performed':False,'gpu_used':False}
    figures(out,summary);summary['elapsed_seconds']=time.time()-started;write_json(out/'summary.json',summary)
    files=[p for p in out.iterdir() if p.is_file() and p.name!='complete.json']
    write_json(out/'complete.json',{'status':'COMPLETE','files_sha256':{p.name:sha256_file(p) for p in files},
        'official_validation_read':False,'official_test_read':False,'training_performed':False,'gpu_used':False})
    print('GEOMETRY COMPLETE',summary['elapsed_seconds'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['check','freeze','run']);parser.add_argument('--base',type=Path,default=BASE);args=parser.parse_args()
    if args.action=='check':print(json.dumps(semantic_checks(),indent=2))
    elif args.action=='freeze':freeze(args.base)
    else:run(args.base)


if __name__=='__main__': main()
