#!/usr/bin/env python3
"""Frozen manuscript supplements; validation first, post-hoc test replicas second.

All reported metric values are percentages (including MRR times 100).
No manuscript edits, checkpoint fitting, or modification of sealed inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Share the execution state when helper modules import this CLI module.
if __name__ == '__main__':
    sys.modules['tools.analysis.revision_supplement'] = sys.modules[__name__]
from glmr.embcos import uniform_tie_metrics

OUT = ROOT / 'outputs/revision_supplement_20261111'
TEST = ROOT / 'outputs/official_s7_test_20261017'
FREEZE = ROOT / 'outputs/official_final_freeze_20261017'
CONTROL = ROOT / 'outputs/official_d3_full_system_controls_20261017/formal'
STAGE = ROOT / 'outputs/stage2_official_datasafe/stage1_candidate_scores/val_three_seed_logits.npz'
EXTERNAL = {'msalign': ROOT/'outputs/msalign_official_e1_20261014', 'embcos': ROOT/'outputs/embcos_official_e2_20261014'}
SEEDS = (20260621, 20260622, 20260623)
METRICS = ('R1', 'R5', 'R20', 'MRRx100')
READS = {}
START = time.time()
BOOT = 10000
SEED = 20261111


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*2**20), b''): h.update(b)
    return h.hexdigest()


def bind(path):
    path = Path(path).resolve()
    key = str(path.relative_to(ROOT))
    if key not in READS: READS[key] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return path


def load(path, keys=None):
    with np.load(bind(path), allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if keys is None else keys)}


def read(path): return json.loads(bind(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.staging')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def groups(values, identities):
    values = np.asarray(values, np.float64)
    if values.ndim == 1: values = values[:, None]
    unique, inverse, counts = np.unique(identities, return_inverse=True, return_counts=True)
    sums = np.zeros((len(unique), values.shape[1]), np.float64)
    np.add.at(sums, inverse, values)
    return unique, inverse, counts, sums, sums/counts[:, None]


def stats(values, identities, names=METRICS):
    """Whole-connectivity paired bootstrap; bounded batches, no query iid intervals."""
    values = np.asarray(values, np.float64)
    if values.ndim == 1: values = values[:, None]
    unique, inverse, counts, sums, means = groups(values, identities)
    assert len(unique) and np.isfinite(values).all()
    rng = np.random.default_rng(SEED)
    qw = np.empty((BOOT, values.shape[1])); bal = np.empty_like(qw)
    for st in range(0, BOOT, 32):
        ix = rng.integers(0, len(unique), size=(min(32, BOOT-st), len(unique)))
        qw[st:st+len(ix)] = sums[ix].sum(1)/counts[ix].sum(1)[:, None]
        bal[st:st+len(ix)] = means[ix].mean(1)
    def cells(point, samples):
        lo, hi = np.quantile(samples, [.025, .975], axis=0)
        return {k: {'point': float(100*v), 'ci95': [float(100*a), float(100*b)]}
                for k, v, a, b in zip(names, point, lo, hi)}
    return {'queries': len(values), 'connectivities': len(unique),
            'query_weighted': cells(values.mean(0), qw),
            'connectivity_balanced': cells(means.mean(0), bal)}


def metrics_scores(scores, counts, gt):
    scores = np.asarray(scores)
    mask = np.arange(scores.shape[1])[None, :] < np.asarray(counts)[:, None]
    assert np.isfinite(scores[mask]).all()
    truth = scores[np.arange(len(scores)), gt]
    greater = ((scores > truth[:, None]) & mask).sum(1)
    equal = ((scores == truth[:, None]) & mask).sum(1)
    assert (equal > 0).all()
    return uniform_tie_metrics(greater, equal)


def base(split):
    return load(FREEZE/'validation_per_query.npz' if split == 'validation' else TEST/'per_query.npz')


def rows(split):
    return load(ROOT/'outputs/embcos_official_e2_20261014/cache/val.rows.npz' if split == 'validation' else TEST/'rows.npz')


def external(split, name, seed):
    return load(EXTERNAL[name]/f'seed{seed}/val_ranks.npz' if split == 'validation' else TEST/('e1' if name == 'msalign' else 'e2')/f'seed{seed}.npz')


def finish(item, split, summary, arrays):
    folder = OUT/item/split; folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'per_query.npz', **arrays)
    summary.update(item=item, split=split, status='FINAL', post_hoc=True,
                   official_test_read=split == 'test', frozen_S7_rows_changed=False,
                   metric_unit='percent; MRR times 100',
                   uncertainty={'unit': 'recomputed connectivity', 'resamples': BOOT, 'seed': SEED, 'interval': 'percentile 95%'},
                   command='OMP_NUM_THREADS=1 '+sys.executable+' '+' '.join(sys.argv),
                   inputs=READS.copy(), source_sha256=sha(Path(__file__)), wall_seconds=time.time()-START)
    summary['per_query_sha256'] = sha(folder/'per_query.npz')
    write(folder/'summary.json', summary)
    print(json.dumps({'item': item, 'split': split, 'status': 'FINAL', 'headline': summary.get('headline', '')}), flush=True)


def r3(split):
    b = base(split); r = rows(split); ids = b['identifiers']; g = b['connectivities']
    assert np.array_equal(ids, r['identifiers']) and np.array_equal(g, r['connectivities'])
    arrays = {'identifiers': ids, 'connectivities': g, 'candidate_counts': r['lengths'], 'gt_slots': r['gt_slots']}
    values = {'sccf_ensemble': b['sccf_ensemble'], 'selected': b['top20_support_fallback']}
    for name in EXTERNAL:
        pool = []
        for seed in SEEDS:
            z = external(split, name, seed)
            for key, ref in [('identifiers', ids), ('connectivities', g), ('candidate_counts', r['lengths']), ('gt_slots', r['gt_slots'])]:
                assert np.array_equal(z[key], ref), (name, seed, key)
            np.testing.assert_array_equal(metrics_scores(z['scores'], r['lengths'], r['gt_slots']), z['expected_metrics'])
            pool.append(z['scores'].astype(np.float64))
        scores = np.mean(pool, axis=0)
        values[name+'_ensemble'] = metrics_scores(scores, r['lengths'], r['gt_slots'])
        arrays[name+'_ensemble_scores'] = scores
    arrays.update(values)
    reports = {k: stats(v, g) for k, v in values.items()}
    contrasts = {}
    for a, c in [('selected', 'msalign_ensemble'), ('sccf_ensemble', 'msalign_ensemble'), ('selected', 'embcos_ensemble'), ('sccf_ensemble', 'embcos_ensemble')]:
        report = stats(values[a]-values[c], g)
        gap = float((values[c][:, 2]-values[c][:, 0]).mean())
        report['R1_to_R20_gap_closure_pct'] = float(100*(values[a][:, 0]-values[c][:, 0]).mean()/gap) if gap else None
        contrasts[a+'_minus_'+c] = report
    headline = {k: {w: v[w]['R1'] for w in ('query_weighted', 'connectivity_balanced')} for k, v in contrasts.items()}
    finish('R3', split, {'rows': reports, 'contrasts': contrasts, 'headline': headline,
                        'aggregation': 'unweighted arithmetic mean of three stored logits, float64 accumulation; no temperature/weight/subset selection',
                        'candidate_alignment': 'identifiers, connectivity, candidate counts and GT slots exact across seeds and frozen shared candidate-row archive'}, arrays)



def all_method_values(split):
    b = base(split); r = rows(split); ids = b['identifiers']; g = b['connectivities']
    assert np.array_equal(ids, r['identifiers']) and np.array_equal(g, r['connectivities'])
    values = {'sccf_ensemble': b['sccf_ensemble'], 'original_d3': b['top20_detail'], 'selected': b['top20_support_fallback']}
    if split == 'validation':
        for seed in SEEDS:
            z = load(ROOT/f'outputs/stage1_submission_0926/runs/full_r2_block_l2/seed{seed}/blind_controls/constant_spectrum_full_val.per_query.npz')
            assert np.array_equal(z['connectivity'], g)
            values[f'sccf_seed{seed}'] = uniform_tie_metrics(z['baseline_greater'], z['baseline_equal'])
    else:
        z = load(TEST/'sccf_seeds.npz')
        for seed in SEEDS: values[f'sccf_seed{seed}'] = z[f'seed{seed}']
    for name in EXTERNAL:
        for seed in SEEDS:
            z = external(split, name, seed)
            assert np.array_equal(z['identifiers'], ids) and np.array_equal(z['connectivities'], g)
            values[f'{name}_seed{seed}'] = z['expected_metrics']
        np.testing.assert_allclose(np.mean([values[f'{name}_seed{seed}'] for seed in SEEDS], axis=0), b[name+'_seed_mean'], atol=1e-14, rtol=0)
    np.testing.assert_allclose(np.mean([values[f'sccf_seed{seed}'] for seed in SEEDS], axis=0), b['sccf_seed_mean'], atol=1e-14, rtol=0)
    system = load(CONTROL/'real.npz' if split == 'validation' else TEST/'real.npz')
    assert np.array_equal(system['identifiers'], ids)
    for name, key in [('original_d3','values_original_d3'), ('selected','values_selected')]:
        np.testing.assert_array_equal(values[name], system[key])
    return ids, g, r, values


def icc_sufficient(n, total, square):
    n = np.asarray(n); total = np.asarray(total); square = np.asarray(square)
    G = n.shape[-1]; N = n.sum(axis=-1)
    k0 = (N-(n*n).sum(axis=-1)/N)/(G-1)
    msb = ((total*total/n).sum(axis=-1)-total.sum(axis=-1)**2/N)/(G-1)
    msw = (square-total*total/n).sum(axis=-1)/(N-G)
    den = msb+(k0-1)*msw
    return np.divide(msb-msw, den, out=np.full_like(den, np.nan, dtype=float), where=den>0)


def clean_interval(v):
    v = np.asarray(v); v = v[np.isfinite(v)]
    return [float(x) for x in np.quantile(v, [.025,.975])] if len(v) else None


def cluster_characterization(hit, identities, minimum, bootstrap=True):
    unique, inverse, n, sums, means = groups(hit, identities)
    total = sums[:,0]; squares = np.bincount(inverse, weights=np.asarray(hit)**2)
    maximum = np.full(len(unique), -np.inf); np.maximum.at(maximum, inverse, hit)
    keep = n >= minimum
    n, total, squares, maximum = (x[keep] for x in (n,total,squares,maximum))
    assert len(n)>1 and n.sum()>len(n)
    wrong = total==0; right = total==n
    mixed = ~(wrong|right)
    def calc(n,t,sq,m,w,ri,mi):
        return np.stack([icc_sufficient(n,t,sq),w.mean(-1),ri.mean(-1),mi.mean(-1),
                         (n*w).sum(-1)/n.sum(-1),t.sum(-1)/n.sum(-1),
                         (n*m).sum(-1)/n.sum(-1),((n*m).sum(-1)-t.sum(-1))/n.sum(-1)],axis=-1)
    labels=['ICC1','all_wrong_molecule_share','all_right_molecule_share','mixed_molecule_share','all_wrong_query_share','baseline','sibling_oracle','oracle_gain']
    point=calc(n,total,squares,maximum,wrong,right,mixed)
    result={'groups':len(n),'queries':int(n.sum()),'all_query_coverage_pct':float(100*n.sum()/len(hit)),
            'all_wrong_groups':int(wrong.sum()),'all_right_groups':int(right.sum()),'mixed_groups':int(mixed.sum()),
            'fractional_tie_queries':int(((hit>0)&(hit<1)).sum()),
            'values':{k:float(v) if np.isfinite(v) else None for k,v in zip(labels,point)}}
    if bootstrap:
        rng=np.random.default_rng(SEED);samples=np.empty((BOOT,len(labels)))
        for st in range(0,BOOT,64):
            ix=rng.integers(0,len(n),size=(min(64,BOOT-st),len(n)))
            samples[st:st+len(ix)]=calc(n[ix],total[ix],squares[ix],maximum[ix],wrong[ix],right[ix],mixed[ix])
        result['ci95']={k:clean_interval(samples[:,i]) for i,k in enumerate(labels)}
    return result


def null_clustering(hit, identities, minimum):
    unique, inv, n = np.unique(identities,return_inverse=True,return_counts=True)
    # Permute the full query vector before selecting >=4 groups; group sizes fixed.
    keep=n>=minimum; rng=np.random.default_rng(SEED); iccs=[];wrong=[]
    for _ in range(1000):
        x=rng.permutation(hit);total=np.bincount(inv,weights=x);square=np.bincount(inv,weights=x*x)
        iccs.append(float(icc_sufficient(n[keep],total[keep],square[keep])))
        wrong.append(float(np.mean(total[keep]==0)))
    return {'permutations':1000,'permutation_scope':'all queries; fixed group sizes; then minimum group-size mask',
            'ICC1_mean':float(np.nanmean(iccs)),'ICC1_reference95':clean_interval(iccs),
            'all_wrong_share_mean':float(np.mean(wrong)),'all_wrong_share_reference95':clean_interval(wrong)}


def r1a(split):
    ids,g,r,values=all_method_values(split)
    reports={};arrays={'identifiers':ids,'connectivities':g,'canonical_isomeric_smiles':r['smiles']}
    for name,v in values.items():
        outcomes={'R1':v[:,0]}
        if name.startswith('sccf'):outcomes['R20']=v[:,2]
        reports[name]={}
        for outcome,hit in outcomes.items():
            arrays[name+'_'+outcome]=hit
            reports[name][outcome]={'minimum4':cluster_characterization(hit,g,4),
                                  'all_molecules':cluster_characterization(hit,g,1),
                                  'null_minimum4':null_clustering(hit,g,4),
                                  'null_all_molecules':null_clustering(hit,g,1),
                                  'canonical_string_sensitivity_minimum4':cluster_characterization(hit,r['smiles'],4)}
        print('R1a_ROW',name,reports[name]['R1']['minimum4']['values']['ICC1'],flush=True)
    unique,inv,n=np.unique(g,return_inverse=True,return_counts=True)
    def wrong_set(name,minimum):
        total=np.bincount(inv,weights=values[name][:,0]);return set(unique[(n>=minimum)&(total==0)])
    pairs=[(f'sccf_seed{a}',f'sccf_seed{b}') for i,a in enumerate(SEEDS) for b in SEEDS[i+1:]]
    trio=['sccf_ensemble','selected','msalign_seed20260621']
    pairs += [(a,b) for i,a in enumerate(trio) for b in trio[i+1:]]
    stability={}
    for minimum in [1,4]:
        for a,b in pairs:
            A,B=wrong_set(a,minimum),wrong_set(b,minimum)
            stability[f'{a}_vs_{b}_minimum{minimum}']={'intersection':len(A&B),'union':len(A|B),'IoU':len(A&B)/len(A|B) if A|B else None}
    headline={name:rep['R1']['minimum4']['values'] for name,rep in reports.items() if name in ['sccf_ensemble','selected','msalign_seed20260621']}
    finish('R1a',split,{'rows':reports,'all_wrong_stability':stability,'headline':headline,
                      'outcome':'stored uniform-tie expected hits, validated against frozen contributions; classes use exactly 0 and 1',
                      'oracle':'maximum sibling expected hit propagated to all queries in that molecule; equals any-correct oracle for binary hits',
                      'historical_reference':{'protocol':'custom post-P0.3 predecessor; never pooled with official split','ICC1':.756,'all_wrong_pct':44.9,'all_right_pct':28.4,'oracle_gain_pp':13.19},
                      'units':'clustering values and intervals are fractions, ICC dimensionless; metric tables elsewhere percent'},arrays)



def r2(split):
    b=base(split);r=rows(split);g=b['connectivities'];ids=b['identifiers'];n=len(ids)
    donor_seeds=(20260820,20260821,20260822)
    modes=['real','zero','train_mean',*[f'wrong_seed{x}' for x in donor_seeds]]
    data={};queries={};donors={};arrays={'identifiers':ids,'connectivities':g}
    for mode in modes:
        z=load(CONTROL/(mode+'.npz') if split=='validation' else TEST/(mode+'.npz'))
        q=z['queries'];queries[mode]=q
        assert np.array_equal(z['identifiers'],ids[q]) and np.array_equal(z['groups'],g[q])
        data[('sccf_ensemble',mode)]=z['values_stage1'];data[('selected',mode)]=z['values_selected']
        if mode.startswith('wrong'):
            donors[mode]=z['source_rows'];assert np.all(g[q]!=g[z['source_rows']])
    common=queries[modes[-1]]
    for mode in modes[3:]:assert np.array_equal(queries[mode],common)
    assert len(common)>0
    for name in EXTERNAL:
        for mode in modes:
            seedvalues=[];seedscores=[]
            for seed in SEEDS:
                path=(EXTERNAL[name]/f'seed{seed}/controls/{mode}.npz' if split=='validation' else OUT/'R2/test/inference'/('e1' if name=='msalign' else 'e2')/mode/f'seed{seed}.npz')
                z=load(path);q=z['recipient_rows']
                assert np.array_equal(q,queries[mode]),(name,seed,mode,'recipient slice')
                assert np.array_equal(z['identifiers'],ids[q]) and np.array_equal(z['connectivities'],g[q])
                assert np.array_equal(z['candidate_counts'],r['lengths'][q]) and np.array_equal(z['gt_slots'],r['gt_slots'][q])
                if mode.startswith('wrong'):assert np.array_equal(z['spectrum_rows'],donors[mode]),(name,mode,'donors')
                seedvalues.append(z['expected_metrics']);seedscores.append(z['scores'].astype(np.float64))
            data[(name+'_seed_mean',mode)]=np.mean(seedvalues,axis=0)
            data[(name+'_ensemble',mode)]=metrics_scores(np.mean(seedscores,axis=0),r['lengths'][q],r['gt_slots'][q])
    prior=load(ROOT/'outputs/official_s7_preparation_20261017/candidate_prior/val_expected_metrics.npz' if split=='validation' else TEST/'candidate_prior.npz')
    if split=='validation':assert np.array_equal(prior['identifiers'],ids) and np.array_equal(prior['connectivities'],g)
    random=uniform_tie_metrics(np.zeros(n,dtype=int),r['lengths'])
    methods=['sccf_ensemble','selected','msalign_seed_mean','msalign_ensemble','embcos_seed_mean','embcos_ensemble']
    for method in methods:
        data[(method,'candidate_prior')]=prior['expected_metrics'];data[(method,'exact_random')]=random
    queries['candidate_prior']=queries['exact_random']=np.arange(n)
    result={}
    for population,index in [('full',np.arange(n)),('strict_common',common)]:
        result[population]={'queries':len(index),'connectivities':len(np.unique(g[index])),'rows':{}}
        for method in methods:
            real=data[(method,'real')][index,:1];table={}
            for mode in [*modes,'candidate_prior','exact_random']:
                if population=='full' and mode.startswith('wrong'):continue
                q=queries[mode];positions=np.searchsorted(q,index)
                assert np.array_equal(q[positions],index)
                v=data[(method,mode)][positions,:1]
                summ=stats(v,g[index],('R1',));delta=stats(real-v,g[index],('R1',)) if mode!='real' else None
                retention={w:(summ[w]['R1']['point']/stats_real if stats_real else None)
                           for w,stats_real in [(w,float(real.mean()*100) if w=='query_weighted' else float(groups(real,g[index])[4].mean()*100)) for w in ['query_weighted','connectivity_balanced']]}
                table[mode]={'metrics':summ,'real_minus_control_pp':delta,'control_over_real_retention':retention}
            result[population]['rows'][method]=table
            print('R2_TABLE',population,method,flush=True)
    for (method,mode),v in data.items():arrays[method+'__'+mode]=v
    for mode,q in queries.items():arrays['queries__'+mode]=q
    for mode,d in donors.items():arrays['donors__'+mode]=d
    headline={method:result['strict_common']['rows'][method]['wrong_seed20260820']['real_minus_control_pp']['query_weighted']['R1'] for method in methods}
    finish('R2',split,{'populations':result,'headline':headline,'donor_alignment':'exact same recipients and donor assignments across all methods and three seeds; no intersection reduction needed',
                      'aggregation':'SCCF and selected use frozen three-seed upstream ensemble; external seed-mean and unweighted three-seed mean-logit ensemble both reported',
                      'controls_scope':'recipient CE/adduct retained for MSAlign zero/mean/wrong inputs; candidate-prior is a separate shared train-only diagnostic',
                      'retention':'descriptive ratio, not additive causal decomposition; strict_common is the matched main-table population'},arrays)



def r5a(split):
    b=base(split);r=rows(split);ids=b['identifiers'];g=b['connectivities']
    meta=load(ROOT/'outputs/msalign_official_e1_20261014/assets/val.metadata.npz' if split=='validation' else TEST/'metadata.npz')
    ce=meta['collision_energy'];assert len(ce)==len(ids) and not np.isinf(ce).any()
    ensembles=load(OUT/'R3'/split/'per_query.npz')
    assert np.array_equal(ensembles['identifiers'],ids)
    values={'sccf_ensemble':b['sccf_ensemble'],'msalign_seed_mean':b['msalign_seed_mean'],'msalign_ensemble':ensembles['msalign_ensemble'],'selected':b['top20_support_fallback']}
    reports={}
    for name,mask in [('CE_present',np.isfinite(ce)),('CE_missing',np.isnan(ce))]:
        count=r['lengths'][mask];u,f=np.unique(count,return_counts=True)
        reports[name]={'queries':int(mask.sum()),'connectivities':len(np.unique(g[mask])),
                       'candidate_counts':{'min':int(count.min()),'median':float(np.median(count)),'max':int(count.max()),'histogram':{str(k):int(v) for k,v in zip(u,f)}},
                       'rows':{method:stats(v[mask],g[mask]) for method,v in values.items()}}
    finish('R5a',split,{'strata':reports,'headline':{k:{'queries':v['queries'],'selected_R1':v['rows']['selected']['query_weighted']['R1']} for k,v in reports.items()},
                       'interpretation':'composition differs between CE strata; not a collision-energy effect estimate; metadata missingness is preserved, never recoded to zero'},
           {'identifiers':ids,'connectivities':g,'collision_energy':ce,'candidate_counts':r['lengths'],**values})


def r4(split):
    from tools.analysis.revision_mass_prior import run
    run(split)

def r1b(split):
    from tools.analysis.revision_proximity import run
    run(split)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('item', choices=['R3', 'R1a', 'R2', 'R5a', 'R4', 'R1b'])
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    args = parser.parse_args()
    if args.split == 'test':
        assert read(OUT/args.item/'validation/summary.json')['status'] == 'FINAL'
    {'R3': r3, 'R1a': r1a, 'R2': r2, 'R5a': r5a, 'R4': r4, 'R1b': r1b}[args.item](args.split)


if __name__ == '__main__': main()
