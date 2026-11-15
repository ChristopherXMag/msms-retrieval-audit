#!/usr/bin/env python3
"""Bounded CPU LambdaRank pilot on existing isolated official-train OOF lists."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb
import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from tools.reranking.train_official_mpfir_cross_candidate import (
    combine_scores, rank_scores, win_interval, wrong_spectrum_channel, write_json,
)
from tools.reranking.eval_official_mpfir_v2_top5 import (
    load_or_build_stores, masked_null_support, candidate_relative_score,
)
from tools.reranking.train_official_magma_forward import digest
from tools.reranking.train_official_mpfir_oof_retriever import memory_guard
from tools.analysis.analyze_stage1_logit_ensemble import comparison
from tools.reranking.train_official_datasafe_score_fusion import endpoint

SEEDS = (20261010, 20261011, 20261012)
TREE_COUNTS = (25, 50, 100, 200)
ARMS = ('score_only', 'coverage_detail', 'coverage_chemistry')
PARAMS = dict(objective='lambdarank', metric='ndcg', label_gain=[0, 1],
    lambdarank_truncation_level=5, num_leaves=15, max_depth=4,
    min_data_in_leaf=50, learning_rate=.03, lambda_l2=10.,
    feature_fraction=.8, bagging_fraction=1., num_threads=2,
    deterministic=True, force_col_wise=True, verbosity=-1)


def relative(values):
    values = np.asarray(values, dtype=np.float64)
    centered = values - symmetric_mean(values)
    scale = np.maximum(np.sqrt(symmetric_mean(centered**2)), .02)
    return centered, centered / scale


def symmetric_mean(values):
    # Sorted float64 accumulation prevents candidate-order rounding changes
    # from crossing hard tree split thresholds.
    return np.sort(values,axis=1).mean(axis=1,keepdims=True,dtype=np.float64)


def score_features(base, anchor):
    """Symmetric set statistics; no candidate position or identity inputs."""
    parts, names = [], []
    for prefix, values in (('sccf', base), ('anchor', anchor)):
        values = np.asarray(values,dtype=np.float64)
        centered, normalized = relative(values)
        exponent = np.exp(values-values.max(axis=1, keepdims=True))
        for name, value in (
            ('centered', centered), ('z', normalized),
            ('to_max', values-values.max(axis=1, keepdims=True)),
            ('prob', exponent/(5*symmetric_mean(exponent))),
            ('std', np.broadcast_to(np.sqrt(symmetric_mean(centered**2)), values.shape)),
        ):
            parts.append(value); names.append(prefix+'_'+name)
    return np.stack(parts, axis=-1).astype(np.float32), names


def evidence_features(inputs, valid, intensity):
    """Input-only candidate feasibility summaries, no GT annotations."""
    counts = np.asarray(valid.sum(-1), dtype=np.float64)
    hit = counts > 0
    intensity = np.maximum(np.asarray(intensity), 0)
    present = np.zeros(intensity.shape, bool)
    mz = np.zeros(intensity.shape, dtype=np.float64)
    for i, query in enumerate(inputs):
        observed = np.asarray(query['observed_spectrum'])
        present[i, :len(observed)] = True
        mz[i, :len(observed)] = observed[:, 0]
    hit &= present[:, None]
    count = hit.sum(-1)
    denom = np.maximum(present.sum(-1), 1)[:, None]
    weight = np.sqrt(intensity)[:, None]
    total = np.maximum(weight.sum(-1), 1e-12)
    unique = hit & (hit.sum(1)[:, None] == 1)
    bands = mz / np.asarray([q['precursor_mz'] for q in inputs])[:, None]
    values = [count, count/denom, np.log1p(counts.sum(-1)),
        counts.sum(-1)/np.maximum(count, 1),
        (hit*weight).sum(-1)/total, (unique*weight).sum(-1)/total,
        unique.sum(-1)/denom, ((counts == 1)*hit).sum(-1)/np.maximum(count, 1),
        (hit*mz[:, None]).sum(-1)/np.maximum(count, 1)]
    names = ['feasible_peaks', 'peak_fraction', 'log_edges', 'edges_per_peak',
        'sqrt_intensity_fraction', 'exclusive_intensity_fraction',
        'exclusive_peak_fraction', 'unambiguous_peak_fraction', 'matched_mean_mz']
    for label, band in (
        ('low_mz', bands < 1/3), ('middle_mz', (bands >= 1/3)&(bands < 2/3)),
        ('high_mz', bands >= 2/3), ('small_loss', bands >= .9),
    ):
        values.append((hit*weight*band[:, None]).sum(-1)/total)
        names.append(label+'_intensity_fraction')
    result = np.stack(values, -1).astype(np.float32)
    # Candidate-relative counterparts let trees use evidence against competitors.
    centered = result-symmetric_mean(result)
    return np.concatenate([result, centered], -1).astype(np.float32), names+['relative_'+s for s in names]


def molecule_features(inputs):
    cache = {}
    names = ['mass_error_ppm', 'heavy_atoms', 'rings', 'aromatic_fraction',
        'rotatable', 'hbd', 'hba', 'logp', 'tpsa', 'fraction_csp3',
        'count_C', 'count_N', 'count_O', 'count_S', 'count_P', 'count_F',
        'count_Cl', 'count_Br', 'count_I']
    values = []
    from glmr.subformula import ion_shift
    for query in inputs:
        candidates = []
        for smiles in query['candidate_smiles']:
            if smiles not in cache:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError('Invalid cached candidate')
                atoms = list(mol.GetAtoms())
                cache[smiles] = [Descriptors.ExactMolWt(mol),
                    mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcNumRings(mol),
                    sum(a.GetIsAromatic() for a in atoms)/max(len(atoms), 1),
                    Lipinski.NumRotatableBonds(mol), Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol), Descriptors.MolLogP(mol),
                    Descriptors.TPSA(mol), rdMolDescriptors.CalcFractionCSP3(mol),
                    *[sum(a.GetSymbol() == element for a in atoms)
                      for element in ('C','N','O','S','P','F','Cl','Br','I')]]
            row = cache[smiles].copy()
            row[0] = (row[0]+ion_shift(query['adduct'])-query['precursor_mz'])/query['precursor_mz']*1e6
            candidates.append(row)
        values.append(candidates)
    raw = np.asarray(values, dtype=np.float32)
    return np.concatenate([raw, raw-symmetric_mean(raw)], -1).astype(np.float32), names+['relative_'+s for s in names]


def feature_sets(inputs, base, anchor, valid, intensity, chemistry=None):
    scalar, score_names = score_features(base, anchor)
    evidence, evidence_names = evidence_features(inputs, valid, intensity)
    if chemistry is None:
        chemistry = molecule_features(inputs)
    chem, chem_names = chemistry
    features = dict(score_only=scalar,
        coverage_detail=np.concatenate([scalar, evidence], -1),
        coverage_chemistry=np.concatenate([scalar, evidence, chem], -1))
    names = dict(score_only=score_names, coverage_detail=score_names+evidence_names,
        coverage_chemistry=score_names+evidence_names+chem_names)
    if any(not np.isfinite(x).all() for x in features.values()):
        raise ValueError('Nonfinite features')
    return features, names, chemistry


def select_count(history):
    eligible = [r for r in history if r['delta_r1'] >= 0 and r['delta_mrr'] > 0 and r['damage'] <= .08]
    return max(eligible, key=lambda r:(r['mrr'], -r['trees']))['trees'] if eligible else 0


def calibration(before, after):
    return dict(r1=float(np.mean(after == 1)*100), mrr=float(np.mean(1/after)*100),
        delta_r1=float(np.mean((after == 1).astype(float)-(before == 1))*100),
        delta_mrr=float(np.mean(1/after-1/before)*100),
        damage=float(((before == 1)&(after > 1)).sum()/max((before == 1).sum(), 1)))


def equal_coverage(p, selected, scores):
    paired = []
    for pos, row in enumerate(selected):
        gt = int(p['labels'][row])
        if gt < 0 or not bool(p['support'][row]):
            continue
        group = p['records'][row]['connectivity']
        decoys = [j for j in range(5) if j != gt and p['candidate_connectivities'][row][j] != group]
        if not decoys:
            continue
        wrong = max(decoys, key=lambda j:float(p['coverage'][row,j]))
        a, b = int(p['feasible_peaks'][row,gt]), int(p['feasible_peaks'][row,wrong])
        paired.append(dict(pos=pos, gt=gt, wrong=wrong,
            equal_count=a == b and a > 0,
            equal_score=a > 0 and b > 0 and abs(float(p['coverage'][row,gt]-p['coverage'][row,wrong])) <= 1e-6,
            same_formula=bool(p['same_formula'][row,wrong])))
    report = {}
    for key in ('equal_count', 'equal_score'):
        for formula in (False, True):
            pairs = [r for r in paired if r[key] and (not formula or r['same_formula'])]
            wins = {}
            for name, score in scores.items():
                diff = np.asarray([score[r['pos'],r['gt']]-score[r['pos'],r['wrong']] for r in pairs])
                wins[name] = np.where(diff > 1e-6, 1., np.where(diff < -1e-6, 0., .5))
            report[key+('_same_formula' if formula else '')] = dict(queries=len(pairs),
                against_coverage={name:win_interval(wins['coverage'], win) for name,win in wins.items()},
                against_score_only={name:win_interval(wins['score_only'], wins[name]) for name in ARMS[1:]})
    return report


def run(args):
    start = time.time()
    memory_guard()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.data/'features/data.pt'
    p = torch.load(str(source), map_location='cpu', weights_only=False, mmap=True)
    source_summary = json.loads((args.data/'features/summary.json').read_text())
    if not source_summary['formal_oof'] or source_summary['official_validation_read'] or source_summary['official_test_read']:
        raise ValueError('Not isolated official-train source')
    for mode in ('mining0','mining1','endpoint'):
        audit = json.loads((args.data/mode/'summary.json').read_text())
        if audit['oof_gt_overlap'] or audit['negative_overlap'] or audit['plumbing_only']:
            raise ValueError('Upstream isolation failed')
    parts = np.asarray([r['pilot_partition'] for r in p['records']])
    groups = np.asarray([r['connectivity'] for r in p['records']])
    if len(np.unique(groups)) != len(groups):
        raise ValueError('Expected one spectrum per connectivity')
    indices = {name:np.flatnonzero(parts == name) for name in ('fit','calibration','evaluation')}
    base = p['base'].numpy(); coverage = p['coverage'].numpy()
    anchor = combine_scores(p['base'],p['coverage']).numpy()
    features, names, chemistry = feature_sets(p['inputs'],base,anchor,p['valid'],p['intensity'])
    labels = p['labels'].numpy(); support = p['support'].numpy()
    train = indices['fit'][(labels[indices['fit']] >= 0)&support[indices['fit']]]
    relevant = (np.arange(5)[None] == labels[:, None]).astype(np.int32)
    fit_permutations = np.asarray([np.random.default_rng(SEEDS[0]+int(i)).permutation(5) for i in train])
    flat_anchor = np.take_along_axis(anchor[train],fit_permutations,1).reshape(-1)
    flat_labels = np.take_along_axis(relevant[train],fit_permutations,1).reshape(-1)
    if not np.all(flat_labels.reshape(-1,5).sum(1) == 1):
        raise ValueError('Ranking group must have exactly one positive')
    frozen = dict(params=PARAMS, seeds=SEEDS, tree_counts=TREE_COUNTS,
        data_sha256=digest(source), source_summary=source_summary,
        script_sha256=digest(Path(__file__)), contract_sha256=digest(args.contract),
        lightgbm=lgb.__version__, rdkit=rdBase.rdkitVersion,
        partitions={name:len(value) for name,value in indices.items()},
        fit_reachable_supported=len(train), feature_names=names,
        evaluation_previously_exposed=True, official_validation_read=False, official_test_read=False)
    write_json(args.output/'freeze_manifest.json',frozen)
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'contract.md').write_bytes(args.contract.read_bytes())
    np.savez_compressed(args.output/'features.npz',**features, identifiers=np.asarray([q['identifier'] for q in p['inputs']]),
        partitions=parts,connectivities=groups,anchor=anchor,support=support)
    def rank(score, ix):
        return rank_scores([p['inputs'][i] for i in ix], [p['targets'][i] for i in ix], score)
    calibration_ix = indices['calibration']; evaluate_ix = indices['evaluation']
    cal_before = rank(anchor[calibration_ix],calibration_ix)
    scores = dict(base=base[evaluate_ix], coverage=anchor[evaluate_ix])
    # Standardization has a floor, so double the z channel itself for exact reference.
    scores['doubled_coverage'] = base[evaluate_ix]+2*(anchor[evaluate_ix]-base[evaluate_ix])
    results, models, selections = {}, {}, {}
    for arm in ARMS:
        folder = args.output/arm; folder.mkdir()
        feature = features[arm]
        flat = np.take_along_axis(feature[train],fit_permutations[:,:,None],1).reshape(-1,feature.shape[-1])
        dataset = lgb.Dataset(flat,label=flat_labels,group=np.full(len(train),5),
            init_score=flat_anchor,feature_name=names[arm],free_raw_data=False)
        models[arm] = []
        for seed in SEEDS:
            model = lgb.train(dict(PARAMS,seed=seed),dataset,num_boost_round=200)
            model.save_model(str(folder/f'seed{seed}.txt'))
            # Explicitly add init_score at inference; Booster.predict returns residual.
            models[arm].append(model)
        history = []
        for count in TREE_COUNTS:
            residual = np.mean([m.predict(feature[calibration_ix].reshape(-1,feature.shape[-1]),num_iteration=count)
                                for m in models[arm]],0).reshape(-1,5)
            residual[~support[calibration_ix]] = 0
            after = rank(anchor[calibration_ix]+residual,calibration_ix)
            history.append(dict(trees=count,**calibration(cal_before,after)))
        selected = select_count(history); selections[arm] = selected
        residuals = []
        for model in models[arm]:
            residual = model.predict(feature[evaluate_ix].reshape(-1,feature.shape[-1]),num_iteration=selected).reshape(-1,5) if selected else np.zeros((len(evaluate_ix),5))
            residual[~support[evaluate_ix]] = 0; residuals.append(residual)
        scores[arm] = anchor[evaluate_ix]+np.mean(residuals,0)
        results[arm] = dict(selected_trees=selected, calibration=history,
            per_seed=[endpoint(rank(anchor[evaluate_ix],evaluate_ix),rank(anchor[evaluate_ix]+r,evaluate_ix),groups[evaluate_ix]) for r in residuals],
            importance_gain={name:float(np.mean([m.feature_importance('gain',iteration=selected)[j] for m in models[arm]])) if selected else 0.
                             for j,name in enumerate(names[arm])})
        write_json(folder/'training_summary.json',results[arm])
        print(json.dumps(dict(arm=arm,selected_trees=selected,calibration=history)),flush=True)
    # Cached neural references have the identical evaluation identities.
    reference = np.load(args.data/'evaluation/per_query.npz')
    identifiers = np.asarray([p['inputs'][i]['identifier'] for i in evaluate_ix])
    if not np.array_equal(identifiers, reference['identifiers']):
        raise ValueError('Reference cohort drift')
    ranks = {name:rank(score,evaluate_ix) for name,score in scores.items()}
    for name in ('base','coverage'):
        if not np.array_equal(ranks[name],reference['ranks_'+name]):
            raise ValueError('Reference baseline replay failed')
    ranks.update({name:reference['ranks_'+name] for name in ('l0_only','listwise_l0')})
    # Recompute feature vectors after actual candidate-input permutation.
    permutation = np.asarray([4,2,0,3,1])
    subset = evaluate_ix[:128]
    perm_inputs = copy.deepcopy([p['inputs'][i] for i in subset])
    for query in perm_inputs:
        for key in ('candidate_smiles','candidate_global_rows'):
            query[key] = [query[key][j] for j in permutation]
    perm_features, _, _ = feature_sets(perm_inputs,base[subset][:,permutation],anchor[subset][:,permutation],
        p['valid'][subset][:,permutation],p['intensity'][subset])
    errors = {}
    for arm in ARMS:
        error = float(np.max(np.abs(perm_features[arm]-features[arm][subset][:,permutation])))
        if not np.allclose(perm_features[arm],features[arm][subset][:,permutation],atol=1e-5,rtol=1e-6):
            raise ValueError('Feature permutation failure')
        selected = selections[arm]
        if selected:
            x = features[arm][subset]; xp = perm_features[arm]
            original = np.mean([m.predict(x.reshape(-1,x.shape[-1]),num_iteration=selected) for m in models[arm]],0).reshape(-1,5)
            permuted = np.mean([m.predict(xp.reshape(-1,xp.shape[-1]),num_iteration=selected) for m in models[arm]],0).reshape(-1,5)
            score_error = float(np.max(np.abs(permuted-original[:,permutation])))
            if score_error > 1e-6:
                raise ValueError('Model permutation failure')
            loaded = lgb.Booster(model_file=str(args.output/arm/f'seed{SEEDS[0]}.txt'))
            if not np.array_equal(loaded.predict(x.reshape(-1,x.shape[-1]),num_iteration=selected),
                                  models[arm][0].predict(x.reshape(-1,x.shape[-1]),num_iteration=selected)):
                raise ValueError('Serialization replay failure')
        else:
            score_error = 0.
        errors[arm] = dict(feature_max_error=error,score_max_error=score_error)
    for value in ranks.values():
        if not np.array_equal(value<=5,ranks['base']<=5) or not np.array_equal(value<=20,ranks['base']<=20):
            raise ValueError('Shortlist changed')
    comparisons = {f'{arm}_vs_{baseline}':comparison(ranks[baseline],ranks[arm],groups[evaluate_ix],10000,20261010)
                   for arm in ARMS for baseline in ('coverage','doubled_coverage','l0_only')}
    comparisons.update({f'{arm}_vs_score_only':comparison(ranks['score_only'],ranks[arm],groups[evaluate_ix],10000,20261010)
                        for arm in ARMS[1:]})
    # Fixed exact-mass/adduct chemistry-channel control, keeping upstream anchor real.
    control_data = SimpleNamespace(payload=p,parts=parts,groups=groups)
    control_data.graph,control_data.fragment,_ = load_or_build_stores(args.data/'features/candidates',p['registry_smiles'])
    token = torch.load(str(args.data/'tokens.pt'),map_location='cpu',weights_only=False,mmap=True)
    control_data.tokens = token
    wrong, eligible = wrong_spectrum_channel(control_data,evaluate_ix)
    controlled = {}; control_scores = {}
    if len(eligible):
        donor = wrong['donor_rows'][eligible].numpy()
        wrong_inputs = copy.deepcopy([p['inputs'][i] for i in eligible])
        for query, row in zip(wrong_inputs,donor,strict=True):
            query['observed_spectrum'] = p['inputs'][row]['observed_spectrum']
        wrong_features,_,_ = feature_sets(wrong_inputs,base[eligible],anchor[eligible],wrong['valid'][eligible],p['intensity'][donor],
            chemistry=(chemistry[0][eligible],chemistry[1]))
        lookup = {row:i for i,row in enumerate(evaluate_ix)}
        positions = np.asarray([lookup[row] for row in eligible])
        for arm in ARMS:
            count = selections[arm]; value = wrong_features[arm]
            residual = np.mean([m.predict(value.reshape(-1,value.shape[-1]),num_iteration=count) for m in models[arm]],0).reshape(-1,5) if count else np.zeros((len(eligible),5))
            residual[~support[eligible]] = 0
            wrong_ranks = rank(anchor[eligible]+residual,eligible)
            controlled[arm] = comparison(wrong_ranks,ranks[arm][positions],groups[eligible],10000,20261010)
            control_scores[arm] = wrong_ranks
        np.savez_compressed(args.output/'wrong_spectrum.npz',recipient_rows=eligible,donor_rows=donor,**control_scores)
    decisions = {}
    for arm in ARMS[1:]:
        checks = dict(nonzero_selection=selections[arm]>0,
            damage_bounded=comparisons[f'{arm}_vs_coverage']['rank1_transition']['baseline_rank1_damage_rate_pct'] <= 8)
        for baseline in ('doubled_coverage','l0_only'):
            delta = comparisons[f'{arm}_vs_{baseline}']['delta_query_weighted_pp']
            checks['positive_r1_'+baseline] = delta['recall@1'] > 0
            checks['positive_mrr_'+baseline] = delta['mrr'] > 0
        checks['mrr_beyond_score_only'] = comparisons[f'{arm}_vs_score_only']['delta_query_weighted_pp']['mrr'] > 0
        decisions[arm] = dict(checks=checks,point_gate_pass=all(checks.values()))
    status = 'COMPONENT_SIGNAL_REQUIRES_FRESH_PROTOCOL' if any(d['point_gate_pass'] for d in decisions.values()) else 'STOP_THIS_CONFIGURATION'
    np.savez_compressed(args.output/'per_query.npz',identifiers=identifiers,connectivities=groups[evaluate_ix],
        **{f'ranks_{name}':value for name,value in ranks.items()},**{f'scores_{name}':value for name,value in scores.items()})
    report = dict(status=status, freeze=frozen, selections=selections, arms=results,
        metrics={name:endpoint(ranks['coverage'],value,groups[evaluate_ix]) for name,value in ranks.items()},
        comparisons=comparisons, decisions=decisions,
        equal_coverage=equal_coverage(p,evaluate_ix,scores),
        wrong_spectrum=dict(queries=len(eligible),real_minus_wrong=controlled,
            scope='Only evidence detail features replaced; real upstream+coverage anchor fixed.'),
        invariance=errors, baseline_replay_exact=True, shortlist_invariant=True,
        elapsed_seconds=time.time()-start, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        official_validation_read=False,official_test_read=False,evaluation_previously_exposed=True)
    write_json(args.output/'summary.json',report)
    print(json.dumps(dict(status=status,selections=selections,elapsed_seconds=report['elapsed_seconds'])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'outputs/mpfir_cross_candidate_official_20261010')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--contract',type=Path,default=ROOT/'docs/reranking/OFFICIAL_LIGHTGBM_PILOT_20261010.md')
    run(parser.parse_args())
