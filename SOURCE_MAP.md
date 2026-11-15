# Reported findings and reproduction inputs

| Finding | Main code | Frozen data group |
| --- | --- | --- |
| Stage 1 SCCF and ensemble | tools/stage1/train_msalign_projection_smoke.py; export_stage1_candidate_scores.py | Validation/test contributions, Stage 1 scores, candidate identities |
| Pair reranking and support-gated mass feasibility | tools/reranking/run_official_lightgbm_top20.py; replay_official_d3_support_fallback.py | S6 endpoint/feature order, S7 real/control scores and route masks |
| MSAlign / EmbCos | glmr/msalign_external.py; glmr/embcos.py; tools/stage1/run_msalign_official.py; run_embcos_official.py | Selected comparator checkpoints and all seed score/rank arrays |
| Controls and candidate priors | tools/reranking/audit_official_d3_full_system.py; tools/audits/fit_frozen_candidate_prior.py | Zero/mean/wrong-spectrum/prior contributions, fixed donor assignments |
| Query vs connectivity weighting | audit_metrics.py; tools/analysis/revision_supplement.py | Per-query fractions, connectivity keys and original paired summaries |
| Clustering and training proximity | tools/analysis/revision_supplement.py; revision_proximity.py | R1a/R1b archives, fixed bins and analysis definitions |
| Mass prior and CE strata | tools/analysis/revision_mass_prior.py; revision_supplement.py | R4/R5a validation/test archives and calibration metadata |
| Coverage intervention | tools/analysis/revision_r6_*.py | Validation-only paired-arm checkpoints, gate and per-query/bootstrap data |
| MCES and latency | tools/analysis/supplement_official_s7.py; complete_official_s7_mces.py; tools/evaluation/benchmark_official_validation_latency.py | Archived distances, completion summaries and validation timing observations |
| Figures 2–4 | tools/analysis/plot_jcheminf_figures.py | R3/R2/R1a/R1b summaries and existing vector figures |
| Figure 5 | tools/analysis/figure5_validation_cases.py; plot_figure5_validation_cases.py | Frozen cases, eligibility/selection metadata, masks, SVG/CDXML/PDF masters |
| Local/negative evidence | MPFIR, cross-candidate, CFM and geometry tools | Separate bounded-pilot summaries and relevant archived predictions |
| Historical target/coverage screen | Historical Stage 1 and coverage-sweep tools | Separately labelled custom/ordinary-DreaMS summaries and ranks |

These source names are provenance identifiers. Historical and official results,
method seed means and actual ensembles, descriptive analyses and interventions
must retain their separate scopes. A source file being supplied does not claim
that its large-job execution has been retested on this draft.
