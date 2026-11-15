# Reproduction paths and tested scope

## Verification snapshot

The 15 November 2026 checks are summarized in [VALIDATION.json](VALIDATION.json).
A fresh NumPy-only environment verified the primary estimates and the R1–R6
archived analyses. All selected neural checkpoints load through CPU mmap in
PyTorch 2.2.1, with model tensor shapes, bytes and hashes checked; six tree
models load and return finite synthetic predictions. These checks do not
exercise the complete inference pipeline. For CPU mmap on this PyTorch
version, pass a string filename to `torch.load`, not a `Path` object.

## Prediction-only verification

Use Python 3.11 and `requirements-verify.txt`. Unpack the verification assets
at the repository root, retaining the relative `outputs/` paths in the manifest.
Run `scripts/verify_predictions.py` as described in the README. It imports only
NumPy and the standard library. It checks identities, row membership, value
ranges, both weighting schemes and agreement with stored primary estimates.
The bootstrap resamples whole connectivity groups with paired rows in bounded
batches. Its optional intervals are independently seeded reproductions; the
original summaries retain each original analysis's exact interval conventions.
No model fitting, spectrum encoding or official-test inference occurs.

Further analyses are provided in their original scientific implementations.
Their historical memory guards, output locks and source bindings are preserved;
copying the source does not make those entry points safe on a small machine.

## Stage 1 training and validation

The frozen SCCF contract uses official v1.5 train/validation, data-safe DreaMS,
radius-2 Morgan/Atom Pair/path/MACCS features, canonical-isomeric molecule rows,
GT plus 127 random candidates, fixed temperature 0.07, one epoch, and seeds
20260621/20260622/20260623. Do not substitute historical custom/ordinary-DreaMS
inputs. Held-out-connectivity identity exclusion remains transductive and must
remain disclosed. Controls retain physical candidate permutation and uniform ties.

The training/validation helper is portable at the Python-interpreter level:

```bash
python scripts/run_stage1.py --phase both --dry-run
python scripts/run_stage1.py --seed 20260621 --phase both --device cuda
python scripts/run_blind_controls.py --dry-run
```

Actual runs require the complete configured metadata, spectrum features,
molecule cache, candidate index and negative exclusion; see `configs/official_v15_sccf.json`
and DATA.md. The recorded helper preloads sparse features. It must not be run
on the present 2 GiB preparation environment. Use the high-memory GPU recipe
only after verifying the host resources; the source retains its operational guards.
A fresh environment installation or retraining is not claimed by this draft.

## Full-system inference and external comparisons

Scientific source for pair, coverage, mass-feasibility, trees and support fallback
is included. The frozen endpoint uses top-20 feasibility when supported, otherwise
complete top-5 feasibility when supported, otherwise top-20 pair plus coverage.
The S6 endpoint/feature-order files in the verification group define its exact
policy. Preserve the documented floating-point, pair-padding and candidate-tail
conventions. Original full-test transaction tools are provenance-bearing research
entry points with exclusive-run guards, not a general overwrite-and-rerun command.
A standalone full-system inference convenience CLI is not claimed in this draft.

MSAlign and EmbCos use their own input representations, metadata and training
budgets. `configs/external_matched.json` records those differences. Fresh MSAlign
feature extraction needs the separate pinned Transformers/tokenizers environment;
SCCF and fragment chemistry use different RDKit versions. Exact cache rebuilds
must use their owning environment rather than changing one shared installation.

Reference executions used an RTX 4080 SUPER with 32,760 MiB and a high-memory
Linux container. This is execution provenance, not a measured minimum hardware
requirement. Training/inference have not been exercised on the release draft.

## Figures and supplementary analyses

`tools/analysis/plot_jcheminf_figures.py` reads archived November summaries.
Run from the repository root, with its plotting environment and locally licensed
Arial fonts, using `--out-dir results/figures`. Exact ChemDraw depiction uses
Windows Python 3.10 and a licensed ChemScript installation. Existing PDF/SVG/CDXML
masters and frozen case data permit checking the scientific content independently
of proprietary artwork tooling. Font binaries and ChemDraw are not distributed.

MCES environment pins are separate from the lightweight verifier. Archived MCES
values can be summarized without running the structural solver. Recomputing
pair distances requires the solver environment and its recorded certificates.
The R6-lite group is validation-only: two arms and two paired seeds, with the
bounded-cohort and training-seed-uncertainty limitations retained.
