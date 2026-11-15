# MS/MS retrieval audit

Reproducibility materials for *Auditing two-stage MS/MS molecular retrieval on
MassSpecGym v1.5: Reranking gains, candidate priors and spectrum dependence*.

Repository: https://github.com/ChristopherXMag/msms-retrieval-audit

Version 1.0.0 packages selected source, configurations and verification tools.
The eight associated ZIPs are catalogued in RELEASE_ASSETS.json; download them
from the corresponding tagged GitHub Release. Scope and executed checks are
recorded in VALIDATION.json. Third-party components retain their own terms.

## Start with archived predictions

The lightweight checking path uses Python 3.11 and NumPy and reads frozen
predictions; it does not import a model framework or run inference. Download
`verification-01.zip` from this repository's tagged Release. Its SHA256 and size
are recorded in [RELEASE_ASSETS.json](RELEASE_ASSETS.json). From the repository
root, verify the downloaded ZIP before extracting it (replace the example path):

```bash
sha256sum /path/to/downloads/verification-01.zip
# Compare with ASSET_SHA256SUMS, then extract with paths preserved:
unzip /path/to/downloads/verification-01.zip -d .
python -m pip install -r requirements-verify.txt
python scripts/check_release.py --artifacts --group verification
python scripts/verify_predictions.py --data-root . --out-dir results/metrics
```

Only the approximately 308 MiB verification ZIP is needed for these metric
checks. Model ZIPs are optional for this workflow. If all eight archives have
been downloaded to one directory, run `sha256sum -c /path/to/repository/ASSET_SHA256SUMS`
from that download directory. Unpacked data/model/figure directories are excluded
from Git; their payloads belong in Release attachments.

The checker recomputes all 15 validation and official-test row estimates with
query and connectivity weighting and compares them with the archived summaries.
Optional independently seeded paired whole-connectivity intervals:

```bash
python scripts/verify_predictions.py --data-root . --out-dir results/intervals \
  --bootstrap 10000 --seed 20261017
```

Additional archived-analysis checks use the same lightweight environment:

```bash
python scripts/verify_analyses.py --data-root . --output results/analysis_checks.json
```

The dated tested scope and unexecuted operations are in [VALIDATION.json](VALIDATION.json).
Use a fresh output directory each time. Per-query contributions are fractions;
reported metrics are percentages, including MRR multiplied by 100.

## Reproducibility scope

Three different capabilities are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md):
retraining, inference with frozen checkpoints, and checking archived predictions.
Only the prediction workflow is designed for a small CPU environment. Source
for the scientific methods is included under `glmr/`, `tools/`, and
`train_spec_encoder/`. Original completed-stage guards are preserved.
Fresh training and full inference are not validated by a successful metric check.

See [DATA.md](DATA.md) for the artifact groups, [SOURCE_MAP.md](SOURCE_MAP.md)
for reported findings and their inputs, and [THIRD_PARTY.md](THIRD_PARTY.md)
for external dependencies. Large files belong in Release assets rather than
ordinary source history. [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) records
remaining validation and publication tasks.

## Licenses and attribution

Original software: MIT, see [LICENSE](LICENSE).
Author-owned predictions, statistical summaries and figure source data:
CC BY 4.0, see [DATA_LICENSE.md](DATA_LICENSE.md).
Third-party datasets, pretrained weights, model-derived material and font files
retain their own terms; they are not automatically relicensed by these choices.
Citation metadata is in [CITATION.cff](CITATION.cff).
