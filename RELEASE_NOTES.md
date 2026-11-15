# v1.0.0 — manuscript reproducibility materials

Release contents and tested scope.

This version accompanies *Auditing two-stage MS/MS molecular retrieval on
MassSpecGym v1.5: Reranking gains, candidate priors and spectrum dependence*.
It supplies the selected method/analysis source, configurations, frozen
predictions, supporting statistics, author-trained checkpoints and figure
masters used to audit the paper's findings.

Start with `verification-01.zip` and README.md for the NumPy-only workflow.
The other seven ZIPs provide historical supplementary evidence, figure masters,
trained models and separate validation-only coverage-intervention models.
The eight ZIPs total 3.27 GiB; see RELEASE_ASSETS.json and ASSET_SHA256SUMS.

Original software and author-trained model files: MIT. Author-owned original
predictions, statistical summaries and figure source data: CC BY 4.0. Third-party
components retain their own notices/terms; upstream pretrained encoders and full
embedding caches are excluded. See THIRD_PARTY.md and DATA_LICENSE.md.

The tested scope and limitations are recorded in VALIDATION.json. A successful
archived-prediction check does not establish a fresh end-to-end training or
inference run. CITATION.cff identifies the software version and authors.
