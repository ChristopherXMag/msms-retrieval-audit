# Third-party provenance and notices

The project MIT license covers author-owned original software. Retain upstream
notices for source ports and check data/model terms separately.

The selected checkpoint files contain the authors' trained projection/reranker
parameters and training state; upstream DreaMS and MolDeBERTa pretrained
encoders are not bundled. The authors' software-license grant applies to their
own trainable model files and tree models. It does not grant rights to upstream
encoder weights, embeddings or any other third-party component. Upstream
input-model terms remain relevant when reconstructing those workflows.

| Component | Pin / provenance | Treatment |
| --- | --- | --- |
| MSAlign and EmbCos source | `c2ef425b68749052a17b9cdd5f5f11709322654e`, https://github.com/ChristopherXMag/MSAlign-NeurIPS2026 | MIT notice retained under third_party/ |
| DreaMS source | `dbec3a0b514a99e5056cfccde4559fda8cfe8129`, https://github.com/pluskal-lab/DreaMS | MIT source notice retained; pretrained checkpoint provenance/terms remain separate |
| MAGMa / ms-pred source | `cd616f629a82040b950fbf8f0a983cc70ce1cf3b`, https://github.com/coleygroup/ms-pred | Exact upstream notice retained; source is required at external/ms-pred for imported fragmentation helpers |
| MolDeBERTa | SaeedLab/MolDeBERTa-base-123M-mtr, revision `7d06a5f39098837a6f33ed095e2fa7706e30f8b9` | Saved upstream card says CC BY-NC-ND 4.0; do not apply MIT or automatically redistribute its weights/embeddings |
| MassSpecGym | Pinned v1.5 revision in DATA.md; its pinned public card declares MIT | Retain upstream attribution/terms for source records; original added predictions/statistics use CC BY 4.0 |
| ChemDraw / Arial | Licensed author-side artwork dependencies | Software/font binaries are excluded; existing editable/vector outputs are supplied for review |

To reproduce feature extraction, acquire the pinned upstream source trees in
`external/DreaMS` and `external/ms-pred`; install dependencies in the appropriate
isolated environment. Model download paths and hashes must match the recorded
configuration. Acquire upstream components separately from their pinned official distributions;
source-code licenses do not replace pretrained-model terms.

Optional/historical imports in the scientific source do not imply that every
upstream teacher or global model is needed for prediction-only verification.
The included source allowlist records the actual local dependency provenance.
