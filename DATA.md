# Data and artifact groups

Large artifacts are distributed as separate assets of the corresponding tagged
GitHub Release. `ARTIFACT_MANIFEST.json` records original-source and exported-file
SHA256 values and export methods. Export records describe checks performed at
export time; later checkpoint-loading results are recorded in VALIDATION.json.
Unpack approved assets at the repository root, preserving their relative paths.
The exact eight-file inventory, bytes and digests are in RELEASE_ASSETS.json and
ASSET_SHA256SUMS. `verification-01.zip` alone supports the lightweight prediction
and archived-analysis checks; model and figure groups can be downloaded separately.

| Group | Scope |
| --- | --- |
| verification | Primary validation/test contribution arrays, candidate/query mappings, controls, external scores, November analyses, R6-lite statistics, MCES and selected local evidence |
| supplementary | Selected historical summaries and rank files actually supporting supplementary claims; these are not official benchmark rows |
| figures | Final vector figures, editable author masters and existing figure-source bundles |
| models | The 15 selected final neural models and six tree models; no optimizer-last duplicates or upstream pretrained encoders |
| coverage_models | Four R6-lite training-intervention checkpoints; separate from the deployed system |

Original research path names are retained for exact traceability and compatibility
with source readers. Public descriptions use the terminology in the manuscript.
The manifest distinguishes original hashes from metadata-sanitized copies.
Non-finite metadata such as an open bin endpoint is encoded explicitly as
`"Infinity"`, `"-Infinity"` or `"NaN"` in standard JSON; finite result values
are unchanged. The original source hash preserves the pre-export identity.
Tensor-storage ZIP members remain byte-identical during checkpoint export;
PyTorch 2.2.1 CPU mmap loading is verified in VALIDATION.json; full-system
inference remains untested on the release draft.

## Original data and upstream model reconstruction

MassSpecGym v1.5 source revision:
`0dc77f670939718f333cd06e8a7b70fdfc5a39a5` at
https://huggingface.co/datasets/roman-bushuiev/MassSpecGym .
The pinned raw TSV SHA256 is
`50cfdd1d22f79543c59555f9ce43c6893bd788a19a42b21fb4e2e3a54673c06a`.
Retain official split membership, candidate order/counts, canonical-isomeric
identity, connectivity recomputation and the documented no-GT-insertion policy.
Training may move an already-present GT to slot zero; evaluation tracks its
actual slot and does not infer correctness from order.

The pinned dataset card declares MIT. Retain its source attribution and terms
for original dataset content; this is separate from the CC BY 4.0 license for
the authors' new predictions, summaries and annotations.

Data-safe DreaMS checkpoint SHA256:
`d9b4e8b920ff367370c5a3fa154eaa951f90bafa58d5224a594424d9c54b1832`.
Follow its official upstream distribution and exact checkpoint provenance.
Ordinary DreaMS is a distinct historical component, not a substitute.
MolDeBERTa uses the revision and terms in THIRD_PARTY.md. Upstream pretrained
weights, the full dataset, the full candidate JSON and large embedding caches
are not automatically included or relicensed.

## Full Stage 1 input layout

The configured training helper expects:

```text
artifacts/metadata/{train,val}.metadata.jsonl
artifacts/spectra/{train,val}.no_head.npy
artifacts/sccf/manifest.json
artifacts/candidate_index/{manifest.json,identifiers.txt,indptr.npy,indices.bin}
artifacts/heldout_connectivity_exclusion/{manifest.json,excluded_rows.npy}
artifacts/molecule_rows/full_smiles.txt
```

The SCCF cache has 7,220,331 molecule rows and uses RDKit 2025.09.6, chiral
Morgan/Atom Pair, log1p count blocks and per-block L2 normalization. The source
builder is `tools/data/build_msalign_sparse_fingerprint_cache.py`. The exact
row order, feature manifest and candidate-index source must be verified before
training. A complete small-machine rebuild has not been tested.

## Redistribution review

The author selected CC BY 4.0 for original predictions, statistics and figure
source data. Third-party identities, observed spectra and model-related material retain
applicable upstream terms. The artifact manifest records per-file provenance
and export methods; THIRD_PARTY.md identifies notices and excluded upstream
weights/caches. The original-output grant covers only rights the authors hold.
