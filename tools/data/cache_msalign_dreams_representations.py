from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "DreaMS"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(REPO_ROOT / "outputs" / ".numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "outputs" / ".matplotlib_cache"))

from dreams.api import PreTrainedModel  # noqa: E402
from dreams.models.dreams.dreams import DreaMS as DreaMSModel  # noqa: E402
from dreams.models.heads.heads import ContrastiveHead  # noqa: E402
from dreams.utils import data as dreams_data  # noqa: E402
from dreams.utils import dformats  # noqa: E402
DEFAULT_V15_TSV = REPO_ROOT / "data/MassSpecGym1.5.tsv"


DEFAULT_DREAMS_CKPT = REPO_ROOT / "external/DreaMS/dreams/models/pretrained/embedding_model.ckpt"


class BareBackboneNoHead(torch.nn.Module):
    """Expose a raw DreaMS backbone through the ``no_head`` calling convention.

    ``ContrastiveHead(spec, no_head=True)`` is exactly ``backbone(spec, charge)[:, 0, :]``
    (see ``FineTuningHead.forward``, which sets ``precursor_emb=True``). Reproducing that
    here lets a backbone with no fine-tuning head — the raw SSL checkpoint, or one adapted
    by another task — produce a cache that is directly comparable to the released
    ``embedding_model.ckpt`` caches.
    """

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    @property
    def dtype(self) -> torch.dtype:
        return next(self.backbone.parameters()).dtype

    def forward(
        self, spec: torch.Tensor, charge: torch.Tensor | None = None, no_head: bool = False
    ) -> torch.Tensor:
        if not no_head:
            raise ValueError("A bare backbone has no fine-tuning head; use --representation no_head")
        return self.backbone(spec, charge)[:, 0, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache MSAlign DreaMS spectrum representations from MassSpecGym v1.5 metadata. "
            "Supports released ContrastiveHead output, no_head backbone precursor token, and concat."
        )
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out-npy", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--v15-tsv", type=Path, default=DEFAULT_V15_TSV)
    parser.add_argument("--dreams-ckpt", type=Path, default=DEFAULT_DREAMS_CKPT)
    parser.add_argument(
        "--expected-dreams-ckpt-sha256",
        type=str,
        default=None,
        help="Optional full SHA256 guard; abort before model loading when it does not match.",
    )
    parser.add_argument(
        "--representation",
        choices=["contrastive", "no_head", "concat_contrastive_no_head"],
        default="no_head",
    )
    parser.add_argument(
        "--backbone-source",
        choices=["contrastive_head", "bare_backbone"],
        default="contrastive_head",
        help=(
            "contrastive_head loads --dreams-ckpt as a ContrastiveHead (released default). "
            "bare_backbone loads --dreams-ckpt as a raw DreaMS backbone and takes the "
            "precursor token directly, which is what ContrastiveHead's no_head path returns; "
            "combine with --backbone-state-dict to cache a backbone adapted by another task."
        ),
    )
    parser.add_argument(
        "--backbone-state-dict",
        type=Path,
        default=None,
        help="Checkpoint holding a DreaMS backbone state dict; requires --backbone-source bare_backbone.",
    )
    parser.add_argument(
        "--backbone-state-key",
        type=str,
        default="last_backbone_state_dict",
        help="Key of the backbone state dict inside --backbone-state-dict.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--intensity-scale", type=float, default=100.0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(name)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_float_list(value: str) -> list[float]:
    if not value:
        return []
    return [float(x) for x in value.split(",") if x.strip()]


def load_v15_spectra(
    path: Path, keep_identifiers: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    spectra: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            identifier = row.get("identifier")
            if not identifier or (keep_identifiers is not None and identifier not in keep_identifiers):
                continue
            mzs = parse_float_list(row.get("mzs", ""))
            intensities = parse_float_list(row.get("intensities", ""))
            if len(mzs) != len(intensities):
                raise ValueError(
                    f"m/z and intensity length mismatch for {identifier}: "
                    f"{len(mzs)} vs {len(intensities)}"
                )
            spectra[identifier] = {
                "mzs": mzs,
                "intensities": intensities,
                "precursor_mz": float(row["precursor_mz"]) if row.get("precursor_mz") else None,
                "adduct": row.get("adduct"),
            }
    return spectra


def state_dict_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(np.ascontiguousarray(state[key].detach().cpu().float().numpy()).tobytes())
    return digest.hexdigest()[:16]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess_spectrum(
    row: dict[str, Any],
    spec_preproc: dreams_data.SpectrumPreprocessor,
    intensity_scale: float,
) -> torch.Tensor:
    peaks = np.asarray(
        list(zip(row["mzs"], [float(value) * intensity_scale for value in row["intensities"]])),
        dtype=np.float32,
    )
    spec = spec_preproc(peaks, prec_mz=row.get("precursor_mz"), high_form=True, augment=False)
    return torch.from_numpy(spec).float()


class SpectrumDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        records: list[dict[str, Any]],
        v15_spectra: dict[str, dict[str, Any]],
        spec_preproc: dreams_data.SpectrumPreprocessor,
        intensity_scale: float,
    ) -> None:
        self.specs: list[torch.Tensor] = []
        self.missing_identifiers: list[str] = []
        for record in records:
            identifier = str(record.get("identifier"))
            spectrum_row = v15_spectra.get(identifier)
            if spectrum_row is None:
                self.missing_identifiers.append(identifier)
                continue
            self.specs.append(preprocess_spectrum(spectrum_row, spec_preproc, intensity_scale))

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.specs[index]


@torch.inference_mode()
def encode_batch(model: torch.nn.Module, specs: torch.Tensor, representation: str) -> torch.Tensor:
    if representation == "contrastive":
        return model(specs)
    if representation == "no_head":
        return model(specs, no_head=True)
    if representation == "concat_contrastive_no_head":
        contrastive = model(specs)
        no_head = model(specs, no_head=True)
        return torch.cat([contrastive, no_head], dim=-1)
    raise ValueError(f"Unsupported representation: {representation}")


def main() -> None:
    args = parse_args()
    started = time.time()
    device = choose_device(args.device)
    checkpoint_sha256 = file_sha256(args.dreams_ckpt)
    if (
        args.expected_dreams_ckpt_sha256 is not None
        and checkpoint_sha256.lower() != args.expected_dreams_ckpt_sha256.lower()
    ):
        raise ValueError(
            "DreaMS checkpoint SHA256 mismatch: "
            f"expected={args.expected_dreams_ckpt_sha256.lower()} actual={checkpoint_sha256}"
        )

    records = load_jsonl(args.metadata)
    if args.max_records > 0:
        records = records[: args.max_records]
    requested_ids = {str(record["identifier"]) for record in records}
    v15_spectra = load_v15_spectra(args.v15_tsv, requested_ids)
    spec_preproc = dreams_data.SpectrumPreprocessor(
        dformat=dformats.DataFormatA(),
        n_highest_peaks=args.n_highest_peaks,
    )
    dataset = SpectrumDataset(records, v15_spectra, spec_preproc, args.intensity_scale)
    if dataset.missing_identifiers:
        raise ValueError(
            f"{len(dataset.missing_identifiers)} metadata rows are missing from v1.5 TSV; "
            f"first={dataset.missing_identifiers[0]!r}"
        )
    if len(dataset) == 0:
        raise ValueError(f"No usable spectra for {args.metadata}")

    backbone_provenance: dict[str, Any] = {
        "backbone_source": args.backbone_source,
        "backbone_state_dict": None,
        "backbone_state_key": None,
        "backbone_state_digest": None,
    }
    if args.backbone_source == "contrastive_head":
        if args.backbone_state_dict is not None:
            raise ValueError("--backbone-state-dict requires --backbone-source bare_backbone")
        wrapper = PreTrainedModel.from_ckpt(
            args.dreams_ckpt,
            ContrastiveHead,
            n_highest_peaks=args.n_highest_peaks,
        )
        model = wrapper.model.to(device).eval()
        loader_class = "ContrastiveHead"
    else:
        if args.representation != "no_head":
            raise ValueError("--backbone-source bare_backbone only supports --representation no_head")
        wrapper = PreTrainedModel.from_ckpt(
            args.dreams_ckpt,
            DreaMSModel,
            n_highest_peaks=args.n_highest_peaks,
        )
        backbone = wrapper.model
        if args.backbone_state_dict is not None:
            payload = torch.load(str(args.backbone_state_dict), map_location="cpu", weights_only=False)
            state = payload.get(args.backbone_state_key)
            if state is None:
                raise ValueError(
                    f"{args.backbone_state_dict} has no key {args.backbone_state_key!r}; "
                    f"available={sorted(k for k in payload if k.endswith('state_dict'))}"
                )
            backbone.load_state_dict(state, strict=True)
            backbone_provenance["backbone_state_dict"] = str(args.backbone_state_dict)
            backbone_provenance["backbone_state_key"] = args.backbone_state_key
            backbone_provenance["backbone_state_digest"] = state_dict_digest(state)
        model = BareBackboneNoHead(backbone).to(device).eval()
        loader_class = "DreaMSModel + BareBackboneNoHead"
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    chunks: list[np.ndarray] = []
    for batch in tqdm(loader, desc=f"Caching DreaMS {args.representation}"):
        specs = batch.to(device=device, dtype=model.dtype)
        embeddings = encode_batch(model, specs, args.representation)
        chunks.append(embeddings.float().cpu().numpy())
    arr = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(arr).all():
        raise ValueError("Nonfinite DreaMS cache")
    staging = args.out_npy.with_name(args.out_npy.stem + ".staging.npy")
    np.save(staging, arr)
    check = np.load(staging, mmap_mode="r")
    if check.shape != arr.shape or check.dtype != np.float32 or not np.isfinite(check).all():
        raise ValueError("DreaMS staging read-integrity failure")
    del check
    staging.replace(args.out_npy)
    summary = {
        "metadata": str(args.metadata),
        "metadata_sha256": file_sha256(args.metadata),
        "cache_sha256": file_sha256(args.out_npy),
        "requested_identifier_filter": True,
        "parsed_spectrum_rows": len(v15_spectra),
        "out_npy": str(args.out_npy),
        "v15_tsv": str(args.v15_tsv),
        "dreams_ckpt": str(args.dreams_ckpt),
        "dreams_ckpt_sha256": checkpoint_sha256,
        "loader_class": loader_class,
        **backbone_provenance,
        "representation": args.representation,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "n_highest_peaks": args.n_highest_peaks,
        "intensity_scale": args.intensity_scale,
        "records_seen": len(records),
        "usable_records": len(dataset),
        "batch_size": args.batch_size,
        "device": str(device),
        "elapsed_seconds": round(time.time() - started, 3),
        "checkpoint_structure": {
            "backbone_class": type(model.backbone).__name__ if hasattr(model, "backbone") else None,
            "d_model": int(getattr(model.backbone, "d_model", arr.shape[1])),
            "n_layers": int(getattr(model.backbone, "n_layers", 0)),
            "n_heads": int(getattr(model.backbone, "n_heads", 0)),
            "head": str(getattr(model, "head", None)),
        },
    }
    out_json = args.out_json or args.out_npy.with_suffix(".summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
