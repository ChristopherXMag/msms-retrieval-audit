#!/usr/bin/env python3
"""Fetch a pinned public TSV, verify it, and extract only official train rows.

The downloaded official file contains all folds. Extraction consults fold only
for non-train rows; no non-train identities, spectra, or metrics are analyzed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REVISION = '0dc77f670939718f333cd06e8a7b70fdfc5a39a5'
EXPECTED_SHA256 = '50cfdd1d22f79543c59555f9ce43c6893bd788a19a42b21fb4e2e3a54673c06a'
EXPECTED_SIZE = 261197365
FILE = 'data/MassSpecGym1.5.tsv'
DEFAULT_OUT = ROOT / f'data/source_massspecgym_v15_{REVISION}'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    api_url = f'https://huggingface.co/api/datasets/roman-bushuiev/MassSpecGym/tree/{REVISION}/data'
    with urllib.request.urlopen(api_url, timeout=45) as response:
        metadata = json.load(response)
    entry = next(item for item in metadata if item['path'] == FILE)
    assert entry['lfs']['oid'] == EXPECTED_SHA256 and entry['size'] == EXPECTED_SIZE
    save(output / 'remote_file_metadata.json', entry)
    url = f'https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/{REVISION}/{FILE}'
    target = output / 'MassSpecGym1.5.tsv'
    if target.exists():
        assert target.stat().st_size == EXPECTED_SIZE and digest(target) == EXPECTED_SHA256
        print('Existing pinned source verified.', flush=True)
    else:
        staging = target.with_suffix('.tsv.download')
        assert not staging.exists(), 'Remove or inspect incomplete staging before retry'
        downloaded = 0
        next_progress = 32 * 1024 * 1024
        with urllib.request.urlopen(url, timeout=60) as response, staging.open('xb') as handle:
            assert response.status == 200, 'Partial response must not become full source'
            while block := response.read(4 * 1024 * 1024):
                handle.write(block)
                downloaded += len(block)
                if downloaded >= next_progress:
                    print(f'Downloaded {downloaded/1024**2:.1f} / {EXPECTED_SIZE/1024**2:.1f} MiB', flush=True)
                    next_progress += 32 * 1024 * 1024
        assert staging.stat().st_size == EXPECTED_SIZE
        assert digest(staging) == EXPECTED_SHA256, 'HF LFS digest mismatch'
        staging.replace(target)
    train = output / 'official_train.raw.tsv'
    temporary = train.with_suffix('.tsv.extracting')
    rows = 0
    with target.open('rb') as source, temporary.open('wb') as destination:
        header_line = source.readline()
        header = header_line.decode('utf-8').rstrip('\r\n').split('\t')
        assert {'fold','precursor_mz','simulation_challenge','smiles'} <= set(header)
        fold_index = header.index('fold')
        destination.write(header_line)
        for line in source:
            fields = line.rstrip(b'\r\n').split(b'\t')
            assert len(fields) == len(header), 'Malformed TSV row'
            if fields[fold_index] != b'train':
                continue
            destination.write(line)
            rows += 1
    assert rows == 194119, rows
    temporary.replace(train)
    manifest = {
        'schema':'glmr_pinned_msgym_source_acquisition_v1',
        'fetched_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset':'roman-bushuiev/MassSpecGym', 'revision':REVISION,
        'source_url':url, 'source_file':str(target.relative_to(ROOT)),
        'sha256':digest(target), 'bytes':target.stat().st_size,
        'train_file':str(train.relative_to(ROOT)), 'train_file_sha256':digest(train),
        'train_bytes':train.stat().st_size, 'train_rows':rows,
        'columns':header, 'preservation':'train TSV lines copied byte-for-byte including precursor strings, all peaks and cohort flag',
        'cohort_field':'simulation_challenge',
        'source_field_available':'source' in header,
        'scope':'downloaded merged official train/val/test file; extraction checks fold and processes/writes train only; no held-out metrics or model evaluation',
        'script':str(Path(__file__).relative_to(ROOT)), 'script_sha256':digest(__file__)}
    save(output / 'acquisition_manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
