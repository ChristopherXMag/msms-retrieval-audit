#!/usr/bin/env python3
"""Check prepared source/artifact hashes using the standard library only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(256*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',action='store_true')
    p.add_argument('--group',choices=('verification','supplementary','figures','models','coverage_models'),help='Limit --artifacts to one downloaded group')
    args=p.parse_args()
    if args.group and not args.artifacts:p.error('--group requires --artifacts')
    manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text());count=0
    for name,record in manifest['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT):raise ValueError('Manifest path escapes repository')
        if digest(path)!=record['public_sha256']:raise ValueError('Source mismatch: '+name)
        if path.suffix=='.py':compile(path.read_text(),str(path),'exec')
        count+=1
    artifacts=0
    if args.artifacts:
        for name,record in json.loads((ROOT/'ARTIFACT_MANIFEST.json').read_text())['files'].items():
            if args.group and record['group']!=args.group:continue
            path=(ROOT/name).resolve()
            if not path.is_relative_to(ROOT):raise ValueError('Manifest path escapes repository')
            if digest(path)!=record['public_sha256']:raise ValueError('Artifact mismatch: '+name)
            artifacts+=1
    print(json.dumps({'source_files_verified':count,'artifact_files_verified':artifacts,'model_deserialization':False}))


if __name__=='__main__':main()
