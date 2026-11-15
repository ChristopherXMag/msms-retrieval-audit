"""Release clean project file pages after large hash/inference stages; preserve bytes."""
from pathlib import Path
import os
from tools.evaluation.official_s7_common import ROOT,is_test_data

def advise_clean(path):
    path=Path(path).resolve();assert path.is_relative_to(ROOT)
    if path.stat().st_size<64*2**20:return
    with path.open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)

def reclaim_inputs(manifest,include_test=None):
    paths={ROOT/p for p in manifest['files']}
    if include_test is not None:
        include_test=Path(include_test).resolve();assert include_test.is_relative_to(ROOT/'outputs')
        paths.update(p for p in include_test.rglob('*') if p.is_file())
    count=0
    for path in paths:
        if path.is_file() and path.stat().st_size>=64*2**20:advise_clean(path);count+=1
    return count
