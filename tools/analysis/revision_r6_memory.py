"""Bound residency of the exact read-only R6 sparse feature cache.

Only file mapping/page-cache advice changes. No feature values, RNG state,
training schedule or candidate pools change. No system-wide cache flushes.
"""
from __future__ import annotations
import json
import mmap
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
BLOCKS=('count_morgan','count_atom_pair','rdkit_path_count','maccs')
FP=ROOT/'outputs/stage1_submission_0926/assets/full_sparse_direct/manifest_historical_projection_order.json'


def state():
    base=Path('/sys/fs/cgroup')
    stat=dict(line.split() for line in (base/'memory.stat').read_text().splitlines())
    current=int((base/'memory.current').read_text());limit=int((base/'memory.max').read_text())
    return dict(time=time.time(),current=current,anon=int(stat['anon']),
        inactive_file=int(stat['inactive_file']),available=limit-current+int(stat['inactive_file']),
        swap=int((base/'memory.swap.current').read_text()))


def advise_files():
    manifest=json.loads(FP.read_text());paths=[]
    for block in BLOCKS:
        paths.extend(FP.parent/manifest['features'][block][key] for key in ('indices','values','indptr'))
    paths.extend(FP.parent/manifest['projection_order'][key] for key in ('indices','sources'))
    for path in paths:
        path=path.resolve();assert path.is_relative_to(ROOT)
        fd=os.open(path,os.O_RDONLY)
        try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
        finally:os.close(fd)


def trim_store(store=None):
    before=state()
    if store is not None:
        seen=set()
        for cache in store.sparse_caches:
            arrays=[cache.row_sources,cache.row_indices]
            for block in BLOCKS:
                matrix=cache.features.get(block)
                if matrix is not None:arrays.extend([matrix.indptr,matrix.indices,matrix.values])
            for array in arrays:
                if array is not None:
                    assert not array.flags.writeable
                    mapping=array._mmap
                    if id(mapping) not in seen:
                        mapping.madvise(mmap.MADV_DONTNEED);seen.add(id(mapping))
    advise_files()
    return dict(before=before,after=state(),method='read-only mmap MADV_DONTNEED then scoped POSIX_FADV_DONTNEED')


def install(directory):
    from tools.stage1.train_msalign_projection_smoke import MoleculeEmbeddingStore
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    original=MoleculeEmbeddingStore.gather
    def bounded_gather(self,indices):
        if state()['available']<40*2**30:
            event=trim_store(self)
            with (directory/'cache_residency_events.jsonl').open('a') as handle:
                handle.write(json.dumps(event)+'\n')
            assert event['after']['available']>=32*2**30 and event['after']['swap']==0
        return original(self,indices)
    MoleculeEmbeddingStore.gather=bounded_gather
