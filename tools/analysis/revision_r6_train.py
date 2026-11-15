#!/usr/bin/env python3
"""Run the four fixed R6-lite fits sequentially, with host-memory monitoring."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'outputs/revision_supplement_20261111/R6_lite/validation'
SEEDS = (20260621, 20260622)
ARMS = ('targeted','size_matched')


def memory_state():
    base = Path('/sys/fs/cgroup')
    stat = dict(line.split() for line in (base/'memory.stat').read_text().splitlines())
    limit = int((base/'memory.max').read_text())
    current = int((base/'memory.current').read_text())
    host = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return dict(time=time.time(), current=current, limit=limit, anon=int(stat['anon']),
        available=limit-current+int(stat['inactive_file']),
        swap=int((base/'memory.swap.current').read_text()),
        host_available=int(host['MemAvailable'].split()[0])*1024,
        disk_free=shutil.disk_usage(ROOT).free)


def check_memory(state):
    assert state['available'] >= 32*2**30, ('cgroup reserve',state)
    assert state['host_available'] >= 32*2**30, ('host reserve',state)
    assert state['swap'] == 0, ('swap growth',state)
    assert state['disk_free'] >= 5*2**30, ('disk reserve',state)


def worker(seed,arm):
    from tools.evaluation.official_s7_common import install_no_test_guard, gpu_preflight, write, sha
    install_no_test_guard()
    import torch
    from tools.stage1 import train_msalign_projection_smoke as trainer
    from tools.stage1.run_stage1_submission_0926 import train_command
    directory = OUT/f'seed{seed}'/arm
    from tools.analysis.revision_r6_memory import install
    install(directory)
    gpu_preflight(directory,'before_fit')
    check_memory(memory_state())
    paths = {'summary':directory/'train_summary.json','checkpoint':directory/'checkpoint.pt'}
    original = train_command('full_r2_block_l2',seed,'cuda',paths)
    command = list(original)
    command[command.index('--train-metadata')+1] = str(OUT/'gate'/f'{arm}.metadata.jsonl')
    command[command.index('--train-dreams')+1] = str(OUT/'gate'/f'{arm}.dreams.npy')
    # Storage-only deviation required by the cgroup budget; exact feature values unchanged.
    command.remove('--preload-sparse-features')
    assert not paths['checkpoint'].exists(), 'Never overwrite an existing fit'
    write(directory/'command.json',dict(argv=command,original_argv=original,
        storage_deviation='Read-only mmap; bounded residency with scoped advisory eviction; no full preload; num_workers=0',
        gate_sha256=sha(OUT/'gate/summary.json'),source_sha256=sha(Path(__file__)),
        torch_version=torch.__version__,cuda_version=torch.version.cuda,
        float32_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        formal_checkpoint='model_state_dict after exactly epoch 1; no validation epoch selection'))
    sys.argv = command[1:]
    trainer.main()
    state = torch.load(str(paths['checkpoint']),map_location='cpu',weights_only=False,mmap=True)
    assert state['last_epoch'] == 1 and [row['epoch'] for row in state['history'] if 'train' in row] == [1]
    assert state['args']['seed'] == seed and state['args']['epochs'] == 1
    write(directory/'fit_complete.json',dict(status='COMPLETE',seed=seed,arm=arm,
        checkpoint_sha256=sha(paths['checkpoint']),summary_sha256=sha(paths['summary']),
        memory=memory_state(),time=time.time()))


def run(resume=False):
    # Parent intentionally avoids importing CUDA: the bound worker performs each fresh CUDA check.
    from tools.evaluation.official_s7_common import read,write,sha
    assert read(OUT/'gate/summary.json')['status'] == 'PASS', 'CPU gate required'
    assert resume or not (OUT/'training_execution.json').exists(), 'Execution already exists'
    provenance=OUT/'provenance';provenance.mkdir(parents=True,exist_ok=True)
    if not resume:
        shutil.copyfile(ROOT/'to_be_run.md', provenance/'contract_execution.md')
    assert sha(provenance/'contract_execution.md') == next(x['sha256'] for x in read(OUT/'gate/summary.json')['inputs'] if x['path']=='to_be_run.md')
    bindings={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'tools/stage1/train_msalign_projection_smoke.py',
        ROOT/'tools/stage1/run_stage1_submission_0926.py',ROOT/'glmr/sparse_features.py',ROOT/'glmr/candidate_indices.py',ROOT/'tools/analysis/revision_r6_memory.py',
        ROOT/'outputs/stage1_submission_0926/assets/heldout_connectivity_exclusion/manifest.json']}
    if resume:
        execution=read(OUT/'training_execution.json')
        assert execution['status'] in ('STOP_FIT_FAILED','STOP_POSTFIT_CHECK_FAILED','STOP_RESOURCE_OR_EXECUTION_FAILURE')
        for path,digest in execution['source_bindings'].items():
            if path!=str(Path(__file__).relative_to(ROOT)):
                assert sha(ROOT/path)==digest, path
        execution['recovery']=dict(time=time.time(),source_bindings=bindings,
            reason='PyTorch mmap checkpoint loader requires str(path); first epoch and checkpoint completed before receipt failure; no retraining')
        execution['status']='RUNNING'
    else:
        execution=dict(status='RUNNING',started=time.time(),fixed_seeds=list(SEEDS),arms=list(ARMS),
            gate_sha256=sha(OUT/'gate/summary.json'),source_bindings=bindings,completed=[])
    write(OUT/'training_execution.json',execution)
    for seed in SEEDS:
        for arm in ARMS:
            directory=OUT/f'seed{seed}'/arm;directory.mkdir(parents=True,exist_ok=True)
            if resume and (directory/'checkpoint.pt').exists():
                import torch
                torch.set_num_threads(1)
                state=torch.load(str(directory/'checkpoint.pt'),map_location='cpu',weights_only=False,mmap=True)
                summary=read(directory/'train_summary.json')
                assert state['last_epoch']==1 and [row['epoch'] for row in state['history'] if 'train' in row] == [1]
                assert state['args']['seed']==seed and state['args']['epochs']==1
                assert summary['args']['seed']==seed and summary['args']['epochs']==1
                assert summary['train_sampling_protocol']['planned_row_exposures']==176615
                for value in state['model_state_dict'].values():
                    assert torch.isfinite(value).all()
                if not (directory/'fit_complete.json').exists():
                    write(directory/'fit_complete.json',dict(status='COMPLETE',seed=seed,arm=arm,
                        checkpoint_sha256=sha(directory/'checkpoint.pt'),summary_sha256=sha(directory/'train_summary.json'),
                        recovered_postfit_receipt=True,time=time.time(),memory=memory_state()))
                else:
                    assert sha(directory/'checkpoint.pt')==read(directory/'fit_complete.json')['checkpoint_sha256']
                if not any(item['seed']==seed and item['arm']==arm for item in execution['completed']):
                    samples=read(directory/'resource_monitor.json')['samples']
                    execution['completed'].append(dict(seed=seed,arm=arm,
                        wall_seconds=samples[-1]['time']-samples[0]['time'],wall_seconds_from_resource_monitor=True,
                        peak_worker_rss_bytes=max(x.get('worker_rss_bytes',0) for x in samples),
                        minimum_container_available_bytes=min(x['available'] for x in samples),
                        minimum_host_available_bytes=min(x['host_available'] for x in samples),
                        max_swap_bytes=max(x['swap'] for x in samples)))
                    write(OUT/'training_execution.json',execution)
                print('RECOVERED_COMPLETED_FIT_WITHOUT_RETRAINING',seed,arm,flush=True)
                del state
                continue
            from tools.analysis.revision_r6_memory import trim_store
            cache_event=trim_store()
            write(directory/'cache_before_fit.json',cache_event)
            check_memory(memory_state())
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True).strip()
            assert os.environ['CUDA_VISIBLE_DEVICES'] in gpu
            row=next(line for line in gpu.splitlines() if os.environ['CUDA_VISIBLE_DEVICES'] in line)
            assert int(row.split(',')[2]) < 512, ('GPU was claimed',row)
            print('FIT_START',seed,arm,time.strftime('%H:%M:%S'),flush=True)
            start=time.time();samples=[]
            with (directory/'fit.log').open('w') as log:
                process=subprocess.Popen([sys.executable,str(Path(__file__)),'--worker',str(seed),arm],
                    stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                try:
                    while True:
                        state=memory_state();check_memory(state)
                        status=Path(f'/proc/{process.pid}/status')
                        if status.exists():
                            fields=dict(line.split(':',1) for line in status.read_text().splitlines() if ':' in line)
                            state['worker_rss_bytes']=int(fields.get('VmRSS','0 kB').split()[0])*1024
                            state['worker_threads']=int(fields.get('Threads','0').strip())
                        samples.append(state)
                        write(directory/'resource_monitor.json',dict(samples=samples))
                        try:
                            code=process.wait(timeout=15)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                except BaseException:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    execution.update(status='STOP_RESOURCE_OR_EXECUTION_FAILURE',failed_seed=seed,failed_arm=arm)
                    write(OUT/'training_execution.json',execution)
                    raise
            if code != 0:
                execution.update(status='STOP_FIT_FAILED',failed_seed=seed,failed_arm=arm,exit_code=code)
                write(OUT/'training_execution.json',execution)
                raise RuntimeError(f'Fit failed: {seed} {arm}; see {directory}/fit.log')
            item=dict(seed=seed,arm=arm,wall_seconds=time.time()-start,
                peak_worker_rss_bytes=max(x.get('worker_rss_bytes',0) for x in samples),
                minimum_container_available_bytes=min(x['available'] for x in samples),
                minimum_host_available_bytes=min(x['host_available'] for x in samples),
                max_swap_bytes=max(x['swap'] for x in samples))
            execution['completed'].append(item);write(OUT/'training_execution.json',execution)
            print('FIT_COMPLETE',json.dumps(item),flush=True)
    execution.update(status='COMPLETE',finished=time.time())
    write(OUT/'training_execution.json',execution)
    print('FOUR_FITS_COMPLETE',flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',nargs=2,metavar=('SEED','ARM'))
    parser.add_argument('--resume',action='store_true',help='Recover completed checkpoint after a post-fit receipt failure; never retrain completed fits')
    args=parser.parse_args()
    if args.worker:
        seed,arm=int(args.worker[0]),args.worker[1]
        assert seed in SEEDS and arm in ARMS
        worker(seed,arm)
    else:run(args.resume)
