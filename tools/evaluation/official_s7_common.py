"""Shared S7 inference semantics. Preparing on train/val never opens test files."""
from __future__ import annotations
import hashlib,json,os,re,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'outputs/official_s7_preparation_20261017'
S6=ROOT/'outputs/official_final_freeze_20261017'
CONTROLS=ROOT/'outputs/official_d3_full_system_controls_20261017'
E1=ROOT/'outputs/msalign_official_e1_20261014'
E2=ROOT/'outputs/embcos_official_e2_20261014'
SEEDS=(20260621,20260622,20260623)
TIE_SEEDS=(20260926,20260927,20260928)
DONOR_SEEDS=(20260820,20260821,20260822)
CONFIG=ROOT/'docs/msalign/configs/external_matched_e0_20261014.json'
GUARD_EVENTS=[]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.staging')
    temp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');temp.replace(p)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def exact(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes()
def compare_npz(actual,reference):
    expected=load(reference);got=load(actual) if isinstance(actual,(str,Path)) else actual
    for key,value in expected.items():
        assert key in got and exact(np.asarray(got[key]),value),('array replay differs',str(reference),key)
    return dict(reference=str(Path(reference).relative_to(ROOT)),arrays=len(expected),status='BITWISE_EXACT')
def project_data_paths(path):
    # Classify the lexical name before following symlinks outside the repository.
    values=(Path(os.path.abspath(path)),Path(path).resolve(strict=False));result=[]
    for value in values:
        try:rel=value.relative_to(ROOT)
        except (ValueError,TypeError):continue
        if rel not in result:result.append(rel)
    return result
def project_data_path(path):
    values=project_data_paths(path);return values[0] if values else None
def is_test_data(path):
    return any(rel.parts and rel.parts[0] in ('outputs','data') and
        any(re.search(r'(^|[_.-])test([_.-]|$)',p) for p in rel.parts) for rel in project_data_paths(path))
def install_no_test_guard():
    # Resolve metadata-free path identity only; do not open the combined dataset.
    combined=(ROOT/'data/MassSpecGym1.5.tsv').resolve(strict=False)
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=os.fsdecode(args[0]);rels=project_data_paths(path)
        blocked_combined=any(str(rel)=='data/MassSpecGym1.5.tsv' for rel in rels) or Path(path).resolve(strict=False)==combined
        if is_test_data(path) or blocked_combined:
            label=str(rels[0]) if rels else path
            GUARD_EVENTS.append(label);raise PermissionError('Official test/combined TSV reads are disabled during preparation: '+label)
    sys.addaudithook(audit)

def expected_segment(scores,target,offset=0):
    scores=np.asarray(scores);value=scores[target]
    g=int((scores>value).sum());e=int((scores==value).sum());assert e
    ranks=np.arange(offset+g+1,offset+g+e+1,dtype=np.float64)
    return np.array([(ranks<=k).mean() for k in (1,5,20)]+[np.mean(1/ranks)])
def expected_reranked(full_scores,selected,scores,target=0):
    """Uniform ties within the executed prefix and within the retained Stage1 tail."""
    selected=np.asarray(selected,np.int64);where=np.flatnonzero(selected==target)
    if len(where):return expected_segment(scores,int(where[0]))
    tail=np.flatnonzero(~np.isin(np.arange(len(full_scores)),selected))
    where=np.flatnonzero(tail==target)
    return expected_segment(np.asarray(full_scores)[tail],int(where[0]),len(selected)) if len(where) else np.zeros(4)
def rows_from_records(records,store):
    from rdkit import Chem
    n=len(records);candidates=np.full((n,256),-1,np.int32);permuted=candidates.copy()
    lengths=np.empty(n,np.int32);gt_slots=np.empty(n,np.int32);groups=[];memo={}
    rng=np.random.default_rng(20261014)
    for i,q in enumerate(records):
        values=np.asarray(store.get(q['identifier'],256),np.int32)
        assert len(values)>0 and len(values)==len(set(values.tolist()))
        candidates[i,:len(values)]=values;lengths[i]=len(values)
        order=rng.permutation(len(values));permuted[i,:len(values)]=values[order];gt_slots[i]=int(np.flatnonzero(order==0)[0])
        smi=q['canonical_isomeric_smiles']
        if smi not in memo:
            mol=Chem.MolFromSmiles(smi);assert mol is not None;memo[smi]=Chem.MolToInchiKey(mol)[:14]
        groups.append(memo[smi])
    return dict(candidates=candidates,permuted_candidates=permuted,lengths=lengths,gt_slots=gt_slots,
        identifiers=np.asarray([q['identifier'] for q in records]),smiles=np.asarray([q['canonical_isomeric_smiles'] for q in records]),
        connectivities=np.asarray(groups))
def make_donors(records,groups):
    from tools.audits.audit_spectrum_permutation import build_balanced_strata,make_assignment,assignment_audit
    strata,summary=build_balanced_strata(records,list(groups),6,20260820)
    arrays={};audits={}
    for seed in DONOR_SEEDS:
        mapping=make_assignment(strata,records,list(groups),seed,.02)
        recipients=np.asarray(sorted(mapping),np.int64);donors=np.asarray([mapping[int(q)] for q in recipients],np.int64)
        if 'recipient_rows' not in arrays:arrays['recipient_rows']=recipients
        else:assert np.array_equal(arrays['recipient_rows'],recipients)
        assert set(recipients)==set(donors) and len(set(donors))==len(donors)
        assert np.all(np.asarray(groups)[recipients]!=np.asarray(groups)[donors])
        arrays[f'seed{seed}_donor_rows']=donors
        audits[str(seed)]=assignment_audit(mapping,records,list(groups)) if len(recipients) else dict(queries=0,status='NO_ELIGIBLE_DONORS')
    return arrays,dict(strata=summary,assignments=audits,selection_seed=20260820,jitter=.02,
        algorithm='existing balanced strata + peak-count-matched Hungarian assignment; no eligibility relaxation')

def gpu_preflight(out,label,require_idle=True):
    import subprocess,torch
    from tools.reranking.run_official_lightgbm_transfer import memory_guard,memory_state
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    binding=os.environ.get('CUDA_VISIBLE_DEVICES','')
    row=next((x for x in raw.splitlines() if binding and binding in x),None)
    assert row is not None,'Bind the freshly checked GPU UUID explicitly'
    if require_idle:assert int(row.split(',')[2].strip())<512,'GPU was claimed before launch'
    assert torch.cuda.is_available();memory_guard()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    item=dict(label=label,time=time.time(),gpu=row,torch_device=torch.cuda.get_device_name(0),resources=memory_state())
    out=Path(out);out.mkdir(parents=True,exist_ok=True);write(out/(label+'.preflight.json'),item)
    return item


def sccf_seed_contributions(pool,index):
    """Use the exact frozen harmonic-prefix metric implementation, including ties."""
    from glmr.embcos import uniform_tie_metrics
    greater=[];equal=[]
    for a,b in zip(pool['indptr'][:-1],pool['indptr'][1:]):
        scores=pool['scores'][index,a:b];greater.append(int((scores>scores[0]).sum()));equal.append(int((scores==scores[0]).sum()))
    return uniform_tie_metrics(np.asarray(greater,np.int32),np.asarray(equal,np.int32))
