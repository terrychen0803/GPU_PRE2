"""Offline numerical regression of the explicit retained datasets; no CUDA work."""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from gpu_pre2.data.storage import REPO, data_root, dataset, inside_data
from gpu_pre2.profiling.audit_v5 import audit

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def dependency_audit():
    bad=[];count=0
    forbidden={'o1_operator','o2_sequence','o4_iteration','o4_research','inference_pilot','o4_i3','o4_i3_v2','o4_i3_context'}
    for root in (REPO/'src',REPO/'e2_calibration'):
        for p in root.rglob('*.py'):
            count+=1
            text=p.read_text();tree=ast.parse(text)
            for node in ast.walk(tree):
                names=[]
                if isinstance(node,ast.Import):names=[x.name.split('.')[0] for x in node.names]
                if isinstance(node,ast.ImportFrom) and node.level==0:names=[(node.module or '').split('.')[0]]
                if forbidden.intersection(names):bad.append(str(p))
            if any(x in text for x in ('sys.path.insert','sys.path.append','entry_points(',"vendor/legacy",'/home/good_egg/RTchen/O')):
                bad.append(str(p))
    if bad:raise ValueError('Forbidden runtime dependencies: '+str(bad))
    return {'python_files':count,'forbidden_dependencies':bad}

def validate_cells(stage):
    records=rows(REPO/f'data/manifests/{stage}_CELLS.csv')
    expected={'I2':189,'I3':3024}[stage]
    if len(records)!=expected:raise ValueError('Cell count changed')
    keys=set();measured=0
    for row in records:
        relative=row.get('path',row.get('root'))
        cell=inside_data(data_root()/relative)
        if relative in keys:raise ValueError('Duplicate canonical cell')
        keys.add(relative)
        samples=rows(cell/'iterations.csv')
        samples=[r for r in samples if r['stage']=='measure']
        if len(samples)!=40 or {int(r['iteration']) for r in samples}!=set(range(40)):
            raise ValueError('Incomplete measured set: '+str(cell))
        measured+=len(samples)
    return records,{'cells':len(records),'measured_iterations':measured}

def predict_regression(out):
    root=dataset('I4_P0_P2',stage='I4')
    result=[]
    env={**os.environ,'PYTHONPATH':str(REPO/'src'),'PYTHONNOUSERSITE':'1',
         'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    # Isolated input bundle deliberately omits evaluator, R and old research folders.
    with tempfile.TemporaryDirectory(prefix='gpu-pre2-source-only-') as directory:
        tmp=Path(directory)
        shutil.copytree(root/'models',tmp/'models')
        shutil.copy2(root/'data/predictor_input_v2/hardware.json',tmp/'hardware.json')
        def run(repeat,label):
            source=tmp/f'repeat{repeat}.json'
            if not source.exists():shutil.copy2(root/f'data/predictor_input_v2/repeat{repeat}.json',source)
            dest=out/f'i4_{label}.json'
            subprocess.run([sys.executable,'-m','gpu_pre2.models.historical.predict',
                '--input',str(source),'--hardware',str(tmp/'hardware.json'),
                '--models',str(tmp/'models'),'--out',str(dest)],check=True,env=env,cwd=tmp,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            if sha(dest)!=sha(root/f'predictions_v2/repeat{repeat}.json'):
                raise ValueError('Historical v2 prediction changed: '+label)
            return {'repeat':repeat,'sha256':sha(dest),'byte_exact':True}
        for repeat in (1,2,3):result.append(run(repeat,f'repeat{repeat}'))
        (tmp/'target_test_trace.json').write_text('{"elapsed_us":1e100,"true_class":"POISON"}')
        (tmp/'reference.json').write_text('{"gold_segments":"POISON","true_shape":[999]}')
        result.append({**run(2,'repeat2_poisoned'),'target_reference_poison_invariant':True})
    return result

def parser_regression(records,stage,out):
    candidates=[r for r in records if r['mode']!='GI' and (stage=='I2' or r['repeat']=='2')]
    checked=byte_exact=0;known_differences=[]
    for i,row in enumerate(candidates):
        cell=inside_data(data_root()/row.get('path',row.get('root')))
        dest=out/f'parser_{stage}_{i:04d}'
        audit(cell,dest)
        original=cell/'audit'
        for p in sorted(dest.rglob('*')):
            if not p.is_file():continue
            relative=p.relative_to(dest);old=original/relative
            if not old.exists():raise ValueError('Unexpected parser artifact: '+str(relative))
            checked+=1
            if p.read_bytes()==old.read_bytes():byte_exact+=1;continue
            # The former Groups4 exporter flattened all streams to s0. Preserve the
            # true capture-local aliases now; all other fields must remain identical.
            if 'predictor' in relative.parts and row.get('config_id')=='I3_conv_00_groups4':
                before=[json.loads(x) for x in old.read_text().splitlines()]
                after=[json.loads(x) for x in p.read_text().splitlines()]
                streams={x['stream_id'] for x in after}
                if len(streams)!=4:raise ValueError('Groups4 stream aliases lost')
                for item in after:item['stream_id']='s0'
                if before!=after:raise ValueError('Unexplained Groups4 numeric difference')
                known_differences.append({'cell':str(cell.relative_to(data_root())),
                    'artifact':str(relative),'reason':'corrected_four_stream_aliases_preserve_all_other_fields'})
            else:
                raise ValueError('Unexpected parser difference: '+str(cell)+' '+str(relative))
        shutil.rmtree(dest)
        if (i+1)%40==0:print(stage,'parser captures',i+1,'/',len(candidates),flush=True)
    return {'captures':len(candidates),'artifacts':checked,'byte_exact_artifacts':byte_exact,
            'versioned_stream_corrections':known_differences}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--skip-parser',action='store_true')
    args=parser.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    result={'dependencies':dependency_audit(),'GPU_runs':0,'strict_reference_verified':False}
    i2,result['I2']=validate_cells('I2')
    i3,result['I3']=validate_cells('I3')
    result['I4']=predict_regression(out)
    if not args.skip_parser:
        result['I2_parser']=parser_regression(i2,'I2',out)
        result['I3_parser']=parser_regression(i3,'I3',out)
    result['status']='PASS_OFFLINE_MIGRATION_ONLY'
    (out/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'I3_parser'}},indent=2))

if __name__=='__main__':main()
