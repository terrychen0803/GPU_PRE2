"""Evaluator-only mapping of real I2 events to measured I3 signatures and I4 gaps."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from gpu_pre2.data.storage import REPO, data_root, inside_data
from gpu_pre2.i3.taxonomy import classify
from gpu_pre2.profiling.audit import pair

def read(p):return json.loads(p.read_text())
def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def shape_signature(r):return (r.get('aten'),encode(r.get('input_shapes',[])))

def build(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    i2=rows(REPO/'data/manifests/I2_CELLS.csv')
    i3=rows(REPO/'data/manifests/I3_CELLS.csv')
    controlled=defaultdict(set)
    contexts=defaultdict(set)
    for row in i3:
        if row['node']!='RTX4090' or row['mode']!='R' or row['repeat']!='2':continue
        cell=inside_data(data_root()/row['path'])
        refs=[r for r in read(cell/'audit/reference_operations.json') if r['stage']=='measure' and int(r['iteration'])==7]
        for ref in refs:
            store=contexts if row['stage']=='S4' else controlled
            store[shape_signature(ref)].add(row['config_id'])
    frozen_classes=read(data_root()/'checkpoints/i4_p0_p2/TAXONOMY.json')
    # The historical closed set is fixed; this mapping does not add new labels.
    supported={'adaptive_average_pool','add','batchnorm_eval','bilinear_resize','bmm','concat','conv','depthwise','gemm','maxpool','relu'}
    aliases={'resize_bilinear':'bilinear_resize','resize_bilinear_native':'bilinear_resize'}
    coverage=[]
    for workload in sorted({r['workload'] for r in i2}):
        bymode={r['mode']:r for r in i2 if r['device']=='RTX4090' and r['workload']==workload and r['repeat']=='2'}
        p=inside_data(data_root()/bymode['P']['root'])
        r=inside_data(data_root()/bymode['R']['root'])
        aligned=pair(p/'audit',r/'audit',out/(workload+'_alignment'))
        selected=[x for x in aligned if x['scope']=='o4.iter/measure/000007']
        if len(selected)!=1 or not selected[0]['ordered_signature_exact']:
            raise ValueError('Cannot transfer R ownership to source P: '+workload)
        events=[e for e in rows(p/'audit/events_diagnostic.csv') if e['iteration_scope']=='o4.iter/measure/000007']
        refs=[x for x in read(r/'audit/reference_operations.json') if x['stage']=='measure' and int(x['iteration'])==7]
        for ref in refs:
            ids=ref['event_indices'];op_events=[events[j] for j in ids]
            family=aliases.get(classify(ref['aten'])['subtype'],classify(ref['aten'])['subtype'])
            signature=shape_signature(ref)
            matches=sorted(controlled.get(signature,[]));ctx=sorted(contexts.get(signature,[]))
            reasons=[]
            if not ids:reasons.append('zero_recorded_work_not_verified_noop')
            if not ref['contiguous']:reasons.append('noncontiguous_recorded_ownership')
            if len(ids)>8:reasons.append('over_L8')
            if family not in supported:reasons.append('outside_frozen_11_class_taxonomy')
            if any(e['event_kind']!='kernel' for e in op_events):reasons.append('nonkernel_geometry_unsupported')
            if len({(e['device'],e['context'],e['stream']) for e in op_events})>1:reasons.append('multistream_unsupported')
            if not ref.get('bound_arguments'):reasons.append('missing_full_semantic_arguments_no_exact_match_claim')
            coverage.append({'workload_id':workload,'operator_index':ref['op_index'],'operator_overload':ref['aten'],
                'operator_family':family,'runtime_signature':hashlib.sha256(encode(signature).encode()).hexdigest(),
                'signature_level':'ATen_name_and_input_shapes_only',
                'source_P_id':bymode['P']['root'],'source_event_range':encode(ids),
                'input_shapes':encode(ref.get('input_shapes',[])),
                'i3_operator_match':'shape_only' if matches else 'not_matched',
                'i3_operator_ids':encode(matches),'i3_context_match':'shape_only' if ctx else 'not_matched',
                'i3_context_ids':encode(ctx),'i4_supported':'unverified' if reasons else 'geometry_and_closed_set_candidate_only',
                'coverage_reason':';'.join(reasons),'reference_quality':'development_unverified',
                'target_GI_ids':encode({d:next(x['root'] for x in i2 if x['device']==d and x['workload']==workload and x['mode']=='GI' and x['repeat']=='2') for d in ('RTX3090','GB10')})})
        unassigned=set(range(len(events)))-{i for ref in refs for i in ref['event_indices']}
        for i in sorted(unassigned):
            coverage.append({'workload_id':workload,'operator_index':'','operator_overload':'','operator_family':'unassigned_reference_event',
                'runtime_signature':'','signature_level':'unassigned','source_P_id':bymode['P']['root'],
                'source_event_range':encode([i]),'input_shapes':'[]','i3_operator_match':'not_matched','i3_operator_ids':'[]',
                'i3_context_match':'not_matched','i3_context_ids':'[]','i4_supported':'unsupported',
                'coverage_reason':'unassigned_reference_'+events[i]['event_kind'],'reference_quality':'development_unverified',
                'target_GI_ids':'{}'})
    with (out/'TASK_COVERAGE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(coverage[0]));w.writeheader();w.writerows(coverage)
    summary={'rows':len(coverage),'source':'RTX4090','repeat':2,'measured_iteration':7,
        'by_workload':dict(Counter(r['workload_id'] for r in coverage)),
        'mapping_level':'ATen name + input shapes; no complete semantic exact match claim',
        'unassigned_events':sum(r['operator_family']=='unassigned_reference_event' for r in coverage),
        'model_evaluation_performed':False,'strict_reference_verified':False,'taxonomy_sha256':hashlib.sha256(encode(frozen_classes).encode()).hexdigest()}
    (out/'TASK_COVERAGE_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    build(p.parse_args().out)
