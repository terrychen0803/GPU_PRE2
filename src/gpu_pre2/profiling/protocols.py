"""Only named retained protocols can be replayed. No plugin entry-point discovery."""
SINGLE_STREAM = {
    'o4_seven_A_40x3_engineering_v2_20261106',
    'o4_i3_config_context_40x3_v1_20261107',
    'o4_i3_context_DIS_40x3_v1_20261107',
    'o4_i3_context_DIS_BC_40x3_v1_20261107',
    'o4_i3_config_smoke_v1_20261107',
    'o4_i3_context_DIS_smoke_v1_20261107',
    'o4_i3_context_DIS_BC_smoke_v1_20261107',
}
GROUPS4 = 'o4_i3_groups4_multistream_conditional_v2_20261107'

def validate_request(request):
    protocol = request.get('protocol', {})
    ident = protocol.get('protocol_id')
    if ident == GROUPS4:
        if request.get('config_id') != 'I3_conv_00_groups4' or protocol.get('single_stream_required') is not False:
            raise ValueError('Groups4 protocol is restricted to its frozen conditional configuration')
    elif ident not in SINGLE_STREAM and ident not in {'gpu_pre2:'+p for p in SINGLE_STREAM}:
        raise ValueError(f'Unregistered capture protocol {ident!r}; no historical fallback')
    return ident

def groups4_evidence(req, events, expected, errors, scope_exact, references, reference_exact):
    """Exact recorded-structure exception; never proves all-stream GI completion."""
    from collections import defaultdict
    per_scope=defaultdict(list)
    for e in events:
        if e['iteration_scope']:per_scope[e['iteration_scope']].append(e)
    streams={(e['device'],e['context'],e['stream']) for e in events}
    passed=(req.get('config_id')=='I3_conv_00_groups4'
        and req['mode'] in ('P','R') and req['protocol']['protocol_id']==GROUPS4
        and req['protocol'].get('single_stream_required') is False
        and not errors and scope_exact and len(events)==4*len(expected)
        and len(streams)==4 and len({(e['device'],e['context']) for e in events})==1
        and set(per_scope)==expected
        and all(len(v)==4 and {(e['device'],e['context'],e['stream']) for e in v}==streams for v in per_scope.values())
        and all(e['iteration_scope'] in expected and not e['preparation_scope'] for e in events)
        and (req['mode']!='R' or (reference_exact and len(references)==len(expected)
             and all(r['recorded_GPU_event_count']==4 and r['contiguous'] for r in references)
             and all(e['op_scope'] for e in events))))
    return {'policy':'groups4_conditional_recorded_structure_v2',
        'conditional_four_stream_recorded_structure':passed,
        'raw_correlation_errors':len(errors),'scope_count':len(per_scope),
        'gpu_events':len(events),'distinct_streams':len(streams),
        'GI_all_stream_completion_verified':False,'strict_reference_verified':False,
        'source_protocol_id':req['protocol']['protocol_id'],
        'note':'Observed event structure and reference ownership only; device-wide timing/submitted work unverified'}
