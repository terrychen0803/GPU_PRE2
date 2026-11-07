"""Evaluator only. Read sealed predictions first, attach historical labels/gates.

No weights, optimizer, predictor calls, source-view selection or test tuning.
"""
from collections import defaultdict, Counter
import numpy as np
from .common import *
from .features import candidates, Unsupported, union_ns


METHODS=('Identity','Ratio','Ridge','N0')


def time_stats(rr,prefix):
    v=[r for r in rr if r[prefix+'_us'] is not None]
    if not v:
        return {'n':0,'MdAPE_pct':None,'P90_APE_pct':None,'MAE_us':None,'signed_error_us':None,'time20_pct':None}
    y=np.array([r['target_P_CTX_us'] for r in v]);p=np.array([r[prefix+'_us'] for r in v])
    ape=np.abs(p/y-1)*100
    return {'n':len(v),'MdAPE_pct':float(np.median(ape)),'P90_APE_pct':float(np.percentile(ape,90)),
            'MAE_us':float(np.mean(np.abs(p-y))),'signed_error_us':float(np.mean(p-y)),
            'time20_pct':100*float(np.mean(ape<=20))}


def evaluate(run, out):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    from gpu_pre2.data.storage import data_root
    if out==data_root() or data_root() in out.parents:raise ValueError('Evaluation output must be outside immutable data')
    seals={}
    predictions={}
    for rep in (2,1,3):
        p=run/f'predictions_v2/repeat{rep}.json'
        seal=read(str(p)+'.seal.json');assert sha(p)==seal['prediction_sha256']
        assert sha(run/'models/FROZEN.json')==seal['frozen_sha256']
        predictions[rep]={(r['seed'],r['id']):r for r in read(p)}
        seals[str(rep)]=seal
    # Only after sealed inference do we attach target values and retrospective masks.
    mapping={r['id']:r for r in rows(run/'data/evaluator/ID_MAP.csv')}
    gold=read(run/'data/evaluator/operations.json')
    label={(r['id'],r['node'],r['op_index']):r['target_P_CTX_us'] for r in read(run/'data/evaluator/context_labels.json')}
    pool={(oid(r['config_id']),r['target'],int(r['op_index'])):r for r in rows(run/'data/evaluator/N0_FULL_POOL.csv')}
    primaryids=set().union(*(set(read(run/f'data/splits/B_{s}.json')) for s in SPLITS))
    primarypairs={(oid(r['config_id']),r['target'],int(r['op_index'])) for r in rows(run/'data/evaluator/PRIMARY_N0.csv')}
    source_rows={rep:{r['id']:r for r in read(run/f'data/predictor_input_v2/repeat{rep}.json')} for rep in (2,1,3)}
    classified=[];timed=[];graphs=[];diagnostic=[]
    for rep in (2,1,3):
        for seed in SEEDS:
            for i,meta in mapping.items():
                p=predictions[rep][seed,i];raw=source_rows[rep][i];gs=gold[i]
                try:
                    cs=candidates(raw['events'],L)
                except Unsupported:
                    cs=[]
                ix={(c['a'],c['b']):j for j,c in enumerate(cs)}
                predicted={(g['a'],g['b']):g for g in p['predicted']}
                for g in gs:
                    key=g['a'],g['b'];j=ix.get(key);match=predicted.get(key)
                    known=g['c'] is not None
                    boundary=match is not None
                    typed=known and boundary and match['c']==g['c']
                    oracle_cls=int(np.argmax(p['candidate_q'][j])) if j is not None else None
                    classified.append({'repeat':rep,'seed':seed,**meta,'op_index':g['op_index'],
                         'class':g['semantic_class'],'true_c':g['c'],'oracle_pred_c':oracle_cls,
                         'a':g['a'],'b':g['b'],'events':g['b']-g['a'],
                         'boundary':boundary,'typed':typed,'known':known,
                         'primary':i in primaryids,'source_status':p['status'],
                         'predicted_segments':len(p['predicted'])})
                    for t in TARGETS:
                        audit=pool[i,t,g['op_index']]
                        reasons=json.loads(audit['reasons'])
                        relaxed=not [x for x in reasons if not x.endswith('CTX_process_spread_gt_5pct')]
                        y=label.get((i,t,g['op_index']))
                        assert y is None or y>0
                        for method in METHODS:
                            oracle=p['candidate_times'][t][method][j] if j is not None else None
                            matched=oracle if typed else None
                            if y is None:
                                failure='unscorable_missing_label'
                            elif not known:
                                failure='unsupported_class'
                            elif p['status']=='unsupported':
                                failure='unsupported_input'
                            elif j is None:
                                failure='candidate_over_L'
                            elif not boundary:
                                failure='boundary'
                            elif not typed:
                                failure='type'
                            elif abs(oracle/y-1)>.2:
                                failure='time'
                            else:
                                failure='success20'
                            timed.append({'repeat':rep,'seed':seed,**meta,'target':t,'method':method,
                                'op_index':g['op_index'],'class':g['semantic_class'],'events':g['b']-g['a'],
                                'target_P_CTX_us':y,'oracle_us':oracle,'matched_us':matched,
                                'boundary':boundary,'typed':typed,'failure':failure,
                                'primary_isolated_pair':(i,t,g['op_index']) in primarypairs,
                                'source_primary_class':i in primaryids,
                                'ctx_main_gate':audit['proposed_exploratory_candidate']=='True',
                                'relaxed_stability_gate':relaxed,'gate_reasons':audit['reasons'],
                                'strict_reference_verified':False})
                # All predicted segments partition the event index; evaluator computes temporal coverage separately.
                matched_ops=[g for g in gs if (g['a'],g['b']) in predicted]
                typed_ops=[g for g in matched_ops if g['c'] is not None and predicted[g['a'],g['b']]['c']==g['c']]
                matched_events={j for g in matched_ops for j in range(g['a'],g['b'])}
                typed_events={j for g in typed_ops for j in range(g['a'],g['b'])}
                diagnostic.append({'repeat':rep,'seed':seed,**meta,'reference_ops':len(gs),'known_ops':sum(g['c'] is not None for g in gs),
                      'predicted_ops':len(p['predicted']),'boundary_hits':len(matched_ops),'typed_hits':len(typed_ops),
                      'reference_events':len(raw['events']),'boundary_matched_events':len(matched_events),
                      'typed_matched_events':len(typed_events),'candidate_over_L_ops':sum(g['b']-g['a']>L for g in gs),
                      'source_status':p['status'],'semantic_complete_evaluator':len(typed_ops)==len(gs),
                      'all_work_reference_verified':False})
                if meta['stage']=='S4':
                    for t in TARGETS:
                        for method in METHODS:
                            predsum=sum(p['candidate_times'][t][method][g['candidate_index']] for g in p['predicted']) if p['status']!='unsupported' else None
                            truthsum=sum(label[i,t,g['op_index']] for g in gs)
                            graphs.append({'repeat':rep,'seed':seed,**meta,'target':t,'method':method,
                                  'sum_predicted_segment_P_CTX_us':predsum,'sum_target_op_P_CTX_us':truthsum,
                                  'semantic_complete_evaluator':len(typed_ops)==len(gs),'reference_ops':len(gs),
                                  'main_gate_positions':sum(pool[i,t,g['op_index']]['proposed_exploratory_candidate']=='True' for g in gs),
                                  'meaning':'sum of contextual spans; not union, GI or causal dependency estimate'})
    table(out/'CLASSIFICATION_ROWS.csv',classified)
    table(out/'TIME_ROWS.csv',timed)
    table(out/'SEQUENCE_COVERAGE.csv',diagnostic)
    table(out/'S4_WORK_SUMS.csv',graphs)
    # Classification main 64 configurations, plus S4 all 84 positions. No repeat pooling.
    class_metrics=[];perclass=[];confusions=[]
    for rep in (2,1,3):
        for seed in SEEDS:
            for scope in (*SPLITS,'S4_all','S4_D_only','isolated_challenge'):
                rr=[r for r in classified if r['repeat']==rep and r['seed']==seed and
                    ((scope in SPLITS and r['primary'] and r['split']==scope) or
                     (scope=='S4_all' and r['stage']=='S4') or
                     (scope=='S4_D_only' and r['config_id'].startswith('CTX_')) or
                     (scope=='isolated_challenge' and r['stage']!='S4' and not r['primary']))]
                ids={r['id'] for r in rr};dd=[d for d in diagnostic if d['repeat']==rep and d['seed']==seed and d['id'] in ids]
                gn=len(rr);pn=sum(d['predicted_ops'] for d in dd);bp=sum(r['boundary'] for r in rr);tp=sum(r['typed'] for r in rr)
                confusion=np.zeros((11,12),dtype=int)
                for r in rr:
                    if r['known']:
                        confusion[r['true_c'],r['oracle_pred_c'] if r['oracle_pred_c'] is not None else 11]+=1
                f=[]
                for c,cls in enumerate(CLASSES):
                    den=confusion[c].sum()+confusion[:,c].sum()
                    val=200*confusion[c,c]/den if den else 0
                    f.append(val)
                    perclass.append({'repeat':rep,'seed':seed,'scope':scope,'class':cls,
                         'reference_ops':int(confusion[c].sum()),'oracle_F1_pct':val,
                         'boundary_hits':sum(r['boundary'] for r in rr if r['true_c']==c),
                         'typed_hits':sum(r['typed'] for r in rr if r['true_c']==c)})
                    for pred_cls in range(12):
                        confusions.append({'repeat':rep,'seed':seed,'scope':scope,'true_class':cls,
                             'predicted_class':CLASSES[pred_cls] if pred_cls<11 else 'unreachable',
                             'n':int(confusion[c,pred_cls]),
                             'row_fraction':float(confusion[c,pred_cls]/confusion[c].sum()) if confusion[c].sum() else None})
                class_metrics.append({'repeat':rep,'seed':seed,'scope':scope,'configs':len(ids),
                     'groups':len({r['group'] for r in rr}),'reference_ops':gn,'known_ops':sum(r['known'] for r in rr),
                     'unknown_ops':sum(not r['known'] for r in rr),'predicted_ops':pn,
                     'boundary_F1_pct':200*bp/(gn+pn) if gn+pn else 0,
                     'typed_F1_pct':200*tp/(gn+pn) if gn+pn else 0,
                     'oracle_macro_F1_pct':float(np.mean(f)),
                     'typed_coverage_pct':100*tp/gn if gn else 0,
                     'boundary_event_coverage_pct':100*sum(d['boundary_matched_events'] for d in dd)/sum(d['reference_events'] for d in dd),
                     'typed_event_coverage_pct':100*sum(d['typed_matched_events'] for d in dd)/sum(d['reference_events'] for d in dd),
                     'candidate_over_L_ops':sum(d['candidate_over_L_ops'] for d in dd),
                     'unsupported_sequences':sum(d['source_status']=='unsupported' for d in dd)})
    table(out/'CLASSIFICATION_METRICS.csv',class_metrics)
    table(out/'PER_CLASS.csv',perclass)
    table(out/'CONFUSION_ROW_NORMALIZED.csv',confusions)
    # Timing method comparison on identical per-scope reference sets.
    metrics=[];strata=[]
    def in_scope(r,scope):
        if scope in SPLITS:
            return r['primary_isolated_pair'] and r['split']==scope
        if scope=='isolated_relaxed_stability':
            return r['source_primary_class'] and r['relaxed_stability_gate']
        if scope=='S4_all_recorded':
            return r['stage']=='S4'
        if scope=='S4_main_gate':
            return r['stage']=='S4' and r['ctx_main_gate']
        if scope=='S4_D_all_recorded':
            return r['config_id'].startswith('CTX_')
        if scope=='S4_D_main_gate':
            return r['config_id'].startswith('CTX_') and r['ctx_main_gate']
        return False
    for rep in (2,1,3):
        for seed in SEEDS:
            for target in TARGETS:
                for method in METHODS:
                    base=[r for r in timed if r['repeat']==rep and r['seed']==seed and r['target']==target and r['method']==method]
                    for scope in (*SPLITS,'isolated_relaxed_stability','S4_all_recorded','S4_main_gate','S4_D_all_recorded','S4_D_main_gate'):
                        rr=[r for r in base if in_scope(r,scope)];failure=Counter(r['failure'] for r in rr)
                        row={'repeat':rep,'seed':seed,'target':target,'method':method,'scope':scope,
                             'reference_ops':len(rr),'configs':len({r['id'] for r in rr}),'groups':len({r['group'] for r in rr}),
                             'scorable_labels':sum(r['target_P_CTX_us'] is not None for r in rr),
                             'success20_pct':100*failure['success20']/len(rr) if rr else None,
                             'failures':json.dumps(dict(failure),sort_keys=True)}
                        for prefix in ('oracle','matched'):
                            row.update({prefix+'_'+k:v for k,v in time_stats(rr,prefix).items()})
                        metrics.append(row)
                        if rep==2:
                            for strat in ('class','events','gate_reasons'):
                                for val in sorted({str(r[strat]) for r in rr}):
                                    ss=[r for r in rr if str(r[strat])==val]
                                    strata.append({'seed':seed,'target':target,'method':method,'scope':scope,'stratum':strat,'value':val,
                                        'reference_ops':len(ss),'success20_pct':100*sum(r['failure']=='success20' for r in ss)/len(ss),
                                        **{'oracle_'+k:v for k,v in time_stats(ss,'oracle').items()}})
    table(out/'TIME_METRICS.csv',metrics)
    table(out/'TIME_STRATA.csv',strata)
    # Paired group bootstrap: seeds stay paired and are averaged within each config group.
    rng=np.random.default_rng(20261107);bootstrap=[]
    for t in TARGETS:
        rr=[r for r in timed if r['repeat']==2 and r['target']==t and r['split']=='test_exploratory' and r['primary_isolated_pair']]
        gs=sorted({r['group'] for r in rr})
        for baseline in ('Identity','Ratio','Ridge'):
            delta=[]
            for g in gs:
                a=[100*abs(r['oracle_us']/r['target_P_CTX_us']-1) for r in rr if r['group']==g and r['method']=='N0']
                b=[100*abs(r['oracle_us']/r['target_P_CTX_us']-1) for r in rr if r['group']==g and r['method']==baseline]
                delta.append(float(np.mean(a)-np.mean(b)))
            draws=np.array([np.mean(rng.choice(delta,size=len(delta),replace=True)) for _ in range(1000)])
            bootstrap.append({'target':t,'scope':'historical_test_oracle','comparison':'N0 minus '+baseline,
                 'metric':'mean_APE_pp; paired groups, seeds equal inside group','groups':len(gs),
                 'delta_pp':float(np.mean(delta)),'p2_5':float(np.percentile(draws,2.5)),
                 'p97_5':float(np.percentile(draws,97.5)),'status':'descriptive_small_historical_pool_not_confirmation'})
    table(out/'PAIRED_GROUP_BOOTSTRAP.csv',bootstrap)
    # D/I/S sums per graph, never presented as GI prediction.
    import re
    dis=[]
    for rep in (2,1,3):
        for seed in SEEDS:
            for t in TARGETS:
                for method in METHODS:
                    gg=[g for g in graphs if g['repeat']==rep and g['seed']==seed and g['target']==t and g['method']==method]
                    for d in [g for g in gg if g['config_id'].startswith('CTX_')]:
                        name=d['config_id'][4:];ind=next(g for g in gg if g['config_id']=='IND_'+name)
                        iso=[g for g in gg if g['config_id'].startswith('ISO_'+name+'_')]
                        assert len(iso)==d['reference_ops']
                        dis.append({'repeat':rep,'seed':seed,'target':t,'method':method,'graph':name,
                             'D_pred_sum_CTX_us':d['sum_predicted_segment_P_CTX_us'],
                             'I_pred_sum_CTX_us':ind['sum_predicted_segment_P_CTX_us'],
                             'S_pred_sum_CTX_us':sum(g['sum_predicted_segment_P_CTX_us'] for g in iso) if all(g['sum_predicted_segment_P_CTX_us'] is not None for g in iso) else None,
                             'D_target_sum_CTX_us':d['sum_target_op_P_CTX_us'],
                             'I_target_sum_CTX_us':ind['sum_target_op_P_CTX_us'],
                             'S_target_sum_CTX_us':sum(g['sum_target_op_P_CTX_us'] for g in iso),
                             'missing_S_predictions':sum(g['sum_predicted_segment_P_CTX_us'] is None for g in iso),
                             'independent_context_groups':1,'GI_prediction':False})
    table(out/'D_I_S_CTX_SUMS.csv',dis)
    save(out/'EVALUATION_COMPLETE.json',{'prediction_seals':seals,'strict_reference_verified':0,
         'formal_training_eligible':False,'fits_after_test':0,'reference_status':'historical_warning_accepted',
         'A0':'planned_only_not_fit','C0':'insufficient_independent_contexts','ITER':'not_trained',
         'unknowns':'evaluator identifies unsupported truth; predictor q cannot detect them',
         'primary_source_repeat':2,'robustness_repeats':[1,3],'all_repeats_separate':True,
         'metric_note':'S4 oracle macro F1 uses fixed 11-class taxonomy, absent classes zero; inspect PER_CLASS denominators',
         'bootstrap':'11 historical test groups/direction; descriptive only'})
    print('evaluation complete; historical test and B0 S4 only',flush=True)


def main():
    import argparse
    from gpu_pre2.data.storage import dataset
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-id',default='I4_P0_P2')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    evaluate(dataset(a.dataset_id,stage='I4'),a.out)


if __name__=='__main__':
    main()
