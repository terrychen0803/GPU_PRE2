"""E1 numeric 27D contract, original timestamps retained verbatim."""
import math
from statistics import mean, median, pstdev

NAMES = ['event_count', 'span_us', 'union_us', 'idle_fraction', 'duration_min_us',
         'duration_median_us', 'duration_max_us', 'duration_mean_us', 'duration_std_us',
         'positive_gap_sum_us', 'positive_gap_max_us', 'positive_gap_mean_us',
         'grid_product_mean', 'block_product_mean', 'dynamic_shared_bytes_mean',
         'first_duration_us', 'last_duration_us', 'preceding_positive_gap_us',
         'following_positive_gap_us', 'has_preceding_event', 'has_following_event',
         'grid_x_mean', 'grid_y_mean', 'grid_z_mean',
         'block_x_mean', 'block_y_mean', 'block_z_mean']
EVENT_FIELDS = {'event_kind', 'start_ns', 'end_ns', 'api_start_ns', 'api_end_ns',
                'grid_xyz', 'block_xyz', 'dynamic_shared_memory_bytes', 'bytes',
                'copy_kind', 'memory_kind', 'memset_value', 'event_id', 'api_event_id',
                'correlation_id', 'stream_id', 'context_id', 'process_id'}


class Unsupported(ValueError):
    pass


def validate(events):
    if not events:
        raise Unsupported('missing_or_zero_recorded_work; never impute time zero')
    for e in events:
        if set(e) != EVENT_FIELDS:
            raise ValueError('numeric event allowlist violation')
        if not isinstance(e['start_ns'], int) or not isinstance(e['end_ns'], int):
            raise ValueError('timestamps must retain integer ns')
        if e['end_ns'] <= e['start_ns']:
            raise Unsupported('nonpositive_duration')
        if e['event_kind'] != 'kernel' or e['grid_xyz'] is None or e['block_xyz'] is None:
            raise Unsupported('E1_missing_geometry: event retained, no invented geometry')
        if any(len(e[k]) != 3 or any(x <= 0 for x in e[k]) for k in ('grid_xyz', 'block_xyz')):
            raise Unsupported('invalid_geometry')
        if e['dynamic_shared_memory_bytes'] is None or e['dynamic_shared_memory_bytes'] < 0:
            raise Unsupported('missing_dynamic_shared_memory')
    if len({(e['process_id'], e['context_id'], e['stream_id']) for e in events}) != 1:
        raise Unsupported('multistream')
    if any((a['start_ns'], a['end_ns']) > (b['start_ns'], b['end_ns']) for a, b in zip(events, events[1:])):
        raise Unsupported('nonmonotone_event_index')


def union_ns(events):
    intervals = sorted((e['start_ns'], e['end_ns']) for e in events)
    if not intervals:
        raise Unsupported('empty union')
    a, b = intervals[0]
    total = 0
    for u, v in intervals[1:]:
        if u <= b:
            b = max(b, v)
        else:
            total += b - a
            a, b = u, v
    return total + b - a


def features(events, a, b):
    if not 0 <= a < b <= len(events):
        raise ValueError('invalid candidate')
    es = events[a:b]
    ds = [(e['end_ns'] - e['start_ns']) / 1000 for e in es]
    span = (max(e['end_ns'] for e in es) - min(e['start_ns'] for e in es)) / 1000
    busy = union_ns(es) / 1000
    gaps = [max(0, (y['start_ns'] - x['end_ns']) / 1000) for x, y in zip(es, es[1:])]
    before = max(0, (events[a]['start_ns'] - events[a-1]['end_ns']) / 1000) if a else 0
    after = max(0, (events[b]['start_ns'] - events[b-1]['end_ns']) / 1000) if b < len(events) else 0
    z = [float(len(es)), span, busy, max(0, 1-busy/span), min(ds), median(ds), max(ds),
         mean(ds), pstdev(ds), sum(gaps), max(gaps, default=0), mean(gaps) if gaps else 0,
         mean(math.prod(e['grid_xyz']) for e in es), mean(math.prod(e['block_xyz']) for e in es),
         mean(e['dynamic_shared_memory_bytes'] for e in es), ds[0], ds[-1], before, after,
         int(a > 0), int(b < len(events))]
    z += [mean(e[k][j] for e in es) for k in ('grid_xyz', 'block_xyz') for j in range(3)]
    if len(z) != 27 or not all(math.isfinite(v) and v >= 0 for v in z):
        raise ValueError('invalid feature vector')
    return z


def candidates(events, limit):
    validate(events)
    if limit < 1:
        raise ValueError('positive candidate limit required')
    return [{'a': a, 'b': b, 'z': features(events, a, b)}
            for b in range(1, len(events)+1) for a in range(max(0, b-limit), b)]


def request(r):
    # IDs only join output; neither IDs nor timestamps enter any model tensor.
    if set(r) != {'id', 'events', 'h_source'}:
        raise ValueError('source-only request schema violation')
    if len(r['h_source']) != 2 or not all(math.isfinite(x) and x > 0 for x in r['h_source']):
        raise ValueError('invalid source hardware')
    return {'id': r['id'], 'n': len(r['events']), 'h_source': r['h_source'],
            'candidates': candidates(r['events'], 8)}
