"""Inspect session cache to diagnose stream length issues."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

session = sys.argv[1] if len(sys.argv) > 1 else 'Y_45_03302026'
import glob
cache_files = glob.glob(f'C:/Users/optilab/desktop/MCCT/session_cache/*{session}*.npz')
if not cache_files:
    print(f"No cache for {session}")
    sys.exit(1)

npz_path = cache_files[0]
print(f"Cache: {npz_path}")
d = np.load(npz_path, allow_pickle=True)

# Check timestamp spans for each stream
streams = {}
for k in sorted(d.keys()):
    arr = d[k]
    if not hasattr(arr, 'shape') or arr.ndim == 0:
        continue
    if k.endswith('_ts'):
        base = k.replace('_ts', '')
        if base not in streams:
            streams[base] = {}
        streams[base]['ts'] = arr
        streams[base]['n'] = len(arr)
        streams[base]['start'] = float(arr[0])
        streams[base]['end'] = float(arr[-1])
        streams[base]['span'] = float(arr[-1] - arr[0])
    elif k.endswith('_valid'):
        base = k.replace('_valid', '')
        if base not in streams:
            streams[base] = {}
        streams[base]['valid_pct'] = float(arr.mean() * 100)
        streams[base]['valid_n'] = int(arr.sum())
    elif not k.endswith('_ts') and not k.endswith('_valid'):
        if k not in streams:
            streams[k] = {}
        streams[k]['shape'] = arr.shape

print(f"\n{'Stream':<25s} {'Shape':>15s} {'Samples':>8s} {'Span(s)':>8s} {'Start':>12s} {'End':>12s} {'Valid%':>7s}")
print("-" * 90)
for name in sorted(streams.keys()):
    s = streams[name]
    shape = str(s.get('shape', ''))
    n = str(s.get('n', ''))
    span = f"{s['span']:.1f}" if 'span' in s else ''
    start = f"{s['start']:.1f}" if 'start' in s else ''
    end = f"{s['end']:.1f}" if 'end' in s else ''
    valid = f"{s['valid_pct']:.1f}" if 'valid_pct' in s else ''
    print(f"{name:<25s} {shape:>15s} {n:>8s} {span:>8s} {start:>12s} {end:>12s} {valid:>7s}")

# Check the alignment — what's the common time range?
ts_streams = {k: v for k, v in streams.items() if 'start' in v}
if ts_streams:
    all_starts = [v['start'] for v in ts_streams.values()]
    all_ends = [v['end'] for v in ts_streams.values()]
    print(f"\nAlignment:")
    print(f"  Latest start: {max(all_starts):.1f} (from {[k for k,v in ts_streams.items() if v['start']==max(all_starts)]})")
    print(f"  Earliest end: {min(all_ends):.1f} (from {[k for k,v in ts_streams.items() if v['end']==min(all_ends)]})")
    print(f"  Common range: {min(all_ends) - max(all_starts):.1f}s ({(min(all_ends) - max(all_starts))/60:.1f} min)")

# Also check the JSON sidecar
json_path = npz_path.replace('.npz', '.json')
if os.path.exists(json_path):
    import json
    with open(json_path) as f:
        meta = json.load(f)
    print(f"\nJSON metadata:")
    print(f"  duration: {meta.get('duration', '?')}s")
    print(f"  p1_role: {meta.get('p1_role', '?')}")
    print(f"  p2_role: {meta.get('p2_role', '?')}")
    markers = meta.get('markers', {})
    if isinstance(markers, dict):
        for k, v in sorted(markers.items()):
            print(f"  marker: {k} = {v}")
    elif isinstance(markers, list):
        for m in markers[:20]:
            print(f"  marker: {m}")
