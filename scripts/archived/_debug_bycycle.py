"""Debug bycycle on single channel from y_06."""
import sys, os, warnings, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.conditions import parse_condition_intervals

cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
name, path = next((n, p) for n, p in entries if 'y_06' in n)
session = load_session_from_cache(path, config=cfg)

p1_eeg = session['p1_eeg']
p1_ts = session['p1_eeg_ts']

intervals = parse_condition_intervals(session)
mk = [(s, e) for s, e, c in intervals if c == 'meditate_K'][0]
m = (p1_ts >= mk[0]) & (p1_ts < mk[1])
sig = p1_eeg[m, 2].astype(np.float64)
sig = sig - sig.mean()
sig = sig / sig.std()
print(f"Signal: {len(sig)} samples, {len(sig)/256:.1f}s")
print(f"mean={sig.mean():.4f}, std={sig.std():.4f}, min={sig.min():.2f}, max={sig.max():.2f}")

# Try without monkey-patch first to see the actual error
from bycycle import Bycycle
bc = Bycycle(center_extrema='peak', burst_method='cycles')
try:
    bc.fit(sig, 256.0, f_range=(4, 8))
    print(f"Cycles: {len(bc.df_features)}")
    if len(bc.df_features) > 0:
        print(bc.df_features[['volt_amp', 'period', 'time_rdsym', 'is_burst']].describe())
        print(f"\nBurst fraction: {bc.df_features['is_burst'].mean():.1%}")
        print(f"N burst cycles: {bc.df_features['is_burst'].sum()}")
        print(f"Cycle rate: {len(bc.df_features) / (len(sig)/256):.1f}/s")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()
    # Try without burst detection
    print("\nRetrying without burst detection...")
    from bycycle.features import compute_features
    try:
        df = compute_features(sig, 256.0, (4, 8))
        print(f"Features OK: {len(df)} cycles")
        print(df[['volt_amp', 'period', 'time_rdsym']].head())
    except Exception as e2:
        print(f"Features also failed: {e2}")
        traceback.print_exc()
