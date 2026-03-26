"""Inspect bycycle feature distributions to tune burst thresholds for EPOC."""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from bycycle import Bycycle
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.conditions import parse_condition_intervals
from cadence.constants import EPOC_CHANNEL_NAMES

cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
name, path = next((n, p) for n, p in entries if 'y_06' in n)
session = load_session_from_cache(path, config=cfg)
intervals = parse_condition_intervals(session)
mk = [(s, e) for s, e, c in intervals if c == 'meditate_K'][0]

p1_eeg = session['p1_eeg']
p2_eeg = session['p2_eeg']
p1_ts = session['p1_eeg_ts']
p2_ts = session['p2_eeg_ts']

FS = 256.0
BAND = (4, 8)

# Extract meditate_K for both participants
for p_label, eeg, ts in [('P1_patient', p1_eeg, p1_ts),
                           ('P2_therapist', p2_eeg, p2_ts)]:
    m = (ts >= mk[0]) & (ts < mk[1])
    sig_all = eeg[m]
    # avg ref
    sig_all = sig_all - sig_all.mean(axis=1, keepdims=True)

    print(f"\n{'='*60}")
    print(f"  {p_label} — meditate_K")
    print(f"{'='*60}")

    # Run bycycle on F3 (channel 2) with NO burst thresholds first
    ch = 2  # F3
    sig = sig_all[:, ch].astype(np.float64)
    sig = (sig - sig.mean()) / max(sig.std(), 1e-8)

    # Extract features without burst detection
    from bycycle.features import compute_shape_features, compute_burst_features
    from bycycle.cyclepoints import find_extrema, find_zerox
    from neurodsp.filt import filter_signal

    sig_filt = filter_signal(sig, FS, 'bandpass', BAND)

    print(f"  {EPOC_CHANNEL_NAMES[ch]}: signal std={sig.std():.2f}")

    bc = Bycycle(center_extrema='peak', burst_method='cycles',
                  thresholds={
                      'amp_fraction_threshold': 0.,
                      'amp_consistency_threshold': 0.5,
                      'period_consistency_threshold': 0.5,
                      'monotonicity_threshold': 0.8,
                      'min_n_cycles': 3,
                  })
    bc.fit(sig, FS, f_range=BAND)
    df = bc.df_features

    # Print distribution of burst-related features
    burst_feats = ['amp_fraction', 'amp_consistency', 'period_consistency',
                    'monotonicity', 'is_burst']
    print(f"\n  Burst feature distributions:")
    for feat in burst_feats:
        if feat in df:
            vals = df[feat].values.astype(float)
            print(f"    {feat:>25}: mean={vals.mean():.3f}  "
                  f"p25={np.percentile(vals,25):.3f}  "
                  f"p50={np.percentile(vals,50):.3f}  "
                  f"p75={np.percentile(vals,75):.3f}  "
                  f"p90={np.percentile(vals,90):.3f}")

    # Test different threshold combinations
    print(f"\n  Burst fraction at different thresholds:")
    for amp_c in [0.3, 0.4, 0.5]:
        for mono in [0.6, 0.7, 0.8]:
            is_b = ((df['amp_fraction'] > 0) &
                     (df['amp_consistency'] > amp_c) &
                     (df['period_consistency'] > 0.5) &
                     (df['monotonicity'] > mono))
            frac = is_b.mean()
            print(f"    amp_c={amp_c}, mono={mono}: {frac:.1%} "
                  f"({is_b.sum()} cycles)")

    # Also check a few more channels
    print(f"\n  Per-channel burst fractions (default thresholds):")
    for ch_i in range(14):
        sig_ch = sig_all[:, ch_i].astype(np.float64)
        sig_ch = (sig_ch - sig_ch.mean()) / max(sig_ch.std(), 1e-8)
        bc2 = Bycycle(center_extrema='peak', burst_method='cycles',
                       thresholds={
                           'amp_fraction_threshold': 0.,
                           'amp_consistency_threshold': 0.5,
                           'period_consistency_threshold': 0.5,
                           'monotonicity_threshold': 0.8,
                           'min_n_cycles': 3,
                       })
        bc2.fit(sig_ch, FS, f_range=BAND)
        bf = bc2.df_features['is_burst'].mean()
        nc = len(bc2.df_features)
        print(f"    {EPOC_CHANNEL_NAMES[ch_i]:>4}: {bf:5.1%} ({nc} cycles)")
