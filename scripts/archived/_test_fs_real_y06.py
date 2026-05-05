"""Real data test: 30 Hz vs 60 Hz (upsampled) BL Stage 2 on y_06.

Compares event detection, co-occurrence, and lag precision on real
facial data — where sharp onsets should benefit from higher sample rate.
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.conditions import parse_condition_intervals
from cadence.significance.bl_coupling import bl_two_stage_coupling

N_SURR = 200; SEED = 42
SMILE_AUS = [43, 44, 17]

cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
name, path = next((n, p) for n, p in entries if 'y_06' in n)
session = load_session_from_cache(path, config=cfg)

p1_role = session.get('p1_role', '?')
p2_role = session.get('p2_role', '?')
intervals = parse_condition_intervals(session)
print(f"{name}: P1={p1_role}, P2={p2_role}\n")


def resolve_bl(sess, role):
    p1r = sess.get('p1_role', 'therapist')
    if (role == 'therapist' and p1r == 'therapist') or \
       (role == 'patient' and p1r == 'patient'):
        return sess['p1_blendshapes'], sess['p1_blendshapes_ts']
    return sess['p2_blendshapes'], sess['p2_blendshapes_ts']


def extract_at_rate(sess, start_s, end_s, target_fs):
    """Extract BL for both roles, resampled to target_fs."""
    t_bl, t_ts = resolve_bl(sess, 'therapist')
    p_bl, p_ts = resolve_bl(sess, 'patient')

    t_m = (t_ts >= start_s) & (t_ts < end_s)
    p_m = (p_ts >= start_s) & (p_ts < end_s)
    dur = end_s - start_s
    N = int(dur * target_fs)
    n_au = min(t_bl.shape[1], p_bl.shape[1], 52)

    t_out = np.linspace(0, dur, N)

    t_seg = np.stack([np.interp(t_out, t_ts[t_m] - start_s, t_bl[t_m, c])
                       for c in range(n_au)], axis=1).astype(np.float32)
    p_seg = np.stack([np.interp(t_out, p_ts[p_m] - start_s, p_bl[p_m, c])
                       for c in range(n_au)], axis=1).astype(np.float32)

    return t_seg[:N], p_seg[:N]


def analyze_events(bl, fs, label):
    """Quick event analysis on smile composite."""
    comp = gaussian_filter1d(sum(bl[:, au] for au in SMILE_AUS if au < bl.shape[1]),
                              0.3 * fs)
    pks, props = find_peaks(comp, prominence=0.3, distance=int(3 * fs))
    proms = props['prominences']
    # Check sharpness: rise time in samples from half-prominence to peak
    rise_times = []
    for pk, prom in zip(pks, proms):
        half = comp[pk] - prom / 2
        # Walk backwards from peak to find half-prominence crossing
        for j in range(pk, max(0, pk - int(2 * fs)), -1):
            if comp[j] < half:
                rise_times.append((pk - j) / fs * 1000)  # ms
                break
    mean_rise = np.mean(rise_times) if rise_times else 0
    print(f"  {label}: {len(pks)} events, "
          f"mean_prom={proms.mean():.3f}, "
          f"mean_rise={mean_rise:.0f}ms")
    return len(pks), proms.mean() if len(pks) > 0 else 0, mean_rise


def run_pipeline(therapist_bl, patient_bl, fs, direction_label):
    """Run full BL pipeline and return key metrics."""
    t0 = time.perf_counter()
    r = bl_two_stage_coupling(
        therapist_bl, patient_bl, fs,
        max_lag_s=5.0, lag_step_s=0.1,
        smooth_s=3.0,  # fixed for fair comparison
        n_surrogates=N_SURR, target_fa=0.05,
        event_prominence=0.3, min_event_iei_s=3.0,
        n_surrogates_event=N_SURR,
        population_prior=(2.5, 1.0), seed=SEED)
    elapsed = time.perf_counter() - t0

    cats = getattr(r, 'catalogs', {})
    out = {}
    for comp_name in ['smile', 'brow', 'general']:
        cat = cats.get(comp_name)
        if cat:
            out[comp_name] = {
                'n_a': cat.n_events_a,
                'n_b': cat.n_events_b,
                'co': cat.n_cooccurrences,
                'p': cat.session_p_value,
                'mim': cat.n_mimicry,
                'ss': cat.n_shared_stimulus,
                'lag': cat.mean_lag if cat.mean_lag else 0,
            }
    return out, elapsed


# ── Run on conv_1 and conv_2 ────────────────────────────────────────────
target_conds = ['conv_1', 'conv_2']

for cond_name in target_conds:
    interval = [(s, e) for s, e, c in intervals if c == cond_name]
    if not interval:
        continue
    start, end = interval[0]
    dur = end - start

    print(f"\n{'='*70}")
    print(f"  {cond_name} ({dur:.0f}s)")
    print(f"{'='*70}")

    for fs_label, fs in [('30 Hz (native)', 30.0), ('60 Hz (upsampled)', 60.0)]:
        print(f"\n  --- {fs_label} ---")

        t_bl, p_bl = extract_at_rate(session, start, end, fs)
        print(f"  Shape: ({len(t_bl)}, {t_bl.shape[1]}) @ {fs} Hz")

        # Event analysis
        analyze_events(t_bl, fs, "Therapist")
        analyze_events(p_bl, fs, "Patient")

        # T→P
        r_tp, t_tp = run_pipeline(t_bl, p_bl, fs, 'T→P')
        print(f"\n  T→P ({t_tp:.1f}s):")
        for comp, m in r_tp.items():
            sig = "*" if m['p'] < 0.05 else ""
            print(f"    {comp:>8}: T={m['n_a']} P={m['n_b']} co={m['co']} "
                  f"p={m['p']:.3f}{sig} mim={m['mim']} ss={m['ss']} "
                  f"lag={m['lag']:.3f}s")

        # P→T
        r_pt, t_pt = run_pipeline(p_bl, t_bl, fs, 'P→T')
        print(f"  P→T ({t_pt:.1f}s):")
        for comp, m in r_pt.items():
            sig = "*" if m['p'] < 0.05 else ""
            print(f"    {comp:>8}: P={m['n_a']} T={m['n_b']} co={m['co']} "
                  f"p={m['p']:.3f}{sig} mim={m['mim']} ss={m['ss']} "
                  f"lag={m['lag']:.3f}s")

print(f"\n{'='*70}")
