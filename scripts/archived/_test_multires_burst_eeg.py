"""Multi-resolution PLV test with realistic theta-burst coupling patterns.

Theta coupling comes in bursts (100-300ms), in trains (3-10 bursts, ~200ms gaps).
This is the scenario where multi-res PLV should outperform single 20s windows:
a 2s window catches an entire burst train while 20s averages it out.

Compares:
  (A) Baseline: 20s window, 15s smooth
  (B) Multi-res: 2+5+10+20s, auto smooth
  (C) Single 2s window (lower bound)
  (D) Single 5s window

Two coupling profiles:
  [sustained] — Standard 5-20s continuous events (reference)
  [burst]     — Burst trains: 3-10 bursts × 100-300ms, 200ms gaps, 1-3s inter-train

Usage:
  python scripts/_test_multires_burst_eeg.py
  python scripts/_test_multires_burst_eeg.py --kappas 0.2 --k 50   # quick
"""
import sys, os, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import (generate_coupling_gate, generate_burst_coupling_gate,
                                find_valid_window, inject_eeg_coupling_spatial)
from cadence.significance.coherence_localization import (
    wpli_temporal_localization, multires_plv_temporal_localization,
    _min_event_filter)
from cadence.constants import EPOC_CHANNEL_NAMES

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=100, help='Surrogates')
parser.add_argument('--duration', type=int, default=1800, help='Duration (s)')
parser.add_argument('--kappas', type=float, nargs='+', default=[0.1, 0.2, 0.4])
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

FS = 256.0
N_CH = 14
CENTER_CH = 2
DECAY_SIGMA = 0.5
LAG_SAMP = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Config: duration={args.duration}s, K={args.k}, kappas={args.kappas}\n")

# ── Load EEG ─────────────────────────────────────────────────────────────
cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
sess = [(n, load_session_from_cache(p, cfg)) for n, p in entries]
sess = [(n, s) for n, s in sess if s is not None and 'p1_eeg' in s]
s1, s2 = sess[-1][1], sess[0][1]
window = find_valid_window(s1, min_duration=args.duration)

p1_ts = s1['p1_eeg_ts']
p1_m = (p1_ts >= window[0]) & (p1_ts < window[1])
p1_eeg = s1['p1_eeg'][p1_m].copy()

p2_ts = s2['p2_eeg_ts']
p2_s = float(p2_ts[0])
p2_m = (p2_ts >= p2_s) & (p2_ts < p2_s + args.duration)
p2_eeg = s2['p2_eeg'][p2_m].copy()

N = min(len(p1_eeg), len(p2_eeg), int(args.duration * FS))
t_target = np.linspace(0, args.duration, N)
p2_tw = p2_ts[p2_m] - p2_s
p2_eeg = np.stack([np.interp(t_target, p2_tw, p2_eeg[:, c])
                    for c in range(N_CH)], axis=1).astype(np.float32)
p1_eeg = p1_eeg[:N]
t_eeg = np.arange(N) / FS

for ch in range(N_CH):
    for sig in [p1_eeg, p2_eeg]:
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

print(f"EEG: ({N}, {N_CH}) @ {FS} Hz, {args.duration}s\n")

# ── Shared params ────────────────────────────────────────────────────────
freqs = np.logspace(np.log10(2.0), np.log10(40.0), 30)


def evaluate(mask, gate_at_win):
    nc = gate_at_win.sum()
    nn = (~gate_at_win).sum()
    hit = float((mask & gate_at_win).sum() / max(nc, 1))
    fa = float((mask & ~gate_at_win).sum() / max(nn, 1))
    return hit, fa


def best_hit_at_fa5(z_agg, gate_win, stride_s):
    """Sweep thresholds, return best hit rate where 3% ≤ FA ≤ 7%."""
    best = 0.0
    for thr in np.arange(0.4, 3.0, 0.1):
        m = z_agg > thr
        m = _min_event_filter(m, max(1, int(5.0 / stride_s)))
        h, f = evaluate(m, gate_win)
        if 0.03 <= f <= 0.07:
            best = max(best, h)
    return best


def roc_sweep(z_agg, gate_win, stride_s, label):
    """Print ROC and return best hit at FA~5%."""
    zc = z_agg[gate_win].mean() if gate_win.any() else 0
    zn = z_agg[~gate_win].mean() if (~gate_win).any() else 0
    print(f"      {label}  z_coupled={zc:.3f} z_null={zn:.3f}")
    print(f"        {'Thr':>5} {'Hit':>6} {'FA':>6}")
    best = 0.0
    for thr in np.arange(0.4, 3.0, 0.2):
        m = z_agg > thr
        m = _min_event_filter(m, max(1, int(5.0 / stride_s)))
        h, f = evaluate(m, gate_win)
        marker = " *" if 0.03 <= f <= 0.07 else ""
        if 0.03 <= f <= 0.07:
            best = max(best, h)
        print(f"        {thr:>5.1f} {h:>5.1%} {f:>5.1%}{marker}")
    return best


def gate_at_windows(win_times, gate, t_eeg):
    """Map gate to window time grid."""
    return np.interp(win_times, t_eeg, gate.astype(float)) > 0.5


# ── Coupling profiles ────────────────────────────────────────────────────
def make_sustained_gate():
    return generate_coupling_gate(N, FS, {
        'duty_cycle': 0.10, 'event_range_s': (5, 20), 'ramp_s': 1.0
    }, seed=args.seed), None


def make_burst_gate():
    gate, info = generate_burst_coupling_gate(N, FS, {
        'macro_duty_cycle': 0.10,
        'macro_event_range_s': (5, 20),
        'bursts_per_train': (3, 10),
        'burst_duration_ms': (100, 300),
        'inter_burst_ms': 200,
        'inter_train_s': (1.0, 3.0),
        'ramp_ms': 20,
    }, seed=args.seed)
    # Return burst gate for INJECTION, macro gate for EVALUATION
    # (period-level detection: did we detect the macro coupling window?)
    info['eval_gate'] = info['macro_gate']
    return gate, info


# ── Main sweep ───────────────────────────────────────────────────────────
results = {}  # {(kappa, profile, method): hit}

profiles = {
    'sustained': make_sustained_gate,
    'burst': make_burst_gate,
}

for profile_name, gate_fn in profiles.items():
    inject_gate, info = gate_fn()
    # inject_gate: used for coupling injection (burst or sustained)
    # eval_gate: used for hit/FA evaluation (always macro-level periods)
    if info and 'eval_gate' in info:
        eval_gate = info['eval_gate']
    else:
        eval_gate = inject_gate  # sustained: inject == eval
    gate = inject_gate

    print(f"\n{'#'*70}")
    print(f"  PROFILE: {profile_name.upper()}")
    if info and 'n_trains' in info:
        print(f"  Burst stats: {info['n_trains']} trains, "
              f"{info['n_bursts_total']} bursts, "
              f"burst duty={info['overall_duty']:.1%}, "
              f"macro duty={eval_gate.mean():.1%}, "
              f"{info['burst_duty_within_macro']:.0%} fill within macro")
    else:
        print(f"  Sustained: duty={gate.mean():.1%}")
    print(f"{'#'*70}")

    for kappa in args.kappas:
        print(f"\n  {'='*60}")
        print(f"  kappa={kappa}, profile={profile_name}")
        print(f"  {'='*60}")

        p2_coupled, kappa_per_ch = inject_eeg_coupling_spatial(
            p1_eeg, p2_eeg, gate, kappa,
            center_ch=CENTER_CH, decay_sigma=DECAY_SIGMA, lag_samp=LAG_SAMP)

        p1_c = p1_eeg.copy()
        for ch in range(N_CH):
            for sig in [p1_c, p2_coupled]:
                mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
                sig[:, ch] = (sig[:, ch] - mu) / sd

        oracle_ch = [i for i in np.argsort(-kappa_per_ch)
                     if kappa_per_ch[i] > 0.03]
        print(f"    Coupled: {len(oracle_ch)} ch, "
              f"max kappa={kappa_per_ch.max():.3f}")

        # (A) Baseline: 20s window
        print(f"\n    (A) Baseline: 20s window")
        t0 = time.perf_counter()
        _, z_a, _, d_a = wpli_temporal_localization(
            p1_c, p2_coupled, FS,
            channels=oracle_ch, center_freqs=freqs, n_surrogates=args.k,
            n_cycles=[3, 7], window_s=20.0, stride_s=0.5, smooth_s=15.0,
            target_fa=0.03, min_event_s=5.0, metric='plv',
            seed=args.seed, device=device)
        gw_a = gate_at_windows(d_a['win_times'], eval_gate, t_eeg)
        h_a = roc_sweep(z_a, gw_a, 0.5, f"({time.perf_counter()-t0:.1f}s)")
        results[(kappa, profile_name, 'baseline_20s')] = h_a

        # (B) Multi-res: 2+5+10+20s
        print(f"\n    (B) Multi-res: 2+5+10+20s")
        t0 = time.perf_counter()
        _, z_b, _, d_b = multires_plv_temporal_localization(
            p1_c, p2_coupled, FS,
            channels=oracle_ch, center_freqs=freqs, n_surrogates=args.k,
            n_cycles=[3, 7], windows_s=(2.0, 5.0, 10.0, 20.0), stride_s=0.5,
            target_fa=0.03, min_event_s=5.0, metric='plv',
            surrogate_method='circular', seed=args.seed, device=device)
        gw_b = gate_at_windows(d_b['win_times'], eval_gate, t_eeg)
        h_b = roc_sweep(z_b, gw_b, 0.5, f"({time.perf_counter()-t0:.1f}s)")
        results[(kappa, profile_name, 'multires')] = h_b
        print(f"      Scale dist: {d_b['scale_distribution']}")
        print(f"      Per-scale z (coupled): {d_b['per_scale_z_coupled']}")

        # (C) Single 2s window (to see if very short windows help alone)
        print(f"\n    (C) Single 2s window")
        t0 = time.perf_counter()
        _, z_c, _, d_c = wpli_temporal_localization(
            p1_c, p2_coupled, FS,
            channels=oracle_ch, center_freqs=freqs, n_surrogates=args.k,
            n_cycles=[3, 7], window_s=2.0, stride_s=0.5, smooth_s=1.5,
            target_fa=0.03, min_event_s=5.0, metric='plv',
            seed=args.seed, device=device)
        gw_c = gate_at_windows(d_c['win_times'], eval_gate, t_eeg)
        h_c = roc_sweep(z_c, gw_c, 0.5, f"({time.perf_counter()-t0:.1f}s)")
        results[(kappa, profile_name, 'single_2s')] = h_c

        # (D) Single 5s window
        print(f"\n    (D) Single 5s window")
        t0 = time.perf_counter()
        _, z_d, _, d_d = wpli_temporal_localization(
            p1_c, p2_coupled, FS,
            channels=oracle_ch, center_freqs=freqs, n_surrogates=args.k,
            n_cycles=[3, 7], window_s=5.0, stride_s=0.5, smooth_s=3.75,
            target_fa=0.03, min_event_s=5.0, metric='plv',
            seed=args.seed, device=device)
        gw_d = gate_at_windows(d_d['win_times'], eval_gate, t_eeg)
        h_d = roc_sweep(z_d, gw_d, 0.5, f"({time.perf_counter()-t0:.1f}s)")
        results[(kappa, profile_name, 'single_5s')] = h_d


# ── Summary ──────────────────────────────────────────────────────────────
methods = ['baseline_20s', 'multires', 'single_2s', 'single_5s']

print(f"\n\n{'='*80}")
print(f"  SUMMARY: Hit rate at FA~5%")
print(f"{'='*80}")

for profile_name in profiles:
    print(f"\n  --- {profile_name.upper()} ---")
    header = f"  {'kappa':>6}"
    for m in methods:
        header += f"  {m:>12}"
    print(header)
    print(f"  {'-'*6}" + f"  {'-'*12}" * len(methods))

    for kappa in args.kappas:
        row = f"  {kappa:>6.2f}"
        baseline = results.get((kappa, profile_name, 'baseline_20s'), 0)
        for m in methods:
            val = results.get((kappa, profile_name, m), 0)
            delta = val - baseline
            sign = '+' if delta >= 0 else ''
            if m == 'baseline_20s':
                row += f"  {val:>11.1%}"
            else:
                row += f"  {val:>5.1%}({sign}{delta:.0%})"
            # Pad to 12
        print(row)

print(f"\n{'='*80}")
