"""3×3 metric × coupling model comparison for EEG temporal localization.

Metrics:  PLV, CCorr, Envelope correlation
Models:   Mixing (linear signal sum), Kuramoto (phase attractor), Amplitude (power co-modulation)

Expected results:
  - PLV optimal for mixing model (phase scales linearly with κ)
  - CCorr potentially better than PLV if lower spurious rate
  - Envelope optimal for amplitude model (designed for it)
  - Kuramoto: pure phase coupling → PLV/CCorr should dominate

Usage:
  python scripts/_test_metric_model_matrix.py
  python scripts/_test_metric_model_matrix.py --kappas 0.2 --k 50   # quick
"""
import sys, os, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import (generate_coupling_gate, find_valid_window,
                                inject_eeg_coupling_spatial,
                                inject_eeg_coupling_kuramoto,
                                inject_eeg_coupling_amplitude)
from cadence.significance.coherence_localization import (
    wpli_temporal_localization, _min_event_filter)
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

# ── Helpers ──────────────────────────────────────────────────────────────
freqs = np.logspace(np.log10(2.0), np.log10(40.0), 30)


def evaluate(mask, gate_at_win):
    nc = gate_at_win.sum()
    nn = (~gate_at_win).sum()
    hit = float((mask & gate_at_win).sum() / max(nc, 1))
    fa = float((mask & ~gate_at_win).sum() / max(nn, 1))
    return hit, fa


def best_hit_at_fa5(z_agg, gate_win, stride_s):
    best = 0.0
    for thr in np.arange(0.4, 3.0, 0.1):
        m = z_agg > thr
        m = _min_event_filter(m, max(1, int(5.0 / stride_s)))
        h, f = evaluate(m, gate_win)
        if 0.03 <= f <= 0.07:
            best = max(best, h)
    return best


def gate_at_windows(win_times, gate, t_arr):
    return np.interp(win_times, t_arr, gate.astype(float)) > 0.5


# ── Coupling models ──────────────────────────────────────────────────────
def inject_mixing(p1, p2, gate, kappa):
    return inject_eeg_coupling_spatial(
        p1, p2, gate, kappa,
        center_ch=CENTER_CH, decay_sigma=DECAY_SIGMA, lag_samp=LAG_SAMP)


def inject_kuramoto(p1, p2, gate, kappa):
    return inject_eeg_coupling_kuramoto(
        p1, p2, gate, kappa,
        center_ch=CENTER_CH, decay_sigma=DECAY_SIGMA, lag_samp=LAG_SAMP,
        band=(4.0, 8.0), fs=FS)


def inject_amplitude(p1, p2, gate, kappa):
    return inject_eeg_coupling_amplitude(
        p1, p2, gate, kappa,
        center_ch=CENTER_CH, decay_sigma=DECAY_SIGMA,
        band=(4.0, 8.0), fs=FS)


models = {
    'mixing': inject_mixing,
    'kuramoto': inject_kuramoto,
    'amplitude': inject_amplitude,
}

metrics = ['plv', 'ccorr', 'envelope']

# ── Main sweep ───────────────────────────────────────────────────────────
# results[kappa][model][metric] = (hit, z_coupled, z_null, time)
results = {}

gate = generate_coupling_gate(N, FS, {
    'duty_cycle': 0.10, 'event_range_s': (5, 20), 'ramp_s': 1.0
}, seed=args.seed)

for kappa in args.kappas:
    results[kappa] = {}
    print(f"\n{'#'*70}")
    print(f"  KAPPA = {kappa}")
    print(f"{'#'*70}")

    for model_name, inject_fn in models.items():
        results[kappa][model_name] = {}
        print(f"\n  --- Model: {model_name} ---")

        t0 = time.perf_counter()
        p2_coupled, kappa_per_ch = inject_fn(
            p1_eeg, p2_eeg, gate, kappa)
        inject_time = time.perf_counter() - t0

        # Re-normalize
        p1_c = p1_eeg.copy()
        for ch in range(N_CH):
            for sig in [p1_c, p2_coupled]:
                mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
                sig[:, ch] = (sig[:, ch] - mu) / sd

        oracle_ch = [i for i in np.argsort(-kappa_per_ch)
                     if kappa_per_ch[i] > 0.03]
        print(f"    Coupled: {len(oracle_ch)} ch (inject: {inject_time:.1f}s)")

        for metric in metrics:
            t0 = time.perf_counter()
            _, z_agg, _, diag = wpli_temporal_localization(
                p1_c, p2_coupled, FS,
                channels=oracle_ch, center_freqs=freqs,
                n_surrogates=args.k, n_cycles=[3, 7],
                window_s=20.0, stride_s=0.5, smooth_s=15.0,
                target_fa=0.03, min_event_s=5.0, metric=metric,
                seed=args.seed, device=device)
            elapsed = time.perf_counter() - t0

            gw = gate_at_windows(diag['win_times'], gate, t_eeg)
            zc = float(z_agg[gw].mean()) if gw.any() else 0
            zn = float(z_agg[~gw].mean()) if (~gw).any() else 0
            hit = best_hit_at_fa5(z_agg, gw, 0.5)

            results[kappa][model_name][metric] = {
                'hit': hit, 'z_coupled': zc, 'z_null': zn, 'time': elapsed
            }
            print(f"    {metric:>10}: hit@FA5%={hit:5.1%}  "
                  f"z_c={zc:+.3f}  z_n={zn:+.3f}  ({elapsed:.1f}s)")

# ── Null test ────────────────────────────────────────────────────────────
print(f"\n{'#'*70}")
print(f"  NULL TEST (kappa=0)")
print(f"{'#'*70}")

null_ch = list(range(5))
for metric in metrics:
    _, z_null, _, d_null = wpli_temporal_localization(
        p1_eeg, p2_eeg, FS,
        channels=null_ch, center_freqs=freqs,
        n_surrogates=args.k, n_cycles=[3, 7],
        window_s=20.0, stride_s=0.5, smooth_s=15.0,
        target_fa=0.03, min_event_s=5.0, metric=metric,
        seed=args.seed, device=device)
    gw = gate_at_windows(d_null['win_times'], gate, t_eeg)
    fa = float((z_null > d_null['z_threshold'])[~gw].mean()) if (~gw).any() else 0
    print(f"  {metric:>10}: FA={fa:.1%}  z_mean={z_null.mean():.3f}  "
          f"z_max={z_null.max():.3f}")

# ── Summary tables ───────────────────────────────────────────────────────
print(f"\n\n{'='*80}")
print(f"  SUMMARY: Hit rate at FA~5%")
print(f"{'='*80}")

for kappa in args.kappas:
    print(f"\n  kappa = {kappa}")
    header = f"  {'model':>12}"
    for m in metrics:
        header += f"  {m:>10}"
    print(header)
    print(f"  {'-'*12}" + f"  {'-'*10}" * len(metrics))

    for model_name in models:
        row = f"  {model_name:>12}"
        for metric in metrics:
            r = results[kappa][model_name][metric]
            row += f"  {r['hit']:>9.1%}"
        print(row)

# z_coupled summary
print(f"\n\n{'='*80}")
print(f"  SUMMARY: z_coupled (mean z during coupling windows)")
print(f"{'='*80}")

for kappa in args.kappas:
    print(f"\n  kappa = {kappa}")
    header = f"  {'model':>12}"
    for m in metrics:
        header += f"  {m:>10}"
    print(header)
    print(f"  {'-'*12}" + f"  {'-'*10}" * len(metrics))

    for model_name in models:
        row = f"  {model_name:>12}"
        for metric in metrics:
            r = results[kappa][model_name][metric]
            row += f"  {r['z_coupled']:>+9.3f}"
        print(row)

# Best metric per model
print(f"\n\n{'='*80}")
print(f"  BEST METRIC PER MODEL (at each kappa)")
print(f"{'='*80}")
for kappa in args.kappas:
    print(f"\n  kappa = {kappa}:")
    for model_name in models:
        best_metric = max(metrics,
                          key=lambda m: results[kappa][model_name][m]['hit'])
        best_hit = results[kappa][model_name][best_metric]['hit']
        print(f"    {model_name:>12} → {best_metric} ({best_hit:.1%})")

print(f"\n{'='*80}")
