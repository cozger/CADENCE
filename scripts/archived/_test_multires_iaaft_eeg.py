"""Multi-resolution PLV + IAAFT surrogate test for EEG temporal localization.

Compares:
  (A) Baseline: single 20s window, circular shift (current best)
  (B) Multi-res: 2+5+10+20s windows, circular shift
  (C) Multi-res + IAAFT: 2+5+10+20s windows, IAAFT surrogates

Across kappa = 0.1, 0.2, 0.4 with spatial decay (center F3, sigma=0.5).

Usage:
  python scripts/_test_multires_iaaft_eeg.py              # full test
  python scripts/_test_multires_iaaft_eeg.py --no-iaaft    # skip IAAFT (faster)
  python scripts/_test_multires_iaaft_eeg.py --k 50        # fewer surrogates
  python scripts/_test_multires_iaaft_eeg.py --duration 600 # shorter session
"""
import sys, os, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import (generate_coupling_gate, find_valid_window,
                                inject_eeg_coupling_spatial)
from cadence.significance.coherence_localization import (
    wpli_temporal_localization, multires_plv_temporal_localization,
    _min_event_filter)
from cadence.constants import EPOC_CHANNEL_NAMES

# ── Args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--no-iaaft', action='store_true', help='Skip IAAFT tests')
parser.add_argument('--k', type=int, default=100, help='Number of surrogates')
parser.add_argument('--duration', type=int, default=1800, help='Session duration (s)')
parser.add_argument('--kappas', type=float, nargs='+', default=[0.1, 0.2, 0.4])
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

FS = 256.0
DURATION = args.duration
N_CH = 14
SEED = args.seed
CENTER_CH = 2   # F3
DECAY_SIGMA = 0.5
LAG_SAMP = 8    # ~30ms
N_SURROGATES = args.k

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Config: duration={DURATION}s, K={N_SURROGATES}, "
      f"kappas={args.kappas}, iaaft={'OFF' if args.no_iaaft else 'ON'}")

# ── Load EEG from different sessions (pseudo-dyad) ──────────────────────
cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
sess = [(n, load_session_from_cache(p, cfg)) for n, p in entries]
sess = [(n, s) for n, s in sess if s is not None and 'p1_eeg' in s]
print(f"Found {len(sess)} sessions with EEG")
s1, s2 = sess[-1][1], sess[0][1]
window = find_valid_window(s1, min_duration=DURATION)

p1_ts = s1['p1_eeg_ts']
p1_m = (p1_ts >= window[0]) & (p1_ts < window[1])
p1_eeg = s1['p1_eeg'][p1_m].copy()

p2_ts = s2['p2_eeg_ts']
p2_s = float(p2_ts[0])
p2_m = (p2_ts >= p2_s) & (p2_ts < p2_s + DURATION)
p2_eeg = s2['p2_eeg'][p2_m].copy()

N = min(len(p1_eeg), len(p2_eeg), int(DURATION * FS))
t_target = np.linspace(0, DURATION, N)
p2_tw = p2_ts[p2_m] - p2_s
p2_eeg = np.stack([np.interp(t_target, p2_tw, p2_eeg[:, c])
                    for c in range(N_CH)], axis=1).astype(np.float32)
p1_eeg = p1_eeg[:N]
t_eeg = np.arange(N) / FS

# Z-normalize
for ch in range(N_CH):
    for sig in [p1_eeg, p2_eeg]:
        mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
        sig[:, ch] = (sig[:, ch] - mu) / sd

print(f"EEG loaded: ({N}, {N_CH}) @ {FS} Hz, {DURATION}s\n")

# ── Shared parameters ────────────────────────────────────────────────────
freqs = np.logspace(np.log10(2.0), np.log10(40.0), 30)


def evaluate(mask, gate_at_win):
    """Hit/FA from binary mask vs ground truth gate."""
    nc = gate_at_win.sum()
    nn = (~gate_at_win).sum()
    hit = float((mask & gate_at_win).sum() / max(nc, 1))
    fa = float((mask & ~gate_at_win).sum() / max(nn, 1))
    return hit, fa


def roc_sweep(z_agg, gate_win, stride_s, label):
    """Sweep thresholds and return best hit at FA~5%."""
    zc = z_agg[gate_win].mean() if gate_win.any() else 0
    zn = z_agg[~gate_win].mean() if (~gate_win).any() else 0
    print(f"    {label}  z_coupled={zc:.3f} z_null={zn:.3f}")
    print(f"      {'Thr':>6} {'Hit':>7} {'FA':>7}")
    best_hit_at_5 = 0
    for thr in np.arange(0.6, 2.8, 0.2):
        m = z_agg > thr
        m = _min_event_filter(m, max(1, int(5.0 / stride_s)))
        h, f = evaluate(m, gate_win)
        marker = ""
        if 0.03 <= f <= 0.07:
            marker = " <-- FA~5%"
            best_hit_at_5 = max(best_hit_at_5, h)
        print(f"      {thr:>6.2f} {h:>6.1%} {f:>6.1%}{marker}")
    return best_hit_at_5


# ── Main sweep ───────────────────────────────────────────────────────────
results = {}  # {(kappa, method): hit_at_fa5}

for kappa in args.kappas:
    print(f"\n{'='*70}")
    print(f"  KAPPA = {kappa}")
    print(f"{'='*70}")

    # Generate coupling gate
    gate = generate_coupling_gate(N, FS, {
        'duty_cycle': 0.10, 'event_range_s': (5, 20), 'ramp_s': 1.0},
        seed=SEED)

    # Inject coupling
    p2_coupled, kappa_per_ch = inject_eeg_coupling_spatial(
        p1_eeg, p2_eeg, gate, kappa,
        center_ch=CENTER_CH, decay_sigma=DECAY_SIGMA, lag_samp=LAG_SAMP)

    # Re-normalize
    p1_c = p1_eeg.copy()
    for ch in range(N_CH):
        for sig in [p1_c, p2_coupled]:
            mu, sd = sig[:, ch].mean(), max(sig[:, ch].std(), 1e-8)
            sig[:, ch] = (sig[:, ch] - mu) / sd

    oracle_ch = [i for i in np.argsort(-kappa_per_ch)
                 if kappa_per_ch[i] > 0.03]
    print(f"  Coupled: {[EPOC_CHANNEL_NAMES[i] for i in oracle_ch]} "
          f"(kappa: {[f'{kappa_per_ch[i]:.3f}' for i in oracle_ch]})")
    print(f"  Duty: {gate.mean():.1%}")

    # ── (A) Baseline: single 20s window, circular shift ──────────────────
    print(f"\n  (A) Baseline: 20s window, circular shift")
    t0 = time.perf_counter()
    mask_a, z_a, _, diag_a = wpli_temporal_localization(
        p1_c, p2_coupled, FS,
        channels=oracle_ch, center_freqs=freqs, n_surrogates=N_SURROGATES,
        n_cycles=[3, 7], window_s=20.0, stride_s=0.5, smooth_s=15.0,
        target_fa=0.03, min_event_s=5.0, metric='plv',
        seed=SEED, device=device)
    t_a = time.perf_counter() - t0
    wt_a = diag_a['win_times']
    gw_a = np.interp(wt_a, t_eeg, gate.astype(float)) > 0.5
    h_a = roc_sweep(z_a, gw_a, 0.5, f"({t_a:.1f}s)")
    results[(kappa, 'baseline')] = h_a

    # ── (B) Multi-res PLV, circular shift ────────────────────────────────
    print(f"\n  (B) Multi-res PLV (2+5+10+20s), circular shift")
    t0 = time.perf_counter()
    mask_b, z_b, psz_b, diag_b = multires_plv_temporal_localization(
        p1_c, p2_coupled, FS,
        channels=oracle_ch, center_freqs=freqs, n_surrogates=N_SURROGATES,
        n_cycles=[3, 7], windows_s=(2.0, 5.0, 10.0, 20.0), stride_s=0.5,
        target_fa=0.03, min_event_s=5.0, metric='plv',
        surrogate_method='circular', seed=SEED, device=device)
    t_b = time.perf_counter() - t0
    wt_b = diag_b['win_times']
    gw_b = np.interp(wt_b, t_eeg, gate.astype(float)) > 0.5
    h_b = roc_sweep(z_b, gw_b, 0.5, f"({t_b:.1f}s)")
    results[(kappa, 'multires')] = h_b
    print(f"    Scale dist: {diag_b['scale_distribution']}")
    print(f"    Per-scale z (coupled): {diag_b['per_scale_z_coupled']}")

    # ── (C) Multi-res PLV + IAAFT (optional) ─────────────────────────────
    if not args.no_iaaft:
        print(f"\n  (C) Multi-res PLV + IAAFT surrogates")
        t0 = time.perf_counter()
        mask_c, z_c, psz_c, diag_c = multires_plv_temporal_localization(
            p1_c, p2_coupled, FS,
            channels=oracle_ch, center_freqs=freqs,
            n_surrogates=N_SURROGATES,
            n_cycles=[3, 7], windows_s=(2.0, 5.0, 10.0, 20.0), stride_s=0.5,
            target_fa=0.03, min_event_s=5.0, metric='plv',
            surrogate_method='iaaft', seed=SEED, device=device)
        t_c = time.perf_counter() - t0
        wt_c = diag_c['win_times']
        gw_c = np.interp(wt_c, t_eeg, gate.astype(float)) > 0.5
        h_c = roc_sweep(z_c, gw_c, 0.5, f"({t_c:.1f}s)")
        results[(kappa, 'multires_iaaft')] = h_c
        print(f"    Scale dist: {diag_c['scale_distribution']}")

# ── Null test ────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  NULL TEST (kappa=0, multi-res circular)")
print(f"{'='*70}")

oracle_ch_null = list(range(5))  # arbitrary 5 channels
mask_null, z_null, _, diag_null = multires_plv_temporal_localization(
    p1_eeg, p2_eeg, FS,
    channels=oracle_ch_null, center_freqs=freqs, n_surrogates=N_SURROGATES,
    n_cycles=[3, 7], windows_s=(2.0, 5.0, 10.0, 20.0), stride_s=0.5,
    target_fa=0.03, min_event_s=5.0, metric='plv',
    surrogate_method='circular', seed=SEED, device=device)
print(f"  Coupling fraction: {mask_null.mean():.1%}")
print(f"  z mean: {z_null.mean():.3f}, max: {z_null.max():.3f}")

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  SUMMARY: Hit rate at FA~5%")
print(f"{'='*70}")

methods = ['baseline', 'multires']
if not args.no_iaaft:
    methods.append('multires_iaaft')

header = f"  {'Kappa':>6}"
for m in methods:
    header += f"  {m:>14}"
print(header)
print(f"  {'-'*6}" + f"  {'-'*14}" * len(methods))

for kappa in args.kappas:
    row = f"  {kappa:>6.2f}"
    for m in methods:
        val = results.get((kappa, m), 0)
        row += f"  {val:>13.1%}"
    print(row)

    # Delta row
    baseline_val = results.get((kappa, 'baseline'), 0)
    delta_row = f"  {'':>6}"
    for m in methods:
        if m == 'baseline':
            delta_row += f"  {'(ref)':>14}"
        else:
            val = results.get((kappa, m), 0)
            d = val - baseline_val
            sign = '+' if d >= 0 else ''
            delta_row += f"  {f'{sign}{d:.1%}':>14}"
    print(delta_row)

print(f"{'='*70}")
print(f"  Null FA: {mask_null.mean():.1%}")
print(f"{'='*70}")
