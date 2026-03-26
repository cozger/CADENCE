"""Temporal Localization V4: Iteration 12 — 3 lag estimation approaches.

1. Event-triggered lag estimation (peak matching)
2. Multi-lag bank + cluster-mass (lag-agnostic)
3. Hierarchical (session anchor + narrow refinement)

All tested on the mixed coupling scenario (30 min, alternating/cyclical/mutual).
Compared against oracle lag and 60s windowed cross-corr baselines.
"""
import numpy as np
import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import find_valid_window, generate_coupling_gate
from cadence.significance.kim_filter import _estimate_ar

DURATION_S = 1800
FS = 30.0
N_JOBS = -1

HETERO_PROFILE = {
    43: 0.30, 44: 0.30, 6: 0.25, 7: 0.25,
    2: 0.15, 0: 0.12, 1: 0.12, 18: 0.10, 19: 0.10, 20: 0.08,
    5: 0.05, 49: 0.05, 50: 0.05, 29: 0.05, 30: 0.05,
    27: 0.04, 28: 0.04, 3: 0.03,
}
HETERO_AUS = sorted(HETERO_PROFILE.keys())
HETERO_KAPPAS = np.array([HETERO_PROFILE[c] for c in HETERO_AUS])


# ─────────────────────────────────────────────────────────────────────
# Data loading + coupling injection (reused from previous iterations)
# ─────────────────────────────────────────────────────────────────────

def load_sessions(cfg):
    session_entries = discover_cached_sessions(cfg['session_cache'])
    sess_list = [(n, load_session_from_cache(p, cfg))
                 for n, p in session_entries]
    sess_list = [(n, s) for n, s in sess_list if s is not None]
    s1, s2 = sess_list[-1][1], sess_list[0][1]
    window = find_valid_window(s1, min_duration=DURATION_S)
    t_start, t_end = window
    t_end = min(t_end, t_start + DURATION_S)

    def extract(sess, pfx):
        k = f'{pfx}_blendshapes'
        d, ts = sess[k], sess[f'{k}_ts']
        m = (ts >= t_start) & (ts < t_end)
        return d[m, :52].copy(), ts[m] - t_start

    p1, p1t = extract(s1, 'p1')
    p2, p2t = extract(s2, 'p2')
    t_s, t_e = max(float(p1t[0]), float(p2t[0])), min(float(p1t[-1]), float(p2t[-1]))
    T = int((t_e - t_s) * FS)
    times = np.linspace(t_s, t_e, T)
    p1s = np.stack([np.interp(times, p1t, p1[:, c]) for c in range(52)], axis=1)
    p2s = np.stack([np.interp(times, p2t, p2[:, c]) for c in range(52)], axis=1)
    for c in range(52):
        p1s[:, c] = (p1s[:, c] - p1s[:, c].mean()) / max(p1s[:, c].std(), 1e-8)
        p2s[:, c] = (p2s[:, c] - p2s[:, c].mean()) / max(p2s[:, c].std(), 1e-8)
    return p1s, p2s, T


def make_gate_segment(T, t_start_s, t_end_s, duty=0.10, seed=42):
    T_seg = int((t_end_s - t_start_s) * FS)
    gate_seg = generate_coupling_gate(T_seg, FS, {
        'duty_cycle': duty, 'event_range_s': (3, 15), 'ramp_s': 1.0,
    }, seed=seed)
    gate = np.zeros(T)
    s = int(t_start_s * FS)
    e = min(s + T_seg, T)
    gate[s:e] = gate_seg[:e-s]
    return gate


def inject_segment(src, tgt, gate, coupled_aus, kappas, lag_samples):
    T = src.shape[0]
    out = tgt.copy()
    for ch, kappa in zip(coupled_aus, kappas):
        if kappa <= 0: continue
        s_lag = np.roll(src[:, ch], lag_samples); s_lag[:lag_samples] = 0
        alpha = kappa * gate
        mask = alpha > 0.001
        out[mask, ch] = (alpha[mask] * s_lag[mask]
                         + np.sqrt(np.maximum(1 - alpha[mask]**2, 0)) * tgt[mask, ch])
    return out


def build_mixed_scenario(p1_orig, p2_orig, T):
    T_s = T / FS
    scale = T_s / 2700.0 if T_s < 2700 else 1.0
    schedule = [
        (0,    300,  'fwd', 2.0, 1.0, 0.15, 42),
        (300,  600,  'rev', 1.5, 1.0, 0.15, 55),
        (600,  900,  'fwd', 2.0, 1.0, 0.15, 70),
        (600,  900,  'rev', 1.5, 1.0, 0.15, 85),
        (900,  1200, 'fwd', 2.0, 1.0, 0.12, 100),
        (900,  1200, 'rev', 2.0, 0.7, 0.12, 115),
        (1500, 2100, 'fwd', 2.5, 0.5, 0.10, 130),
        (2100, 2400, 'fwd', 2.0, 1.0, 0.15, 145),
        (2100, 2400, 'rev', 1.0, 0.7, 0.15, 160),
        (2400, 2700, 'fwd', 2.0, 1.0, 0.15, 175),
    ]
    if scale < 1.0:
        schedule = [(int(s*scale), int(e*scale), d, l, k, du, sd)
                    for s, e, d, l, k, du, sd in schedule]

    p1_c, p2_c = p1_orig.copy(), p2_orig.copy()
    gate_fwd, gate_rev = np.zeros(T), np.zeros(T)
    for t_s, t_e, direction, lag_s, k_scale, duty, seed in schedule:
        if t_s >= T_s: continue
        gate = make_gate_segment(T, t_s, t_e, duty=duty, seed=seed)
        lag = int(lag_s * FS)
        kappas = np.clip(HETERO_KAPPAS * k_scale, 0, 0.95)
        if direction == 'fwd':
            p2_c = inject_segment(p1_orig, p2_c, gate, HETERO_AUS, kappas, lag)
            gate_fwd = np.maximum(gate_fwd, gate)
        else:
            p1_c = inject_segment(p2_orig, p1_c, gate, HETERO_AUS, kappas, lag)
            gate_rev = np.maximum(gate_rev, gate)

    for c in range(52):
        for sig in [p1_c, p2_c]:
            mu, sd = sig[:, c].mean(), max(sig[:, c].std(), 1e-8)
            sig[:, c] = (sig[:, c] - mu) / sd

    return p1_c, p2_c, gate_fwd > 0.5, gate_rev > 0.5


def ar_parallel(signals, channels):
    def _ar(y):
        a, _ = _estimate_ar(y, 3)
        T = len(y)
        yr = y.copy()
        for k in range(3): yr[3:] -= a[k] * y[2-k:T-k-1]
        yr[:3] = 0.0
        return yr
    results = Parallel(n_jobs=N_JOBS)(delayed(_ar)(signals[:, c]) for c in channels)
    out = signals.copy()
    for i, c in enumerate(channels):
        out[:, c] = results[i]
    return out


def _metrics(d, gm):
    nc, nn = max(gm.sum(), 1), max((~gm).sum(), 1)
    hit = float((gm & d).sum() / nc)
    fa = float((~gm & d).sum() / nn)
    inter = float((gm & d).sum())
    union = float((gm | d).sum())
    iou = inter / max(union, 1)
    prec = inter / max(d.sum(), 1)
    f1 = 2 * prec * hit / max(prec + hit, 1e-8)
    return {'hit': hit, 'fa': fa, 'iou': iou, 'f1': f1}


def detect_fixed_lag(src, tgt_res, coupled_aus, lag):
    T = src.shape[0]; C = len(coupled_aus); cc = np.zeros(T)
    for ci, c in enumerate(coupled_aus):
        cc[lag:] += src[:T-lag, c] * tgt_res[lag:, c]
    return cc / C


def detect_varying_lag(src, tgt_res, coupled_aus, lag_per_t):
    T = src.shape[0]; C = len(coupled_aus); cc = np.zeros(T)
    for ci, c in enumerate(coupled_aus):
        for t in range(T):
            L = lag_per_t[t]
            if t >= L: cc[t] += src[t-L, c] * tgt_res[t, c]
    return cc / C


def evaluate_detection(cc_c, cc_n, gm, smooth_s):
    c_sm = gaussian_filter1d(cc_c, sigma=smooth_s*FS) if smooth_s > 0 else cc_c
    n_sm = gaussian_filter1d(cc_n, sigma=smooth_s*FS) if smooth_s > 0 else cc_n
    best_f1, best = -1, None
    for tfa in [0.03, 0.05, 0.07, 0.10]:
        thr = np.percentile(n_sm, 100*(1-tfa))
        mc = _metrics(c_sm > thr, gm)
        if mc['fa'] <= 0.10 and mc['f1'] > best_f1:
            best_f1, best = mc['f1'], (tfa, mc)
    if best is None:
        thr = np.percentile(n_sm, 90)
        best = (0.10, _metrics(c_sm > thr, gm))
    return best


# ─────────────────────────────────────────────────────────────────────
# Approach 1: Event-Triggered Lag Estimation
# ─────────────────────────────────────────────────────────────────────

def detect_events(signal, channels, min_dist_s=0.3):
    """Detect activation events per channel. Returns dict of {ch: event_times_in_samples}."""
    min_dist = int(min_dist_s * FS)
    events = {}
    for c in channels:
        x = signal[:, c]
        mad = np.median(np.abs(x - np.median(x)))
        threshold = 1.5 * mad if mad > 0.01 else 0.5
        peaks, props = find_peaks(x, height=threshold, distance=min_dist)
        events[c] = peaks
    return events


def estimate_lag_events(src, tgt_res, coupled_aus, win_s=120, stride_s=60,
                        lag_range=(0.5, 5.0), bin_s=0.1):
    """Event-triggered lag estimation via lag histogram."""
    T = src.shape[0]
    lag_min_samp = int(lag_range[0] * FS)
    lag_max_samp = int(lag_range[1] * FS)
    n_bins = int((lag_range[1] - lag_range[0]) / bin_s) + 1
    bin_edges = np.linspace(lag_range[0], lag_range[1], n_bins + 1)

    # Detect events globally
    src_events = detect_events(src, coupled_aus)
    tgt_events = detect_events(tgt_res, coupled_aus)

    win_samp = int(win_s * FS)
    stride_samp = int(stride_s * FS)

    centers, lags, confidences = [], [], []
    pos = 0
    while pos + win_samp <= T:
        s, e = pos, pos + win_samp
        # Collect event-pair lags across all coupled channels
        pair_lags = []
        for c in coupled_aus:
            src_ev = src_events[c]
            tgt_ev = tgt_events[c]
            src_in_win = src_ev[(src_ev >= s) & (src_ev < e)]
            for t1 in src_in_win:
                # Find nearest tgt event in [t1+lag_min, t1+lag_max]
                candidates = tgt_ev[(tgt_ev > t1 + lag_min_samp) &
                                     (tgt_ev <= t1 + lag_max_samp)]
                if len(candidates) > 0:
                    nearest = candidates[0]  # first (closest lag)
                    pair_lags.append((nearest - t1) / FS)

        if len(pair_lags) >= 3:
            hist, _ = np.histogram(pair_lags, bins=bin_edges)
            peak_bin = np.argmax(hist)
            peak_lag_s = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2
            peak_lag_samp = int(peak_lag_s * FS)
            conf = hist[peak_bin] / max(len(pair_lags), 1)
        else:
            peak_lag_samp = int(2.0 * FS)  # fallback
            conf = 0.0

        centers.append(pos + win_samp // 2)
        lags.append(peak_lag_samp)
        confidences.append(conf)
        pos += stride_samp

    centers = np.array(centers)
    lags = np.array(lags)
    confidences = np.array(confidences)

    # Confidence-weighted interpolation: high-confidence windows dominate
    if len(centers) > 0 and confidences.max() > 0:
        lag_per_t = np.zeros(T)
        for t in range(T):
            # Weighted average of nearby window lags
            dists = np.abs(centers - t) / (win_samp / 2)
            w = confidences * np.exp(-dists**2)
            w_sum = w.sum()
            if w_sum > 0:
                lag_per_t[t] = int(np.round((w * lags).sum() / w_sum))
            else:
                lag_per_t[t] = int(2.0 * FS)
        lag_per_t = lag_per_t.astype(int)
        lag_per_t = np.clip(lag_per_t, int(lag_range[0]*FS), int(lag_range[1]*FS))
    else:
        lag_per_t = np.full(T, int(2.0 * FS), dtype=int)

    return lag_per_t, centers, lags, confidences


# ─────────────────────────────────────────────────────────────────────
# Approach 2: Multi-Lag Bank + Cluster-Mass
# ─────────────────────────────────────────────────────────────────────

def detect_multilag_bank(src, tgt_c_res, tgt_n_res, coupled_aus, gm,
                         smooth_s=3.0, lag_step=3):
    """Max-over-lags detection without lag estimation."""
    T = src.shape[0]; C = len(coupled_aus)
    lag_min = int(0.5 * FS); lag_max = int(5.0 * FS)
    lag_grid = range(lag_min, lag_max + 1, lag_step)
    n_lags = len(list(lag_grid))

    # Build cc matrix: (n_lags, T)
    cc_c = np.zeros((n_lags, T))
    cc_n = np.zeros((n_lags, T))
    for li, L in enumerate(lag_grid):
        for ci, c in enumerate(coupled_aus):
            cc_c[li, L:] += src[:T-L, c] * tgt_c_res[L:, c]
            cc_n[li, L:] += src[:T-L, c] * tgt_n_res[L:, c]
        cc_c[li] /= C
        cc_n[li] /= C

    # Smooth each lag row
    sigma = smooth_s * FS
    for li in range(n_lags):
        cc_c[li] = gaussian_filter1d(cc_c[li], sigma=sigma)
        cc_n[li] = gaussian_filter1d(cc_n[li], sigma=sigma)

    # Max over lags
    cc_max_c = cc_c.max(axis=0)
    cc_max_n = cc_n.max(axis=0)

    # Argmax gives per-timepoint lag estimate (bonus)
    lag_est = np.array(list(lag_grid))[cc_c.argmax(axis=0)]

    # Simple threshold detection (surrogate-calibrated)
    best_f1, best = -1, None
    for tfa in [0.03, 0.05, 0.07, 0.10]:
        thr = np.percentile(cc_max_n, 100*(1-tfa))
        mc = _metrics(cc_max_c > thr, gm)
        if mc['fa'] <= 0.10 and mc['f1'] > best_f1:
            best_f1, best = mc['f1'], (tfa, mc)
    if best is None:
        thr = np.percentile(cc_max_n, 90)
        best = (0.10, _metrics(cc_max_c > thr, gm))

    return best, lag_est, cc_max_c, cc_max_n


# ─────────────────────────────────────────────────────────────────────
# Approach 3: Hierarchical (Session Anchor + Narrow Refinement)
# ─────────────────────────────────────────────────────────────────────

def estimate_lag_hierarchical(src, tgt_res, coupled_aus, win_s=120, stride_s=60,
                               narrow_range_s=0.5):
    """Session anchor → narrow per-window refinement → median filter."""
    T = src.shape[0]; C = len(coupled_aus)
    lag_min = max(1, int(0.3 * FS)); lag_max = int(5.0 * FS)

    # Stage 1: session-level anchor
    cc_session = np.zeros(lag_max - lag_min + 1)
    for li, L in enumerate(range(lag_min, lag_max + 1)):
        if L >= T: continue
        for ci, c in enumerate(coupled_aus):
            cc_session[li] += (src[:T-L, c] * tgt_res[L:, c]).mean()
        cc_session[li] /= C
    anchor_lag = lag_min + np.argmax(cc_session)

    # Stage 2: per-window narrow search
    narrow = int(narrow_range_s * FS)
    search_min = max(lag_min, anchor_lag - narrow)
    search_max = min(lag_max, anchor_lag + narrow)
    win_samp = int(win_s * FS)
    stride_samp = int(stride_s * FS)

    centers, raw_lags = [], []
    pos = 0
    while pos + win_samp <= T:
        s, e = pos, pos + win_samp
        cc = np.zeros(search_max - search_min + 1)
        for li, L in enumerate(range(search_min, search_max + 1)):
            if L >= win_samp: continue
            for ci, c in enumerate(coupled_aus):
                n = min(e-s-L, e-s)
                if n > 10:
                    cc[li] += np.mean(src[s:s+n, c] * tgt_res[s+L:s+L+n, c])
            cc[li] /= C
        raw_lags.append(search_min + np.argmax(cc))
        centers.append(pos + win_samp // 2)
        pos += stride_samp

    centers = np.array(centers)
    raw_lags = np.array(raw_lags)

    # Median filter (window of 5)
    from scipy.ndimage import median_filter
    if len(raw_lags) >= 5:
        filtered = median_filter(raw_lags, size=5)
    else:
        filtered = raw_lags

    lag_per_t = np.interp(np.arange(T), centers, filtered).astype(int)
    lag_per_t = np.clip(lag_per_t, lag_min, lag_max)
    return lag_per_t, anchor_lag, centers, raw_lags, filtered


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def run_direction(label, src_orig, tgt_c_res, tgt_n_res, gm, T):
    """Run all 3 approaches for one direction."""
    print(f"\n{'='*80}", flush=True)
    print(f"{label} — ground truth: {gm.mean():.1%} coupled", flush=True)
    print(f"{'='*80}", flush=True)

    if gm.sum() == 0:
        print("  No coupling, skipping.", flush=True)
        return

    smooth_s = 2.0

    # ── Baseline: Oracle lag ──
    # Find true dominant lag from ground truth cross-corr peak
    cc_oracle = np.zeros(int(5.0*FS))
    for L in range(int(0.3*FS), int(5.0*FS)):
        if L >= T: continue
        for ci, c in enumerate(HETERO_AUS):
            cc_oracle[L] += (src_orig[:T-L, c] * tgt_c_res[L:, c]).mean()
    oracle_lag = int(0.3*FS) + np.argmax(cc_oracle[int(0.3*FS):])
    cc_c_oracle = detect_fixed_lag(src_orig, tgt_c_res, HETERO_AUS, oracle_lag)
    cc_n_oracle = detect_fixed_lag(src_orig, tgt_n_res, HETERO_AUS, oracle_lag)
    tfa_o, mc_o = evaluate_detection(cc_c_oracle, cc_n_oracle, gm, smooth_s)
    print(f"\n  Oracle (lag={oracle_lag/FS:.2f}s): "
          f"hit={mc_o['hit']:.1%} FA={mc_o['fa']:.1%} F1={mc_o['f1']:.2f}",
          flush=True)

    # ── Approach 1: Event-Triggered ──
    t0 = time.perf_counter()
    lag1, ctrs1, lags1, confs1 = estimate_lag_events(
        src_orig, tgt_c_res, HETERO_AUS, win_s=120, stride_s=60)
    cc_c1 = detect_varying_lag(src_orig, tgt_c_res, HETERO_AUS, lag1)
    lag1_null, _, _, _ = estimate_lag_events(
        src_orig, tgt_n_res, HETERO_AUS, win_s=120, stride_s=60)
    cc_n1 = detect_varying_lag(src_orig, tgt_n_res, HETERO_AUS, lag1_null)
    tfa1, mc1 = evaluate_detection(cc_c1, cc_n1, gm, smooth_s)
    t1 = time.perf_counter() - t0

    # Lag accuracy
    correct1 = sum(1 for l in lags1 if abs(l/FS - oracle_lag/FS) < 0.15)
    print(f"\n  Approach 1 (Event Matching, {t1:.1f}s):", flush=True)
    print(f"    Lag accuracy: {correct1}/{len(lags1)} windows correct (±0.15s)",
          flush=True)
    print(f"    Lags: {[f'{l/FS:.1f}' for l in lags1]}", flush=True)
    print(f"    Confs: {[f'{c:.2f}' for c in confs1]}", flush=True)
    print(f"    Detection: hit={mc1['hit']:.1%} FA={mc1['fa']:.1%} "
          f"F1={mc1['f1']:.2f}", flush=True)

    # ── Approach 2: Multi-Lag Bank ──
    t0 = time.perf_counter()
    (tfa2, mc2), lag_est2, cc_max_c2, cc_max_n2 = detect_multilag_bank(
        src_orig, tgt_c_res, tgt_n_res, HETERO_AUS, gm,
        smooth_s=smooth_s, lag_step=3)
    t2 = time.perf_counter() - t0
    print(f"\n  Approach 2 (Multi-Lag Bank, {t2:.1f}s):", flush=True)
    print(f"    Detection: hit={mc2['hit']:.1%} FA={mc2['fa']:.1%} "
          f"F1={mc2['f1']:.2f}", flush=True)
    # Show FA sweep
    for tfa in [0.03, 0.05, 0.07, 0.10, 0.15]:
        thr = np.percentile(cc_max_n2, 100*(1-tfa))
        mc = _metrics(cc_max_c2 > thr, gm)
        print(f"      FA={tfa:.0%}: hit={mc['hit']:.1%} FA={mc['fa']:.1%} "
              f"F1={mc['f1']:.2f}", flush=True)

    # ── Approach 3: Hierarchical ──
    t0 = time.perf_counter()
    lag3, anchor3, ctrs3, raw3, filt3 = estimate_lag_hierarchical(
        src_orig, tgt_c_res, HETERO_AUS, win_s=120, stride_s=60)
    cc_c3 = detect_varying_lag(src_orig, tgt_c_res, HETERO_AUS, lag3)
    lag3_null, _, _, _, _ = estimate_lag_hierarchical(
        src_orig, tgt_n_res, HETERO_AUS, win_s=120, stride_s=60)
    cc_n3 = detect_varying_lag(src_orig, tgt_n_res, HETERO_AUS, lag3_null)
    tfa3, mc3 = evaluate_detection(cc_c3, cc_n3, gm, smooth_s)
    t3 = time.perf_counter() - t0

    correct3 = sum(1 for l in filt3 if abs(l/FS - oracle_lag/FS) < 0.15)
    print(f"\n  Approach 3 (Hierarchical, {t3:.1f}s):", flush=True)
    print(f"    Anchor lag: {anchor3/FS:.2f}s", flush=True)
    print(f"    Refined lags: {[f'{l/FS:.1f}' for l in filt3]}", flush=True)
    print(f"    Lag accuracy: {correct3}/{len(filt3)} correct (±0.15s)", flush=True)
    print(f"    Detection: hit={mc3['hit']:.1%} FA={mc3['fa']:.1%} "
          f"F1={mc3['f1']:.2f}", flush=True)

    # ── Summary ──
    print(f"\n  {'Method':<30} | {'hit':>6} {'FA':>6} {'F1':>6}", flush=True)
    print(f"  {'-'*50}", flush=True)
    print(f"  {'Oracle':.<30} | {mc_o['hit']:>5.1%} {mc_o['fa']:>5.1%} "
          f"{mc_o['f1']:>5.2f}", flush=True)
    print(f"  {'1. Event Matching':.<30} | {mc1['hit']:>5.1%} {mc1['fa']:>5.1%} "
          f"{mc1['f1']:>5.2f}", flush=True)
    print(f"  {'2. Multi-Lag Bank':.<30} | {mc2['hit']:>5.1%} {mc2['fa']:>5.1%} "
          f"{mc2['f1']:>5.2f}", flush=True)
    print(f"  {'3. Hierarchical':.<30} | {mc3['hit']:>5.1%} {mc3['fa']:>5.1%} "
          f"{mc3['f1']:>5.2f}", flush=True)


def main():
    print("=" * 80, flush=True)
    print("3 LAG ESTIMATION APPROACHES — Mixed Coupling Scenario", flush=True)
    print(f"18 AUs, mean κ={HETERO_KAPPAS.mean():.3f}", flush=True)
    print("=" * 80, flush=True)

    cfg = load_config('configs/default.yaml')
    p1_orig, p2_orig, T = load_sessions(cfg)
    p1_c, p2_c, gm_fwd, gm_rev = build_mixed_scenario(p1_orig, p2_orig, T)
    print(f"T={T} ({T/FS:.0f}s), fwd={gm_fwd.mean():.1%}, rev={gm_rev.mean():.1%}",
          flush=True)

    # AR residualize
    t0 = time.perf_counter()
    p2_c_res = ar_parallel(p2_c, list(range(52)))
    p1_c_res = ar_parallel(p1_c, list(range(52)))
    p2_n_res = ar_parallel(p2_orig, list(range(52)))
    p1_n_res = ar_parallel(p1_orig, list(range(52)))
    print(f"AR: {time.perf_counter()-t0:.1f}s", flush=True)

    # Run both directions
    run_direction("P1→P2", p1_orig, p2_c_res, p2_n_res, gm_fwd, T)
    run_direction("P2→P1", p2_orig, p1_c_res, p1_n_res, gm_rev, T)


if __name__ == '__main__':
    main()
