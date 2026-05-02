"""TE reliability pooled across bands — same math, same threshold.

Pools theta + alpha timepoints by burst rate to get proper sample sizes
at all rate bins. Since the TE computation is identical (binary grids,
k=3, 120 windows, 200 surrogates), reliability should depend only on
burst rate, not on which band produced the bursts.

Usage:
    python scripts/_analyze_te_reliability_pooled.py
"""

import sys, os, time, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import spearmanr

from cadence.config import load_config
from cadence.data.alignment import (
    discover_cached_sessions, load_session_from_cache, apply_modality_exclusions
)
from cadence.significance.fast_cycles import extract_burst_grids, EEG_BANDS
from scripts.run_session_v6 import load_xdf_session
from scripts._run_scaffold_v82 import CONDITION_ORDER, FS_OUT
from scripts.run_condition_statistics import (
    discover_sessions, load_session, classify_protocol, get_condition_mask,
)

RAW_DIR = 'C:/Users/optilab/Desktop/CADENCE/raw sessions'
ROLLING_WINDOW = 120


def compute_rolling_rate(burst_grid, window):
    """Rolling mean burst rate, channel-averaged. (C, N) -> (N,)"""
    C, N = burst_grid.shape
    rates = np.zeros((C, N), dtype=np.float32)
    for c in range(C):
        cs = np.cumsum(burst_grid[c].astype(np.float32))
        cs = np.insert(cs, 0, 0)
        for t in range(N):
            t_start = max(0, t - window // 2)
            t_end = min(N, t + window // 2)
            rates[c, t] = (cs[t_end] - cs[t_start]) / (t_end - t_start)
    return rates.mean(axis=0)


def main():
    config = load_config()
    t_wall = time.time()

    scaffold_sessions = discover_sessions('results/v11')
    print(f"Found {len(scaffold_sessions)} sessions")

    cached_sessions = discover_cached_sessions(config['session_cache'])
    cache_map = {n.lower(): p for n, p in cached_sessions}

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Collect ALL (burst_rate, te_z) pairs, pooled across bands
    all_rate = []
    all_te_z = []
    all_band = []
    all_cond = []

    for sess in scaffold_sessions:
        sname = sess['name']
        npz, meta, rslds = load_session(sess)
        sess['protocol'] = classify_protocol(meta['segments'])
        t_common = npz['t_common']
        segments = meta['segments']

        cache_key = sname.lower()
        if cache_key not in cache_map:
            continue
        cached = load_session_from_cache(cache_map[cache_key], config)
        apply_modality_exclusions(cached, sname)
        if 'p1_eeg' not in cached or 'p2_eeg' not in cached:
            continue

        xdf_files = glob.glob(os.path.join(RAW_DIR, f'{sname}*.xdf'))
        if not xdf_files:
            continue
        session_data = load_xdf_session(xdf_files[0])
        lsl_ts_p1 = session_data['landmarks']['P1'][0]
        lsl_offset = float(lsl_ts_p1[0]) - float(cached.get('p1_blendshapes_ts', lsl_ts_p1)[0])

        p1_eeg = cached['p1_eeg'].astype(np.float64)
        p2_eeg = cached['p2_eeg'].astype(np.float64)
        p1_ts = cached['p1_eeg_ts']
        n_ch = min(14, p1_eeg.shape[1], p2_eeg.shape[1])
        p1_eeg = p1_eeg[:, :n_ch]; p2_eeg = p2_eeg[:, :n_ch]
        mlen = min(len(p1_eeg), len(p2_eeg))
        p1_eeg = p1_eeg[:mlen]; p2_eeg = p2_eeg[:mlen]

        fs_eeg = 1.0 / np.median(np.diff(p1_ts[:1000])) if len(p1_ts) > 10 else 256.0
        dur = mlen / fs_eeg
        t_grid_local = np.arange(0, dur, 1.0 / FS_OUT)
        t_grid_lsl = t_grid_local + (p1_ts[0] + lsl_offset)

        grids = extract_burst_grids(p1_eeg, p2_eeg, fs_eeg, t_grid_local,
                                     bands=EEG_BANDS, device=device)

        # Segment mask + condition labels
        in_segment = np.zeros(len(t_common), dtype=bool)
        cond_labels = np.full(len(t_common), '', dtype=object)
        for seg in segments:
            m = get_condition_mask(t_common, segments, seg[0])
            if m is not None:
                in_segment |= m
                cond_labels[m] = seg[0]

        for band_name in ['theta', 'alpha']:
            bg = grids.get(band_name)
            if bg is None:
                continue
            p1_ch_rates = bg['p1_burst'].mean(axis=1)
            p2_ch_rates = bg['p2_burst'].mean(axis=1)
            ch_mask = (p1_ch_rates > 0.01) & (p2_ch_rates > 0.01)
            if ch_mask.sum() < 3:
                continue

            p1_rate_local = compute_rolling_rate(bg['p1_burst'][ch_mask], ROLLING_WINDOW)
            p2_rate_local = compute_rolling_rate(bg['p2_burst'][ch_mask], ROLLING_WINDOW)

            p1_rate = np.interp(t_common, t_grid_lsl, p1_rate_local, left=np.nan, right=np.nan)
            p2_rate = np.interp(t_common, t_grid_lsl, p2_rate_local, left=np.nan, right=np.nan)
            min_rate = np.minimum(p1_rate, p2_rate)

            te_key = f'u_te_asym_{band_name}'
            if te_key not in npz:
                continue
            te_z = npz[te_key]

            valid = in_segment & np.isfinite(min_rate) & np.isfinite(te_z)
            all_rate.extend(min_rate[valid].tolist())
            all_te_z.extend(te_z[valid].tolist())
            all_band.extend([band_name] * valid.sum())
            all_cond.extend(cond_labels[valid].tolist())

        print(f"  {sname}: done")

    rate = np.array(all_rate)
    te_z = np.array(all_te_z)
    band = np.array(all_band)
    cond = np.array(all_cond)

    print(f"\nTotal pooled timepoints: {len(rate)} (theta: {(band=='theta').sum()}, alpha: {(band=='alpha').sum()})")

    # ══════════════════════════════════════════════════════════════════
    #  POOLED analysis: TE z-score properties by burst rate
    # ══════════════════════════════════════════════════════════════════

    rate_bins = [0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 1.0]

    print(f"\n{'='*90}")
    print(f"  POOLED TE z-score reliability by burst rate (both bands)")
    print(f"{'='*90}")

    print(f"\n  {'Rate bin':>12s} | {'n':>6s} | {'n_th':>5s} | {'n_al':>5s} | "
          f"{'mean|z|':>8s} | {'std(z)':>7s} | {'%>+2':>6s} | {'%<-2':>6s} | "
          f"{'kurtosis':>9s} | {'quality':>10s}")
    print(f"  {'-'*105}")

    for i in range(len(rate_bins) - 1):
        lo, hi = rate_bins[i], rate_bins[i+1]
        mask = (rate >= lo) & (rate < hi)
        n = mask.sum()
        n_th = ((band == 'theta') & mask).sum()
        n_al = ((band == 'alpha') & mask).sum()

        if n < 30:
            print(f"  [{lo:.2f},{hi:.2f}) | {n:6d} | {n_th:5d} | {n_al:5d} | "
                  f"{'--':>8s} | {'--':>7s} | {'--':>6s} | {'--':>6s} | "
                  f"{'--':>9s} | {'--':>10s}")
            continue

        z_bin = te_z[mask]
        mean_abs = np.mean(np.abs(z_bin))
        std_z = np.std(z_bin)
        mean_z = np.mean(z_bin)
        frac_pos = np.mean(z_bin > 2.0) * 100
        frac_neg = np.mean(z_bin < -2.0) * 100
        # Excess kurtosis (normal = 0)
        kurt = float(np.mean((z_bin - mean_z)**4) / std_z**4 - 3) if std_z > 0.01 else float('nan')

        if std_z < 0.3:
            quality = 'COLLAPSED'
        elif std_z > 2.0:
            quality = 'INFLATED'
        elif 0.6 <= std_z <= 1.4 and 0.4 <= mean_abs <= 1.2:
            quality = 'GOOD'
        elif 0.3 <= std_z <= 2.0:
            quality = 'MARGINAL'
        else:
            quality = 'POOR'

        print(f"  [{lo:.2f},{hi:.2f}) | {n:6d} | {n_th:5d} | {n_al:5d} | "
              f"{mean_abs:8.3f} | {std_z:7.3f} | {frac_pos:5.1f}% | {frac_neg:5.1f}% | "
              f"{kurt:+9.2f} | {quality:>10s}")

    # ── Same-rate comparison: theta vs alpha ──────────────────────────
    print(f"\n{'='*90}")
    print(f"  SAME-RATE COMPARISON: theta vs alpha (is the math really the same?)")
    print(f"{'='*90}")

    print(f"\n  {'Rate bin':>12s} | {'th std':>7s} | {'th n':>6s} | {'al std':>7s} | {'al n':>6s} | {'diff':>7s}")
    print(f"  {'-'*60}")

    for i in range(len(rate_bins) - 1):
        lo, hi = rate_bins[i], rate_bins[i+1]
        th_mask = (rate >= lo) & (rate < hi) & (band == 'theta')
        al_mask = (rate >= lo) & (rate < hi) & (band == 'alpha')
        n_th = th_mask.sum()
        n_al = al_mask.sum()

        if n_th >= 30 and n_al >= 30:
            th_std = np.std(te_z[th_mask])
            al_std = np.std(te_z[al_mask])
            print(f"  [{lo:.2f},{hi:.2f}) | {th_std:7.3f} | {n_th:6d} | {al_std:7.3f} | {n_al:6d} | {al_std-th_std:+7.3f}")
        elif n_th >= 30:
            th_std = np.std(te_z[th_mask])
            print(f"  [{lo:.2f},{hi:.2f}) | {th_std:7.3f} | {n_th:6d} | {'--':>7s} | {n_al:6d} | {'--':>7s}")
        elif n_al >= 30:
            al_std = np.std(te_z[al_mask])
            print(f"  [{lo:.2f},{hi:.2f}) | {'--':>7s} | {n_th:6d} | {al_std:7.3f} | {n_al:6d} | {'--':>7s}")
        else:
            print(f"  [{lo:.2f},{hi:.2f}) | {'--':>7s} | {n_th:6d} | {'--':>7s} | {n_al:6d} | {'--':>7s}")

    # ── Cumulative: what's the overall std if we gate at threshold X? ──
    print(f"\n{'='*90}")
    print(f"  CUMULATIVE: overall TE z-score std if we gate at threshold X")
    print(f"  (i.e., std of all z-scores at rate >= X)")
    print(f"{'='*90}")

    thresholds = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    print(f"\n  {'Threshold':>12s} | {'n':>7s} | {'std(z)':>7s} | {'mean|z|':>8s} | "
          f"{'%>+2':>6s} | {'%<-2':>6s} | {'kurt':>7s}")
    print(f"  {'-'*70}")

    for t in thresholds:
        mask = rate >= t
        n = mask.sum()
        if n < 30:
            continue
        z = te_z[mask]
        std_z = np.std(z)
        mean_abs = np.mean(np.abs(z))
        frac_pos = np.mean(z > 2.0) * 100
        frac_neg = np.mean(z < -2.0) * 100
        kurt = float(np.mean((z - z.mean())**4) / std_z**4 - 3) if std_z > 0.01 else 0
        print(f"  >= {t:.2f}       | {n:7d} | {std_z:7.3f} | {mean_abs:8.3f} | "
              f"{frac_pos:5.1f}% | {frac_neg:5.1f}% | {kurt:+7.2f}")

    print(f"\n  For reference, ideal N(0,1): std=1.000, mean|z|=0.798, tail=2.28% each, kurtosis=0.00")
    print(f"\n  Total time: {time.time() - t_wall:.0f}s")


if __name__ == '__main__':
    main()
