"""TE reliability as a function of burst rate.

For each timepoint, pairs the local burst rate with the TE z-score.
Bins by burst rate and checks whether TE z-scores behave like proper
z-distributions (std~1.0, mean|z|~0.8, ~2.5% tail exceedance).

If TE is unreliable at low burst rates, z-scores will have inflated
variance (noise/noise) or collapsed variance (0/0 → 0).

Usage:
    python scripts/_analyze_te_reliability.py
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
ROLLING_WINDOW = 120  # 60s at 2 Hz


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

    # Collect per-timepoint (burst_rate, te_z) pairs across all sessions
    pairs = {band: {'rate': [], 'te_z': [], 'condition': []}
             for band in ['theta', 'alpha']}

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

            # Only include timepoints within segments
            in_segment = np.zeros(len(t_common), dtype=bool)
            cond_labels = np.full(len(t_common), '', dtype=object)
            for seg in segments:
                m = get_condition_mask(t_common, segments, seg[0])
                if m is not None:
                    in_segment |= m
                    cond_labels[m] = seg[0]

            valid = in_segment & np.isfinite(min_rate) & np.isfinite(te_z)
            pairs[band_name]['rate'].extend(min_rate[valid].tolist())
            pairs[band_name]['te_z'].extend(te_z[valid].tolist())
            pairs[band_name]['condition'].extend(cond_labels[valid].tolist())

        print(f"  {sname}: done")

    # ══════════════════════════════════════════════════════════════════
    #  ANALYSIS: TE z-score properties binned by burst rate
    # ══════════════════════════════════════════════════════════════════

    rate_bins = [0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 1.0]

    for band in ['theta', 'alpha']:
        rate = np.array(pairs[band]['rate'])
        te_z = np.array(pairs[band]['te_z'])
        conds = np.array(pairs[band]['condition'])

        print(f"\n{'='*80}")
        print(f"  BAND: {band.upper()} — TE z-score reliability by burst rate")
        print(f"  Total timepoints: {len(rate)}")
        print(f"{'='*80}")

        # Overall binned analysis
        print(f"\n  {'Rate bin':>12s} | {'n':>6s} | {'mean|z|':>8s} | {'std(z)':>7s} | "
              f"{'%>+2':>6s} | {'%<-2':>6s} | {'% any>2':>8s} | {'mean(z)':>8s} | {'quality':>10s}")
        print(f"  {'-'*95}")

        for i in range(len(rate_bins) - 1):
            lo, hi = rate_bins[i], rate_bins[i+1]
            mask = (rate >= lo) & (rate < hi)
            n = mask.sum()
            if n < 20:
                print(f"  [{lo:.2f},{hi:.2f}) | {n:6d} | {'--':>8s} | {'--':>7s} | "
                      f"{'--':>6s} | {'--':>6s} | {'--':>8s} | {'--':>8s} | {'--':>10s}")
                continue

            z_bin = te_z[mask]
            mean_abs = np.mean(np.abs(z_bin))
            std_z = np.std(z_bin)
            mean_z = np.mean(z_bin)
            frac_pos = np.mean(z_bin > 2.0) * 100
            frac_neg = np.mean(z_bin < -2.0) * 100
            frac_any = (frac_pos + frac_neg)

            # Quality assessment
            # For a proper N(0,1): mean|z| ≈ 0.80, std ≈ 1.0, tail ≈ 4.6%
            # Noise: std >> 1 or std << 1
            if std_z < 0.3:
                quality = 'COLLAPSED'
            elif std_z > 2.0:
                quality = 'INFLATED'
            elif 0.5 <= std_z <= 1.5 and 0.4 <= mean_abs <= 1.2:
                quality = 'GOOD'
            elif 0.3 <= std_z <= 2.0:
                quality = 'MARGINAL'
            else:
                quality = 'POOR'

            print(f"  [{lo:.2f},{hi:.2f}) | {n:6d} | {mean_abs:8.3f} | {std_z:7.3f} | "
                  f"{frac_pos:5.1f}% | {frac_neg:5.1f}% | {frac_any:6.1f}% | "
                  f"{mean_z:+8.3f} | {quality:>10s}")

        # Per-condition breakdown at key thresholds
        print(f"\n  Per-condition: std(TE z) by burst rate bin")
        cond_order = ['base_EO', 'base_EC', 'conv_1', 'meditate_B', 'meditate_K',
                      'PE_1', 'PE_2', 'conv_2']
        header = f"  {'Condition':>15s}"
        bin_labels = []
        for i in range(len(rate_bins) - 1):
            lo, hi = rate_bins[i], rate_bins[i+1]
            lbl = f'[{lo:.0%},{hi:.0%})'
            bin_labels.append(lbl)
            header += f" | {lbl:>12s}"
        print(header)
        print(f"  {'-'*(18 + 15*len(bin_labels))}")

        for cond in cond_order:
            cond_mask = conds == cond
            if cond_mask.sum() < 10:
                continue
            parts = [f"  {cond:>15s}"]
            for i in range(len(rate_bins) - 1):
                lo, hi = rate_bins[i], rate_bins[i+1]
                bin_mask = cond_mask & (rate >= lo) & (rate < hi)
                n = bin_mask.sum()
                if n < 10:
                    parts.append(f" {'--':>12s}")
                else:
                    std_z = np.std(te_z[bin_mask])
                    parts.append(f" {std_z:>11.3f}s")
            print(" |".join(parts))

        # Recommend threshold based on where std stabilizes near 1.0
        print(f"\n  Threshold recommendation ({band}):")
        for i in range(len(rate_bins) - 1):
            lo, hi = rate_bins[i], rate_bins[i+1]
            mask = (rate >= lo) & (rate < hi)
            n = mask.sum()
            if n < 20:
                continue
            z_bin = te_z[mask]
            std_z = np.std(z_bin)
            # TE is reliable when std is in [0.5, 1.5] range
            if std_z >= 0.5:
                print(f"    First bin with std >= 0.5: [{lo:.2f}, {hi:.2f}) — std={std_z:.3f}, n={n}")
                print(f"    -> Recommended min_rate for {band}: {lo:.2f}")
                break

    print(f"\n  Total time: {time.time() - t_wall:.0f}s")


if __name__ == '__main__':
    main()
