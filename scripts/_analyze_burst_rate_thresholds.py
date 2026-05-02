"""Empirical burst rate analysis for TE gating threshold selection.

Computes per-participant rolling burst rates from EEG burst grids,
checks collinearity with concordance channels, and reports gate fractions
at various thresholds per condition.

Key questions:
1. What are burst rates per condition? (to set threshold)
2. Is burst rate collinear with concordance? (redundancy check)
3. What gate fraction does each threshold give per condition?
4. Does gating remove the burst-rate/TE-episode correlation?

Usage:
    python scripts/_analyze_burst_rate_thresholds.py
    python scripts/_analyze_burst_rate_thresholds.py --session y_06  # single session debug
"""

import sys, os, time, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

from cadence.config import load_config
from cadence.data.alignment import (
    discover_cached_sessions, load_session_from_cache, apply_modality_exclusions
)
from cadence.significance.fast_cycles import extract_burst_grids, EEG_BANDS
from scripts.run_session_v6 import load_xdf_session
from scripts._run_scaffold_v82 import CONDITION_ORDER, FS_OUT
from scripts.run_condition_statistics import (
    discover_sessions, load_session, classify_protocol,
    get_condition_mask,
)

RAW_DIR = 'C:/Users/optilab/Desktop/CADENCE/raw sessions'
ROLLING_WINDOW = 120  # 60s at 2 Hz, matching TE window
THRESHOLDS = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
TE_EPISODE_Z = 2.0


def compute_rolling_rate(burst_grid, window):
    """Rolling mean burst rate, channel-averaged.

    Args:
        burst_grid: (C, N) bool — per-channel burst at 2 Hz
        window: int — rolling window in samples

    Returns:
        (N,) float32 — channel-averaged rolling burst rate
    """
    C, N = burst_grid.shape
    # Cumsum trick for rolling mean, per channel
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    t_wall = time.time()

    # Discover V11 scaffold sessions (need both scaffold NPZ and raw EEG)
    scaffold_sessions = discover_sessions('results/v11')
    if args.session:
        scaffold_sessions = [s for s in scaffold_sessions if args.session in s['name']]
    print(f"Found {len(scaffold_sessions)} sessions with V11 scaffolds")

    # Load cached session data for raw EEG access
    cached_sessions = discover_cached_sessions(config['session_cache'])
    cache_map = {n.lower(): p for n, p in cached_sessions}

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Collect per-condition per-session rows
    all_rows = []

    for sess in scaffold_sessions:
        sname = sess['name']
        print(f"\n{'='*60}")
        print(f"  {sname}")

        # Load scaffold NPZ (has concordance + TE values)
        npz, meta, rslds = load_session(sess)
        sess['protocol'] = classify_protocol(meta['segments'])
        t_common = npz['t_common']
        segments = meta['segments']
        N = len(t_common)

        # Load raw EEG from cache
        cache_key = sname.lower()
        if cache_key not in cache_map:
            print(f"  No cache for {sname}, skipping")
            continue
        cached = load_session_from_cache(cache_map[cache_key], config)
        apply_modality_exclusions(cached, sname)

        if 'p1_eeg' not in cached or 'p2_eeg' not in cached:
            print(f"  No EEG data, skipping")
            continue

        # Get LSL offset (same logic as scaffold)
        xdf_files = glob.glob(os.path.join(RAW_DIR, f'{sname}*.xdf'))
        if not xdf_files:
            print(f"  No XDF for {sname}, skipping")
            continue
        session_data = load_xdf_session(xdf_files[0])
        lsl_ts_p1 = session_data['landmarks']['P1'][0]
        lsl_offset = float(lsl_ts_p1[0]) - float(cached.get('p1_blendshapes_ts', lsl_ts_p1)[0])

        # Prepare EEG (same as compute_burst_features)
        p1_eeg = cached['p1_eeg'].astype(np.float64)
        p2_eeg = cached['p2_eeg'].astype(np.float64)
        p1_ts = cached['p1_eeg_ts']
        n_ch = min(14, p1_eeg.shape[1], p2_eeg.shape[1])
        p1_eeg = p1_eeg[:, :n_ch]
        p2_eeg = p2_eeg[:, :n_ch]
        mlen = min(len(p1_eeg), len(p2_eeg))
        p1_eeg = p1_eeg[:mlen]
        p2_eeg = p2_eeg[:mlen]

        fs_eeg = 1.0 / np.median(np.diff(p1_ts[:1000])) if len(p1_ts) > 10 else 256.0
        dur = mlen / fs_eeg
        t_grid_local = np.arange(0, dur, 1.0 / FS_OUT)
        t_grid_lsl = t_grid_local + (p1_ts[0] + lsl_offset)

        # Extract burst grids
        t0 = time.time()
        grids = extract_burst_grids(p1_eeg, p2_eeg, fs_eeg, t_grid_local,
                                     bands=EEG_BANDS, device=device)
        print(f"  Burst grids: {time.time() - t0:.1f}s")

        # Per-band: compute rolling burst rates, interp to t_common
        for band_name in ['theta', 'alpha']:
            bg = grids.get(band_name)
            if bg is None:
                continue

            valid = bg['valid_channels']
            p1_ch_rates = bg['p1_burst'].mean(axis=1)
            p2_ch_rates = bg['p2_burst'].mean(axis=1)
            ch_mask = (p1_ch_rates > 0.01) & (p2_ch_rates > 0.01)
            if ch_mask.sum() < 3:
                print(f"    {band_name}: <3 valid channels, skipping")
                continue

            p1b = bg['p1_burst'][ch_mask]
            p2b = bg['p2_burst'][ch_mask]

            # Rolling burst rates
            p1_rate_local = compute_rolling_rate(p1b, ROLLING_WINDOW)
            p2_rate_local = compute_rolling_rate(p2b, ROLLING_WINDOW)

            # Interp to t_common
            p1_rate = np.interp(t_common, t_grid_lsl, p1_rate_local,
                                left=0, right=0).astype(np.float32)
            p2_rate = np.interp(t_common, t_grid_lsl, p2_rate_local,
                                left=0, right=0).astype(np.float32)
            min_rate = np.minimum(p1_rate, p2_rate)

            # Get concordance and TE from scaffold
            conc_key = f'z_raw_conc_{band_name}'
            te_key = f'u_te_asym_{band_name}'
            conc = npz[conc_key] if conc_key in npz else None
            te_asym = npz[te_key] if te_key in npz else None

            # Per-condition analysis
            for seg in segments:
                cond_name = seg[0]
                mask = get_condition_mask(t_common, segments, cond_name)
                if mask is None or mask.sum() < 20:
                    continue

                p1r = p1_rate[mask]
                p2r = p2_rate[mask]
                mr = min_rate[mask]

                row = {
                    'session': sname,
                    'protocol': sess['protocol'],
                    'condition': cond_name,
                    'band': band_name,
                    'n_samples': int(mask.sum()),
                    'p1_rate_mean': float(p1r.mean()),
                    'p2_rate_mean': float(p2r.mean()),
                    'min_rate_mean': float(mr.mean()),
                    'p1_rate_p25': float(np.percentile(p1r, 25)),
                    'p2_rate_p25': float(np.percentile(p2r, 25)),
                    'min_rate_p25': float(np.percentile(mr, 25)),
                }

                # Gate fractions at various thresholds
                for thresh in THRESHOLDS:
                    gate = (p1r >= thresh) & (p2r >= thresh)
                    row[f'gate_frac_{thresh}'] = float(gate.mean())

                # Concordance correlation
                if conc is not None:
                    c = conc[mask]
                    rho_p1, _ = spearmanr(p1r, c)
                    rho_p2, _ = spearmanr(p2r, c)
                    rho_min, _ = spearmanr(mr, c)
                    row['rho_p1rate_conc'] = float(rho_p1)
                    row['rho_p2rate_conc'] = float(rho_p2)
                    row['rho_minrate_conc'] = float(rho_min)

                # TE episode fraction (ungated + gated at each threshold)
                if te_asym is not None:
                    te = te_asym[mask]
                    te_any_ungated = float(np.mean(np.abs(te) > TE_EPISODE_Z))
                    row['te_any_ungated'] = te_any_ungated

                    for thresh in THRESHOLDS:
                        gate = (p1r >= thresh) & (p2r >= thresh)
                        if gate.sum() >= 10:
                            te_gated = te[gate]
                            row[f'te_any_gated_{thresh}'] = float(
                                np.mean(np.abs(te_gated) > TE_EPISODE_Z))
                        else:
                            row[f'te_any_gated_{thresh}'] = float('nan')

                all_rows.append(row)

        print(f"  {sname}: {len([r for r in all_rows if r['session'] == sname])} condition-band rows")

    # ══════════════════════════════════════════════════════════════════
    #  AGGREGATE RESULTS
    # ══════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"  AGGREGATE RESULTS ({len(all_rows)} rows)")
    print(f"{'='*80}")

    for band in ['theta', 'alpha']:
        band_rows = [r for r in all_rows if r['band'] == band]
        if not band_rows:
            continue

        print(f"\n{'─'*70}")
        print(f"  BAND: {band.upper()}")
        print(f"{'─'*70}")

        # 1. Per-condition burst rate statistics
        print(f"\n  1. Per-condition burst rates (rolling 60s, channel-averaged):")
        cond_order = ['base_EO', 'base_EC', 'conv_1', 'meditate_B', 'meditate_K',
                      'PE_1', 'PE_2', 'conv_2']
        header = f"    {'Condition':15s} | {'n':>3s} | {'P1 rate':>8s} | {'P2 rate':>8s} | {'min rate':>9s} | {'min p25':>8s}"
        print(header)
        print(f"    {'-'*65}")
        for cond in cond_order:
            cond_rows = [r for r in band_rows if r['condition'] == cond]
            if not cond_rows:
                continue
            n = len(cond_rows)
            p1 = np.mean([r['p1_rate_mean'] for r in cond_rows])
            p2 = np.mean([r['p2_rate_mean'] for r in cond_rows])
            mn = np.mean([r['min_rate_mean'] for r in cond_rows])
            p25 = np.mean([r['min_rate_p25'] for r in cond_rows])
            print(f"    {cond:15s} | {n:3d} | {p1:7.3f}  | {p2:7.3f}  | {mn:8.3f}  | {p25:7.3f}")

        # 2. Collinearity with concordance
        print(f"\n  2. Burst rate vs concordance (Spearman rho, per-condition):")
        rhos_min = [r.get('rho_minrate_conc', float('nan')) for r in band_rows
                    if not np.isnan(r.get('rho_minrate_conc', float('nan')))]
        if rhos_min:
            print(f"    min(p1,p2)_rate vs conc_{band}: "
                  f"median rho = {np.median(rhos_min):.3f}, "
                  f"mean = {np.mean(rhos_min):.3f}, "
                  f"range = [{min(rhos_min):.3f}, {max(rhos_min):.3f}]")
        rhos_p1 = [r.get('rho_p1rate_conc', float('nan')) for r in band_rows
                   if not np.isnan(r.get('rho_p1rate_conc', float('nan')))]
        if rhos_p1:
            print(f"    p1_rate vs conc_{band}:          "
                  f"median rho = {np.median(rhos_p1):.3f}, "
                  f"mean = {np.mean(rhos_p1):.3f}")

        # 3. Gate fractions per threshold per condition
        print(f"\n  3. Gate fractions at various thresholds (% of condition retained):")
        header = f"    {'Condition':15s} | " + " | ".join(f'{t:6.0%}' for t in THRESHOLDS)
        print(header)
        print(f"    {'-'*(18 + 9*len(THRESHOLDS))}")
        for cond in cond_order:
            cond_rows = [r for r in band_rows if r['condition'] == cond]
            if not cond_rows:
                continue
            parts = [f"    {cond:15s} |"]
            for thresh in THRESHOLDS:
                gf = np.mean([r.get(f'gate_frac_{thresh}', 0) for r in cond_rows])
                parts.append(f" {gf:5.1%} ")
            print(" |".join(parts))

        # 4. Burst rate vs TE episode correlation (the confound)
        print(f"\n  4. Burst rate vs TE episode fraction (the confound):")
        for thresh_label, key_suffix in [('ungated', None)] + \
                [(f'gated@{t}', t) for t in THRESHOLDS]:
            rates_all = []
            te_all = []
            for r in band_rows:
                mr = r['min_rate_mean']
                if key_suffix is None:
                    te_val = r.get('te_any_ungated', float('nan'))
                else:
                    te_val = r.get(f'te_any_gated_{key_suffix}', float('nan'))
                if np.isfinite(te_val):
                    rates_all.append(mr)
                    te_all.append(te_val)
            if len(rates_all) >= 6:
                rho, p = spearmanr(rates_all, te_all)
                print(f"    {thresh_label:15s}: rho={rho:+.3f}, p={p:.4f}, n={len(rates_all)}")
            else:
                print(f"    {thresh_label:15s}: insufficient data (n={len(rates_all)})")

    # 5. Overall recommendation
    print(f"\n{'='*70}")
    print(f"  RECOMMENDATION")
    print(f"{'='*70}")
    print(f"  Review the gate fractions above to find a threshold where:")
    print(f"    - Conversation retains >30% of timepoints (enough data)")
    print(f"    - Baselines retain >80% (don't over-gate resting state)")
    print(f"    - The burst-rate/TE correlation drops below |rho|=0.10")

    # Save raw data
    out_path = 'results/v11/burst_rate_analysis.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'rows': all_rows, 'thresholds': THRESHOLDS}, f, indent=2)
    print(f"\n  Raw data saved: {out_path}")
    print(f"  Total time: {time.time() - t_wall:.0f}s")


if __name__ == '__main__':
    main()
