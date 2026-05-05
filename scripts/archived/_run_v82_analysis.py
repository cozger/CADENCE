"""V8.2: Shared-state concordance + classic EEG coherence analysis.

Computes TWO new feature types alongside existing V8 coupling z-timecourses:
  1. Concordance: (z_P1 + z_P2)/2 per modality — shared state level
  2. Classic EEG coherence: Welch MSC + imaginary coherence per band per ROI pair

Runs per-condition analysis across all sessions to test whether these features
separate conditions better than coupling z-scores alone.

Usage:
    python scripts/_run_v82_analysis.py
    python scripts/_run_v82_analysis.py --session y_06
"""

import sys, os, time, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import hilbert
from scipy.stats import wilcoxon
import torch

from cadence.significance.fast_cycles import _fft_bandpass, EEG_BANDS
from cadence.preprocess.eeg.coherence import eeg_band_coherence
from cadence.config import load_config
from cadence.data import discover_cached_sessions, load_session_from_cache
from scripts.run_session_v6 import load_xdf_session
from scripts._run_rslds_phase2 import MODALITY_KEYS, FS_OUT

CONDITION_ORDER = ['base_EO', 'base_EC', 'baseline', 'conv_1',
                   'PE', 'PE_1', 'PE_2',
                   'meditate_B', 'meditate_K', 'conv_2']
RAW_DIR = 'C:/Users/optilab/Desktop/CADENCE/raw sessions'


def compute_eeg_concordance(p1_eeg, p2_eeg, fs_eeg, t_grid, device=None):
    """Compute per-band EEG concordance: shared alpha/theta/beta power level.

    For each band:
      1. Bandpass filter both participants
      2. Hilbert envelope (instantaneous power)
      3. Average across channels
      4. Z-score each participant across session
      5. Concordance = (z_P1 + z_P2) / 2

    Returns dict of {band_name: (N,) concordance on t_grid}.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    T_eeg, n_ch = p1_eeg.shape
    dur = T_eeg / fs_eeg
    result = {}

    # Stack both participants for single GPU bandpass call
    both = np.vstack([p1_eeg.T, p2_eeg.T])  # (2*n_ch, T)
    both_t = torch.as_tensor(both, dtype=torch.float32, device=device)

    for band_name, (lo, hi) in EEG_BANDS.items():
        filt = _fft_bandpass(both_t, fs_eeg, (lo, hi), device).cpu().numpy()
        p1_filt = filt[:n_ch].T  # (T, n_ch)
        p2_filt = filt[n_ch:].T

        # Hilbert envelope = instantaneous amplitude
        p1_env = np.abs(hilbert(p1_filt, axis=0)).mean(axis=1)  # (T,)
        p2_env = np.abs(hilbert(p2_filt, axis=0)).mean(axis=1)

        # Resample to t_grid (2 Hz)
        t_eeg = np.arange(T_eeg) / fs_eeg
        p1_2hz = np.interp(t_grid, t_eeg, p1_env)
        p2_2hz = np.interp(t_grid, t_eeg, p2_env)

        # Z-score each participant across session
        for arr in [p1_2hz, p2_2hz]:
            mu, sd = arr.mean(), arr.std()
            if sd > 1e-8:
                arr[:] = (arr - mu) / sd
            else:
                arr[:] = 0.0

        # Concordance = average of z-scored power
        result[band_name] = ((p1_2hz + p2_2hz) / 2.0).astype(np.float32)

    return result


def compute_other_concordance(cached, t_common, lsl_offset):
    """Compute concordance for ECG HR, BL activity, Pose activity.

    Uses per-participant features already in cache.
    """
    result = {}

    # ECG HR concordance
    if ('p1_ecg_features' in cached and 'p2_ecg_features' in cached
            and 'p1_ecg_features_ts' in cached and 'p2_ecg_features_ts' in cached):
        p1_hr = np.interp(t_common, cached['p1_ecg_features_ts'] + lsl_offset,
                          cached['p1_ecg_features'][:, 0], left=np.nan, right=np.nan)
        p2_hr = np.interp(t_common, cached['p2_ecg_features_ts'] + lsl_offset,
                          cached['p2_ecg_features'][:, 0], left=np.nan, right=np.nan)
        valid = np.isfinite(p1_hr) & np.isfinite(p2_hr)
        for arr in [p1_hr, p2_hr]:
            v = arr[valid]
            if len(v) > 10:
                mu, sd = v.mean(), v.std()
                if sd > 1e-8:
                    arr[valid] = (arr[valid] - mu) / sd
                else:
                    arr[valid] = 0.0
            arr[~valid] = 0.0
        result['ecg_hr'] = ((p1_hr + p2_hr) / 2.0).astype(np.float32)

    # BL activity concordance (channel 52 = activity in raw blendshapes)
    if ('p1_blendshapes' in cached and 'p2_blendshapes' in cached
            and 'p1_blendshapes_ts' in cached and 'p2_blendshapes_ts' in cached):
        n_ch1 = cached['p1_blendshapes'].shape[1]
        n_ch2 = cached['p2_blendshapes'].shape[1]
        act_ch = min(n_ch1, n_ch2) - 1  # last channel is activity
        if act_ch >= 52:
            p1_act = np.interp(t_common, cached['p1_blendshapes_ts'] + lsl_offset,
                               cached['p1_blendshapes'][:, act_ch])
            p2_act = np.interp(t_common, cached['p2_blendshapes_ts'] + lsl_offset,
                               cached['p2_blendshapes'][:, act_ch])
            for arr in [p1_act, p2_act]:
                mu, sd = arr.mean(), arr.std()
                if sd > 1e-8: arr[:] = (arr - mu) / sd
                else: arr[:] = 0.0
            result['bl_activity'] = ((p1_act + p2_act) / 2.0).astype(np.float32)

    # Pose activity concordance
    if ('p1_pose_features' in cached and 'p2_pose_features' in cached
            and 'p1_pose_features_ts' in cached and 'p2_pose_features_ts' in cached):
        n_pch1 = cached['p1_pose_features'].shape[1]
        n_pch2 = cached['p2_pose_features'].shape[1]
        act_ch = min(n_pch1, n_pch2) - 1
        if act_ch >= 40:
            p1_pa = np.interp(t_common, cached['p1_pose_features_ts'] + lsl_offset,
                              cached['p1_pose_features'][:, act_ch])
            p2_pa = np.interp(t_common, cached['p2_pose_features_ts'] + lsl_offset,
                              cached['p2_pose_features'][:, act_ch])
            for arr in [p1_pa, p2_pa]:
                mu, sd = arr.mean(), arr.std()
                if sd > 1e-8: arr[:] = (arr - mu) / sd
                else: arr[:] = 0.0
            result['pose_activity'] = ((p1_pa + p2_pa) / 2.0).astype(np.float32)

    return result


def process_session(session_name, config):
    """Compute concordance + classic coherence for one session."""
    xdf_files = glob.glob(os.path.join(RAW_DIR, f'{session_name}*.xdf'))
    if not xdf_files:
        return None

    try:
        session_data = load_xdf_session(xdf_files[0])
    except Exception:
        return None

    markers = session_data['markers']
    cached_sessions = discover_cached_sessions(config['session_cache'])
    cache_matches = [p for n, p in cached_sessions if session_name.lower() in n.lower()]
    if not cache_matches:
        return None
    cached = load_session_from_cache(cache_matches[0], config)

    lsl_ts_p1 = session_data['landmarks']['P1'][0]
    lsl_offset = float(lsl_ts_p1[0]) - float(cached.get('p1_blendshapes_ts', lsl_ts_p1)[0])

    # Build segments
    segments = []
    for seg_name in CONDITION_ORDER:
        t_start = markers.get(f'{seg_name}_start')
        t_end = markers.get(f'{seg_name}_stop')
        if t_start is not None and t_end is not None and t_end > t_start:
            segments.append((seg_name, t_start, t_end))

    if not segments:
        return None

    session_start = min(t for _, t, _ in segments) - 30
    session_end = max(t for _, _, t in segments) + 30
    t_common = np.arange(session_start, session_end, 1.0 / FS_OUT)
    N = len(t_common)
    t_grid_relative = t_common - (cached['p1_eeg_ts'][0] + lsl_offset) if 'p1_eeg_ts' in cached else np.arange(N) / FS_OUT

    # ── EEG concordance ──────────────────────────────────────────────
    eeg_conc = {}
    has_eeg = ('p1_eeg' in cached and 'p2_eeg' in cached and 'p1_eeg_ts' in cached)
    if has_eeg:
        p1_eeg = cached['p1_eeg'].astype(np.float64)
        p2_eeg = cached['p2_eeg'].astype(np.float64)
        p1_ts = cached['p1_eeg_ts']
        n_ch = min(14, p1_eeg.shape[1], p2_eeg.shape[1])
        p1_eeg = p1_eeg[:, :n_ch]; p2_eeg = p2_eeg[:, :n_ch]
        mlen = min(len(p1_eeg), len(p2_eeg))
        p1_eeg = p1_eeg[:mlen]; p2_eeg = p2_eeg[:mlen]
        p1_eeg -= p1_eeg.mean(axis=1, keepdims=True)
        p2_eeg -= p2_eeg.mean(axis=1, keepdims=True)
        for ch in range(n_ch):
            for arr in [p1_eeg, p2_eeg]:
                sd = arr[:, ch].std()
                if sd > 1e-8: arr[:, ch] = (arr[:, ch] - arr[:, ch].mean()) / sd
        fs_eeg = len(p1_ts) / (p1_ts[-1] - p1_ts[0])

        eeg_conc = compute_eeg_concordance(
            p1_eeg, p2_eeg, fs_eeg, t_grid_relative)

        # Map to LSL time
        eeg_conc_lsl = {}
        eeg_t_lsl = t_grid_relative + (p1_ts[0] + lsl_offset)
        for bn, z in eeg_conc.items():
            eeg_conc_lsl[bn] = np.interp(t_common, eeg_t_lsl, z, left=0, right=0)

        # ── Classic EEG coherence ─────────────────────────────────────
        try:
            coh_real, coh_imag, coh_times, feat_names = eeg_band_coherence(
                p1_eeg, p2_eeg, fs=fs_eeg)
            coh_times_lsl = coh_times + (p1_ts[0] + lsl_offset)
            # Average coherence across ROI pairs per band → (n_bands, n_windows)
            coh_band_avg = coh_real.mean(axis=1)  # (n_bands, n_windows)
        except Exception:
            coh_band_avg = None
            coh_times_lsl = None
    else:
        eeg_conc_lsl = {}
        coh_band_avg = None
        coh_times_lsl = None

    # ── Other concordance ─────────────────────────────────────────────
    other_conc = compute_other_concordance(cached, t_common, lsl_offset)

    # ── Per-condition analysis ────────────────────────────────────────
    # Also load V8 coupling z-scores for comparison
    v8_npz = f'results/rslds/{session_name}/rslds_scaffold_v8_ztimecourses.npz'
    v8_coupling = {}
    if os.path.exists(v8_npz):
        v8_data = np.load(v8_npz)
        v8_t = v8_data['t_common']
        for k in MODALITY_KEYS:
            v8_coupling[k] = np.interp(t_common, v8_t, v8_data[f'z_{k}'], left=0, right=0)

    cond_results = {}
    for seg_name, t0, t1 in segments:
        mask = (t_common >= t0) & (t_common <= t1)
        if mask.sum() < 10:
            continue
        r = {'duration_s': t1 - t0}

        # Concordance per band
        for bn in ['theta', 'alpha', 'beta']:
            if bn in eeg_conc_lsl:
                r[f'conc_{bn}'] = float(eeg_conc_lsl[bn][mask].mean())

        # Classic coherence per band
        if coh_band_avg is not None and coh_times_lsl is not None:
            band_names = list(EEG_BANDS.keys())
            coh_mask = (coh_times_lsl >= t0) & (coh_times_lsl <= t1)
            if coh_mask.sum() > 2:
                for bi, bn in enumerate(band_names):
                    if bi < coh_band_avg.shape[0]:
                        r[f'coh_{bn}'] = float(coh_band_avg[bi, coh_mask].mean())

        # V8 coupling z per band
        for bn in ['eeg_theta', 'eeg_alpha', 'eeg_beta']:
            if bn in v8_coupling:
                r[f'coup_{bn}'] = float(v8_coupling[bn][mask].mean())

        # Other concordance
        for k, v in other_conc.items():
            r[f'conc_{k}'] = float(v[mask].mean())

        cond_results[seg_name] = r

    return {
        'session': session_name,
        'n_segments': len(segments),
        'segments': [(n, float(t0), float(t1)) for n, t0, t1 in segments],
        'conditions': cond_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', nargs='+', default=None)
    args = parser.parse_args()

    config = load_config()
    print("=" * 100)
    print("  V8.2: Concordance + Classic EEG Coherence Analysis")
    print("=" * 100)

    if args.session:
        sessions = args.session
    else:
        xdf_files = sorted(glob.glob(os.path.join(RAW_DIR, '*.xdf')))
        sessions = [os.path.splitext(os.path.basename(f))[0].lower() for f in xdf_files]

    print(f"  Sessions: {len(sessions)}")

    t0 = time.time()
    results = Parallel(n_jobs=-1)(
        delayed(process_session)(s, config) for s in sessions)
    results = [r for r in results if r is not None]
    print(f"  Processed {len(results)} sessions in {time.time()-t0:.0f}s\n")

    # ── Per-session per-condition table ────────────────────────────────
    features = ['conc_theta', 'conc_alpha', 'conc_beta',
                'coh_theta', 'coh_alpha', 'coh_beta',
                'coup_eeg_theta', 'coup_eeg_alpha', 'coup_eeg_beta',
                'conc_ecg_hr', 'conc_bl_activity', 'conc_pose_activity']

    for r in results:
        print(f"\n{'='*100}")
        print(f"  {r['session']} ({r['n_segments']} conditions)")
        print(f"{'='*100}")
        header = f"  {'Condition':>12s} |"
        for f in features:
            header += f" {f[:10]:>10s}"
        print(header)
        print(f"  " + "-" * (14 + 11 * len(features)))

        for cname, cr in r['conditions'].items():
            row = f"  {cname:>12s} |"
            for f in features:
                v = cr.get(f)
                if v is not None:
                    row += f" {v:+10.3f}"
                else:
                    row += f"        ---"
            print(row)

    # ── Grand summary: mean per condition type ────────────────────────
    print(f"\n{'='*100}")
    print(f"  GRAND SUMMARY: Mean feature values by condition type")
    print(f"{'='*100}")

    cond_type_data = {}
    for r in results:
        for cname, cr in r['conditions'].items():
            if cname.startswith('base') or cname == 'baseline':
                ct = 'Baseline'
            elif cname.startswith('conv'):
                ct = 'Conversation'
            elif cname.startswith('PE'):
                ct = 'PE'
            elif cname.startswith('meditate'):
                ct = 'Meditation'
            else:
                ct = cname
            if ct not in cond_type_data:
                cond_type_data[ct] = {f: [] for f in features}
            for f in features:
                v = cr.get(f)
                if v is not None:
                    cond_type_data[ct][f].append(v)

    header = f"  {'Condition':>14s} | n  |"
    for f in features:
        header += f" {f[:10]:>10s}"
    print(header)
    print(f"  " + "-" * (18 + 11 * len(features)))

    for ct in ['Conversation', 'PE', 'Meditation', 'Baseline']:
        if ct not in cond_type_data:
            continue
        cd = cond_type_data[ct]
        n = max(len(v) for v in cd.values()) if cd else 0
        row = f"  {ct:>14s} | {n:2d} |"
        for f in features:
            vals = cd[f]
            if vals:
                row += f" {np.mean(vals):+10.3f}"
            else:
                row += f"        ---"
        print(row)

    # ── Paired Wilcoxon tests: conversation vs baseline ───────────────
    print(f"\n  WILCOXON TESTS: Conversation vs Baseline (paired by session)")
    print(f"  " + "-" * 70)

    for f in features:
        conv_vals = []; base_vals = []
        for r in results:
            conv_v = []; base_v = []
            for cname, cr in r['conditions'].items():
                v = cr.get(f)
                if v is None:
                    continue
                if cname.startswith('conv'):
                    conv_v.append(v)
                elif cname.startswith('base') or cname == 'baseline':
                    base_v.append(v)
            if conv_v and base_v:
                conv_vals.append(np.mean(conv_v))
                base_vals.append(np.mean(base_v))

        if len(conv_vals) >= 5:
            diff = np.mean(conv_vals) - np.mean(base_vals)
            stat, p = wilcoxon(conv_vals, base_vals)
            sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
            print(f"    {f:>20s}: conv-base = {diff:+.4f}, p={p:.4f} {sig} (n={len(conv_vals)})")
        else:
            print(f"    {f:>20s}: insufficient pairs (n={len(conv_vals)})")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs('results/rslds', exist_ok=True)
    with open('results/rslds/v82_concordance_coherence_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved results/rslds/v82_concordance_coherence_results.json")


if __name__ == '__main__':
    main()
