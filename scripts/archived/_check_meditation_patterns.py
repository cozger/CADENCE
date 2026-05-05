"""Check if meditate_B (alpha PLV) vs meditate_K (theta volt_amp) replicates across sessions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyxdf, glob

from cadence.significance.fast_cycles import analyze_interbrain_cycles_multiband
from scripts.run_session_v6 import load_xdf_session
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache

config = load_config()
cached_sessions = discover_cached_sessions(config['session_cache'])
raw_dir = 'C:/Users/optilab/Desktop/CADENCE/raw sessions'
xdf_files = sorted(glob.glob(os.path.join(raw_dir, '*.xdf')))

print(f"{'session':>20} {'condition':>12} | "
      f"{'th_va':>6} {'al_va':>6} {'be_va':>6} {'comb':>6} | "
      f"{'th_plv':>6} {'al_plv':>6} {'be_plv':>6}")
print("-" * 95)

for xdf_path in xdf_files:
    session_name = os.path.splitext(os.path.basename(xdf_path))[0]

    try:
        session_data = load_xdf_session(xdf_path)
    except Exception as e:
        continue

    markers = session_data['markers']

    # Find cache
    cache_path = None
    for name, path in cached_sessions:
        if any(part in name for part in session_name.replace('_', ' ').split() if len(part) > 2):
            cache_path = path
            break
    if cache_path is None:
        continue

    try:
        cached = load_session_from_cache(cache_path, config)
    except Exception:
        continue

    # LSL offset
    p1_bl_ts = cached.get('p1_blendshapes_ts')
    if p1_bl_ts is None or 'P1' not in session_data['landmarks']:
        continue
    lsl_ts = session_data['landmarks']['P1'][0]
    lsl_offset = float(lsl_ts[0]) - float(p1_bl_ts[0])

    for seg_name in ['meditate_B', 'meditate_K', 'conv_1', 'conv_2']:
        t_start_lsl = markers.get(f'{seg_name}_start')
        t_end_lsl = markers.get(f'{seg_name}_stop')
        if t_start_lsl is None or t_end_lsl is None:
            continue

        t_start = t_start_lsl - lsl_offset
        t_end = t_end_lsl - lsl_offset

        sigs = {}
        fs_eeg = 256
        for person in ['p1', 'p2']:
            eeg_key = f'{person}_eeg'
            ts_key = f'{person}_eeg_ts'
            if eeg_key not in cached or ts_key not in cached:
                break
            eeg_all = cached[eeg_key]
            ts_all = cached[ts_key]
            m = (ts_all >= t_start) & (ts_all <= t_end)
            if m.sum() < 1000:
                break
            eeg_seg = eeg_all[m, :14].astype(np.float64)
            ts_seg = ts_all[m]
            fs_eeg = len(ts_seg) / (ts_seg[-1] - ts_seg[0])
            eeg_seg -= eeg_seg.mean(axis=1, keepdims=True)
            for ch in range(14):
                std = eeg_seg[:, ch].std()
                if std > 1e-8:
                    eeg_seg[:, ch] = (eeg_seg[:, ch] - eeg_seg[:, ch].mean()) / std
            sigs[person] = eeg_seg

        if 'p1' not in sigs or 'p2' not in sigs:
            continue

        min_len = min(len(sigs['p1']), len(sigs['p2']))
        try:
            result = analyze_interbrain_cycles_multiband(
                sigs['p1'][:min_len], sigs['p2'][:min_len], fs_eeg,
                n_surrogates=200, seed=42)
        except Exception as e:
            print(f"{session_name:>20} {seg_name:>12} | ERROR: {e}")
            continue

        vals = {}
        for band in ['theta', 'alpha', 'beta']:
            br = result.get('per_band', {}).get(band, {})
            vals[f'{band}_va'] = br.get('volt_amp', {}).get('pooled_z', 0)
            vals[f'{band}_plv'] = br.get('cycle_plv', {}).get('pooled_z', 0)

        comb = result.get('combined', {}).get('volt_amp', {})
        comb_z = comb.get('stouffer_z', comb.get('pooled_z', 0))

        print(f"{session_name:>20} {seg_name:>12} | "
              f"{vals['theta_va']:+6.1f} {vals['alpha_va']:+6.1f} "
              f"{vals['beta_va']:+6.1f} {comb_z:+6.1f} | "
              f"{vals['theta_plv']:+6.1f} {vals['alpha_plv']:+6.1f} "
              f"{vals['beta_plv']:+6.1f}")

print("\n\nSummary: meditate_B vs meditate_K pattern")
print("Prediction: meditate_B -> alpha PLV high, meditate_K -> theta volt_amp high")
