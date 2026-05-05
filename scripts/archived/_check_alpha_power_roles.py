"""Check per-person alpha power during each condition.

Who has high alpha and who has low alpha during meditate_B?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyxdf, glob
import torch

from cadence.significance.fast_cycles import _fft_bandpass, extract_cycle_features
from scripts.run_session_v6 import load_xdf_session
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache

CONDITION_ORDER = ['base_EO', 'base_EC', 'conv_1', 'meditate_B', 'meditate_K', 'conv_2']

# Load
xdf_path = glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0]
session_data = load_xdf_session(xdf_path)
markers = session_data['markers']
p1_role = session_data['p1_role']
p2_role = session_data['p2_role']

config = load_config()
cached_sessions = discover_cached_sessions(config['session_cache'])
cache_path = [p for n, p in cached_sessions if 'y_06' in n][0]
cached = load_session_from_cache(cache_path, config)

p1_bl_ts = cached.get('p1_blendshapes_ts')
lsl_ts = session_data['landmarks']['P1'][0]
lsl_offset = float(lsl_ts[0]) - float(p1_bl_ts[0])

print(f"P1 = {p1_role}, P2 = {p2_role}\n")
print(f"{'condition':>12} {'dur':>5} | "
      f"{'patient_alpha':>14} {'therapist_alpha':>15} {'ratio_pat/ther':>14} | "
      f"{'patient_theta':>14} {'therapist_theta':>15}")
print("-" * 100)

for seg_name in CONDITION_ORDER:
    t_start_lsl = markers.get(f'{seg_name}_start')
    t_end_lsl = markers.get(f'{seg_name}_stop')
    if t_start_lsl is None:
        continue

    t_start = t_start_lsl - lsl_offset
    t_end = t_end_lsl - lsl_offset
    dur = t_end_lsl - t_start_lsl

    powers = {}
    for person in ['p1', 'p2']:
        eeg_all = cached[f'{person}_eeg']
        ts_all = cached[f'{person}_eeg_ts']
        m = (ts_all >= t_start) & (ts_all <= t_end)
        if m.sum() < 1000:
            continue

        eeg_seg = eeg_all[m, :14].astype(np.float64)
        ts_seg = ts_all[m]
        fs = len(ts_seg) / (ts_seg[-1] - ts_seg[0])

        # Average reference
        eeg_seg -= eeg_seg.mean(axis=1, keepdims=True)

        # Compute per-band power via FFT bandpass + RMS
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        eeg_t = torch.as_tensor(eeg_seg.T, dtype=torch.float32, device=device)

        band_powers = {}
        for band_name, (lo, hi) in [('theta', (4, 8)), ('alpha', (8, 13)), ('beta', (13, 30))]:
            filtered = _fft_bandpass(eeg_t, fs, (lo, hi), device)
            # RMS power per channel, then average across channels
            rms = filtered.pow(2).mean(dim=1).sqrt().mean().item()
            band_powers[band_name] = rms

        role = p1_role if person == 'p1' else p2_role
        powers[role] = band_powers

    if 'patient' not in powers or 'therapist' not in powers:
        continue

    pat_alpha = powers['patient']['alpha']
    ther_alpha = powers['therapist']['alpha']
    ratio = pat_alpha / max(ther_alpha, 1e-10)

    pat_theta = powers['patient']['theta']
    ther_theta = powers['therapist']['theta']

    print(f"{seg_name:>12} {dur:5.0f} | "
          f"{pat_alpha:14.4f} {ther_alpha:15.4f} {ratio:14.2f} | "
          f"{pat_theta:14.4f} {ther_theta:15.4f}")

# Also show per-channel alpha power for meditate_B
print(f"\n--- Per-channel alpha RMS during meditate_B ---")
from cadence.constants import EPOC_CHANNEL_NAMES

t_start = markers['meditate_B_start'] - lsl_offset
t_end = markers['meditate_B_stop'] - lsl_offset

for person in ['p1', 'p2']:
    eeg_all = cached[f'{person}_eeg']
    ts_all = cached[f'{person}_eeg_ts']
    m = (ts_all >= t_start) & (ts_all <= t_end)
    eeg_seg = eeg_all[m, :14].astype(np.float64)
    ts_seg = ts_all[m]
    fs = len(ts_seg) / (ts_seg[-1] - ts_seg[0])
    eeg_seg -= eeg_seg.mean(axis=1, keepdims=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eeg_t = torch.as_tensor(eeg_seg.T, dtype=torch.float32, device=device)
    filtered = _fft_bandpass(eeg_t, fs, (8, 13), device)
    per_ch_rms = filtered.pow(2).mean(dim=1).sqrt().cpu().numpy()

    role = p1_role if person == 'p1' else p2_role
    ch_str = '  '.join(f"{EPOC_CHANNEL_NAMES[i]}={per_ch_rms[i]:.3f}" for i in range(14))
    print(f"\n  {role}: {ch_str}")
    print(f"    mean={per_ch_rms.mean():.4f}, std={per_ch_rms.std():.4f}, "
          f"max_ch={EPOC_CHANNEL_NAMES[per_ch_rms.argmax()]}={per_ch_rms.max():.4f}")
