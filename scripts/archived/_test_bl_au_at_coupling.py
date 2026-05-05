"""What do the faces look like during Stage 1 coupling windows?

Stage 1 gives z-score timecourse from all 52 AUs. At high-z timepoints,
extract the raw AU profiles for both people. Compare coupled vs uncoupled
windows to see which expressions drive the coupling signal.
"""
import sys
sys.path.insert(0, 'C:/Users/optilab/desktop/CADENCE')

import numpy as np
import pyxdf, glob

from cadence.significance.bl_coupling import bl_two_stage_coupling
from cadence.significance.hawkes_coupling import MP_BLENDSHAPE_NAMES

FS = 30.0

# Load y_06
xdf_path = glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0]
data, _ = pyxdf.load_xdf(xdf_path, dejitter_timestamps=True)

marker_times = {}
for stream in data:
    if stream['info']['type'][0] == 'Markers':
        for t, v in zip(stream['time_stamps'], stream['time_series']):
            marker_times[v[0]] = t

landmarks = {}
for stream in data:
    name = stream['info']['name'][0]
    if 'landmarks' in name.lower():
        person = 'P1' if 'P1' in name else 'P2'
        n_ch = int(stream['info']['channel_count'][0])
        if n_ch >= 52 and person not in landmarks:
            landmarks[person] = (np.array(stream['time_stamps']),
                                 np.array(stream['time_series'], dtype=np.float32))

for segment in ['conv_1', 'conv_2']:
    t_start = marker_times[f'{segment}_start']
    t_end = marker_times[f'{segment}_stop']
    dur = t_end - t_start
    T = int(dur * FS)
    t_grid = np.linspace(0, dur, T)

    sigs = {}
    for p in ['P1', 'P2']:
        ts, d = landmarks[p]
        m = (ts >= t_start) & (ts <= t_end)
        sigs[p] = np.stack([np.interp(t_grid, ts[m] - t_start, d[m, c])
                            for c in range(52)], axis=1)

    p1_raw = sigs['P1']
    p2_raw = sigs['P2']

    # Run Stage 1 + 2
    result = bl_two_stage_coupling(p1_raw, p2_raw, FS, seed=42)
    z = result.z_continuous
    mask = result.mask_continuous
    out_rate = result.output_rate

    print(f"\n{'='*70}")
    print(f"  {segment} ({dur:.0f}s) — coupling={mask.mean():.1%}, z_max={z.max():.2f}")
    print(f"{'='*70}")

    # Map z-score timepoints back to raw signal indices
    z_times = np.arange(len(z)) / out_rate  # seconds
    coupled_times = z_times[mask]
    uncoupled_times = z_times[~mask]

    # For each timepoint in z, get the corresponding raw AU window (±0.5s)
    half_win = int(0.5 * FS)  # ±0.5s = 1s window

    def get_au_profiles(times_s, p1, p2):
        """Average raw AU values at given timepoints."""
        profiles_p1 = []
        profiles_p2 = []
        for t in times_s:
            idx = int(t * FS)
            s = max(0, idx - half_win)
            e = min(len(p1), idx + half_win)
            if e > s:
                profiles_p1.append(p1[s:e].mean(axis=0))
                profiles_p2.append(p2[s:e].mean(axis=0))
        if profiles_p1:
            return np.array(profiles_p1).mean(axis=0), np.array(profiles_p2).mean(axis=0)
        return np.zeros(52), np.zeros(52)

    au_coupled_p1, au_coupled_p2 = get_au_profiles(coupled_times, p1_raw, p2_raw)
    au_uncoupled_p1, au_uncoupled_p2 = get_au_profiles(uncoupled_times, p1_raw, p2_raw)

    # Difference: coupled - uncoupled
    diff_p1 = au_coupled_p1 - au_uncoupled_p1
    diff_p2 = au_coupled_p2 - au_uncoupled_p2

    # Cross-product contribution: which AUs have correlated activation
    # during coupled windows?
    coupled_mask_raw = np.zeros(T, dtype=bool)
    for t in coupled_times:
        idx = int(t * FS)
        coupled_mask_raw[max(0, idx - half_win):min(T, idx + half_win)] = True

    # Per-AU cross-correlation during coupled vs uncoupled windows
    au_xcorr_coupled = np.zeros(52)
    au_xcorr_uncoupled = np.zeros(52)
    for c in range(52):
        p1c = p1_raw[:, c]
        p2c = p2_raw[:, c]
        # Z-score
        p1z = (p1c - p1c.mean()) / max(p1c.std(), 1e-8)
        p2z = (p2c - p2c.mean()) / max(p2c.std(), 1e-8)
        cp = p1z * p2z  # instantaneous cross-product

        if coupled_mask_raw.sum() > 0:
            au_xcorr_coupled[c] = cp[coupled_mask_raw].mean()
        if (~coupled_mask_raw).sum() > 0:
            au_xcorr_uncoupled[c] = cp[~coupled_mask_raw].mean()

    xcorr_diff = au_xcorr_coupled - au_xcorr_uncoupled

    # Report top AUs by different criteria
    print(f"\n  Top AUs by cross-correlation DURING coupling (P1×P2):")
    top_xcorr = np.argsort(au_xcorr_coupled)[::-1][:10]
    for i in top_xcorr:
        print(f"    [{i:2d}] {MP_BLENDSHAPE_NAMES[i]:25s}  "
              f"coupled={au_xcorr_coupled[i]:+.4f}  "
              f"uncoupled={au_xcorr_uncoupled[i]:+.4f}  "
              f"diff={xcorr_diff[i]:+.4f}")

    print(f"\n  Top AUs by coupling EXCESS (coupled - uncoupled cross-corr):")
    top_excess = np.argsort(xcorr_diff)[::-1][:10]
    for i in top_excess:
        print(f"    [{i:2d}] {MP_BLENDSHAPE_NAMES[i]:25s}  "
              f"coupled={au_xcorr_coupled[i]:+.4f}  "
              f"uncoupled={au_xcorr_uncoupled[i]:+.4f}  "
              f"diff={xcorr_diff[i]:+.4f}")

    print(f"\n  P1 face during coupling (top elevated AUs vs baseline):")
    top_p1 = np.argsort(diff_p1)[::-1][:8]
    for i in top_p1:
        print(f"    [{i:2d}] {MP_BLENDSHAPE_NAMES[i]:25s}  "
              f"coupled={au_coupled_p1[i]:.3f}  uncoupled={au_uncoupled_p1[i]:.3f}  "
              f"diff={diff_p1[i]:+.3f}")

    print(f"\n  P2 face during coupling (top elevated AUs vs baseline):")
    top_p2 = np.argsort(diff_p2)[::-1][:8]
    for i in top_p2:
        print(f"    [{i:2d}] {MP_BLENDSHAPE_NAMES[i]:25s}  "
              f"coupled={au_coupled_p2[i]:.3f}  uncoupled={au_uncoupled_p2[i]:.3f}  "
              f"diff={diff_p2[i]:+.3f}")

    # ── NMF on coupled-window AU snapshots ────────────────────────
    # Stack P1+P2 AU profiles from coupled timepoints, run NMF to
    # discover what expression patterns cluster during coupling
    from sklearn.decomposition import NMF

    coupled_snapshots = []
    for t in coupled_times:
        idx = int(t * FS)
        s = max(0, idx - half_win)
        e = min(T, idx + half_win)
        if e > s:
            # Concatenate P1 and P2 AU profiles: (52+52,) = 104-dim
            snap = np.concatenate([p1_raw[s:e].mean(axis=0),
                                   p2_raw[s:e].mean(axis=0)])
            coupled_snapshots.append(snap)

    if len(coupled_snapshots) > 10:
        X_coupled = np.array(coupled_snapshots)
        np.maximum(X_coupled, 0, out=X_coupled)

        print(f"\n  NMF on {len(X_coupled)} coupled snapshots (P1+P2 = 104 dims):")

        for k in [3, 4, 5]:
            nmf = NMF(n_components=k, init='nndsvda', max_iter=300, random_state=42)
            W = nmf.fit_transform(X_coupled)
            H = nmf.components_  # (k, 104)
            recon = 1 - nmf.reconstruction_err_ / np.linalg.norm(X_coupled, 'fro')
            print(f"    k={k}: {recon:.1%} explained")

        # Detailed k=4
        k = 4
        nmf = NMF(n_components=k, init='nndsvda', max_iter=300, random_state=42)
        W = nmf.fit_transform(X_coupled)
        H = nmf.components_

        print(f"\n  Coupled-window expression patterns (k={k}):")
        for comp in range(k):
            h_p1 = H[comp, :52]
            h_p2 = H[comp, 52:]

            # Fraction of coupled windows dominated by this pattern
            frac = (W[:, comp] > W.mean(axis=0)[comp]).mean()

            top_p1 = np.argsort(h_p1)[::-1][:4]
            top_p2 = np.argsort(h_p2)[::-1][:4]

            p1_str = ', '.join(f'{MP_BLENDSHAPE_NAMES[i]}={h_p1[i]:.2f}' for i in top_p1 if h_p1[i] > 0.01)
            p2_str = ', '.join(f'{MP_BLENDSHAPE_NAMES[i]}={h_p2[i]:.2f}' for i in top_p2 if h_p2[i] > 0.01)

            print(f"\n    Pattern {comp} ({frac:.0%} of coupled windows):")
            print(f"      P1: {p1_str}")
            print(f"      P2: {p2_str}")
    else:
        print(f"\n  Too few coupled windows ({len(coupled_snapshots)}) for NMF")
