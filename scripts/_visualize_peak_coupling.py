"""Visualize what faces and bodies look like at peak facial/postural coupling moments.

Finds the top coupling timepoints from the V11 scaffold, maps back to raw
blendshape AUs and pose keypoints, and draws side-by-side comparisons of
peak coupling vs. low coupling moments.
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from cadence.data.alignment import discover_cached_sessions, load_session_from_cache

# ── Apple ARKit blendshape names (52 AUs, MediaPipe order) ───────────
APPLE_AU_NAMES = [
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
    'eyeBlinkLeft', 'eyeBlinkRight',
    'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight',
    'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight',
    'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight',
    'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
    'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight',
    'mouthFrownLeft', 'mouthFrownRight',
    'mouthFunnel', 'mouthLeft',
    'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthPressLeft', 'mouthPressRight',
    'mouthPucker', 'mouthRight',
    'mouthRollLower', 'mouthRollUpper',
    'mouthShrugLower', 'mouthShrugUpper',
    'mouthSmileLeft', 'mouthSmileRight',
    'mouthStretchLeft', 'mouthStretchRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight',
    'noseSneerLeft', 'noseSneerRight',
    'tongueOut',
]

# Affect AUs used by the wavelet coherence pipeline
AFFECT_AU_INDICES = [7, 8, 28, 29, 30, 31, 44, 45, 50, 51]
AFFECT_AU_NAMES = [APPLE_AU_NAMES[i] for i in AFFECT_AU_INDICES]

# MediaPipe pose connections for stick figure
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),      # right face
    (0, 4), (4, 5), (5, 6), (6, 8),      # left face
    (9, 10),                               # mouth
    (11, 12),                              # shoulders
    (11, 13), (13, 15),                    # left arm
    (12, 14), (14, 16),                    # right arm
    (11, 23), (12, 24), (23, 24),         # torso
    (23, 25), (25, 27),                    # left leg
    (24, 26), (26, 28),                    # right leg
]

CACHE_DIR = 'C:/Users/optilab/desktop/MCCT/session_cache'
OUT_DIR = 'results/v11/peak_coupling_viz'


def find_peak_timepoints(scaffold_npz, meta, metric_key, n_peaks=5, n_low=5):
    """Find highest and lowest coupling timepoints per condition."""
    t = scaffold_npz['t_common']
    values = scaffold_npz[metric_key]
    segments = meta['segments']

    peaks = {}
    for seg in segments:
        cond, t0, t1 = seg[0], seg[1], seg[2]
        mask = (t >= t0) & (t <= t1)
        if mask.sum() < 20:
            continue

        idx = np.where(mask)[0]
        vals = values[idx]
        valid = np.isfinite(vals) & (vals != 0)
        if valid.sum() < 10:
            continue

        valid_idx = idx[valid]
        valid_vals = vals[valid]

        sorted_i = np.argsort(valid_vals)
        high_idx = valid_idx[sorted_i[-n_peaks:]][::-1]
        low_idx = valid_idx[sorted_i[:n_low]]

        peaks[cond] = {
            'high_scaffold_idx': high_idx,
            'high_t': t[high_idx],
            'high_vals': values[high_idx],
            'low_scaffold_idx': low_idx,
            'low_t': t[low_idx],
            'low_vals': values[low_idx],
        }

    return peaks


def get_raw_au_at_time(cached, participant, target_t_lsl, lsl_offset, window_s=1.0):
    """Get mean AU values in a window around a scaffold timepoint."""
    bl = cached[f'{participant}_blendshapes'][:, :52]  # 52 AUs only
    ts = cached[f'{participant}_blendshapes_ts']

    # Convert LSL scaffold time to local (zero-based) cached time
    local_t = target_t_lsl - lsl_offset
    mask = (ts >= local_t - window_s / 2) & (ts <= local_t + window_s / 2)
    if mask.sum() < 5:
        return np.full(52, np.nan)
    return bl[mask].mean(axis=0)


def get_raw_pose_at_time(cached, participant, target_t_lsl, lsl_offset, window_s=1.0):
    """Get mean pose keypoints in a window around a scaffold timepoint."""
    pose = cached[f'{participant}_pose']  # (T, 99) = 33 keypoints x 3
    ts = cached[f'{participant}_pose_ts']

    local_t = target_t_lsl - lsl_offset
    mask = (ts >= local_t - window_s / 2) & (ts <= local_t + window_s / 2)
    if mask.sum() < 3:
        return np.full((33, 3), np.nan)
    mean_pose = pose[mask].mean(axis=0)
    return mean_pose.reshape(33, 3)


def plot_au_comparison(au_high_p1, au_high_p2, au_low_p1, au_low_p2,
                       condition, metric_name, out_path):
    """Bar chart comparing AU activations at high vs low coupling."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    x = np.arange(len(AFFECT_AU_INDICES))
    width = 0.2

    # High coupling
    ax = axes[0]
    h_p1 = [au_high_p1[i] for i in AFFECT_AU_INDICES]
    h_p2 = [au_high_p2[i] for i in AFFECT_AU_INDICES]
    ax.bar(x - width / 2, h_p1, width, label='Therapist', color='#1565C0', alpha=0.8)
    ax.bar(x + width / 2, h_p2, width, label='Patient', color='#D32F2F', alpha=0.8)
    ax.set_ylabel('AU activation')
    ax.set_title(f'HIGH {metric_name} moments ({condition})', fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=-3)

    # Low coupling
    ax = axes[1]
    l_p1 = [au_low_p1[i] for i in AFFECT_AU_INDICES]
    l_p2 = [au_low_p2[i] for i in AFFECT_AU_INDICES]
    ax.bar(x - width / 2, l_p1, width, label='Therapist', color='#1565C0', alpha=0.8)
    ax.bar(x + width / 2, l_p2, width, label='Patient', color='#D32F2F', alpha=0.8)
    ax.set_ylabel('AU activation')
    ax.set_title(f'LOW {metric_name} moments ({condition})', fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=-3)

    ax.set_xticks(x)
    ax.set_xticklabels(AFFECT_AU_NAMES, rotation=45, ha='right', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def draw_stick_figure(ax, keypoints, color, label, x_offset=0):
    """Draw a stick figure from 33 MediaPipe pose keypoints."""
    kp = keypoints.copy()
    # MediaPipe: x=right, y=down, z=toward camera
    # Plot x vs -y (flip y so up is up)
    kp[:, 0] += x_offset

    for i, j in POSE_CONNECTIONS:
        if np.any(kp[i] == 0) or np.any(kp[j] == 0):
            continue
        ax.plot([kp[i, 0], kp[j, 0]], [-kp[i, 1], -kp[j, 1]],
                color=color, linewidth=2, alpha=0.8)

    # Draw joints
    valid = ~np.all(kp == 0, axis=1)
    ax.scatter(kp[valid, 0], -kp[valid, 1], c=color, s=15, zorder=5, alpha=0.8)
    ax.text(kp[0, 0], -kp[0, 1] + 0.05, label, fontsize=8, ha='center',
            color=color, fontweight='bold')


def plot_pose_comparison(pose_high_p1, pose_high_p2, pose_low_p1, pose_low_p2,
                         condition, metric_name, out_path):
    """Stick figure comparison at high vs low coupling."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ax = axes[0]
    draw_stick_figure(ax, pose_high_p1, '#1565C0', 'Therapist', x_offset=-0.3)
    draw_stick_figure(ax, pose_high_p2, '#D32F2F', 'Patient', x_offset=0.3)
    ax.set_title(f'HIGH {metric_name} ({condition})', fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.axis('off')

    ax = axes[1]
    draw_stick_figure(ax, pose_low_p1, '#1565C0', 'Therapist', x_offset=-0.3)
    draw_stick_figure(ax, pose_low_p2, '#D32F2F', 'Patient', x_offset=0.3)
    ax.set_title(f'LOW {metric_name} ({condition})', fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.axis('off')

    fig.suptitle(f'Pose at peak vs low coupling — {condition}', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    session_name = 'y_06'

    # Load scaffold
    npz = np.load(f'results/v11/{session_name}/scaffold_v11_ztimecourses.npz')
    with open(f'results/v11/{session_name}/scaffold_v11_results.json') as f:
        meta = json.load(f)

    # Load raw cached data
    cached_sessions = discover_cached_sessions(CACHE_DIR)
    cache_path = [p for n, p in cached_sessions if session_name in n][0]
    cached = load_session_from_cache(cache_path)

    # Compute LSL offset: scaffold t_common is LSL time, cached ts is zero-based
    scaffold_t = npz['t_common']
    lsl_offset = scaffold_t[0]  # approximate: scaffold[0] corresponds to cached[0]
    print(f"LSL offset: {lsl_offset:.1f}")

    # ── Facial coupling peaks ────────────────────────────────────────
    print("\nFinding facial coupling peaks...")
    bl_peaks = find_peak_timepoints(npz, meta, 'z_raw_bl_expr', n_peaks=5, n_low=5)

    for cond, data in bl_peaks.items():
        print(f"\n  {cond}: high z={data['high_vals'].mean():.2f}, low z={data['low_vals'].mean():.2f}")

        # Average AUs across the top-5 / bottom-5 timepoints
        au_high_p1 = np.nanmean([get_raw_au_at_time(cached, 'p1', t, lsl_offset) for t in data['high_t']], axis=0)
        au_high_p2 = np.nanmean([get_raw_au_at_time(cached, 'p2', t, lsl_offset) for t in data['high_t']], axis=0)
        au_low_p1 = np.nanmean([get_raw_au_at_time(cached, 'p1', t, lsl_offset) for t in data['low_t']], axis=0)
        au_low_p2 = np.nanmean([get_raw_au_at_time(cached, 'p2', t, lsl_offset) for t in data['low_t']], axis=0)

        # Print top differing AUs
        diff_high = np.abs(au_high_p1 - au_high_p2)
        diff_low = np.abs(au_low_p1 - au_low_p2)
        similarity_high = 1.0 - np.mean(diff_high[AFFECT_AU_INDICES])
        similarity_low = 1.0 - np.mean(diff_low[AFFECT_AU_INDICES])

        print(f"    Affect AU similarity: high={similarity_high:.3f}, low={similarity_low:.3f}")

        # Which AUs are most similar at high coupling?
        for idx in AFFECT_AU_INDICES:
            name = APPLE_AU_NAMES[idx]
            h1, h2 = au_high_p1[idx], au_high_p2[idx]
            l1, l2 = au_low_p1[idx], au_low_p2[idx]
            print(f"    {name:25s}: high T={h1:+.2f} P={h2:+.2f} (diff={abs(h1-h2):.2f}) | "
                  f"low T={l1:+.2f} P={l2:+.2f} (diff={abs(l1-l2):.2f})")

        plot_au_comparison(au_high_p1, au_high_p2, au_low_p1, au_low_p2,
                          cond, 'Facial Coupling',
                          os.path.join(OUT_DIR, f'face_{cond}.png'))

    # ── Postural coupling peaks ──────────────────────────────────────
    print("\n\nFinding postural coupling peaks...")
    pose_peaks = find_peak_timepoints(npz, meta, 'z_raw_pose', n_peaks=5, n_low=5)

    for cond, data in pose_peaks.items():
        print(f"\n  {cond}: high z={data['high_vals'].mean():.2f}, low z={data['low_vals'].mean():.2f}")

        pose_high_p1 = np.nanmean([get_raw_pose_at_time(cached, 'p1', t, lsl_offset) for t in data['high_t']], axis=0)
        pose_high_p2 = np.nanmean([get_raw_pose_at_time(cached, 'p2', t, lsl_offset) for t in data['high_t']], axis=0)
        pose_low_p1 = np.nanmean([get_raw_pose_at_time(cached, 'p1', t, lsl_offset) for t in data['low_t']], axis=0)
        pose_low_p2 = np.nanmean([get_raw_pose_at_time(cached, 'p2', t, lsl_offset) for t in data['low_t']], axis=0)

        plot_pose_comparison(pose_high_p1, pose_high_p2, pose_low_p1, pose_low_p2,
                            cond, 'Postural Coupling',
                            os.path.join(OUT_DIR, f'pose_{cond}.png'))

    print(f"\nFigures saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
