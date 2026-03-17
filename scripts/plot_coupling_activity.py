"""Plot coupling activity: when and where significant couplings are active.

Usage:
    python scripts/plot_coupling_activity.py --results results/cluster_session/y_06
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import uniform_filter1d

# Short names for display
MOD_SHORT = {
    'eeg_features': 'EEG', 'eeg_wavelet': 'EEG',
    'ecg_features': 'ECG', 'ecg_features_v2': 'ECG',
    'blendshapes': 'BL', 'blendshapes_v2': 'BL',
    'pose_features': 'Pose',
    'eeg_interbrain': 'EEGib',
}

# Colors by source modality
MOD_COLORS = {
    'eeg_features': '#2196F3', 'eeg_wavelet': '#2196F3',
    'ecg_features': '#F44336', 'ecg_features_v2': '#F44336',
    'blendshapes': '#4CAF50', 'blendshapes_v2': '#4CAF50',
    'pose_features': '#FF9800',
    'eeg_interbrain': '#9C27B0',
}

# Map feature name prefix to source modality key
FEAT_PREFIX_TO_SRC = {
    'bl_': 'blendshapes', 'eeg_': 'eeg_features', 'pose_': 'pose_features',
    'ecg_': 'ecg_features',
}


def _direction_to_prefix(direction):
    """Convert 'Patient -> Therapist' to 'patient_to_therapist' for filenames."""
    d = direction.replace('\u2192', '->').replace(' -> ', '_to_').replace(' ', '_')
    return d.lower()


def _direction_display(direction):
    """Ensure direction uses arrow for display."""
    return direction.replace('->', '\u2192')


def _feat_display(feat_name):
    """Strip modality prefix for display (bl_brow -> brow)."""
    for prefix in FEAT_PREFIX_TO_SRC:
        if feat_name.startswith(prefix):
            return feat_name[len(prefix):]
    return feat_name


def load_direction(npz_path, json_path):
    """Load NPZ + JSON for one direction, return structured data."""
    d = np.load(npz_path, allow_pickle=True)
    with open(json_path) as f:
        meta = json.load(f)

    times = d['times']

    sig_keys = d['pathway_significant/_keys']
    sig_vals = d['pathway_significant/_vals']

    pathways = []
    for k, is_sig in zip(sig_keys, sig_vals):
        src, tgt = k.split('||')
        dr2 = d[f'pathway_dr2/{k}']
        # Load per-timepoint p-values if available
        pval_key = f'pathway_pvalues/{k}'
        pvalues = d[pval_key] if pval_key in d else None
        pathways.append({
            'key': k, 'src': src, 'tgt': tgt,
            'dr2': dr2, 'pvalues': pvalues,
            'significant': bool(is_sig),
            'mean_dr2': float(np.nanmean(dr2)),
        })

    # Load per-feature dR2 timecourses
    feat_keys_raw = [k.replace('feature_dr2/', '')
                     for k in d.keys()
                     if k.startswith('feature_dr2/') and k != 'feature_dr2/_keys']
    features = {}
    for fk in feat_keys_raw:
        feat_name, tgt = fk.split('||')
        features[(feat_name, tgt)] = d[f'feature_dr2/{fk}']

    direction = meta['direction']
    return {
        'times': times,
        'pathways': pathways,
        'features': features,
        'direction': direction,
        'direction_display': _direction_display(direction),
        'file_prefix': _direction_to_prefix(direction),
        'session': meta['session'],
    }


def smooth(x, window_sec, dt):
    """Smooth with uniform filter, NaN-aware."""
    w = max(1, int(window_sec / dt))
    mask = np.isnan(x)
    x_filled = np.where(mask, 0.0, x)
    count = uniform_filter1d((~mask).astype(float), w, mode='nearest')
    summed = uniform_filter1d(x_filled, w, mode='nearest')
    result = np.where(count > 0, summed / (count + 1e-20), np.nan)
    return result


def _trim_times(times, trim_sec):
    dt = times[1] - times[0]
    trim_idx = int(trim_sec / dt)
    t_sl = slice(trim_idx, len(times) - trim_idx)
    return times[t_sl] / 60.0, t_sl, dt


def plot_activity_heatmap(data, save_path, smooth_sec=30, trim_sec=60):
    """Coupling activity heatmap: significant pathways x time."""
    times_plot, t_sl, dt = _trim_times(data['times'], trim_sec)

    sig_pw = [p for p in data['pathways'] if p['significant']]
    if not sig_pw:
        return

    sig_pw.sort(key=lambda p: (p['tgt'], -p['mean_dr2']))
    n_pw = len(sig_pw)

    heatmap = np.full((n_pw, len(times_plot)), np.nan)
    labels = []
    for i, pw in enumerate(sig_pw):
        dr2_s = smooth(pw['dr2'], smooth_sec, dt)[t_sl]
        # Use per-timepoint p-values if available, else fall back to dR2 > 0
        if pw.get('pvalues') is not None:
            pval_s = smooth(pw['pvalues'], smooth_sec, dt)[t_sl]
            # Mask: significant (p < 0.05) AND positive dR2
            active = (pval_s < 0.05) & (dr2_s > 0)
        else:
            active = dr2_s > 0
        heatmap[i] = np.where(active, dr2_s, 0.0)
        src_s = MOD_SHORT.get(pw['src'], pw['src'])
        tgt_s = MOD_SHORT.get(pw['tgt'], pw['tgt'])
        labels.append(f"{src_s}\u2192{tgt_s}")

    cmap = LinearSegmentedColormap.from_list('coupling', [
        (0.0, '#e0e0e0'), (0.001, '#e8f0fe'),
        (0.3, '#64b5f6'), (0.6, '#1976d2'), (1.0, '#0d47a1'),
    ])

    heatmap_masked = heatmap  # already masked above
    pos_vals = heatmap_masked[heatmap_masked > 0]
    vmax = np.percentile(pos_vals, 95) if len(pos_vals) > 0 else 0.3

    fig, ax = plt.subplots(figsize=(18, 0.6 * n_pw + 2.0), dpi=150)
    im = ax.imshow(heatmap_masked, cmap=cmap, vmin=0, vmax=vmax,
                   aspect='auto', interpolation='nearest',
                   extent=[times_plot[0], times_plot[-1], n_pw - 0.5, -0.5],
                   rasterized=True)

    ax.set_yticks(range(n_pw))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Time (min)', fontsize=11)
    has_pvals = any(pw.get('pvalues') is not None for pw in sig_pw)
    mask_label = 'p < 0.05 & dR\u00b2 > 0' if has_pvals else 'dR\u00b2 > 0'
    ax.set_title(
        f'Coupling Activity  \u2014  {data["direction_display"]}  ({data["session"]})\n'
        f'{mask_label} = coupling active  |  {smooth_sec}s smoothing  |  '
        f'{n_pw} significant pathways', fontsize=12)

    for i, pw in enumerate(sig_pw):
        color = MOD_COLORS.get(pw['src'], '#999')
        ax.barh(i, 0.3, left=times_plot[0] - 0.8, color=color,
                height=0.8, clip_on=False, zorder=5)

    cbar = fig.colorbar(im, ax=ax, label='dR\u00b2', shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_timecourses(data, save_path, smooth_sec=30, trim_sec=60):
    """dR2 timecourses for significant pathways."""
    times_plot, t_sl, dt = _trim_times(data['times'], trim_sec)

    sig_pw = [p for p in data['pathways'] if p['significant']]
    if not sig_pw:
        return

    sig_pw.sort(key=lambda p: -p['mean_dr2'])
    n_pw = len(sig_pw)

    fig, axes = plt.subplots(n_pw, 1, figsize=(18, 2.2 * n_pw + 1.5),
                              sharex=True, dpi=150)
    if n_pw == 1:
        axes = [axes]

    for ax, pw in zip(axes, sig_pw):
        dr2_raw = pw['dr2'][t_sl]
        dr2_s = smooth(pw['dr2'], smooth_sec, dt)[t_sl]
        color = MOD_COLORS.get(pw['src'], '#333')
        src_s = MOD_SHORT.get(pw['src'], pw['src'])
        tgt_s = MOD_SHORT.get(pw['tgt'], pw['tgt'])

        # Determine per-timepoint significance mask
        if pw.get('pvalues') is not None:
            pval_s = smooth(pw['pvalues'], smooth_sec, dt)[t_sl]
            active_mask = (pval_s < 0.05) & (dr2_s > 0)
        else:
            active_mask = dr2_s > 0

        ax.plot(times_plot, dr2_raw, color=color, linewidth=0.3, alpha=0.15)
        ax.plot(times_plot, dr2_s, color=color, linewidth=1.2, alpha=0.9)
        ax.fill_between(times_plot, 0, dr2_s,
                         where=active_mask, color=color, alpha=0.3)
        ax.fill_between(times_plot, 0, dr2_s,
                         where=~active_mask, color='#cccccc', alpha=0.2)
        ax.axhline(0, color='#888', linewidth=0.5, linestyle='--', alpha=0.5)

        active_frac = np.mean(active_mask)
        mean_active = np.nanmean(dr2_s[active_mask]) if np.any(active_mask) else 0

        ax.set_ylabel('dR\u00b2', fontsize=9)
        ax.set_title(
            f'{src_s} \u2192 {tgt_s}    '
            f'mean={pw["mean_dr2"]:.3f}    '
            f'active={active_frac:.0%}    '
            f'mean(active)={mean_active:.3f}',
            fontsize=9, loc='left', fontweight='bold')

        ylim = min(1.0, np.percentile(np.abs(dr2_s[~np.isnan(dr2_s)]), 99) * 1.3)
        ax.set_ylim(-ylim, ylim)

    axes[-1].set_xlabel('Time (min)', fontsize=11)
    has_pvals = any(pw.get('pvalues') is not None for pw in sig_pw)
    fill_label = 'p < 0.05 & dR\u00b2 > 0' if has_pvals else 'dR\u00b2 > 0'
    fig.suptitle(
        f'dR\u00b2 Timecourses  \u2014  {data["direction_display"]}  ({data["session"]})\n'
        f'Colored fill = coupling active ({fill_label})  |  {smooth_sec}s smoothing',
        fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_summary_matrix(data_list, save_path):
    """Bidirectional coupling matrix: mean dR2 with significance markers."""
    all_mods = set()
    for data in data_list:
        for pw in data['pathways']:
            all_mods.add(pw['src'])
            all_mods.add(pw['tgt'])

    order_pref = ['eeg_features', 'eeg_wavelet', 'eeg_interbrain',
                  'ecg_features', 'ecg_features_v2',
                  'blendshapes', 'blendshapes_v2', 'pose_features']
    mod_order = [m for m in order_pref if m in all_mods]
    n = len(mod_order)
    mod_idx = {m: i for i, m in enumerate(mod_order)}
    mod_labels = [MOD_SHORT.get(m, m) for m in mod_order]

    matrices = []
    sig_matrices = []
    titles = []
    for data in data_list:
        mat = np.full((n, n), np.nan)
        sig = np.zeros((n, n), dtype=bool)
        for pw in data['pathways']:
            if pw['src'] in mod_idx and pw['tgt'] in mod_idx:
                i, j = mod_idx[pw['src']], mod_idx[pw['tgt']]
                mat[i, j] = pw['mean_dr2']
                sig[i, j] = pw['significant']
        matrices.append(mat)
        sig_matrices.append(sig)
        titles.append(data['direction_display'])

    fig, axes = plt.subplots(1, len(data_list), figsize=(7 * len(data_list), 5.5),
                              dpi=150)
    if len(data_list) == 1:
        axes = [axes]

    vmax = max(np.nanmax(m) for m in matrices)
    vmax = min(vmax, 0.3)

    for ax, mat, sig, title in zip(axes, matrices, sig_matrices, titles):
        im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=vmax,
                       aspect='equal', origin='upper')
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                if np.isnan(val):
                    continue
                is_sig = sig[i, j]
                txt = f'{val:.3f}'
                if is_sig:
                    txt += '\n***'
                color = 'white' if val > vmax * 0.6 else 'black'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=7, fontweight='bold' if is_sig else 'normal',
                        color=color)
        ax.set_xticks(range(n))
        ax.set_xticklabels(mod_labels, fontsize=9, rotation=45, ha='right')
        ax.set_yticks(range(n))
        ax.set_yticklabels(mod_labels, fontsize=9)
        ax.set_xlabel('Target', fontsize=10)
        ax.set_ylabel('Source', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')

    fig.colorbar(im, ax=list(axes), label='Mean dR\u00b2', shrink=0.8)
    session = data_list[0]['session']
    fig.suptitle(f'Coupling Matrix  \u2014  {session}  (*** = p < 0.05)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _get_pathway_features(data, src, tgt):
    """Get features for a specific src->tgt pathway, sorted by mean dR2."""
    feats = []
    for (feat_name, feat_tgt), dr2 in data['features'].items():
        if feat_tgt != tgt:
            continue
        matched_src = None
        for prefix, src_mod in FEAT_PREFIX_TO_SRC.items():
            if feat_name.startswith(prefix):
                matched_src = src_mod
                break
        if matched_src == src:
            feats.append({
                'name': feat_name,
                'display': _feat_display(feat_name),
                'dr2': dr2,
                'mean_dr2': float(np.nanmean(dr2)),
            })
    feats.sort(key=lambda f: -f['mean_dr2'])
    return feats


def plot_feature_timecourses(data, save_dir, smooth_sec=30, trim_sec=60):
    """Per-feature dR2 timecourses for each significant pathway."""
    times_plot, t_sl, dt = _trim_times(data['times'], trim_sec)

    sig_pw = [p for p in data['pathways'] if p['significant']]
    if not sig_pw:
        return

    for pw in sig_pw:
        feats = _get_pathway_features(data, pw['src'], pw['tgt'])
        if not feats:
            continue

        n_feat = len(feats)
        src_s = MOD_SHORT.get(pw['src'], pw['src'])
        tgt_s = MOD_SHORT.get(pw['tgt'], pw['tgt'])

        fig, axes = plt.subplots(n_feat, 1,
                                  figsize=(18, 1.6 * n_feat + 1.5),
                                  sharex=True, dpi=150)
        if n_feat == 1:
            axes = [axes]

        cmap = plt.cm.tab20
        for i, (ax, feat) in enumerate(zip(axes, feats)):
            dr2_s = smooth(feat['dr2'], smooth_sec, dt)[t_sl]
            color = cmap(i / max(n_feat, 1))

            ax.plot(times_plot, feat['dr2'][t_sl], color=color,
                    linewidth=0.2, alpha=0.1)
            ax.plot(times_plot, dr2_s, color=color, linewidth=1.2, alpha=0.9)
            ax.fill_between(times_plot, 0, dr2_s,
                             where=dr2_s > 0, color=color, alpha=0.3)
            ax.fill_between(times_plot, 0, dr2_s,
                             where=dr2_s <= 0, color='#cccccc', alpha=0.15)
            ax.axhline(0, color='#888', linewidth=0.4, linestyle='--', alpha=0.4)

            active_frac = np.mean(dr2_s > 0)
            ax.set_ylabel('dR\u00b2', fontsize=7)
            ax.set_title(
                f'{feat["display"]}    '
                f'mean={feat["mean_dr2"]:.4f}    '
                f'active={active_frac:.0%}',
                fontsize=8, loc='left', fontweight='bold', color=color)

            ylim = min(0.5, np.nanpercentile(np.abs(dr2_s[~np.isnan(dr2_s)]), 99) * 1.5)
            ylim = max(ylim, 0.02)
            ax.set_ylim(-ylim, ylim)
            ax.tick_params(labelsize=7)

        axes[-1].set_xlabel('Time (min)', fontsize=10)
        fig.suptitle(
            f'{src_s} \u2192 {tgt_s} Feature Breakdown  \u2014  '
            f'{data["direction_display"]}  ({data["session"]})\n'
            f'{n_feat} features  |  {smooth_sec}s smoothing',
            fontsize=11, y=1.01)
        fig.tight_layout()

        fname = f'{data["file_prefix"]}_{src_s}-{tgt_s}_features.png'
        save_path = os.path.join(save_dir, fname)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")


def plot_feature_heatmaps(data, save_dir, smooth_sec=30, trim_sec=60):
    """Per-feature activity heatmaps for each significant pathway."""
    times_plot, t_sl, dt = _trim_times(data['times'], trim_sec)

    cmap = LinearSegmentedColormap.from_list('feat_coupling', [
        (0.0, '#f5f5f5'), (0.001, '#e8f5e9'),
        (0.3, '#66bb6a'), (0.6, '#2e7d32'), (1.0, '#1b5e20'),
    ])

    sig_pw = [p for p in data['pathways'] if p['significant']]

    for pw in sig_pw:
        feats = _get_pathway_features(data, pw['src'], pw['tgt'])
        if not feats:
            continue

        n_feat = len(feats)
        src_s = MOD_SHORT.get(pw['src'], pw['src'])
        tgt_s = MOD_SHORT.get(pw['tgt'], pw['tgt'])

        heatmap = np.full((n_feat, len(times_plot)), np.nan)
        labels = []
        for i, feat in enumerate(feats):
            dr2_s = smooth(feat['dr2'], smooth_sec, dt)[t_sl]
            heatmap[i] = np.where(dr2_s > 0, dr2_s, 0.0)
            labels.append(feat['display'])

        pos_vals = heatmap[heatmap > 0]
        vmax = np.percentile(pos_vals, 95) if len(pos_vals) > 0 else 0.1

        fig, ax = plt.subplots(figsize=(18, 0.45 * n_feat + 1.8), dpi=150)
        im = ax.imshow(heatmap, cmap=cmap, vmin=0, vmax=vmax,
                       aspect='auto', interpolation='nearest',
                       extent=[times_plot[0], times_plot[-1],
                               n_feat - 0.5, -0.5],
                       rasterized=True)

        ax.set_yticks(range(n_feat))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Time (min)', fontsize=10)
        ax.set_title(
            f'{src_s} \u2192 {tgt_s} Feature Activity  \u2014  '
            f'{data["direction_display"]}  ({data["session"]})\n'
            f'{n_feat} features  |  dR\u00b2 > 0 = active  |  '
            f'{smooth_sec}s smoothing', fontsize=10)

        cbar = fig.colorbar(im, ax=ax, label='dR\u00b2', shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        fig.tight_layout()

        fname = f'{data["file_prefix"]}_{src_s}-{tgt_s}_feature_activity.png'
        save_path = os.path.join(save_dir, fname)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, help='Results directory')
    parser.add_argument('--smooth', type=float, default=30,
                        help='Smoothing window in seconds (default: 30)')
    parser.add_argument('--trim', type=float, default=60,
                        help='Trim edge seconds (default: 60)')
    args = parser.parse_args()

    results_dir = args.results
    out_dir = results_dir

    # Load both directions from fixed NPZ/JSON filenames
    data_list = []
    for npz_name, json_name in [('p1-p2_full.npz', 'p1-p2_results.json'),
                                 ('p2-p1_full.npz', 'p2-p1_results.json')]:
        npz_path = os.path.join(results_dir, npz_name)
        json_path = os.path.join(results_dir, json_name)
        if os.path.exists(npz_path) and os.path.exists(json_path):
            data_list.append(load_direction(npz_path, json_path))

    def _p(s):
        return s.replace('\u2192', '->')

    print(f"Session: {data_list[0]['session']}")
    for data in data_list:
        n_sig = sum(1 for p in data['pathways'] if p['significant'])
        print(f"  {_p(data['direction'])}: {n_sig} sig pathways")
        print(f"    file prefix: {data['file_prefix']}")

    for data in data_list:
        prefix = data['file_prefix']

        plot_activity_heatmap(data,
                              os.path.join(out_dir, f'{prefix}_activity.png'),
                              smooth_sec=args.smooth, trim_sec=args.trim)

        plot_timecourses(data,
                         os.path.join(out_dir, f'{prefix}_timecourse.png'),
                         smooth_sec=args.smooth, trim_sec=args.trim)

        plot_feature_timecourses(data, out_dir,
                                  smooth_sec=args.smooth, trim_sec=args.trim)

        plot_feature_heatmaps(data, out_dir,
                               smooth_sec=args.smooth, trim_sec=args.trim)

    # Coupling matrix (both directions side by side)
    if len(data_list) == 2:
        plot_summary_matrix(data_list,
                            os.path.join(out_dir, 'coupling_matrix.png'))


if __name__ == '__main__':
    main()
