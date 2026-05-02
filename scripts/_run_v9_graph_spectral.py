"""V9: Spectral Graph Theory Analysis for CADENCE.

Three applications of spectral graph theory on real session data:
  1. Inter-brain bipartite Laplacian — λ₂(t) algebraic connectivity
  2. AU spectral clustering — data-driven AU grouping
  3. Graph spectral filtering — multimodal coupling flexibility
     + time-varying modality graph

Usage:
    python scripts/_run_v9_graph_spectral.py                 # y_06, all apps
    python scripts/_run_v9_graph_spectral.py --session y_17   # single session
    python scripts/_run_v9_graph_spectral.py --all            # batch: App 3 on all sessions + cross-session comparison
"""

import sys, os, time, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

from cadence.preprocess.eeg.coherence import eeg_band_coherence, DEFAULT_EEG_BANDS
from cadence.data import discover_cached_sessions, load_session_from_cache
from cadence.config import load_config
from cadence.constants import EEG_ROI_NAMES
from cadence.significance.spectral_graph import (
    reshape_roi_pairs_to_matrix,
    interbrain_laplacian_timeseries,
    interbrain_fiedler_summary,
    au_similarity_graph,
    spectral_cluster_aus,
    format_cluster_report,
    build_modality_graph,
    build_modality_graph_windowed,
    graph_lowpass, graph_highpass, graph_heat_kernel,
    coupling_flexibility_index,
    graph_fourier_transform,
    graph_spectral_summary,
    coupling_shuffle_null,
    protocol_label_permutation,
    edge_dynamics,
    per_condition_coupling_test,
    MEDIAPIPE_AU_NAMES,
    MODALITY_GROUPS,
)
from scripts.run_session_v6 import load_xdf_session, extract_bl_segment

# ── Constants ─────────────────────────────────────────────────────────

RAW_DIR = 'C:/Users/optilab/Desktop/CADENCE/raw sessions'
FS_OUT = 2.0

CONDITION_ORDER = ['base_EO', 'base_EC', 'baseline', 'conv_1',
                   'PE', 'PE_1', 'PE_2',
                   'meditate_B', 'meditate_K', 'conv_2']
CONDITION_COLORS = {
    'base_EO': '#E3F2FD', 'base_EC': '#E8EAF6', 'baseline': '#E3F2FD',
    'conv_1': '#FFF3E0', 'conv_2': '#FFF3E0',
    'PE': '#FCE4EC', 'PE_1': '#FCE4EC', 'PE_2': '#FCE4EC',
    'meditate_B': '#F3E5F5', 'meditate_K': '#E8F5E9',
}

# Condition type grouping for cross-session comparison
CONDITION_TYPES = {
    'baseline': ['base_EO', 'base_EC', 'baseline'],
    'conversation': ['conv_1', 'conv_2'],
    'meditation': ['meditate_B', 'meditate_K'],
    'psychoeducation': ['PE', 'PE_1', 'PE_2'],
}
CONDITION_TYPE_COLORS = {
    'baseline': '#90CAF9',
    'conversation': '#FFB74D',
    'meditation': '#CE93D8',
    'psychoeducation': '#EF9A9A',
}

MODALITY_KEYS = [
    'imcoh_theta', 'imcoh_alpha', 'imcoh_beta',
    'conc_theta', 'conc_alpha', 'conc_beta',
    'dyn_theta', 'dyn_alpha', 'dyn_beta',
    'asym_theta', 'asym_alpha', 'asym_beta',
    'bl_expr', 'bl_activity_conc',
    'ecg_lf', 'ecg_hf',
    'resp', 'pose',
]
MODALITY_NAMES_SHORT = [
    'ImCoh θ', 'ImCoh α', 'ImCoh β',
    'Conc θ', 'Conc α', 'Conc β',
    'Dyn θ', 'Dyn α', 'Dyn β',
    'Asym θ', 'Asym α', 'Asym β',
    'BL expr', 'BL act',
    'ECG LF', 'ECG HF',
    'Resp', 'Pose',
]

BAND_COLORS = {'theta': '#1565C0', 'alpha': '#E65100', 'beta': '#2E7D32'}

# Session protocol mapping
MEDITATION_SESSIONS = ['y_06', 'y_17', 'y_19', 'y11', 'y04', 'y24', 'y_45']
PE_SESSIONS = ['y01', 'y05', 'y_10', 'y_32', 'y_41']


def _session_protocol(session_name):
    """Determine protocol from session name."""
    sn = session_name.lower().replace('_', '').replace('-', '')
    # Strip date suffixes
    for med in MEDITATION_SESSIONS:
        if med.lower().replace('_', '') in sn:
            return 'meditation'
    for pe in PE_SESSIONS:
        if pe.lower().replace('_', '') in sn:
            return 'PE'
    return 'unknown'


# ═════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═════════════════════════════════════════════════════════════════════

def _shade_conditions(ax, segments, t_offset=0):
    for seg_name, t0, t1 in segments:
        color = CONDITION_COLORS.get(seg_name, '#F5F5F5')
        ax.axvspan(t0 - t_offset, t1 - t_offset, alpha=0.3, color=color,
                   linewidth=0, zorder=0)


def _condition_legend(segments):
    seen = {}
    for seg_name, _, _ in segments:
        if seg_name not in seen:
            seen[seg_name] = CONDITION_COLORS.get(seg_name, '#F5F5F5')
    return [Patch(facecolor=c, alpha=0.3, label=n) for n, c in seen.items()]


# ═════════════════════════════════════════════════════════════════════
#  Application 1: Inter-Brain Bipartite Laplacian
# ═════════════════════════════════════════════════════════════════════

def run_app1_interbrain(cached, session_data, segments, out_dir):
    """Compute and plot inter-brain bipartite Laplacian spectrum."""
    print("\n[App 1] Inter-Brain Bipartite Laplacian")
    print("-" * 50)

    if 'p1_eeg' not in cached or 'p2_eeg' not in cached:
        print("  No EEG data, skipping App 1")
        return None

    p1_eeg = cached['p1_eeg'].astype(np.float64)
    p2_eeg = cached['p2_eeg'].astype(np.float64)
    p1_ts = cached['p1_eeg_ts']
    n_ch = min(14, p1_eeg.shape[1], p2_eeg.shape[1])
    p1_eeg, p2_eeg = p1_eeg[:, :n_ch], p2_eeg[:, :n_ch]
    mlen = min(len(p1_eeg), len(p2_eeg))
    p1_eeg, p2_eeg = p1_eeg[:mlen], p2_eeg[:mlen]

    p1_eeg -= p1_eeg.mean(axis=1, keepdims=True)
    p2_eeg -= p2_eeg.mean(axis=1, keepdims=True)
    for ch in range(n_ch):
        for arr in [p1_eeg, p2_eeg]:
            sd = arr[:, ch].std()
            if sd > 1e-8:
                arr[:, ch] /= sd

    fs_eeg = len(p1_ts) / (p1_ts[-1] - p1_ts[0])

    print("  Computing ROI-pair ImCoh...", flush=True)
    t0 = time.time()
    _, coh_imag, coh_times, _ = eeg_band_coherence(
        p1_eeg, p2_eeg, fs=fs_eeg, use_imcoh=True)
    print(f"  ImCoh: {time.time() - t0:.1f}s, shape={coh_imag.shape}")

    coh_matrix = reshape_roi_pairs_to_matrix(coh_imag, n_rois=4)
    band_names = list(DEFAULT_EEG_BANDS.keys())

    print("  Computing bipartite Laplacian spectrum...", flush=True)
    graph_results = interbrain_laplacian_timeseries(coh_matrix, band_names)

    lsl_ts_p1 = session_data['landmarks']['P1'][0]
    lsl_offset = float(lsl_ts_p1[0]) - float(p1_ts[0])
    coh_times_lsl = coh_times + float(p1_ts[0]) + lsl_offset

    results = {'bands': {}}
    for bn in band_names:
        lam2 = graph_results[bn]['lambda2']
        fiedler = graph_results[bn]['fiedler']
        _, _, coupling_mat = interbrain_fiedler_summary(fiedler)
        band_result = {
            'mean_lambda2': float(lam2.mean()),
            'coupling_matrix': coupling_mat.tolist(),
            'per_condition': {},
        }
        for seg_name, t0_seg, t1_seg in segments:
            mask = (coh_times_lsl >= t0_seg) & (coh_times_lsl <= t1_seg)
            if mask.sum() > 0:
                band_result['per_condition'][seg_name] = {
                    'mean_lambda2': float(lam2[mask].mean()),
                    'std_lambda2': float(lam2[mask].std()),
                }
                print(f"    {bn:>6s} {seg_name:>12s}: λ₂ = {lam2[mask].mean():.4f}")
        results['bands'][bn] = band_result

    # ── Plot ──
    fig, axes = plt.subplots(len(band_names) + 1, 2, figsize=(18, 3.5 * (len(band_names) + 1)),
                             gridspec_kw={'width_ratios': [4, 1]})
    for bi, bn in enumerate(band_names):
        ax_ts, ax_f = axes[bi, 0], axes[bi, 1]
        lam2 = graph_results[bn]['lambda2']
        _shade_conditions(ax_ts, segments)
        ax_ts.plot(coh_times_lsl, lam2, color=BAND_COLORS[bn], linewidth=0.8)
        ax_ts.set_ylabel(f'λ₂ ({bn})')
        ax_ts.set_xlim(coh_times_lsl[0], coh_times_lsl[-1])
        ax_ts.set_ylim(bottom=0)
        if bi == 0:
            ax_ts.set_title('Algebraic Connectivity λ₂(t) — Inter-Brain Bipartite Graph')
        _, _, cmat = interbrain_fiedler_summary(graph_results[bn]['fiedler'])
        ax_f.imshow(cmat, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        roi_short = ['Fro', 'LT', 'RT', 'Pos']
        ax_f.set_xticks(range(4)); ax_f.set_yticks(range(4))
        ax_f.set_xticklabels(roi_short, fontsize=8); ax_f.set_yticklabels(roi_short, fontsize=8)
        ax_f.set_title(f'Fiedler ({bn})')

    ax_comb = axes[-1, 0]
    _shade_conditions(ax_comb, segments)
    combined_lam2 = np.mean([graph_results[bn]['lambda2'] for bn in band_names], axis=0)
    ax_comb.plot(coh_times_lsl, combined_lam2, color='k', linewidth=1.0)
    ax_comb.set_ylabel('λ₂ (mean)'); ax_comb.set_xlabel('Time (LSL seconds)')
    ax_comb.set_xlim(coh_times_lsl[0], coh_times_lsl[-1]); ax_comb.set_ylim(bottom=0)
    axes[-1, 1].axis('off')
    axes[-1, 1].legend(handles=_condition_legend(segments), loc='center', fontsize=8, frameon=False)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'v9_interbrain_lambda2.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: v9_interbrain_lambda2.png")
    return results


# ═════════════════════════════════════════════════════════════════════
#  Application 1b: Channel-Level (14×14) Bipartite Graph
# ═════════════════════════════════════════════════════════════════════

def run_app1b_channel_bipartite(cached, session_data, segments, out_dir):
    """14×14 channel-level bipartite graph — rich spatial spectrum."""
    from cadence.constants import EPOC_2D_POS, EPOC_CHANNEL_NAMES

    print("\n[App 1b] Channel-Level Bipartite Graph (14×14)")
    print("-" * 50)

    if 'p1_eeg' not in cached or 'p2_eeg' not in cached:
        print("  No EEG data, skipping")
        return None

    p1_eeg = cached['p1_eeg'].astype(np.float64)
    p2_eeg = cached['p2_eeg'].astype(np.float64)
    p1_ts = cached['p1_eeg_ts']
    n_ch = min(14, p1_eeg.shape[1], p2_eeg.shape[1])
    p1_eeg, p2_eeg = p1_eeg[:, :n_ch], p2_eeg[:, :n_ch]
    mlen = min(len(p1_eeg), len(p2_eeg))
    p1_eeg, p2_eeg = p1_eeg[:mlen], p2_eeg[:mlen]

    p1_eeg -= p1_eeg.mean(axis=1, keepdims=True)
    p2_eeg -= p2_eeg.mean(axis=1, keepdims=True)
    for ch in range(n_ch):
        for arr in [p1_eeg, p2_eeg]:
            sd = arr[:, ch].std()
            if sd > 1e-8:
                arr[:, ch] /= sd

    fs_eeg = len(p1_ts) / (p1_ts[-1] - p1_ts[0])

    # Per-channel roi_map: each "ROI" is one electrode
    ch_roi_map = {EPOC_CHANNEL_NAMES[i]: [i] for i in range(n_ch)}

    print("  Computing per-channel ImCoh (14×14)...", flush=True)
    t0 = time.time()
    _, coh_imag, coh_times, _ = eeg_band_coherence(
        p1_eeg, p2_eeg, fs=fs_eeg, use_imcoh=True,
        roi_map=ch_roi_map)
    print(f"  ImCoh: {time.time() - t0:.1f}s, shape={coh_imag.shape}")

    # Reshape (3, 196, W) → (3, 14, 14, W)
    coh_matrix = reshape_roi_pairs_to_matrix(coh_imag, n_rois=n_ch)
    band_names = list(DEFAULT_EEG_BANDS.keys())
    n_total = 2 * n_ch  # 28 nodes in bipartite graph

    # Compute bipartite Laplacian spectrum
    print("  Computing 28-node bipartite Laplacian...", flush=True)
    t0 = time.time()
    graph_results = interbrain_laplacian_timeseries(coh_matrix, band_names)
    print(f"  Laplacian: {time.time() - t0:.1f}s")

    # LSL time alignment
    lsl_ts_p1 = session_data['landmarks']['P1'][0]
    lsl_offset = float(lsl_ts_p1[0]) - float(p1_ts[0])
    coh_times_lsl = coh_times + float(p1_ts[0]) + lsl_offset

    # Per-condition analysis
    results = {'bands': {}, 'n_channels': n_ch}
    for bn in band_names:
        lam2 = graph_results[bn]['lambda2']
        fiedler = graph_results[bn]['fiedler']

        band_result = {
            'mean_lambda2': float(lam2.mean()),
            'std_lambda2': float(lam2.std()),
            'per_condition': {},
            'per_condition_fiedler': {},
        }

        for seg_name, t0_seg, t1_seg in segments:
            mask = (coh_times_lsl >= t0_seg) & (coh_times_lsl <= t1_seg)
            if mask.sum() > 0:
                band_result['per_condition'][seg_name] = {
                    'mean_lambda2': float(lam2[mask].mean()),
                    'std_lambda2': float(lam2[mask].std()),
                }
                # Condition-averaged Fiedler vector
                mean_fiedler = np.abs(fiedler[mask]).mean(axis=0)
                band_result['per_condition_fiedler'][seg_name] = mean_fiedler.tolist()
                print(f"    {bn:>6s} {seg_name:>12s}: λ₂ = {lam2[mask].mean():.4f} "
                      f"± {lam2[mask].std():.4f}")
        results['bands'][bn] = band_result

    # ── Plot 1: λ₂ timeseries (all bands) ─────────────────────────
    fig, axes = plt.subplots(len(band_names), 1, figsize=(18, 3 * len(band_names)),
                             sharex=True)
    for bi, bn in enumerate(band_names):
        ax = axes[bi]
        lam2 = graph_results[bn]['lambda2']
        _shade_conditions(ax, segments)
        ax.plot(coh_times_lsl, lam2, color=BAND_COLORS[bn], linewidth=0.6, alpha=0.8)
        ax.set_ylabel(f'λ₂ ({bn})')
        ax.set_xlim(coh_times_lsl[0], coh_times_lsl[-1])
        ax.set_ylim(bottom=0)
        if bi == 0:
            ax.set_title('Channel-Level (14×14) Bipartite Algebraic Connectivity')
    axes[-1].set_xlabel('Time (LSL seconds)')
    axes[-1].legend(handles=_condition_legend(segments), loc='lower right',
                    fontsize=7, ncol=3, frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'v9_channel_lambda2.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot 2: Per-condition Fiedler scalp topographies ──────────
    # For each band, show Fiedler loadings on P1 and P2 scalp maps per condition
    cond_list = [(s, t0s, t1s) for s, t0s, t1s in segments
                 if s in ['base_EO', 'base_EC', 'conv_1', 'meditate_B',
                          'meditate_K', 'conv_2', 'PE', 'PE_1', 'PE_2']]
    if not cond_list:
        cond_list = segments[:6]

    n_conds = len(cond_list)
    pos = EPOC_2D_POS[:n_ch]

    for bn in band_names:
        fiedler = graph_results[bn]['fiedler']
        fig, axes = plt.subplots(2, n_conds, figsize=(3.5 * n_conds, 7))
        if n_conds == 1:
            axes = axes[:, np.newaxis]

        vmax = 0
        for ci, (seg_name, t0_seg, t1_seg) in enumerate(cond_list):
            mask = (coh_times_lsl >= t0_seg) & (coh_times_lsl <= t1_seg)
            if mask.sum() > 0:
                mf = np.abs(fiedler[mask]).mean(axis=0)
                vmax = max(vmax, mf.max())

        vmax = max(vmax, 0.01)

        for ci, (seg_name, t0_seg, t1_seg) in enumerate(cond_list):
            mask = (coh_times_lsl >= t0_seg) & (coh_times_lsl <= t1_seg)
            if mask.sum() == 0:
                axes[0, ci].set_visible(False)
                axes[1, ci].set_visible(False)
                continue

            mf = np.abs(fiedler[mask]).mean(axis=0)
            p1_f = mf[:n_ch]
            p2_f = mf[n_ch:]

            for row, (participant, f_vals) in enumerate([('Therapist', p1_f), ('Patient', p2_f)]):
                ax = axes[row, ci]
                # Draw head circle
                theta = np.linspace(0, 2 * np.pi, 100)
                ax.plot(1.1 * np.cos(theta), 1.1 * np.sin(theta), 'k-', linewidth=0.5)
                # Nose marker
                ax.plot([0], [1.2], 'k^', markersize=5)

                sc = ax.scatter(pos[:, 0], pos[:, 1], c=f_vals, cmap='YlOrRd',
                               s=200, vmin=0, vmax=vmax, edgecolors='black',
                               linewidths=0.5, zorder=3)
                for ch_i in range(n_ch):
                    ax.annotate(EPOC_CHANNEL_NAMES[ch_i],
                               (pos[ch_i, 0], pos[ch_i, 1] - 0.15),
                               fontsize=5, ha='center', alpha=0.6)

                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
                ax.set_aspect('equal')
                ax.axis('off')
                if ci == 0:
                    ax.set_ylabel(participant, fontsize=11)
                    ax.yaxis.set_visible(True)
                    ax.yaxis.label.set_visible(True)
                if row == 0:
                    ax.set_title(seg_name, fontsize=10)

        fig.suptitle(f'Fiedler Topography — {bn} band (|loadings|, higher = more coupling)',
                     fontsize=12, y=1.02)
        fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.5, label='|Fiedler loading|',
                     pad=0.02)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'v9_channel_fiedler_{bn}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved: v9_channel_lambda2.png + v9_channel_fiedler_{{band}}.png")
    return results


# ═════════════════════════════════════════════════════════════════════
#  Application 2: AU Spectral Clustering
# ═════════════════════════════════════════════════════════════════════

def run_app2_au_clustering(session_data, segments, out_dir):
    """Compute and plot AU spectral clustering."""
    print("\n[App 2] AU Spectral Clustering")
    print("-" * 50)

    landmarks = session_data['landmarks']
    all_p1, all_p2 = [], []
    for seg_name, t0_seg, t1_seg in segments:
        p1_bl, p2_bl, dur = extract_bl_segment(landmarks, t0_seg, t1_seg)
        if p1_bl is not None and dur > 5:
            all_p1.append(p1_bl[:, :52])
            all_p2.append(p2_bl[:, :52])

    if not all_p1:
        print("  No BL data, skipping App 2")
        return None

    bl_p1, bl_p2 = np.vstack(all_p1), np.vstack(all_p2)
    print(f"  Total: {bl_p1.shape[0]} frames ({bl_p1.shape[0] / 30:.0f}s)")

    W, active_mask = au_similarity_graph(bl_p1, bl_p2)
    sc = spectral_cluster_aus(W, k_max=10)
    print(f"  k={sc['k']} clusters, {active_mask.sum()}/52 active AUs")

    report = format_cluster_report(sc['labels'], active_mask=active_mask)
    for cid, aus in report:
        print(f"    Cluster {cid} ({len(aus)}): {', '.join(aus[:6])}"
              + ("..." if len(aus) > 6 else ""))

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    eigvals = sc['eigvals']
    n_show = min(20, len(eigvals))
    axes[0].bar(range(n_show), eigvals[:n_show], color='#1565C0', alpha=0.7)
    axes[0].axvline(sc['k'] - 0.5, color='red', linestyle='--', label=f'k={sc["k"]}')
    axes[0].set_xlabel('Eigenvalue index'); axes[0].set_ylabel('λ')
    axes[0].set_title('Laplacian Eigenvalues (AU Graph)'); axes[0].legend()

    emb = sc['embedding']
    x_emb = emb[:, 0]; y_emb = emb[:, 1] if emb.shape[1] >= 2 else np.zeros(52)
    colors = plt.cm.tab10(sc['labels'] % 10)
    for i in range(52):
        if active_mask[i]:
            axes[1].scatter(x_emb[i], y_emb[i], c=[colors[i]], s=60, zorder=3)
            name = MEDIAPIPE_AU_NAMES[i] if i < len(MEDIAPIPE_AU_NAMES) else f'AU{i}'
            short = name.replace('mouth', 'm').replace('eye', 'e').replace('brow', 'b')
            axes[1].annotate(short, (x_emb[i], y_emb[i]), fontsize=5, alpha=0.7, ha='center', va='bottom')
    axes[1].set_xlabel('φ₁'); axes[1].set_ylabel('φ₂')
    axes[1].set_title(f'Spectral Embedding ({sc["k"]} clusters)')

    order = np.argsort(sc['labels'])
    im = axes[2].imshow(W[np.ix_(order, order)], cmap='YlOrRd', aspect='auto', interpolation='nearest')
    plt.colorbar(im, ax=axes[2], shrink=0.6)
    axes[2].set_title('AU Similarity (reordered)')

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'v9_au_clusters.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: v9_au_clusters.png")

    return {'k': sc['k'], 'n_active_aus': int(active_mask.sum()),
            'clusters': {str(cid): aus for cid, aus in report}}


# ═════════════════════════════════════════════════════════════════════
#  Application 3: Modality Graph + Time-Varying + Flexibility
# ═════════════════════════════════════════════════════════════════════

def _load_scaffold(session_name):
    """Load scaffold z-timecourses + metadata for a session."""
    scaffold_dir = f'results/rslds/{session_name}'
    npz_path = os.path.join(scaffold_dir, 'scaffold_v82_ztimecourses.npz')
    json_path = os.path.join(scaffold_dir, 'scaffold_v82_results.json')

    if not os.path.exists(npz_path):
        return None, None, None, None

    data = np.load(npz_path)
    t_common = data['t_common']
    N = len(t_common)
    z_cols = [data.get(f'z_{k}', np.zeros(N)) for k in MODALITY_KEYS]
    z_matrix = np.column_stack(z_cols)

    segments = []
    if os.path.exists(json_path):
        with open(json_path) as f:
            meta = json.load(f)
        for s in meta.get('segments', []):
            segments.append((s[0], s[1], s[2]))

    return z_matrix, t_common, segments, data


def run_app3_modality_graph(session_name, segments, out_dir, plot=True):
    """Build modality graph, time-varying graph, and coupling flexibility."""
    print(f"\n[App 3] Graph Spectral Analysis — {session_name}")
    print("-" * 50)

    z_matrix, t_common, scaffold_segments, _ = _load_scaffold(session_name)
    if z_matrix is None:
        print(f"  Scaffold not found for {session_name}")
        return None

    # Use scaffold segments if caller didn't provide them
    if not segments:
        segments = scaffold_segments

    N, D = z_matrix.shape
    print(f"  Loaded: {N} pts × {D} modalities, {(t_common[-1] - t_common[0]):.0f}s")

    # ── Static modality graph ─────────────────────────────────────
    W, L, eigvals, eigvecs = build_modality_graph(
        z_matrix, method='correlation', threshold=0.05)
    summary = graph_spectral_summary(eigvals, eigvecs, MODALITY_NAMES_SHORT)
    print(f"  Static graph: {summary['n_components']} components, "
          f"spectral gap={summary['spectral_gap']:.4f}")

    # Static flexibility
    flexibility, energy_low, energy_high = coupling_flexibility_index(
        z_matrix, eigvals, eigvecs, smooth_s=15, fs=FS_OUT)

    # ── Time-varying modality graph ───────────────────────────────
    print("  Computing time-varying graph (90s windows)...", flush=True)
    t0 = time.time()
    tv = build_modality_graph_windowed(z_matrix, window_s=90, stride_s=15,
                                       fs=FS_OUT, threshold=0.05)
    # Convert window centers to LSL times
    tv_times = tv['t_centers'] * (1.0 / FS_OUT) + t_common[0]
    # Correct: t_centers is already in seconds (samples / fs), add session start
    tv_times = tv['t_centers'] + t_common[0]
    print(f"  Time-varying: {len(tv_times)} windows, {time.time() - t0:.1f}s")
    print(f"    λ₂ range: [{tv['lambda2'].min():.3f}, {tv['lambda2'].max():.3f}]")
    print(f"    Components range: [{tv['n_components'].min()}, {tv['n_components'].max()}]")
    print(f"    Flexibility range: [{tv['flexibility'].min():.3f}, {tv['flexibility'].max():.3f}]")

    # Per-condition statistics
    cond_flex = {}
    cond_flex_tv = {}
    for seg_name, t0_seg, t1_seg in segments:
        # Static flexibility
        mask = (t_common >= t0_seg) & (t_common <= t1_seg)
        if mask.sum() > 10:
            cond_flex[seg_name] = {
                'mean': float(flexibility[mask].mean()),
                'std': float(flexibility[mask].std()),
            }
        # Time-varying
        tv_mask = (tv_times >= t0_seg) & (tv_times <= t1_seg)
        if tv_mask.sum() > 0:
            cond_flex_tv[seg_name] = {
                'flex_mean': float(tv['flexibility'][tv_mask].mean()),
                'flex_std': float(tv['flexibility'][tv_mask].std()),
                'lambda2_mean': float(tv['lambda2'][tv_mask].mean()),
                'n_components_mean': float(tv['n_components'][tv_mask].mean()),
                'n_edges_mean': float(tv['n_edges'][tv_mask].mean()),
            }
            print(f"    {seg_name:>12s}: flex={tv['flexibility'][tv_mask].mean():.3f}, "
                  f"λ₂={tv['lambda2'][tv_mask].mean():.3f}, "
                  f"comp={tv['n_components'][tv_mask].mean():.1f}, "
                  f"edges={tv['n_edges'][tv_mask].mean():.0f}")

    # ── Edge dynamics ────────────────────────────────────────────
    print("  Computing edge dynamics...", flush=True)
    ed = edge_dynamics(z_matrix, window_s=90, stride_s=15, fs=FS_OUT, threshold=0.05)
    ed_times = ed['t_centers'] + t_common[0]

    # Key cross-modal pairs summary
    cross_modal_pairs = [gp for gp in ed['group_pairs']
                         if gp.split('↔')[0] != gp.split('↔')[1]]
    print(f"    Cross-modal edges: mean={ed['cross_modal_count'].mean():.0f}, "
          f"range=[{ed['cross_modal_count'].min()}, {ed['cross_modal_count'].max()}]")
    for gp in cross_modal_pairs:
        mc = ed['group_pair_counts'][gp]
        if mc.max() > 0:
            mw = ed['group_pair_weights'][gp]
            print(f"      {gp:>12s}: mean_edges={mc.mean():.1f}, "
                  f"mean_weight={mw[mw > 0].mean():.3f}" if (mw > 0).any() else
                  f"      {gp:>12s}: mean_edges={mc.mean():.1f}")

    # Per-condition edge counts
    edge_cond = {}
    for seg_name, t0_seg, t1_seg in segments:
        emask = (ed_times >= t0_seg) & (ed_times <= t1_seg)
        if emask.sum() > 0:
            edge_cond[seg_name] = {
                'cross_modal': float(ed['cross_modal_count'][emask].mean()),
                'within_modal': float(ed['within_modal_count'][emask].mean()),
            }

    # ── Per-condition coupling-shuffle ────────────────────────────
    print("  Per-condition coupling-shuffle (300 perms)...", flush=True)
    t0 = time.time()
    pcc = per_condition_coupling_test(z_matrix, t_common, segments,
                                       n_perms=300, threshold=0.05)
    for cname, cr in pcc.items():
        sig = '*' if cr['p_two_sided'] < 0.05 else ' '
        print(f"    {cname:>12s}: real={cr['real_flex']:.3f}, "
              f"null={cr['null_mean']:.3f}±{cr['null_std']:.3f}, "
              f"d={cr['cohens_d']:+.2f} {sig}")
    print(f"  Permutation: {time.time() - t0:.1f}s")

    base_result = {
        'graph_spectral_summary': summary,
        'eigvals': eigvals.tolist(),
        'flexibility_mean': float(flexibility.mean()),
        'flexibility_std': float(flexibility.std()),
        'per_condition_flexibility': cond_flex,
        'per_condition_timevarying': cond_flex_tv,
        'per_condition_edges': edge_cond,
        'per_condition_coupling_test': {k: {kk: vv for kk, vv in v.items()}
                                         for k, v in pcc.items()},
        'protocol': _session_protocol(session_name),
    }

    if not plot:
        return base_result

    # ── Plot 1: Modality graph structure ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    im = axes[0].imshow(W, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    plt.colorbar(im, ax=axes[0], shrink=0.7, label='|corr|')
    axes[0].set_xticks(range(D)); axes[0].set_yticks(range(D))
    axes[0].set_xticklabels(MODALITY_NAMES_SHORT, rotation=90, fontsize=6)
    axes[0].set_yticklabels(MODALITY_NAMES_SHORT, fontsize=6)
    axes[0].set_title('Modality Graph Adjacency')

    axes[1].bar(range(D), eigvals, color='#1565C0', alpha=0.7)
    cutoff = float(np.median(eigvals[eigvals > 1e-8])) if (eigvals > 1e-8).any() else 0.5
    axes[1].axhline(cutoff, color='red', linestyle='--', label=f'cutoff={cutoff:.2f}')
    axes[1].set_xlabel('Graph frequency'); axes[1].set_ylabel('λ')
    axes[1].set_title('Graph Laplacian Eigenvalues'); axes[1].legend()

    for mi in range(min(3, D)):
        axes[2].barh(np.arange(D) + mi * 0.15, eigvecs[:, mi], height=0.13,
                     alpha=0.7, label=f'φ_{mi} (λ={eigvals[mi]:.2f})')
    axes[2].set_yticks(range(D)); axes[2].set_yticklabels(MODALITY_NAMES_SHORT, fontsize=6)
    axes[2].set_title('Top 3 Eigenmodes'); axes[2].legend(fontsize=7, loc='lower right')
    axes[2].axvline(0, color='gray', linewidth=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'v9_modality_graph.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot 2: Time-varying graph + flexibility timeline ─────────
    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)

    # Row 1: Time-varying λ₂ (modality graph connectivity)
    ax = axes[0]
    _shade_conditions(ax, segments)
    ax.plot(tv_times, tv['lambda2'], color='#1565C0', linewidth=1.0)
    ax.set_ylabel('λ₂ (modality graph)')
    ax.set_title(f'{session_name} — Time-Varying Modality Graph Spectrum')

    # Row 2: Number of connected components
    ax = axes[1]
    _shade_conditions(ax, segments)
    ax.plot(tv_times, tv['n_components'], color='#E65100', linewidth=1.0)
    ax.set_ylabel('# Components')
    ax.set_ylim(0, D + 1)

    # Row 3: Number of edges
    ax = axes[2]
    _shade_conditions(ax, segments)
    ax.plot(tv_times, tv['n_edges'], color='#2E7D32', linewidth=1.0)
    ax.set_ylabel('# Edges')

    # Row 4: Time-varying flexibility
    ax = axes[3]
    _shade_conditions(ax, segments)
    ax.plot(tv_times, tv['flexibility'], color='k', linewidth=1.0)
    ax.fill_between(tv_times, 0, tv['flexibility'], alpha=0.15, color='steelblue')
    ax.set_ylabel('Flexibility')
    ax.set_xlabel('Time (LSL seconds)')
    ax.set_ylim(0, 1)
    ax.axhline(tv['flexibility'].mean(), color='red', linestyle='--', linewidth=0.8,
               alpha=0.6, label=f'mean={tv["flexibility"].mean():.2f}')
    ax.legend(handles=_condition_legend(segments) + [
        Patch(facecolor='white', edgecolor='red', linestyle='--',
              label=f'mean={tv["flexibility"].mean():.2f}')
    ], fontsize=7, loc='upper right', ncol=4)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'v9_timevarying_graph.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot 3: Static flexibility (original) ─────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    ax = axes[0]
    _shade_conditions(ax, segments)
    show_mods = [0, 3, 12, 17]
    for mi in show_mods:
        ax.plot(t_common, z_matrix[:, mi], linewidth=0.5, alpha=0.5,
                label=MODALITY_NAMES_SHORT[mi])
    ax.set_ylabel('z-score'); ax.set_title('Original Signals')
    ax.legend(fontsize=7, loc='upper right', ncol=2)

    z_lp = graph_lowpass(z_matrix, eigvals, eigvecs)
    ax = axes[1]
    _shade_conditions(ax, segments)
    for mi in show_mods:
        ax.plot(t_common, z_lp[:, mi], linewidth=0.7, alpha=0.7,
                label=f'{MODALITY_NAMES_SHORT[mi]} (LP)')
    ax.set_ylabel('z-score'); ax.set_title('Graph Low-Pass (Shared Component)')
    ax.legend(fontsize=7, loc='upper right', ncol=2)

    ax = axes[2]
    _shade_conditions(ax, segments)
    ax.plot(t_common, flexibility, color='k', linewidth=1.0)
    ax.fill_between(t_common, 0, flexibility, alpha=0.15, color='steelblue')
    ax.set_ylabel('Flexibility'); ax.set_xlabel('Time (LSL seconds)')
    ax.set_ylim(0, 1)
    ax.axhline(flexibility.mean(), color='red', linestyle='--', linewidth=0.8,
               label=f'mean={flexibility.mean():.2f}')
    ax.set_title('Static Coupling Flexibility'); ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'v9_flexibility.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    # ── Plot 4: Edge dynamics ─────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

    # Row 1: Cross-modal vs within-modal edge count
    ax = axes[0]
    _shade_conditions(ax, segments)
    ax.plot(ed_times, ed['cross_modal_count'], color='#E65100', linewidth=1.0,
            label='Cross-modal edges')
    ax.plot(ed_times, ed['within_modal_count'], color='#1565C0', linewidth=1.0,
            alpha=0.5, label='Within-modal edges')
    ax.set_ylabel('# Edges')
    ax.set_title(f'{session_name} — Edge Dynamics: Cross-Modal Coupling Formation')
    ax.legend(fontsize=8, loc='upper right')

    # Row 2: Key cross-modal pair edge counts (stacked area)
    ax = axes[1]
    _shade_conditions(ax, segments)
    cm_colors = ['#E53935', '#FB8C00', '#43A047', '#1E88E5', '#8E24AA',
                 '#00897B', '#D81B60', '#5E35B1', '#F4511E', '#3949AB']
    ci = 0
    for gp in cross_modal_pairs:
        mc = ed['group_pair_counts'][gp]
        if mc.max() > 0:
            ax.plot(ed_times, mc, linewidth=0.9, alpha=0.8,
                    color=cm_colors[ci % len(cm_colors)], label=gp)
            ci += 1
    ax.set_ylabel('# Edges')
    ax.legend(fontsize=6, loc='upper right', ncol=3)
    ax.set_title('Cross-Modal Edge Counts by Modality Pair')

    # Row 3: Cross-modal ratio (cross / total)
    ax = axes[2]
    _shade_conditions(ax, segments)
    total = ed['cross_modal_count'] + ed['within_modal_count']
    ratio = np.where(total > 0, ed['cross_modal_count'] / total, 0)
    ax.plot(ed_times, ratio, color='k', linewidth=1.0)
    ax.fill_between(ed_times, 0, ratio, alpha=0.15, color='#E65100')
    ax.set_ylabel('Cross-modal fraction')
    ax.set_xlabel('Time (LSL seconds)')
    ax.set_ylim(0, 1)
    ax.set_title('Cross-Modal Edge Fraction (higher = more inter-modality coupling)')

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'v9_edge_dynamics.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot 5: Per-condition coupling-shuffle ─────────────────────
    n_conds = len(pcc)
    if n_conds > 0:
        fig, axes = plt.subplots(1, n_conds, figsize=(4 * n_conds, 4), squeeze=False)
        for ci, (cname, cr) in enumerate(pcc.items()):
            ax = axes[0, ci]
            color = CONDITION_COLORS.get(cname, '#E0E0E0')
            # Null bar
            ax.barh(0, cr['null_mean'], xerr=cr['null_std'], height=0.4,
                    color='#E0E0E0', capsize=5, ecolor='gray', alpha=0.7,
                    label='Null (shuffled)')
            # Real dot
            ax.scatter(cr['real_flex'], 0, c=color, s=120, zorder=3,
                       edgecolors='black', linewidths=1, label='Real')
            sig_str = f"d={cr['cohens_d']:+.1f}"
            if cr['p_two_sided'] < 0.05:
                sig_str += ' *'
            ax.set_title(f"{cname}\n{sig_str}", fontsize=10)
            ax.set_yticks([])
            ax.set_xlim(0, max(cr['real_flex'], cr['null_mean'] + 2 * cr['null_std']) * 1.2)
            if ci == 0:
                ax.set_xlabel('Mean Flexibility')
                ax.legend(fontsize=7, loc='lower right')

        fig.suptitle(f'{session_name} — Per-Condition Coupling Test', fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, 'v9_percondition_permutation.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved: v9_modality_graph.png, v9_timevarying_graph.png, v9_flexibility.png, "
          f"v9_edge_dynamics.png, v9_percondition_permutation.png")

    return base_result


# ═════════════════════════════════════════════════════════════════════
#  Single-Session Pipeline
# ═════════════════════════════════════════════════════════════════════

def run_session(session_name, raw_dir=RAW_DIR, full=True):
    """Run V9 analysis on one session. full=True runs all 3 apps."""
    print(f"\n{'=' * 70}")
    print(f"  V9 Spectral Graph Theory — {session_name}")
    print(f"{'=' * 70}")
    t_wall = time.time()

    out_dir = f'results/v9/{session_name}'
    os.makedirs(out_dir, exist_ok=True)
    cfg = load_config()

    # Discover segments from scaffold metadata first (fast path)
    scaffold_dir = f'results/rslds/{session_name}'
    json_path = os.path.join(scaffold_dir, 'scaffold_v82_results.json')
    segments = []
    if os.path.exists(json_path):
        with open(json_path) as f:
            meta = json.load(f)
        segments = [(s[0], s[1], s[2]) for s in meta.get('segments', [])]

    all_results = {
        'session': session_name,
        'version': 'v9',
        'protocol': _session_protocol(session_name),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    if full:
        # Load XDF + cache for Apps 1 & 2
        xdf_files = glob.glob(os.path.join(raw_dir, f'{session_name}*.xdf'))
        if not xdf_files:
            xdf_files = glob.glob(os.path.join(raw_dir, f'{session_name.upper()}*.xdf'))
        if xdf_files:
            session_data = load_xdf_session(xdf_files[0])
            markers = session_data['markers']

            # Rediscover segments from markers if not from scaffold
            if not segments:
                for seg_name in CONDITION_ORDER:
                    t_start = markers.get(f'{seg_name}_start')
                    t_end = markers.get(f'{seg_name}_stop')
                    if t_start is not None and t_end is not None and t_end > t_start:
                        segments.append((seg_name, t_start, t_end))

            cached_sessions = discover_cached_sessions(cfg['session_cache'])
            cache_matches = [p for n, p in cached_sessions
                             if session_name.lower() in n.lower()]
            if cache_matches:
                cached = load_session_from_cache(cache_matches[0], cfg)
                r1 = run_app1_interbrain(cached, session_data, segments, out_dir)
                if r1:
                    all_results['app1_interbrain_laplacian'] = {
                        k: v for k, v in r1.items() if k != 'coh_times_lsl'}
                r1b = run_app1b_channel_bipartite(cached, session_data, segments, out_dir)
                if r1b:
                    all_results['app1b_channel_bipartite'] = r1b
                r2 = run_app2_au_clustering(session_data, segments, out_dir)
                if r2:
                    all_results['app2_au_clustering'] = r2

    # App 3 always runs (only needs scaffold NPZ)
    r3 = run_app3_modality_graph(session_name, segments, out_dir, plot=True)
    if r3:
        r3_json = {k: v for k, v in r3.items() if k != 'adjacency_matrix'}
        all_results['app3_modality_graph'] = r3_json

    json_out = os.path.join(out_dir, 'v9_results.json')
    with open(json_out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {json_out}  ({time.time() - t_wall:.1f}s)")
    return all_results


# ═════════════════════════════════════════════════════════════════════
#  Batch Mode: All Sessions + Cross-Session Comparison
# ═════════════════════════════════════════════════════════════════════

def run_all_sessions():
    """Run App 3 on all sessions with scaffold data, then cross-session comparison."""
    print(f"\n{'=' * 70}")
    print(f"  V9 Batch — All Sessions")
    print(f"{'=' * 70}")
    t_wall = time.time()

    # Find all sessions with scaffold data
    rslds_dir = 'results/rslds'
    sessions = []
    for entry in sorted(os.listdir(rslds_dir)):
        npz = os.path.join(rslds_dir, entry, 'scaffold_v82_ztimecourses.npz')
        if os.path.isfile(npz):
            sessions.append(entry)
    print(f"  Found {len(sessions)} sessions with scaffold data")

    # Run App 3 on each session (fast — no XDF loading needed)
    all_results = {}
    for session_name in sessions:
        out_dir = f'results/v9/{session_name}'
        os.makedirs(out_dir, exist_ok=True)
        r3 = run_app3_modality_graph(session_name, [], out_dir, plot=True)
        if r3:
            all_results[session_name] = r3
            # Save per-session JSON
            session_result = {
                'session': session_name,
                'version': 'v9',
                'protocol': r3.get('protocol', 'unknown'),
                'app3_modality_graph': {k: v for k, v in r3.items()
                                         if k != 'adjacency_matrix'},
            }
            with open(os.path.join(out_dir, 'v9_results.json'), 'w') as f:
                json.dump(session_result, f, indent=2)

    if not all_results:
        print("  No results to compare")
        return

    # ── Cross-session comparison ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  Cross-Session Comparison ({len(all_results)} sessions)")
    print(f"{'=' * 70}")

    grand_dir = 'results/v9/grand_comparison'
    os.makedirs(grand_dir, exist_ok=True)

    # Collect per-condition-type flexibility across all sessions
    # condition_type → [(session, protocol, mean_flex, std_flex), ...]
    ct_data = {ct: [] for ct in CONDITION_TYPES}
    session_summaries = []

    for session_name, r3 in all_results.items():
        protocol = r3.get('protocol', 'unknown')
        cond_tv = r3.get('per_condition_timevarying', {})
        cond_static = r3.get('per_condition_flexibility', {})

        session_sum = {
            'session': session_name,
            'protocol': protocol,
            'overall_flex': r3.get('flexibility_mean', np.nan),
        }

        for ct_name, ct_conditions in CONDITION_TYPES.items():
            flex_vals = []
            for cname in ct_conditions:
                if cname in cond_tv:
                    flex_vals.append(cond_tv[cname]['flex_mean'])
                elif cname in cond_static:
                    flex_vals.append(cond_static[cname]['mean'])
            if flex_vals:
                mean_f = float(np.mean(flex_vals))
                ct_data[ct_name].append({
                    'session': session_name,
                    'protocol': protocol,
                    'flex_mean': mean_f,
                })
                session_sum[f'flex_{ct_name}'] = mean_f

        session_summaries.append(session_sum)

    # Print summary table
    print(f"\n  {'Session':>20s} | {'Protocol':>10s} | {'Baseline':>8s} | "
          f"{'Conv':>8s} | {'Med':>8s} | {'PE':>8s} | {'Overall':>8s}")
    print("  " + "-" * 85)
    for ss in sorted(session_summaries, key=lambda x: x['protocol']):
        print(f"  {ss['session']:>20s} | {ss['protocol']:>10s} | "
              f"{ss.get('flex_baseline', np.nan):8.3f} | "
              f"{ss.get('flex_conversation', np.nan):8.3f} | "
              f"{ss.get('flex_meditation', np.nan):8.3f} | "
              f"{ss.get('flex_psychoeducation', np.nan):8.3f} | "
              f"{ss['overall_flex']:8.3f}")

    # ── Statistical tests ─────────────────────────────────────────
    print("\n  Statistical tests (Mann-Whitney U):")
    stats = {}

    # Test: conversation flexibility — meditation vs PE sessions
    for ct_name in ['conversation', 'baseline']:
        med_flex = [d['flex_mean'] for d in ct_data[ct_name] if d['protocol'] == 'meditation']
        pe_flex = [d['flex_mean'] for d in ct_data[ct_name] if d['protocol'] == 'PE']
        if len(med_flex) >= 2 and len(pe_flex) >= 2:
            U, p = mannwhitneyu(med_flex, pe_flex, alternative='two-sided')
            print(f"    {ct_name}: meditation({len(med_flex)}) vs PE({len(pe_flex)}): "
                  f"med={np.mean(med_flex):.3f} vs {np.mean(pe_flex):.3f}, "
                  f"U={U:.0f}, p={p:.4f}")
            stats[f'{ct_name}_med_vs_pe'] = {
                'U': float(U), 'p': float(p),
                'med_mean': float(np.mean(med_flex)),
                'pe_mean': float(np.mean(pe_flex)),
                'med_n': len(med_flex), 'pe_n': len(pe_flex),
            }

    # Test: within-session conversation vs baseline
    conv_all = [d['flex_mean'] for d in ct_data['conversation']]
    base_all = [d['flex_mean'] for d in ct_data['baseline']]
    if len(conv_all) >= 2 and len(base_all) >= 2:
        U, p = mannwhitneyu(conv_all, base_all, alternative='two-sided')
        print(f"    conv vs baseline (all): "
              f"conv={np.mean(conv_all):.3f} vs base={np.mean(base_all):.3f}, "
              f"U={U:.0f}, p={p:.4f}")
        stats['conv_vs_baseline'] = {
            'U': float(U), 'p': float(p),
            'conv_mean': float(np.mean(conv_all)),
            'base_mean': float(np.mean(base_all)),
        }

    # ── Permutation Test 1: Protocol-label permutation ───────────
    print("\n  Permutation test: protocol-label (conversation flexibility):")
    conv_flex_by_session = {}
    conv_protocol_by_session = {}
    for d in ct_data['conversation']:
        conv_flex_by_session[d['session']] = d['flex_mean']
        conv_protocol_by_session[d['session']] = d['protocol']

    perm_result = protocol_label_permutation(
        conv_flex_by_session, conv_protocol_by_session,
        group_a='meditation', group_b='PE', exhaustive=True)
    if 'error' not in perm_result:
        print(f"    Observed diff: {perm_result['observed_diff']:+.4f} "
              f"(med={perm_result['group_a_mean']:.3f}, PE={perm_result['group_b_mean']:.3f})")
        print(f"    Permutation p = {perm_result['p_value']:.4f} "
              f"(exhaustive, {perm_result['n_perms_used']} arrangements)")
        stats['protocol_permutation_conversation'] = {
            'observed_diff': perm_result['observed_diff'],
            'p_value': perm_result['p_value'],
            'n_perms': perm_result['n_perms_used'],
            'n_meditation': perm_result['n_a'],
            'n_pe': perm_result['n_b'],
        }
    else:
        print(f"    {perm_result['error']}")

    # ── Permutation Test 2: Coupling-shuffle per session ──────────
    print("\n  Coupling-shuffle permutation (500 per session):")
    shuffle_results = {}
    for session_name in all_results:
        z_matrix, t_common, _, _ = _load_scaffold(session_name)
        if z_matrix is None:
            continue
        sr = coupling_shuffle_null(z_matrix, n_perms=500, threshold=0.05)
        p_str = f"p_low={sr['p_value_low']:.3f}, p_high={sr['p_value_high']:.3f}"
        direction = "LOWER" if sr['real_flex_mean'] < np.median(sr['null_flex_mean']) else "HIGHER"
        print(f"    {session_name:>20s}: real={sr['real_flex_mean']:.3f}, "
              f"null={np.mean(sr['null_flex_mean']):.3f}±{np.std(sr['null_flex_mean']):.3f}, "
              f"{direction}, {p_str}")
        shuffle_results[session_name] = {
            'real_flex': sr['real_flex_mean'],
            'null_mean': float(np.mean(sr['null_flex_mean'])),
            'null_std': float(np.std(sr['null_flex_mean'])),
            'p_low': sr['p_value_low'],
            'p_high': sr['p_value_high'],
            'p_two_sided': sr['p_value_two_sided'],
        }
    stats['coupling_shuffle'] = shuffle_results

    # Summary: how many sessions have significantly low flexibility?
    n_sig_low = sum(1 for sr in shuffle_results.values() if sr['p_low'] < 0.05)
    n_sig_high = sum(1 for sr in shuffle_results.values() if sr['p_high'] < 0.05)
    print(f"\n    Sessions with significantly LOW flex (coupling-driven): "
          f"{n_sig_low}/{len(shuffle_results)}")
    print(f"    Sessions with significantly HIGH flex: "
          f"{n_sig_high}/{len(shuffle_results)}")

    # ── Grand comparison plots ────────────────────────────────────

    # Plot 1: Flexibility by condition type, colored by protocol
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Box/strip plot by condition type
    ax = axes[0]
    ct_order = ['baseline', 'conversation', 'meditation', 'psychoeducation']
    positions = []
    all_flex_data = []
    all_colors = []
    tick_labels = []

    for ci, ct_name in enumerate(ct_order):
        data = ct_data[ct_name]
        if not data:
            continue
        x_pos = ci
        positions.append(x_pos)
        tick_labels.append(ct_name.replace('psychoeducation', 'PE'))

        for d in data:
            color = '#CE93D8' if d['protocol'] == 'meditation' else '#EF9A9A'
            marker = 'o' if d['protocol'] == 'meditation' else 's'
            ax.scatter(x_pos + np.random.uniform(-0.15, 0.15), d['flex_mean'],
                      c=color, marker=marker, s=80, alpha=0.8, zorder=3,
                      edgecolors='black', linewidths=0.5)

        flex_vals = [d['flex_mean'] for d in data]
        ax.bar(x_pos, np.mean(flex_vals), width=0.5, alpha=0.15,
               color=CONDITION_TYPE_COLORS.get(ct_name, 'gray'))
        ax.errorbar(x_pos, np.mean(flex_vals), yerr=np.std(flex_vals),
                    fmt='_', color='black', linewidth=2, capsize=8)

    ax.set_xticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_ylabel('Coupling Flexibility', fontsize=12)
    ax.set_title('Flexibility by Condition Type', fontsize=13)
    ax.legend(handles=[Patch(facecolor='#CE93D8', label='Meditation protocol'),
                      Patch(facecolor='#EF9A9A', label='PE protocol')],
              loc='upper right', fontsize=9)

    # Panel B: Per-session overall flexibility, colored by protocol
    ax = axes[1]
    med_sessions = [(ss['session'], ss['overall_flex']) for ss in session_summaries
                    if ss['protocol'] == 'meditation' and not np.isnan(ss['overall_flex'])]
    pe_sessions = [(ss['session'], ss['overall_flex']) for ss in session_summaries
                   if ss['protocol'] == 'PE' and not np.isnan(ss['overall_flex'])]
    unk_sessions = [(ss['session'], ss['overall_flex']) for ss in session_summaries
                    if ss['protocol'] == 'unknown' and not np.isnan(ss['overall_flex'])]

    all_sorted = sorted(med_sessions + pe_sessions + unk_sessions, key=lambda x: x[1])
    for i, (sname, flex) in enumerate(all_sorted):
        protocol = _session_protocol(sname)
        color = '#CE93D8' if protocol == 'meditation' else (
            '#EF9A9A' if protocol == 'PE' else '#BDBDBD')
        ax.barh(i, flex, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.text(flex + 0.005, i, sname, fontsize=8, va='center')

    ax.set_xlabel('Overall Coupling Flexibility', fontsize=12)
    ax.set_ylabel('Session')
    ax.set_yticks([])
    ax.set_title('Per-Session Flexibility (sorted)', fontsize=13)
    ax.legend(handles=[Patch(facecolor='#CE93D8', label='Meditation'),
                      Patch(facecolor='#EF9A9A', label='PE'),
                      Patch(facecolor='#BDBDBD', label='Unknown')],
              loc='lower right', fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(grand_dir, 'v9_grand_flexibility_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {grand_dir}/v9_grand_flexibility_comparison.png")

    # Plot 2: Flexibility delta (conversation - baseline) per session
    fig, ax = plt.subplots(figsize=(12, 6))
    deltas = []
    for ss in session_summaries:
        conv_f = ss.get('flex_conversation', np.nan)
        base_f = ss.get('flex_baseline', np.nan)
        if not np.isnan(conv_f) and not np.isnan(base_f):
            deltas.append((ss['session'], ss['protocol'], conv_f - base_f))

    if deltas:
        deltas.sort(key=lambda x: x[2])
        for i, (sname, protocol, delta) in enumerate(deltas):
            color = '#CE93D8' if protocol == 'meditation' else (
                '#EF9A9A' if protocol == 'PE' else '#BDBDBD')
            ax.barh(i, delta, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.text(delta + 0.002 * np.sign(delta), i, sname, fontsize=8, va='center')
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlabel('Δ Flexibility (conversation − baseline)', fontsize=12)
        ax.set_yticks([])
        ax.set_title('Coupling Flexibility Change: Conversation vs Baseline', fontsize=13)
        ax.legend(handles=[Patch(facecolor='#CE93D8', label='Meditation'),
                          Patch(facecolor='#EF9A9A', label='PE')],
                  loc='lower right', fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(grand_dir, 'v9_grand_flexibility_delta.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {grand_dir}/v9_grand_flexibility_delta.png")

    # Plot 3: Permutation test results
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Protocol-label permutation null distribution
    ax = axes[0]
    if 'error' not in perm_result:
        ax.hist(perm_result['null_diffs'], bins=30, color='#90CAF9', alpha=0.7,
                edgecolor='black', linewidth=0.5, label='Null distribution')
        ax.axvline(perm_result['observed_diff'], color='red', linewidth=2,
                   linestyle='--', label=f"Observed: {perm_result['observed_diff']:+.3f}")
        ax.set_xlabel('Difference in mean flexibility (meditation − PE)', fontsize=11)
        ax.set_ylabel('Count')
        ax.set_title(f"Protocol-Label Permutation (p={perm_result['p_value']:.4f}, "
                     f"n={perm_result['n_perms_used']})", fontsize=12)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)

    # Panel B: Coupling-shuffle: real vs null per session
    ax = axes[1]
    sessions_sorted = sorted(shuffle_results.items(),
                             key=lambda x: x[1]['real_flex'] - x[1]['null_mean'])
    for i, (sname, sr) in enumerate(sessions_sorted):
        protocol = _session_protocol(sname)
        color = '#CE93D8' if protocol == 'meditation' else (
            '#EF9A9A' if protocol == 'PE' else '#BDBDBD')
        # Null range
        ax.barh(i, sr['null_mean'], xerr=sr['null_std'], color='#E0E0E0',
                alpha=0.5, capsize=3, ecolor='gray')
        # Real value
        marker = 'o' if protocol == 'meditation' else 's'
        ax.scatter(sr['real_flex'], i, c=color, marker=marker, s=80, zorder=3,
                   edgecolors='black', linewidths=0.5)
        sig = '*' if sr['p_two_sided'] < 0.05 else ''
        ax.text(max(sr['real_flex'], sr['null_mean'] + sr['null_std']) + 0.01,
                i, f"{sname} {sig}", fontsize=7, va='center')

    ax.set_xlabel('Mean Coupling Flexibility', fontsize=11)
    ax.set_yticks([])
    ax.set_title('Coupling-Shuffle: Real (dots) vs Null (bars)', fontsize=12)
    ax.legend(handles=[
        Patch(facecolor='#E0E0E0', label='Null (shuffled coupling)'),
        Patch(facecolor='#CE93D8', label='Real — Meditation'),
        Patch(facecolor='#EF9A9A', label='Real — PE'),
    ], fontsize=8, loc='lower right')

    plt.tight_layout()
    fig.savefig(os.path.join(grand_dir, 'v9_permutation_tests.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {grand_dir}/v9_permutation_tests.png")

    # Save grand results
    grand_results = {
        'version': 'v9',
        'n_sessions': len(all_results),
        'session_summaries': session_summaries,
        'condition_type_data': {k: v for k, v in ct_data.items()},
        'statistics': stats,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    json_path = os.path.join(grand_dir, 'v9_grand_results.json')
    with open(json_path, 'w') as f:
        json.dump(grand_results, f, indent=2)
    print(f"  Saved: {json_path}")
    print(f"\n  Total batch time: {time.time() - t_wall:.1f}s")

    return grand_results


# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V9: Spectral Graph Theory Analysis')
    parser.add_argument('--session', default='y_06', help='Session name')
    parser.add_argument('--all', action='store_true', help='Run all sessions + cross-session comparison')
    args = parser.parse_args()

    if args.all:
        run_all_sessions()
    else:
        run_session(args.session)
