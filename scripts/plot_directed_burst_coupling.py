"""Visualize V11 Directed Burst Coupling Results.

4 figures:
  1. Per-condition ECA asymmetry bars (+ = therapist leads)
  2. Per-condition TE asymmetry bars
  3. Lag profile: directed ECA rates per condition
  4. ECA vs TE scatter

Usage:
    python scripts/plot_directed_burst_coupling.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONDITION_COLORS = {
    'base_EO': '#90CAF9', 'base_EC': '#9FA8DA',
    'conv_1': '#FFE0B2', 'conv_2': '#FFCC80',
    'PE_1': '#F8BBD0', 'PE_2': '#F8BBD0',
    'meditate_B': '#CE93D8', 'meditate_K': '#A5D6A7',
}
BAND_COLORS = {'theta': '#1565C0', 'alpha': '#E65100', 'beta': '#2E7D32'}

CONDS_ALL = ['base_EO', 'base_EC', 'conv_1', 'conv_2',
             'meditate_B', 'meditate_K', 'PE_1', 'PE_2']

OUT_DIR = 'results/v11/directed_burst_coupling'


def load_results():
    with open(os.path.join(OUT_DIR, 'directed_coupling_results.json')) as f:
        return json.load(f)


def plot_condition_eca_bars():
    """Per-condition ECA asymmetry (+ = therapist leads)."""
    results = load_results()
    sessions = results['sessions']
    band_names = ['theta', 'alpha', 'beta']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for bi, band in enumerate(band_names):
        ax = axes[bi]
        means, ses, colors, labels = [], [], [], []

        for cond in CONDS_ALL:
            vals = []
            for s in sessions:
                if cond in s['per_condition'] and band in s['per_condition'][cond]:
                    vals.append(s['per_condition'][cond][band]['eca_asym'])
            if len(vals) >= 1:
                means.append(np.mean(vals))
                ses.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
                colors.append(CONDITION_COLORS.get(cond, '#E0E0E0'))
                labels.append(cond.replace('_', '\n'))

        if not means:
            continue

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=[s * 1.96 for s in ses], capsize=4,
               color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        for xi, m in enumerate(means):
            y_off = 0.005 if m >= 0 else -0.005
            va = 'bottom' if m >= 0 else 'top'
            ax.text(xi, m + y_off, f'{m:+.3f}', ha='center', va=va,
                    fontsize=7, fontweight='bold')

        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, fontweight='bold')
        ax.set_title(f'{band.upper()}', fontsize=11,
                     fontweight='bold', color=BAND_COLORS[band])
        if bi == 0:
            ax.set_ylabel('ECA Asymmetry\n(+ = therapist leads)',
                          fontsize=9, fontweight='bold')

    n = len(sessions)
    fig.suptitle(f'Directed ECA: Burst Lead/Lag Asymmetry (n={n} sessions)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'condition_eca_asymmetry.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_condition_te_bars():
    """Per-condition TE asymmetry (+ = therapist leads)."""
    results = load_results()
    sessions = results['sessions']
    band_names = ['theta', 'alpha', 'beta']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for bi, band in enumerate(band_names):
        ax = axes[bi]
        means, ses, colors, labels = [], [], [], []

        for cond in CONDS_ALL:
            vals = []
            for s in sessions:
                if cond in s['per_condition'] and band in s['per_condition'][cond]:
                    vals.append(s['per_condition'][cond][band]['te_asym'])
            if len(vals) >= 1:
                means.append(np.mean(vals))
                ses.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
                colors.append(CONDITION_COLORS.get(cond, '#E0E0E0'))
                labels.append(cond.replace('_', '\n'))

        if not means:
            continue

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=[s * 1.96 for s in ses], capsize=4,
               color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        for xi, m in enumerate(means):
            y_off = 0.001 if m >= 0 else -0.001
            va = 'bottom' if m >= 0 else 'top'
            ax.text(xi, m + y_off, f'{m:+.4f}', ha='center', va=va,
                    fontsize=6, fontweight='bold')

        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, fontweight='bold')
        ax.set_title(f'{band.upper()}', fontsize=11,
                     fontweight='bold', color=BAND_COLORS[band])
        if bi == 0:
            ax.set_ylabel('TE Asymmetry (nats)\n(+ = therapist leads)',
                          fontsize=9, fontweight='bold')

    n = len(sessions)
    fig.suptitle(f'Directed TE: Burst Information Transfer Asymmetry (n={n} sessions)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'condition_te_asymmetry.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_lag_profile():
    """Directed ECA rates at each lag, per condition."""
    results = load_results()
    sessions = results['sessions']
    band_names = ['theta', 'alpha', 'beta']

    # Group conditions
    cond_groups = {
        'Conversation': ['conv_1', 'conv_2'],
        'Meditation': ['meditate_B', 'meditate_K'],
        'Baseline': ['base_EO', 'base_EC'],
        'PE': ['PE_1', 'PE_2'],
    }
    group_colors = {
        'Conversation': '#FF9800', 'Meditation': '#9C27B0',
        'Baseline': '#607D8B', 'PE': '#E91E63',
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for bi, band in enumerate(band_names):
        ax = axes[bi]

        for gname, gconds in cond_groups.items():
            # Collect lag profiles for this group
            all_p1_leads = []
            all_p2_leads = []

            for s in sessions:
                bs = s.get('per_band_summary', {}).get(band, {})
                p1l = bs.get('eca_p1_leads', [])
                p2l = bs.get('eca_p2_leads', [])
                lags = bs.get('eca_lags_s', [])
                if p1l and p2l:
                    all_p1_leads.append(p1l)
                    all_p2_leads.append(p2l)

            if not all_p1_leads or not lags:
                continue

            mean_p1 = np.mean(all_p1_leads, axis=0)
            mean_p2 = np.mean(all_p2_leads, axis=0)
            lags_arr = np.array(lags)

            color = group_colors[gname]
            ax.plot(lags_arr, mean_p1, '-o', color=color, markersize=4,
                    label=f'{gname} (therapist leads)', alpha=0.8)
            ax.plot(-lags_arr, mean_p2, '-s', color=color, markersize=4,
                    alpha=0.4, linestyle='--')

        ax.axvline(0, color='gray', ls=':', lw=0.5)
        ax.axhline(0, color='gray', ls=':', lw=0.5)
        ax.set_xlabel('Lag (s)\n(+ = therapist leads)', fontsize=8, fontweight='bold')
        ax.set_title(f'{band.upper()}', fontsize=11,
                     fontweight='bold', color=BAND_COLORS[band])
        if bi == 0:
            ax.set_ylabel('ECA Rate', fontsize=9, fontweight='bold')
            ax.legend(fontsize=6, loc='upper right')

    fig.suptitle('Directed ECA Lag Profile (session-averaged)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'lag_profile.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_eca_vs_te_scatter():
    """ECA asymmetry vs TE asymmetry per session per band."""
    results = load_results()
    sessions = results['sessions']
    band_names = ['theta', 'alpha', 'beta']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for bi, band in enumerate(band_names):
        ax = axes[bi]
        eca_vals, te_vals, labels = [], [], []

        for s in sessions:
            pc = s.get('per_condition', {})
            # Average across all conditions for session-level
            eca_all = [pc[c][band]['eca_asym'] for c in pc if band in pc[c]]
            te_all = [pc[c][band]['te_asym'] for c in pc if band in pc[c]]
            if eca_all and te_all:
                eca_vals.append(np.mean(eca_all))
                te_vals.append(np.mean(te_all))
                labels.append(s['session'])

        if not eca_vals:
            continue

        ax.scatter(eca_vals, te_vals, c=BAND_COLORS[band], s=60,
                   alpha=0.7, edgecolors='black', linewidths=0.5)

        for i, lab in enumerate(labels):
            short = lab.replace('_0', '').replace('_', '')[:6]
            ax.annotate(short, (eca_vals[i], te_vals[i]),
                        fontsize=6, ha='left', va='bottom',
                        xytext=(3, 3), textcoords='offset points')

        # Correlation
        if len(eca_vals) > 2:
            r = np.corrcoef(eca_vals, te_vals)[0, 1]
            ax.set_title(f'{band.upper()} (r={r:.2f})', fontsize=11,
                         fontweight='bold', color=BAND_COLORS[band])
        else:
            ax.set_title(f'{band.upper()}', fontsize=11,
                         fontweight='bold', color=BAND_COLORS[band])

        ax.axhline(0, color='gray', ls=':', lw=0.5)
        ax.axvline(0, color='gray', ls=':', lw=0.5)
        ax.set_xlabel('ECA Asymmetry', fontsize=9, fontweight='bold')
        if bi == 0:
            ax.set_ylabel('TE Asymmetry', fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)

    fig.suptitle('ECA vs TE Asymmetry (per session)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'eca_vs_te_scatter.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating directed burst coupling visualizations...")
    print()
    print("[1/4] Per-condition ECA asymmetry...")
    plot_condition_eca_bars()
    print("[2/4] Per-condition TE asymmetry...")
    plot_condition_te_bars()
    print("[3/4] Lag profile...")
    plot_lag_profile()
    print("[4/4] ECA vs TE scatter...")
    plot_eca_vs_te_scatter()
    print("\nDone.")
