"""Plots for the per-region wavelet semi-synthetic battery.

Reads results/regional_wavelet_semisynthetic/results.json and produces:
  1. Region-specificity heatmap per kappa (n_scenarios × n_regions AUC)
  2. Dose-response curves (target-region AUC vs kappa, one line per scenario)
  3. PC1 loading bars per region (if --collect-pca was used in the battery)

Usage:
    python scripts/_plot_regional_battery.py
    python scripts/_plot_regional_battery.py --in-dir results/regional_wavelet_semisynthetic
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SCEN_TARGET = {
    'R_mouth_smile':  ['mouth'],
    'R_mouth_frown':  ['mouth'],
    'R_brow_furrow':  ['brow'],
    'R_eye_squint':   ['eye'],
    'R_cheek_squint': ['cheek'],
    'R_nose_sneer':   ['nose'],
    'R_duchenne':     ['mouth', 'cheek'],
}


def build_matrix(results, kappa_str, region_order):
    """Return (scenarios, matrix) where matrix[i,j] is mean AUC."""
    scenarios = [s for s in results if kappa_str in results[s]]
    mat = np.full((len(scenarios), len(region_order)), np.nan, dtype=float)
    for i, s in enumerate(scenarios):
        for j, r in enumerate(region_order):
            entry = results[s][kappa_str].get(r)
            if isinstance(entry, dict) and 'auc_mean' in entry:
                mat[i, j] = entry['auc_mean']
    return scenarios, mat


def plot_specificity_heatmap(results, kappas, region_order, out_path):
    """One subplot per kappa: scenarios on Y, regions on X."""
    n = len(kappas)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n + 1.5, 4.0),
                              constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, k in zip(axes, kappas):
        ks = f'kappa_{k:.2f}'
        scenarios, mat = build_matrix(results, ks, region_order)
        if mat.size == 0:
            ax.set_visible(False)
            continue
        im = ax.imshow(mat, vmin=0.40, vmax=0.90, aspect='auto',
                        cmap='RdBu_r', origin='upper')
        ax.set_xticks(range(len(region_order)))
        ax.set_xticklabels(region_order, rotation=45, ha='right')
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels(scenarios)
        ax.set_title(f'kappa={k:.2f}')
        # Mark target cells
        for i, scen in enumerate(scenarios):
            tgts = SCEN_TARGET.get(scen, [])
            for t in tgts:
                if t in region_order:
                    j = region_order.index(t)
                    ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                                fill=False, ec='black', lw=1.5))
        # AUC text
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                             color='white' if abs(mat[i, j] - 0.65) > 0.15 else 'black',
                             fontsize=8)

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.8, label='AUC')
    cbar.ax.axhline(0.5, color='black', lw=0.8)
    fig.suptitle('Region-specificity: scenario × region AUC at each kappa',
                  fontsize=12)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f'  wrote {out_path}')


def plot_dose_response(results, kappas, region_order, out_path):
    """One subplot per scenario: target region AUC vs kappa,
    plus mean of non-target regions for comparison."""
    scenarios = list(results.keys())
    n = len(scenarios)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols + 1, 2.8 * rows + 1),
                              constrained_layout=True)
    axes = np.atleast_2d(axes).flatten()

    for idx, scen in enumerate(scenarios):
        ax = axes[idx]
        targets = SCEN_TARGET.get(scen, [])
        # Per-region traces
        for region in region_order:
            xs, ys, errs = [], [], []
            for k in kappas:
                ks = f'kappa_{k:.2f}'
                entry = results[scen].get(ks, {}).get(region)
                if isinstance(entry, dict) and 'auc_mean' in entry:
                    xs.append(k)
                    ys.append(entry['auc_mean'])
                    errs.append(entry.get('auc_std', 0.0) /
                                 max(np.sqrt(entry.get('n', 1)), 1))
            if not xs:
                continue
            is_target = region in targets
            ax.errorbar(xs, ys, yerr=errs,
                         marker='o', linewidth=2 if is_target else 1,
                         alpha=1.0 if is_target else 0.45,
                         label=region + (' (target)' if is_target else ''))
        ax.axhline(0.50, color='gray', linestyle='--', lw=0.8)
        ax.axhline(0.65, color='black', linestyle=':', lw=0.6)
        ax.set_xlabel('kappa')
        ax.set_ylabel('AUC')
        ax.set_ylim(0.40, 1.0)
        ax.set_title(scen)
        ax.legend(loc='lower right', fontsize=7)

    for k in range(n, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle('Dose-response per scenario × region', fontsize=12)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f'  wrote {out_path}')


def plot_pca_loadings(loadings_per_region, region_au_lists, out_path):
    """Bar plot of PC1 (and PC2) loadings per region.

    loadings_per_region: dict region_name -> {loadings: [[..], [..]],
                                              au_indices: [...]}
    region_au_lists:     dict from AU_REGIONS for AU labeling.
    """
    if not loadings_per_region:
        return
    regions = list(loadings_per_region.keys())
    fig, axes = plt.subplots(len(regions), 1,
                              figsize=(8, 1.8 * len(regions) + 1),
                              constrained_layout=True)
    if len(regions) == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        info = loadings_per_region[region]
        load = np.asarray(info['loadings'], dtype=float)
        aus = info['au_indices']
        if load.shape[0] == 0:
            ax.set_visible(False)
            continue
        x = np.arange(len(aus))
        width = 0.40
        ax.bar(x - width / 2, load[0], width=width, label='PC1', color='C0')
        if load.shape[0] >= 2:
            ax.bar(x + width / 2, load[1], width=width, label='PC2', color='C1')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f'AU{a}' for a in aus], rotation=45, ha='right',
                            fontsize=8)
        ax.set_title(f'{region} ({len(aus)} AUs)', fontsize=10)
        ax.legend(loc='upper right', fontsize=7)
        ax.set_ylabel('loading')

    fig.suptitle('Regional PC loadings (sample pair)', fontsize=12)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f'  wrote {out_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in-dir', default='results/regional_wavelet_semisynthetic')
    ap.add_argument('--out-dir', default=None,
                    help='Default: same as --in-dir')
    args = ap.parse_args()

    in_path = os.path.join(args.in_dir, 'results.json')
    if not os.path.exists(in_path):
        print(f'Input not found: {in_path}')
        return 1

    with open(in_path) as f:
        payload = json.load(f)

    results = payload['results']
    meta = payload['meta']
    kappas = meta['kappas']
    region_order = list(meta['regions'].keys())

    out_dir = args.out_dir or args.in_dir
    os.makedirs(out_dir, exist_ok=True)

    print('Plotting region-specificity heatmaps...')
    plot_specificity_heatmap(
        results, kappas, region_order,
        os.path.join(out_dir, 'specificity_heatmap.png'))

    print('Plotting dose-response curves...')
    plot_dose_response(
        results, kappas, region_order,
        os.path.join(out_dir, 'dose_response.png'))

    # PCA loadings if collected
    for scen, scen_data in results.items():
        if 'loadings_sample' in scen_data and scen_data['loadings_sample']:
            print(f'Plotting PCA loadings from {scen}...')
            plot_pca_loadings(
                scen_data['loadings_sample'], meta['regions'],
                os.path.join(out_dir, f'pca_loadings_{scen}.png'))
            break  # one set is enough; loadings are scenario-invariant on raw BL

    print('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
