"""Audit z-score clipping rates across all cached sessions.

Checks how many samples sit at exactly ±10.0 (the clip boundary) in
preprocessed features. Also reports tail distribution percentiles to
assess whether the [-10, 10] range is appropriate.

Usage:
    python scripts/audit_clipping.py
"""

import os
import sys
import glob
import json
import numpy as np
from collections import defaultdict

# Session cache location
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         '..', 'MCCT', 'session_cache')
# Fallback
if not os.path.isdir(CACHE_DIR):
    CACHE_DIR = 'C:/Users/optilab/desktop/MCCT/session_cache'

CLIP_VAL = 10.0
THRESHOLDS = [3, 5, 7, 10]

# Modalities to audit (keys in the .npz cache)
MODALITIES = {
    'eeg': {'channels': 14, 'desc': 'EEG (14ch)'},
    'ecg': {'channels': 1, 'desc': 'ECG (1ch)'},
    'blendshapes': {'channels': None, 'desc': 'Blendshapes'},
    'pose': {'channels': None, 'desc': 'Pose'},
}


def audit_session(npz_path):
    """Audit clipping for one session cache file.

    Returns dict: modality -> {
        'n_total': int,
        'n_clipped': int,  # samples at exactly ±CLIP_VAL
        'threshold_rates': {threshold: fraction},
        'percentiles': {99.9: val, 99.99: val, max: val},
        'participant': str,
    }
    """
    data = np.load(npz_path, allow_pickle=True)
    results = {}

    for participant in ['p1', 'p2']:
        for mod_key, mod_info in MODALITIES.items():
            feat_key = f'{participant}_{mod_key}'
            valid_key = f'{participant}_{mod_key}_valid'

            if feat_key not in data:
                continue

            features = data[feat_key]
            if features.ndim == 1:
                features = features[:, None]

            # Use validity mask if available
            if valid_key in data:
                valid = data[valid_key]
                if valid.ndim == 1 and features.ndim == 2:
                    # Broadcast per-sample validity to all channels
                    mask = valid.astype(bool)
                elif valid.ndim == 2:
                    mask = valid.astype(bool)
                else:
                    mask = np.ones(features.shape[0], dtype=bool)
            else:
                mask = np.ones(features.shape[0], dtype=bool)

            # Get valid samples
            if mask.ndim == 1:
                valid_feats = features[mask]
            else:
                valid_feats = features[mask]

            if valid_feats.size == 0:
                continue

            abs_vals = np.abs(valid_feats.ravel())
            n_total = abs_vals.size

            # Count samples at exactly ±CLIP_VAL (clipped)
            n_clipped = int(np.sum(np.abs(valid_feats.ravel() - CLIP_VAL) < 1e-6) +
                           np.sum(np.abs(valid_feats.ravel() + CLIP_VAL) < 1e-6))

            # Threshold exceedance rates
            threshold_rates = {}
            for t in THRESHOLDS:
                n_exceed = int(np.sum(abs_vals >= t))
                threshold_rates[t] = n_exceed / n_total if n_total > 0 else 0

            # Tail percentiles
            pcts = {}
            for p in [99.0, 99.9, 99.99]:
                pcts[p] = float(np.percentile(abs_vals, p))
            pcts['max'] = float(abs_vals.max())

            label = f'{participant}_{mod_key}'
            results[label] = {
                'n_total': n_total,
                'n_clipped': n_clipped,
                'clip_rate': n_clipped / n_total if n_total > 0 else 0,
                'threshold_rates': threshold_rates,
                'percentiles': pcts,
                'n_channels': features.shape[1] if features.ndim == 2 else 1,
            }

    return results


def main():
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, '*.npz')))
    if not npz_files:
        print(f"No .npz files found in {CACHE_DIR}")
        sys.exit(1)

    print(f"Auditing {len(npz_files)} session cache files in {CACHE_DIR}\n")
    print(f"Clip boundary: ±{CLIP_VAL}")
    print("=" * 90)

    # Aggregate per modality
    mod_agg = defaultdict(lambda: {
        'total_samples': 0,
        'total_clipped': 0,
        'session_clip_rates': [],
        'all_percentiles': defaultdict(list),
        'threshold_totals': defaultdict(lambda: {'exceed': 0, 'total': 0}),
    })

    for npz_path in npz_files:
        fname = os.path.basename(npz_path)
        # Extract session name from filename
        session_name = fname.replace('.npz', '')
        # Try to get human-readable name from JSON
        json_path = npz_path.replace('.npz', '.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                meta = json.load(f)
            session_name = meta.get('session_name', session_name)

        results = audit_session(npz_path)

        for label, info in results.items():
            # Extract modality (strip p1_/p2_ prefix)
            mod = label.split('_', 1)[1]

            agg = mod_agg[mod]
            agg['total_samples'] += info['n_total']
            agg['total_clipped'] += info['n_clipped']
            agg['session_clip_rates'].append(info['clip_rate'])

            for p, v in info['percentiles'].items():
                agg['all_percentiles'][p].append(v)

            for t, rate in info['threshold_rates'].items():
                agg['threshold_totals'][t]['exceed'] += int(rate * info['n_total'])
                agg['threshold_totals'][t]['total'] += info['n_total']

    # Print summary table
    print(f"\n{'Modality':<20} {'Samples':>12} {'Clipped':>10} {'Clip%':>8} "
          f"{'|z|>3':>8} {'|z|>5':>8} {'|z|>7':>8} {'|z|>10':>8}")
    print("-" * 90)

    for mod in sorted(mod_agg.keys()):
        agg = mod_agg[mod]
        total = agg['total_samples']
        clipped = agg['total_clipped']
        clip_pct = 100 * clipped / total if total > 0 else 0

        cols = [f"{mod:<20}", f"{total:>12,}", f"{clipped:>10,}", f"{clip_pct:>7.4f}%"]
        for t in THRESHOLDS:
            tt = agg['threshold_totals'][t]
            rate = 100 * tt['exceed'] / tt['total'] if tt['total'] > 0 else 0
            cols.append(f"{rate:>7.4f}%")
        print(" ".join(cols))

    # Percentile summary
    print(f"\n{'Modality':<20} {'P99 |z|':>10} {'P99.9 |z|':>10} {'P99.99 |z|':>12} {'Max |z|':>10}")
    print("-" * 70)

    for mod in sorted(mod_agg.keys()):
        agg = mod_agg[mod]
        pcts = agg['all_percentiles']
        row = [f"{mod:<20}"]
        for p in [99.0, 99.9, 99.99]:
            vals = pcts.get(p, [0])
            row.append(f"{np.mean(vals):>10.3f}")
        vals = pcts.get('max', [0])
        row.append(f"{np.mean(vals):>10.3f}")
        print(" ".join(row))

    # Per-session detail for modalities with non-trivial clipping
    print("\n" + "=" * 90)
    print("Per-session clip rates for modalities with >0.01% clipping:")
    print("-" * 90)

    for mod in sorted(mod_agg.keys()):
        agg = mod_agg[mod]
        total = agg['total_samples']
        clip_pct = 100 * agg['total_clipped'] / total if total > 0 else 0
        if clip_pct > 0.01:
            rates = agg['session_clip_rates']
            rates_pct = [r * 100 for r in rates]
            print(f"\n  {mod}:")
            print(f"    Mean: {np.mean(rates_pct):.4f}%  "
                  f"Median: {np.median(rates_pct):.4f}%  "
                  f"Max: {np.max(rates_pct):.4f}%  "
                  f"Min: {np.min(rates_pct):.4f}%")
            print(f"    Sessions > 0.1%: {sum(1 for r in rates_pct if r > 0.1)}/{len(rates_pct)}")

    print("\nDone.")


if __name__ == '__main__':
    main()
