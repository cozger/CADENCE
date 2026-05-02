"""Rate-conditional TE analysis: restrict TE episode detection to timepoints
where both participants have above-average alpha power (conc_alpha > 0).

Tests whether the TE condition differences survive after removing the
burst-rate detectability confound.
"""

import numpy as np
import os
from collections import defaultdict
from scipy.stats import wilcoxon, spearmanr

from scripts.run_condition_statistics import (
    discover_sessions, load_session, classify_protocol,
    get_condition_mask, FS_OUT, CONTRASTS
)

RESULTS_DIR = 'results/v11'
GATE_THRESH = 0.0  # conc_alpha > 0 means above session-average shared alpha


def main():
    sessions = discover_sessions(RESULTS_DIR)

    print("=== Rate-conditional TE: only count episodes when conc_alpha > 0 ===")
    print("(i.e., both participants have above-average alpha power)\n")

    rows = []
    for sess in sessions:
        npz, meta, rslds = load_session(sess)
        sess['protocol'] = classify_protocol(meta['segments'])
        t_common = npz['t_common']
        segments = meta['segments']
        conc_alpha = npz['z_raw_conc_alpha']
        te_asym = npz['u_te_asym_alpha']

        for seg in segments:
            cond_name = seg[0]
            mask = get_condition_mask(t_common, segments, cond_name)
            if mask is None or mask.sum() < 20:
                continue

            cond_te = te_asym[mask]
            cond_conc = conc_alpha[mask]
            n_total = len(cond_te)

            # Ungated
            tp_ungated = float(np.mean(cond_te > 2.0))
            pt_ungated = float(np.mean(cond_te < -2.0))
            any_ungated = tp_ungated + pt_ungated

            # Gated
            gate = cond_conc > GATE_THRESH
            n_gated = int(gate.sum())

            if n_gated >= 10:
                gated_te = cond_te[gate]
                tp_gated = float(np.mean(gated_te > 2.0))
                pt_gated = float(np.mean(gated_te < -2.0))
                any_gated = tp_gated + pt_gated
                gate_frac = n_gated / n_total
            else:
                tp_gated = pt_gated = any_gated = float('nan')
                gate_frac = 0.0

            rows.append({
                'session': sess['name'], 'protocol': sess['protocol'],
                'condition': cond_name,
                'tp_ungated': tp_ungated, 'pt_ungated': pt_ungated,
                'any_ungated': any_ungated,
                'tp_gated': tp_gated, 'pt_gated': pt_gated,
                'any_gated': any_gated,
                'gate_frac': gate_frac, 'n_total': n_total,
                'n_gated': n_gated,
                'mean_conc_alpha': float(np.mean(cond_conc)),
            })

    # ── Per-condition summary ────────────────────────────────────────
    cond_data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not np.isnan(r['any_gated']):
            for k in ['any_ungated', 'any_gated', 'tp_ungated', 'tp_gated',
                       'pt_ungated', 'pt_gated', 'gate_frac', 'mean_conc_alpha']:
                cond_data[r['condition']][k].append(r[k])

    header = (f"{'Condition':15s} | {'n':>3s} | {'gate%':>6s} | "
              f"{'raw_any':>8s} | {'gated_any':>10s} | "
              f"{'raw_T>P':>8s} | {'gated_T>P':>10s} | "
              f"{'raw_P>T':>8s} | {'gated_P>T':>10s}")
    print(header)
    print("-" * len(header))

    order = ['base_EO', 'base_EC', 'conv_1', 'meditate_B', 'meditate_K',
             'PE_1', 'PE_2', 'conv_2']
    for c in order:
        if c not in cond_data:
            continue
        d = cond_data[c]
        n = len(d['any_ungated'])
        gf = np.mean(d['gate_frac']) * 100
        au = np.mean(d['any_ungated']) * 100
        ag = np.mean(d['any_gated']) * 100
        tu = np.mean(d['tp_ungated']) * 100
        tg = np.mean(d['tp_gated']) * 100
        pu = np.mean(d['pt_ungated']) * 100
        pg = np.mean(d['pt_gated']) * 100
        print(f"{c:15s} | {n:3d} | {gf:5.1f}% | "
              f"{au:6.2f}% | {ag:8.2f}% | "
              f"{tu:6.2f}% | {tg:8.2f}% | "
              f"{pu:6.2f}% | {pg:8.2f}%")

    # ── Correlation: does gating remove the rate confound? ───────────
    print("\n=== Correlation with alpha power: ungated vs gated ===")
    valid = [r for r in rows if not np.isnan(r['any_gated'])]
    conc_all = [r['mean_conc_alpha'] for r in valid]
    ungated_all = [r['any_ungated'] for r in valid]
    gated_all = [r['any_gated'] for r in valid]

    rho_u, p_u = spearmanr(conc_all, ungated_all)
    rho_g, p_g = spearmanr(conc_all, gated_all)
    print(f"  Conc_alpha vs TE_any UNGATED: rho={rho_u:.3f}, p={p_u:.4f}")
    print(f"  Conc_alpha vs TE_any GATED:   rho={rho_g:.3f}, p={p_g:.4f}")

    # ── Condition group summary ──────────────────────────────────────
    print("\n=== Condition groups: ungated vs gated ===")
    groups = {
        'Baseline':     [r for r in valid if r['condition'] in ('base_EO', 'base_EC')],
        'Conversation': [r for r in valid if r['condition'] in ('conv_1', 'conv_2')],
        'Meditation':   [r for r in valid if r['condition'] in ('meditate_B', 'meditate_K')],
        'PE':           [r for r in valid if r['condition'] in ('PE_1', 'PE_2', 'PE')],
    }
    header2 = (f"{'Group':15s} | {'n':>3s} | {'gate%':>6s} | "
               f"{'raw_T>P':>8s} | {'gated_T>P':>10s} | "
               f"{'raw_P>T':>8s} | {'gated_P>T':>10s} | "
               f"{'raw_any':>8s} | {'gated_any':>10s}")
    print(header2)
    print("-" * len(header2))
    for label, subset in groups.items():
        if not subset:
            continue
        n = len(subset)
        gf = np.mean([r['gate_frac'] for r in subset]) * 100
        tu = np.mean([r['tp_ungated'] for r in subset]) * 100
        tg = np.mean([r['tp_gated'] for r in subset]) * 100
        pu = np.mean([r['pt_ungated'] for r in subset]) * 100
        pg = np.mean([r['pt_gated'] for r in subset]) * 100
        au = np.mean([r['any_ungated'] for r in subset]) * 100
        ag = np.mean([r['any_gated'] for r in subset]) * 100
        print(f"{label:15s} | {n:3d} | {gf:5.1f}% | "
              f"{tu:6.2f}% | {tg:8.2f}% | "
              f"{pu:6.2f}% | {pg:8.2f}% | "
              f"{au:6.2f}% | {ag:8.2f}%")

    # ── Paired tests: gated TE across contrasts ──────────────────────
    print("\n=== Paired Wilcoxon on gated TE (T>P alpha) ===")
    proto_map = {s['name']: s['protocol'] for s in sessions}
    row_map = {(r['session'], r['condition']): r for r in rows}

    for contrast_name, ca, cb, filt in CONTRASTS:
        pairs_a, pairs_b = [], []
        for sess in sessions:
            sn = sess['name']
            proto = proto_map.get(sn, 'unknown')
            if filt == 'meditation' and proto != 'meditation':
                continue
            if filt == 'pe' and proto != 'pe':
                continue

            ca_eff = ca
            if ca == 'PE_1' and (sn, 'PE') in row_map and (sn, 'PE_1') not in row_map:
                ca_eff = 'PE'

            ra = row_map.get((sn, ca_eff))
            rb = row_map.get((sn, cb))
            if (ra and rb
                    and not np.isnan(ra.get('tp_gated', float('nan')))
                    and not np.isnan(rb.get('tp_gated', float('nan')))):
                pairs_a.append(ra['tp_gated'])
                pairs_b.append(rb['tp_gated'])

        n = len(pairs_a)
        if n >= 6:
            a_arr = np.array(pairs_a)
            b_arr = np.array(pairs_b)
            d = a_arr - b_arr
            if np.any(d != 0):
                stat, p = wilcoxon(a_arr, b_arr)
                md = float(d.mean())
                print(f"  {contrast_name:30s}: n={n}, diff={md*100:+.2f}%, p={p:.4f}")
            else:
                print(f"  {contrast_name:30s}: n={n}, all diffs=0")
        elif n > 0:
            md = float((np.array(pairs_a) - np.array(pairs_b)).mean())
            print(f"  {contrast_name:30s}: n={n} (too few), diff={md*100:+.2f}%")
        else:
            print(f"  {contrast_name:30s}: n=0")

    # ── Same for P>T ─────────────────────────────────────────────────
    print("\n=== Paired Wilcoxon on gated TE (P>T alpha) ===")
    for contrast_name, ca, cb, filt in CONTRASTS:
        pairs_a, pairs_b = [], []
        for sess in sessions:
            sn = sess['name']
            proto = proto_map.get(sn, 'unknown')
            if filt == 'meditation' and proto != 'meditation':
                continue
            if filt == 'pe' and proto != 'pe':
                continue

            ca_eff = ca
            if ca == 'PE_1' and (sn, 'PE') in row_map and (sn, 'PE_1') not in row_map:
                ca_eff = 'PE'

            ra = row_map.get((sn, ca_eff))
            rb = row_map.get((sn, cb))
            if (ra and rb
                    and not np.isnan(ra.get('pt_gated', float('nan')))
                    and not np.isnan(rb.get('pt_gated', float('nan')))):
                pairs_a.append(ra['pt_gated'])
                pairs_b.append(rb['pt_gated'])

        n = len(pairs_a)
        if n >= 6:
            a_arr = np.array(pairs_a)
            b_arr = np.array(pairs_b)
            d = a_arr - b_arr
            if np.any(d != 0):
                stat, p = wilcoxon(a_arr, b_arr)
                md = float(d.mean())
                print(f"  {contrast_name:30s}: n={n}, diff={md*100:+.2f}%, p={p:.4f}")
            else:
                print(f"  {contrast_name:30s}: n={n}, all diffs=0")
        elif n > 0:
            md = float((np.array(pairs_a) - np.array(pairs_b)).mean())
            print(f"  {contrast_name:30s}: n={n} (too few), diff={md*100:+.2f}%")
        else:
            print(f"  {contrast_name:30s}: n=0")


if __name__ == '__main__':
    main()
