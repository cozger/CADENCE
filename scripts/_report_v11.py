"""V11 results report: hierarchical rSLDS + condition statistics."""
import json, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Hierarchical rSLDS ──────────────────────────────────────────────
r = json.load(open('results/v11/hierarchical/v11_hierarchical_results.json'))
sl = r['state_labels']

print("=" * 80)
print("  V11 HIERARCHICAL rSLDS RESULTS")
print("=" * 80)
print(f"  Sessions: {r['n_sessions']}")
print(f"  BIC: {r['bic']:.0f}")
print(f"  Final LL: {r['final_ll']:.0f}")
print(f"  States: {sl}")
print(f"  Config: K={r['config']['K']}, D_obs={r['config']['D_obs']}, "
      f"D_input={r['config']['D_input']}, D_latent={r['config']['D_latent']}")

usages = np.array([s['usage'] for s in r['sessions']])
print(f"\n  Mean state usage:")
for k in range(4):
    print(f"    {sl[k]:>6s}: {usages[:, k].mean():.1%} +/- {usages[:, k].std():.1%}")

print(f"\n  Per-session usage:")
for s in r['sessions']:
    usage = s['usage']
    u_str = '  '.join(f'{sl[k]}={usage[k]:.0%}' for k in range(4))
    print(f"    {s['session']:20s}: {u_str}  trans={s['n_transitions']}")

# Per-condition state usage aggregated
print(f"\n  Per-condition state usage (aggregated across sessions):")
cond_usage = {}
for s in r['sessions']:
    for pname, dur, usage in s.get('periods', []):
        if pname not in cond_usage:
            cond_usage[pname] = []
        cond_usage[pname].append(usage)

cond_order = ['base_EO', 'base_EC', 'conv_1', 'meditate_B', 'meditate_K',
              'PE_1', 'PE_2', 'PE', 'conv_2']
header = f"    {'Condition':15s} | {'n':>3s} | " + ' | '.join(f'{sl[k]:>6s}' for k in range(4))
print(header)
print(f"    {'-' * 55}")
for cond in cond_order:
    if cond not in cond_usage:
        continue
    arr = np.array(cond_usage[cond])
    n = len(arr)
    parts = [f"    {cond:15s} | {n:3d} |"]
    for k in range(4):
        parts.append(f" {arr[:, k].mean():5.1%} ")
    print(' |'.join(parts))

# ── Condition Statistics ────────────────────────────────────────────
cs_path = 'results/v11/condition_statistics/condition_statistics.json'
if os.path.exists(cs_path):
    cs = json.load(open(cs_path))
    results = cs['results']
    with_p = [r for r in results if r['p_uncorrected'] is not None]
    sig_005 = [r for r in with_p if r['p_uncorrected'] < 0.05]
    sig_fdr = [r for r in with_p if r.get('q_fdr') is not None and r['q_fdr'] < 0.05]

    print(f"\n{'=' * 80}")
    print(f"  CONDITION STATISTICS")
    print(f"{'=' * 80}")
    print(f"  Total tests: {len(results)}, with p-values: {len(with_p)}")
    print(f"  Significant uncorrected (p<0.05): {len(sig_005)}")
    print(f"  Significant FDR (q<0.05): {len(sig_fdr)}")

    print(f"\n  Top results (p<0.05):")
    sig_005.sort(key=lambda r: r['p_uncorrected'])
    for r in sig_005[:25]:
        q = r.get('q_fdr', 1.0)
        print(f"    {r['contrast']:30s} | {r['metric']:25s} | n={r['n_pairs']:2d} | "
              f"diff={r['mean_diff']:+.4f} | r={r['effect_size_r']:+.3f} | "
              f"p={r['p_uncorrected']:.4f} | q={q:.4f}")

    # TE-specific results
    te_results = [r for r in with_p if 'te_' in r['metric']]
    print(f"\n  TE-specific results ({len(te_results)} tests):")
    te_sig = [r for r in te_results if r['p_uncorrected'] < 0.10]
    te_sig.sort(key=lambda r: r['p_uncorrected'])
    if te_sig:
        for r in te_sig:
            q = r.get('q_fdr', 1.0)
            print(f"    {r['contrast']:30s} | {r['metric']:25s} | n={r['n_pairs']:2d} | "
                  f"diff={r['mean_diff']:+.4f} | p={r['p_uncorrected']:.4f} | q={q:.4f}")
    else:
        print(f"    No TE results at p<0.10")

    # Gate fractions per condition
    gate_results = [r for r in results if 'te_gate_frac' in r['metric'] and r['n_pairs'] > 0]
    if gate_results:
        print(f"\n  TE gate fractions (burst-rate gating at MIN_BURST_RATE=0.05):")
        for r in gate_results:
            if r['mean_a'] is not None:
                print(f"    {r['contrast']:30s} | {r['metric']:25s} | "
                      f"A={r['mean_a']:.1%}  B={r['mean_b']:.1%}")

print(f"\n{'=' * 80}")
print(f"  y_51 SESSION DETAILS")
print(f"{'=' * 80}")
for s in r['sessions']:
    if 'y_51' not in s['session']:
        continue
    usage = s['usage']
    u_str = '  '.join(f'{sl[k]}={usage[k]:.1%}' for k in range(4))
    print(f"  Usage: {u_str}")
    print(f"  Transitions: {s['n_transitions']}")
    print(f"\n  Per-condition:")
    for pname, dur, usage in s.get('periods', []):
        u_str = '  '.join(f'{sl[k]}={usage[k]:.1%}' for k in range(4))
        print(f"    {pname:15s} ({dur:.0f}s): {u_str}")
