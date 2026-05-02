"""Compare Y_45 burst results to cross-session group means."""
import json, numpy as np

d = json.load(open('results/rslds/burst_analysis/Y_45_03302026_bursts.json'))
a = json.load(open('results/rslds/burst_analysis/cross_session_burst_results.json'))

segs = ['base_EO', 'base_EC', 'conv_1', 'meditate_B', 'meditate_K', 'conv_2']
gs = ['EEG phase', 'EEG power', 'Face', 'Autonomic', 'Body']

print("Y_45 vs Group Mean — Burst Rates (Y_45 / group mean)")
print(f"{'Segment':>12s} |" + ''.join(f"{'':>3s}{g:>9s}" for g in gs))
print("-" * 75)
for s in segs:
    if s not in d['segments']:
        continue
    parts = []
    for g in gs:
        y45 = d['segments'][s]['burst_rates'].get(g, 0)
        grp = a['segments'].get(s, {}).get('rates', {}).get(g, {})
        gm = grp.get('mean', 0)
        parts.append(f"{y45:4.1f}/{gm:4.1f}")
    print(f"{s:>12s} |" + ''.join(f"{'':>1s}{p:>11s}" for p in parts))

print()
print("Y_45 EEG Power Burst Asymmetry (theta/alpha/beta, + = therapist):")
for s in segs:
    if s not in d['segments']:
        continue
    asym = d['segments'][s].get('burst_asym', {}).get('EEG power')
    grp_asym = a['segments'].get(s, {}).get('asym', {}).get('EEG power', {})
    gm = grp_asym.get('mean', [0, 0, 0])
    if asym:
        print(f"  {s:>12s}: Y45=[{asym[0]:+.2f},{asym[1]:+.2f},{asym[2]:+.2f}]  "
              f"group=[{gm[0]:+.2f},{gm[1]:+.2f},{gm[2]:+.2f}]")
    else:
        print(f"  {s:>12s}: insufficient bursts")

print()
print("Y_45 EEG Phase Burst Asymmetry (theta/alpha/beta):")
for s in segs:
    if s not in d['segments']:
        continue
    asym = d['segments'][s].get('burst_asym', {}).get('EEG phase')
    grp_asym = a['segments'].get(s, {}).get('asym', {}).get('EEG phase', {})
    gm = grp_asym.get('mean', [0, 0, 0])
    if asym:
        print(f"  {s:>12s}: Y45=[{asym[0]:+.2f},{asym[1]:+.2f},{asym[2]:+.2f}]  "
              f"group=[{gm[0]:+.2f},{gm[1]:+.2f},{gm[2]:+.2f}]")
    else:
        print(f"  {s:>12s}: insufficient bursts")
