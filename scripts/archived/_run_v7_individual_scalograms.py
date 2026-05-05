"""V7 individual CWT scalograms: per-participant affect AU power for conv_2 and meditations."""
import os, sys, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.run_session_v6 import load_xdf_session, extract_bl_segment
from cadence.significance.bl_wavelet import compute_au_cwt, AFFECT_AUS, SPEECH_AUS

raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'raw sessions')
xdf_files = glob.glob(os.path.join(raw_dir, '*y_06*.xdf'))
print(f"Loading {xdf_files[0]}...", flush=True)
session = load_xdf_session(xdf_files[0])
markers = session['markers']
FS = 30.0

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'v6', 'y_06', 'wavelet_diag')
os.makedirs(out_dir, exist_ok=True)

SEGMENTS = ['conv_2', 'meditate_K', 'meditate_B']

for seg in SEGMENTS:
    t0 = markers.get(f'{seg}_start')
    t1 = markers.get(f'{seg}_stop')
    if t0 is None:
        continue

    p1, p2, dur = extract_bl_segment(session['landmarks'], t0, t1)
    if p1 is None:
        continue

    t_start = time.time()
    scal_p1 = compute_au_cwt(p1)
    scal_p2 = compute_au_cwt(p2)
    elapsed = time.time() - t_start
    print(f"  {seg}: {dur:.0f}s, CWT {elapsed:.1f}s")

    freqs = scal_p1.freqs
    T = scal_p1.power.shape[1]
    t_axis = np.linspace(0, dur, T)

    # Sum power over AU groups
    groups = {
        'Affect (smile+frown+dimple+cheekSq+noseSneer)': AFFECT_AUS,
        'Speech (mouthLowerDown L/R)': [34, 35],
    }

    fig, axes = plt.subplots(len(groups) * 2, 1,
                             figsize=(22, 3.5 * len(groups) * 2),
                             sharex=True)

    row = 0
    for group_name, aus in groups.items():
        for person, scal, role in [('P1', scal_p1, 'patient'),
                                    ('P2', scal_p2, 'therapist')]:
            power_sum = sum(scal.power[:, :T, au] for au in aus)  # (n_freqs, T)

            ax = axes[row]
            vmax = np.percentile(power_sum, 98)
            ax.pcolormesh(t_axis, freqs, power_sum, shading='auto',
                          cmap='hot', vmin=0, vmax=max(vmax, 1e-8))
            ax.set_yscale('log')
            ax.set_ylim(0.3, 8)
            ax.set_ylabel(f'{person} ({role})\nFreq (Hz)', fontsize=8)

            # Band boundaries
            for f, ls in [(0.5, '--'), (2.0, '--')]:
                ax.axhline(f, color='cyan', linewidth=0.5, linestyle=ls, alpha=0.6)

            ax.text(0.01, 0.95, group_name, transform=ax.transAxes,
                    fontsize=8, color='white', va='top', fontweight='bold')
            row += 1

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'{seg}: Individual CWT scalograms (per participant, per AU group)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{seg}_individual_scalograms.png'), dpi=120)
    plt.close(fig)
    print(f"    Saved: {seg}_individual_scalograms.png")

print(f"\nAll outputs in: {out_dir}")
