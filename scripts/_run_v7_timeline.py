"""V7 full-session timeline: CWT band-power + wavelet coherence spectrogram.

Panels:
  1-3. EEG per-band z(t) (same as V6)
  4.   BL expression-band power: patient (up) / therapist (down) — continuous
  5.   BL speech-band power: patient (up) / therapist (down) — continuous
  6.   Wavelet coherence spectrogram (affect AUs) — time-frequency heatmap
  7.   Per-band coherence timecourses (state, expression, speech)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from scripts.run_session_v6 import (
    load_xdf_session, extract_bl_segment, extract_eeg_segment, DEFAULT_SEGMENTS,
)
from cadence.significance.bl_wavelet import (
    compute_au_cwt, detect_speech, wavelet_coherence, coherence_band_summary,
    AFFECT_AUS, SMILE_AUS, BAND_EXPRESSION, BAND_SPEECH, BAND_STATE,
)
from cadence.significance.fast_cycles import eeg_coupling_timecourse
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache

FS_BL = 30.0

CONDITION_COLORS = {
    'base_EO': '#E3F2FD', 'base_EC': '#E8EAF6',
    'conv_1': '#FFF3E0', 'conv_2': '#FFF3E0',
    'meditate_B': '#F3E5F5', 'meditate_K': '#E8F5E9',
}
BAND_COLORS = {'theta': '#2196F3', 'alpha': '#4CAF50', 'beta': '#FF9800'}
CONDITION_ORDER = ['base_EO', 'base_EC', 'conv_1', 'meditate_B', 'meditate_K', 'conv_2']

# ── Load session ──────────────────────────────────────────────────────

xdf_path = glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0]
print("Loading XDF...", flush=True)
session_data = load_xdf_session(xdf_path)
markers = session_data['markers']
p1_role = session_data['p1_role']
p2_role = session_data['p2_role']

config = load_config()
cached_sessions = discover_cached_sessions(config['session_cache'])
cache_path = [p for n, p in cached_sessions if 'y_06' in n][0]
cached = load_session_from_cache(cache_path, config)
lsl_ts = session_data['landmarks']['P1'][0]
lsl_offset = float(lsl_ts[0]) - float(cached.get('p1_blendshapes_ts', lsl_ts)[0])
session_data['cached'] = cached
session_data['lsl_offset'] = lsl_offset

# Collect segments
segments = []
for seg_name in CONDITION_ORDER:
    t_start = markers.get(f'{seg_name}_start')
    t_end = markers.get(f'{seg_name}_stop')
    if t_start is not None and t_end is not None:
        segments.append((seg_name, t_start, t_end))

session_start = min(t for _, t, _ in segments) - 30
session_end = max(t for _, _, t in segments) + 30

# ── Run analysis per segment ─────────────────────────────────────────

eeg_tl_data = {}
bl_wavelet_data = {}

for seg_name, t_start_lsl, t_end_lsl in segments:
    print(f"  {seg_name}...", flush=True)

    # BL wavelet
    p1_bl, p2_bl, bl_dur = extract_bl_segment(
        session_data['landmarks'], t_start_lsl, t_end_lsl)
    if p1_bl is not None:
        t0 = time.time()
        scal_p1 = compute_au_cwt(p1_bl)
        scal_p2 = compute_au_cwt(p2_bl)
        coh = wavelet_coherence(scal_p1, scal_p2)
        summary = coherence_band_summary(coh, scal_p1.freqs)
        elapsed = time.time() - t0

        T = min(scal_p1.expression_power.shape[0], scal_p2.expression_power.shape[0])
        t_arr = np.linspace(t_start_lsl, t_start_lsl + bl_dur, T)

        # Expression-band power on affect AUs
        expr_p1 = sum(scal_p1.expression_power[:T, au] for au in AFFECT_AUS)
        expr_p2 = sum(scal_p2.expression_power[:T, au] for au in AFFECT_AUS)

        # Speech-band power on speech AUs
        speech_p1 = scal_p1.speech_power[:T, 34] + scal_p1.speech_power[:T, 35]
        speech_p2 = scal_p2.speech_power[:T, 34] + scal_p2.speech_power[:T, 35]

        bl_wavelet_data[seg_name] = {
            't': t_arr, 'freqs': scal_p1.freqs,
            'expr_p1': expr_p1, 'expr_p2': expr_p2,
            'speech_p1': speech_p1, 'speech_p2': speech_p2,
            'coh_affect': coh['affect']['coherence'][:, :T],
            'summary': summary, 'elapsed': elapsed,
        }
        ec = summary['affect']['expression']['mean']
        print(f"    BL wavelet: expr_coh={ec:.3f} ({elapsed:.1f}s)", flush=True)

    # EEG
    p1_eeg, p2_eeg, eeg_dur, fs_eeg = extract_eeg_segment(
        cached, markers, seg_name, lsl_offset=lsl_offset)
    if p1_eeg is not None:
        t0 = time.time()
        tl = eeg_coupling_timecourse(p1_eeg, p2_eeg, fs_eeg,
                                      smooth_samples=3, n_surrogates=100, seed=42)
        elapsed = time.time() - t0
        tl['times_lsl'] = tl['times'] + t_start_lsl
        eeg_tl_data[seg_name] = tl
        mz = tl['combined']['mean_z']
        cf = tl['combined']['coupling_fraction']
        print(f"    EEG: z={mz:+.1f}, coupling={cf:.0%} ({elapsed:.1f}s)", flush=True)

# ── Plot ──────────────────────────────────────────────────────────────

print("\nGenerating plot...", flush=True)

fig, axes = plt.subplots(7, 1, figsize=(24, 22),
                         gridspec_kw={'height_ratios': [1, 1, 1, 1.2, 0.8, 1.5, 1]},
                         sharex=True)

ax_theta, ax_alpha, ax_beta, ax_expr, ax_speech, ax_coh_spec, ax_coh_bands = axes
eeg_axes = [('theta', ax_theta), ('alpha', ax_alpha), ('beta', ax_beta)]

# Condition backgrounds
for ax in axes:
    for seg_name, t0, t1 in segments:
        color = CONDITION_COLORS.get(seg_name, '#F5F5F5')
        ax.axvspan(t0, t1, alpha=0.3, color=color, zorder=0)
        ax.axvline(t0, color='gray', linewidth=0.5, alpha=0.3)

# Segment labels
for seg_name, t0, t1 in segments:
    ax_theta.text((t0 + t1) / 2, 1.08, seg_name.replace('_', ' '),
                  ha='center', va='bottom', fontsize=9, fontweight='bold',
                  transform=ax_theta.get_xaxis_transform())

# ── EEG panels (same as V6) ──────────────────────────────────────────

for band_name, ax in eeg_axes:
    ax.axhline(0, color='black', linewidth=0.3, alpha=0.3)
    ax.axhline(2, color='gray', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.axhline(-2, color='gray', linewidth=0.5, linestyle='--', alpha=0.4)
    for seg_name in [s[0] for s in segments]:
        tl = eeg_tl_data.get(seg_name)
        if tl is None:
            continue
        band_data = tl['per_band'].get(band_name)
        if band_data is None or band_data.get('n_valid', 0) == 0:
            continue
        t_lsl = tl['times_lsl']
        z = band_data['z']
        mask = band_data['mask']
        ax.plot(t_lsl, z, color=BAND_COLORS[band_name], linewidth=0.8, alpha=0.7, zorder=2)
        sig_regions = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        starts = np.where(sig_regions == 1)[0]
        ends = np.where(sig_regions == -1)[0]
        for s, e in zip(starts, ends):
            if s < len(t_lsl) and e <= len(t_lsl):
                ax.axvspan(t_lsl[max(0, s)], t_lsl[min(e-1, len(t_lsl)-1)],
                           alpha=0.25, color=BAND_COLORS[band_name], zorder=1)
    ax.set_ylabel(f'{band_name}\nz-score', fontsize=9)
    ax.set_ylim(-6, 12)

# ── BL expression-band power (patient up, therapist down) ────────────

for seg_name, t0, t1 in segments:
    bw = bl_wavelet_data.get(seg_name)
    if bw is None:
        continue
    t = bw['t']
    ax_expr.fill_between(t, 0, bw['expr_p1'], alpha=0.6, color='#E91E63',
                         linewidth=0, zorder=2)
    ax_expr.fill_between(t, 0, -bw['expr_p2'], alpha=0.6, color='#2196F3',
                         linewidth=0, zorder=2)

ax_expr.axhline(0, color='black', linewidth=0.5)
ax_expr.set_ylabel(f'Expression\n{p1_role} (+) / {p2_role} (-)', fontsize=9)
# Auto-scale symmetrically
expr_max = max(abs(ax_expr.get_ylim()[0]), abs(ax_expr.get_ylim()[1]))
ax_expr.set_ylim(-expr_max, expr_max)

# ── BL speech-band power ─────────────────────────────────────────────

for seg_name, t0, t1 in segments:
    bw = bl_wavelet_data.get(seg_name)
    if bw is None:
        continue
    t = bw['t']
    ax_speech.fill_between(t, 0, bw['speech_p1'], alpha=0.6, color='#E91E63',
                           linewidth=0, zorder=2)
    ax_speech.fill_between(t, 0, -bw['speech_p2'], alpha=0.6, color='#2196F3',
                           linewidth=0, zorder=2)

ax_speech.axhline(0, color='black', linewidth=0.5)
ax_speech.set_ylabel(f'Speech\n{p1_role} (+) / {p2_role} (-)', fontsize=9)
speech_max = max(abs(ax_speech.get_ylim()[0]), abs(ax_speech.get_ylim()[1]))
ax_speech.set_ylim(-speech_max, speech_max)

# ── Coherence spectrogram (affect AUs) ───────────────────────────────

for seg_name, t0, t1 in segments:
    bw = bl_wavelet_data.get(seg_name)
    if bw is None:
        continue
    t = bw['t']
    freqs = bw['freqs']
    coh = bw['coh_affect']  # (n_freqs, T)

    ax_coh_spec.pcolormesh(t, freqs, coh, shading='auto',
                           cmap='hot', vmin=0, vmax=0.8, zorder=2)

ax_coh_spec.set_yscale('log')
ax_coh_spec.set_ylim(0.3, 8)
ax_coh_spec.set_ylabel('Affect coherence\nFreq (Hz)', fontsize=9)
# Band boundary lines
for f in [0.5, 2.0]:
    ax_coh_spec.axhline(f, color='white', linewidth=0.5, linestyle='--', alpha=0.5)

# ── Per-band coherence timecourses ───────────────────────────────────

band_colors_coh = {'state': '#9C27B0', 'expression': '#E91E63', 'speech': '#FF9800'}

for seg_name, t0, t1 in segments:
    bw = bl_wavelet_data.get(seg_name)
    if bw is None:
        continue
    t = bw['t']
    s = bw['summary']
    for band_name, color in band_colors_coh.items():
        tc = s['affect'].get(band_name, {}).get('timecourse')
        if tc is not None and len(tc) == len(t):
            ax_coh_bands.plot(t, tc, color=color, linewidth=0.7, alpha=0.7)

ax_coh_bands.set_ylabel('Band\ncoherence', fontsize=9)
ax_coh_bands.set_ylim(0, 1)
ax_coh_bands.set_xlabel('LSL time (s)')
ax_coh_bands.set_xlim(session_start, session_end)

# ── Legend ────────────────────────────────────────────────────────────

legend_elements = [
    Line2D([0], [0], color=BAND_COLORS['theta'], label='Theta (4-8 Hz)', linewidth=2),
    Line2D([0], [0], color=BAND_COLORS['alpha'], label='Alpha (8-13 Hz)', linewidth=2),
    Line2D([0], [0], color=BAND_COLORS['beta'], label='Beta (13-30 Hz)', linewidth=2),
    Patch(facecolor='#E91E63', alpha=0.6, label=f'{p1_role.capitalize()}'),
    Patch(facecolor='#2196F3', alpha=0.6, label=f'{p2_role.capitalize()}'),
    Line2D([0], [0], color=band_colors_coh['state'], label='State coh (<0.5 Hz)', linewidth=2),
    Line2D([0], [0], color=band_colors_coh['expression'], label='Expr coh (0.5-2 Hz)', linewidth=2),
    Line2D([0], [0], color=band_colors_coh['speech'], label='Speech coh (2-7 Hz)', linewidth=2),
]
for seg_name in ['conv_1', 'meditate_K', 'meditate_B', 'base_EO']:
    legend_elements.append(Patch(facecolor=CONDITION_COLORS.get(seg_name, 'white'),
                                  label=seg_name.replace('_', ' '), alpha=0.5))

fig.legend(handles=legend_elements, loc='upper right', fontsize=8,
           bbox_to_anchor=(0.99, 0.98), ncol=2)

fig.suptitle(f'y_06 V7: EEG Coupling + Facial CWT Band Power + Wavelet Coherence ({p1_role} vs {p2_role})',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_path = 'results/v6/y_06/v7_wavelet_timeline.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved {out_path}")

# Print summary
print(f"\n{'seg':>12} | {'EEG_z':>8} {'EEG_cf':>8} | {'expr_coh':>8} {'speech_coh':>10} {'state_coh':>9}")
print("-" * 65)
for seg_name, t0, t1 in segments:
    tl = eeg_tl_data.get(seg_name)
    bw = bl_wavelet_data.get(seg_name)
    eeg_z = tl['combined']['mean_z'] if tl else 0
    eeg_cf = tl['combined']['coupling_fraction'] if tl else 0
    ec = bw['summary']['affect']['expression']['mean'] if bw else 0
    sc = bw['summary']['affect']['speech']['mean'] if bw else 0
    stc = bw['summary']['affect']['state']['mean'] if bw else 0
    print(f"{seg_name:>12} | {eeg_z:+8.1f} {eeg_cf:7.0%} | {ec:8.3f} {sc:10.3f} {stc:9.3f}")
