"""Comprehensive validation of V7 wavelet pipeline against known ground truths.

Tests:
1. Speech detection: therapist speaks in meditation, patient silent
2. Speech in conversation: both active, turn-taking
3. Coherence at shared smile times vs non-smile times
4. Coherence in conversation vs meditation vs baseline (expected hierarchy)
5. Expression events: meditation should have far fewer than conversation
6. Band structure: speech coherence should be LOW (anti-sync turn-taking),
   expression coherence should be HIGHER in conversation
"""
import os, sys, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scripts.run_session_v6 import load_xdf_session, extract_bl_segment
from cadence.significance.bl_wavelet import (
    compute_au_cwt, detect_speech, detect_expression_events,
    wavelet_coherence, coherence_band_summary, AFFECT_AUS, SMILE_AUS
)
from cadence.significance.bl_coupling import facial_event_catalog

raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'raw sessions')
xdf_files = glob.glob(os.path.join(raw_dir, '*y_06*.xdf'))
print(f"Loading {xdf_files[0]}...", flush=True)
session = load_xdf_session(xdf_files[0])
markers = session['markers']
FS = 30.0

SEGMENTS = ['conv_1', 'conv_2', 'meditate_K', 'meditate_B', 'base_EO', 'base_EC']

# -- Compute CWT + coherence for all segments --------------------------

seg_data = {}
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
    speech_p1 = detect_speech(scal_p1)
    speech_p2 = detect_speech(scal_p2)
    events_p1 = detect_expression_events(scal_p1, speech_prob=speech_p1, lsl_start=t0)
    events_p2 = detect_expression_events(scal_p2, speech_prob=speech_p2, lsl_start=t0)
    coh = wavelet_coherence(scal_p1, scal_p2)
    summary = coherence_band_summary(coh, scal_p1.freqs)
    elapsed = time.time() - t_start

    # Also run V6 for shared smile times (phasic scoring)
    cat = facial_event_catalog(p1, p2, FS, lsl_start=t0, segment_name=seg)

    seg_data[seg] = {
        'scal_p1': scal_p1, 'scal_p2': scal_p2,
        'speech_p1': speech_p1, 'speech_p2': speech_p2,
        'events_p1': events_p1, 'events_p2': events_p2,
        'coh': coh, 'summary': summary, 'cat': cat,
        'dur': dur, 't0': t0, 'elapsed': elapsed,
    }
    print(f"  {seg}: {dur:.0f}s, {elapsed:.2f}s")

# ======================================================================
print("\n" + "="*70)
print("TEST 1: Speech detection — meditation ground truth")
print("  Patient should be SILENT, therapist should speak intermittently")
print("="*70)

for seg in ['meditate_K', 'meditate_B', 'conv_1', 'conv_2', 'base_EO', 'base_EC']:
    sd = seg_data.get(seg)
    if sd is None:
        continue
    sp1 = sd['speech_p1']
    sp2 = sd['speech_p2']
    print(f"  {seg:>12s}: P1(patient) active={(sp1>0.5).mean()*100:5.1f}%  "
          f"P2(therapist) active={(sp2>0.5).mean()*100:5.1f}%  "
          f"ratio P2/P1={((sp2>0.5).mean()+1e-6)/((sp1>0.5).mean()+1e-6):6.1f}x")

# ======================================================================
print("\n" + "="*70)
print("TEST 2: Expression event counts — conversation vs meditation vs baseline")
print("  Expect: conv >> meditation >= baseline")
print("="*70)

for seg in SEGMENTS:
    sd = seg_data.get(seg)
    if sd is None:
        continue
    n_p1 = len(sd['events_p1'])
    n_p2 = len(sd['events_p2'])
    rate_p1 = n_p1 / sd['dur'] * 60
    rate_p2 = n_p2 / sd['dur'] * 60
    print(f"  {seg:>12s}: P1={n_p1:3d} ({rate_p1:4.1f}/min)  "
          f"P2={n_p2:3d} ({rate_p2:4.1f}/min)")

# ======================================================================
print("\n" + "="*70)
print("TEST 3: Coherence band structure across conditions")
print("  Expect: expression coherence: conv > med > base")
print("  Expect: speech coherence: low everywhere (anti-sync turn-taking)")
print("="*70)

print(f"\n  {'Segment':>12s} | {'Affect':^30s} | {'Speech':^30s}")
print(f"  {'':>12s} | {'state':>8s} {'expr':>8s} {'speech':>8s} | {'state':>8s} {'expr':>8s} {'speech':>8s}")
print(f"  {'-'*12}-+-{'-'*30}-+-{'-'*30}")

for seg in SEGMENTS:
    sd = seg_data.get(seg)
    if sd is None:
        continue
    s = sd['summary']
    print(f"  {seg:>12s} | "
          f"{s['affect']['state']['mean']:8.3f} "
          f"{s['affect']['expression']['mean']:8.3f} "
          f"{s['affect']['speech']['mean']:8.3f} | "
          f"{s['speech']['state']['mean']:8.3f} "
          f"{s['speech']['expression']['mean']:8.3f} "
          f"{s['speech']['speech']['mean']:8.3f}")

# ======================================================================
print("\n" + "="*70)
print("TEST 4: Coherence at shared smile times vs non-smile times")
print("  Expect: affect expression coherence HIGHER during shared smiles")
print("="*70)

for seg in ['conv_1', 'conv_2']:
    sd = seg_data.get(seg)
    if sd is None:
        continue
    cat = sd['cat']
    coh_affect = sd['coh']['affect']['coherence']  # (n_freqs, T)
    freqs = sd['scal_p1'].freqs

    # Expression-band coherence timecourse
    expr_mask = (freqs >= 0.5) & (freqs < 2.0)
    expr_coh_tc = coh_affect[expr_mask].mean(axis=0)  # (T,)

    # Sample coherence at shared smile times
    smile_cohs = []
    for ss in cat.shared_smiles:
        t_mid = (ss.event_a.time + ss.event_b.time) / 2
        idx = int(t_mid * FS)
        if 0 <= idx < len(expr_coh_tc):
            # Average over +-1s window
            s = max(0, idx - int(FS))
            e = min(len(expr_coh_tc), idx + int(FS))
            smile_cohs.append(float(expr_coh_tc[s:e].mean()))

    # Non-smile baseline: random samples
    rng = np.random.default_rng(42)
    n_random = max(len(smile_cohs) * 5, 50)
    random_idx = rng.integers(int(3*FS), len(expr_coh_tc) - int(3*FS), size=n_random)
    baseline_cohs = [float(expr_coh_tc[max(0,i-int(FS)):i+int(FS)].mean())
                     for i in random_idx]

    smile_mean = np.mean(smile_cohs) if smile_cohs else 0
    base_mean = np.mean(baseline_cohs)
    ratio = smile_mean / (base_mean + 1e-6)

    print(f"  {seg}: {len(cat.shared_smiles)} shared smiles")
    print(f"    Smile coherence:   {smile_mean:.3f}")
    print(f"    Baseline coherence: {base_mean:.3f}")
    print(f"    Ratio: {ratio:.2f}x  {'PASS (>1.0)' if ratio > 1.0 else 'FAIL'}")

# ======================================================================
print("\n" + "="*70)
print("TEST 5: Meditation null — shared smile count should be near zero")
print("="*70)

for seg in SEGMENTS:
    sd = seg_data.get(seg)
    if sd is None:
        continue
    cat = sd['cat']
    print(f"  {seg:>12s}: {cat.n_shared_smiles:2d} shared smiles (V6 phasic)")

# ======================================================================
print("\n" + "="*70)
print("TEST 6: Pseudo-pair coherence control")
print("  conv_2 P1 x meditate_K P2 should have LOWER coherence than conv_2 real pair")
print("="*70)

if 'conv_2' in seg_data and 'meditate_K' in seg_data:
    sd_conv = seg_data['conv_2']
    sd_med = seg_data['meditate_K']

    # Trim to same length for pseudo-pair
    T_min = min(sd_conv['scal_p1'].coeffs.shape[1],
                sd_med['scal_p2'].coeffs.shape[1])

    from cadence.significance.bl_wavelet import AUScalogram
    # Create trimmed scalograms for pseudo pair
    from copy import copy
    pseudo_p2 = copy(sd_med['scal_p2'])
    pseudo_p2_coeffs = sd_med['scal_p2'].coeffs[:, :T_min, :]
    real_p1_coeffs = sd_conv['scal_p1'].coeffs[:, :T_min, :]

    # Manual coherence for pseudo pair (affect AUs)
    from scipy.ndimage import gaussian_filter1d as gf1d
    sigma = 0.5 * FS
    cross_real = np.zeros((len(freqs), T_min), dtype=np.complex64)
    cross_pseudo = np.zeros_like(cross_real)
    auto1 = np.zeros((len(freqs), T_min))
    auto2_real = np.zeros_like(auto1)
    auto2_pseudo = np.zeros_like(auto1)

    for au in AFFECT_AUS:
        w1 = real_p1_coeffs[:, :, au]
        w2r = sd_conv['scal_p2'].coeffs[:, :T_min, au]
        w2p = pseudo_p2_coeffs[:, :, au]
        cross_real += w1 * np.conj(w2r)
        cross_pseudo += w1 * np.conj(w2p)
        auto1 += np.abs(w1)**2
        auto2_real += np.abs(w2r)**2
        auto2_pseudo += np.abs(w2p)**2

    cross_real_s = gf1d(cross_real, sigma=sigma, axis=1)
    cross_pseudo_s = gf1d(cross_pseudo, sigma=sigma, axis=1)
    auto1_s = gf1d(auto1, sigma=sigma, axis=1)
    auto2r_s = gf1d(auto2_real, sigma=sigma, axis=1)
    auto2p_s = gf1d(auto2_pseudo, sigma=sigma, axis=1)

    coh_real = (np.abs(cross_real_s)**2 / (auto1_s * auto2r_s + 1e-10))
    coh_pseudo = (np.abs(cross_pseudo_s)**2 / (auto1_s * auto2p_s + 1e-10))

    # Expression band
    expr_mask = (freqs >= 0.5) & (freqs < 2.0)
    real_expr = coh_real[expr_mask].mean()
    pseudo_expr = coh_pseudo[expr_mask].mean()

    print(f"  Real pair (conv_2):       affect expression coh = {real_expr:.3f}")
    print(f"  Pseudo pair (conv_P1 x med_P2): affect expression coh = {pseudo_expr:.3f}")
    print(f"  Ratio real/pseudo: {real_expr/(pseudo_expr+1e-6):.2f}x  "
          f"{'PASS (>1.0)' if real_expr > pseudo_expr else 'FAIL'}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
