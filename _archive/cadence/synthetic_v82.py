"""V8.2 Semi-Synthetic Injection Module.

Provides raw-level coupling injection for all 18D V8.2 features.
Each injection operates on the raw signal (256 Hz EEG, 30 Hz BL, IBI, EDR, pose)
so the full scaffold pipeline validates end-to-end.

Base = pseudo-dyad (P1 from session A, P2 from session B).
kappa=0 must produce AUC~0.50.

Reuses existing primitives from cadence.synthetic where possible.
"""

import os
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from scipy.interpolate import interp1d

from cadence.constants import (
    EPOC_DISTANCE, EEG_ROIS,
    V82_EEG_SCENARIOS, V82_BL_SCENARIOS,
    V82_ECG_SCENARIOS, V82_RESP_SCENARIOS, V82_POSE_SCENARIOS,
    V82_MULTIMODAL_PATTERNS, V82_KAPPA_RANGES, V82_EEG_AMP_FACTOR,
)
from cadence.synthetic import generate_coupling_gate


# =========================================================================
# Pseudo-dyad base builder
# =========================================================================

def build_v82_pseudo_dyad(cached_a, cached_b, t_start_lsl, t_end_lsl,
                          lsl_offset_a=0.0, lsl_offset_b=0.0):
    """Build a pseudo-dyad base from two real sessions (cross-session pairing).

    Takes P1 data from session A and P2 data from session B.
    At kappa=0, this guarantees no pre-existing coupling.

    Args:
        cached_a, cached_b: Cached session dicts (from load_session_from_cache).
        t_start_lsl, t_end_lsl: LSL time window to extract.
        lsl_offset_a, lsl_offset_b: LSL offsets for each session.

    Returns:
        dict with raw arrays: p1_eeg, p2_eeg, p1_ecg, p2_ecg, etc.
    """
    base = {
        'duration': t_end_lsl - t_start_lsl,
        't_start': t_start_lsl,
        't_end': t_end_lsl,
    }

    # EEG: (T, 14) at ~256 Hz
    for role, cached, offset in [('p1', cached_a, lsl_offset_a),
                                  ('p2', cached_b, lsl_offset_b)]:
        eeg_key = f'{role}_eeg'
        ts_key = f'{role}_eeg_ts'
        if eeg_key in cached and ts_key in cached:
            ts = cached[ts_key] + offset
            mask = (ts >= t_start_lsl) & (ts < t_end_lsl)
            base[eeg_key] = cached[eeg_key][mask].copy()
            base[ts_key] = ts[mask].copy()

        # ECG raw
        ecg_key = f'{role}_ecg'
        ecg_ts_key = f'{role}_ecg_ts'
        if ecg_key in cached and ecg_ts_key in cached:
            ts = cached[ecg_ts_key] + offset
            mask = (ts >= t_start_lsl) & (ts < t_end_lsl)
            base[ecg_key] = cached[ecg_key][mask].copy()
            base[ecg_ts_key] = ts[mask].copy()

        # Blendshapes raw
        bl_key = f'{role}_blendshapes'
        bl_ts_key = f'{role}_blendshapes_ts'
        if bl_key in cached and bl_ts_key in cached:
            ts = cached[bl_ts_key] + offset
            mask = (ts >= t_start_lsl) & (ts < t_end_lsl)
            base[bl_key] = cached[bl_key][mask].copy()
            base[bl_ts_key] = ts[mask].copy()

        # Pose features
        pose_key = f'{role}_pose_features'
        pose_ts_key = f'{role}_pose_features_ts'
        if pose_key in cached and pose_ts_key in cached:
            ts = cached[pose_ts_key] + offset
            mask = (ts >= t_start_lsl) & (ts < t_end_lsl)
            base[pose_key] = cached[pose_key][mask].copy()
            base[pose_ts_key] = ts[mask].copy()

        # Validity arrays
        for suffix in ['_valid']:
            for mod in ['blendshapes', 'pose_features']:
                vk = f'{role}_{mod}{suffix}'
                tk = f'{role}_{mod}_ts'
                if vk in cached and tk in cached:
                    ts_full = cached[tk] + offset
                    m = (ts_full >= t_start_lsl) & (ts_full < t_end_lsl)
                    base[vk] = cached[vk][m].copy() if len(cached[vk]) == len(ts_full) else None

    return base


# =========================================================================
# EEG injection: composes phase + envelope + asymmetry primitives
# =========================================================================

def inject_eeg_v82(p1_eeg, p2_eeg, kappa, scenario_name, gate, seed=42, fs=256.0):
    """Inject EEG coupling at raw 256 Hz level using scenario-specific config.

    Composes Kuramoto phase rotation, envelope co-modulation, and power
    asymmetry according to scenario weights.

    Args:
        p1_eeg, p2_eeg: (T, 14) raw EEG, z-scored per channel.
        kappa: coupling strength [0, 0.40].
        scenario_name: key into V82_EEG_SCENARIOS.
        gate: (T,) coupling gate in [0, 1].
        seed: random seed.
        fs: sampling rate (256 Hz).

    Returns:
        p1_out, p2_out: (T, 14) with injected coupling.
    """
    cfg = V82_EEG_SCENARIOS[scenario_name]
    band = cfg['band']
    channels = cfg['channels']
    decay_sigma = cfg['decay_sigma']
    lag_samp = max(1, int(cfg['lag_ms'] * fs / 1000))
    w_phase = cfg['phase_weight']
    w_env = cfg['envelope_weight']
    w_asym = cfg['asymmetry_weight']

    # Amplify kappa to compensate for pipeline dilution (ROI averaging,
    # ImCoh extraction, prewhitening). Ecological kappa=0.30 → raw kappa=1.50.
    kappa_raw = kappa * V82_EEG_AMP_FACTOR

    p1_out = p1_eeg.copy().astype(np.float64)
    p2_out = p2_eeg.copy().astype(np.float64)
    T, C = p1_eeg.shape

    # Spatial decay from center of target channels
    center_ch = channels[len(channels) // 2]
    dist = EPOC_DISTANCE[center_ch]
    spatial_weight = np.exp(-dist ** 2 / (2 * decay_sigma ** 2))
    spatial_weight[spatial_weight < 0.01] = 0.0

    # Only process channels with nonzero spatial weight
    sos = butter(4, [band[0], band[1]], btype='band', fs=fs, output='sos')

    for ch in range(C):
        if spatial_weight[ch] <= 0:
            continue
        k_ch = kappa_raw * spatial_weight[ch]

        p1_filt = sosfiltfilt(sos, p1_eeg[:, ch].astype(np.float64))
        p2_filt_orig = sosfiltfilt(sos, p2_eeg[:, ch].astype(np.float64))

        amp1 = np.abs(hilbert(p1_filt))  # P1 envelope (for concordance)

        # Start with original P2 narrowband
        p2_filt_new = p2_filt_orig.copy()

        # Primitive 1: Narrowband signal mixing → targets ImCoh
        # Signal mixing creates a shared component between P1 and P2 with
        # a consistent time lag. The cross-spectrum of the shared component
        # has a non-zero imaginary part that ImCoh detects.
        if w_phase > 0:
            alpha_p = np.clip(k_ch * w_phase * gate, 0, 0.95)  # cap at 0.95
            p1_lagged = np.roll(p1_filt, lag_samp)
            p1_lagged[:lag_samp] = 0
            noise_scale = np.sqrt(np.maximum(1 - alpha_p ** 2, 0.0))
            p2_filt_new = noise_scale * p2_filt_orig + alpha_p * p1_lagged

        # Primitive 2: Envelope concordance (no lag — shared state)
        if w_env > 0:
            p2_an = hilbert(p2_filt_new)
            amp2 = np.abs(p2_an)
            phi2 = np.angle(p2_an)
            amp1_norm = amp1 / max(amp1.mean(), 1e-10)
            alpha_e = k_ch * w_env * gate
            amp2_new = amp2 * (1 + alpha_e * (amp1_norm - 1))
            amp2_new = np.maximum(amp2_new, 0)
            p2_filt_new = amp2_new * np.cos(phi2)

        # Replace P2's narrowband: subtract original, add modified
        p2_out[:, ch] = p2_eeg[:, ch].astype(np.float64) - p2_filt_orig + p2_filt_new

        # Primitive 3: Power asymmetry (boost P1 narrowband only)
        if w_asym > 0:
            alpha_a = kappa_raw * w_asym * gate
            p1_filt_boosted = sosfiltfilt(sos, p1_eeg[:, ch].astype(np.float64))
            p1_filt_boosted = p1_filt_boosted * (1 + alpha_a)
            p1_out[:, ch] = (p1_eeg[:, ch].astype(np.float64)
                             - sosfiltfilt(sos, p1_eeg[:, ch].astype(np.float64))
                             + p1_filt_boosted)

    return p1_out.astype(np.float32), p2_out.astype(np.float32)


# =========================================================================
# BL face injection: template-based additive on raw blendshapes
# =========================================================================

def _load_bl_template(template_name, template_dir='results/v6/y_06/coupling_patterns'):
    """Load a BL expression template (.npy file)."""
    path = os.path.join(template_dir, f'{template_name}_template.npy')
    if os.path.exists(path):
        return np.load(path)
    # Fallback: use smile_delta_avg if available
    fallback = os.path.join(template_dir, 'smile_delta_avg.npy')
    if os.path.exists(fallback):
        return np.load(fallback)
    # Generate synthetic Gaussian pulse template (52 channels, 6s @ 30 Hz)
    T_tpl = 180  # 6s at 30 Hz
    t = np.arange(T_tpl) / 30.0
    onset = 2.0  # seconds
    sigma = 0.7  # seconds
    pulse = np.exp(-0.5 * ((t - onset) / sigma) ** 2)
    template = np.zeros((T_tpl, 52), dtype=np.float32)
    # Only populate smile AUs with the pulse
    for au in [44, 45]:
        if au < 52:
            template[:, au] = pulse * 0.4  # peak amplitude
    return template


def inject_bl_v82(p1_bl, p2_bl, kappa, scenario_name, gate, seed=42, fs=30.0,
                  template_dir='results/v6/y_06/coupling_patterns'):
    """Inject BL coupling at raw 52-channel blendshape level.

    Uses continuous narrowband signal mixing in the expression band (0.5-2 Hz)
    on target AUs. This creates sustained wavelet coherence that the CWT +
    surrogate z-scoring pipeline reliably detects.

    The template-based approach (discrete smile events) was too sparse:
    individual events are diluted by ~70% uncoupled time in the CWT.

    Args:
        p1_bl, p2_bl: (T, C) raw blendshapes in [0, 1] at 30 Hz.
        kappa: coupling strength [0, 0.40].
        scenario_name: key into V82_BL_SCENARIOS.
        gate: (T,) coupling gate in [0, 1].
        seed: random seed.
        fs: sampling rate (30 Hz).
        template_dir: directory containing .npy templates (unused but kept for API).

    Returns:
        p1_out, p2_out: (T, C) with injected coupling.
        n_events: int (0 — continuous injection, no discrete events).
    """
    cfg = V82_BL_SCENARIOS[scenario_name]
    au_subset = cfg['au_subset']
    lag_samp = int(cfg['lag_s'] * fs)

    T, C = p1_bl.shape
    p1_out = p1_bl.copy().astype(np.float64)
    p2_out = p2_bl.copy().astype(np.float64)

    # Injection strategy: EXPRESSION-BAND SIGNAL MIXING on AFFECT_AUS.
    # Template-based discrete events don't survive CWT coherence because:
    # (1) The lag separates templates in time, reducing instantaneous coherence
    # (2) Only 4/10 AFFECT_AUS have template energy, diluting cross-AU sum
    # Instead, mix P1's expression-band signal directly into P2 during gate-ON.
    # The real BL data has STRUCTURED expression-band content (natural smile
    # dynamics) that circular-shift surrogates WILL break, producing detectable z.
    from cadence.significance.bl_wavelet import AFFECT_AUS as _AFFECT_AUS

    rng = np.random.default_rng(seed)

    # Expression band: 0.5-2 Hz
    lo, hi = 0.5, 2.0
    nyq = fs / 2.0
    if hi >= nyq:
        hi = nyq * 0.95
    sos = butter(4, [lo / nyq, hi / nyq], btype='band', output='sos')

    n_events = 0  # not event-based — continuous mixing
    for au in _AFFECT_AUS:
        if au >= C:
            continue

        # Extract expression-band from P1 (structured: real smile dynamics)
        p1_expr = sosfiltfilt(sos, p1_bl[:, au].astype(np.float64))
        p2_expr_orig = sosfiltfilt(sos, p2_bl[:, au].astype(np.float64))

        # Lag P1's expression band by lag_s (short — within CWT resolution)
        p1_lagged = np.roll(p1_expr, lag_samp)
        p1_lagged[:lag_samp] = 0.0

        # Scale: strong enough to be detectable in CWT coherence.
        # With 10 AUs, need SNR > 0.5 per AU for coherence > noise.
        # P1's expression-band std is the natural coupling strength.
        p1_expr_std = max(p1_expr.std(), 1e-6)
        au_boost = 3.0 if au in au_subset else 1.5
        alpha = np.clip(kappa * au_boost * gate[:T], 0, 0.95)

        # Additive: preserve P2's original signal, ADD the shared component
        injection = alpha * p1_lagged * (p2_bl[:, au].std() / p1_expr_std)
        p2_out[:, au] += injection

    # Also inject into the ACTIVITY CHANNEL (last column) for bl_activity_conc.
    # The pipeline reads: activity_channel = min(n_ch_P1, n_ch_P2) - 1
    # Activity = RMS deviation from 30s trailing mean (computed by preprocessor).
    # To create correlated activity: add P1's activity fluctuation to P2.
    act_ch = C - 1  # last column
    if act_ch >= 52:  # only if activity channel exists
        p1_act = p1_bl[:, act_ch].astype(np.float64)
        p2_act = p2_bl[:, act_ch].astype(np.float64)

        # Expression-band filter on activity channel
        p1_act_filt = sosfiltfilt(sos, p1_act)
        p1_act_lagged = np.roll(p1_act_filt, lag_samp)
        p1_act_lagged[:lag_samp] = 0.0

        act_alpha = np.clip(kappa * 2.0 * gate[:T], 0, 0.95)
        p1_std = max(p1_act_filt.std(), 1e-6)
        p2_std = max(p2_act.std(), 1e-6)
        p2_out[:, act_ch] += act_alpha * p1_act_lagged * (p2_std / p1_std)

    return p1_out.astype(np.float32), p2_out.astype(np.float32), n_events


# =========================================================================
# ECG injection: IBI-level band-specific coupling
# =========================================================================

FS_IBI = 4.0  # IBI interpolation rate (matches scaffold pipeline)

def inject_ecg_v82(ibi1_uniform, ibi2_uniform, kappa, band, gate, lag_s=4.0,
                   seed=42):
    """Inject ECG coupling at the interpolated IBI level.

    Mixes P1's bandpass-filtered IBI into P2's IBI during coupling windows.
    Must be called AFTER R-peak detection and cubic spline interpolation,
    BEFORE the pipeline's bandpass + Hilbert + cross-product.

    Args:
        ibi1_uniform, ibi2_uniform: (T,) IBI at 4 Hz, in seconds.
        kappa: coupling strength [0, 0.60].
        band: 'lf' (0.04-0.15 Hz) or 'hf' (0.15-0.4 Hz).
        gate: (T,) coupling gate in [0, 1].
        lag_s: coupling lag in seconds.
        seed: random seed.

    Returns:
        ibi2_coupled: (T,) modified IBI.
    """
    T = len(ibi2_uniform)
    lag_samp = int(lag_s * FS_IBI)

    # Band limits
    if band == 'lf':
        lo, hi = 0.04, 0.15
    elif band == 'hf':
        lo, hi = 0.15, 0.4
    else:
        raise ValueError(f"Unknown ECG band: {band}")

    nyq = FS_IBI / 2.0
    if hi >= nyq:
        hi = nyq * 0.95
    sos = butter(4, [lo / nyq, hi / nyq], btype='band', output='sos')

    # Extract P1's band-limited IBI as coupling source
    try:
        source = sosfiltfilt(sos, ibi1_uniform.astype(np.float64))
    except ValueError:
        return ibi2_uniform.copy()

    # Lag the source
    source_lagged = np.roll(source, lag_samp)
    # Normalize to match P2's IBI scale
    src_std = max(source_lagged.std(), 1e-10)
    ibi2_std = max(ibi2_uniform.std(), 1e-10)
    source_norm = source_lagged / src_std * ibi2_std

    # Mix
    alpha = kappa * gate[:T]
    ibi2_coupled = np.sqrt(np.maximum(1.0 - alpha ** 2, 0.0)) * ibi2_uniform + alpha * source_norm

    # Clip to physiological range
    return np.clip(ibi2_coupled, 0.3, 2.0).astype(np.float64)


# =========================================================================
# Respiratory injection: Kuramoto phase rotation on fused EDR
# =========================================================================

def inject_resp_v82(fused1, fused2, t_edr, kappa, gate, lag_s=0.0, seed=42):
    """Inject respiratory coupling via Kuramoto phase rotation on fused EDR.

    Rotates P2's instantaneous respiratory phase toward P1's phase
    during coupling windows. Phase-only — preserves P2's amplitude.

    Args:
        fused1, fused2: (T,) fused EDR waveforms at 4 Hz.
        t_edr: (T,) timestamps for EDR.
        kappa: coupling strength [0, 0.90].
        gate: (T,) coupling gate in [0, 1].
        lag_s: phase coupling lag in seconds.
        seed: random seed.

    Returns:
        fused2_coupled: (T,) modified EDR.
    """
    fs_edr = 4.0
    T = len(fused2)
    lag_samp = int(lag_s * fs_edr)

    # Bandpass 0.1-0.5 Hz (respiratory range)
    nyq = fs_edr / 2.0
    sos = butter(4, [0.1 / nyq, 0.5 / nyq], btype='band', output='sos')

    try:
        sig1_bp = sosfiltfilt(sos, fused1.astype(np.float64))
        sig2_bp = sosfiltfilt(sos, fused2.astype(np.float64))
    except ValueError:
        return fused2.copy()

    # Hilbert transform for instantaneous phase/amplitude
    an1 = hilbert(sig1_bp)
    an2 = hilbert(sig2_bp)

    phi1 = np.angle(np.roll(an1, lag_samp))
    phi2 = np.angle(an2)
    amp2 = np.abs(an2)

    # Kuramoto phase rotation
    alpha = kappa * gate[:T]
    phi2_new = phi2 + alpha * np.sin(phi1 - phi2)

    # Reconstruct narrowband with new phase
    sig2_bp_new = amp2 * np.cos(phi2_new)

    # Replace narrowband in original
    fused2_coupled = fused2.astype(np.float64) - sig2_bp + sig2_bp_new

    return fused2_coupled.astype(np.float64)


# =========================================================================
# Pose injection: position-level mixing + event-triggered nods
# =========================================================================

def inject_pose_v82(p1_pose, p2_pose, kappa, scenario_name, gate, lag_s=3.0,
                    seed=42, fs=12.0):
    """Inject pose coupling at the VELOCITY level, then reconstruct positions.

    The pipeline measures velocity cross-product at 2 Hz:
      positions → interp to 2 Hz → diff (velocity) → z-score → cross_product_z

    Injecting at position level fails because first-differencing + downsampling
    destroys the position coupling. Instead:
      1. Compute velocities at 12 Hz for both participants
      2. Mix P1's velocity into P2's velocity (with lag and gate)
      3. Integrate (cumsum) back to positions
      4. The pipeline's diff recovers the coupled velocity

    Args:
        p1_pose, p2_pose: (T, C) pose features at 12 Hz.
        kappa: coupling strength [0, 0.40].
        scenario_name: key into V82_POSE_SCENARIOS.
        gate: (T,) coupling gate in [0, 1].
        lag_s: coupling lag in seconds.
        seed: random seed.
        fs: sampling rate (12 Hz).

    Returns:
        p2_coupled: (T, C) with injected coupling (position-level output).
    """
    cfg = V82_POSE_SCENARIOS[scenario_name]
    channels = cfg['channels']
    mode = cfg['mode']
    lag_samp = int(cfg['lag_s'] * fs)

    rng = np.random.default_rng(seed)
    T, C = p2_pose.shape
    p2_out = p2_pose.copy().astype(np.float64)

    # Amplification: pipeline downsamples 12→2 Hz, dilutes across 11 channels,
    # then 200 surrogate z-scoring. Need strong injection.
    POSE_AMP = 8.0

    for ch in channels:
        if ch >= C:
            continue

        p1_vel = np.diff(p1_pose[:, ch].astype(np.float64), prepend=0)
        p2_vel = np.diff(p2_pose[:, ch].astype(np.float64), prepend=0)

        p1_vel_lagged = np.roll(p1_vel, lag_samp)
        p1_vel_lagged[:lag_samp] = 0.0

        p1v_std = max(p1_vel_lagged.std(), 1e-8)
        p2v_std = max(p2_vel.std(), 1e-8)
        scale_factor = p2v_std / p1v_std

        if mode == 'continuous':
            alpha = np.clip(kappa * POSE_AMP * gate[:T], 0, 0.95)
            p2_vel_coupled = p2_vel + alpha * p1_vel_lagged * scale_factor

        elif mode == 'event':
            vel_std = max(np.abs(p1_vel).std(), 1e-8)
            peaks, _ = find_peaks(np.abs(p1_vel), prominence=1.5 * vel_std,
                                  distance=int(1.0 * fs))
            p2_vel_coupled = p2_vel.copy()
            half_w = int(0.5 * fs)

            for pk in peaks:
                if pk >= T or gate[min(pk, T - 1)] < 0.5:
                    continue
                if rng.random() > min(kappa * POSE_AMP, 0.95):
                    continue
                resp_center = pk + lag_samp
                if resp_center < half_w or resp_center + half_w >= T:
                    continue
                src_start = max(0, pk - half_w)
                src_end = min(T, pk + half_w + 1)
                dst_start = max(0, resp_center - half_w)
                dst_end = min(T, resp_center + half_w + 1)
                wf_len = min(src_end - src_start, dst_end - dst_start)
                if wf_len > 0:
                    p2_vel_coupled[dst_start:dst_start + wf_len] += (
                        kappa * POSE_AMP * p1_vel[src_start:src_start + wf_len] * scale_factor)
        else:
            continue

        # Integrate velocity back to position
        p2_out[:, ch] = p2_pose[0, ch] + np.cumsum(p2_vel_coupled)

    return p2_out.astype(np.float32)


# =========================================================================
# Gate generation helpers
# =========================================================================

def generate_v82_gate(n_samples, fs, scenario_cfg, seed=42):
    """Generate a coupling gate from a V82 scenario config."""
    gate_cfg = scenario_cfg.get('gate')
    if gate_cfg is None:
        # Full-on gate (for respiratory — entire segment)
        return np.ones(n_samples, dtype=np.float32)
    return generate_coupling_gate(n_samples, fs, gate_cfg, seed=seed)


def generate_meditation_gate(n_samples, fs, ramp_s=20.0, seed=42):
    """Generate a block gate for meditation (full ON with slow ramps)."""
    gate = np.ones(n_samples, dtype=np.float32)
    ramp_samp = int(ramp_s * fs)
    if ramp_samp > 0 and ramp_samp < n_samples // 2:
        # Cosine ramp on
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_samp) / ramp_samp))
        gate[:ramp_samp] = ramp.astype(np.float32)
        gate[-ramp_samp:] = ramp[::-1].astype(np.float32)
    return gate


def generate_multimodal_gates(n_samples, fs, pattern_name, seed=42):
    """Generate coordinated per-modality gates for a multimodal pattern.

    Returns dict: modality_key -> (T,) gate array.
    """
    pattern = V82_MULTIMODAL_PATTERNS[pattern_name]
    rng = np.random.default_rng(seed)
    gates = {}

    for mod_key in ['eeg', 'bl', 'ecg', 'resp', 'pose']:
        spec = pattern.get(mod_key)
        if spec is None:
            gates[mod_key] = np.zeros(n_samples, dtype=np.float32)
            continue

        scenario_name, _ = spec

        if mod_key == 'resp':
            # Respiratory: block gate for meditation
            gates[mod_key] = generate_meditation_gate(n_samples, fs, ramp_s=20.0,
                                                       seed=rng.integers(0, 2**31))
        else:
            # Look up the scenario config
            if mod_key == 'eeg':
                cfg = V82_EEG_SCENARIOS[scenario_name]
            elif mod_key == 'bl':
                cfg = V82_BL_SCENARIOS[scenario_name]
            elif mod_key == 'ecg':
                cfg = V82_ECG_SCENARIOS[scenario_name]
            elif mod_key == 'pose':
                cfg = V82_POSE_SCENARIOS[scenario_name]
            else:
                cfg = {}
            gates[mod_key] = generate_v82_gate(n_samples, fs, cfg,
                                                seed=rng.integers(0, 2**31))

    return gates


# =========================================================================
# Full injection orchestrator
# =========================================================================

def inject_all_v82(base, kappa_dict, scenario_dict, gates, seed=42):
    """Apply per-modality injections to a pseudo-dyad base.

    Args:
        base: dict from build_v82_pseudo_dyad.
        kappa_dict: {'eeg': float, 'bl': float, 'ecg': float, 'resp': float, 'pose': float}
        scenario_dict: {'eeg': 'E1_mutual_gaze', 'bl': 'B1_smile', ...} or None per modality.
        gates: {'eeg': (T,), ...} per-modality gates.
        seed: random seed.

    Returns:
        modified base dict (shallow copy with replaced arrays).
    """
    rng = np.random.default_rng(seed)
    out = dict(base)  # shallow copy

    # EEG
    if (kappa_dict.get('eeg', 0) > 0 and scenario_dict.get('eeg') and
            'p1_eeg' in out and 'p2_eeg' in out):
        n_eeg = min(len(out['p1_eeg']), len(out['p2_eeg']))
        gate_eeg = gates.get('eeg', np.ones(n_eeg, dtype=np.float32))[:n_eeg]
        p1_e, p2_e = inject_eeg_v82(
            out['p1_eeg'][:n_eeg], out['p2_eeg'][:n_eeg],
            kappa_dict['eeg'], scenario_dict['eeg'], gate_eeg,
            seed=rng.integers(0, 2**31))
        out['p1_eeg'] = p1_e
        out['p2_eeg'] = p2_e

    # BL
    if (kappa_dict.get('bl', 0) > 0 and scenario_dict.get('bl') and
            'p1_blendshapes' in out and 'p2_blendshapes' in out):
        n_bl = min(len(out['p1_blendshapes']), len(out['p2_blendshapes']))
        gate_bl = gates.get('bl', np.ones(n_bl, dtype=np.float32))[:n_bl]
        p1_b, p2_b, n_ev = inject_bl_v82(
            out['p1_blendshapes'][:n_bl], out['p2_blendshapes'][:n_bl],
            kappa_dict['bl'], scenario_dict['bl'], gate_bl,
            seed=rng.integers(0, 2**31))
        out['p1_blendshapes'] = p1_b
        out['p2_blendshapes'] = p2_b

    # ECG (needs IBI extraction first — defer to test battery)
    # Resp (needs EDR extraction first — defer to test battery)
    # Pose
    if (kappa_dict.get('pose', 0) > 0 and scenario_dict.get('pose') and
            'p1_pose_features' in out and 'p2_pose_features' in out):
        n_pose = min(len(out['p1_pose_features']), len(out['p2_pose_features']))
        gate_pose = gates.get('pose', np.ones(n_pose, dtype=np.float32))[:n_pose]
        p2_p = inject_pose_v82(
            out['p1_pose_features'][:n_pose], out['p2_pose_features'][:n_pose],
            kappa_dict['pose'], scenario_dict['pose'], gate_pose,
            seed=rng.integers(0, 2**31))
        out['p2_pose_features'] = p2_p

    return out


# =========================================================================
# AUC computation
# =========================================================================

def compute_auc_within_session(z_timecourse, gate_2hz):
    """Compute within-session AUC: gate-ON vs gate-OFF discrimination.

    This is the correct test: can the pipeline distinguish coupled windows
    from uncoupled windows within the same session?

    Args:
        z_timecourse: (T,) z-scored feature timecourse from coupled data.
        gate_2hz: (T,) coupling gate resampled to 2 Hz.

    Returns:
        auc: float in [0, 1]. 0.5 = no discrimination.
    """
    T = min(len(z_timecourse), len(gate_2hz))
    z = z_timecourse[:T]
    g = gate_2hz[:T]

    on_mask = g > 0.5
    off_mask = g < 0.2

    on_vals = z[on_mask]
    off_vals = z[off_mask]

    on_vals = on_vals[~np.isnan(on_vals)]
    off_vals = off_vals[~np.isnan(off_vals)]

    if len(on_vals) < 10 or len(off_vals) < 10:
        return 0.5

    n1 = len(on_vals)
    n0 = len(off_vals)
    scores = np.concatenate([on_vals, off_vals])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

    rank_sum = ranks[:n1].sum()
    u = rank_sum - n1 * (n1 + 1) / 2
    auc = u / (n1 * n0)

    return float(np.clip(auc, 0, 1))


def compute_peak_ratio(z_timecourse, gate_2hz, percentile=90):
    """Peak-based metric for transient coupling (BL events).

    Compares the percentile-th z-score during gate-ON vs gate-OFF.
    Transient but strong events should produce higher peaks during ON.

    Returns:
        ratio: p90_ON / p90_OFF. > 1.0 means coupling detected.
    """
    T = min(len(z_timecourse), len(gate_2hz))
    z = z_timecourse[:T]
    g = gate_2hz[:T]

    on_vals = z[g > 0.5]
    off_vals = z[g < 0.2]

    on_vals = on_vals[~np.isnan(on_vals)]
    off_vals = off_vals[~np.isnan(off_vals)]

    if len(on_vals) < 5 or len(off_vals) < 5:
        return 1.0

    p_on = float(np.percentile(on_vals, percentile))
    p_off = float(np.percentile(off_vals, percentile))

    if abs(p_off) < 0.01:
        return 1.0 + (p_on - p_off)
    return p_on / p_off


def compute_auc_cross_session(z_coupled, z_null):
    """Compute AUC between coupled and null session z-timecourses.

    Uses full timecourse comparison (no gating).

    Args:
        z_coupled: (T,) from coupled run.
        z_null: (T,) from null run.

    Returns:
        auc: float in [0, 1].
    """
    coupled_vals = z_coupled[~np.isnan(z_coupled)]
    null_vals = z_null[~np.isnan(z_null)]

    if len(coupled_vals) < 10 or len(null_vals) < 10:
        return 0.5

    n1 = len(coupled_vals)
    n0 = len(null_vals)
    scores = np.concatenate([coupled_vals, null_vals])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

    rank_sum = ranks[:n1].sum()
    u = rank_sum - n1 * (n1 + 1) / 2
    auc = u / (n1 * n0)

    return float(np.clip(auc, 0, 1))
