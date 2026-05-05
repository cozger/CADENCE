"""V10 Semi-Synthetic Injection Module.

Extends V8.2 injection with LZ complexity scenarios.
LZ coupling is injected by modulating the amplitude envelope of frontal EEG
in a coordinated way between participants, which changes the LZ complexity
concordance/asymmetry measured by the V10 pipeline.

Reuses all V8.2 injection infrastructure (DRY).
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from cadence.constants import (
    V10_LZ_SCENARIOS, V10_KAPPA_RANGES, FRONTAL_ROI,
    V82_EEG_SCENARIOS, V82_BL_SCENARIOS,
    V82_ECG_SCENARIOS, V82_RESP_SCENARIOS, V82_POSE_SCENARIOS,
    V82_MULTIMODAL_PATTERNS,
)
from cadence.synthetic_v82 import (
    build_v82_pseudo_dyad,
    inject_eeg_v82, inject_bl_v82, inject_ecg_v82,
    inject_resp_v82, inject_pose_v82,
    inject_all_v82,
    generate_v82_gate, generate_meditation_gate, generate_multimodal_gates,
    compute_auc_within_session, compute_auc_cross_session,
    compute_peak_ratio,
)
from cadence.synthetic import generate_coupling_gate


def inject_lz_v10(p1_eeg, p2_eeg, kappa, scenario_name, gate, seed=42, fs=256.0):
    """Inject shared amplitude modulation that changes LZ complexity.

    Mechanism: inject a shared structured envelope pattern into frontal EEG.
    This creates temporal regularity (lower LZ) when coupling is ON and
    shared between participants → detectable as LZ concordance.

    The injection uses periodic bursting (amplitude modulation at ~1 Hz)
    which reduces LZ complexity by introducing predictable structure.
    When kappa > 0, both participants receive correlated burst patterns.

    Args:
        p1_eeg, p2_eeg: (T, 14) raw EEG.
        kappa: coupling strength [0, 0.40].
        scenario_name: key into V10_LZ_SCENARIOS.
        gate: (T,) coupling gate in [0, 1].
        seed: random seed.
        fs: sampling rate (256 Hz).

    Returns:
        p1_out, p2_out: (T, 14) with LZ-modulated frontal channels.
    """
    cfg = V10_LZ_SCENARIOS[scenario_name]
    band = cfg['band']
    channels = cfg['channels']

    rng = np.random.default_rng(seed)
    T, C = p1_eeg.shape
    p1_out = p1_eeg.copy().astype(np.float64)
    p2_out = p2_eeg.copy().astype(np.float64)

    if kappa <= 0:
        return p1_out, p2_out

    # Generate shared amplitude modulation pattern
    # Periodic bursting at ~1 Hz creates predictable temporal structure
    # that reduces LZ complexity for both participants when coupled
    t = np.arange(T) / fs
    burst_freq = 1.0 + rng.uniform(-0.2, 0.2)  # ~1 Hz ± jitter
    shared_envelope = 0.5 * (1 + np.cos(2 * np.pi * burst_freq * t))

    # Add slower modulation for more realistic structure
    slow_mod = 0.5 * (1 + np.cos(2 * np.pi * 0.15 * t + rng.uniform(0, 2 * np.pi)))
    shared_envelope *= slow_mod

    # Independent noise for each participant (decorrelates when kappa=0)
    noise_p1 = rng.standard_normal(T)
    noise_p2 = rng.standard_normal(T)

    # Mix shared + independent: higher kappa → more shared structure
    env_p1 = kappa * shared_envelope + (1 - kappa) * np.abs(noise_p1)
    env_p2 = kappa * shared_envelope + (1 - kappa) * np.abs(noise_p2)

    # Bandpass filter to target band
    sos = butter(4, [band[0], band[1]], btype='band', fs=fs, output='sos')

    for ch in channels:
        if ch >= C:
            continue

        # Apply amplitude modulation to the narrowband signal
        p1_filt = sosfiltfilt(sos, p1_eeg[:, ch].astype(np.float64))
        p2_filt = sosfiltfilt(sos, p2_eeg[:, ch].astype(np.float64))

        # Scale modulation by gate (only inject during coupled windows)
        mod_p1 = gate * env_p1
        mod_p2 = gate * env_p2

        # Modulate: multiply narrowband by (1 + kappa * modulation)
        # This changes the temporal structure of the amplitude envelope
        p1_out[:, ch] += p1_filt * mod_p1 * kappa * 3.0
        p2_out[:, ch] += p2_filt * mod_p2 * kappa * 3.0

    return p1_out, p2_out


def inject_all_v10(base, kappa_dict, scenario_dict, gates, seed=42):
    """Apply per-modality injections including V10 LZ scenarios.

    Extends inject_all_v82 with LZ injection.

    Args:
        base: dict from build_v82_pseudo_dyad.
        kappa_dict: {'eeg': float, 'bl': float, ..., 'lz': float}
        scenario_dict: {'eeg': 'E1_...', 'lz': 'L1_...', ...}
        gates: {'eeg': (T,), 'lz': (T,), ...} per-modality gates.
        seed: random seed.

    Returns:
        injected: dict with modified raw arrays.
    """
    # Apply V8.2 injections first
    v82_kappa = {k: v for k, v in kappa_dict.items() if k != 'lz'}
    v82_scenario = {k: v for k, v in scenario_dict.items() if k != 'lz'}
    v82_gates = {k: v for k, v in gates.items() if k != 'lz'}

    injected = inject_all_v82(base, v82_kappa, v82_scenario, v82_gates, seed=seed)

    # Apply LZ injection on top
    lz_kappa = kappa_dict.get('lz', 0.0)
    lz_scenario = scenario_dict.get('lz')
    lz_gate = gates.get('lz')

    if lz_kappa > 0 and lz_scenario and lz_gate is not None:
        if 'p1_eeg' in injected and 'p2_eeg' in injected:
            fs_eeg = 256.0
            if 'p1_eeg_ts' in injected and len(injected['p1_eeg_ts']) > 10:
                fs_eeg = 1.0 / np.median(np.diff(injected['p1_eeg_ts'][:1000]))

            # Resample gate to EEG rate
            n_eeg = len(injected['p1_eeg'])
            gate_eeg = np.interp(
                np.linspace(0, 1, n_eeg),
                np.linspace(0, 1, len(lz_gate)),
                lz_gate)

            injected['p1_eeg'], injected['p2_eeg'] = inject_lz_v10(
                injected['p1_eeg'], injected['p2_eeg'],
                lz_kappa, lz_scenario, gate_eeg, seed=seed + 999, fs=fs_eeg)

    return injected
