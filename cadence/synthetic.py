"""Synthetic session generation: Lorenz dynamics, coupling gates, session builder.

Self-contained -- only depends on numpy/scipy, plus cadence.data.preprocessors
for activity channel computation.
"""

import numpy as np

from cadence.constants import (
    SYNTH_MODALITY_CONFIG, COUPLING_PROFILES,
    SYNTH_MODALITY_CONFIG_V2, COUPLING_PROFILES_V2,
)


# =========================================================================
# Lorenz dynamical system
# =========================================================================

def integrate_lorenz(duration, output_hz=2.0, dt=0.005, transient=50.0, seed=42):
    """Integrate autonomous Lorenz system, return states at output_hz.

    Uses 4th-order Runge-Kutta with standard chaotic parameters
    (sigma=10, rho=28, beta=8/3).

    Args:
        duration: Session length in seconds.
        output_hz: Resampling rate for output (default 2 Hz).
        dt: Integration timestep (default 0.005s).
        transient: Pre-integration burn-in time (default 50s, discarded).
        seed: Random seed for initial conditions.

    Returns:
        (n_samples, 3) array of Lorenz states [x, y, z].
    """
    rng = np.random.default_rng(seed)
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    total_time = duration + transient
    n_steps = int(total_time / dt)
    output_every = max(1, int(1.0 / (output_hz * dt)))

    s = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.1, 3)
    out = []
    for i in range(n_steps):
        def f(s):
            x, y, z = s
            return np.array([sigma*(y-x), x*(rho-z)-y, x*y - beta*z])
        k1 = f(s); k2 = f(s + 0.5*dt*k1)
        k3 = f(s + 0.5*dt*k2); k4 = f(s + dt*k3)
        s = s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        if i % output_every == 0:
            out.append(s.copy())

    out = np.array(out)
    skip = int(transient * output_hz)
    return out[skip:]


def lorenz_to_features(states, n_channels, seed=0):
    """Map 3 Lorenz states to n_channels via random projection + nonlinear.

    Creates n_channels features: random linear projections from 3 Lorenz
    states plus 3 nonlinear channels (xy product, tanh, trig).

    Args:
        states: (n_samples, 3) Lorenz states from integrate_lorenz().
        n_channels: Number of output feature channels.
        seed: Random seed for projection matrix.

    Returns:
        (n_samples, n_channels) array, z-scored and clipped to [-10, 10].
    """
    rng = np.random.default_rng(seed)
    n_linear = max(0, n_channels - 3)

    W = rng.standard_normal((3, n_linear)).astype(np.float32) * 0.5
    linear = states @ W

    x, y, z = states[:, 0], states[:, 1], states[:, 2]
    nl = np.column_stack([
        x * y / (np.abs(x * y).max() + 1e-8),
        np.tanh(x - z),
        np.sin(0.1 * y) * np.cos(0.1 * z),
    ])

    if n_linear > 0:
        features = np.concatenate([linear, nl], axis=1).astype(np.float32)
    else:
        features = nl[:, :n_channels].astype(np.float32)

    mu = features.mean(axis=0, keepdims=True)
    sd = features.std(axis=0, keepdims=True) + 1e-8
    return np.clip((features - mu) / sd, -10, 10)


# =========================================================================
# Event-gated coupling
# =========================================================================

def _generate_single_band_gate(n_samples, hz, duty, ev_min, ev_max, ramp_s,
                                rng):
    """Generate gate for a single event-duration band."""
    from scipy.ndimage import gaussian_filter1d

    duration_s = n_samples / hz
    gate = np.zeros(n_samples, dtype=np.float32)
    target_coupled_s = duty * duration_s
    total_coupled_s = 0.0

    t = rng.uniform(5, max(10, ev_max))
    while total_coupled_s < target_coupled_s and t < duration_s - ev_min:
        event_dur = rng.uniform(ev_min, ev_max)
        start = int(t * hz)
        end = min(int((t + event_dur) * hz), n_samples)
        gate[start:end] = 1.0
        total_coupled_s += event_dur

        mean_gap = event_dur * (1 - duty) / max(duty, 0.01)
        gap = rng.exponential(mean_gap)
        gap = max(gap, ev_min)
        t += event_dur + gap

    sigma = ramp_s * hz / 2.35  # FWHM = ramp_s
    if sigma > 0.5:
        gate = gaussian_filter1d(gate, sigma)
    return gate


def generate_coupling_gate(n_samples, hz, profile, seed=42):
    """Generate time-varying coupling gate g(t) in [0, 1].

    Creates realistic episodic coupling: random events with smooth ramps.
    Between events, gate = 0. During events, gate ramps to 1.

    Supports bimodal profiles via a ``bands`` key: a list of dicts each with
    ``event_range_s``, ``ramp_s``, and ``weight`` (fraction of duty budget).
    When ``bands`` is absent, falls back to the single-band ``event_range_s``
    and ``ramp_s`` keys.

    Args:
        n_samples: Number of time samples.
        hz: Sampling rate.
        profile: Dict with duty_cycle, event_range_s/ramp_s OR bands.
        seed: Random seed.

    Returns:
        (n_samples,) array in [0, 1].
    """
    rng = np.random.default_rng(seed)
    duty = profile['duty_cycle']
    bands = profile.get('bands')

    if bands is None:
        # Single-band (original behaviour)
        ev_min, ev_max = profile['event_range_s']
        ramp_s = profile['ramp_s']
        gate = _generate_single_band_gate(
            n_samples, hz, duty, ev_min, ev_max, ramp_s, rng)
    else:
        # Bimodal / multi-band: each band gets weight * duty budget
        gate = np.zeros(n_samples, dtype=np.float32)
        for band in bands:
            w = band['weight']
            ev_min, ev_max = band['event_range_s']
            ramp_s = band['ramp_s']
            band_gate = _generate_single_band_gate(
                n_samples, hz, duty * w, ev_min, ev_max, ramp_s, rng)
            gate = np.maximum(gate, band_gate)

    gate = np.clip(gate, 0, 1).astype(np.float32)
    return gate


# =========================================================================
# Synthetic session builder
# =========================================================================

def build_synthetic_session_permod(duration, kappa_dict, seed=42,
                                   duty_cycle_override=None,
                                   activity_boost=2.0):
    """Build session with per-modality coupling strengths.

    Activity-coupled events: Both participants get amplitude-modulated
    "activity events". Coupling only happens during a subset of these events.

    Args:
        duration: Session length in seconds.
        kappa_dict: {modality_name: kappa} where kappa=0 means uncoupled.
        seed: Base random seed.
        duty_cycle_override: If set, override all modalities' duty_cycle.
        activity_boost: Amplitude multiplier during activity events (default 2.0).

    Returns:
        Session dict with all modality data, timestamps, validity, and
        coupling_gates for ground truth.
    """
    from cadence.data.preprocessors import compute_activity_channel

    session = {'duration': float(duration)}
    session['p1_role'] = 'therapist'
    session['p2_role'] = 'patient'
    session['role_mapping'] = {'therapist': 'p1', 'patient': 'p2'}
    session['coupling_gates'] = {}
    session['kappa_dict'] = kappa_dict

    for mod_idx, (mod, cfg) in enumerate(SYNTH_MODALITY_CONFIG.items()):
        base_ch = cfg.get('base_ch', cfg['n_ch'])
        hz, lag_s = cfg['hz'], cfg['lag_s']
        lag_samples = int(lag_s * hz)
        kappa = kappa_dict.get(mod, 0.0)

        seed_p1 = seed + mod_idx * 100
        seed_p2 = seed + mod_idx * 100 + 1000
        states_p1 = integrate_lorenz(duration, output_hz=hz, seed=seed_p1)
        states_p2 = integrate_lorenz(duration, output_hz=hz, seed=seed_p2)
        n_samples = min(len(states_p1), len(states_p2))
        states_p1, states_p2 = states_p1[:n_samples], states_p2[:n_samples]

        feat_p1 = lorenz_to_features(states_p1, base_ch, seed=seed_p1 + 50)
        feat_p2_indep = lorenz_to_features(states_p2, base_ch, seed=seed_p2 + 50)

        # Activity event generation — INDEPENDENT for each participant
        # to avoid false coupling from shared amplitude modulation
        profile = COUPLING_PROFILES[mod]
        coupling_duty = duty_cycle_override if duty_cycle_override is not None \
            else profile['duty_cycle']
        activity_duty = min(0.5, max(coupling_duty * 3, 0.15))

        activity_profile = dict(profile)
        activity_profile['duty_cycle'] = activity_duty
        activity_env_p1 = generate_coupling_gate(
            n_samples, hz, activity_profile,
            seed=seed + mod_idx * 100 + 8000)
        activity_env_p2 = generate_coupling_gate(
            n_samples, hz, activity_profile,
            seed=seed + mod_idx * 100 + 9000)

        amp_mod_p1 = (1.0 + activity_boost * activity_env_p1)[:, None]
        amp_mod_p2 = (1.0 + activity_boost * activity_env_p2)[:, None]
        feat_p1 = feat_p1 * amp_mod_p1
        feat_p2_indep = feat_p2_indep * amp_mod_p2

        for arr in [feat_p1, feat_p2_indep]:
            mu = arr.mean(axis=0, keepdims=True)
            sd = arr.std(axis=0, keepdims=True) + 1e-8
            arr[:] = np.clip((arr - mu) / sd, -10, 10).astype(np.float32)

        if kappa > 0:
            # Generate a dedicated coupling gate (independent of activity envelopes)
            coupling_profile = dict(profile)
            coupling_profile['duty_cycle'] = coupling_duty
            gate = generate_coupling_gate(
                n_samples, hz, coupling_profile,
                seed=seed + mod_idx * 100 + 5000)

            session['coupling_gates'][mod] = gate

            alpha_t = kappa * gate
            feat_p2 = np.copy(feat_p2_indep)
            p1_lagged = np.roll(feat_p1, lag_samples, axis=0)
            p1_lagged[:lag_samples] = 0

            a = alpha_t[:, None]
            noise_scale = np.sqrt(np.maximum(1 - a**2, 0.0))
            feat_p2[lag_samples:] = (a[lag_samples:] * p1_lagged[lag_samples:]
                                     + noise_scale[lag_samples:]
                                     * feat_p2_indep[lag_samples:])

            mu = feat_p2.mean(axis=0, keepdims=True)
            sd = feat_p2.std(axis=0, keepdims=True) + 1e-8
            feat_p2 = np.clip((feat_p2 - mu) / sd, -10, 10).astype(np.float32)
        else:
            feat_p2 = feat_p2_indep

        ts = np.arange(n_samples) / hz

        for p, data in [('p1', feat_p1), ('p2', feat_p2)]:
            if base_ch < cfg['n_ch']:
                activity = compute_activity_channel(data[:n_samples], hz,
                                                     trailing_seconds=30.0)
                data_with_act = np.concatenate(
                    [data[:n_samples], activity], axis=1)
            else:
                data_with_act = data[:n_samples]
            session[f'{p}_{mod}'] = data_with_act
            session[f'{p}_{mod}_ts'] = ts.copy()
            session[f'{p}_{mod}_valid'] = np.ones(n_samples, dtype=bool)

    return session


# =========================================================================
# V2 Synthetic: Wavelet-aware session generation
# =========================================================================

def build_synthetic_wavelet_session(duration, coupling_freq=6.5,
                                     coupling_roi='frontal', coupling_lag_s=1.0,
                                     kappa=0.6, seed=42):
    """Build synthetic session with sinusoidal EEG coupling at a specific frequency.

    Two participants have sum-of-sinusoids EEG (broadband 2-45 Hz).
    Coupling is injected at a specific frequency in a specific ROI with
    known lag, gated by episodic coupling windows.

    Args:
        duration: Session length in seconds.
        coupling_freq: Frequency of coupling in Hz (e.g., 6.5 Hz theta-alpha border).
        coupling_roi: ROI name where coupling is injected.
        coupling_lag_s: Lag in seconds for the coupling.
        kappa: Coupling strength (0 = none, 1 = full).
        seed: Random seed.

    Returns:
        session dict with raw EEG (p1_eeg, p2_eeg), timestamps, validity,
        plus ground_truth metadata.
    """
    from cadence.constants import EEG_ROIS, WAVELET_CENTER_FREQS

    rng = np.random.default_rng(seed)
    srate = 256
    n_samples = int(duration * srate)
    n_channels = 14
    t = np.arange(n_samples) / srate

    # Generate broadband EEG for both participants
    freqs = np.logspace(np.log10(2), np.log10(45), 30)
    eeg_p1 = np.zeros((n_samples, n_channels), dtype=np.float32)
    eeg_p2 = np.zeros((n_samples, n_channels), dtype=np.float32)

    for ch in range(n_channels):
        for f in freqs:
            amp = 1.0 / f  # 1/f spectrum
            phase_p1 = rng.uniform(0, 2 * np.pi)
            phase_p2 = rng.uniform(0, 2 * np.pi)
            eeg_p1[:, ch] += amp * np.sin(2 * np.pi * f * t + phase_p1)
            eeg_p2[:, ch] += amp * np.sin(2 * np.pi * f * t + phase_p2)

    # Coupling gate (episodic)
    profile = {'duty_cycle': 0.30, 'event_range_s': (5, 20), 'ramp_s': 2.0}
    gate = generate_coupling_gate(n_samples, srate, profile, seed=seed + 1000)

    # Inject coupling at specific freq + ROI
    roi_channels = EEG_ROIS[coupling_roi]
    lag_samples = int(coupling_lag_s * srate)

    # P1's signal at coupling frequency
    coupling_signal = np.sin(2 * np.pi * coupling_freq * t)

    for ch in roi_channels:
        p1_contrib = np.roll(coupling_signal * eeg_p1[:, ch], lag_samples)
        p1_contrib[:lag_samples] = 0
        alpha_t = kappa * gate
        eeg_p2[:, ch] = (1 - alpha_t) * eeg_p2[:, ch] + alpha_t * p1_contrib

    # Z-score
    for arr in [eeg_p1, eeg_p2]:
        for ch in range(n_channels):
            std = arr[:, ch].std()
            if std > 1e-8:
                arr[:, ch] = (arr[:, ch] - arr[:, ch].mean()) / std

    ts = np.arange(n_samples) / srate
    valid = np.ones((n_samples, n_channels), dtype=bool)

    session = {
        'duration': float(duration),
        'p1_eeg': eeg_p1, 'p1_eeg_ts': ts.copy(), 'p1_eeg_valid': valid.copy(),
        'p2_eeg': eeg_p2, 'p2_eeg_ts': ts.copy(), 'p2_eeg_valid': valid.copy(),
        'p1_role': 'therapist', 'p2_role': 'patient',
        'role_mapping': {'therapist': 'p1', 'patient': 'p2'},
        'coupling_gates': {'eeg_wavelet': gate},
        'ground_truth': {
            'coupling_freq': coupling_freq,
            'coupling_roi': coupling_roi,
            'coupling_lag_s': coupling_lag_s,
            'kappa': kappa,
        },
    }
    return session


def build_synthetic_interbrain_session(duration, coupling_freq=10.0,
                                        coupling_roi='posterior',
                                        kappa=0.6, seed=42):
    """Build synthetic session with inter-brain phase synchrony coupling.

    Phase synchrony between participants at a target frequency during
    coupling windows predicts one participant's behavior.

    Args:
        duration: Session length in seconds.
        coupling_freq: Frequency of phase synchrony in Hz.
        coupling_roi: ROI where synchrony occurs.
        kappa: Coupling strength.
        seed: Random seed.

    Returns:
        session dict with raw EEG and synthetic face/pose data.
    """
    from cadence.constants import EEG_ROIS, SYNTH_MODALITY_CONFIG_V2, COUPLING_PROFILES_V2
    from cadence.data.preprocessors import compute_activity_channel

    rng = np.random.default_rng(seed)
    srate = 256
    n_samples = int(duration * srate)
    n_channels = 14
    t = np.arange(n_samples) / srate

    # Generate EEG with shared phase at coupling_freq during coupling windows
    profile = {'duty_cycle': 0.30, 'event_range_s': (5, 20), 'ramp_s': 2.0}
    gate = generate_coupling_gate(n_samples, srate, profile, seed=seed + 2000)

    eeg_p1 = np.zeros((n_samples, n_channels), dtype=np.float32)
    eeg_p2 = np.zeros((n_samples, n_channels), dtype=np.float32)

    # Broadband background
    freqs = np.logspace(np.log10(2), np.log10(45), 30)
    for ch in range(n_channels):
        for f in freqs:
            amp = 1.0 / f
            eeg_p1[:, ch] += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2*np.pi))
            eeg_p2[:, ch] += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2*np.pi))

    # Inject shared phase in coupling ROI during coupling windows
    roi_channels = EEG_ROIS[coupling_roi]
    shared_phase_signal = np.sin(2 * np.pi * coupling_freq * t)
    for ch in roi_channels:
        eeg_p1[:, ch] += kappa * gate * shared_phase_signal
        eeg_p2[:, ch] += kappa * gate * shared_phase_signal

    # Z-score
    for arr in [eeg_p1, eeg_p2]:
        for ch in range(n_channels):
            std = arr[:, ch].std()
            if std > 1e-8:
                arr[:, ch] = (arr[:, ch] - arr[:, ch].mean()) / std

    ts_eeg = np.arange(n_samples) / srate
    valid_eeg = np.ones((n_samples, n_channels), dtype=bool)

    session = {
        'duration': float(duration),
        'p1_eeg': eeg_p1, 'p1_eeg_ts': ts_eeg.copy(), 'p1_eeg_valid': valid_eeg.copy(),
        'p2_eeg': eeg_p2, 'p2_eeg_ts': ts_eeg.copy(), 'p2_eeg_valid': valid_eeg.copy(),
        'p1_role': 'therapist', 'p2_role': 'patient',
        'role_mapping': {'therapist': 'p1', 'patient': 'p2'},
        'coupling_gates': {'eeg_interbrain': gate},
        'ground_truth': {
            'coupling_freq': coupling_freq,
            'coupling_roi': coupling_roi,
            'kappa': kappa,
            'type': 'interbrain_phase_sync',
        },
    }

    # Also add simple blendshape data influenced by phase sync
    bl_hz = 30.0
    n_bl = int(duration * bl_hz)
    bl_t = np.arange(n_bl) / bl_hz

    # P2's blendshapes partially driven by phase sync
    bl_p1 = rng.standard_normal((n_bl, 52)).astype(np.float32)
    bl_p2 = rng.standard_normal((n_bl, 52)).astype(np.float32)

    # Downsample gate to blendshape rate
    gate_bl = np.interp(bl_t, ts_eeg, gate)
    # Phase sync predicts blendshape movement
    sync_signal = np.interp(bl_t, ts_eeg, shared_phase_signal * gate)
    for ch in range(min(10, 52)):
        bl_p2[:, ch] += 0.3 * kappa * sync_signal

    for p, bl in [('p1', bl_p1), ('p2', bl_p2)]:
        activity = compute_activity_channel(bl, bl_hz, trailing_seconds=30.0)
        session[f'{p}_blendshapes'] = np.concatenate([bl, activity], axis=1)
        session[f'{p}_blendshapes_ts'] = bl_t.copy()
        session[f'{p}_blendshapes_valid'] = np.ones(n_bl, dtype=bool)

    return session


def plan_corpus(n_coupled=50, n_null=150, base_seed=7000):
    """Build list of session specs for corpus generation.

    Returns:
        list of (name, kappa_dict, seed) tuples.
    """
    rng = np.random.default_rng(base_seed)
    specs = []

    eeg_kappas = np.linspace(0.3, 0.9, n_coupled)
    bl_kappas = np.linspace(0.3, 0.9, n_coupled)
    rng.shuffle(bl_kappas)

    for i in range(n_coupled):
        ek = float(round(eeg_kappas[i], 3))
        bk = float(round(bl_kappas[i], 3))
        kappa_dict = {
            'eeg_features': ek,
            'ecg_features': 0.0,
            'blendshapes': bk,
            'pose_features': 0.0,
        }
        specs.append((
            f'C{i:03d}_eeg{ek:.2f}_bl{bk:.2f}',
            kappa_dict,
            base_seed + i * 100,
        ))

    for i in range(n_null):
        kappa_dict = {
            'eeg_features': 0.0,
            'ecg_features': 0.0,
            'blendshapes': 0.0,
            'pose_features': 0.0,
        }
        specs.append((
            f'N{i:03d}_null',
            kappa_dict,
            base_seed + (n_coupled + i) * 100,
        ))

    return specs


# =========================================================================
# V2 Synthetic: Per-modality Lorenz session with V2 modality keys
# =========================================================================

def build_synthetic_session_v2(duration, kappa_dict, seed=42,
                                duty_cycle_override=None,
                                activity_boost=2.0):
    """Build synthetic session with V2 modality names and channel counts.

    Same Lorenz-based coupling injection as build_synthetic_session_permod()
    but uses V2 modality keys (eeg_wavelet, ecg_features_v2, blendshapes_v2,
    pose_features) and their associated channel counts / sample rates from
    SYNTH_MODALITY_CONFIG_V2.

    Args:
        duration: Session length in seconds.
        kappa_dict: {v2_modality_name: kappa} where kappa=0 means uncoupled.
        seed: Base random seed.
        duty_cycle_override: If set, override all modalities' duty_cycle.
        activity_boost: Amplitude multiplier during activity events (default 2.0).

    Returns:
        Session dict with V2 modality data, timestamps, validity, and
        coupling_gates for ground truth.
    """
    from cadence.data.preprocessors import compute_activity_channel

    session = {'duration': float(duration)}
    session['p1_role'] = 'therapist'
    session['p2_role'] = 'patient'
    session['role_mapping'] = {'therapist': 'p1', 'patient': 'p2'}
    session['coupling_gates'] = {}
    session['kappa_dict'] = kappa_dict

    for mod_idx, (mod, cfg) in enumerate(SYNTH_MODALITY_CONFIG_V2.items()):
        base_ch = cfg.get('base_ch', cfg['n_ch'])
        hz, lag_s = cfg['hz'], cfg['lag_s']
        lag_samples = int(lag_s * hz)
        kappa = kappa_dict.get(mod, 0.0)

        seed_p1 = seed + mod_idx * 100
        seed_p2 = seed + mod_idx * 100 + 1000
        states_p1 = integrate_lorenz(duration, output_hz=hz, seed=seed_p1)
        states_p2 = integrate_lorenz(duration, output_hz=hz, seed=seed_p2)
        n_samples = min(len(states_p1), len(states_p2))
        states_p1, states_p2 = states_p1[:n_samples], states_p2[:n_samples]

        feat_p1 = lorenz_to_features(states_p1, base_ch, seed=seed_p1 + 50)
        feat_p2_indep = lorenz_to_features(states_p2, base_ch, seed=seed_p2 + 50)

        # Activity event generation — INDEPENDENT for each participant
        profile = COUPLING_PROFILES_V2[mod]
        coupling_duty = (duty_cycle_override if duty_cycle_override is not None
                         else profile['duty_cycle'])
        activity_duty = min(0.5, max(coupling_duty * 3, 0.15))

        activity_profile = dict(profile)
        activity_profile['duty_cycle'] = activity_duty
        activity_env_p1 = generate_coupling_gate(
            n_samples, hz, activity_profile,
            seed=seed + mod_idx * 100 + 8000)
        activity_env_p2 = generate_coupling_gate(
            n_samples, hz, activity_profile,
            seed=seed + mod_idx * 100 + 9000)

        amp_mod_p1 = (1.0 + activity_boost * activity_env_p1)[:, None]
        amp_mod_p2 = (1.0 + activity_boost * activity_env_p2)[:, None]
        feat_p1 = feat_p1 * amp_mod_p1
        feat_p2_indep = feat_p2_indep * amp_mod_p2

        for arr in [feat_p1, feat_p2_indep]:
            mu = arr.mean(axis=0, keepdims=True)
            sd = arr.std(axis=0, keepdims=True) + 1e-8
            arr[:] = np.clip((arr - mu) / sd, -10, 10).astype(np.float32)

        if kappa > 0:
            coupling_profile = dict(profile)
            coupling_profile['duty_cycle'] = coupling_duty
            gate = generate_coupling_gate(
                n_samples, hz, coupling_profile,
                seed=seed + mod_idx * 100 + 5000)

            session['coupling_gates'][mod] = gate

            # Sparse coupling: only inject into n_coupled channels
            # (matches real data where only specific freq-ROI combos couple)
            n_coupled = cfg.get('n_coupled', base_ch)
            n_coupled = min(n_coupled, base_ch)

            alpha_t = kappa * gate
            feat_p2 = np.copy(feat_p2_indep)
            p1_lagged = np.roll(feat_p1, lag_samples, axis=0)
            p1_lagged[:lag_samples] = 0

            a = alpha_t[:, None]
            noise_scale = np.sqrt(np.maximum(1 - a**2, 0.0))
            # Only coupled channels get the mixing; rest stay independent
            feat_p2[lag_samples:, :n_coupled] = (
                a[lag_samples:] * p1_lagged[lag_samples:, :n_coupled]
                + noise_scale[lag_samples:]
                * feat_p2_indep[lag_samples:, :n_coupled])

            mu = feat_p2.mean(axis=0, keepdims=True)
            sd = feat_p2.std(axis=0, keepdims=True) + 1e-8
            feat_p2 = np.clip((feat_p2 - mu) / sd, -10, 10).astype(np.float32)
        else:
            feat_p2 = feat_p2_indep

        # --- Fix 2: Recompute derivative channels from coupled PCA signal ---
        # In real data, d(coupled_PCA)/dt also shows coupling (chain rule).
        # Without this, derivative channels are pure noise, diluting the
        # source-side group lasso gradient for behavioral modalities.
        if kappa > 0 and cfg.get('has_derivatives'):
            from cadence.data.preprocessors import _compute_temporal_derivatives
            n_pca = cfg.get('n_pca', 15)
            for feat_arr in [feat_p1, feat_p2]:
                if feat_arr.shape[1] >= 2 * n_pca:
                    pca_part = feat_arr[:, :n_pca]
                    deriv = _compute_temporal_derivatives(pca_part, hz, sigma_s=0.5)
                    mu_d = deriv.mean(axis=0, keepdims=True)
                    sd_d = deriv.std(axis=0, keepdims=True) + 1e-8
                    feat_arr[:, n_pca:2*n_pca] = np.clip(
                        (deriv - mu_d) / sd_d, -10, 10).astype(np.float32)

        ts = np.arange(n_samples) / hz

        for p, data in [('p1', feat_p1), ('p2', feat_p2)]:
            if base_ch < cfg['n_ch']:
                activity = compute_activity_channel(data[:n_samples], hz,
                                                     trailing_seconds=30.0)
                data_with_act = np.concatenate(
                    [data[:n_samples], activity], axis=1)
            else:
                data_with_act = data[:n_samples]
            session[f'{p}_{mod}'] = data_with_act
            session[f'{p}_{mod}_ts'] = ts.copy()
            session[f'{p}_{mod}_valid'] = np.ones(n_samples, dtype=bool)

    # ---------------------------------------------------------------
    # Generate interbrain features (source-only, no participant prefix)
    # These are 160-ch (2 components x 20 freqs x 4 ROIs) at 5 Hz,
    # stored without p1/p2 prefix since they represent cross-brain
    # phase-locking values computed from BOTH participants' EEG.
    # For synthetic tests the interbrain data is uncoupled noise
    # unless a specific interbrain test injects coupling.
    # ---------------------------------------------------------------
    ib_mod = 'eeg_interbrain'
    ib_n_ch = 160
    ib_hz = 5.0
    ib_n_samples = int(duration * ib_hz)
    rng_ib = np.random.default_rng(seed + 77777)
    ib_data = rng_ib.standard_normal((ib_n_samples, ib_n_ch)).astype(np.float32)
    # Z-score
    mu_ib = ib_data.mean(axis=0, keepdims=True)
    sd_ib = ib_data.std(axis=0, keepdims=True) + 1e-8
    ib_data = np.clip((ib_data - mu_ib) / sd_ib, -10, 10).astype(np.float32)

    ib_ts = np.arange(ib_n_samples) / ib_hz
    session[ib_mod] = ib_data
    session[f'{ib_mod}_ts'] = ib_ts
    session[f'{ib_mod}_valid'] = np.ones(ib_n_samples, dtype=bool)

    return session


def plan_corpus_v2(n_coupled=50, n_null=20, base_seed=7000):
    """Build list of session specs for V2 corpus generation.

    Returns:
        list of (name, kappa_dict, seed) tuples using V2 modality keys.
    """
    rng = np.random.default_rng(base_seed)
    specs = []

    eeg_kappas = np.linspace(0.3, 0.9, n_coupled)
    bl_kappas = np.linspace(0.3, 0.9, n_coupled)
    rng.shuffle(bl_kappas)

    for i in range(n_coupled):
        ek = float(round(eeg_kappas[i], 3))
        bk = float(round(bl_kappas[i], 3))
        kappa_dict = {
            'eeg_wavelet': ek,
            'ecg_features_v2': 0.0,
            'blendshapes_v2': bk,
            'pose_features': 0.0,
        }
        specs.append((
            f'C{i:03d}_eeg{ek:.2f}_bl{bk:.2f}',
            kappa_dict,
            base_seed + i * 100,
        ))

    for i in range(n_null):
        kappa_dict = {
            'eeg_wavelet': 0.0,
            'ecg_features_v2': 0.0,
            'blendshapes_v2': 0.0,
            'pose_features': 0.0,
        }
        specs.append((
            f'N{i:03d}_null',
            kappa_dict,
            base_seed + (n_coupled + i) * 100,
        ))

    return specs
