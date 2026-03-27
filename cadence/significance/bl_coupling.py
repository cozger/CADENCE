"""Two-stage BL coupling pipeline: continuous detection + per-event characterization.

Stage 1: Cross-product multi-lag bank → coupling windows + lag estimation.
Stage 2: Per-event mimicry test → which events were mimicked, at what lag, per-event p-values.

The cross-product is optimal for DETECTING coupling (integrates all timepoints).
The per-event test is optimal for CHARACTERIZING coupling (uses the event structure).
They are complementary: Stage 1 provides the data-driven lag estimate that Stage 2
needs for tight-window per-event significance.
"""

import numpy as np
from scipy.signal import find_peaks, butter, sosfiltfilt, hilbert
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class ExpressionEvent:
    """A single detected expression event for one person."""
    time: float                 # seconds from segment start
    amplitude: float            # raw composite peak value
    composite: str = 'smile'


@dataclass
class CoOccurrence:
    """A co-occurrence: both people expressed within a time window."""
    person_a_time: float
    person_b_time: float
    person_a_amplitude: float
    person_b_amplitude: float
    lag: float                  # B_time - A_time (positive = A led)
    leader: str                 # 'A' or 'B'
    composite: str = 'smile'
    # Causal attribution
    attribution: Optional[str] = None  # 'mimicry', 'shared_stimulus', 'coincidence'
    causal_confidence: float = 0.0
    follower_pre_slope: float = 0.0
    follower_baseline: float = 0.0


@dataclass
class ExpressionCatalog:
    """Per-composite expression and co-occurrence catalog."""
    name: str
    # Individual counts
    n_events_a: int
    n_events_b: int
    events_a: List[ExpressionEvent] = field(default_factory=list)
    events_b: List[ExpressionEvent] = field(default_factory=list)
    # Co-occurrences
    cooccurrences: List[CoOccurrence] = field(default_factory=list)
    n_cooccurrences: int = 0
    cooccurrence_rate: float = 0.0      # fraction of A events with B co-occurrence
    null_cooccurrence_rate: float = 0.0
    session_p_value: float = 1.0
    # Directionality
    n_a_led: int = 0
    n_b_led: int = 0
    mean_lag: Optional[float] = None
    # Causal breakdown
    n_mimicry: int = 0
    n_shared_stimulus: int = 0
    n_coincidence: int = 0


@dataclass
class BLCouplingResult:
    """Complete BL coupling analysis output."""
    # Stage 1: continuous coupling
    z_continuous: np.ndarray     # (T_out,) cross-product z-score timecourse
    mask_continuous: np.ndarray  # (T_out,) binary coupling mask
    best_lag_s: np.ndarray       # (T_out,) estimated lag per timepoint
    times: np.ndarray            # (T_out,) timestamps
    output_rate: float           # Hz

    # Lag estimation
    estimated_lag_s: float
    lag_confidence: float
    lag_window_s: Tuple[float, float]

    # Stage 2: per-composite catalogs
    catalogs: Dict[str, ExpressionCatalog] = field(default_factory=dict)

    diagnostics: Dict = field(default_factory=dict)


# ── Composite definitions ──────────────────────────────────────────────
# Each composite is a named set of AUs whose sum captures a
# meaningful expression category.  'general' uses velocity across ALL AUs
# to detect any facial event regardless of type.

EXPRESSION_COMPOSITES = {
    'smile':  (44, 45),        # mouthSmileLeft + mouthSmileRight
    'brow':   (3, 4, 5),       # browInnerUp + browOuterUpLeft + browOuterUpRight
    'frown':  (30, 31),        # mouthFrownLeft + mouthFrownRight
    'speech': (25, 33, 39),    # jawOpen + mouthLeft + mouthRight
}


def _build_au_composite(signal_2d_raw, aus, fs, smooth_s=0.3):
    """Build composite from RAW AU values (not z-scored).

    Uses raw activation values so that event detection thresholds
    correspond to visible expressions, not statistical noise.
    A light smoothing merges sub-frame jitter.

    Returns the raw composite (sum of AU activations).
    Peak detection should use absolute prominence thresholds
    (e.g., prominence=0.3 means the peak is 0.3 raw units above
    the local baseline — a clearly visible expression).
    """
    T = signal_2d_raw.shape[0]
    comp = np.zeros(T)
    for ch in aus:
        if ch >= signal_2d_raw.shape[1]:
            continue
        comp += signal_2d_raw[:, ch]
    if smooth_s > 0 and fs > 0:
        comp = gaussian_filter1d(comp, sigma=smooth_s * fs)
    return comp


def _build_general_composite(signal_2d_raw, fs, smooth_s=0.3):
    """Build a general facial activity composite from ALL AUs.

    Uses sum of raw activation across all AUs — captures any
    facial event regardless of type. Light smoothing merges jitter.
    """
    T, C = signal_2d_raw.shape
    n_aus = min(C, 52)
    activity = np.zeros(T)
    for ch in range(n_aus):
        activity += signal_2d_raw[:, ch]
    if smooth_s > 0 and fs > 0:
        activity = gaussian_filter1d(activity, sigma=smooth_s * fs)
    return activity


def _estimate_characteristic_lag(best_lag_arr, z_scores, mask, fs_out,
                                  prior_mean=2.5, prior_std=1.0):
    """Estimate the characteristic coupling lag with hierarchical shrinkage.

    Uses lag values from high-z timepoints (coupled windows) as the
    session-level likelihood, combined with a population-level Gaussian
    prior via Bayesian shrinkage.

    When the session has strong evidence (many high-z timepoints with
    consistent lags), the posterior stays close to the session estimate.
    When evidence is weak or scattered, the posterior shrinks toward the
    population prior (mean ~2.5s).

    This is a Normal-Normal conjugate update:
        posterior_mean = (prior_mean/prior_var + session_mean/session_var) /
                         (1/prior_var + 1/session_var)
        posterior_var  = 1 / (1/prior_var + 1/session_var)

    Args:
        best_lag_arr: (T_out,) lag in samples at original fs.
        z_scores: (T_out,) z-score timecourse.
        mask: (T_out,) boolean coupling mask (or None).
        fs_out: Original sampling rate for lag→seconds conversion.
        prior_mean: Population prior mean lag in seconds.
        prior_std: Population prior std in seconds.

    Returns:
        posterior_lag: Shrinkage-estimated lag in seconds.
        confidence: 0-1 measure of how much the estimate relies on data vs prior.
        window: (lo, hi) lag window for per-event matching.
    """
    prior_var = prior_std ** 2

    # Extract lag values from high-z timepoints
    strong = z_scores > 2.0
    if mask is not None:
        strong = strong & mask

    if strong.sum() < 5:
        threshold = np.percentile(z_scores, 90)
        strong = z_scores > threshold

    if strong.sum() < 3:
        # No usable session data — fall back to pure prior
        posterior_lag = prior_mean
        posterior_std = prior_std
        confidence = 0.0
    else:
        lag_samples = best_lag_arr[strong]
        lag_seconds = lag_samples / fs_out if fs_out > 0 else lag_samples

        session_mean = float(np.median(lag_seconds))
        session_mad = float(np.median(np.abs(lag_seconds - session_mean)))
        # Convert MAD to std estimate (MAD ≈ 0.6745 * std for normal)
        session_std = max(session_mad / 0.6745, 0.1)
        # Reduce session variance with more evidence (effective n)
        n_eff = min(strong.sum(), 50)  # cap to avoid overconfidence
        session_var = (session_std ** 2) / n_eff

        # Normal-Normal conjugate posterior
        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / session_var)
        posterior_lag = posterior_var * (prior_mean / prior_var +
                                         session_mean / session_var)
        posterior_std = np.sqrt(posterior_var)

        # Confidence = fraction of posterior precision from session data
        # (1 = all data, 0 = all prior)
        confidence = float((1.0 / session_var) /
                           (1.0 / prior_var + 1.0 / session_var))

    # Window: posterior_lag ± k * posterior_std
    # Use 1.5σ for ~87% coverage of the posterior
    half_width = max(0.3, min(1.5 * posterior_std, 1.5))
    window = (max(0.0, posterior_lag - half_width),
              posterior_lag + half_width)

    return float(posterior_lag), float(confidence), window


def _attribute_event(event, comp_target, fs, pre_window_s=1.5,
                     baseline_window_s=1.0):
    """Compute causal attribution for a single matched mimicry event.

    Analyzes the target's expression trajectory BEFORE the response
    to distinguish:
      - MIMICRY: target was flat, then rose sharply at the lag → caused by source
      - SHARED_STIMULUS: target was already rising → both responding to same thing
      - COINCIDENCE: target was already at high amplitude → independent event

    Args:
        event: MimicryEvent with matched=True.
        comp_target: (T,) raw composite signal of the target.
        fs: Sampling rate.
        pre_window_s: How far before the response to check slope.
        baseline_window_s: Window for computing baseline before the pre-window.

    Returns:
        Updated event with causal_confidence, attribution, target_pre_slope,
        target_baseline.
    """
    if not event.matched or event.target_time is None:
        return event

    resp_samp = int(event.target_time * fs)
    T = len(comp_target)

    # Baseline: target level well before the response
    # [resp - pre_window - baseline_window : resp - pre_window]
    bl_end = max(0, resp_samp - int(pre_window_s * fs))
    bl_start = max(0, bl_end - int(baseline_window_s * fs))
    if bl_start >= bl_end:
        bl_start = max(0, bl_end - int(0.5 * fs))

    baseline = float(comp_target[bl_start:bl_end].mean()) if bl_end > bl_start else 0.0

    # Pre-slope: derivative in the 1s before response onset
    pre_start = max(0, resp_samp - int(pre_window_s * fs))
    pre_end = resp_samp
    if pre_end > pre_start + 3:
        pre_signal = comp_target[pre_start:pre_end]
        # Linear regression slope
        x = np.arange(len(pre_signal)) / fs
        slope = np.polyfit(x, pre_signal, 1)[0]
    else:
        slope = 0.0

    # Response amplitude relative to baseline
    resp_amp = float(comp_target[min(resp_samp, T - 1)])
    rise = resp_amp - baseline

    # Attribution logic:
    # 1. If baseline is already high (> 50% of response) → coincidence
    # 2. If pre-slope is steep positive (target already rising) → shared stimulus
    # 3. If baseline low AND flat before response → mimicry

    # Normalize slope by the rise magnitude to get relative pre-activation
    rise_safe = max(rise, 0.01)
    pre_activation = max(slope * pre_window_s, 0) / rise_safe  # fraction of rise
    # already explained by pre-existing trend

    baseline_ratio = baseline / max(resp_amp, 0.01)

    if baseline_ratio > 0.6:
        # Target was already activated — coincidental overlap
        attribution = 'coincidence'
        causal_confidence = max(0.0, 0.3 - baseline_ratio)
    elif pre_activation > 0.5:
        # Target was already rising before the expected lag
        attribution = 'shared_stimulus'
        causal_confidence = max(0.0, 0.7 - pre_activation)
    else:
        # Target was flat, then rose at the lag — mimicry
        attribution = 'mimicry'
        causal_confidence = min(1.0, 0.7 + 0.3 * (1.0 - pre_activation))

    event.causal_confidence = round(causal_confidence, 3)
    event.attribution = attribution
    event.target_pre_slope = round(slope, 4)
    event.target_baseline = round(baseline, 4)

    return event


def _attribute_cooccurrence(co, follower_comp, follower_time, fs,
                            pre_window_s=1.5, baseline_window_s=1.0):
    """Attribute a co-occurrence as mimicry, shared_stimulus, or coincidence.

    Analyzes the FOLLOWER's expression trajectory before their response.
    """
    resp_samp = int(follower_time * fs)
    T = len(follower_comp)

    bl_end = max(0, resp_samp - int(pre_window_s * fs))
    bl_start = max(0, bl_end - int(baseline_window_s * fs))
    baseline = float(follower_comp[bl_start:bl_end].mean()) if bl_end > bl_start else 0.0

    pre_start = max(0, resp_samp - int(pre_window_s * fs))
    pre_end = resp_samp
    if pre_end > pre_start + 3:
        pre_signal = follower_comp[pre_start:pre_end]
        x = np.arange(len(pre_signal)) / fs
        slope = float(np.polyfit(x, pre_signal, 1)[0])
    else:
        slope = 0.0

    resp_amp = float(follower_comp[min(resp_samp, T - 1)])
    rise = resp_amp - baseline
    rise_safe = max(rise, 0.01)
    pre_activation = max(slope * pre_window_s, 0) / rise_safe
    baseline_ratio = baseline / max(resp_amp, 0.01)

    if baseline_ratio > 0.6:
        co.attribution = 'coincidence'
        co.causal_confidence = max(0.0, 0.3 - baseline_ratio)
    elif pre_activation > 0.5:
        co.attribution = 'shared_stimulus'
        co.causal_confidence = max(0.0, 0.7 - pre_activation)
    else:
        co.attribution = 'mimicry'
        co.causal_confidence = min(1.0, 0.7 + 0.3 * (1.0 - pre_activation))

    co.follower_pre_slope = round(slope, 4)
    co.follower_baseline = round(baseline, 4)
    return co


def _per_event_p_value(target_event_rate, window_width_s):
    """P-value for a single event match under Poisson null.

    P(at least one target event in window) = 1 - exp(-rate * width).
    """
    expected = target_event_rate * window_width_s
    return 1.0 - np.exp(-expected)


def bl_two_stage_coupling(p1_raw, p2_raw, fs,
                          composites=None,
                          # Stage 1 params
                          xcorr_channels=None,
                          max_lag_s=5.0, lag_step_s=0.1,
                          smooth_s=None, n_surrogates=100,
                          target_fa=0.05,
                          # Stage 2 params
                          event_prominence=0.3,
                          min_event_iei_s=3.0,
                          response_threshold=0.15,
                          n_surrogates_event=500,
                          lag_prior_s=None,
                          population_prior=(2.5, 1.0),
                          seed=42):
    """Two-stage BL coupling: continuous detection + per-event characterization.

    Stage 1: Cross-product multi-lag bank on z-scored AUs → coupling mask + lag.
    Stage 2: Per-event mimicry on RAW AU composites → events are real visible
        expressions, not z-score artifacts.

    Args:
        p1_raw, p2_raw: (T, C) raw AU signals at native rate.
            NOT z-scored — raw blendshape values [0, 1].
            Z-scoring is done internally for Stage 1 only.
        fs: Sampling rate.
        composites: Dict of {name: tuple_of_au_indices}.
            None = use defaults (smile, brow, frown, speech).

        xcorr_channels: AU indices for Stage 1 cross-product (None = all 52).
        max_lag_s, lag_step_s, smooth_s, n_surrogates, target_fa: Stage 1 params.

        event_prominence: Prominence for peak detection in RAW composite units.
            Default 0.3 — a peak must rise 0.3 above local baseline to count.
            For smile composite (AU43+44+17), 0.3 ≈ a visible smile.
        min_event_iei_s: Minimum inter-event interval in seconds.
        response_threshold: Minimum prominence for target responses.
        n_surrogates_event: Surrogates for per-event significance.
        lag_prior_s: Optional (min, max) hard lag window override.
        population_prior: (mean, std) for hierarchical Bayesian shrinkage.
        seed: Random seed.

    Returns:
        BLCouplingResult with continuous mask + per-composite event catalogs.
    """
    from cadence.significance.coherence_localization import xcorr_temporal_localization
    from cadence.significance.kim_filter import _estimate_ar

    T, C = p1_raw.shape
    duration_s = T / fs

    # Adaptive smoothing.  Empirically validated: d' is governed by
    # effective DOF (≈ smooth_s / decorrelation_time), NOT by sample
    # count.  AU signals decorrelate in ~0.2 s, so DOF ≈ smooth_s/0.2.
    # At 30 Hz × 3 s this gives ~15 DOF (baseline).  To keep d' within
    # ~7 % of baseline at higher rates, use 2.5 s (12.5 DOF, −7 % d')
    # which buys 17 % better temporal resolution.
    #   30 Hz → 3.0 s  (15 DOF, baseline)
    #   60 Hz → 2.5 s  (12.5 DOF, −7 % d', +17 % temporal res)
    if smooth_s is None:
        smooth_s = 2.5 if fs > 50 else 3.0

    # ── Stage 1: Cross-product multi-lag bank (z-scored internally) ──

    # Differentiate: co-MOVEMENT (velocity) not co-LEVEL (tonic values)
    p1_z = np.diff(p1_raw, axis=0, prepend=p1_raw[:1])
    p2_z = np.diff(p2_raw, axis=0, prepend=p2_raw[:1])

    # Expressiveness weighting: downweight channels with high baseline
    # variance (eye squint, blink — always active), upweight channels
    # that are mostly silent but spike during expressions (smile, frown).
    #
    # Weight = 1 / (baseline_std + epsilon).  Channels with constant
    # high-frequency jitter (blinks) have large velocity std → low weight.
    # Channels that are mostly zero but occasionally spike (smiles) have
    # small velocity std → high weight.
    #
    # Computed on the AVERAGE of both participants so the weighting is
    # symmetric and doesn't leak coupling information.
    n_ch = min(C, 52)
    ch_std = np.zeros(n_ch)
    for c in range(n_ch):
        ch_std[c] = 0.5 * (p1_z[:, c].std() + p2_z[:, c].std())
    ch_weights = 1.0 / (ch_std + 1e-6)
    ch_weights /= ch_weights.sum()  # normalize to sum to 1
    ch_weights *= n_ch              # scale so mean weight = 1

    # Z-score then apply expressiveness weight
    for c in range(n_ch):
        for sig in [p1_z, p2_z]:
            mu, sd = sig[:, c].mean(), max(sig[:, c].std(), 1e-8)
            sig[:, c] = (sig[:, c] - mu) / sd * ch_weights[c]

    if xcorr_channels is None:
        xcorr_channels = list(range(min(C, 52)))
    p2_res = p2_z.copy()
    for ch in xcorr_channels:
        a, _ = _estimate_ar(p2_z[:, ch], 3)
        pred = np.zeros(T)
        for i in range(len(a)):
            pred[i+1:] += a[i] * p2_z[:T-i-1, ch]
        p2_res[:, ch] = p2_z[:, ch] - pred

    mask_cont, z_cont, lag_arr, diag_s1 = xcorr_temporal_localization(
        p1_z[:, xcorr_channels], p2_res[:, xcorr_channels], fs,
        max_lag_s=max_lag_s, lag_step_s=lag_step_s,
        smooth_s=smooth_s, n_surrogates=n_surrogates,
        target_fa=target_fa, min_event_s=5.0, seed=seed)

    output_rate = diag_s1['output_rate']
    times = np.arange(len(z_cont)) / output_rate

    # Estimate characteristic lag from Stage 1 with hierarchical shrinkage
    # lag_arr is in samples at ORIGINAL fs (not output rate)
    prior_args = {}
    if population_prior is not None:
        prior_args = {'prior_mean': population_prior[0],
                      'prior_std': population_prior[1]}
    estimated_lag, lag_conf, lag_window = _estimate_characteristic_lag(
        lag_arr, z_cont, mask_cont, fs, **prior_args)

    # ── Stage 2: Per-event mimicry across multiple composites ──

    # Build composite dict: always include 'general' + user-specified
    if composites is None:
        composites = dict(EXPRESSION_COMPOSITES)
    comp_dict = dict(composites)  # copy

    # Lag window from Stage 1 (shared across composites — lag is a
    # property of the dyad, not the expression type)
    if lag_prior_s is not None:
        win_lo, win_hi = lag_prior_s
        estimated_lag = (win_lo + win_hi) / 2
        lag_window = (win_lo, win_hi)
    else:
        win_lo, win_hi = lag_window

    def _build_catalog(comp_name, comp_a, comp_b):
        """Build expression catalog with co-occurrence detection."""
        # Detect events for both people (same threshold)
        pks_a, _ = find_peaks(comp_a, prominence=event_prominence,
                               distance=int(min_event_iei_s * fs))
        pks_b, _ = find_peaks(comp_b, prominence=event_prominence,
                               distance=int(min_event_iei_s * fs))

        events_a = [ExpressionEvent(time=pk/fs, amplitude=float(comp_a[pk]),
                                     composite=comp_name) for pk in pks_a]
        events_b = [ExpressionEvent(time=pk/fs, amplitude=float(comp_b[pk]),
                                     composite=comp_name) for pk in pks_b]

        times_a = pks_a / fs
        times_b = pks_b / fs

        # Find co-occurrences: A event and B event within ±cooccurrence_window
        cooc_window = (win_hi + win_lo) / 2  # use the lag estimate as window
        cooc_window = max(cooc_window, 2.0)   # at least 2s

        cooccurrences = []
        used_b = set()

        for i, t_a in enumerate(times_a):
            # Find closest B event within ±window
            diffs = times_b - t_a
            in_window = np.where(np.abs(diffs) <= cooc_window)[0]
            if len(in_window) == 0:
                continue
            # Pick closest unused B event
            for j in in_window[np.argsort(np.abs(diffs[in_window]))]:
                if j in used_b:
                    continue
                used_b.add(j)
                t_b = times_b[j]
                lag = t_b - t_a  # positive = A led
                leader = 'A' if lag > 0 else 'B'

                co = CoOccurrence(
                    person_a_time=t_a, person_b_time=t_b,
                    person_a_amplitude=float(comp_a[pks_a[i]]),
                    person_b_amplitude=float(comp_b[pks_b[j]]),
                    lag=lag, leader=leader, composite=comp_name)

                # Causal attribution on the follower
                follower_comp = comp_b if leader == 'A' else comp_a
                follower_time = t_b if leader == 'A' else t_a
                co = _attribute_cooccurrence(co, follower_comp, follower_time, fs)
                cooccurrences.append(co)
                break

        n_co = len(cooccurrences)
        co_rate = n_co / max(len(events_a), 1)

        # Surrogate: shift A events, count co-occurrences
        rng_c = np.random.default_rng(seed + hash(comp_name) % 10000)
        null_co = np.zeros(n_surrogates_event)
        for k in range(n_surrogates_event):
            shift = rng_c.uniform(0.1 * duration_s, 0.9 * duration_s)
            shifted_a = (times_a + shift) % duration_s
            n_null = 0
            for t_a in shifted_a:
                if np.any(np.abs(times_b - t_a) <= cooc_window):
                    n_null += 1
            null_co[k] = n_null

        null_rate = float(null_co.mean() / max(len(events_a), 1))
        sess_p = float((null_co >= n_co).mean())
        if sess_p == 0:
            sess_p = 1.0 / (n_surrogates_event + 1)

        lags = [c.lag for c in cooccurrences]
        n_a_led = sum(1 for c in cooccurrences if c.leader == 'A')
        n_b_led = sum(1 for c in cooccurrences if c.leader == 'B')

        return ExpressionCatalog(
            name=comp_name,
            n_events_a=len(events_a), n_events_b=len(events_b),
            events_a=events_a, events_b=events_b,
            cooccurrences=cooccurrences,
            n_cooccurrences=n_co,
            cooccurrence_rate=co_rate,
            null_cooccurrence_rate=null_rate,
            session_p_value=sess_p,
            n_a_led=n_a_led, n_b_led=n_b_led,
            mean_lag=float(np.mean(lags)) if lags else None,
            n_mimicry=sum(1 for c in cooccurrences if c.attribution == 'mimicry'),
            n_shared_stimulus=sum(1 for c in cooccurrences if c.attribution == 'shared_stimulus'),
            n_coincidence=sum(1 for c in cooccurrences if c.attribution == 'coincidence'),
        )

    # Run all composites
    catalogs = {}

    gen_a = _build_general_composite(p1_raw, fs)
    gen_b = _build_general_composite(p2_raw, fs)
    catalogs['general'] = _build_catalog('general', gen_a, gen_b)

    for name, aus in comp_dict.items():
        ca = _build_au_composite(p1_raw, aus, fs)
        cb = _build_au_composite(p2_raw, aus, fs)
        catalogs[name] = _build_catalog(name, ca, cb)

    return BLCouplingResult(
        z_continuous=z_cont,
        mask_continuous=mask_cont,
        best_lag_s=lag_arr / fs,
        times=times,
        output_rate=output_rate,
        estimated_lag_s=estimated_lag,
        lag_confidence=lag_conf,
        lag_window_s=lag_window,
        catalogs=catalogs,
        diagnostics={'stage1': diag_s1},
    )


# ── V6: Saliency-based facial event detection + confidence scoring ───
#
# Detects ANY salient facial event via velocity norm on non-eye AUs,
# then scores confidence (is this a real smile?) from AU profiles.
# No significance testing — a shared smile during conversation is a
# real interpersonal event by definition.

# Non-eye AU indices (MediaPipe FaceLandmarker ordering)
_EYE_AUS = set(range(9, 23))  # eyeBlink*, eyeLook*, eyeSquint*, eyeWide*
_EXPR_AUS = [i for i in range(52) if i not in _EYE_AUS]

# Speech AU partition (from constants.BLENDSHAPE_COUPLING_GROUPS['speech'])
# Empirically validated: AU34/35 are 50x discriminative for speech vs silence
_SPEECH_AUS = {23, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 39,
               40, 41, 42, 43, 46, 47, 48, 49}
_NONSPEECH_EXPR_AUS = [i for i in _EXPR_AUS if i not in _SPEECH_AUS]
# Index mappings into the _EXPR_AUS array for velocity partitioning
_NONSPEECH_INDICES = [j for j, i in enumerate(_EXPR_AUS) if i not in _SPEECH_AUS]
_SPEECH_INDICES = [j for j, i in enumerate(_EXPR_AUS) if i in _SPEECH_AUS]

# Blink-associated AU partition (brow AUs that co-activate during blinks)
# Empirically validated: browInnerUp 3.0x, browOuterUp 1.6-2.3x during blinks
# cheekSquint (7, 8) NOT included — minimal blink co-activation, needed as smile marker
_BLINK_ASSOC_AUS = {0, 1, 2, 3, 4, 5}  # neutral + brow group
_BLINK_INDICES = [j for j, i in enumerate(_EXPR_AUS) if i in _BLINK_ASSOC_AUS]

# Affect AUs: the expression channels we actually care about.
# This is the residual after subtracting speech (jaw/mouth), blinks (brow),
# and eyes (already excluded). Only emotional expression remains.
_AFFECT_AUS = [7, 8, 28, 29, 30, 31, 44, 45, 50, 51]
# cheekSquintL/R, mouthDimpleL/R, mouthFrownL/R, mouthSmileL/R, noseSneerL/R


@dataclass
class FacialEvent:
    """A detected facial event for one person."""
    time: float                  # seconds from segment start
    lsl_time: float              # LSL timestamp (for cross-modal anchoring)
    saliency: float              # velocity norm amplitude at peak
    au_snapshot: np.ndarray      # (52,) raw AU values at event
    smile_composite: float       # AU44 + AU45
    smile_confidence: float      # amplitude x dominance x bilaterality
    top_aus: List[Tuple[int, str, float]]  # [(idx, name, value), ...]
    speech_prob: float = 0.0     # speech probability at event time [0, 1]
    blink_prob: float = 0.0      # blink mask value at event time [0, 1]
    smile_phasic: float = 0.0   # deviation of AU44+AU45 from trailing baseline
    smile_velocity: float = 0.0  # d/dt of smile signal at event (positive = onset)


@dataclass
class SharedEvent:
    """A co-occurring facial event between two people."""
    event_a: FacialEvent
    event_b: FacialEvent
    lag: float                   # B_time - A_time (positive = A led)
    leader: str                  # 'A' or 'B'
    joint_smile_confidence: float  # geometric mean of both smile confidences
    is_shared_smile: bool        # both have smile_composite > threshold


@dataclass
class FacialEventCatalog:
    """Complete facial event catalog for a segment."""
    events_p1: List[FacialEvent]
    events_p2: List[FacialEvent]
    shared_events: List[SharedEvent]
    shared_smiles: List[SharedEvent]  # filtered to is_shared_smile=True
    # Summary stats
    n_events_p1: int
    n_events_p2: int
    n_shared: int
    n_shared_smiles: int
    duration_s: float
    segment_name: str
    lsl_start: float = 0.0
    # Speech detection masks (None when speech_gating=False)
    speech_p1: Optional[np.ndarray] = None  # (T,) speech probability for P1
    speech_p2: Optional[np.ndarray] = None  # (T,) speech probability for P2
    # Blink detection masks (None when blink_gating=False)
    blink_p1: Optional[np.ndarray] = None   # (T,) blink mask for P1
    blink_p2: Optional[np.ndarray] = None   # (T,) blink mask for P2


def _detect_speech_from_blendshapes(signal, fs=30.0, lo=2.0, hi=8.0,
                                    smooth_s=0.5, min_bout_s=0.3,
                                    margin_s=0.15):
    """Detect speech from blendshape temporal texture.

    Uses mouthLowerDown L/R (AU34+35) bandpass [2-8 Hz] + Hilbert envelope.
    These AUs are 50x more discriminative for speech vs silence than jawOpen,
    empirically validated on y_06 (therapist-speaking vs patient-silent).

    Args:
        signal: (T, 52) raw blendshapes [0, 1].
        fs: sampling rate (Hz).
        lo, hi: bandpass frequency range (Hz).
        smooth_s: envelope smoothing (seconds).
        min_bout_s: minimum speech bout duration (seconds).
        margin_s: pad speech regions by this amount on each side (seconds).

    Returns:
        speech_prob: (T,) array, continuous [0, 1] speech probability.
    """
    T = signal.shape[0]
    nyq = fs / 2.0
    if hi >= nyq:
        hi = nyq - 0.5

    # Primary: mouthLowerDown L/R (AU34 + AU35) — best speech indicators
    lip_open = signal[:, 34].astype(np.float64) + signal[:, 35].astype(np.float64)

    # Bandpass filter to isolate syllable-rate oscillation
    sos = butter(4, [lo / nyq, hi / nyq], btype='band', output='sos')
    filtered = sosfiltfilt(sos, lip_open)

    # Analytic envelope via Hilbert transform
    env = np.abs(hilbert(filtered)).astype(np.float32)

    # Smooth
    if smooth_s > 0:
        env = gaussian_filter1d(env, sigma=smooth_s * fs)

    # Threshold using peak-relative scaling with absolute noise floor.
    # Calibrated from y_06 meditation ground truth:
    #   Patient silence: max envelope ~0.001 (true null)
    #   Therapist speech: peaks at 0.010-0.012 (soft guided meditation)
    #   Conversation: peaks at 0.06-0.26 (normal speech)
    # Noise floor 0.0005 cleanly separates silence from soft speech (20x headroom).
    noise_floor = 0.0005
    p95 = np.percentile(env, 95)
    if p95 < noise_floor:
        return np.zeros(T, dtype=np.float32)

    threshold = max(0.15 * p95, noise_floor)
    speech_prob = np.clip((env - threshold) / (p95 - threshold + 1e-8), 0.0, 1.0)

    # Enforce minimum bout duration: suppress short blips
    if min_bout_s > 0:
        min_samples = int(min_bout_s * fs)
        binary = speech_prob > 0.3
        changes = np.diff(binary.astype(np.int8), prepend=0, append=0)
        starts = np.where(changes == 1)[0]
        stops = np.where(changes == -1)[0]
        for s, e in zip(starts, stops):
            if (e - s) < min_samples:
                speech_prob[s:e] *= 0.2

    # Pad speech regions with margin
    if margin_s > 0:
        margin_samples = int(margin_s * fs)
        padded = gaussian_filter1d(speech_prob, sigma=margin_samples)
        speech_prob = np.maximum(speech_prob, padded)

    return np.clip(speech_prob, 0.0, 1.0).astype(np.float32)


def _detect_blinks(signal, fs=30.0, height=0.3, prominence=0.2,
                   min_iei_s=0.3, margin_s=0.15):
    """Detect blinks from eyeBlink AUs and return a binary mask.

    Uses bilateral average of eyeBlinkLeft (AU9) + eyeBlinkRight (AU10).
    When eyes are closed (mean eyeBlink > 0.5), raises the height threshold
    to avoid detecting fluctuations in the closed-eye baseline.

    35-42% of conversation facial events coincide with blinks (y_06 empirical).

    Args:
        signal: (T, 52) raw blendshapes [0, 1].
        fs: sampling rate (Hz).
        height: minimum peak height for blink detection.
        prominence: minimum peak prominence.
        min_iei_s: minimum inter-blink interval (seconds).
        margin_s: mask extends ± this many seconds around each blink peak.

    Returns:
        blink_mask: (T,) float32, 1.0 during blinks, 0.0 otherwise.
    """
    T = signal.shape[0]
    blink_avg = (signal[:, 9].astype(np.float64) + signal[:, 10].astype(np.float64)) / 2

    # When eyes are closed (baseline > 0.5), raise height threshold
    # to only detect genuine blinks above the elevated baseline
    baseline = np.median(blink_avg)
    effective_height = max(height, baseline + 0.15) if baseline > 0.5 else height

    pks, _ = find_peaks(blink_avg, height=effective_height,
                        distance=int(min_iei_s * fs),
                        prominence=prominence)

    mask = np.zeros(T, dtype=np.float32)
    half = int(margin_s * fs)
    for pk in pks:
        s = max(0, pk - half)
        e = min(T, pk + half + 1)
        mask[s:e] = 1.0

    return mask


def _smile_baseline(signal, fs=30.0, window_s=30.0):
    """Compute trailing median baseline for smile signal (AU44 + AU45).

    Uses a causal (trailing) window so the baseline at time t only depends
    on data before t. This prevents smile peaks from inflating their own
    baseline.

    Args:
        signal: (T, 52) raw blendshapes.
        fs: sampling rate.
        window_s: trailing window duration in seconds.

    Returns:
        baseline: (T,) trailing median of AU44+AU45.
    """
    smile = (signal[:, 44] + signal[:, 45]).astype(np.float64)
    T = len(smile)
    win = int(window_s * fs)

    # Causal trailing median: pad the signal by prepending `win` samples,
    # apply centered median_filter, then strip the padding.
    # This makes the result at time t depend only on t-win..t.
    from scipy.ndimage import median_filter
    padded = np.concatenate([np.full(win, smile[0]), smile])
    filt = median_filter(padded, size=win)
    baseline = filt[win:]  # strip padding to recover causal result

    return baseline.astype(np.float32)


def _smile_confidence(au_snapshot, speech_prob=0.0, phasic_amp=None):
    """Score confidence that this AU snapshot is a genuine smile.

    Uses phasic amplitude (deviation from trailing baseline) instead of
    absolute AU44+AU45 level. This eliminates tonic facial tone noise
    that produces false smiles in meditation and resting conditions.

    confidence = phasic_amplitude * dominance * bilaterality

    Returns (confidence, smile_composite, phasic_amp, dominance, bilaterality).
    """
    smile_l = au_snapshot[44]
    smile_r = au_snapshot[45]
    smile_comp = smile_l + smile_r

    # Use phasic amplitude if provided, otherwise fall back to absolute
    amp = phasic_amp if phasic_amp is not None else smile_comp

    # During speech, compute dominance over non-speech AUs only
    if speech_prob > 0.3:
        total_expr = sum(au_snapshot[i] for i in _NONSPEECH_EXPR_AUS) + 1e-6
    else:
        total_expr = sum(au_snapshot[i] for i in _EXPR_AUS) + 1e-6
    dominance = smile_comp / total_expr

    max_side = max(smile_l, smile_r, 1e-6)
    bilaterality = 1.0 - abs(smile_l - smile_r) / max_side

    confidence = max(amp, 0.0) * dominance * bilaterality

    if speech_prob > 0.3:
        confidence *= (1.0 + 0.3 * speech_prob)

    return confidence, smile_comp, dominance, bilaterality


def _detect_facial_events(signal, fs=30.0, smooth_s=0.1, prominence=0.03,
                          min_iei_s=1.0, lsl_start=0.0,
                          speech_mask=None, blink_mask=None):
    """Detect facial events via velocity norm on non-eye AUs.

    Subtractive gating: speech and blink masks suppress AU velocity in
    their respective AU groups before computing the L2 saliency norm.
    This removes known mechanical contamination (articulatory mouth
    movement, brow co-contraction) to reveal genuine expression events.

    Args:
        signal: (T, 52) raw blendshapes [0, 1].
        fs: sampling rate.
        smooth_s: smoothing on velocity norm.
        prominence: min prominence for peak detection.
        min_iei_s: min inter-event interval.
        lsl_start: LSL timestamp of segment start (for absolute timestamps).
        speech_mask: (T,) speech probability [0, 1], or None.
        blink_mask: (T,) blink mask [0, 1], or None.

    Returns:
        list of FacialEvent.
    """
    from cadence.significance.hawkes_coupling import MP_BLENDSHAPE_NAMES

    T = signal.shape[0]

    # Affect-only saliency: compute velocity norm on just the 10 AUs that
    # encode emotional expression (smile, frown, dimple, cheekSquint, noseSneer).
    vel = np.diff(signal[:, _AFFECT_AUS], axis=0, prepend=signal[:1, _AFFECT_AUS])
    saliency = np.sqrt((vel ** 2).sum(axis=1))

    if smooth_s > 0:
        saliency = gaussian_filter1d(saliency, sigma=smooth_s * fs)

    pks, _ = find_peaks(saliency, prominence=prominence,
                        distance=int(min_iei_s * fs))

    # Pre-compute phasic smile signals for the full timeseries
    smile_raw = (signal[:, 44] + signal[:, 45]).astype(np.float64)
    smile_base = _smile_baseline(signal, fs)
    smile_phasic_ts = (smile_raw - smile_base).astype(np.float32)
    smile_vel_ts = np.diff(smile_raw, prepend=smile_raw[0])
    smile_vel_ts = gaussian_filter1d(smile_vel_ts, sigma=0.1 * fs).astype(np.float32)

    half = int(0.25 * fs)
    events = []
    for pk in pks:
        s = max(0, pk - half)
        e = min(T, pk + half + 1)
        snapshot = signal[s:e].mean(axis=0)

        sp = float(speech_mask[pk]) if speech_mask is not None else 0.0
        bp = float(blink_mask[pk]) if blink_mask is not None else 0.0
        ph_amp = float(smile_phasic_ts[pk])
        ph_vel = float(smile_vel_ts[pk])

        conf, smile_comp, dom, bilat = _smile_confidence(
            snapshot, speech_prob=sp, phasic_amp=ph_amp)
        top = sorted([(i, MP_BLENDSHAPE_NAMES[i], float(snapshot[i]))
                      for i in _EXPR_AUS if snapshot[i] > 0.1],
                     key=lambda x: -x[2])[:5]

        events.append(FacialEvent(
            time=pk / fs,
            lsl_time=lsl_start + pk / fs,
            saliency=float(saliency[pk]),
            au_snapshot=snapshot,
            smile_composite=float(smile_comp),
            smile_confidence=float(conf),
            top_aus=top,
            speech_prob=sp,
            blink_prob=bp,
            smile_phasic=ph_amp,
            smile_velocity=ph_vel,
        ))

    return events


def _find_shared_events(events_a, events_b, max_lag_s=3.0,
                        smile_threshold=0.3):
    """Match co-occurring facial events between two people.

    Returns list of SharedEvent, sorted by joint_smile_confidence descending.
    """
    shared = []
    used_b = set()
    times_b = np.array([e.time for e in events_b]) if events_b else np.array([])

    for ev_a in events_a:
        if len(times_b) == 0:
            break
        diffs = times_b - ev_a.time
        in_window = np.where(np.abs(diffs) <= max_lag_s)[0]
        if len(in_window) == 0:
            continue

        for j in in_window[np.argsort(np.abs(diffs[in_window]))]:
            if j in used_b:
                continue
            used_b.add(j)
            ev_b = events_b[j]
            lag = ev_b.time - ev_a.time
            joint_conf = np.sqrt(max(ev_a.smile_confidence, 0) *
                                 max(ev_b.smile_confidence, 0))
            is_smile = (ev_a.smile_phasic > smile_threshold and
                        ev_b.smile_phasic > smile_threshold and
                        ev_a.smile_velocity > 0 and
                        ev_b.smile_velocity > 0)

            shared.append(SharedEvent(
                event_a=ev_a, event_b=ev_b,
                lag=lag,
                leader='A' if lag > 0 else 'B',
                joint_smile_confidence=float(joint_conf),
                is_shared_smile=is_smile,
            ))
            break

    shared.sort(key=lambda x: -x.joint_smile_confidence)
    return shared


def facial_event_catalog(p1_raw, p2_raw, fs,
                         lsl_start=0.0, segment_name='',
                         smooth_s=0.1, prominence=0.03,
                         min_iei_s=1.0, max_lag_s=3.0,
                         smile_threshold=0.1,
                         speech_gating=True, blink_gating=True):
    """Detect all facial events, match co-occurrences, score confidence.

    This is the V6 production function for BL coupling analysis.
    No significance testing — events are facts, confidence is about
    detection quality.

    Subtractive gating strategy:
      - speech_gating: suppress articulatory mouth-AU velocity during speech
      - blink_gating: suppress brow-AU velocity during blinks
    Together these remove known mechanical contamination, revealing
    genuine expression events (smiles, frowns, etc.).

    Args:
        p1_raw, p2_raw: (T, 52) raw blendshapes [0, 1].
        fs: sampling rate (Hz).
        lsl_start: LSL timestamp of segment start.
        segment_name: e.g., 'conv_1'.
        smooth_s: saliency smoothing.
        prominence: event detection prominence.
        min_iei_s: min inter-event interval.
        max_lag_s: co-occurrence matching window.
        smile_threshold: min smile composite for shared smile.
        speech_gating: if True, detect speech and suppress articulatory saliency.
        blink_gating: if True, detect blinks and suppress brow-AU saliency.

    Returns:
        FacialEventCatalog.
    """
    T = min(p1_raw.shape[0], p2_raw.shape[0])
    duration = T / fs

    # Speech detection (when enabled)
    speech_p1 = speech_p2 = None
    if speech_gating:
        speech_p1 = _detect_speech_from_blendshapes(p1_raw[:T], fs)
        speech_p2 = _detect_speech_from_blendshapes(p2_raw[:T], fs)

    # Blink detection (when enabled)
    blink_p1 = blink_p2 = None
    if blink_gating:
        blink_p1 = _detect_blinks(p1_raw[:T], fs)
        blink_p2 = _detect_blinks(p2_raw[:T], fs)

    events_p1 = _detect_facial_events(p1_raw[:T], fs, smooth_s, prominence,
                                       min_iei_s, lsl_start,
                                       speech_mask=speech_p1,
                                       blink_mask=blink_p1)
    events_p2 = _detect_facial_events(p2_raw[:T], fs, smooth_s, prominence,
                                       min_iei_s, lsl_start,
                                       speech_mask=speech_p2,
                                       blink_mask=blink_p2)

    shared = _find_shared_events(events_p1, events_p2, max_lag_s,
                                  smile_threshold)
    shared_smiles = [s for s in shared if s.is_shared_smile]

    return FacialEventCatalog(
        events_p1=events_p1,
        events_p2=events_p2,
        shared_events=shared,
        shared_smiles=shared_smiles,
        n_events_p1=len(events_p1),
        n_events_p2=len(events_p2),
        n_shared=len(shared),
        n_shared_smiles=len(shared_smiles),
        duration_s=duration,
        segment_name=segment_name,
        lsl_start=lsl_start,
        speech_p1=speech_p1,
        speech_p2=speech_p2,
        blink_p1=blink_p1,
        blink_p2=blink_p2,
    )
