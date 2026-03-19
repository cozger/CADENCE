"""CouplingEstimator: orchestrates the full CADENCE analysis pipeline.

V2 Pipeline:
  Stage 1: Group lasso discovery (all pathways, full feature set).
  Stage 2: EWLS estimation on selected features + moderation + nonlinear.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import sys
import time as _time

import numpy as np
import torch


def _log(msg):
    """Unbuffered progress logging."""
    print(msg, flush=True)

from cadence.config import load_config
from cadence.constants import (
    MODALITY_ORDER_V2, MODALITY_SPECS_V2, INTERBRAIN_MODALITY,
)
from cadence.basis.raised_cosine import raised_cosine_basis, multi_band_basis
from cadence.basis.design_matrix import DesignMatrixBuilder
from cadence.regression.ewls import EWLSSolver
from cadence.regression.ridge import (
    batched_ridge, batched_ridge_multi,
    batched_ridge_per_target, batched_ridge_multi_per_ch,
)
from cadence.regression.ftest import f_test_timecourse, f_test_static
from cadence.regression.group_lasso import GroupLassoSolver
from cadence.surrogates import circular_shift_surrogate_batched
from cadence.coupling.pathways import (
    get_pathway_n_predictors,
    get_modality_pathways_v2, get_pathway_category, get_feature_groups_v2,
)


def _max_cluster_mass(stat_array, threshold):
    """Compute max cluster mass from a 1D statistic array.

    Clusters are contiguous runs of positions where stat > threshold.
    Cluster mass is the sum of (stat - threshold) within each cluster.
    Returns the maximum cluster mass, or 0.0 if no clusters found.
    """
    above = stat_array > threshold
    if not above.any():
        return 0.0
    diff = np.diff(above.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    max_mass = 0.0
    for s, e in zip(starts, ends):
        mass = np.sum(stat_array[s:e] - threshold)
        if mass > max_mass:
            max_mass = mass
    return max_mass


@dataclass
class CouplingResult:
    """Result container for one direction of coupling analysis."""

    direction: str  # e.g. 'p1_to_p2' or 'therapist_to_patient'
    times: np.ndarray  # (T_eval,) seconds

    # Phase 1: modality-level
    pathway_dr2: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    pathway_r2_full: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    pathway_r2_restricted: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    pathway_kernels: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    pathway_pvalues: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    pathway_significant: Dict[Tuple[str, str], bool] = field(default_factory=dict)
    pathway_f_stat: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)

    # Phase 2: feature-level (populated only for significant pathways)
    feature_dr2: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    feature_kernels: Dict = field(default_factory=dict)

    # Per-pathway time grids (when per-modality eval rates differ)
    pathway_times: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)

    # Per-feature dr2 decomposition: pathway -> {feat_idx: dr2_timecourse}
    pathway_feature_dr2: Dict[Tuple[str, str], Dict[int, np.ndarray]] = field(default_factory=dict)

    # Per-feature per-timepoint p-values: pathway -> {feat_idx: (T,) pvalues}
    pathway_feature_pvalues: Dict[Tuple[str, str], Dict[int, np.ndarray]] = field(default_factory=dict)

    # Source x Target feature decomposition: pathway -> {(src_feat_idx, tgt_ch_idx): (T,) dr2}
    # Only top-N pairs stored to keep NPZ manageable.
    pathway_src_tgt_dr2: Dict[Tuple[str, str], Dict[Tuple[int, int], np.ndarray]] = field(default_factory=dict)

    # Auxiliary data for interpretation (e.g., PCA loadings)
    aux: Dict[str, np.ndarray] = field(default_factory=dict)

    # Surrogate null statistics per pathway (for HMM detection)
    pathway_null_stats: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)

    # HMM coupling posterior per pathway
    pathway_coupling_posterior: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)

    # Summary statistics
    overall_dr2: Optional[np.ndarray] = None  # average across pathways
    n_significant_pathways: int = 0

    # Stage 1 discovery result (populated in V2 pipeline)
    discovery: Optional[object] = None


class CouplingEstimator:
    """Main CADENCE analysis engine.

    Usage:
        estimator = CouplingEstimator(config)
        result = estimator.analyze_session(session, 'p1_to_p2')
    """

    def __init__(self, config=None):
        if config is None:
            config = load_config()
        self.config = config
        self.device = config.get('device', 'cuda')

        # Build default basis functions (Layer 1)
        basis_cfg = config['basis']['layer1']
        ewls_cfg = config['ewls']
        self.basis, self.lag_times = raised_cosine_basis(
            n_basis=basis_cfg['n_basis'],
            max_lag_s=basis_cfg['max_lag_seconds'],
            min_lag_s=basis_cfg.get('min_lag_seconds', 0.0),
            sample_rate=ewls_cfg['eval_rate'],
            log_spacing=basis_cfg.get('log_spacing', True),
        )
        self.n_basis = basis_cfg['n_basis']

        # Design matrix builder (default, single-band)
        ar_order = config['autoregressive']['order'] if config['autoregressive']['include'] else 0
        self.ar_order = ar_order
        self.dm_builder = DesignMatrixBuilder(
            self.basis, ar_order=ar_order, device=self.device)

        # Per-modality eval rate overrides (output resolution)
        self.eval_rate_overrides = config.get('eval_rate_overrides', {})

        # Per-modality basis and design matrix builders
        # Basis sample_rate must match internal_rate so conv1d lag coverage is correct.
        # Source signals are pre-resampled to internal_rate before convolution.
        self._mod_dm_builders = {}
        self._mod_basis = {}
        self._mod_n_basis = {}

        lag_bands_cfg = config.get('lag_bands', {})
        # Multi-band basis builders (e.g., ECG with short+long lag bands)
        for mod, bands in lag_bands_cfg.items():
            internal_rate = max(
                self.eval_rate_overrides.get(mod, ewls_cfg['eval_rate']),
                ewls_cfg['eval_rate'])
            mb, ml, slices = multi_band_basis(bands, sample_rate=internal_rate)
            self._mod_basis[mod] = mb
            self._mod_n_basis[mod] = mb.shape[1]
            self._mod_dm_builders[mod] = DesignMatrixBuilder(
                mb, ar_order=ar_order, device=self.device)

        # Single-band basis for modalities with rate overrides (not already multi-band)
        for mod, desired_rate in self.eval_rate_overrides.items():
            if mod in self._mod_dm_builders:
                continue  # already has multi-band builder
            internal_rate = max(desired_rate, ewls_cfg['eval_rate'])
            if internal_rate != ewls_cfg['eval_rate']:
                mod_basis, _ = raised_cosine_basis(
                    n_basis=basis_cfg['n_basis'],
                    max_lag_s=basis_cfg['max_lag_seconds'],
                    min_lag_s=basis_cfg.get('min_lag_seconds', 0.0),
                    sample_rate=internal_rate,
                    log_spacing=basis_cfg.get('log_spacing', True),
                )
                self._mod_basis[mod] = mod_basis
                self._mod_n_basis[mod] = basis_cfg['n_basis']
                self._mod_dm_builders[mod] = DesignMatrixBuilder(
                    mod_basis, ar_order=ar_order, device=self.device)

        # EWLS solver (default tau)
        self.solver = EWLSSolver(
            tau_seconds=ewls_cfg['tau_seconds'],
            lambda_ridge=ewls_cfg['lambda_ridge'],
            eval_rate=ewls_cfg['eval_rate'],
            device=self.device,
            min_effective_n=ewls_cfg.get('min_effective_n', 20),
        )
        self.eval_rate = ewls_cfg['eval_rate']

        # Per-modality EWLS solvers (only tau overrides — eval_rate always default).
        # EWLS scan always runs at default eval_rate (2Hz). This is optimal because
        # tau >> 1/rate (e.g., tau=30s at 2Hz = 60 samples/time-constant), so higher
        # rates just add Python loop overhead with no benefit to the estimates.
        # Conv1d runs at the higher rate for correct lag coverage, then the design
        # matrix is downsampled to eval_rate before EWLS.
        self._mod_solvers = {}
        tau_overrides = config.get('ewls_tau_overrides', {})
        for mod, tau in tau_overrides.items():
            if tau != ewls_cfg['tau_seconds']:
                self._mod_solvers[mod] = EWLSSolver(
                    tau_seconds=tau,
                    lambda_ridge=ewls_cfg['lambda_ridge'],
                    eval_rate=ewls_cfg['eval_rate'],
                    device=self.device,
                    min_effective_n=ewls_cfg.get('min_effective_n', 20),
                )

    def analyze_session(self, session, direction='p1_to_p2'):
        """Run full CADENCE analysis on one session for one direction.

        Args:
            session: Session dict with modality data, timestamps, validity.
            direction: 'p1_to_p2' or 'p2_to_p1'.

        Returns:
            CouplingResult with all pathway analyses.
        """
        return self._analyze_session_v2(session, direction)

    def analyze_session_both(self, session):
        """Run both directions concurrently on separate CUDA streams.

        Uses threading + CUDA streams so GPU work from both directions
        overlaps. The GIL is released during CUDA kernel execution,
        allowing true GPU parallelism between threads.

        Returns:
            dict: {'p1_to_p2': CouplingResult, 'p2_to_p1': CouplingResult}
        """
        import threading

        device = self.device
        use_streams = (isinstance(device, str) and 'cuda' in device) or \
                      (hasattr(device, 'type') and device.type == 'cuda')

        results = {}
        errors = {}

        def _run_direction(direction, stream=None):
            try:
                if stream is not None:
                    with torch.cuda.stream(stream):
                        results[direction] = self.analyze_session(
                            session, direction)
                else:
                    results[direction] = self.analyze_session(
                        session, direction)
            except Exception as e:
                errors[direction] = e

        if use_streams:
            stream1 = torch.cuda.Stream(device=device)
            stream2 = torch.cuda.Stream(device=device)

            t1 = threading.Thread(
                target=_run_direction, args=('p1_to_p2', stream1))
            t2 = threading.Thread(
                target=_run_direction, args=('p2_to_p1', stream2))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            # Sync streams
            stream1.synchronize()
            stream2.synchronize()
        else:
            # CPU fallback: sequential
            _run_direction('p1_to_p2')
            _run_direction('p2_to_p1')

        for d, e in errors.items():
            raise RuntimeError(f"Direction {d} failed: {e}")

        return results

    def _analyze_session_v2(self, session, direction):
        """V2 pipeline: group lasso discovery + EWLS on selected features."""
        from cadence.coupling.discovery import DiscoveryResult

        if direction == 'p1_to_p2':
            src_p, tgt_p = 'p1', 'p2'
        elif direction == 'p2_to_p1':
            src_p, tgt_p = 'p2', 'p1'
        else:
            raise ValueError(f"Unknown direction: {direction}")

        # Extract v2 signals
        source_sigs = self._extract_signals_v2(session, src_p)
        target_sigs = self._extract_signals_v2(session, tgt_p)

        # Add inter-brain signals (source-only, not tied to either participant)
        ib_key = INTERBRAIN_MODALITY
        for key in [f'{ib_key}', f'{ib_key}_ts', f'{ib_key}_valid']:
            pass  # inter-brain stored without participant prefix
        if ib_key in session:
            ib_signal = session[ib_key].copy()
            ib_ts = session[f'{ib_key}_ts']
            ib_valid = session.get(f'{ib_key}_valid',
                                    np.ones(len(ib_signal), dtype=bool))

            # Delta floor: zero out interbrain channels below min_freq_hz
            ib_min_freq = self.config.get('interbrain', {}).get(
                'min_freq_hz', 0.0)
            if ib_min_freq > 0:
                from cadence.constants import (
                    WAVELET_CENTER_FREQS, EEG_WAVELET_N_FREQS,
                    EEG_WAVELET_N_ROIS,
                )
                n_freqs = EEG_WAVELET_N_FREQS
                n_rois = EEG_WAVELET_N_ROIS
                n_zeroed = 0
                for comp_idx in range(2):  # cos, sin
                    for f_idx in range(n_freqs):
                        if WAVELET_CENTER_FREQS[f_idx] < ib_min_freq:
                            for r_idx in range(n_rois):
                                col = comp_idx * (n_freqs * n_rois) + \
                                      f_idx * n_rois + r_idx
                                if col < ib_signal.shape[1]:
                                    ib_signal[:, col] = 0.0
                                    n_zeroed += 1
                if n_zeroed > 0:
                    _log(f"[{direction}] Interbrain: zeroed {n_zeroed}/160 "
                         f"channels below {ib_min_freq} Hz")

            if ib_valid.sum() >= 10:
                source_sigs[ib_key] = (ib_signal, ib_ts, ib_valid)

        duration = session.get('duration', 0)
        eval_rate = self.config.get('wavelet', {}).get('output_hz', 5.0)
        eval_times = np.arange(0, duration, 1.0 / eval_rate)

        result = CouplingResult(direction=direction, times=eval_times)

        # Initialize all testable pathways as not significant so that
        # FPR denominators include pathways correctly rejected at Stage 1.
        for key in get_modality_pathways_v2():
            src_mod, tgt_mod = key
            if src_mod in source_sigs and tgt_mod in target_sigs:
                result.pathway_significant.setdefault(key, False)

        # Stage 1: Group lasso discovery
        _log(f"[{direction}] Stage 1: Discovery (group lasso)...")
        _t0 = _time.perf_counter()
        discovery = self._stage1_discovery(
            source_sigs, target_sigs, duration, eval_rate)
        result.discovery = discovery
        _log(f"[{direction}] Stage 1 done in {_time.perf_counter()-_t0:.1f}s")

        # Stage 1.5: Surrogate significance screen
        _log(f"[{direction}] Stage 1.5: Surrogate screening...")
        _t0 = _time.perf_counter()
        self._stage1_5_surrogate_screen(
            source_sigs, target_sigs, duration, discovery, result, eval_rate)
        _log(f"[{direction}] Stage 1.5 done in {_time.perf_counter()-_t0:.1f}s")

        # Stage 2: EWLS on surviving pathways only
        _log(f"[{direction}] Stage 2: EWLS estimation...")
        _t0 = _time.perf_counter()
        self._stage2_estimate(
            source_sigs, target_sigs, duration, discovery, result, eval_rate)
        _log(f"[{direction}] Stage 2 done in {_time.perf_counter()-_t0:.1f}s")

        # Post-Stage-2 filter: pathways with clearly negative mean dr2
        # are not genuine coupling (source predictions worse than null).
        # Use -min_dr2 as tolerance: diluted coupling across many channels
        # can produce slightly negative aggregate dr2 even when real.
        sig_cfg = self.config['significance']
        min_dr2 = sig_cfg.get('session_level', {}).get('min_dr2', 0.001)
        for key in list(result.pathway_significant.keys()):
            if result.pathway_significant[key] and key in result.pathway_dr2:
                mean_dr2_val = float(np.nanmean(result.pathway_dr2[key]))
                # Structural confound: interbrain source contains target EEG
                # information (PLV computed from both participants). Require
                # positive dr2 to confirm the source adds NEW predictive
                # power beyond structural prediction.
                src_mod, tgt_mod = key
                if (src_mod == INTERBRAIN_MODALITY
                        and 'eeg' in tgt_mod):
                    threshold = 0.0
                else:
                    threshold = -min_dr2
                if mean_dr2_val < threshold:
                    result.pathway_significant[key] = False
                    result.n_significant_pathways -= 1

        # Post-Stage-2 HMM: final arbiter for pathway_significant
        from cadence.significance.detection import detect_coupling_hmm
        hmm_cfg = sig_cfg.get('session_level', {}).get('hmm', {})
        hmm_cfg['min_dr2'] = min_dr2
        eval_rate_overrides = self.config.get('eval_rate_overrides', {})

        for key in result.pathway_dr2:
            if key not in result.pathway_pvalues:
                continue
            dr2_arr = result.pathway_dr2[key]
            null_stats = result.pathway_null_stats.get(key)
            pw_eval_rate = eval_rate_overrides.get(key[0], eval_rate)
            detected, details = detect_coupling_hmm(
                dr2_arr, null_stats, pw_eval_rate, hmm_cfg)
            result.pathway_significant[key] = detected
            if 'coupling_posterior' in details:
                result.pathway_coupling_posterior[key] = details['coupling_posterior']
        result.n_significant_pathways = sum(result.pathway_significant.values())

        # Overall dR2
        if result.pathway_dr2:
            dr2_stack = []
            for key, dr2 in result.pathway_dr2.items():
                dr2_clean = np.nan_to_num(dr2, nan=0.0)
                pw_times = result.pathway_times.get(key, eval_times)
                if len(dr2_clean) != len(eval_times):
                    dr2_clean = np.interp(eval_times, pw_times, dr2_clean)
                dr2_stack.append(dr2_clean)
            if dr2_stack:
                result.overall_dr2 = np.mean(dr2_stack, axis=0)

        # Store PCA loadings for blendshapes interpretation
        for p in [src_p, tgt_p]:
            loadings_key = f'{p}_blendshapes_v2_pca_loadings'
            if loadings_key in session:
                result.aux['bl_pca_loadings'] = session[loadings_key]
                break  # same PCA for both participants

        return result

    def _get_dm_builder(self, src_mod):
        """Get the design matrix builder for a source modality.

        Returns the multi-band builder if the modality has lag_bands configured,
        otherwise the default single-band builder.
        """
        return self._mod_dm_builders.get(src_mod, self.dm_builder)

    def _get_n_basis(self, src_mod):
        """Get total number of basis functions for a source modality."""
        return self._mod_n_basis.get(src_mod, self.n_basis)

    def _get_basis(self, src_mod):
        """Get the basis matrix for a source modality."""
        return self._mod_basis.get(src_mod, self.basis)

    def _get_solver(self, src_mod):
        """Get the EWLS solver for a source modality.

        Returns the per-modality solver if a tau or eval_rate override is configured,
        otherwise the default solver.
        """
        return self._mod_solvers.get(src_mod, self.solver)

    def _get_internal_rate(self, src_mod):
        """Get the internal EWLS computation rate (always >= default)."""
        desired = self.eval_rate_overrides.get(src_mod, self.eval_rate)
        return max(desired, self.eval_rate)

    def _get_output_rate(self, src_mod):
        """Get the desired output eval rate for a source modality."""
        return self.eval_rate_overrides.get(src_mod, self.eval_rate)

    def _get_internal_times(self, src_mod, duration):
        """Get internal eval time grid (for EWLS computation)."""
        rate = self._get_internal_rate(src_mod)
        return np.arange(0, duration, 1.0 / rate)

    def _get_output_times(self, src_mod, duration):
        """Get output eval time grid (may be coarser than internal)."""
        rate = self._get_output_rate(src_mod)
        return np.arange(0, duration, 1.0 / rate)

    def _resample_to_output(self, data, internal_times, output_times):
        """Resample EWLS output from internal rate to output rate.

        Uses linear interpolation (fine for smooth dR2 timecourses).
        Handles both 1D (T,) and 2D (T, C) arrays.
        """
        if len(internal_times) == len(output_times):
            return data
        data_clean = np.nan_to_num(data, nan=0.0)
        if data_clean.ndim == 1:
            return np.interp(output_times, internal_times, data_clean)
        # 2D: resample each column independently
        n_cols = data_clean.shape[1]
        out = np.zeros((len(output_times), n_cols), dtype=data_clean.dtype)
        for c in range(n_cols):
            out[:, c] = np.interp(output_times, internal_times, data_clean[:, c])
        return out

    def _resample_source_to_internal(self, signal, src_ts, src_valid, internal_times):
        """Pre-resample source signal to internal_rate for correct basis lag coverage.

        The basis conv1d kernel must operate at internal_rate to ensure the
        number of kernel samples corresponds to the intended lag range.
        """
        n_out = len(internal_times)
        n_ch = signal.shape[1]
        resampled = np.zeros((n_out, n_ch), dtype=np.float32)
        for c in range(n_ch):
            resampled[:, c] = np.interp(internal_times, src_ts, signal[:, c])
        # Propagate validity via nearest-neighbor
        idx = np.clip(np.searchsorted(src_ts, internal_times), 0, len(src_valid) - 1)
        valid_out = src_valid[idx]
        return resampled, internal_times, valid_out

    def _reconstruct_kernel(self, basis_coeffs, n_source_channels, basis=None):
        """Reconstruct coupling kernel h(s) from basis coefficients.

        The kernel is the impulse response: h(s) = sum_j alpha_j * phi_j(s)

        Args:
            basis_coeffs: (T, n_basis * n_src, C_tgt) coefficients.
            n_source_channels: Number of source channels.
            basis: Optional basis matrix override (for multi-band pathways).

        Returns:
            kernel: (T, n_lags) average kernel across source and target channels.
        """
        if basis is None:
            basis = self.basis
        T = basis_coeffs.shape[0]
        n_lags, n_basis = basis.shape

        # Reshape to (T, n_src, n_basis, C_tgt)
        C_tgt = basis_coeffs.shape[2]
        coeffs_reshaped = basis_coeffs.reshape(T, n_source_channels, n_basis, C_tgt)

        # Average across source channels and target channels
        coeffs_avg = coeffs_reshaped.mean(axis=(1, 3))  # (T, n_basis)

        # Reconstruct kernel: h(s) = sum_j alpha_j * phi_j(s)
        # basis is (n_lags, n_basis)
        kernel = coeffs_avg @ basis.T  # (T, n_lags)

        return kernel

    # ==================================================================
    # V2 Pipeline: Group Lasso Discovery + EWLS Estimation
    # ==================================================================

    def _extract_signals_v2(self, session, participant):
        """Extract v2 modality signals for one participant.

        Returns:
            dict[mod_name] -> (signal, timestamps, valid)
        """
        signals = {}
        for mod in MODALITY_ORDER_V2:
            data_key = f'{participant}_{mod}'
            ts_key = f'{participant}_{mod}_ts'
            valid_key = f'{participant}_{mod}_valid'

            if data_key not in session:
                continue

            signal = session[data_key]
            ts = session[ts_key]
            valid = session.get(valid_key, np.ones(len(signal), dtype=bool))

            if valid.sum() < 10:
                continue

            signals[mod] = (signal, ts, valid)

        return signals

    @staticmethod
    def _detrend_block_target(y_block):
        """Remove polynomial time confounds [1, t, t²] from y_block.

        Unconditionally projects out intercept, linear, and quadratic
        time components from the target.  Called only when the session-
        level trend check has confirmed a genuine polynomial drift.

        Args:
            y_block: (T_block, C) target block

        Returns:
            y_clean: (T_block, C) detrended target block
        """
        T = y_block.shape[0]
        dev = y_block.device
        dtype = y_block.dtype

        t = torch.linspace(0, 1, T, device=dev, dtype=dtype)
        C = torch.stack([
            torch.ones(T, device=dev, dtype=dtype),
            t,
            t * t,
        ], dim=1)  # (T, 3)

        CtC = C.T @ C
        Cy = C.T @ y_block
        return y_block - C @ torch.linalg.solve(CtC, Cy)

    def _get_pathway_basis_v2(self, src_mod, tgt_mod, eval_rate):
        """Build pathway-specific basis for v2 (based on pathway category).

        Returns:
            basis: (n_lag_samples, n_basis) array
            n_basis: int
        """
        category = get_pathway_category(src_mod, tgt_mod, self.config)
        temporal_cfg = self.config.get('pathway_temporal', {}).get(
            category, {'max_lag_seconds': 18.0, 'n_basis': 10})

        max_lag = temporal_cfg['max_lag_seconds']
        n_basis = temporal_cfg['n_basis']

        basis, _ = raised_cosine_basis(
            n_basis=n_basis,
            max_lag_s=max_lag,
            min_lag_s=0.0,
            sample_rate=eval_rate,
            log_spacing=True,
        )
        return basis, n_basis

    def _ar_whiten_target(self, X_ar, y, valid, C_tgt, lam):
        """AR-whiten all target channels independently via batched ridge.

        Fits per-channel AR(p) model and returns residuals. Whitening
        reduces the cross-covariance noise floor from autocorrelation-
        inflated ~30/T back to ~1/T (crucial for Lorenz-derived signals
        with lag-1 autocorrelation ~0.95).

        Returns:
            y_resid: (T_v, C_tgt) AR residuals on valid samples
        """
        ar_order = self.ar_order

        if valid is not None:
            X_ar_v = X_ar[valid]
            y_v = y[valid]
        else:
            X_ar_v = X_ar
            y_v = y

        T_v = X_ar_v.shape[0]
        dev = X_ar_v.device
        dtype = X_ar_v.dtype

        # Build per-channel AR design: (C_tgt, T_v, ar_order)
        X_ar_per_ch = torch.zeros(
            T_v, C_tgt, ar_order, device=dev, dtype=dtype)
        for lag_idx in range(ar_order):
            X_ar_per_ch[:, :, lag_idx] = X_ar_v[
                :, lag_idx * C_tgt : (lag_idx + 1) * C_tgt]
        X_ar_batch = X_ar_per_ch.permute(1, 0, 2)  # (C_tgt, T_v, ar)

        y_batch = y_v.T.unsqueeze(2)  # (C_tgt, T_v, 1)

        XtX = torch.bmm(X_ar_batch.transpose(1, 2), X_ar_batch)
        Xty = torch.bmm(X_ar_batch.transpose(1, 2), y_batch)
        reg = lam * torch.eye(ar_order, device=dev, dtype=dtype)
        beta_ar = torch.linalg.solve(XtX + reg, Xty)
        y_hat_ar = torch.bmm(X_ar_batch, beta_ar)
        y_resid = (y_batch - y_hat_ar).squeeze(2)  # (C_tgt, T_v)

        return y_resid.T  # (T_v, C_tgt)

    def _compute_matched_cc_sq(self, X_source, X_ar, y, valid,
                               n_basis, n_feat, C_tgt, lam):
        """Compute matched cross-cov norms with AR-whitened target.

        Used by Stage 1 for feature ranking/selection (not significance).
        For significance testing, Stage 1.5 uses SVD instead.

        Returns:
            cc_sq: (C_min,) squared cross-cov norms per matched feature
            y_resid_mat: (T_v, C_min) AR-whitened target
        """
        C_min = min(n_feat, C_tgt)
        ar_order = self.ar_order

        if valid is not None:
            X_src_v = X_source[valid]
            X_ar_v = X_ar[valid]
            y_v = y[valid]
        else:
            X_src_v = X_source
            X_ar_v = X_ar
            y_v = y

        T_v = X_src_v.shape[0]
        dev = X_src_v.device
        dtype = X_src_v.dtype

        # Build per-channel AR batch: (C_min, T_v, ar_order)
        X_ar_per_ch = torch.zeros(
            T_v, C_min, ar_order, device=dev, dtype=dtype)
        for lag_idx in range(ar_order):
            X_ar_per_ch[:, :, lag_idx] = X_ar_v[
                :, lag_idx * C_tgt : lag_idx * C_tgt + C_min]
        X_ar_batch = X_ar_per_ch.permute(1, 0, 2)  # (C_min, T_v, ar)

        # Target batch: (C_min, T_v, 1)
        y_batch = y_v[:, :C_min].T.unsqueeze(2)

        # Per-channel AR fit and residualize
        XtX = torch.bmm(X_ar_batch.transpose(1, 2), X_ar_batch)
        Xty = torch.bmm(X_ar_batch.transpose(1, 2), y_batch)
        reg = lam * torch.eye(ar_order, device=dev, dtype=dtype)
        beta_ar = torch.linalg.solve(XtX + reg, Xty)
        y_hat_ar = torch.bmm(X_ar_batch, beta_ar)  # (C_min, T_v, 1)
        y_resid = (y_batch - y_hat_ar).squeeze(2)  # (C_min, T_v)
        y_resid_mat = y_resid.T  # (T_v, C_min)

        # Matched cross-cov: source[i]_basis.T @ y_resid[i]
        cross_cov = X_src_v[:, :C_min * n_basis].T @ y_resid_mat / T_v
        cc_3d = cross_cov.view(C_min, n_basis, C_min)
        idx = torch.arange(C_min, device=dev)
        cc_matched = cc_3d[idx, :, idx]  # (C_min, B)

        cc_sq = (cc_matched ** 2).sum(dim=1)  # (C_min,)
        return cc_sq, y_resid_mat, cc_matched

    # ==================================================================
    # Doubly-Sparse Detection Methods (Fixes 1-5)
    # ==================================================================

    def _pregroup_features(self, src_signal, src_mod, src_valid=None):
        """Pre-group features by modality-appropriate semantic structure.

        For EEG wavelet/interbrain: merge adjacent frequency bins with high
        correlation within each (component, ROI) pair.

        For pose_features: group by body segment (head, arms, torso, legs,
        global) using POSE_SEGMENT_MAP.

        For blendshapes_v2: pair each PCA component with its temporal
        derivative (PC_i with PC_i_dt).

        For other modalities: each feature is its own cluster (identity map).

        Returns:
            grouped_signal: (T, n_clusters) averaged signal
            cluster_map: dict {cluster_idx: [original_feature_indices]}
        """
        from cadence.constants import (
            EEG_WAVELET_N_COMPONENTS, EEG_WAVELET_N_FREQS, EEG_WAVELET_N_ROIS,
        )

        n_ch = src_signal.shape[1]
        threshold = self.config.get('doubly_sparse', {}).get(
            'pregroup', {}).get('correlation_threshold', 0.8)

        # Use valid samples for correlation computation
        if src_valid is not None:
            sig_valid = src_signal[src_valid]
        else:
            sig_valid = src_signal

        # --- Pose: hybrid anatomical + correlation-based grouping (Fix 3) ---
        # Use POSE_SEGMENT_MAP as constraint boundaries (head channels don't
        # merge with arm channels). Within each segment, apply correlation-
        # based merging for finer resolution.
        if src_mod == 'pose_features':
            from cadence.constants import POSE_SEGMENT_MAP
            cluster_map = {}
            c_idx = 0
            for seg_name, (seg_start, seg_end) in POSE_SEGMENT_MAP.items():
                seg_end = min(seg_end, n_ch)
                if seg_start >= n_ch:
                    continue
                seg_indices = list(range(seg_start, seg_end))
                if len(seg_indices) <= 1:
                    cluster_map[c_idx] = seg_indices
                    c_idx += 1
                    continue
                # Correlation-based merging within segment
                seg_signals = sig_valid[:, seg_indices]
                std = seg_signals.std(axis=0)
                const_mask = std < 1e-10
                if const_mask.all():
                    cluster_map[c_idx] = seg_indices
                    c_idx += 1
                    continue
                corr = np.corrcoef(seg_signals.T)
                corr = np.nan_to_num(corr, nan=0.0)
                # Greedy merge adjacent channels with |rho| > threshold
                groups = []
                current_group = [0]
                for j in range(1, len(seg_indices)):
                    if abs(corr[current_group[-1], j]) > threshold:
                        current_group.append(j)
                    else:
                        groups.append(current_group)
                        current_group = [j]
                groups.append(current_group)
                for group in groups:
                    cluster_map[c_idx] = [seg_indices[j] for j in group]
                    c_idx += 1
            n_clusters = len(cluster_map)
            grouped = np.zeros(
                (src_signal.shape[0], n_clusters), dtype=np.float32)
            for ci, orig_indices in cluster_map.items():
                grouped[:, ci] = src_signal[:, orig_indices].mean(axis=1)
            return grouped, cluster_map

        # --- Blendshapes v2: correlation-based merging (Fix 3) ---
        # Channels: 0-14 = PCA components, 15-29 = derivatives, 30 = activity.
        # Replace fixed triplets with correlation-based merging of PCA channels.
        # Mirror the same group structure for derivative channels.
        # In synthetic data (uncorrelated PCA), each becomes its own group →
        # maximum resolution. In real data, correlated components merge for SNR.
        if src_mod == 'blendshapes_v2':
            n_pca = 15
            cluster_map = {}
            c_idx = 0

            # Correlation-based merging of PCA channels (0-14)
            pca_signals = sig_valid[:, :n_pca]
            std = pca_signals.std(axis=0)
            const_mask = std < 1e-10
            if const_mask.all():
                pca_groups = [list(range(n_pca))]
            else:
                corr = np.corrcoef(pca_signals.T)
                corr = np.nan_to_num(corr, nan=0.0)
                pca_groups = []
                current_group = [0]
                for j in range(1, n_pca):
                    if abs(corr[current_group[-1], j]) > threshold:
                        current_group.append(j)
                    else:
                        pca_groups.append(current_group)
                        current_group = [j]
                pca_groups.append(current_group)

            # PCA groups
            for group in pca_groups:
                cluster_map[c_idx] = group
                c_idx += 1
            # Mirror derivative groups (same structure, offset by n_pca)
            for group in pca_groups:
                deriv_indices = [j + n_pca for j in group
                                 if j + n_pca < n_ch]
                if deriv_indices:
                    cluster_map[c_idx] = deriv_indices
                    c_idx += 1
            # Activity channel (index 30 if present)
            if n_ch > 2 * n_pca:
                cluster_map[c_idx] = [2 * n_pca]
                c_idx += 1

            n_clusters = len(cluster_map)
            grouped = np.zeros(
                (src_signal.shape[0], n_clusters), dtype=np.float32)
            for ci, orig_indices in cluster_map.items():
                grouped[:, ci] = src_signal[:, orig_indices].mean(axis=1)
            return grouped, cluster_map

        # --- Other non-EEG modalities: identity map ---
        if src_mod not in ('eeg_wavelet', 'eeg_interbrain'):
            cluster_map = {i: [i] for i in range(n_ch)}
            return src_signal.copy(), cluster_map

        n_comp = EEG_WAVELET_N_COMPONENTS
        n_freq = EEG_WAVELET_N_FREQS
        n_roi = EEG_WAVELET_N_ROIS
        threshold = self.config.get('doubly_sparse', {}).get(
            'pregroup', {}).get('correlation_threshold', 0.8)

        # Use valid samples for correlation computation
        if src_valid is not None:
            sig_valid = src_signal[src_valid]
        else:
            sig_valid = src_signal

        cluster_map = {}
        cluster_idx = 0

        for comp in range(n_comp):
            for roi in range(n_roi):
                # Feature indices for this (comp, roi): 20 frequency bins
                feat_indices = []
                for freq in range(n_freq):
                    col = comp * (n_freq * n_roi) + freq * n_roi + roi
                    feat_indices.append(col)

                # Correlation between adjacent frequency bins
                signals = sig_valid[:, feat_indices]
                # Handle constant signals
                std = signals.std(axis=0)
                const_mask = std < 1e-10
                if const_mask.all():
                    # All constant — one big cluster
                    cluster_map[cluster_idx] = feat_indices
                    cluster_idx += 1
                    continue

                corr = np.corrcoef(signals.T)
                corr = np.nan_to_num(corr, nan=0.0)

                # Greedy merge: group adjacent bins with |rho| > threshold
                groups = []
                current_group = [0]
                for f in range(1, n_freq):
                    if abs(corr[current_group[-1], f]) > threshold:
                        current_group.append(f)
                    else:
                        groups.append(current_group)
                        current_group = [f]
                groups.append(current_group)

                for group in groups:
                    orig_indices = [feat_indices[f] for f in group]
                    cluster_map[cluster_idx] = orig_indices
                    cluster_idx += 1

        # Build clustered signal: average features within each cluster
        n_clusters = len(cluster_map)
        grouped = np.zeros((src_signal.shape[0], n_clusters), dtype=np.float32)
        for c_idx, orig_indices in cluster_map.items():
            grouped[:, c_idx] = src_signal[:, orig_indices].mean(axis=1)

        return grouped, cluster_map

    def _univariate_prescreen(self, X_source, X_ar, y, valid,
                               n_basis, n_feat, C_tgt, lam, max_k=200):
        """Sure Independence Screening: keep top K features by gradient norm.

        Computes per-feature gradient norms on AR-whitened target
        (Fan & Lv 2008 marginal screening). Very permissive — catches all
        real features plus many false positives. Reduces candidate set
        before the more expensive group lasso.

        Returns:
            prescreened: sorted list of feature indices (into grouped features)
            grad_norms_np: (n_feat,) gradient norm per feature
        """
        # AR-whiten target
        y_resid = self._ar_whiten_target(X_ar, y, valid, C_tgt, lam)

        if valid is not None:
            X_src_v = X_source[valid]
            T_v = int(valid.sum().item())
        else:
            X_src_v = X_source
            T_v = X_source.shape[0]

        # Per-group gradient norm: ||X_g' y_resid||_F / sqrt(T_v)
        ge = n_feat * n_basis
        Xty = X_src_v[:, :ge].T @ y_resid / T_v  # (ge, C_tgt)
        Xty_3d = Xty.view(n_feat, n_basis, C_tgt)
        grad_norms = Xty_3d.norm(dim=(1, 2))  # (n_feat,)

        K = min(max_k, max(1, n_feat // 2))
        _, top_indices = grad_norms.topk(min(K, n_feat))
        prescreened = sorted(top_indices.tolist())

        return prescreened, grad_norms.cpu().numpy()

    def _stage1_stability_selection(self, X_source, X_ar, y, valid,
                                     n_basis, prescreened_indices, C_tgt, lam,
                                     ds_cfg):
        """Stability selection with group lasso (Meinshausen & Buhlmann 2010).

        Runs group lasso on B random subsamples of valid timepoints.
        Features selected in >threshold fraction of subsamples are "stable".
        Provides finite-sample FDR control without surrogates.

        Args:
            X_source: (T, n_feat*n_basis) source basis columns
            X_ar: (T, ar_order*C_tgt) AR columns
            y: (T, C_tgt) target
            valid: (T,) bool mask
            n_basis: basis functions per feature
            prescreened_indices: list of feature indices surviving SIS
            C_tgt: target channels
            lam: ridge lambda for AR whitening
            ds_cfg: doubly_sparse config dict

        Returns:
            stable_indices: list of prescreened feature indices that are stable
            stability_scores: (n_prescreened,) fraction selected per feature
        """
        stab_cfg = ds_cfg.get('stability_selection', {})
        n_subsamples = stab_cfg.get('n_subsamples', 50)
        subsample_frac = stab_cfg.get('subsample_fraction', 0.5)
        threshold = stab_cfg.get('selection_threshold', 0.6)

        n_prescreened = len(prescreened_indices)
        if n_prescreened == 0:
            return [], np.array([])

        # Extract prescreened columns from X_source
        sel_cols = []
        for feat_idx in prescreened_indices:
            start = feat_idx * n_basis
            sel_cols.extend(range(start, start + n_basis))
        sel_cols_t = torch.tensor(sel_cols, device=self.device, dtype=torch.long)
        X_sel = X_source[:, sel_cols_t]

        # AR-whiten target once (reused across subsamples)
        y_resid = self._ar_whiten_target(X_ar, y, valid, C_tgt, lam)

        if valid is not None:
            X_sel_v = X_sel[valid]
            valid_indices = torch.where(valid)[0]
            n_valid = len(valid_indices)
        else:
            X_sel_v = X_sel
            n_valid = X_sel.shape[0]
            valid_indices = torch.arange(n_valid, device=self.device)

        # Define groups for group lasso (over prescreened features)
        groups = [(i * n_basis, (i + 1) * n_basis)
                  for i in range(n_prescreened)]
        solver = GroupLassoSolver(groups, device=self.device)

        # Adaptive lambda: find lambda where ~15-25% of groups are selected.
        # Fixed lambda_fraction often selects too many features on null data
        # (inflating FPR). Binary search for target selection rate is more robust.
        lambda_max = solver._compute_lambda_max(X_sel_v, y_resid, valid=None)
        # Selection budget: target 25% of features per subsample so each
        # coupled feature has high enough per-subsample probability (~70-80%)
        # to cross the 60% stability threshold.  At 10%, coupled features in
        # modalities with 20+ prescreened groups only reach ~50% per-subsample,
        # making stability selection underpowered.  FPR stays controlled:
        # P(Bin(50, 0.25) >= 30) < 0.001.
        target_sel_frac = 0.25
        lam_lo = 0.01 * lambda_max
        lam_hi = lambda_max
        for _ in range(10):
            lam_mid = float(np.sqrt(lam_lo * lam_hi))
            _, sel_trial, _ = solver.fit(
                X_sel_v, y_resid, alpha=lam_mid,
                valid=None, max_iter=200, tol=1e-4)
            frac = len(sel_trial) / max(1, n_prescreened)
            if frac > target_sel_frac:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
        fit_lambda = float(np.sqrt(lam_lo * lam_hi))

        n_subsample = max(20, int(n_valid * subsample_frac))
        selection_counts = np.zeros(n_prescreened)

        gen = torch.Generator(device=self.device)
        gen.manual_seed(42)

        # Streaming weighted Gram: convert subsample indices to {0,1} weight
        # masks, then accumulate Gram via contiguous chunks. Eliminates the
        # random gather bottleneck (X_sel_v[idx]) that saturated GPU Copy Engine
        # while producing numerically identical Gram matrices.
        p_sel = X_sel_v.shape[1]
        C_y = y_resid.shape[1]
        dev = X_sel_v.device
        dtype = X_sel_v.dtype

        # Convert randperm indices → weight mask via scatter (100K writes
        # vs 155M+ random reads for the original gather approach)
        idx_all = torch.zeros(
            n_subsamples, n_subsample, device=dev, dtype=torch.long)
        for b in range(n_subsamples):
            perm = torch.randperm(n_valid, generator=gen, device=dev)
            idx_all[b] = perm[:n_subsample]
        weights = torch.zeros(n_subsamples, n_valid, device=dev, dtype=dtype)
        weights.scatter_(1, idx_all, 1.0)
        del idx_all

        # Streaming Gram: process X in contiguous chunks to limit memory.
        # (K, chunk, p) with chunk=512, K=50, p~1550 -> ~155 MB per chunk.
        CHUNK = 512
        XtX_batch = torch.zeros(n_subsamples, p_sel, p_sel, device=dev, dtype=dtype)
        Xty_batch = torch.zeros(n_subsamples, p_sel, C_y, device=dev, dtype=dtype)
        yty_batch = torch.zeros(n_subsamples, device=dev, dtype=dtype)

        for start in range(0, n_valid, CHUNK):
            end = min(start + CHUNK, n_valid)
            X_chunk = X_sel_v[start:end]                    # (chunk, p) contiguous
            y_chunk = y_resid[start:end]                    # (chunk, C) contiguous
            w = weights[:, start:end].unsqueeze(2)          # (K, chunk, 1) {0,1}
            Xw = X_chunk.unsqueeze(0) * w                   # (K, chunk, p) broadcast
            yw = y_chunk.unsqueeze(0) * w                   # (K, chunk, C)
            Xw_T = Xw.transpose(1, 2)                       # (K, p, chunk)
            XtX_batch += torch.bmm(Xw_T, Xw)               # (K, p, p)
            Xty_batch += torch.bmm(Xw_T, yw)               # (K, p, C)
            yty_batch += (yw ** 2).sum(dim=(1, 2))          # (K,)

        # Normalize by subsample size (same as original)
        XtX_batch /= n_subsample
        Xty_batch /= n_subsample
        yty_batch /= n_subsample

        try:
            _, selected_batch, _ = solver.fit_batched(
                XtX_batch, Xty_batch, yty_batch,
                alpha=float(fit_lambda), max_iter=300, tol=1e-4)
            for b in range(n_subsamples):
                for g_idx in selected_batch[b]:
                    if g_idx < n_prescreened:
                        selection_counts[g_idx] += 1
        except Exception:
            pass

        # Features selected in > threshold fraction of subsamples
        stability_scores = selection_counts / max(1, n_subsamples)
        stable_mask = stability_scores > threshold
        stable_indices = [prescreened_indices[i]
                          for i in range(n_prescreened) if stable_mask[i]]

        return stable_indices, stability_scores

    def _stage1_block_selection(self, X_source, X_ar, y, valid,
                                 n_basis, prescreened_indices, C_tgt, lam,
                                 eval_times, eval_rate, ds_cfg,
                                 max_lag_samples=0):
        """Shift-calibrated block selection for temporally sparse coupling.

        Two levels of shift calibration:
        1. **Feature-level**: For each block, features selected on real data
           but not on most shifted data are "genuinely selected." Aggregated
           via binomial test across blocks.
        2. **Pathway-level**: Training SSE rank test. For each block, the real
           source's prediction quality (training SSE) is ranked among null
           (shifted) fits. Under coupling, real SSE is consistently lower.
           Ranks are aggregated via normal approximation of rank sum.

        The pathway-level test provides significance for Stage 1.5, while
        the feature-level test provides feature selection for Stage 1.

        Args:
            max_lag_samples: number of lag samples in the basis. Shifts must
                exceed this to fully break coupling within the lag range.

        Returns:
            block_selected: list of prescreened feature indices surviving binomial test
            hit_counts: (n_prescreened,) raw hits per feature across blocks
            block_pvals: (n_prescreened,) binomial p-values
            n_blocks_used: number of blocks analyzed
            pathway_pvalue: float, pathway-level shift calibration p-value
        """
        from scipy.stats import binom as binom_dist

        block_cfg = ds_cfg.get('block_selection', {})
        block_duration = block_cfg.get('block_duration_s', 120.0)
        selection_rate = block_cfg.get('selection_rate', 0.05)
        binomial_alpha = block_cfg.get('binomial_alpha', 0.05)
        min_block_samples = block_cfg.get('min_block_samples', 20)
        n_shifts = block_cfg.get('block_n_shifts', 5)

        n_prescreened = len(prescreened_indices)
        if n_prescreened == 0:
            return [], np.zeros(0), np.ones(0), 0, 1.0

        # Extract prescreened columns
        sel_cols = []
        for feat_idx in prescreened_indices:
            start = feat_idx * n_basis
            sel_cols.extend(range(start, start + n_basis))
        sel_cols_t = torch.tensor(sel_cols, device=self.device, dtype=torch.long)
        X_sel = X_source[:, sel_cols_t]

        # AR-whiten target once
        y_resid = self._ar_whiten_target(X_ar, y, valid, C_tgt, lam)

        # Groups for group lasso
        groups = [(i * n_basis, (i + 1) * n_basis)
                  for i in range(n_prescreened)]
        solver = GroupLassoSolver(groups, device=self.device)

        # Divide into blocks
        T = len(eval_times)
        block_samples = max(1, int(block_duration * eval_rate))
        n_blocks = max(1, T // block_samples)

        genuine_counts = np.zeros(n_prescreened)
        hit_counts = np.zeros(n_prescreened)  # raw hits (for diagnostics)
        block_ranks = []  # SSE-based rank of real fit per block
        n_blocks_used = 0

        # RNG for circular shifts
        rng = np.random.default_rng(42)

        for b in range(n_blocks):
            t_start = b * block_samples
            t_end = min((b + 1) * block_samples, T)

            # Block-level validity
            block_mask = torch.zeros(T, dtype=torch.bool, device=self.device)
            block_mask[t_start:t_end] = True
            if valid is not None:
                block_mask = block_mask & valid

            n_block_valid = int(block_mask.sum().item())
            if n_block_valid < min_block_samples:
                continue

            # Extract block data (valid only)
            X_block = X_sel[block_mask]
            if valid is not None:
                valid_cumsum = valid.cumsum(dim=0)
                block_indices = torch.where(block_mask)[0]
                block_in_valid = torch.zeros(
                    len(block_indices), device=self.device, dtype=torch.long)
                for ii, idx in enumerate(block_indices):
                    block_in_valid[ii] = int(valid_cumsum[idx].item()) - 1
                y_block = y_resid[block_in_valid]
            else:
                y_block = y_resid[t_start:t_end]

            try:
                # Compute lambda for this block
                lmax_block = solver._compute_lambda_max(
                    X_block, y_block, valid=None)
                lambda_frac = block_cfg.get('lambda_fraction', 0.3)
                fit_lambda = lambda_frac * lmax_block

                # --- Real fit ---
                real_beta, real_selected, _ = solver.fit(
                    X_block, y_block, alpha=float(fit_lambda),
                    valid=None, max_iter=200, tol=1e-4)
                real_set = set(g for g in real_selected if g < n_prescreened)

                for g_idx in real_set:
                    hit_counts[g_idx] += 1

                # Detrend y and predictions for SSE rank test.
                # Removes [1, t, t²] from both y and predicted y so the
                # rank test measures prediction quality for non-polynomial
                # components only.  This prevents shared slow drifts from
                # inflating SSE advantage of real vs shifted sources.
                y_dt = self._detrend_block_target(y_block)
                pred_real_dt = self._detrend_block_target(
                    X_block @ real_beta)
                sse_real = float(((y_dt - pred_real_dt) ** 2).sum())
                sse_nulls = []

                if not real_set:
                    n_blocks_used += 1
                    continue

                # --- Null fits (circular-shifted source) ---
                # Min shift must exceed max_lag_samples so the shifted source
                # falls entirely outside the basis's lag range.
                null_counts = np.zeros(n_prescreened)
                n_block_t = X_block.shape[0]
                min_shift = max(max_lag_samples + 1, int(0.1 * n_block_t))
                max_shift_val = n_block_t - min_shift

                if max_shift_val <= min_shift:
                    # Block too short for valid shifts — count as genuine
                    for feat in real_set:
                        genuine_counts[feat] += 1
                    n_blocks_used += 1
                    continue

                # Batched null fits: stack shifted sources into (K, T, p),
                # bmm for Gram matrices, solve in one GPU pass.
                p_bl = X_block.shape[1]
                C_bl = y_block.shape[1]
                X_shifted_list = []
                yty_val = (y_block ** 2).sum() / n_block_t

                # Stack all shifted X into (K, T_block, p)
                for s in range(n_shifts):
                    shift = rng.integers(min_shift, max_shift_val)
                    X_shifted_list.append(
                        torch.roll(X_block, int(shift), dims=0))
                X_shifted_batch = torch.stack(X_shifted_list)  # (K, T, p)

                # Batched Gram via bmm
                X_sh_T = X_shifted_batch.transpose(1, 2)  # (K, p, T)
                y_exp = y_block.unsqueeze(0).expand(
                    n_shifts, -1, -1)  # (K, T, C)
                XtX_null = torch.bmm(X_sh_T, X_shifted_batch) / n_block_t
                Xty_null = torch.bmm(X_sh_T, y_exp) / n_block_t
                yty_null = yty_val.expand(n_shifts)
                del X_sh_T, y_exp

                try:
                    null_betas, null_selected_batch, _ = solver.fit_batched(
                        XtX_null, Xty_null, yty_null,
                        alpha=float(fit_lambda), max_iter=200, tol=1e-4)
                    for s in range(n_shifts):
                        for g_idx in null_selected_batch[s]:
                            if g_idx < n_prescreened:
                                null_counts[g_idx] += 1
                        pred_null_dt = self._detrend_block_target(
                            X_shifted_list[s] @ null_betas[s])
                        sse_null = float(
                            ((y_dt - pred_null_dt) ** 2).sum())
                        sse_nulls.append(sse_null)
                except Exception:
                    pass

                # Feature is genuinely selected if: real-selected AND
                # null selection rate < 0.5
                for feat in real_set:
                    null_rate = null_counts[feat] / max(1, n_shifts)
                    if null_rate < 0.5:
                        genuine_counts[feat] += 1

                # Block-level rank: 1 = best fit (lowest SSE)
                if sse_nulls:
                    rank = 1 + sum(1 for sn in sse_nulls if sn <= sse_real)
                    block_ranks.append(rank)

                n_blocks_used += 1
            except Exception:
                continue

        # --- Pathway-level p-value from SSE rank test ---
        # For each block, the real source's training SSE is ranked among
        # null (shifted) SSEs. Under coupling, real fits consistently better
        # (lower rank). Aggregate via normal approximation of rank sum.
        pathway_pvalue = 1.0
        if len(block_ranks) >= 2:
            from scipy.stats import norm as norm_dist
            rank_sum = sum(block_ranks)
            B = len(block_ranks)
            N = n_shifts
            # Under null: each rank ~ Uniform({1, ..., N+1})
            E_rank = (N + 2) / 2.0
            V_rank = ((N + 1) ** 2 - 1) / 12.0
            E_sum = B * E_rank
            V_sum = B * V_rank
            z_score = (rank_sum - E_sum) / max(np.sqrt(V_sum), 1e-10)
            pathway_pvalue = float(norm_dist.cdf(z_score))

        # --- Feature-level binomial test on genuine counts ---
        # Uses shift-calibrated genuine_counts (not raw hits) to control FPs.
        block_pvals = np.ones(n_prescreened)
        if n_blocks_used > 0:
            effective_rate = max(selection_rate, 0.05)
            effective_rate = min(effective_rate, 0.5)

            for i in range(n_prescreened):
                if genuine_counts[i] > 0:
                    block_pvals[i] = 1.0 - binom_dist.cdf(
                        genuine_counts[i] - 1, n_blocks_used, effective_rate)

        block_selected = [prescreened_indices[i]
                          for i in range(n_prescreened)
                          if block_pvals[i] < binomial_alpha]

        return block_selected, genuine_counts, block_pvals, n_blocks_used, pathway_pvalue

    # ------------------------------------------------------------------
    # Per-target methods for behavioral modalities (Fix 1)
    # ------------------------------------------------------------------
    # Instead of multi-output group lasso (Frobenius norm across C_tgt),
    # fit C_tgt independent single-output group lassos and union results.
    # Eliminates gradient dilution when coupling concentrates in tail PCs.

    def _prescreen_per_target(self, X_source, X_ar, y, valid,
                               n_basis, n_feat, C_tgt, lam, max_k=200):
        """Per-target SIS: union of top-K features across all target channels.

        For each target channel c, AR-whiten independently (single-channel AR)
        and compute per-group gradient ||X_g^T y_resid_c||_2 (L2, not Frobenius).
        Return union of all per-channel top-K sets.
        """
        ar_order = self.ar_order

        if valid is not None:
            X_src_v = X_source[valid]
            X_ar_v = X_ar[valid]
            y_v = y[valid]
            T_v = int(valid.sum().item())
        else:
            X_src_v = X_source
            X_ar_v = X_ar
            y_v = y
            T_v = X_source.shape[0]

        ge = n_feat * n_basis
        dev = X_src_v.device
        dtype = X_src_v.dtype
        K = min(max_k, max(1, n_feat // 2))

        union_set = set()
        all_grad_norms = torch.zeros(n_feat, device=dev, dtype=dtype)

        for c in range(C_tgt):
            # Single-channel AR whitening
            y_c = y_v[:, c:c+1]  # (T_v, 1)
            X_ar_c = torch.zeros(T_v, ar_order, device=dev, dtype=dtype)
            for lag_idx in range(ar_order):
                X_ar_c[:, lag_idx] = X_ar_v[:, lag_idx * C_tgt + c]

            if ar_order > 0:
                XtX_ar = X_ar_c.T @ X_ar_c
                reg = lam * torch.eye(ar_order, device=dev, dtype=dtype)
                Xty_ar = X_ar_c.T @ y_c
                beta_ar = torch.linalg.solve(XtX_ar + reg, Xty_ar)
                y_resid_c = y_c - X_ar_c @ beta_ar  # (T_v, 1)
            else:
                y_resid_c = y_c

            # Per-group gradient: ||X_g^T y_resid_c||_2 / sqrt(T_v)
            Xty_c = X_src_v[:, :ge].T @ y_resid_c / T_v  # (ge, 1)
            Xty_3d = Xty_c.view(n_feat, n_basis, 1)
            grad_norms_c = Xty_3d.norm(dim=(1, 2))  # (n_feat,)

            # Track max gradient per feature across targets
            all_grad_norms = torch.maximum(all_grad_norms, grad_norms_c)

            # Top-K for this channel
            _, top_idx = grad_norms_c.topk(min(K, n_feat))
            union_set.update(top_idx.tolist())

        prescreened = sorted(union_set)
        return prescreened, all_grad_norms.cpu().numpy()

    def _stability_selection_per_target(self, X_source, X_ar, y, valid,
                                         n_basis, prescreened_indices, C_tgt,
                                         lam, ds_cfg):
        """Per-target stability selection: group is stable if stable for ANY target.

        For each target c, runs independent group lasso with C=1 on
        prescreened features. A group's stability score = max across targets.
        """
        stab_cfg = ds_cfg.get('stability_selection', {})
        n_subsamples = stab_cfg.get('n_subsamples', 50)
        subsample_frac = stab_cfg.get('subsample_fraction', 0.5)
        threshold = stab_cfg.get('selection_threshold', 0.6)

        n_prescreened = len(prescreened_indices)
        if n_prescreened == 0:
            return [], np.array([]), np.zeros((0, C_tgt), dtype=bool)

        # Extract prescreened columns
        sel_cols = []
        for feat_idx in prescreened_indices:
            start = feat_idx * n_basis
            sel_cols.extend(range(start, start + n_basis))
        sel_cols_t = torch.tensor(sel_cols, device=self.device, dtype=torch.long)
        X_sel = X_source[:, sel_cols_t]

        ar_order = self.ar_order

        if valid is not None:
            X_sel_v = X_sel[valid]
            X_ar_v = X_ar[valid]
            y_v = y[valid]
            valid_indices = torch.where(valid)[0]
            n_valid = len(valid_indices)
        else:
            X_sel_v = X_sel
            X_ar_v = X_ar
            y_v = y
            n_valid = X_sel.shape[0]

        groups = [(i * n_basis, (i + 1) * n_basis)
                  for i in range(n_prescreened)]
        solver = GroupLassoSolver(groups, device=self.device)

        # Max stability score across all targets
        max_stability = np.zeros(n_prescreened)
        # Per-target stability flags for same-target intersection filtering
        per_target_stable = np.zeros((n_prescreened, C_tgt), dtype=bool)

        dev = X_sel_v.device
        dtype = X_sel_v.dtype

        # Select top targets by gradient norm (avoid wasting compute on
        # targets with zero coupling signal)
        max_targets = min(C_tgt, 10)
        if C_tgt > max_targets:
            ge = n_prescreened * n_basis
            target_norms = []
            for c in range(C_tgt):
                grad_c = float((X_sel_v.T @ y_v[:, c:c+1]).norm())
                target_norms.append(grad_c)
            top_targets = sorted(range(C_tgt),
                                  key=lambda i: target_norms[i],
                                  reverse=True)[:max_targets]
        else:
            top_targets = list(range(C_tgt))

        # Phase 1: AR-whiten + adaptive lambda search per target (sequential,
        # ~10 binary search fits per target — not the bottleneck).
        n_subsample = max(20, int(n_valid * subsample_frac))
        p_sel = X_sel_v.shape[1]
        per_target_resid = {}  # c -> y_resid_c
        per_target_lambda = {}  # c -> fit_lambda

        for c in top_targets:
            y_c = y_v[:, c:c+1]
            X_ar_c = torch.zeros(n_valid, ar_order, device=dev, dtype=dtype)
            for lag_idx in range(ar_order):
                X_ar_c[:, lag_idx] = X_ar_v[:, lag_idx * C_tgt + c]

            if ar_order > 0:
                XtX_ar = X_ar_c.T @ X_ar_c
                reg = lam * torch.eye(ar_order, device=dev, dtype=dtype)
                Xty_ar = X_ar_c.T @ y_c
                beta_ar = torch.linalg.solve(XtX_ar + reg, Xty_ar)
                y_resid_c = y_c - X_ar_c @ beta_ar
            else:
                y_resid_c = y_c
            per_target_resid[c] = y_resid_c

            lambda_max = solver._compute_lambda_max(
                X_sel_v, y_resid_c, valid=None)
            # Per-target block selection: keep 0.10 (not 0.25) because the
            # per-target union across C_tgt channels inflates the stable set.
            target_sel_frac = 0.10
            lam_lo = 0.01 * lambda_max
            lam_hi = lambda_max
            for _ in range(10):
                lam_mid = float(np.sqrt(lam_lo * lam_hi))
                _, sel_trial, _ = solver.fit(
                    X_sel_v, y_resid_c, alpha=lam_mid,
                    valid=None, max_iter=200, tol=1e-4)
                frac = len(sel_trial) / max(1, n_prescreened)
                if frac > target_sel_frac:
                    lam_lo = lam_mid
                else:
                    lam_hi = lam_mid
            per_target_lambda[c] = float(np.sqrt(lam_lo * lam_hi))

        # Phase 2: Streaming weighted Gram mega-batch with SHARED XtX.
        # XtX depends only on X and weights (not y), so we compute K weight
        # masks once and reuse Xw_T for both XtX and all targets' Xty in a
        # single streaming pass. Reduces XtX compute from n_tgt×K to K.
        n_tgt = len(top_targets)
        K_total = n_tgt * n_subsamples
        alpha_mega = torch.zeros(K_total, device=dev, dtype=dtype)

        for ci, c in enumerate(top_targets):
            offset = ci * n_subsamples
            alpha_mega[offset:offset + n_subsamples] = per_target_lambda[c]

        # Phase 2a: Generate ONE set of K weight masks (shared across targets)
        gen_shared = torch.Generator(device=dev)
        gen_shared.manual_seed(42)
        idx_shared = torch.zeros(
            n_subsamples, n_subsample, device=dev, dtype=torch.long)
        for b in range(n_subsamples):
            perm = torch.randperm(n_valid, generator=gen_shared, device=dev)
            idx_shared[b] = perm[:n_subsample]
        weights = torch.zeros(n_subsamples, n_valid, device=dev, dtype=dtype)
        weights.scatter_(1, idx_shared, 1.0)
        del idx_shared

        # Phase 2b: Single streaming pass — shared XtX + all targets' Xty
        CHUNK = 512
        XtX_shared = torch.zeros(
            n_subsamples, p_sel, p_sel, device=dev, dtype=dtype)
        Xty_mega = torch.zeros(K_total, p_sel, 1, device=dev, dtype=dtype)
        yty_mega = torch.zeros(K_total, device=dev, dtype=dtype)

        for start in range(0, n_valid, CHUNK):
            end = min(start + CHUNK, n_valid)
            X_chunk = X_sel_v[start:end]                    # (chunk, p)
            w = weights[:, start:end].unsqueeze(2)          # (K, chunk, 1)
            Xw = X_chunk.unsqueeze(0) * w                   # (K, chunk, p)
            Xw_T = Xw.transpose(1, 2)                       # (K, p, chunk)

            # Shared XtX accumulation (computed once for all targets)
            XtX_shared += torch.bmm(Xw_T, Xw)              # (K, p, p)

            # Per-target Xty (reuses Xw_T — no redundant Xw computation)
            for ci, c in enumerate(top_targets):
                y_chunk_c = per_target_resid[c][start:end]  # (chunk, 1)
                yw_c = y_chunk_c.unsqueeze(0) * w           # (K, chunk, 1)
                offset = ci * n_subsamples
                Xty_mega[offset:offset + n_subsamples] += torch.bmm(
                    Xw_T, yw_c)                             # (K, p, 1)
                yty_mega[offset:offset + n_subsamples] += (
                    yw_c ** 2).sum(dim=(1, 2))              # (K,)
        del weights

        # Normalize and tile shared XtX for mega-batch
        XtX_shared /= n_subsample
        Xty_mega /= n_subsample
        yty_mega /= n_subsample

        XtX_mega = XtX_shared.repeat(n_tgt, 1, 1)          # (K_total, p, p)
        del XtX_shared

        # Phase 3: Single mega-batched FISTA solve
        try:
            _, selected_batch, _ = solver.fit_batched(
                XtX_mega, Xty_mega, yty_mega,
                alpha=0.0, max_iter=300, tol=1e-4,
                alpha_batch=alpha_mega)

            # Phase 4: Aggregate results per target
            for ci, c in enumerate(top_targets):
                offset = ci * n_subsamples
                selection_counts_c = np.zeros(n_prescreened)
                for b in range(n_subsamples):
                    for g_idx in selected_batch[offset + b]:
                        if g_idx < n_prescreened:
                            selection_counts_c[g_idx] += 1
                stability_c = selection_counts_c / max(1, n_subsamples)
                max_stability = np.maximum(max_stability, stability_c)
                per_target_stable[:, c] = stability_c > threshold
        except Exception:
            pass

        # Feature is stable if stable for ANY target
        stable_mask = max_stability > threshold
        stable_indices = [prescreened_indices[i]
                          for i in range(n_prescreened) if stable_mask[i]]

        return stable_indices, max_stability, per_target_stable

    def _block_selection_per_target(self, X_source, X_ar, y, valid,
                                     n_basis, prescreened_indices, C_tgt,
                                     lam, eval_times, eval_rate, ds_cfg,
                                     max_lag_samples=0):
        """Per-target block selection: feature is genuine in a block if
        genuine for ANY target channel.

        For pathway SSE rank test: uses sum of per-target SSE improvements
        for the K best targets (largest real-vs-shifted gap).
        """
        from scipy.stats import binom as binom_dist

        block_cfg = ds_cfg.get('block_selection', {})
        block_duration = block_cfg.get('block_duration_s', 120.0)
        selection_rate = block_cfg.get('selection_rate', 0.05)
        binomial_alpha = block_cfg.get('binomial_alpha', 0.05)
        min_block_samples = block_cfg.get('min_block_samples', 20)
        n_shifts = block_cfg.get('block_n_shifts', 20)

        n_prescreened = len(prescreened_indices)
        if n_prescreened == 0:
            return [], np.zeros(0), np.ones(0), 0, 1.0, np.zeros((0, C_tgt))

        # Extract prescreened columns
        sel_cols = []
        for feat_idx in prescreened_indices:
            start = feat_idx * n_basis
            sel_cols.extend(range(start, start + n_basis))
        sel_cols_t = torch.tensor(sel_cols, device=self.device, dtype=torch.long)
        X_sel = X_source[:, sel_cols_t]

        ar_order = self.ar_order
        groups = [(i * n_basis, (i + 1) * n_basis)
                  for i in range(n_prescreened)]
        solver = GroupLassoSolver(groups, device=self.device)

        T = len(eval_times)
        block_samples = max(1, int(block_duration * eval_rate))
        n_blocks = max(1, T // block_samples)

        genuine_counts = np.zeros(n_prescreened)
        hit_counts = np.zeros(n_prescreened)
        # Per-feature-per-target genuine tracking: (n_prescreened, C_tgt)
        # A feature must be genuine for a *specific* target consistently
        # across blocks, not just union-genuine for any target each block.
        per_target_genuine = np.zeros((n_prescreened, C_tgt))
        block_ranks = []
        n_blocks_used = 0
        max_targets_tested = 0  # Track for adjusted binomial rate

        rng = np.random.default_rng(42)
        dev = X_sel.device
        dtype = X_sel.dtype

        for b in range(n_blocks):
            t_start = b * block_samples
            t_end = min((b + 1) * block_samples, T)

            block_mask = torch.zeros(T, dtype=torch.bool, device=self.device)
            block_mask[t_start:t_end] = True
            if valid is not None:
                block_mask = block_mask & valid

            n_block_valid = int(block_mask.sum().item())
            if n_block_valid < min_block_samples:
                continue

            X_block = X_sel[block_mask]
            X_ar_block = X_ar[block_mask]
            y_block = y[block_mask]

            try:
                # Per-target: fit single-output group lasso for each target,
                # union selections and accumulate SSE
                block_selected_union = set()
                block_null_counts = np.zeros(n_prescreened)

                sse_real_sum = 0.0
                sse_nulls_per_shift = [0.0] * n_shifts

                # Determine which targets to use: pick top targets by
                # gradient norm (avoid wasting compute on zero-coupling targets)
                max_targets = min(C_tgt, 15)  # cap for efficiency
                if C_tgt > max_targets:
                    # Quick gradient scan to pick best targets
                    target_norms = []
                    for c in range(C_tgt):
                        grad_c = float((X_block.T @ y_block[:, c:c+1]).norm())
                        target_norms.append(grad_c)
                    top_targets = sorted(range(C_tgt),
                                         key=lambda i: target_norms[i],
                                         reverse=True)[:max_targets]
                else:
                    top_targets = list(range(C_tgt))

                min_shift = max(max_lag_samples + 1,
                                int(0.1 * n_block_valid))
                max_shift_val = n_block_valid - min_shift
                shifts_valid = max_shift_val > min_shift

                # Pre-generate shifts and stack into (K, T, p) for bmm
                if shifts_valid:
                    shift_vals = [int(rng.integers(min_shift, max_shift_val))
                                  for _ in range(n_shifts)]
                    X_shifted_list = [torch.roll(X_block, sv, dims=0)
                                      for sv in shift_vals]
                    X_shifted_stack = torch.stack(X_shifted_list)  # (K, T, p)
                    X_sh_T_block = X_shifted_stack.transpose(1, 2)  # (K, p, T)

                for c in top_targets:
                    # Single-channel AR whitening for block
                    y_c = y_block[:, c:c+1]
                    X_ar_c = torch.zeros(n_block_valid, ar_order,
                                         device=dev, dtype=dtype)
                    for lag_idx in range(ar_order):
                        X_ar_c[:, lag_idx] = X_ar_block[
                            :, lag_idx * C_tgt + c]

                    if ar_order > 0:
                        XtX_ar = X_ar_c.T @ X_ar_c
                        reg = lam * torch.eye(ar_order, device=dev, dtype=dtype)
                        Xty_ar = X_ar_c.T @ y_c
                        beta_ar = torch.linalg.solve(XtX_ar + reg, Xty_ar)
                        y_resid_c = y_c - X_ar_c @ beta_ar
                    else:
                        y_resid_c = y_c

                    # Lambda for this target in this block
                    lmax_block = solver._compute_lambda_max(
                        X_block, y_resid_c, valid=None)
                    lambda_frac = block_cfg.get('lambda_fraction', 0.3)
                    fit_lambda = lambda_frac * lmax_block

                    # Real fit (C=1)
                    real_beta, real_selected, _ = solver.fit(
                        X_block, y_resid_c, alpha=float(fit_lambda),
                        valid=None, max_iter=200, tol=1e-4)
                    real_set = set(g for g in real_selected
                                   if g < n_prescreened)
                    block_selected_union.update(real_set)

                    # SSE for real fit
                    y_dt = self._detrend_block_target(y_resid_c)
                    pred_real_dt = self._detrend_block_target(
                        X_block @ real_beta)
                    sse_real_sum += float(
                        ((y_dt - pred_real_dt) ** 2).sum())

                    # Null fits — batched, each shift gets its own lambda_max
                    # to prevent systematic SSE bias from lambda calibration
                    if shifts_valid and real_set:
                        null_counts_c = np.zeros(n_prescreened)
                        yty_val = (y_resid_c ** 2).sum() / n_block_valid

                        # Batched Gram via pre-stacked X_sh_T_block
                        y_exp = y_resid_c.unsqueeze(0).expand(
                            n_shifts, -1, -1)  # (K, T, 1)
                        XtX_null = torch.bmm(
                            X_sh_T_block, X_shifted_stack) / n_block_valid
                        Xty_null = torch.bmm(
                            X_sh_T_block, y_exp) / n_block_valid
                        yty_null = yty_val.expand(n_shifts)
                        del y_exp

                        # Per-shift lambda_max from Xty (already /n)
                        ng_s = solver.n_groups
                        gs_s = solver.group_size
                        ge_s = solver._group_end
                        alpha_vals = torch.zeros(
                            n_shifts, device=dev, dtype=dtype)
                        if ng_s > 0:
                            Xty_groups = Xty_null[:, :ge_s, :].view(
                                n_shifts, ng_s, gs_s, 1)
                            lmax_per_shift = Xty_groups.norm(
                                dim=(2, 3)).max(dim=1).values  # (K,)
                            lmax_per_shift = torch.clamp(
                                lmax_per_shift, min=1e-10)
                            alpha_vals = lambda_frac * lmax_per_shift

                        try:
                            null_betas, null_sel_batch, _ = \
                                solver.fit_batched(
                                    XtX_null, Xty_null, yty_null,
                                    alpha=0.0,  # ignored, alpha_batch used
                                    max_iter=200, tol=1e-4,
                                    alpha_batch=alpha_vals)
                            for s_idx in range(n_shifts):
                                for g_idx in null_sel_batch[s_idx]:
                                    if g_idx < n_prescreened:
                                        null_counts_c[g_idx] += 1
                                pred_null_dt = self._detrend_block_target(
                                    X_shifted_list[s_idx]
                                    @ null_betas[s_idx])
                                sse_nulls_per_shift[s_idx] += float(
                                    ((y_dt - pred_null_dt) ** 2).sum())
                        except Exception:
                            pass

                        # Feature genuine if null selection rate < 0.5
                        for feat in real_set:
                            null_rate = null_counts_c[feat] / max(1, n_shifts)
                            if null_rate < 0.5:
                                genuine_counts[feat] += 1
                                per_target_genuine[feat, c] += 1

                # Track hits from union across targets
                for g_idx in block_selected_union:
                    hit_counts[g_idx] += 1

                # For features selected but without per-target null test
                # (only happens when shifts_valid is False)
                if not shifts_valid:
                    for feat in block_selected_union:
                        genuine_counts[feat] += 1
                        for c in top_targets:
                            per_target_genuine[feat, c] += 1

                # Block rank from summed SSE across targets
                if shifts_valid and block_selected_union:
                    rank = 1 + sum(1 for sn in sse_nulls_per_shift
                                   if sn <= sse_real_sum)
                    block_ranks.append(rank)

                max_targets_tested = max(max_targets_tested, len(top_targets))
                n_blocks_used += 1
            except Exception:
                continue

        # Pathway-level p-value from SSE rank test
        pathway_pvalue = 1.0
        if len(block_ranks) >= 2:
            from scipy.stats import norm as norm_dist
            rank_sum = sum(block_ranks)
            B = len(block_ranks)
            N = n_shifts
            E_rank = (N + 2) / 2.0
            V_rank = ((N + 1) ** 2 - 1) / 12.0
            E_sum = B * E_rank
            V_sum = B * V_rank
            z_score = (rank_sum - E_sum) / max(np.sqrt(V_sum), 1e-10)
            pathway_pvalue = float(
                norm_dist.cdf(z_score))

        # Feature-level binomial test using per-target consistency.
        # A feature's genuine count = max across targets of its per-target
        # genuine counts.  This avoids union inflation: under null, a feature
        # might be union-genuine in every block (for different targets each
        # time), but its best SINGLE-target genuine count will be low.
        # The binomial rate stays at the base rate (single-target calibrated).
        block_pvals = np.ones(n_prescreened)
        if n_blocks_used > 0:
            effective_rate = max(selection_rate, 0.05)
            effective_rate = min(effective_rate, 0.5)

            for i in range(n_prescreened):
                # Use best single-target genuine count (not union)
                best_target_count = int(per_target_genuine[i].max())
                if best_target_count > 0:
                    block_pvals[i] = 1.0 - binom_dist.cdf(
                        best_target_count - 1, n_blocks_used, effective_rate)

        block_selected = [prescreened_indices[i]
                          for i in range(n_prescreened)
                          if block_pvals[i] < binomial_alpha]

        return (block_selected, genuine_counts, block_pvals,
                n_blocks_used, pathway_pvalue, per_target_genuine)

    def _stage1_discovery(self, source_sigs, target_sigs, duration, eval_rate):
        """Stage 1: Combined doubly-sparse feature selection.

        When doubly_sparse is enabled, applies five-fix pipeline:
          1. Pre-group correlated wavelet features (Fix 1)
          2. Univariate pre-screen / SIS (Fix 3)
          3. Stability selection via group lasso (Fix 2)
          4. Block-level selection for temporal sparsity (Fix 4)
          5. Combined evidence: intersect stable AND block-significant

        Falls back to legacy cc_sq screening when doubly_sparse is disabled.

        Returns:
            DiscoveryResult with selected features per pathway
        """
        from cadence.coupling.discovery import DiscoveryResult

        pathways = get_modality_pathways_v2()
        eval_times = np.arange(0, duration, 1.0 / eval_rate)

        discovery = DiscoveryResult()

        ds_cfg = self.config.get('doubly_sparse', {})
        use_ds = ds_cfg.get('enabled', True)

        # Determine concurrency level for pathway parallelism.
        # Running 2-4 pathways concurrently fills GPU warps that a single
        # pathway's batched FISTA leaves idle (~30-40% → 60-80% utilization).
        n_workers = ds_cfg.get('pathway_workers', 3)  # default matches config
        lam = self.config['ewls']['lambda_ridge']

        # Filter to valid pathways
        valid_pathways = [
            (src_mod, tgt_mod) for src_mod, tgt_mod in pathways
            if src_mod in source_sigs and tgt_mod in target_sigs
        ]

        if n_workers <= 1 or len(valid_pathways) <= 1:
            # Sequential fallback
            for src_mod, tgt_mod in valid_pathways:
                key = (src_mod, tgt_mod)
                try:
                    if use_ds:
                        self._stage1_doubly_sparse(
                            key, *source_sigs[src_mod],
                            *target_sigs[tgt_mod],
                            src_mod, tgt_mod, eval_times, eval_rate,
                            lam, ds_cfg, discovery)
                    else:
                        self._stage1_legacy(
                            key, *source_sigs[src_mod],
                            *target_sigs[tgt_mod],
                            src_mod, tgt_mod, eval_times, eval_rate,
                            lam, discovery)
                except Exception as e:
                    _log(f"  Stage 1 {src_mod}->{tgt_mod}: FAILED ({e})")
                    discovery.selected_features[key] = []
                    discovery.n_selected[key] = 0
                torch.cuda.empty_cache()
        else:
            # Concurrent pathway execution via thread pool.
            # Each pathway gets its own CUDA stream so GPU kernels overlap.
            import threading
            from concurrent.futures import ThreadPoolExecutor, as_completed

            _lock = threading.Lock()

            def _run_pathway(src_mod, tgt_mod):
                key = (src_mod, tgt_mod)
                stream = torch.cuda.Stream(device=self.device)
                try:
                    with torch.cuda.stream(stream):
                        if use_ds:
                            self._stage1_doubly_sparse(
                                key, *source_sigs[src_mod],
                                *target_sigs[tgt_mod],
                                src_mod, tgt_mod, eval_times, eval_rate,
                                lam, ds_cfg, discovery)
                        else:
                            self._stage1_legacy(
                                key, *source_sigs[src_mod],
                                *target_sigs[tgt_mod],
                                src_mod, tgt_mod, eval_times, eval_rate,
                                lam, discovery)
                except Exception as e:
                    _log(f"  Stage 1 {src_mod}->{tgt_mod}: FAILED ({e})")
                    with _lock:
                        discovery.selected_features[key] = []
                        discovery.n_selected[key] = 0
                finally:
                    stream.synchronize()

            # Run the first pathway sequentially to warm up lazy-initialized
            # CUDA/Triton kernels. Avoids "lazy wrapper called at most once"
            # race when multiple threads JIT-compile the same kernel.
            _run_pathway(*valid_pathways[0])

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(_run_pathway, sm, tm)
                    for sm, tm in valid_pathways[1:]
                ]
                for f in as_completed(futures):
                    f.result()  # propagate exceptions

            torch.cuda.empty_cache()

        return discovery

    def _stage1_doubly_sparse(self, key, src_signal, src_ts, src_valid,
                                tgt_signal, tgt_ts, tgt_valid,
                                src_mod, tgt_mod, eval_times, eval_rate,
                                lam, ds_cfg, discovery):
        """Doubly-sparse selection pipeline for one pathway.

        Dispatches to per-target methods (Fix 1) for behavioral modalities
        where multi-target Frobenius gradient dilution is the primary blocker.
        EEG keeps the existing multi-target path which works well.
        """
        # --- Determine per-target vs multi-target dispatch ---
        # Per-target only for high-dimensional TARGETS where Frobenius gradient
        # dilution is the problem. Low-dim targets (ECG 7ch, EEG 10ch) don't
        # suffer dilution and per-target union inflates FPs via multiple testing.
        behavioral_tgt_mods = {'blendshapes_v2', 'pose_features', 'blendshapes'}
        # Per-target mode for cross-modal pathways into behavioral targets
        # (e.g., EEG→BL where coupling targets specific PCA components).
        # Same-modality pathways use single-output (Frobenius norm) because
        # coupling affects many target channels simultaneously, and the
        # per-target union across 15-41 channels inflates stable/block sets.
        use_per_target = (tgt_mod in behavioral_tgt_mods
                          and src_mod != tgt_mod)

        # --- Step 1: Pre-group correlated features ---
        pregroup_enabled = ds_cfg.get('pregroup', {}).get('enabled', True)
        if pregroup_enabled:
            grouped_signal, cluster_map = self._pregroup_features(
                src_signal, src_mod, src_valid)
        else:
            n_ch = src_signal.shape[1]
            grouped_signal = src_signal
            cluster_map = {i: [i] for i in range(n_ch)}

        n_grouped = grouped_signal.shape[1]

        # --- Target PCA reduction (Fix 5: skip for per-target path) ---
        # Per-target fitting handles each target independently, so PCA is
        # unnecessary and harmful (puts coupling in tail components).
        # Multi-target path still uses PCA for non-EEG high-dim targets.
        if not use_per_target:
            max_tgt_dim = ds_cfg.get('max_target_dim', 10)
            if (tgt_signal.shape[1] > max_tgt_dim
                    and tgt_mod not in ('eeg_wavelet', 'eeg_interbrain')):
                tgt_mask = tgt_valid if tgt_valid is not None else np.ones(
                    tgt_signal.shape[0], dtype=bool)
                tgt_centered = tgt_signal - tgt_signal[tgt_mask].mean(axis=0)
                _, _, Vh = np.linalg.svd(
                    tgt_centered[tgt_mask], full_matrices=False)
                tgt_signal = (tgt_centered @ Vh[:max_tgt_dim].T).astype(
                    np.float32)

        C_tgt = tgt_signal.shape[1]

        # --- Per-pathway eval rate (Fix 4) ---
        # Higher eval rates only for per-target behavioral pathways where
        # more samples improve p/T ratio. Multi-target pathways use default
        # rate to avoid inflating feature count → FPs.
        if use_per_target:
            pw_eval_rate = self.eval_rate_overrides.get(src_mod, eval_rate)
        else:
            pw_eval_rate = eval_rate
        pw_eval_times = (np.arange(0, eval_times[-1] + 1.0 / pw_eval_rate,
                                    1.0 / pw_eval_rate)
                         if pw_eval_rate != eval_rate else eval_times)

        basis, n_basis = self._get_pathway_basis_v2(
            src_mod, tgt_mod, pw_eval_rate)

        src_r, src_ts_r, src_valid_r = self._resample_source_to_internal(
            grouped_signal, src_ts, src_valid, pw_eval_times)

        dm = DesignMatrixBuilder(basis, ar_order=self.ar_order,
                                  device=self.device)
        X_full, y, valid = dm.build(
            src_r, tgt_signal,
            source_valid=src_valid_r, target_valid=tgt_valid,
            eval_times=pw_eval_times, source_times=src_ts_r,
            target_times=tgt_ts, eval_rate=pw_eval_rate,
        )
        n_source_cols = n_basis * n_grouped
        X_source = X_full[:, :n_source_cols]
        X_ar = dm._build_ar_terms(
            tgt_signal, tgt_ts, pw_eval_times, pw_eval_rate)

        del X_full  # free memory; we only need X_source and X_ar

        # --- Step 2: SIS univariate pre-screen ---
        _pw_t0 = _time.perf_counter()
        prescreen_cfg = ds_cfg.get('prescreen', {})
        max_k = prescreen_cfg.get('max_features', 200)
        if prescreen_cfg.get('enabled', True):
            if use_per_target:
                prescreened, grad_norms = self._prescreen_per_target(
                    X_source, X_ar, y, valid,
                    n_basis, n_grouped, C_tgt, lam, max_k)
            else:
                prescreened, grad_norms = self._univariate_prescreen(
                    X_source, X_ar, y, valid,
                    n_basis, n_grouped, C_tgt, lam, max_k)
        else:
            prescreened = list(range(n_grouped))
            grad_norms = np.zeros(n_grouped)
        _pw_t1 = _time.perf_counter()

        if not prescreened:
            discovery.selected_features[key] = []
            discovery.n_selected[key] = 0
            discovery.selection_method[key] = 'doubly_sparse'
            del X_source, X_ar, y, valid, dm
            return

        # --- Step 3: Stability selection ---
        stab_cfg = ds_cfg.get('stability_selection', {})
        per_target_stable = None
        if stab_cfg.get('enabled', True):
            if use_per_target:
                stable_groups, stability_scores, per_target_stable = \
                    self._stability_selection_per_target(
                        X_source, X_ar, y, valid,
                        n_basis, prescreened, C_tgt, lam, ds_cfg)
            else:
                stable_groups, stability_scores = \
                    self._stage1_stability_selection(
                        X_source, X_ar, y, valid,
                        n_basis, prescreened, C_tgt, lam, ds_cfg)
            discovery.stability_scores[key] = stability_scores
        else:
            stable_groups = list(prescreened)
            stability_scores = np.ones(len(prescreened))
        _pw_t2 = _time.perf_counter()

        # --- Step 4: Block-level selection ---
        block_cfg = ds_cfg.get('block_selection', {})
        pathway_pvalue = 1.0
        per_target_genuine = None
        if block_cfg.get('enabled', True):
            max_lag_samples = basis.shape[0]
            if use_per_target:
                block_groups, hit_counts, block_pvals, n_blocks, pathway_pvalue, \
                    per_target_genuine = \
                    self._block_selection_per_target(
                        X_source, X_ar, y, valid,
                        n_basis, prescreened, C_tgt, lam,
                        pw_eval_times, pw_eval_rate, ds_cfg,
                        max_lag_samples=max_lag_samples)
            else:
                block_groups, hit_counts, block_pvals, n_blocks, pathway_pvalue = \
                    self._stage1_block_selection(
                        X_source, X_ar, y, valid,
                        n_basis, prescreened, C_tgt, lam,
                        pw_eval_times, pw_eval_rate, ds_cfg,
                        max_lag_samples=max_lag_samples)
            discovery.block_hit_counts[key] = hit_counts
            discovery.block_pvalues[key] = block_pvals
            discovery.n_blocks[key] = n_blocks
            discovery.block_pathway_pvalue[key] = pathway_pvalue
        else:
            block_groups = list(prescreened)
            hit_counts = np.zeros(len(prescreened))
            block_pvals = np.zeros(len(prescreened))
            n_blocks = 0

        # --- Step 5: Combined evidence with intersection significance ---
        n_stable = len(stable_groups)
        n_block = len(block_groups)
        raw_intersection = sorted(set(stable_groups) & set(block_groups))

        # For per-target pathways, require same-target evidence:
        # a feature must be stable AND block-genuine for the SAME target
        # with genuine count >= 2 (significant under binom(B, 0.05) for B >= 2).
        # Under null, P(stable_c AND genuine_c >= 2) ≈ 0.05 * 0.0025 per target,
        # much lower than the union-correlated rate that inflates hypergeom p.
        if (use_per_target and per_target_stable is not None
                and per_target_genuine is not None):
            prescreen_lookup = {feat: i for i, feat in enumerate(prescreened)}
            intersection = []
            for feat_idx in raw_intersection:
                ps_idx = prescreen_lookup.get(feat_idx)
                if ps_idx is None:
                    continue
                stable_targets = set(np.where(per_target_stable[ps_idx])[0])
                # Require genuine count >= 2 for the same target: a single
                # block match is too easy under null (~10-14% per target).
                # Count >= 2 is approximately binomial-significant for
                # typical block counts (B=2-15) at rate 0.05.
                genuine_targets = set(
                    np.where(per_target_genuine[ps_idx] >= 2)[0])
                if stable_targets & genuine_targets:
                    intersection.append(feat_idx)
            intersection = sorted(intersection)
        else:
            intersection = raw_intersection

        n_intersect = len(intersection)
        K = len(prescreened)

        from scipy.stats import hypergeom as hypergeom_dist
        if K > 0 and n_stable > 0 and n_block > 0 and n_intersect > 0:
            p_intersect = float(hypergeom_dist.sf(
                n_intersect - 1, K, n_stable, n_block))
        else:
            p_intersect = 1.0

        fallback_used = 'none'
        if p_intersect < 0.05 and n_intersect >= 2:
            final_groups = intersection
            fallback_used = 'intersection'
        elif pathway_pvalue < 0.01 and n_block >= 2:
            final_groups = block_groups
            fallback_used = 'secondary'
        elif (pathway_pvalue < 0.001 and n_stable >= 2
              and src_mod != tgt_mod):
            final_groups = stable_groups
            fallback_used = 'tertiary'
        else:
            final_groups = []

        # Map cluster indices back to original feature indices
        if final_groups:
            original_indices = []
            for g_idx in final_groups:
                original_indices.extend(cluster_map[g_idx])
            original_indices = sorted(set(original_indices))
        else:
            original_indices = []

        discovery.selected_features[key] = original_indices
        discovery.n_selected[key] = len(original_indices)
        discovery.feature_clusters[key] = cluster_map
        discovery.selection_method[key] = 'doubly_sparse'

        _pw_t3 = _time.perf_counter()
        pt_str = " [per-target]" if use_per_target else ""
        fb_str = f", via={fallback_used}" if fallback_used != 'none' else ""
        _log(f"  Stage 1 {src_mod}->{tgt_mod} [doubly-sparse{pt_str}]: "
              f"grouped={n_grouped}, prescreened={K}, "
              f"stable={n_stable}, block={n_block}, "
              f"intersect={n_intersect} (p={p_intersect:.4f}), "
              f"pathway_p={pathway_pvalue:.4f}, "
              f"final={len(original_indices)}{fb_str} "
              f"[SIS={_pw_t1-_pw_t0:.1f}s stab={_pw_t2-_pw_t1:.1f}s blk={_pw_t3-_pw_t2:.1f}s]")

        del X_source, X_ar, y, valid, dm

    def _stage1_legacy(self, key, src_signal, src_ts, src_valid,
                         tgt_signal, tgt_ts, tgt_valid,
                         src_mod, tgt_mod, eval_times, eval_rate,
                         lam, discovery):
        """Legacy Stage 1: univariate cc_sq screening (original v2 pipeline)."""
        n_src_ch = src_signal.shape[1]
        C_tgt = tgt_signal.shape[1]
        C_min = min(n_src_ch, C_tgt)

        basis, n_basis = self._get_pathway_basis_v2(
            src_mod, tgt_mod, eval_rate)

        src_r, src_ts_r, src_valid_r = self._resample_source_to_internal(
            src_signal, src_ts, src_valid, eval_times)

        dm = DesignMatrixBuilder(basis, ar_order=self.ar_order,
                                  device=self.device)
        X_full, y, valid = dm.build(
            src_r, tgt_signal,
            source_valid=src_valid_r, target_valid=tgt_valid,
            eval_times=eval_times, source_times=src_ts_r,
            target_times=tgt_ts, eval_rate=eval_rate,
        )

        n_source_cols = n_basis * n_src_ch
        X_source = X_full[:, :n_source_cols]

        X_ar = dm._build_ar_terms(
            tgt_signal, tgt_ts, eval_times, eval_rate)

        cc_sq, _, _ = self._compute_matched_cc_sq(
            X_source, X_ar, y, valid,
            n_basis, n_src_ch, C_tgt, lam)

        cc_np = cc_sq.cpu().numpy()

        # Select features above adaptive threshold
        median_cc = float(np.median(cc_np))
        mad = float(np.median(np.abs(cc_np - median_cc)))
        threshold = median_cc + 2.0 * max(mad, 1e-6)

        selected = [i for i in range(C_min) if cc_np[i] > threshold]
        if not selected:
            n_sel = min(20, C_min)
            top_idx = np.argsort(cc_np)[-n_sel:]
            selected = sorted(top_idx.tolist())

        discovery.selected_features[key] = selected
        discovery.n_selected[key] = len(selected)
        discovery.selection_method[key] = 'legacy'

        sum_cc = float(cc_np.sum())
        max_cc = float(cc_np.max())
        best_feat = int(cc_np.argmax())
        _log(f"  Stage 1 {src_mod}->{tgt_mod}: "
              f"n_selected={len(selected)}/{C_min}, "
              f"sum_cc_sq={sum_cc:.6f}, "
              f"max_cc_sq={max_cc:.6f} (feat#{best_feat})")

        del X_full, X_source, X_ar, y, valid, dm

    def _stage1_5_surrogate_screen(self, source_sigs, target_sigs, duration,
                                     discovery, result, eval_rate):
        """Stage 1.5: Significance gate with combined evidence.

        For doubly-sparse pathways:
          Auto-pass: shift-calibrated block selection already provides
          surrogate validation within each block. The hypergeometric
          intersection test from Stage 1 provides the p-value.
          No separate PCA-reduced surrogate test needed.

        For legacy pathways:
          Same-modality (src==tgt, C_min >= 4):
            Matched diagonal with Gram whitening + Max-T combined test.
          Cross-modal or small modalities:
            Targeted SVD surrogate test.
        """
        sig_cfg = self.config['significance']
        n_surr = sig_cfg['surrogate'].get('n_screen_surrogates', 100)
        alpha = sig_cfg['f_test_alpha']
        lam = self.config['ewls']['lambda_ridge']
        ds_cfg = self.config.get('doubly_sparse', {})
        use_surrogate_fallback = ds_cfg.get('surrogate_fallback', True)
        max_pathway_p = sig_cfg.get('max_pathway_p', 1.0)

        eval_times = np.arange(0, duration, 1.0 / eval_rate)
        raw_pvalues = {}

        keys_to_screen = [k for k, sel in discovery.selected_features.items()
                          if sel]

        # Separate doubly-sparse and legacy pathways
        ds_keys = []
        legacy_keys = []
        for key in keys_to_screen:
            method = discovery.selection_method.get(key, 'legacy')
            if method == 'doubly_sparse':
                ds_keys.append(key)
            else:
                legacy_keys.append(key)

        # --- Doubly-sparse pathways: shift-calibrated significance ---
        # The pathway-level shift test from block selection provides coupling
        # evidence: real data selects more features than shifted data across
        # blocks. This replaces the PCA-reduced SVD surrogate test.
        for key in ds_keys:
            src_mod, tgt_mod = key
            n_sel = len(discovery.selected_features[key])

            # Use pathway-level shift p-value from block selection
            pw_p = discovery.block_pathway_pvalue.get(key, 1.0)
            # Also consider block-level feature p-values
            block_pvals = discovery.block_pvalues.get(key)
            if block_pvals is not None and len(block_pvals) > 0:
                best_feat_p = float(np.min(block_pvals))
            else:
                best_feat_p = 1.0
            # Use the more significant of the two
            p_combined = min(pw_p, best_feat_p)

            result.pathway_significant[key] = True
            result.n_significant_pathways += 1
            result.pathway_pvalues[key] = np.full(1, p_combined)
            _log(f"  Screen {src_mod}->{tgt_mod}: PASS [doubly-sparse, shift-calibrated] "
                  f"(n_features={n_sel}, pathway_p={pw_p:.4f}, feat_p={best_feat_p:.4f})")

        # --- Surrogate fallback ---
        # Run for pathways where:
        # 1. Doubly-sparse was NOT used (legacy pathways), OR
        # 2. Doubly-sparse ran but found 0 features AND pathway_p < max_pathway_p
        #    (episodic coupling can pass surrogate test but fail block selection)
        if use_surrogate_fallback and ds_cfg.get('enabled', True):
            all_pathway_keys = set(discovery.selected_features.keys())
            ds_found = set(ds_keys)
            # Legacy pathways (doubly-sparse not applicable)
            ds_legacy = [k for k in all_pathway_keys - ds_found
                         if discovery.selection_method.get(k, 'legacy') != 'doubly_sparse'
                         and not discovery.selected_features.get(k, [])]
            # Doubly-sparse pathways that found 0 features but have low pathway_p
            ds_rejected = [k for k in all_pathway_keys
                           if discovery.selection_method.get(k) == 'doubly_sparse'
                           and not discovery.selected_features.get(k, [])
                           and discovery.block_pathway_pvalue.get(k, 1.0) < max_pathway_p]
            ds_empty = ds_legacy + ds_rejected
            # Re-run these with surrogate screening
            for key in ds_empty:
                src_mod, tgt_mod = key
                if src_mod not in source_sigs or tgt_mod not in target_sigs:
                    continue

                src_signal, src_ts, src_valid = source_sigs[src_mod]
                tgt_signal, tgt_ts, tgt_valid = target_sigs[tgt_mod]

                basis, n_basis = self._get_pathway_basis_v2(
                    src_mod, tgt_mod, eval_rate)

                n_src_ch = src_signal.shape[1]
                C_tgt = tgt_signal.shape[1]
                C_min = min(n_src_ch, C_tgt)

                src_r, src_ts_r, src_valid_r = \
                    self._resample_source_to_internal(
                        src_signal, src_ts, src_valid, eval_times)

                dm = DesignMatrixBuilder(basis, ar_order=self.ar_order,
                                          device=self.device)
                X_full, y, valid = dm.build(
                    src_r, tgt_signal,
                    source_valid=src_valid_r, target_valid=tgt_valid,
                    eval_times=eval_times, source_times=src_ts_r,
                    target_times=tgt_ts, eval_rate=eval_rate,
                )
                X_ar = dm._build_ar_terms(
                    tgt_signal, tgt_ts, eval_times, eval_rate)

                n_source_cols = n_basis * n_src_ch
                X_source = X_full[:, :n_source_cols].clone()
                T_full = X_full.shape[0]
                del X_full

                is_same_mod = (src_mod == tgt_mod)
                use_matched = is_same_mod and C_min >= 4

                try:
                    if use_matched:
                        p_val = self._screen_matched_diagonal(
                            X_source, X_ar, y, valid, n_basis, n_src_ch,
                            C_tgt, lam, n_surr, src_mod, tgt_mod, T_full)
                    else:
                        all_features = list(range(n_src_ch))
                        p_val = self._screen_targeted_svd(
                            X_source, X_ar, y, valid, n_basis, n_src_ch,
                            C_tgt, lam, n_surr, src_mod, tgt_mod, T_full,
                            all_features)

                    if p_val < alpha:
                        # Sustained coupling detected — use legacy cc_sq selection
                        cc_sq, _, _ = self._compute_matched_cc_sq(
                            X_source, X_ar, y, valid,
                            n_basis, n_src_ch, C_tgt, lam)
                        cc_np = cc_sq.cpu().numpy()
                        n_sel = min(20, C_min)
                        top_idx = np.argsort(cc_np)[-n_sel:]
                        selected = sorted(top_idx.tolist())
                        discovery.selected_features[key] = selected
                        discovery.n_selected[key] = len(selected)
                        discovery.selection_method[key] = 'surrogate_fallback'
                        result.pathway_significant[key] = True
                        result.n_significant_pathways += 1
                        result.pathway_pvalues[key] = np.full(1, p_val)
                        _log(f"  Screen {src_mod}->{tgt_mod}: PASS "
                              f"[surrogate fallback] (p={p_val:.4f})")
                    else:
                        _log(f"  Screen {src_mod}->{tgt_mod}: FAIL "
                              f"[surrogate fallback] (p={p_val:.4f})")
                except Exception as e:
                    _log(f"  Screen {src_mod}->{tgt_mod}: FALLBACK FAILED ({e})")

                del X_source, X_ar, y, valid, dm
                torch.cuda.empty_cache()

        # --- Legacy pathways: original surrogate screening ---
        for key in legacy_keys:
            src_mod, tgt_mod = key
            if src_mod not in source_sigs or tgt_mod not in target_sigs:
                continue

            src_signal, src_ts, src_valid = source_sigs[src_mod]
            tgt_signal, tgt_ts, tgt_valid = target_sigs[tgt_mod]

            basis, n_basis = self._get_pathway_basis_v2(
                src_mod, tgt_mod, eval_rate)

            n_src_ch = src_signal.shape[1]
            C_tgt = tgt_signal.shape[1]
            C_min = min(n_src_ch, C_tgt)

            src_r, src_ts_r, src_valid_r = \
                self._resample_source_to_internal(
                    src_signal, src_ts, src_valid, eval_times)

            dm = DesignMatrixBuilder(basis, ar_order=self.ar_order,
                                      device=self.device)
            X_full, y, valid = dm.build(
                src_r, tgt_signal,
                source_valid=src_valid_r, target_valid=tgt_valid,
                eval_times=eval_times, source_times=src_ts_r,
                target_times=tgt_ts, eval_rate=eval_rate,
            )
            X_ar = dm._build_ar_terms(
                tgt_signal, tgt_ts, eval_times, eval_rate)

            n_source_cols = n_basis * n_src_ch
            X_source = X_full[:, :n_source_cols].clone()
            T_full = X_full.shape[0]
            del X_full

            is_same_mod = (src_mod == tgt_mod)
            use_matched = is_same_mod and C_min >= 4

            if use_matched:
                p_val = self._screen_matched_diagonal(
                    X_source, X_ar, y, valid, n_basis, n_src_ch,
                    C_tgt, lam, n_surr, src_mod, tgt_mod, T_full)
            else:
                all_features = list(range(n_src_ch))
                p_val = self._screen_targeted_svd(
                    X_source, X_ar, y, valid, n_basis, n_src_ch,
                    C_tgt, lam, n_surr, src_mod, tgt_mod, T_full,
                    all_features)

            raw_pvalues[key] = p_val

            del X_source, X_ar, y, valid, dm
            torch.cuda.empty_cache()

        # Significance determination for legacy pathways (with optional BH-FDR)
        if raw_pvalues:
            fdr_method = sig_cfg.get('fdr_correction', False)

            all_keys = list(raw_pvalues.keys())
            same_keys = [k for k in all_keys if k[0] == k[1]]
            cross_keys = [k for k in all_keys if k[0] != k[1]]

            pvals_adj = {}
            for group_keys in [same_keys, cross_keys]:
                if not group_keys:
                    continue
                group_pvals = np.array([raw_pvalues[k] for k in group_keys])
                if fdr_method == 'bh' and len(group_pvals) > 1:
                    n_tests = len(group_pvals)
                    sorted_idx = np.argsort(group_pvals)
                    sorted_pvals = group_pvals[sorted_idx]
                    adj = sorted_pvals * n_tests / np.arange(1, n_tests + 1)
                    adj_sorted = np.minimum.accumulate(adj[::-1])[::-1]
                    adj_sorted = np.minimum(adj_sorted, 1.0)
                    adj_back = np.empty_like(group_pvals)
                    adj_back[sorted_idx] = adj_sorted
                    for k, pa in zip(group_keys, adj_back):
                        pvals_adj[k] = pa
                else:
                    for k, pv in zip(group_keys, group_pvals):
                        pvals_adj[k] = pv

            for k in all_keys:
                p_raw = raw_pvalues[k]
                p_adj = pvals_adj[k]
                is_significant = p_adj < alpha
                result.pathway_significant[k] = is_significant
                if is_significant:
                    result.n_significant_pathways += 1
                    result.pathway_pvalues[k] = np.full(1, p_raw)
                    fdr_tag = f" [adj={p_adj:.4f}]" if fdr_method else ""
                    _log(f"  Screen {k[0]}->{k[1]}: PASS "
                          f"(p={p_raw:.4f}{fdr_tag})")
                else:
                    discovery.selected_features[k] = []
                    discovery.n_selected[k] = 0
                    fdr_tag = f" [adj={p_adj:.4f}]" if fdr_method else ""
                    _log(f"  Screen {k[0]}->{k[1]}: FAIL "
                          f"(p={p_raw:.4f}{fdr_tag})")

    @staticmethod
    def _higher_criticism(p_values):
        """Compute Higher Criticism statistic (Donoho & Jin 2004).

        HC detects sparse departures from uniformity in p-value vectors.
        Optimal for the "moderately sparse" regime (5-30% non-null).

        Args:
            p_values: 1D array/tensor of p-values in [0, 1].

        Returns:
            hc: float, the HC* statistic (max over lower half of p-values).
        """
        C = len(p_values)
        if C == 0:
            return 0.0
        p_sorted = np.sort(np.asarray(p_values))
        hc_max = 0.0
        for i in range(C // 2):  # scan lower half (HC* convention)
            p_i = p_sorted[i]
            if p_i <= 0 or p_i >= 1:
                continue
            rank_frac = (i + 1) / C
            hc = np.sqrt(C) * (rank_frac - p_i) / np.sqrt(
                p_i * (1 - p_i))
            if hc > hc_max:
                hc_max = hc
        return hc_max

    def _screen_matched_diagonal(self, X_source, X_ar, y, valid,
                                   n_basis, n_src_ch, C_tgt, lam,
                                   n_surr, src_mod, tgt_mod, T_full):
        """Matched diagonal surrogate test for same-modality pathways.

        Uses ALL features (not just Stage 1 selected) for maximum power.
        Feature i→channel i mapping is structurally valid for same-modality.

        Gram whitening: decorrelates cross-covariance components across basis
        functions to reduce null variance. Equivalent to per-pair F-statistic.

        Max-T combined test (Nichols & Holmes 2002): three test statistics
        (sum, max, topK) capture different coupling profiles:
          sum: total coupling energy (dense coupling, all features coupled)
          max: single strongest feature (very sparse, few features coupled)
          topK: sum of K largest (moderate sparsity)
        Each is standardized by its surrogate distribution (z-scored), and
        the max z-score is compared against the surrogate max-T distribution.
        This avoids multiple testing correction while automatically selecting
        the most powerful statistic for the observed coupling pattern.
        """

        C_min = min(n_src_ch, C_tgt)

        # Compute real matched cc with AR-whitened target
        _, y_resid_mat, cc_matched = self._compute_matched_cc_sq(
            X_source, X_ar, y, valid,
            n_basis, n_src_ch, C_tgt, lam)
        # cc_matched: (C_min, n_basis)

        if valid is not None:
            X_src_v = X_source[valid]
            T_v = int(valid.sum().item())
        else:
            X_src_v = X_source
            T_v = T_full

        # --- Gram whitening ---
        # Compute average Gram matrix across matched pairs to decorrelate
        # the basis function components of the cross-covariance.
        X_reshaped = X_src_v[:, :C_min * n_basis].reshape(
            T_v, C_min, n_basis)
        Xp = X_reshaped.permute(1, 2, 0)  # (C_min, n_basis, T_v)
        G_all = torch.bmm(Xp, Xp.transpose(1, 2)) / T_v
        G_avg = G_all.mean(dim=0)  # (n_basis, n_basis)
        del X_reshaped, Xp, G_all

        eigvals, eigvecs = torch.linalg.eigh(G_avg)
        eigvals = torch.clamp(eigvals, min=1e-8)
        G_inv_half = eigvecs @ torch.diag(
            1.0 / eigvals.sqrt()) @ eigvecs.T
        del eigvals, eigvecs, G_avg

        # Whiten real cc: wcc_sq ~ chi²(n_basis) under H0
        wcc = cc_matched @ G_inv_half.T  # (C_min, n_basis)
        wcc_sq = (wcc ** 2).sum(dim=1)  # (C_min,)
        del wcc

        wcc_sq_cpu = wcc_sq.cpu()
        sum_stat_real = wcc_sq.sum().item()
        max_stat_real = wcc_sq.max().item()
        best_feat = int(wcc_sq.argmax().item())

        K = max(1, C_min // 3)
        topk_real = wcc_sq.topk(min(K, C_min)).values.sum().item()

        # Free intermediate tensors before surrogate loop
        del X_src_v, wcc_sq, cc_matched

        # Surrogate testing
        X_src_3d = X_source.T.unsqueeze(0)
        gen = torch.Generator(device=self.device)
        pathway_seed = sum(ord(c) for c in f"{src_mod}_{tgt_mod}")
        gen.manual_seed(42 + pathway_seed)

        # Store surrogate statistics for max-T combined test
        surr_sums_list = []
        surr_maxs_list = []
        surr_topks_list = []

        n_surr_done = 0
        n_source_cols = n_basis * n_src_ch

        # Account for shifted + valid-masked copy + bmm intermediates
        bytes_per_surr = n_source_cols * T_full * 4 * 3
        # Hard cap: never exceed 2 GB per batch (reliable on Windows WDDM)
        max_batch_mem = max(1, int(2 * 1024**3 / bytes_per_surr))
        try:
            torch.cuda.empty_cache()
            free_vram = torch.cuda.mem_get_info(self.device)[0]
            max_batch_free = max(1, int(0.25 * free_vram / bytes_per_surr))
            max_batch = min(max_batch_mem, max_batch_free)
        except Exception:
            max_batch = max_batch_mem
        batch_sz = min(max_batch, n_surr)

        i = 0
        while i < n_surr:
            n_batch = min(batch_sz, n_surr - i)
            try:
                shifted = circular_shift_surrogate_batched(
                    X_src_3d, n_batch, min_shift_frac=0.1,
                    generator=gen)
                shifted_2d = shifted.permute(0, 2, 1)
                del shifted
                if valid is not None:
                    shifted_v = shifted_2d[:, valid]
                else:
                    shifted_v = shifted_2d
                del shifted_2d

                cc_batch = torch.bmm(
                    shifted_v[:, :, :C_min * n_basis].transpose(1, 2),
                    y_resid_mat.unsqueeze(0).expand(n_batch, -1, -1)
                ) / T_v
                del shifted_v

                cc_4d = cc_batch.view(n_batch, C_min, n_basis, C_min)
                del cc_batch
                cc_matched_surr = cc_4d.diagonal(dim1=1, dim2=3)
                # (n_batch, n_basis, C_min)
                del cc_4d

                # Gram whiten surrogates (same G_inv_half as real data)
                wcc_surr = torch.matmul(G_inv_half, cc_matched_surr)
                # (n_batch, n_basis, C_min)
                del cc_matched_surr
                wcc_sq_surr = (wcc_surr ** 2).sum(dim=1)
                # (n_batch, C_min)
                del wcc_surr

                # Store per-surrogate statistics on CPU for max-T
                k_act = min(K, wcc_sq_surr.shape[1])
                surr_sums_list.append(
                    wcc_sq_surr.sum(dim=1).cpu().numpy())
                surr_maxs_list.append(
                    wcc_sq_surr.max(dim=1).values.cpu().numpy())
                surr_topks_list.append(
                    wcc_sq_surr.topk(k_act, dim=1).values
                    .sum(dim=1).cpu().numpy())

                del wcc_sq_surr

                n_surr_done += n_batch
                i += n_batch

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch_sz = max(1, batch_sz // 2)
                continue

        # --- Max-T combined test (Nichols & Holmes 2002) ---
        # Standardize each statistic by surrogate distribution, take
        # max z-score, compare against surrogate max-T distribution.
        # Properly handles correlation between sum/max/topK without
        # any multiple testing penalty.
        surr_sums = np.concatenate(surr_sums_list)
        surr_maxs = np.concatenate(surr_maxs_list)
        surr_topks = np.concatenate(surr_topks_list)
        del surr_sums_list, surr_maxs_list, surr_topks_list

        mu_s, sd_s = surr_sums.mean(), surr_sums.std() + 1e-20
        mu_m, sd_m = surr_maxs.mean(), surr_maxs.std() + 1e-20
        mu_t, sd_t = surr_topks.mean(), surr_topks.std() + 1e-20

        z_sum = (sum_stat_real - mu_s) / sd_s
        z_max = (max_stat_real - mu_m) / sd_m
        z_topk = (topk_real - mu_t) / sd_t
        maxT_real = max(z_sum, z_max, z_topk)

        z_surr_s = (surr_sums - mu_s) / sd_s
        z_surr_m = (surr_maxs - mu_m) / sd_m
        z_surr_t = (surr_topks - mu_t) / sd_t
        maxT_surr = np.maximum(z_surr_s, np.maximum(z_surr_m, z_surr_t))

        ge_maxT = int((maxT_surr >= maxT_real).sum())
        p_maxT = (ge_maxT + 1) / (n_surr_done + 1)

        # Individual p-values for diagnostics
        p_sum = (int((surr_sums >= sum_stat_real).sum()) + 1) / (n_surr_done + 1)
        p_max = (int((surr_maxs >= max_stat_real).sum()) + 1) / (n_surr_done + 1)
        p_topk = (int((surr_topks >= topk_real).sum()) + 1) / (n_surr_done + 1)

        best_z = ['sum', 'max', f'top{K}'][np.argmax([z_sum, z_max, z_topk])]
        _log(f"  Screen {src_mod}->{tgt_mod} [matched+Gram]: "
              f"sum (p={p_sum:.4f}), "
              f"max feat#{best_feat} (p={p_max:.4f}), "
              f"top{K} (p={p_topk:.4f}), "
              f"maxT[{best_z}] (p={p_maxT:.4f})")
        return p_maxT

    def _screen_scan_matched(self, X_source, X_ar, y, valid,
                               n_basis, n_src_ch, C_tgt, lam,
                               n_surr, src_mod, tgt_mod, T_full):
        """Gram-whitened multi-statistic scan for episodic coupling.

        Slides windows of multiple sizes and computes Gram-whitened
        matched-diagonal wcc_sq per window. Three scan statistics
        (sum, max, topK over pairs) capture different sparsity levels:
          sum:  total coupling energy across all pairs (dense)
          max:  single strongest pair (very sparse)
          topK: sum of K strongest pairs (moderate sparsity)

        The scan maximum over windows × positions is the test statistic.
        Surrogate calibration accounts for all scan multiplicity.
        """
        C_min = min(n_src_ch, C_tgt)
        K = max(1, C_min // 3)

        # AR-whiten target
        y_resid = self._ar_whiten_target(X_ar, y, valid, C_tgt, lam)
        y_resid = y_resid[:, :C_min]  # (T_v, C_min)

        if valid is not None:
            X_src_v = X_source[valid]
            T_v = int(valid.sum().item())
        else:
            X_src_v = X_source
            T_v = T_full

        # --- Gram whitening matrix (same as matched diagonal) ---
        X_reshaped = X_src_v[:, :C_min * n_basis].reshape(
            T_v, C_min, n_basis)
        Xp = X_reshaped.permute(1, 2, 0)  # (C_min, n_basis, T_v)
        G_all = torch.bmm(Xp, Xp.transpose(1, 2)) / T_v
        G_avg = G_all.mean(dim=0)  # (n_basis, n_basis)
        del Xp, G_all

        eigvals, eigvecs = torch.linalg.eigh(G_avg)
        eigvals = torch.clamp(eigvals, min=1e-8)
        G_inv_half = eigvecs @ torch.diag(
            1.0 / eigvals.sqrt()) @ eigvecs.T
        del eigvals, eigvecs, G_avg

        # Matched instantaneous products: z[t, i, k] = X[t, i, k] * y[t, i]
        z_real = X_reshaped * y_resid.unsqueeze(2)  # (T_v, C_min, n_basis)
        del X_reshaped

        # Cumulative sums with prepended zero for easy windowing
        cumZ_real = z_real.cumsum(dim=0)  # (T_v, C_min, n_basis)
        cumZ_real_pad = torch.cat([
            torch.zeros(1, C_min, n_basis, device=self.device,
                        dtype=cumZ_real.dtype),
            cumZ_real
        ], dim=0)  # (T_v + 1, C_min, n_basis)
        del z_real, cumZ_real

        # Window sizes: small windows for brief events (Pose 1-4s),
        # large windows for sustained coupling.
        scan_sizes = [5, 10, 25, 50, 100, 200]

        # Compute real scan statistics (sum, max, topK)
        smax_sum = 0.0
        smax_max = 0.0
        smax_topk = 0.0
        for W in scan_sizes:
            if W >= T_v:
                continue
            stride = max(W // 2, 1)
            starts = torch.arange(0, T_v - W + 1, stride,
                                  device=self.device)
            ends = starts + W
            cc_win = (cumZ_real_pad[ends] - cumZ_real_pad[starts]) / W
            wcc_win = cc_win @ G_inv_half.T
            # (n_windows, C_min)
            wcc_sq_win = (wcc_win ** 2).sum(dim=2)
            del cc_win, wcc_win

            s = wcc_sq_win.sum(dim=1).max().item()
            if s > smax_sum:
                smax_sum = s
            m = wcc_sq_win.max(dim=1).values.max().item()
            if m > smax_max:
                smax_max = m
            k_act = min(K, wcc_sq_win.shape[1])
            t = wcc_sq_win.topk(k_act, dim=1).values.sum(
                dim=1).max().item()
            if t > smax_topk:
                smax_topk = t
            del wcc_sq_win

        # Free X_src_v before surrogate loop (no longer needed)
        del X_src_v

        # Surrogate testing
        X_src_3d = X_source.T.unsqueeze(0)
        gen = torch.Generator(device=self.device)
        pathway_seed = sum(ord(c) for c in f"{src_mod}_{tgt_mod}_scan")
        gen.manual_seed(42 + pathway_seed)

        ge_sum = 0
        ge_max = 0
        ge_topk = 0
        n_surr_done = 0

        # shifted (n_source_cols * T_full) + intermediates (T_v * C_min * n_basis * 3)
        n_source_cols = n_basis * n_src_ch
        bytes_per_surr = (n_source_cols * T_full + T_v * C_min * n_basis * 3) * 4
        max_batch_mem = max(1, int(2 * 1024**3 / bytes_per_surr))
        try:
            torch.cuda.empty_cache()
            free_vram = torch.cuda.mem_get_info(self.device)[0]
            max_batch_free = max(1, int(0.15 * free_vram / bytes_per_surr))
            max_batch = min(max_batch_mem, max_batch_free)
        except Exception:
            max_batch = max_batch_mem
        batch_sz = min(max_batch, n_surr)

        i = 0
        while i < n_surr:
            n_batch = min(batch_sz, n_surr - i)
            try:
                shifted = circular_shift_surrogate_batched(
                    X_src_3d, n_batch, min_shift_frac=0.1,
                    generator=gen)
                shifted_2d = shifted.permute(0, 2, 1)
                del shifted
                if valid is not None:
                    shifted_v = shifted_2d[:, valid]
                else:
                    shifted_v = shifted_2d
                del shifted_2d

                sh_matched = shifted_v[:, :, :C_min * n_basis].reshape(
                    n_batch, T_v, C_min, n_basis)
                del shifted_v
                z_surr = sh_matched * y_resid.unsqueeze(0).unsqueeze(3)
                del sh_matched

                cumZ_surr = z_surr.cumsum(dim=1)
                del z_surr
                zeros_pad = torch.zeros(
                    n_batch, 1, C_min, n_basis,
                    device=self.device, dtype=cumZ_surr.dtype)
                cumZ_pad_s = torch.cat([zeros_pad, cumZ_surr], dim=1)
                del cumZ_surr, zeros_pad

                # Per-surrogate scan max for each statistic
                surr_smax_sum = torch.zeros(n_batch, device=self.device)
                surr_smax_max = torch.zeros(n_batch, device=self.device)
                surr_smax_topk = torch.zeros(n_batch, device=self.device)

                for W in scan_sizes:
                    if W >= T_v:
                        continue
                    stride = max(W // 2, 1)
                    starts = torch.arange(0, T_v - W + 1, stride,
                                          device=self.device)
                    ends = starts + W
                    cc_win = (cumZ_pad_s[:, ends] -
                              cumZ_pad_s[:, starts]) / W
                    wcc_win = cc_win @ G_inv_half.T
                    # (B, n_windows, C_min)
                    wcc_sq_win = (wcc_win ** 2).sum(dim=3)
                    del cc_win, wcc_win

                    # sum over pairs → max over windows
                    s_per = wcc_sq_win.sum(dim=2).max(dim=1).values
                    surr_smax_sum = torch.maximum(surr_smax_sum, s_per)
                    # max over pairs → max over windows
                    m_per = wcc_sq_win.max(dim=2).values.max(
                        dim=1).values
                    surr_smax_max = torch.maximum(surr_smax_max, m_per)
                    # topK over pairs → max over windows
                    k_act = min(K, wcc_sq_win.shape[2])
                    t_per = wcc_sq_win.topk(k_act, dim=2).values.sum(
                        dim=2).max(dim=1).values
                    surr_smax_topk = torch.maximum(
                        surr_smax_topk, t_per)
                    del wcc_sq_win

                del cumZ_pad_s
                ge_sum += int(
                    (surr_smax_sum >= smax_sum).sum().item())
                ge_max += int(
                    (surr_smax_max >= smax_max).sum().item())
                ge_topk += int(
                    (surr_smax_topk >= smax_topk).sum().item())
                n_surr_done += n_batch
                i += n_batch

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch_sz = max(1, batch_sz // 2)
                continue

        p_sum = (ge_sum + 1) / (n_surr_done + 1)
        p_max = (ge_max + 1) / (n_surr_done + 1)
        p_topk = (ge_topk + 1) / (n_surr_done + 1)
        # Single statistic: topK adapts to coupling density
        p_scan = p_topk

        _log(f"  Screen {src_mod}->{tgt_mod} [scan+Gram]: "
              f"sum (p={p_sum:.4f}), max (p={p_max:.4f}), "
              f"top{K} (p={p_topk:.4f})")
        return p_scan

    def _screen_cluster_mass_matched(self, X_source, X_ar, y, valid,
                                       n_basis, n_src_ch, C_tgt, lam,
                                       n_surr, src_mod, tgt_mod, T_full):
        """Max-cluster-mass test for episodic coupling (Maris & Oostenveld 2007).

        Computes Gram-whitened matched cross-covariance in sliding windows,
        forms clusters of contiguous supra-threshold positions, and tests
        whether the max cluster mass exceeds what surrogates produce.

        Sensitive to brief coupling episodes that the session-global matched
        diagonal test misses. Uses a 5-second smoothing window and cluster
        formation to accumulate evidence across episodes.
        """
        C_min = min(n_src_ch, C_tgt)
        K = max(1, C_min // 3)

        # AR-whiten target
        y_resid = self._ar_whiten_target(X_ar, y, valid, C_tgt, lam)
        y_resid = y_resid[:, :C_min]  # (T_v, C_min)

        if valid is not None:
            X_src_v = X_source[valid]
            T_v = int(valid.sum().item())
        else:
            X_src_v = X_source
            T_v = T_full

        # --- Gram whitening matrix (same as matched diagonal) ---
        X_reshaped = X_src_v[:, :C_min * n_basis].reshape(
            T_v, C_min, n_basis)
        Xp = X_reshaped.permute(1, 2, 0)  # (C_min, n_basis, T_v)
        G_all = torch.bmm(Xp, Xp.transpose(1, 2)) / T_v
        G_avg = G_all.mean(dim=0)  # (n_basis, n_basis)
        del Xp, G_all

        eigvals, eigvecs = torch.linalg.eigh(G_avg)
        eigvals = torch.clamp(eigvals, min=1e-8)
        G_inv_half = eigvecs @ torch.diag(
            1.0 / eigvals.sqrt()) @ eigvecs.T
        del eigvals, eigvecs, G_avg

        # Matched instantaneous products: z[t, i, k] = X[t, i, k] * y[t, i]
        z_real = X_reshaped * y_resid.unsqueeze(2)  # (T_v, C_min, n_basis)
        del X_reshaped

        # Cumulative sums for efficient windowing
        cumZ_real = z_real.cumsum(dim=0)
        cumZ_real_pad = torch.cat([
            torch.zeros(1, C_min, n_basis, device=self.device,
                        dtype=cumZ_real.dtype),
            cumZ_real
        ], dim=0)  # (T_v + 1, C_min, n_basis)
        del z_real, cumZ_real

        # Window: ~5 seconds (10 samples at 2Hz eval_rate)
        W = 10
        W = max(5, min(W, T_v // 4))
        stride = max(1, W // 5)  # ~5 positions per window width
        n_pos = (T_v - W) // stride + 1

        starts = torch.arange(0, T_v - W + 1, stride, device=self.device)
        ends = starts + W
        n_pos = len(starts)

        # Per-position windowed Gram-whitened matched cc²
        cc_win = (cumZ_real_pad[ends] - cumZ_real_pad[starts]) / W
        wcc_win = cc_win @ G_inv_half.T
        wcc_sq_win = (wcc_win ** 2).sum(dim=2)  # (n_pos, C_min)
        del cc_win, wcc_win

        # Per-position statistic: top-K wcc_sq
        k_act = min(K, C_min)
        stat_real = wcc_sq_win.topk(k_act, dim=1).values.sum(dim=1)
        stat_real_cpu = stat_real.cpu().numpy()
        del stat_real, wcc_sq_win

        del X_src_v, cumZ_real_pad

        # --- Collect surrogate per-position stats ---
        X_src_3d = X_source.T.unsqueeze(0)
        gen = torch.Generator(device=self.device)
        pathway_seed = sum(ord(c) for c in f"{src_mod}_{tgt_mod}_clust")
        gen.manual_seed(42 + pathway_seed)

        all_surr_stats = []  # list of (n_batch, n_pos) numpy arrays
        n_surr_done = 0
        n_source_cols = n_basis * n_src_ch
        bytes_per_surr = (
            n_source_cols * T_full + T_v * C_min * n_basis * 3) * 4
        max_batch_mem = max(1, int(2 * 1024**3 / bytes_per_surr))
        try:
            torch.cuda.empty_cache()
            free_vram = torch.cuda.mem_get_info(self.device)[0]
            max_batch_free = max(1, int(0.15 * free_vram / bytes_per_surr))
            max_batch = min(max_batch_mem, max_batch_free)
        except Exception:
            max_batch = max_batch_mem
        batch_sz = min(max_batch, n_surr)

        i = 0
        while i < n_surr:
            n_batch = min(batch_sz, n_surr - i)
            try:
                shifted = circular_shift_surrogate_batched(
                    X_src_3d, n_batch, min_shift_frac=0.1,
                    generator=gen)
                shifted_2d = shifted.permute(0, 2, 1)
                del shifted
                if valid is not None:
                    shifted_v = shifted_2d[:, valid]
                else:
                    shifted_v = shifted_2d
                del shifted_2d

                sh_matched = shifted_v[
                    :, :, :C_min * n_basis].reshape(
                    n_batch, T_v, C_min, n_basis)
                del shifted_v
                z_surr = sh_matched * y_resid.unsqueeze(0).unsqueeze(3)
                del sh_matched

                cumZ_surr = z_surr.cumsum(dim=1)
                del z_surr
                zeros_pad = torch.zeros(
                    n_batch, 1, C_min, n_basis,
                    device=self.device, dtype=cumZ_surr.dtype)
                cumZ_pad_s = torch.cat([zeros_pad, cumZ_surr], dim=1)
                del cumZ_surr, zeros_pad

                # Windowed stats for surrogates
                cc_win_s = (cumZ_pad_s[:, ends] -
                            cumZ_pad_s[:, starts]) / W
                del cumZ_pad_s
                wcc_win_s = cc_win_s @ G_inv_half.T
                wcc_sq_win_s = (wcc_win_s ** 2).sum(dim=3)
                del cc_win_s, wcc_win_s
                # (n_batch, n_pos, C_min)

                k_a = min(K, wcc_sq_win_s.shape[2])
                stat_surr = wcc_sq_win_s.topk(
                    k_a, dim=2).values.sum(dim=2)  # (batch, n_pos)
                all_surr_stats.append(stat_surr.cpu().numpy())
                del stat_surr, wcc_sq_win_s

                n_surr_done += n_batch
                i += n_batch

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch_sz = max(1, batch_sz // 2)
                continue

        del y_resid, G_inv_half

        # --- Cluster mass via per-position z-scoring ---
        surr_stats_all = np.concatenate(all_surr_stats, axis=0)
        del all_surr_stats
        # surr_stats_all: (n_surr_done, n_pos)

        # Per-position mean and std from surrogates
        mu = surr_stats_all.mean(axis=0)  # (n_pos,)
        sigma = surr_stats_all.std(axis=0) + 1e-20  # (n_pos,)

        # Z-score real data at each position
        z_real = (stat_real_cpu - mu) / sigma  # (n_pos,)

        # Z-score each surrogate
        z_surr = (surr_stats_all - mu) / sigma  # (n_surr, n_pos)

        # Cluster-forming threshold: z > 2.0
        z_thresh = 2.0
        real_cm = _max_cluster_mass(z_real, z_thresh)

        ge_count = 0
        for k in range(n_surr_done):
            surr_cm = _max_cluster_mass(z_surr[k], z_thresh)
            if surr_cm >= real_cm:
                ge_count += 1

        p_clust = (ge_count + 1) / (n_surr_done + 1)

        n_above = int((z_real > z_thresh).sum())
        _log(f"  Screen {src_mod}->{tgt_mod} [cluster-mass]: "
              f"p={p_clust:.4f}, max_cm={real_cm:.2f}, "
              f"z>{z_thresh} at {n_above}/{len(z_real)} pos")
        return p_clust

    def _screen_targeted_svd(self, X_source, X_ar, y, valid,
                               n_basis, n_src_ch, C_tgt, lam,
                               n_surr, src_mod, tgt_mod, T_full,
                               selected):
        """Targeted SVD surrogate test for cross-modal / small pathways.

        Restricts source to Stage 1 selected features. SVD finds optimal
        coupling direction without assuming channel correspondence.
        """
        # Extract selected feature columns
        sel_cols = []
        for feat_idx in selected:
            start = feat_idx * n_basis
            sel_cols.extend(range(start, start + n_basis))
        sel_cols_t = torch.tensor(sel_cols, device=self.device,
                                   dtype=torch.long)
        X_src_sel = X_source[:, sel_cols_t]
        n_sel = len(selected)
        p_sel = n_sel * n_basis

        # AR-whiten all target channels
        y_resid = self._ar_whiten_target(X_ar, y, valid, C_tgt, lam)

        if valid is not None:
            X_sel_v = X_src_sel[valid]
            T_v = int(valid.sum().item())
        else:
            X_sel_v = X_src_sel
            T_v = T_full

        CC_real = X_sel_v.T @ y_resid / T_v
        s1_real = torch.linalg.svdvals(CC_real)[0].item()
        del X_sel_v, CC_real

        X_sel_3d = X_src_sel.T.unsqueeze(0)
        gen = torch.Generator(device=self.device)
        pathway_seed = sum(ord(c) for c in f"{src_mod}_{tgt_mod}")
        gen.manual_seed(42 + pathway_seed)

        ge_count = 0
        n_surr_done = 0

        bytes_per_item = (p_sel * T_full + 2 * p_sel * C_tgt) * 4 * 2
        max_batch_mem = max(1, int(2 * 1024**3 / bytes_per_item))
        try:
            torch.cuda.empty_cache()
            free_vram = torch.cuda.mem_get_info(self.device)[0]
            max_batch_free = max(1, int(0.3 * free_vram / bytes_per_item))
            max_batch = min(max_batch_mem, max_batch_free)
        except Exception:
            max_batch = max_batch_mem
        batch_sz = min(max_batch, n_surr)

        i = 0
        while i < n_surr:
            n_batch = min(batch_sz, n_surr - i)
            try:
                shifted = circular_shift_surrogate_batched(
                    X_sel_3d, n_batch, min_shift_frac=0.1,
                    generator=gen)
                shifted_2d = shifted.permute(0, 2, 1)
                del shifted
                if valid is not None:
                    shifted_v = shifted_2d[:, valid]
                else:
                    shifted_v = shifted_2d
                del shifted_2d

                CC_surr = torch.bmm(
                    shifted_v.transpose(1, 2),
                    y_resid.unsqueeze(0).expand(n_batch, -1, -1)
                ) / T_v
                del shifted_v

                s1_surr = torch.linalg.svdvals(CC_surr)[:, 0]
                del CC_surr

                ge_count += int((s1_surr >= s1_real).sum().item())
                del s1_surr

                n_surr_done += n_batch
                i += n_batch

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch_sz = max(1, batch_sz // 2)
                continue

        p_val = (ge_count + 1) / (n_surr_done + 1)

        _log(f"  Screen {src_mod}->{tgt_mod} [svd]: "
              f"s1={s1_real:.6f}, p={p_val:.4f}, "
              f"n_sel={n_sel}, dims=({p_sel}x{C_tgt})")
        return p_val

    def _stage2_estimate(self, source_sigs, target_sigs, duration,
                          discovery, result, eval_rate):
        """Stage 2: EWLS estimation on selected features + moderation + nonlinear.

        For each pathway with surviving features from Stage 1:
        1. Build reduced design matrix (selected features only)
        2. Optionally add nonlinear channels (x² for selected predictors)
        3. Optionally add moderation terms (moderator * basis columns)
        4. Run EWLS forward-backward
        5. Store dR2 timecourse and kernels

        For large pathways (high-dim target), automatically falls back to
        chunked processing: target channels are processed in groups to
        reduce the (T, p, p) outer product memory from O(T * (n_src + C*ar)^2)
        to O(T * (n_src + group*ar)^2).
        """
        stage2_cfg = self.config.get('stage2', {})
        use_nonlinear = stage2_cfg.get('nonlinear', {}).get('enabled', True)
        use_moderation = stage2_cfg.get('moderation', {}).get('enabled', True)
        moderator_names = stage2_cfg.get('moderation', {}).get(
            'moderators', ['ecg_hr', 'ecg_rmssd'])
        lam_ridge = self.config['ewls']['lambda_ridge']
        tau = self.config['ewls']['tau_seconds']

        # Per-timepoint significance config
        tp_cfg = self.config['significance'].get('timepoint', {})
        tp_enabled = tp_cfg.get('enabled', True)
        tp_n_surr = tp_cfg.get('n_surrogates', 20)
        tp_smooth_sec = tp_cfg.get('smooth_sec', 30)
        tp_seed = tp_cfg.get('seed', 42)
        tp_surr_rate = tp_cfg.get('surrogate_eval_rate', None)

        # Interbrain surrogate method
        ib_cfg = self.config.get('interbrain', {})
        ib_surrogate_method = ib_cfg.get('surrogate_method', 'circular_shift')
        ib_min_freq = ib_cfg.get('min_freq_hz', 0.0)

        # Pathway-p gate for Stage 2
        max_pathway_p = self.config['significance'].get('max_pathway_p', 1.0)

        eval_times = np.arange(0, duration, 1.0 / eval_rate)

        for key, selected in discovery.selected_features.items():
            if not selected:
                continue

            # Only run Stage 2 for pathways that passed significance screening
            if not result.pathway_significant.get(key, False):
                continue

            # Gate on pathway-level p-value (skip clearly non-significant)
            pw_p = discovery.block_pathway_pvalue.get(key, 1.0)
            if pw_p > max_pathway_p:
                _log(f"  Stage 2 {key[0]}->{key[1]}: skipped "
                     f"(pathway_p={pw_p:.4f} > {max_pathway_p})")
                result.pathway_significant[key] = False
                result.n_significant_pathways = max(
                    0, result.n_significant_pathways - 1)
                continue

            src_mod, tgt_mod = key
            if src_mod not in source_sigs or tgt_mod not in target_sigs:
                continue

            src_signal, src_ts, src_valid = source_sigs[src_mod]
            tgt_signal, tgt_ts, tgt_valid = target_sigs[tgt_mod]

            basis, n_basis = self._get_pathway_basis_v2(
                src_mod, tgt_mod, eval_rate)

            selected_cols = np.array(selected)
            src_selected = src_signal[:, selected_cols]

            src_r, src_ts_r, src_valid_r = self._resample_source_to_internal(
                src_selected, src_ts, src_valid, eval_times)

            n_sel = len(selected)
            n_source_cols = n_basis * n_sel
            C_tgt = tgt_signal.shape[1]
            T_eval = len(eval_times)

            # Estimate EWLS memory for full model
            p_ar = self.ar_order * C_tgt
            p_aug = n_source_cols
            if use_nonlinear:
                p_aug += n_source_cols
            if use_moderation:
                p_aug += n_source_cols * len(moderator_names)
            p_full = p_aug + p_ar

            try:
                free_vram = torch.cuda.mem_get_info(self.device)[0]
            except Exception:
                free_vram = 8 * (1024 ** 3)

            triton_grid = p_full ** 2
            triton_unsafe = triton_grid > 50_000

            # Memory estimate: use checkpointed estimate since solve()
            # auto-dispatches to checkpointed path for large p
            K_est = max(1, int(T_eval ** 0.5))
            n_ckpts_est = (T_eval + K_est - 1) // K_est
            # Peak: checkpoints + 3 segment copies (fwd + bwd + combined)
            ewls_bytes = (n_ckpts_est + 3 * K_est) * 4 * (p_full**2 + p_full * C_tgt)

            if ewls_bytes > 0.4 * free_vram:
                # Even checkpointed path would OOM — must chunk
                self._stage2_chunked(
                    key, src_r, src_ts_r, src_valid_r,
                    tgt_signal, tgt_ts, tgt_valid,
                    basis, n_basis, n_sel, n_source_cols,
                    eval_times, eval_rate, tau, lam_ridge,
                    free_vram, result, target_sigs,
                    use_nonlinear=use_nonlinear,
                    use_moderation=use_moderation,
                    moderator_names=moderator_names,
                    force_sequential=triton_unsafe,
                    selected=list(selected),
                    tp_enabled=tp_enabled,
                    tp_n_surr=tp_n_surr,
                    tp_surr_rate=tp_surr_rate,
                    tp_smooth_sec=tp_smooth_sec,
                    tp_seed=tp_seed,
                    ib_surrogate_method=ib_surrogate_method)
                continue

            # --- Standard full EWLS ---
            # solve() auto-selects checkpointed path when full xx
            # would exceed 40% VRAM; force_sequential bypasses Triton
            dm = DesignMatrixBuilder(basis, ar_order=self.ar_order,
                                      device=self.device)
            X_base, y, valid = dm.build(
                src_r, tgt_signal,
                source_valid=src_valid_r, target_valid=tgt_valid,
                eval_times=eval_times, source_times=src_ts_r,
                target_times=tgt_ts, eval_rate=eval_rate,
            )
            X_restricted = dm._build_ar_terms(
                tgt_signal, tgt_ts, eval_times, eval_rate)

            X_augmented = X_base

            if use_nonlinear and n_source_cols > 0:
                X_source = X_base[:, :n_source_cols]
                X_sq = X_source ** 2
                X_augmented = torch.cat([X_augmented, X_sq], dim=1)

            if use_moderation and moderator_names:
                moderator_signals = self._get_moderator_signals(
                    tgt_signal, tgt_ts, tgt_mod, eval_times,
                    source_sigs, moderator_names)

                if moderator_signals is not None:
                    X_source = X_base[:, :n_source_cols]
                    for mod_sig in moderator_signals:
                        X_mod = X_source * mod_sig.unsqueeze(1)
                        X_augmented = torch.cat([X_augmented, X_mod], dim=1)

            solver = EWLSSolver(
                tau_seconds=tau,
                lambda_ridge=lam_ridge,
                eval_rate=eval_rate,
                device=self.device,
                min_effective_n=self.config['ewls'].get('min_effective_n', 20),
                force_sequential=triton_unsafe,
            )

            try:
                dr2_tv, r2_full_tv, r2_restr_tv, beta_full, _ = \
                    solver.solve_restricted(X_augmented, X_restricted, y, valid)

                dr2_np = dr2_tv.cpu().numpy()
                result.pathway_dr2[key] = dr2_np
                result.pathway_r2_full[key] = r2_full_tv.cpu().numpy()
                result.pathway_r2_restricted[key] = r2_restr_tv.cpu().numpy()
                result.pathway_times[key] = eval_times

                if key not in result.pathway_significant:
                    result.pathway_significant[key] = True
                    result.n_significant_pathways += 1

                if beta_full is not None:
                    basis_coeffs = beta_full[:, :n_source_cols, :].cpu().numpy()
                    kernel = self._reconstruct_kernel(
                        basis_coeffs, n_sel, basis=basis)
                    result.pathway_kernels[key] = kernel

                    # --- Per-feature dr2 decomposition (Fix 6) ---
                    # Coefficient-based: contribution_i(t) proportional to
                    # ||beta_i(t)||² relative to total. Multiply by dr2(t)
                    # so per-feature dr2 sums to total dr2 at each timepoint.
                    if n_sel > 1 and n_source_cols > 0:
                        beta_src = beta_full[:, :n_source_cols, :]  # (T, n_src_cols, C)
                        # Total beta energy at each timepoint
                        total_energy = (beta_src ** 2).sum(dim=(1, 2))  # (T,)
                        total_np = total_energy.cpu().numpy()

                        feat_dr2_dict = {}
                        for i, feat_idx in enumerate(selected):
                            col_start = i * n_basis
                            col_end = (i + 1) * n_basis
                            feat_beta = beta_src[:, col_start:col_end, :]
                            feat_energy = (feat_beta ** 2).sum(dim=(1, 2))
                            feat_frac = feat_energy.cpu().numpy() / (total_np + 1e-20)
                            feat_dr2_dict[feat_idx] = dr2_np * feat_frac

                        result.pathway_feature_dr2[key] = feat_dr2_dict

                    # --- Source x Target feature decomposition ---
                    # Beta energy per (source_feat, target_channel) pair,
                    # apportioned from total dr2. Keep top-N pairs by
                    # mean contribution to avoid storage bloat.
                    max_src_tgt_pairs = 20
                    src_tgt_dict = {}
                    for i, feat_idx in enumerate(selected):
                        col_start = i * n_basis
                        col_end = (i + 1) * n_basis
                        for c in range(C_tgt):
                            pair_beta = beta_src[:, col_start:col_end, c]
                            pair_energy = (pair_beta ** 2).sum(dim=1)
                            pair_frac = pair_energy.cpu().numpy() / (total_np + 1e-20)
                            pair_dr2 = dr2_np * pair_frac
                            src_tgt_dict[(feat_idx, c)] = pair_dr2

                    # Prune to top-N by mean absolute dr2
                    if len(src_tgt_dict) > max_src_tgt_pairs:
                        ranked = sorted(
                            src_tgt_dict.items(),
                            key=lambda kv: np.nanmean(np.abs(kv[1])),
                            reverse=True)
                        src_tgt_dict = dict(ranked[:max_src_tgt_pairs])

                    result.pathway_src_tgt_dr2[key] = src_tgt_dict

                # --- Per-timepoint significance via surrogate threshold ---
                if tp_enabled and n_source_cols > 0:
                    from cadence.significance.surrogate import (
                        surrogate_pvalues_from_design,
                    )
                    # Choose surrogate method (Fourier for interbrain)
                    surr_method = 'circular_shift'
                    if src_mod == INTERBRAIN_MODALITY:
                        surr_method = ib_surrogate_method

                    # Optionally subsample to lower eval rate for surrogates
                    surr_rate = tp_surr_rate or eval_rate
                    if surr_rate < eval_rate:
                        step = max(1, int(round(eval_rate / surr_rate)))
                        X_aug_s = X_augmented[::step]
                        X_res_s = X_restricted[::step]
                        y_s = y[::step]
                        valid_s = valid[::step] if valid is not None else None
                        dr2_s = dr2_np[::step]
                        smooth_s = max(1, int(tp_smooth_sec * surr_rate))
                    else:
                        X_aug_s, X_res_s, y_s = X_augmented, X_restricted, y
                        valid_s, dr2_s = valid, dr2_np
                        smooth_s = max(1, int(tp_smooth_sec * eval_rate))
                        step = 1

                    _log(f"  Stage 2 {src_mod}->{tgt_mod}: "
                         f"per-timepoint p-values "
                         f"({tp_n_surr} {surr_method} surrogates"
                         f"{f' @{surr_rate}Hz' if step > 1 else ''})...")

                    try:
                        # Create solver at surrogate rate if subsampled
                        if step > 1:
                            surr_solver = EWLSSolver(
                                tau_seconds=tau,
                                lambda_ridge=lam_ridge,
                                eval_rate=surr_rate,
                                device=self.device,
                                min_effective_n=self.config['ewls'].get(
                                    'min_effective_n', 20),
                                force_sequential=triton_unsafe,
                            )
                        else:
                            surr_solver = solver

                        # Prepare per-feature dR2 at surrogate rate
                        feat_dr2_s = None
                        if key in result.pathway_feature_dr2 and n_sel > 1:
                            feat_dr2_s = {
                                fi: arr[::step] if step > 1 else arr
                                for fi, arr in result.pathway_feature_dr2[key].items()
                            }

                        # GPD config for continuous p-values
                        tp_cfg = self.config['significance'].get('timepoint', {})
                        gpd_ph = tp_cfg.get('gpd_pool_half', 50)
                        gpd_tq = tp_cfg.get('gpd_threshold_quantile', 0.9)

                        tp_pvalues_raw, tp_dr2_null, feat_pv_raw = \
                            surrogate_pvalues_from_design(
                                surr_solver, X_aug_s, X_res_s,
                                y_s, valid_s, n_source_cols, dr2_s,
                                n_surrogates=tp_n_surr,
                                min_shift_frac=0.1,
                                seed=tp_seed,
                                smooth_samples=smooth_s,
                                surrogate_method=surr_method,
                                n_basis=n_basis,
                                selected=list(selected),
                                feat_dr2_real=feat_dr2_s,
                                gpd_pool_half=gpd_ph,
                                gpd_threshold_quantile=gpd_tq)

                        # Store null stats for HMM detection
                        result.pathway_null_stats[key] = {
                            'mu_0': float(np.mean(tp_dr2_null)),
                            'sigma_0': float(np.std(tp_dr2_null)),
                        }

                        # Interpolate back to full eval rate if subsampled
                        if step > 1:
                            T_full = len(dr2_np)
                            T_sub = len(tp_pvalues_raw)
                            t_sub = np.arange(T_sub) * step
                            t_full = np.arange(T_full)
                            tp_pvalues = np.interp(t_full, t_sub, tp_pvalues_raw)
                        else:
                            tp_pvalues = tp_pvalues_raw

                        result.pathway_pvalues[key] = tp_pvalues

                        # Store per-feature p-values
                        if feat_pv_raw is not None:
                            if step > 1:
                                feat_pv_interp = {}
                                for fi, fp in feat_pv_raw.items():
                                    feat_pv_interp[fi] = np.interp(
                                        t_full, t_sub, fp)
                                result.pathway_feature_pvalues[key] = feat_pv_interp
                            else:
                                result.pathway_feature_pvalues[key] = feat_pv_raw
                    except Exception as tp_e:
                        _log(f"  Stage 2 {src_mod}->{tgt_mod}: "
                             f"timepoint p-values failed ({tp_e})")
                        # Broadcast session-level p-value to match dr2 length
                        if key in result.pathway_pvalues:
                            sess_p = result.pathway_pvalues[key].flat[0]
                            result.pathway_pvalues[key] = np.full_like(
                                dr2_np, sess_p)

            except Exception as e:
                _log(f"  Stage 2 {src_mod}->{tgt_mod}: FAILED ({e}), retrying chunked")
                try:
                    torch.cuda.synchronize(self.device)
                except Exception:
                    pass
                torch.cuda.empty_cache()

                is_triton_error = ('Triton' in str(e) or 'illegal memory' in str(e))

                if is_triton_error:
                    from cadence.regression.ewls import disable_triton_scan
                    disable_triton_scan()
                    _log(f"  Triton disabled globally after crash")

                try:
                    self._stage2_chunked(
                        key, src_r, src_ts_r, src_valid_r,
                        tgt_signal, tgt_ts, tgt_valid,
                        basis, n_basis, n_sel, n_source_cols,
                        eval_times, eval_rate, tau, lam_ridge,
                        free_vram, result, source_sigs,
                        use_nonlinear=use_nonlinear,
                        use_moderation=use_moderation,
                        moderator_names=moderator_names,
                        force_sequential=is_triton_error,
                        selected=list(selected),
                        tp_enabled=tp_enabled,
                        tp_n_surr=tp_n_surr,
                        tp_surr_rate=tp_surr_rate,
                        tp_smooth_sec=tp_smooth_sec,
                        tp_seed=tp_seed,
                        ib_surrogate_method=ib_surrogate_method,
                    )
                except Exception as e2:
                    _log(f"  Stage 2 {src_mod}->{tgt_mod}: chunked retry ALSO FAILED ({e2})")

            # Free GPU memory between pathways
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _stage2_chunked(self, key, src_r, src_ts_r, src_valid_r,
                          tgt_signal, tgt_ts, tgt_valid,
                          basis, n_basis, n_sel, n_source_cols,
                          eval_times, eval_rate, tau, lam_ridge,
                          free_vram, result, source_sigs,
                          use_nonlinear=False, use_moderation=False,
                          moderator_names=None, force_sequential=False,
                          selected=None, tp_enabled=False,
                          tp_n_surr=20, tp_surr_rate=None,
                          tp_smooth_sec=30, tp_seed=42,
                          ib_surrogate_method='circular_shift'):
        """Chunked Stage 2 for large pathways that would OOM full EWLS.

        Processes target channels in groups. Each group gets its own
        AR terms (small), keeping p manageable. Nonlinear and moderation
        terms are built per-chunk (they depend only on source features).
        """
        src_mod, tgt_mod = key
        C_tgt = tgt_signal.shape[1]
        T_eval = len(eval_times)

        # Compute augmented source column count for memory estimation
        n_aug_source = n_source_cols
        if use_nonlinear:
            n_aug_source += n_source_cols
        n_moderators = 0
        if use_moderation and moderator_names:
            n_moderators = len(moderator_names)
            n_aug_source += n_source_cols * n_moderators

        # Pre-compute moderator signals once (source participant's ECG RMSSD)
        moderator_signals = None
        if use_moderation and moderator_names:
            moderator_signals = self._get_moderator_signals(
                tgt_signal, tgt_ts, tgt_mod, eval_times,
                source_sigs, moderator_names)

        # Determine group size: find largest C_g where EWLS fits in memory
        # p = n_aug_source + ar_order * C_g
        # With checkpointing: peak ≈ (n_ckpts + 3*K) * 4 * (p² + p*C_g)
        K_est = max(1, int(T_eval ** 0.5))
        n_ckpts_est = (T_eval + K_est - 1) // K_est
        ckpt_factor = n_ckpts_est + 3 * K_est  # replaces T_eval

        max_ch = C_tgt
        for trial_ch in [32, 24, 16, 12, 8, 4, 2, 1]:
            if trial_ch >= C_tgt:
                continue
            p_trial = n_aug_source + self.ar_order * trial_ch
            mem_trial = ckpt_factor * 4 * (p_trial**2 + p_trial * trial_ch)
            if mem_trial < 0.5 * free_vram:
                max_ch = trial_ch
                break
        if max_ch == C_tgt:
            max_ch = 1  # fallback: single channel at a time

        n_groups = (C_tgt + max_ch - 1) // max_ch
        _log(f"  Stage 2 {src_mod}->{tgt_mod} [chunked]: "
              f"{C_tgt} channels in {n_groups} groups of {max_ch}"
              f"{' (sequential)' if force_sequential else ''}")

        solver = EWLSSolver(
            tau_seconds=tau,
            lambda_ridge=lam_ridge,
            eval_rate=eval_rate,
            device=self.device,
            min_effective_n=self.config['ewls'].get('min_effective_n', 20),
            force_sequential=force_sequential,
        )

        all_dr2 = []
        all_r2_full = []
        all_r2_restr = []
        # Collect per-feature beta energy across chunks for decomposition
        # feat_energy[feat_idx] = (T,) cumulative squared beta energy
        feat_energy_accum = {}
        total_energy_accum = np.zeros(T_eval, dtype=np.float64)
        # src×tgt pairs: (src_feat_idx, global_tgt_ch) -> (T,) energy
        src_tgt_energy = {}

        if selected is None:
            selected = list(range(n_sel))

        _surr_saved = False

        for g in range(n_groups):
            ch_start = g * max_ch
            ch_end = min((g + 1) * max_ch, C_tgt)
            tgt_chunk = tgt_signal[:, ch_start:ch_end]
            tgt_valid_chunk = tgt_valid
            C_g = ch_end - ch_start

            dm_g = DesignMatrixBuilder(basis, ar_order=self.ar_order,
                                        device=self.device)
            X_base_g, y_g, valid_g = dm_g.build(
                src_r, tgt_chunk,
                source_valid=src_valid_r, target_valid=tgt_valid_chunk,
                eval_times=eval_times, source_times=src_ts_r,
                target_times=tgt_ts, eval_rate=eval_rate,
            )
            X_restr_g = dm_g._build_ar_terms(
                tgt_chunk, tgt_ts, eval_times, eval_rate)

            # Augment with nonlinear and moderation terms
            X_augmented_g = X_base_g
            if use_nonlinear and n_source_cols > 0:
                X_source_g = X_base_g[:, :n_source_cols]
                X_sq_g = X_source_g ** 2
                X_augmented_g = torch.cat([X_augmented_g, X_sq_g], dim=1)

            if moderator_signals is not None and n_source_cols > 0:
                X_source_g = X_base_g[:, :n_source_cols]
                for mod_sig in moderator_signals:
                    X_mod_g = X_source_g * mod_sig.unsqueeze(1)
                    X_augmented_g = torch.cat([X_augmented_g, X_mod_g], dim=1)

            try:
                dr2_g, r2f_g, r2r_g, beta_g, _ = solver.solve_restricted(
                    X_augmented_g, X_restr_g, y_g, valid_g)
                all_dr2.append(dr2_g.cpu())
                all_r2_full.append(r2f_g.cpu())
                all_r2_restr.append(r2r_g.cpu())

                # Save first chunk's matrices for per-timepoint surrogates
                if g == 0 and tp_enabled and not _surr_saved:
                    _surr_X_aug = X_augmented_g.clone()
                    _surr_X_res = X_restr_g.clone()
                    _surr_y = y_g.clone()
                    _surr_valid = valid_g.clone() if valid_g is not None else None
                    _surr_n_source = n_source_cols
                    _surr_saved = True

                # Accumulate per-feature beta energy from this chunk
                if beta_g is not None and n_source_cols > 0 and len(selected) > 1:
                    beta_src_g = beta_g[:, :n_source_cols, :]  # (T, n_src_cols, C_g)
                    chunk_total = (beta_src_g ** 2).sum(dim=(1, 2)).cpu().numpy()
                    total_energy_accum += chunk_total

                    for i, feat_idx in enumerate(selected):
                        col_s = i * n_basis
                        col_e = (i + 1) * n_basis
                        feat_beta = beta_src_g[:, col_s:col_e, :]
                        feat_e = (feat_beta ** 2).sum(dim=(1, 2)).cpu().numpy()
                        if feat_idx not in feat_energy_accum:
                            feat_energy_accum[feat_idx] = np.zeros(T_eval, dtype=np.float64)
                        feat_energy_accum[feat_idx] += feat_e

                        # Per src×tgt channel
                        for c_local in range(C_g):
                            c_global = ch_start + c_local
                            pair_e = (feat_beta[:, :, c_local] ** 2).sum(dim=1).cpu().numpy()
                            pair_key = (feat_idx, c_global)
                            if pair_key not in src_tgt_energy:
                                src_tgt_energy[pair_key] = np.zeros(T_eval, dtype=np.float64)
                            src_tgt_energy[pair_key] += pair_e

                    del beta_src_g

                del dr2_g, r2f_g, r2r_g, beta_g
                torch.cuda.empty_cache()
            except Exception as e:
                _log(f"    Chunk {g} ({ch_start}:{ch_end}): FAILED ({e})")
                torch.cuda.empty_cache()
                continue

        if all_dr2:
            # Average dR2 across channel groups (nanmean for robustness)
            dr2_stack = torch.stack(all_dr2)  # (n_groups, T_eval)
            dr2_avg = dr2_stack.nanmean(dim=0).numpy()
            r2f_avg = torch.stack(all_r2_full).nanmean(dim=0).numpy()
            r2r_avg = torch.stack(all_r2_restr).nanmean(dim=0).numpy()

            result.pathway_dr2[key] = dr2_avg
            result.pathway_r2_full[key] = r2f_avg
            result.pathway_r2_restricted[key] = r2r_avg
            result.pathway_times[key] = eval_times

            if key not in result.pathway_significant:
                result.pathway_significant[key] = True
                result.n_significant_pathways += 1

            mean_dr2 = float(np.nanmean(dr2_avg))
            _log(f"  Stage 2 {src_mod}->{tgt_mod} [chunked]: "
                  f"mean_dr2={mean_dr2:.6f}")

            # Per-feature dR2 decomposition from accumulated beta energy
            if feat_energy_accum and total_energy_accum.sum() > 0:
                feat_dr2_dict = {}
                for feat_idx, feat_e in feat_energy_accum.items():
                    feat_frac = feat_e / (total_energy_accum + 1e-20)
                    feat_dr2_dict[feat_idx] = dr2_avg * feat_frac
                result.pathway_feature_dr2[key] = feat_dr2_dict

                # Source x Target decomposition (top-20 pairs)
                max_pairs = 20
                if src_tgt_energy:
                    pair_means = {k: np.nanmean(np.abs(
                        dr2_avg * (v / (total_energy_accum + 1e-20))))
                        for k, v in src_tgt_energy.items()}
                    ranked = sorted(pair_means.items(),
                                    key=lambda kv: -kv[1])
                    src_tgt_dict = {}
                    for (si, ti), _ in ranked[:max_pairs]:
                        frac = src_tgt_energy[(si, ti)] / (total_energy_accum + 1e-20)
                        src_tgt_dict[(si, ti)] = dr2_avg * frac
                    result.pathway_src_tgt_dr2[key] = src_tgt_dict

            # --- Per-timepoint significance via surrogate threshold ---
            if tp_enabled and n_source_cols > 0 and _surr_saved:
                from cadence.significance.surrogate import (
                    surrogate_pvalues_from_design,
                )
                surr_method = 'circular_shift'
                if src_mod == INTERBRAIN_MODALITY:
                    surr_method = ib_surrogate_method

                surr_rate = tp_surr_rate or eval_rate
                if surr_rate < eval_rate:
                    step = max(1, int(round(eval_rate / surr_rate)))
                    X_aug_s = _surr_X_aug[::step]
                    X_res_s = _surr_X_res[::step]
                    y_s = _surr_y[::step]
                    valid_s = _surr_valid[::step] if _surr_valid is not None else None
                    dr2_s = dr2_avg[::step]
                    smooth_s = max(1, int(tp_smooth_sec * surr_rate))
                else:
                    X_aug_s, X_res_s, y_s = _surr_X_aug, _surr_X_res, _surr_y
                    valid_s, dr2_s = _surr_valid, dr2_avg
                    smooth_s = max(1, int(tp_smooth_sec * eval_rate))
                    step = 1

                _log(f"  Stage 2 {src_mod}->{tgt_mod} [chunked]: "
                     f"per-timepoint p-values "
                     f"({tp_n_surr} {surr_method} surrogates"
                     f"{f' @{surr_rate}Hz' if step > 1 else ''})...")

                try:
                    if step > 1:
                        surr_solver = EWLSSolver(
                            tau_seconds=tau,
                            lambda_ridge=lam_ridge,
                            eval_rate=surr_rate,
                            device=self.device,
                            min_effective_n=self.config['ewls'].get(
                                'min_effective_n', 20),
                            force_sequential=True,
                        )
                    else:
                        surr_solver = EWLSSolver(
                            tau_seconds=tau,
                            lambda_ridge=lam_ridge,
                            eval_rate=eval_rate,
                            device=self.device,
                            min_effective_n=self.config['ewls'].get(
                                'min_effective_n', 20),
                            force_sequential=True,
                        )

                    # GPD config for continuous p-values
                    tp_cfg = self.config['significance'].get('timepoint', {})
                    gpd_ph = tp_cfg.get('gpd_pool_half', 50)
                    gpd_tq = tp_cfg.get('gpd_threshold_quantile', 0.9)

                    tp_pvalues_raw, tp_dr2_null, _ = \
                        surrogate_pvalues_from_design(
                            surr_solver, X_aug_s, X_res_s,
                            y_s, valid_s, _surr_n_source, dr2_s,
                            n_surrogates=tp_n_surr,
                            min_shift_frac=0.1,
                            seed=tp_seed,
                            smooth_samples=smooth_s,
                            surrogate_method=surr_method,
                            gpd_pool_half=gpd_ph,
                            gpd_threshold_quantile=gpd_tq)

                    # Store null stats for HMM detection
                    result.pathway_null_stats[key] = {
                        'mu_0': float(np.mean(tp_dr2_null)),
                        'sigma_0': float(np.std(tp_dr2_null)),
                    }

                    if step > 1:
                        T_full = len(dr2_avg)
                        T_sub = len(tp_pvalues_raw)
                        t_sub = np.arange(T_sub) * step
                        t_full = np.arange(T_full)
                        tp_pvalues = np.interp(t_full, t_sub, tp_pvalues_raw)
                    else:
                        tp_pvalues = tp_pvalues_raw

                    result.pathway_pvalues[key] = tp_pvalues
                except Exception as tp_e:
                    _log(f"  Stage 2 {src_mod}->{tgt_mod} [chunked]: "
                         f"timepoint p-values failed ({tp_e})")

                # Clean up surrogate matrices
                try:
                    del _surr_X_aug, _surr_X_res, _surr_y, _surr_valid
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        else:
            _log(f"  Stage 2 {src_mod}->{tgt_mod} [chunked]: "
                  f"ALL chunks failed")

    def _get_moderator_signals(self, tgt_signal, tgt_ts, tgt_mod,
                                eval_times, source_sigs, moderator_names):
        """Extract cardiac moderator signals for moderation terms.

        Uses the SOURCE participant's ECG (RMSSD/HR) as the moderator:
        the source's autonomic state modulates how strongly their own
        features couple to the target participant.

        Returns list of (T_eval,) tensors on device, or None if unavailable.
        """
        # Look for ECG features in SOURCE signals
        ecg_mod = None
        for mod_name in ['ecg_features_v2', 'ecg_features']:
            if mod_name in source_sigs:
                ecg_mod = mod_name
                break

        if ecg_mod is None:
            return None

        ecg_signal, ecg_ts, ecg_valid = source_sigs[ecg_mod]
        moderators = []

        for mod_name in moderator_names:
            # Map moderator name to ECG channel index
            if mod_name == 'ecg_hr':
                ch_idx = 0
            elif mod_name == 'ecg_rmssd':
                ch_idx = 2
            else:
                continue

            if ch_idx >= ecg_signal.shape[1]:
                continue

            # Resample to eval grid
            mod_vals = np.interp(eval_times, ecg_ts, ecg_signal[:, ch_idx])
            # Z-score the moderator
            std = np.std(mod_vals)
            if std > 1e-8:
                mod_vals = (mod_vals - np.mean(mod_vals)) / std
            mod_tensor = torch.tensor(mod_vals, dtype=torch.float32,
                                       device=self.device)
            moderators.append(mod_tensor)

        return moderators if moderators else None
