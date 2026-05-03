"""Tests for Hölder α extraction in detect_au_events.

Verifies the augmentation against synthetic singularities of known
regularity (Mallat & Hwang 1992: |W_f(s, t)| ∝ s^α near a singularity).
Also covers backward-compatibility with the 2-tuple legacy return.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch as _torch  # noqa: F401  -- precede numpy on Windows
import numpy as np
import pytest

from cadence.significance.face_event_coincidence import detect_au_events


FS = 30.0
T = int(20 * FS)


def _signal_with_breakpoint(kind: str) -> np.ndarray:
    """Build a synthetic signal with a known singularity around t=10s.

    Tiny noise is added to break exact symmetry ties in the discrete-grid
    local-maximum detection (real face data has noise that does this).
    """
    rng = np.random.default_rng(0)
    noise = 1e-3 * rng.standard_normal(T)
    if kind == 'step':
        sig = np.zeros(T)
        sig[int(10 * FS):] = 1.0
    elif kind == 'ramp':
        sig = np.zeros(T)
        s, e = int(8 * FS), int(12 * FS)
        sig[s:e] = np.linspace(0, 1, e - s)
        sig[e:] = 1.0
    elif kind == 'gaussian_bump':
        t = np.arange(T) / FS
        sig = np.exp(-((t - 10) ** 2) / (2 * 1.0 ** 2))
    elif kind == 'noise':
        sig = 0.05 * rng.standard_normal(T)
    else:
        raise ValueError(kind)
    return sig + noise


def test_step_alpha_near_zero():
    """A sharp step jump should yield α ≈ 0."""
    sig = _signal_with_breakpoint('step')
    events_s, amps, alphas, chain_lens = detect_au_events(
        sig, FS, return_extra=True)
    assert len(events_s) >= 1, 'Step should produce at least one event'
    # Pick the event closest to the actual step location (t=10s)
    closest = int(np.argmin(np.abs(events_s - 10.0)))
    assert abs(alphas[closest]) < 0.20, \
        f'Expected α≈0 for step; got {alphas[closest]:+.3f}'
    assert chain_lens[closest] >= 4, \
        f'Step chain should span most scales; got {chain_lens[closest]}'


def test_ramp_alpha_near_one():
    """A linear ramp should yield α ≈ 1."""
    sig = _signal_with_breakpoint('ramp')
    events_s, amps, alphas, chain_lens = detect_au_events(
        sig, FS, return_extra=True)
    assert len(events_s) >= 1
    # Ramp event time is somewhere in the ramp interval [8, 12]
    in_ramp = (events_s >= 8.0) & (events_s <= 12.0)
    assert in_ramp.any(), f'No event in ramp interval; events at {events_s}'
    ramp_alphas = alphas[in_ramp]
    assert abs(ramp_alphas.mean() - 1.0) < 0.30, \
        f'Expected α≈1 for ramp; got mean {ramp_alphas.mean():+.3f}'


def test_noise_no_events():
    """Pure noise should be rejected by chain-length filter (most chains
    won't span ≥50% of scales because peaks at different scales are
    spurious and don't align temporally).
    """
    sig = _signal_with_breakpoint('noise')
    events_s, amps, alphas, chain_lens = detect_au_events(
        sig, FS, return_extra=True)
    assert len(events_s) <= 5, \
        f'Pure noise should produce ≤5 chain-confirmed events; got {len(events_s)}'


def test_legacy_return_is_2tuple():
    """Default return (return_extra=False) must remain (events, amps)."""
    sig = _signal_with_breakpoint('gaussian_bump')
    result = detect_au_events(sig, FS)
    assert isinstance(result, tuple) and len(result) == 2, \
        f'Legacy return must be 2-tuple; got {type(result)} of len {len(result)}'
    events, amps = result
    assert isinstance(events, np.ndarray)
    assert isinstance(amps, np.ndarray)
    assert len(events) == len(amps)


def test_extra_return_matches_legacy():
    """events/amps must be identical between legacy and extra return paths."""
    sig = _signal_with_breakpoint('gaussian_bump')
    e_legacy, a_legacy = detect_au_events(sig, FS)
    e_extra, a_extra, alphas, chain_lens = detect_au_events(
        sig, FS, return_extra=True)
    np.testing.assert_array_equal(e_legacy, e_extra)
    np.testing.assert_array_equal(a_legacy, a_extra)
    assert alphas.dtype == np.float32
    assert chain_lens.dtype == np.int8


def test_alpha_bounds_drop_pathological():
    """Events with α outside [-0.5, 1.5] should be dropped."""
    # Constant signal (no singularity → no events)
    sig = np.ones(T)
    events_s, amps, alphas, chain_lens = detect_au_events(
        sig, FS, return_extra=True)
    # All returned events must have α within bounds (or be empty)
    if len(alphas):
        assert (alphas >= -0.5).all() and (alphas <= 1.5).all()


def test_noise_floor_override():
    """Passing explicit noise_floor_override should suppress all events
    when set very high."""
    sig = _signal_with_breakpoint('step')
    events_s, _, _, _ = detect_au_events(sig, FS, return_extra=True,
                                            noise_floor_override=1e9)
    assert len(events_s) == 0
