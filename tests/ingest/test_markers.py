"""Tests for legacy baseline normalization + widest-range conflict resolution."""

from __future__ import annotations

import pytest

from cadence.ingest.xdf_reader import (
    _normalize_legacy_baseline_markers,
    _resolve_widest_range_conflicts,
)


# --------------------------------------------------------------------------
# Legacy baseline_start/stop pairing
# --------------------------------------------------------------------------

def test_baseline_first_pair_becomes_eo_second_pair_becomes_ec():
    raw = [
        (0.0, "baseline_start"),
        (60.0, "baseline_stop"),
        (90.0, "baseline_start"),
        (180.0, "baseline_stop"),
    ]
    out = _normalize_legacy_baseline_markers(raw)
    labels = {lab for _t, lab in out}
    # Original events preserved + EO/EC added
    assert "base_EO_start" in labels
    assert "base_EO_stop" in labels
    assert "base_EC_start" in labels
    assert "base_EC_stop" in labels
    # Specifically the timestamps were copied through
    eo_start = next(t for t, l in out if l == "base_EO_start")
    ec_stop = next(t for t, l in out if l == "base_EC_stop")
    assert eo_start == 0.0
    assert ec_stop == 180.0


def test_baseline_only_one_pair_does_not_emit_ec():
    raw = [(0.0, "baseline_start"), (60.0, "baseline_stop")]
    out = _normalize_legacy_baseline_markers(raw)
    labels = {lab for _t, lab in out}
    assert "base_EO_start" in labels
    assert "base_EC_start" not in labels


def test_baseline_no_events_passes_through():
    raw = [(0.0, "conv_1_start"), (300.0, "conv_1_stop")]
    out = _normalize_legacy_baseline_markers(raw)
    assert out == raw


def test_baseline_third_pair_dropped():
    """Per spec: 1st pair → EO, 2nd → EC, 3rd+ dropped (no current session has 3+)."""
    raw = [
        (0.0, "baseline_start"), (60.0, "baseline_stop"),
        (90.0, "baseline_start"), (180.0, "baseline_stop"),
        (200.0, "baseline_start"), (260.0, "baseline_stop"),
    ]
    out = _normalize_legacy_baseline_markers(raw)
    labels = [(t, l) for t, l in out if l.startswith("base_E")]
    # Only EO + EC start/stop emitted.
    pairs = sorted(labels)
    assert any(l == "base_EO_start" for _t, l in pairs)
    assert any(l == "base_EC_start" for _t, l in pairs)
    # No third (extra) condition emitted.
    assert sum(1 for _t, l in pairs if l.endswith("_start")) == 2


# --------------------------------------------------------------------------
# Widest-range conflict resolution
# --------------------------------------------------------------------------

def test_widest_range_keeps_earliest_start_and_latest_stop():
    raw = [
        (10.0, "conv_1_start"),  # earliest
        (20.0, "conv_1_start"),
        (50.0, "conv_1_stop"),
        (30.0, "conv_1_start"),
        (60.0, "conv_1_stop"),   # latest
    ]
    out = _resolve_widest_range_conflicts(raw)
    starts = [t for t, l in out if l == "conv_1_start"]
    stops = [t for t, l in out if l == "conv_1_stop"]
    assert starts == [10.0]   # only the earliest survives
    assert stops == [60.0]    # only the latest survives


def test_widest_range_y_03_conv_1_pattern():
    """y_03 has conv_1_start at 336.3, 731.6, 731.7 and conv_1_stop at 729.4.
    Widest range should be [336.3, 729.4].
    """
    raw = [
        (336.3, "conv_1_start"),
        (729.4, "conv_1_stop"),
        (731.6, "conv_1_start"),
        (731.7, "conv_1_start"),
    ]
    out = _resolve_widest_range_conflicts(raw)
    out_dict = dict((l, t) for t, l in out)
    assert out_dict["conv_1_start"] == pytest.approx(336.3)
    assert out_dict["conv_1_stop"] == pytest.approx(729.4)


def test_widest_range_passes_unknown_markers_through():
    raw = [
        (0.0, "some_custom_marker"),
        (10.0, "conv_1_start"),
        (50.0, "conv_1_stop"),
    ]
    out = _resolve_widest_range_conflicts(raw)
    assert (0.0, "some_custom_marker") in out


def test_widest_range_handles_meditate_blocks():
    raw = [
        (0.0, "meditate_B_start"),
        (300.0, "meditate_B_stop"),
        (310.0, "meditate_K_start"),
        (610.0, "meditate_K_stop"),
        (620.0, "meditate_B_start"),  # duplicate (e.g. from operator double-press)
        (621.0, "meditate_B_stop"),   # duplicate stop
    ]
    out = _resolve_widest_range_conflicts(raw)
    out_dict = {l: t for t, l in out}
    # meditate_B widest = [0.0, 621.0] (earliest start, latest stop)
    assert out_dict["meditate_B_start"] == 0.0
    assert out_dict["meditate_B_stop"] == 621.0
    # meditate_K untouched
    assert out_dict["meditate_K_start"] == 310.0
    assert out_dict["meditate_K_stop"] == 610.0


# --------------------------------------------------------------------------
# Combined pipeline: normalize then resolve
# --------------------------------------------------------------------------

def test_combined_baseline_then_widest_range():
    """Older session pattern: legacy baseline pair + duplicate conv_1."""
    raw = [
        (0.0, "baseline_start"),       # -> EO
        (60.0, "baseline_stop"),
        (90.0, "baseline_start"),      # -> EC
        (180.0, "baseline_stop"),
        (200.0, "conv_1_start"),
        (300.0, "conv_1_start"),       # duplicate
        (500.0, "conv_1_stop"),
    ]
    norm = _normalize_legacy_baseline_markers(raw)
    norm = _resolve_widest_range_conflicts(norm)
    out = {l: t for t, l in norm}
    assert out["base_EO_start"] == 0.0
    assert out["base_EO_stop"] == 60.0
    assert out["base_EC_start"] == 90.0
    assert out["base_EC_stop"] == 180.0
    assert out["conv_1_start"] == 200.0
    assert out["conv_1_stop"] == 500.0
