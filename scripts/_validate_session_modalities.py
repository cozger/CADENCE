"""Validate all modalities in all cached sessions.

Checks per participant per modality:
  - Array present and non-empty
  - Not all-zero / all-constant
  - Sufficient valid frames (>50%)
  - Reasonable sampling rate coverage
  - No degenerate signals (e.g., near-constant pose with tiny variance)

Usage:
    python scripts/_validate_session_modalities.py
"""

import sys, os, glob, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache

# Minimum fraction of valid frames for a modality to be usable
MIN_VALID_FRAC = 0.30
# Minimum variance for a signal to be considered "real" (not tracker noise/constant)
MIN_VARIANCE = 1e-6
# Minimum duration in seconds for a modality to be usable
MIN_DURATION_S = 60.0


def validate_modality(session, participant, modality, session_duration):
    """Check one modality for one participant. Returns (ok, issues)."""
    data_key = f'{participant}_{modality}'
    valid_key = f'{participant}_{modality}_valid'
    ts_key = f'{participant}_{modality}_ts'

    issues = []

    # Check existence
    if data_key not in session or session[data_key] is None:
        return False, ['MISSING']

    data = session[data_key]
    if data.size == 0:
        return False, ['EMPTY']

    # Shape info
    if data.ndim == 1:
        n_samples = len(data)
        n_channels = 1
        data = data.reshape(-1, 1)
    else:
        n_samples, n_channels = data.shape

    # Duration coverage
    if ts_key in session and session[ts_key] is not None:
        ts = session[ts_key]
        if len(ts) > 1:
            actual_dur = ts[-1] - ts[0]
            coverage = actual_dur / session_duration if session_duration > 0 else 0
            if actual_dur < MIN_DURATION_S:
                issues.append(f'SHORT_DURATION({actual_dur:.0f}s/{session_duration:.0f}s)')
            if coverage < 0.5:
                issues.append(f'LOW_COVERAGE({coverage:.1%})')

    # Validity mask
    if valid_key in session and session[valid_key] is not None:
        valid = session[valid_key]
        if valid.ndim > 1:
            valid_frac = valid.any(axis=1).mean()
        else:
            valid_frac = valid.mean()
        if valid_frac < MIN_VALID_FRAC:
            issues.append(f'LOW_VALIDITY({valid_frac:.1%})')
    else:
        valid_frac = 1.0  # no validity mask = assume all valid

    # All-zero check
    if np.all(data == 0):
        issues.append('ALL_ZERO')
        return False, issues

    # All-constant check (per channel)
    n_const = 0
    for ch in range(n_channels):
        col = data[:, ch]
        if np.nanstd(col) < MIN_VARIANCE:
            n_const += 1
    if n_const == n_channels:
        issues.append(f'ALL_CONSTANT({n_const}/{n_channels}ch)')
        return False, issues
    elif n_const > n_channels * 0.5:
        issues.append(f'MANY_CONSTANT({n_const}/{n_channels}ch)')

    # NaN check
    nan_frac = np.isnan(data).mean()
    if nan_frac > 0.5:
        issues.append(f'HIGH_NAN({nan_frac:.1%})')
        return False, issues
    elif nan_frac > 0.1:
        issues.append(f'SOME_NAN({nan_frac:.1%})')

    # Variance check — is the signal real or just noise floor?
    overall_var = np.nanvar(data)
    if overall_var < MIN_VARIANCE:
        issues.append(f'NEAR_ZERO_VAR({overall_var:.2e})')
        return False, issues

    # Pose-specific: check if body is actually tracked (not just default skeleton)
    if modality in ('pose', 'pose_features'):
        # If >70% of frames have identical values, tracker is outputting defaults
        if data.ndim == 2 and data.shape[1] > 3:
            # Check first few coordinate columns
            unique_ratio = np.array([len(np.unique(data[:, c])) / n_samples
                                     for c in range(min(9, n_channels))])
            if np.median(unique_ratio) < 0.01:
                issues.append('POSE_DEFAULT_VALUES')
                return False, issues

    ok = len([i for i in issues if not i.startswith('SOME_')]) == 0
    return ok, issues


def main():
    config = load_config(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'default.yaml'))
    cache_dir = config['session_cache']
    excluded = set(config.get('excluded_sessions', []))

    sessions = discover_cached_sessions(cache_dir)
    print(f"Found {len(sessions)} cached sessions")
    print(f"Excluded: {excluded}\n")

    modalities = [
        ('eeg', '256 Hz, 14ch'),
        ('ecg', '130 Hz, 1ch'),
        ('blendshapes', '30 Hz, 53ch'),
        ('pose', '12 Hz, 99ch'),
        ('eeg_features', '2 Hz, 8ch'),
        ('ecg_features', '2 Hz, 6ch'),
        ('pose_features', '12 Hz, 41ch'),
    ]

    all_results = {}
    bad_modalities = {}  # session -> {participant -> [modality_list]}

    for session_name, cache_prefix in sorted(sessions):
        if session_name in excluded:
            print(f"\n{'='*60}")
            print(f"  SKIPPED (excluded): {session_name}")
            continue

        print(f"\n{'='*60}")
        print(f"  {session_name}")
        print(f"{'='*60}")

        try:
            session = load_session_from_cache(cache_prefix)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        duration = session.get('duration', 0)
        print(f"  Duration: {duration:.0f}s ({duration/60:.1f} min)")

        session_results = {}
        session_bad = {}

        for participant in ['p1', 'p2']:
            role = session.get(f'{participant}_role', '?')
            print(f"\n  {participant.upper()} ({role}):")

            part_bad = []
            for mod_name, mod_desc in modalities:
                ok, issues = validate_modality(session, participant, mod_name, duration)

                status = 'OK' if ok else 'BAD'
                issue_str = ', '.join(issues) if issues else ''
                marker = '  ' if ok else '**'

                # Get shape
                data_key = f'{participant}_{mod_name}'
                shape_str = ''
                if data_key in session and session[data_key] is not None:
                    shape_str = str(session[data_key].shape)

                print(f"  {marker} {mod_name:20s} {shape_str:>20s}  [{status}] {issue_str}")

                session_results[f'{participant}_{mod_name}'] = {
                    'ok': ok, 'issues': issues
                }

                if not ok:
                    part_bad.append(mod_name)

            if part_bad:
                session_bad[participant] = part_bad

        all_results[session_name] = session_results
        if session_bad:
            bad_modalities[session_name] = session_bad

    # Summary
    print(f"\n\n{'='*60}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*60}")

    if bad_modalities:
        print(f"\n  Sessions with bad modalities:")
        for sess, parts in sorted(bad_modalities.items()):
            for part, mods in sorted(parts.items()):
                print(f"    {sess} / {part}: {', '.join(mods)}")
    else:
        print(f"\n  All modalities OK across all sessions!")

    # Write exclusion dict for alignment.py
    print(f"\n  Recommended EXCLUDED_MODALITIES update:")
    print(f"  {{")
    for sess, parts in sorted(bad_modalities.items()):
        for part, mods in sorted(parts.items()):
            print(f"    '{sess}': {{'{part}': {mods}}},")
    print(f"  }}")

    # Save results
    out_path = 'results/v11/modality_validation.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'results': all_results, 'bad_modalities': bad_modalities}, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
