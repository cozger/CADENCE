"""MVP scaffold builder — slice 7D obs + 2D cov from V11 scaffold per session.

Per docs/superpowers/specs/2026-05-01-mvp-rslds-grant-figures-design.md
§Implementation outline step 1.

Per session:
  1. Verify V11 scaffold exists; check preproc-artifact-content-hash freshness
     (V11 scaffolds predating data-pipeline-v1 lack `digest_xdf_md5` in
     sidecar — fall back to verifying all 4 preproc sidecars carry the same
     `digest_xdf_md5` as `data/digest/v1/<sid>.json`'s `xdf_md5`).
  2. Slice the 7 observation channels:
        conc_theta, conc_alpha, bl_expr, bl_activity_conc, pose, resp, ecg_hf
     and 2 covariates: coupling_flexibility, lambda_2.
  3. Pose channel sourcing:
        * --pose-channel auto (default): read results/mvp/phase1_pose/phase1_report.md
          (Phase 1 pose candidates, docs/superpowers/plans/2026-09-09-pose-coupling-phase1.md),
          else results/mvp/phase0/phase0_report.md, else the V11 multi-lag baseline.
        * --pose-channel ddtw: force DDTW (phase0/pose_ddtw_per_session.npz).
        * --pose-channel angles | angle_speed | evt_landing | evt_peak: force a
          Phase 1 candidate (phase1_pose/pose_<mode>_per_session.npz, key {sid}__z).
        * --pose-channel baseline: force V11 multi-lag baseline (read z_pose).
  4. Write results/mvp/<sid>/mvp_scaffold.{npz,json}.

Cohort-level:
  - Emit results/mvp/cohort_protocol_assignment.csv with one row per canonical
    session: session_id, protocol, n_meditation_phases_present,
    n_pe_phases_present, has_eeg_preproc, included_in_production_fit, status_note.
"""

from __future__ import annotations

# Windows torch+numpy DLL ordering: must import torch BEFORE numpy.
import torch as _torch  # noqa: F401

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadence.ingest.quality import list_canonical_sessions
from cadence.io.resources import limit_blas_threads, pick_n_jobs

REPO_ROOT = Path(__file__).resolve().parents[1]
DIGEST_ROOT = REPO_ROOT / 'data' / 'digest' / 'v1'
PREPROC_ROOTS = {m: REPO_ROOT / 'data' / 'preproc' / m / 'v1'
                 for m in ('eeg', 'face', 'ecg', 'pose')}
V11_SCAFFOLD_ROOT = REPO_ROOT / 'results' / 'v11'
PHASE0_DIR = REPO_ROOT / 'results' / 'mvp' / 'phase0'
PHASE1_DIR = REPO_ROOT / 'results' / 'mvp' / 'phase1_pose'
OUT_ROOT = REPO_ROOT / 'results' / 'mvp'

# Pose channel sourcing: CLI choice -> decision name (as written in the
# phase reports and recorded as `pose_source` in the scaffold sidecar).
POSE_CHANNEL_CHOICES = ('auto', 'baseline', 'ddtw', 'angles', 'angle_speed',
                        'evt_landing', 'evt_peak')
POSE_DECISION_BY_CHOICE = {c: f'pose_{c}' for c in POSE_CHANNEL_CHOICES if c != 'auto'}
POSE_DECISIONS = tuple(POSE_DECISION_BY_CHOICE.values())

# 6-channel MVP observation set
# - 5 channels (conc_theta/alpha, pose, resp, ecg_hf) are sliced from V11
# - bl_event_coincidence is computed directly from face/v1 preproc
#   (replaces obsolete bl_expr + bl_activity_conc — see
#   cadence/significance/face_event_coincidence.py docstring for rationale).
MVP_OBS_CHANNELS = ['conc_theta', 'conc_alpha', 'bl_event_coincidence',
                    'pose', 'resp', 'ecg_hf']
# Channels sourced from V11 ztimecourses (everything except bl_event_coincidence)
MVP_V11_CHANNELS = {'conc_theta', 'conc_alpha', 'pose', 'resp', 'ecg_hf'}
MVP_COV_CHANNELS = ['coupling_flexibility', 'lambda2']

# Conditions per protocol (for cohort_protocol_assignment.csv counts)
MEDITATION_PHASES = ('meditate_B', 'meditate_K')
PE_PHASES = ('PE_1', 'PE_2', 'PE')


# ── Phase decision parsing ──────────────────────────────────────────

def _parse_decision_line(text: str) -> str | None:
    """Value between backticks on the '**Pose channel for MVP scaffold: `x`**' line."""
    for line in text.splitlines():
        if 'Pose channel for MVP scaffold' in line and '`' in line:
            parts = line.split('`')
            if len(parts) >= 2:
                return parts[1].strip()
    return None


def _read_report_decision(report: Path) -> str | None:
    if not report.exists():
        return None
    return _parse_decision_line(report.read_text(encoding='utf-8'))


def read_phase0_decision() -> str | None:
    """Read phase0_report.md and return 'pose_ddtw' or 'pose_baseline' or None."""
    return _read_report_decision(PHASE0_DIR / 'phase0_report.md')


def read_phase1_decision() -> str | None:
    """Read phase1_report.md and return its `pose_<mode>` decision or None."""
    return _read_report_decision(PHASE1_DIR / 'phase1_report.md')


def read_phase_decision() -> str:
    """Pose decision for --pose-channel auto.

    phase1_report.md first, then phase0_report.md, then 'pose_baseline'. A
    report whose decision is not one of POSE_DECISIONS is skipped, so a
    malformed Phase 1 report cannot mask a valid Phase 0 one.
    """
    for reader in (read_phase1_decision, read_phase0_decision):
        decision = reader()
        if decision in POSE_DECISIONS:
            return decision
    return 'pose_baseline'


# ── Freshness check ─────────────────────────────────────────────────

def check_freshness(sid: str) -> dict:
    """Verify V11 scaffold + preproc artifacts are coherent with current digest.

    Returns {'fresh': bool, 'reason': str, 'digest_xdf_md5': str,
             'has_preproc': {modality: bool}, 'v11_scaffold_present': bool}.
    """
    out = {'sid': sid, 'has_preproc': {}, 'v11_scaffold_present': False}
    digest_path = DIGEST_ROOT / f'{sid}.json'
    if not digest_path.exists():
        return {**out, 'fresh': False, 'reason': 'no digest'}
    digest = json.loads(digest_path.read_text())
    digest_md5 = digest.get('xdf_md5', '')
    out['digest_xdf_md5'] = digest_md5

    for mod, root in PREPROC_ROOTS.items():
        out['has_preproc'][mod] = (root / f'{sid}.npz').exists()

    v11_path = V11_SCAFFOLD_ROOT / sid / 'scaffold_v11_ztimecourses.npz'
    out['v11_scaffold_present'] = v11_path.exists()
    if not v11_path.exists():
        return {**out, 'fresh': False, 'reason': 'no V11 scaffold'}

    # Check V11 sidecar for digest_xdf_md5
    sidecar = json.loads((V11_SCAFFOLD_ROOT / sid / 'scaffold_v11_results.json'
                           ).read_text())
    if 'digest_xdf_md5' in sidecar:
        if sidecar['digest_xdf_md5'] == digest_md5:
            return {**out, 'fresh': True, 'reason': 'V11 sidecar digest_xdf_md5 matches'}
        return {**out, 'fresh': False,
                'reason': f'V11 sidecar digest_xdf_md5 mismatch'}

    # Fallback: cross-check preproc sidecars
    pre_md5s = {}
    for mod, root in PREPROC_ROOTS.items():
        side = root / f'{sid}.json'
        if side.exists():
            pre_md5s[mod] = json.loads(side.read_text()).get('digest_xdf_md5')
        else:
            pre_md5s[mod] = None
    matching = {m for m, v in pre_md5s.items() if v == digest_md5}
    missing = [m for m, v in pre_md5s.items() if v is None]
    mismatch = [m for m in pre_md5s if pre_md5s[m] is not None
                and pre_md5s[m] != digest_md5]
    if missing or mismatch:
        return {**out, 'fresh': False,
                'reason': f'V11 sidecar lacks digest_xdf_md5; preproc fallback '
                f'incomplete (missing={missing}, mismatch={mismatch})'}
    return {**out, 'fresh': True, 'reason': 'preproc-artifact fallback all match'}


# ── Pose channel loader (Phase 0 DDTW + Phase 1 candidates) ─────────

def pose_channel_npz_candidates(mode: str) -> list[Path]:
    """NPZ files holding per-session z for a pose decision, in priority order.

    'pose_ddtw' -> phase0/pose_ddtw_per_session.npz (Phase 0), falling back to
    the Phase 1 harness re-run of the same channel (pose_pca_per_session.npz);
    any other Phase 1 decision -> phase1_pose/pose_<mode>_per_session.npz.
    """
    if mode == 'pose_ddtw':
        return [PHASE0_DIR / 'pose_ddtw_per_session.npz',
                PHASE1_DIR / 'pose_pca_per_session.npz']
    if mode in POSE_DECISIONS and mode != 'pose_baseline':
        return [PHASE1_DIR / f'pose_{mode[len("pose_"):]}_per_session.npz']
    return []


def load_pose_channel_for_session(sid: str, t_common: np.ndarray, mode: str
                                  ) -> tuple[np.ndarray | None, str]:
    """Load a pose-coupling z for a session and interp it onto V11 t_common.

    Every candidate NPZ stores z at 2 Hz on stream-relative seconds under
    `{sid}__z` (Phase 1) or `{sid}__ddtw_z` (Phase 0) with `{sid}__stride_ts`;
    V11 t_common is absolute LSL, so the digest's t_start_lsl converts.
    Returns (z_on_v11_grid, note); z is None when the channel is unavailable.
    """
    candidates = pose_channel_npz_candidates(mode)
    if not candidates:
        return None, f'no NPZ source for pose channel {mode!r}'
    npz_path = next((p for p in candidates if p.exists()), None)
    if npz_path is None:
        return None, f'no {candidates[0].name}'
    with np.load(npz_path) as npz:
        z_key = next((k for k in (f'{sid}__z', f'{sid}__ddtw_z') if k in npz.files), None)
        ts_key = f'{sid}__stride_ts'
        if z_key is None or ts_key not in npz.files:
            return None, f'session {sid} missing from {npz_path.name}'
        z = np.asarray(npz[z_key], dtype=np.float64)
        ts_rel = np.asarray(npz[ts_key], dtype=np.float64)
    # Convert V11 t_common (absolute LSL) to stream-relative
    digest = json.loads((DIGEST_ROOT / f'{sid}.json').read_text())
    t_start_lsl = float(digest.get('t_start_lsl', 0.0))
    t_common_rel = np.asarray(t_common, dtype=np.float64) - t_start_lsl
    # Mask NaNs for safe interp
    finite = np.isfinite(z)
    if finite.sum() < 5:
        return None, f'too few finite samples ({int(finite.sum())}) in {npz_path.name}'
    z_on_grid = np.interp(t_common_rel, ts_rel[finite], z[finite],
                           left=0.0, right=0.0).astype(np.float32)
    return z_on_grid, 'OK'


def load_phase0_ddtw_for_session(sid: str, t_common: np.ndarray
                                   ) -> tuple[np.ndarray | None, str]:
    """Thin wrapper: Phase 0 DDTW channel via `load_pose_channel_for_session`."""
    return load_pose_channel_for_session(sid, t_common, 'pose_ddtw')


# ── Per-session slice ───────────────────────────────────────────────

def slice_session(sid: str, pose_channel: str, freshness: dict) -> dict:
    """Slice 7D obs + 2D cov from V11 scaffold for one session.

    Returns {'sid', 'status', 'n_timepoints', 'pose_source', 'note'}.
    """
    info = {'sid': sid, 'status': 'ok', 'pose_source': pose_channel,
            'note': '', 'n_timepoints': 0}
    if not freshness['v11_scaffold_present']:
        info['status'] = 'skipped'
        info['note'] = 'V11 scaffold missing — must regenerate before MVP slice'
        return info

    v11_npz_path = V11_SCAFFOLD_ROOT / sid / 'scaffold_v11_ztimecourses.npz'
    v11 = np.load(v11_npz_path)
    sidecar = json.loads((V11_SCAFFOLD_ROOT / sid / 'scaffold_v11_results.json'
                           ).read_text())
    modality_keys = sidecar['modality_keys']
    cov_keys = sidecar['covariate_keys']
    t_common = v11['t_common']

    # Pre-compute bl_event_coincidence (face/v1 preproc + auto-detected
    # timestamp offset) — see cadence/significance/face_event_coincidence.py
    # for rationale.
    face_path = REPO_ROOT / 'data' / 'preproc' / 'face' / 'v1' / f'{sid}.npz'
    digest_path = DIGEST_ROOT / f'{sid}.json'
    bl_evt_z = None
    bl_evt_valid = None
    bl_evt_info = None
    if face_path.exists() and digest_path.exists():
        from cadence.significance.face_event_coincidence import (
            compute_bl_event_coincidence,
        )
        face_npz = dict(np.load(face_path))
        digest = json.loads(digest_path.read_text())
        t_start_lsl = digest.get('t_start_lsl', 0.0)

        # Auto-detect timestamp convention: face_ts may be session-local
        # (≈0–10000) or Unix epoch (≈1.7e9) depending on session vintage.
        # The pipeline docstring says "session-relative" but enforcement is
        # inconsistent. Pick the offset that lands face_ts in t_common's range.
        if 'p1_au52_ts' in face_npz:
            face_ts0 = float(face_npz['p1_au52_ts'][0])
        elif 'p2_au52_ts' in face_npz:
            face_ts0 = float(face_npz['p2_au52_ts'][0])
        else:
            face_ts0 = None

        if face_ts0 is not None:
            # Candidate 1: assume session-local → add t_start_lsl
            cand_offset = t_start_lsl
            mapped = face_ts0 + cand_offset
            if not (t_common[0] - 600 < mapped < t_common[-1] + 600):
                # Doesn't fit → align face start with t_common start directly
                cand_offset = float(t_common[0] - face_ts0)
            lsl_offset = cand_offset
        else:
            lsl_offset = t_start_lsl

        bl_evt_z, bl_evt_info = compute_bl_event_coincidence(
            face_npz, t_common, lsl_offset)
        bl_evt_info = bl_evt_info or {}
        bl_evt_info['lsl_offset_used'] = lsl_offset

        # Validity: True for the whole session if the channel computed
        # successfully (face data present + alignment worked + non-degenerate
        # surrogate distribution). Mark the WHOLE session invalid only when
        # the channel itself is unusable; per-frame face dropouts are
        # already handled inside compute_bl_event_coincidence via the
        # au_valid masks. If we inherited V11's bl_expr mask we'd silently
        # drop sessions where Morlet bl_expr happened to fail unrelated to
        # the new channel.
        if (bl_evt_info.get('status') != 'ok'
                or float(bl_evt_z.std()) < 1e-6):
            bl_evt_valid = np.zeros(len(t_common), dtype=bool)
            bl_evt_info['marked_invalid'] = True
        else:
            bl_evt_valid = np.ones(len(t_common), dtype=bool)

    # Build observation matrix
    T = len(t_common)
    obs = np.empty((T, len(MVP_OBS_CHANNELS)), dtype=np.float32)
    obs_valid = np.empty((T, len(MVP_OBS_CHANNELS)), dtype=bool)
    for i, ch in enumerate(MVP_OBS_CHANNELS):
        if ch == 'bl_event_coincidence':
            if bl_evt_z is None:
                info['status'] = 'error'
                info['note'] = ('bl_event_coincidence: missing face/v1 preproc '
                                 'or digest')
                return info
            obs[:, i] = bl_evt_z
            obs_valid[:, i] = bl_evt_valid
            continue
        if ch == 'pose' and pose_channel != 'pose_baseline':
            z, note = load_pose_channel_for_session(sid, t_common, pose_channel)
            if z is None:
                info['status'] = 'fallback'
                info['note'] = (f'{pose_channel} requested but unavailable ({note}); '
                                f'falling back to multi-lag baseline')
                info['pose_source'] = 'pose_baseline_fallback'
                obs[:, i] = v11[f'z_{ch}']
            else:
                obs[:, i] = z
        else:
            key = f'z_{ch}'
            if key not in v11.files:
                info['status'] = 'error'
                info['note'] = f'V11 scaffold missing {key}'
                return info
            obs[:, i] = v11[key]
        # obs_mask in V11 is (T, 26); locate the V11 column for this channel
        v11_idx = modality_keys.index(ch)
        if 'obs_mask' in v11.files:
            obs_valid[:, i] = v11['obs_mask'][:, v11_idx]
        else:
            obs_valid[:, i] = True

    # Build covariate matrix
    cov = np.empty((T, len(MVP_COV_CHANNELS)), dtype=np.float32)
    for i, ch in enumerate(MVP_COV_CHANNELS):
        # V11 stores u_lambda2 (no underscore), but spec uses lambda_2 — handle both
        candidates = [f'u_{ch}', f'u_{ch.replace("_", "")}']
        key = next((k for k in candidates if k in v11.files), None)
        if key is None:
            info['status'] = 'error'
            info['note'] = f'V11 scaffold missing covariate {ch} (tried {candidates})'
            return info
        cov[:, i] = v11[key]

    # Save MVP scaffold
    out_dir = OUT_ROOT / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'mvp_scaffold.npz',
             obs=obs, obs_valid=obs_valid, cov=cov,
             t_common=v11['t_common'])

    # Compute V11 NPZ content hash for provenance
    h = hashlib.sha256()
    with open(v11_npz_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    v11_md5 = h.hexdigest()

    sidecar_out = {
        'session_id': sid,
        'mvp_version': 'v1',
        'n_timepoints': T,
        'fs_out': sidecar.get('fs_out'),
        'duration_s': sidecar.get('duration_s'),
        'observation_channels': MVP_OBS_CHANNELS,
        'covariate_channels': MVP_COV_CHANNELS,
        'pose_source': info['pose_source'],
        'pose_channel_decision': pose_channel,
        'digest_xdf_md5': freshness['digest_xdf_md5'],
        'v11_scaffold_sha256': v11_md5,
        'v11_freshness_check': freshness['reason'],
        'bl_event_coincidence_info': bl_evt_info,
        'written_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    (out_dir / 'mvp_scaffold.json').write_text(json.dumps(sidecar_out, indent=2))

    info['n_timepoints'] = T
    return info


# ── Cohort-level cohort_protocol_assignment.csv ─────────────────────

def build_cohort_table(sids: list[str], pose_decision: str
                        ) -> pd.DataFrame:
    """Per spec §Model spec: cohort_protocol_assignment.csv."""
    rows = []
    for sid in sids:
        digest_path = DIGEST_ROOT / f'{sid}.json'
        if not digest_path.exists():
            rows.append({'session_id': sid, 'protocol': '', 'n_meditation_phases_present': 0,
                         'n_pe_phases_present': 0, 'has_eeg_preproc': False,
                         'included_in_production_fit': False,
                         'status_note': 'no digest'})
            continue
        digest = json.loads(digest_path.read_text())
        protocol = digest.get('protocol', '')
        markers = digest.get('markers', [])
        marker_names = {m for _, m in markers}
        n_med = sum(1 for c in MEDITATION_PHASES if f'{c}_start' in marker_names)
        n_pe = sum(1 for c in PE_PHASES if f'{c}_start' in marker_names)
        has_eeg = (PREPROC_ROOTS['eeg'] / f'{sid}.npz').exists()
        v11_present = (V11_SCAFFOLD_ROOT / sid / 'scaffold_v11_ztimecourses.npz').exists()
        included = has_eeg and v11_present
        notes = []
        if not has_eeg:
            notes.append('missing EEG preproc (likely needs MATLAB clean.mat)')
        if not v11_present:
            notes.append('missing V11 scaffold')
        rows.append({'session_id': sid, 'protocol': protocol,
                     'n_meditation_phases_present': n_med,
                     'n_pe_phases_present': n_pe,
                     'has_eeg_preproc': has_eeg,
                     'v11_scaffold_present': v11_present,
                     'pose_channel': pose_decision,
                     'included_in_production_fit': included,
                     'status_note': '; '.join(notes)})
    df = pd.DataFrame(rows)
    return df


# ── Main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true',
                    help='Run on all canonical sessions')
    ap.add_argument('--session', type=str, default=None,
                    help='Run on a single session')
    ap.add_argument('--pose-channel', choices=list(POSE_CHANNEL_CHOICES),
                    default='auto',
                    help='Pose channel source: auto (phase1_report.md, then '
                    'phase0_report.md, then baseline), baseline (force V11 '
                    'multi-lag), ddtw (force Phase 0 DDTW), or a Phase 1 '
                    'candidate: angles | angle_speed | evt_landing | evt_peak')
    ap.add_argument('--n-jobs', type=int, default=-1)
    args = ap.parse_args()

    if not (args.all or args.session):
        ap.error('Provide --all or --session <sid>')

    if args.all:
        sids = list_canonical_sessions()
    else:
        sids = [args.session]

    # Resolve pose channel decision
    if args.pose_channel == 'auto':
        decision = read_phase_decision()
        if read_phase1_decision() is None and read_phase0_decision() is None:
            print('No phase1_report.md / phase0_report.md found — defaulting to '
                  'multi-lag baseline.')
    else:
        decision = POSE_DECISION_BY_CHOICE[args.pose_channel]
    print(f'Pose channel decision: {decision}')

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Cohort table FIRST — written even if individual slices fail
    cohort_df = build_cohort_table(sids, decision)
    cohort_csv = OUT_ROOT / 'cohort_protocol_assignment.csv'
    cohort_df.to_csv(cohort_csv, index=False)
    print(f'\nCohort table: {cohort_csv}')
    print(cohort_df.to_string(index=False))

    # Per-session slice
    print(f'\n=== Slicing MVP scaffold for {len(sids)} sessions ===')

    def _do_one(sid):
        with limit_blas_threads(1):
            fr = check_freshness(sid)
            if not fr['v11_scaffold_present']:
                return {'sid': sid, 'status': 'skipped',
                        'note': fr['reason'], 'n_timepoints': 0,
                        'pose_source': decision}
            info = slice_session(sid, decision, fr)
            info['freshness'] = fr['reason']
            return info

    # Slice-and-write per session is small NPZ I/O — 0.2 GB per worker max.
    n_jobs_eff = pick_n_jobs(per_worker_ram_gb=0.2, requested=args.n_jobs,
                              max_jobs_hard_cap=len(sids))
    results = Parallel(n_jobs=n_jobs_eff, prefer='threads')(
        delayed(_do_one)(sid) for sid in sids)

    print(f'\n{"sid":25s} | {"status":10s} | T     | pose_source             | note')
    print('-' * 110)
    for r in results:
        print(f'{r["sid"]:25s} | {r["status"]:10s} | {r["n_timepoints"]:5d} | '
              f'{r["pose_source"]:23s} | {r["note"]}')

    # Summary
    n_ok = sum(1 for r in results if r['status'] == 'ok')
    n_skip = sum(1 for r in results if r['status'] == 'skipped')
    n_err = sum(1 for r in results if r['status'] == 'error')
    n_fb = sum(1 for r in results if r['status'] == 'fallback')
    print(f'\nSummary: {n_ok} OK, {n_fb} fallback, {n_skip} skipped, {n_err} errors')


if __name__ == '__main__':
    main()
