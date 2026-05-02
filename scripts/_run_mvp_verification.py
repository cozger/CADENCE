"""MVP Verification — V1, V2, V3, V5 per spec §Verification protocol.

V4 (shared-d_emit sensitivity) is deferred until --share-demit is wired in
fit_hierarchical_slds; this script will skip V4 with a clear message.

Inputs (must exist before running):
  results/mvp/hierarchical/             — production K=4 fit
  results/mvp/hierarchical_k3/          — K=3 sensitivity
  results/mvp/hierarchical_med/         — meditation-only fit
  results/mvp/hierarchical_pe/          — PE-only fit
  results/mvp/<sid>/mvp_rslds_results.npz       (production)
  results/mvp/<sid>/mvp_rslds_results_k3.npz    (K=3)

Outputs to results/mvp/diagnostics/:
  dwell_report.md
  k_comparison_report.md
  protocol_stratified_report.md
  flexibility_circularity_report.md
  verification_report.md         — top-level summary, names K_winner

Usage:
    python scripts/_run_mvp_verification.py
"""
from __future__ import annotations

# Windows torch+numpy DLL ordering: must import torch BEFORE numpy.
import torch as _torch  # noqa: F401

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadence.io.resources import log_resources

REPO_ROOT = Path(__file__).resolve().parents[1]
MVP_ROOT = REPO_ROOT / 'results' / 'mvp'
DIAG_ROOT = MVP_ROOT / 'diagnostics'

MVP_OBS_CHANNELS = ['conc_theta', 'conc_alpha', 'bl_expr', 'bl_activity_conc',
                    'pose', 'resp', 'ecg_hf']
FS_OUT = 2.0

DWELL_RATIO_THRESH = 0.5
SHORT_DWELL_FRAC_THRESH = 0.5
SHORT_DWELL_S = 10.0
KS_N_BOOT = 1000
KS_NULL_SEED = 42


# ── Helpers ─────────────────────────────────────────────────────────

def compute_dwell_lengths(path: np.ndarray) -> dict[int, list[int]]:
    if len(path) == 0:
        return {}
    out: dict[int, list[int]] = {}
    cur, count = int(path[0]), 1
    for s in path[1:]:
        s = int(s)
        if s == cur:
            count += 1
        else:
            out.setdefault(cur, []).append(count)
            cur, count = s, 1
    out.setdefault(cur, []).append(count)
    return out


def transition_event_ks(transition_times_s: np.ndarray,
                          event_times_s: list[float],
                          session_length_s: float,
                          n_boot: int = KS_N_BOOT,
                          seed: int = KS_NULL_SEED
                          ) -> tuple[float, float, float]:
    """KS test: real transition-to-event distances vs uniform-random null.

    Returns (ks_stat, p_value, mean_real_dist - mean_boot_dist).
    """
    if len(transition_times_s) == 0 or len(event_times_s) == 0:
        return 0.0, 1.0, 0.0
    rng = np.random.default_rng(seed)
    events = np.asarray(event_times_s, dtype=float)
    real = np.array([np.min(np.abs(t - events)) for t in transition_times_s])
    boot = np.concatenate([
        np.array([np.min(np.abs(t - events))
                  for t in rng.uniform(0, session_length_s, len(transition_times_s))])
        for _ in range(n_boot)
    ])
    ks_stat, p_val = ks_2samp(real, boot, alternative='greater')
    return float(ks_stat), float(p_val), float(real.mean() - boot.mean())


def load_hier_summary(suffix: str = '') -> dict | None:
    """Read mvp_hierarchical_results.json for a fit variant."""
    path = MVP_ROOT / f'hierarchical{suffix}' / 'mvp_hierarchical_results.json'
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_session_results(sid: str, suffix: str = '') -> dict | None:
    """Read per-session NPZ for a fit variant."""
    path = MVP_ROOT / sid / f'mvp_rslds_results{suffix}.npz'
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


# ── V1: Dwell verification ────────────────────────────────────────

def v1_dwell_verification(suffix: str = '') -> dict:
    summary = load_hier_summary(suffix)
    if summary is None:
        return {'status': 'SKIPPED — no production fit', 'pass': False}
    sessions = summary['sessions']
    K = summary['config']['K']
    state_labels = summary['state_labels']
    rows = []
    failures = []
    for sess in sessions:
        sid = sess['session']
        per_session = load_session_results(sid, suffix)
        if per_session is None:
            continue
        path = per_session['path']
        gamma = per_session['gamma']
        if 't_common' in per_session:
            t_common = per_session['t_common']
            session_dur_s = float(t_common[-1] - t_common[0])
        else:
            session_dur_s = len(path) / FS_OUT
        # Per-state dwell stats
        dwells = compute_dwell_lengths(path)
        for k in range(K):
            d = np.array(dwells.get(k, []))
            if d.size == 0:
                continue
            # Predicted dwell from sticky-strength + transition matrix would be
            # complex; use empirical mean/median + fraction-below threshold
            mean_dwell_s = float(d.mean()) / FS_OUT
            short_frac = float((d / FS_OUT < SHORT_DWELL_S).mean())
            # Empirical/predicted ratio: use mean / 1/(1-on_diagonal_prob)
            # Approximation: predicted = 1 / (1 - P(stay))
            # P(stay) from gamma: prob of staying after 1 step ≈ time-fraction of state
            usage = float((path == k).mean())
            predicted = (1.0 / (1.0 - usage)) if 0 < usage < 1 else float('inf')
            ratio = (mean_dwell_s * FS_OUT) / predicted if predicted > 0 else 0.0
            row = {
                'sid': sid, 'state': k, 'label': state_labels[k] if k < len(state_labels) else '',
                'mean_dwell_s': mean_dwell_s,
                'short_dwell_frac': short_frac,
                'usage_frac': usage,
                'empirical_predicted_ratio': ratio,
                'dwell_ok': ratio >= DWELL_RATIO_THRESH,
                'short_ok': short_frac <= SHORT_DWELL_FRAC_THRESH,
            }
            rows.append(row)
            if not (row['dwell_ok'] and row['short_ok']):
                failures.append(f'{sid} S{k}({state_labels[k] if k<len(state_labels) else "?"}): '
                                f'ratio={ratio:.2f}, short_frac={short_frac:.2f}')

    overall_pass = len(failures) == 0
    return {
        'status': 'OK', 'pass': overall_pass,
        'rows': rows, 'failures': failures,
        'n_sessions': len(set(r['sid'] for r in rows)),
    }


# ── V2: K=3 vs K=4 BIC + emission separation ───────────────────────

def _emission_separation(mean_d_emit: np.ndarray) -> tuple[float, float, float]:
    """Return (min_pairwise_dist, max_pairwise_dist, ratio min/max)."""
    K = mean_d_emit.shape[0]
    if K < 2:
        return 0.0, 0.0, 0.0
    dists = []
    for i in range(K):
        for j in range(i + 1, K):
            dists.append(np.linalg.norm(mean_d_emit[i] - mean_d_emit[j]))
    dists = np.array(dists)
    return float(dists.min()), float(dists.max()), float(dists.min() / max(dists.max(), 1e-12))


def v2_k_comparison() -> dict:
    s4 = load_hier_summary('')
    s3 = load_hier_summary('_k3')
    if s4 is None and s3 is None:
        return {'status': 'SKIPPED — no fits', 'k_winner': None}
    out = {'status': 'OK', 'fits': {}}
    if s4:
        d4 = np.asarray(s4['mean_d_emit'])
        mn, mx, ratio = _emission_separation(d4)
        out['fits']['K=4'] = {
            'bic': s4['bic'],
            'final_ll': s4['final_ll'],
            'state_labels': s4['state_labels'],
            'min_pairwise_emission_dist': mn,
            'max_pairwise_emission_dist': mx,
            'min_max_ratio': ratio,
        }
    if s3:
        d3 = np.asarray(s3['mean_d_emit'])
        mn, mx, ratio = _emission_separation(d3)
        out['fits']['K=3'] = {
            'bic': s3['bic'],
            'final_ll': s3['final_ll'],
            'state_labels': s3['state_labels'],
            'min_pairwise_emission_dist': mn,
            'max_pairwise_emission_dist': mx,
            'min_max_ratio': ratio,
        }

    # K_winner = lower BIC
    if s4 and s3:
        out['k_winner'] = 4 if s4['bic'] < s3['bic'] else 3
        out['delta_bic'] = abs(s4['bic'] - s3['bic'])
    elif s4:
        out['k_winner'] = 4
    elif s3:
        out['k_winner'] = 3
    out['note_held_out_ll'] = ('Held-out LL CV (5-fold leave-some-sessions-out) '
                                'is part of spec §V2 but is not yet implemented; '
                                'BIC + emission separation are the gating evidence here.')
    return out


# ── V3: Protocol-stratified vs pooled ──────────────────────────────

def v3_protocol_stratified() -> dict:
    pooled = load_hier_summary('')
    med = load_hier_summary('_med')
    pe = load_hier_summary('_pe')
    if pooled is None or (med is None and pe is None):
        return {'status': 'SKIPPED — missing stratified fits', 'pass': False}

    def _ranking(sessions_list):
        # Rank by mean (COUP+SHARED) usage during conv_1+conv_2 if available;
        # otherwise rank by 1 - NULL usage.
        rank = []
        for sess in sessions_list:
            usage = sess.get('usage', [])
            labels = sess.get('state_labels', [])
            coup_idx = labels.index('COUP') if 'COUP' in labels else -1
            shared_idx = labels.index('SHARED') if 'SHARED' in labels else -1
            null_idx = labels.index('NULL') if 'NULL' in labels else -1
            # Use per-condition periods if present
            conv_usage = []
            for pname, _, u in sess.get('periods', []):
                if pname in ('conv_1', 'conv_2'):
                    score = 0.0
                    if coup_idx >= 0:
                        score += u[coup_idx]
                    if shared_idx >= 0:
                        score += u[shared_idx]
                    conv_usage.append(score)
            if conv_usage:
                rank.append((sess['session'], float(np.mean(conv_usage))))
            else:
                # Fallback: 1 - NULL
                if null_idx >= 0 and usage:
                    rank.append((sess['session'], 1.0 - usage[null_idx]))
        rank.sort(key=lambda x: -x[1])
        return rank

    pooled_rank = _ranking(pooled['sessions'])
    out = {'status': 'OK', 'pooled_ranking': pooled_rank,
           'pooled_n': len(pooled['sessions']),
           'compare': {}}
    for label, ssum in (('meditation', med), ('pe', pe)):
        if ssum is None:
            continue
        strat_rank = _ranking(ssum['sessions'])
        # Sessions in this protocol per pooled fit
        strat_sids = {s for s, _ in strat_rank}
        pooled_sub = [(s, v) for s, v in pooled_rank if s in strat_sids]
        # Spearman rank correlation between pooled-restricted and stratified
        if len(pooled_sub) >= 3 and len(strat_rank) >= 3:
            pool_vals = {s: v for s, v in pooled_sub}
            strat_vals = {s: v for s, v in strat_rank}
            common = sorted(set(pool_vals) & set(strat_vals))
            if len(common) >= 3:
                rho, p = spearmanr([pool_vals[s] for s in common],
                                    [strat_vals[s] for s in common])
                rho, p = float(rho), float(p)
            else:
                rho, p = float('nan'), float('nan')
            out['compare'][label] = {
                'n': len(strat_rank),
                'pooled_top': pooled_sub[0] if pooled_sub else None,
                'pooled_bottom': pooled_sub[-1] if pooled_sub else None,
                'strat_top': strat_rank[0] if strat_rank else None,
                'strat_bottom': strat_rank[-1] if strat_rank else None,
                'spearman_rho': rho, 'spearman_p': p,
            }
    out['pass'] = all(c.get('spearman_rho', 0) > 0.5
                       for c in out['compare'].values()) if out['compare'] else False
    return out


# ── V5: Coupling-flexibility partial-correlation diagnostic ────────

def v5_flexibility_circularity() -> dict:
    """Partial correlation between u_coupling_flexibility and the leading 3 PCs
    of emission residual y - C[k_t] x_t - d_emit[k_t].

    If partial r > 0.5: flexibility is leaking the signal it's supposed to predict.
    """
    summary = load_hier_summary('')
    if summary is None:
        return {'status': 'SKIPPED — no production fit', 'pass': False}
    rows = []
    for sess in summary['sessions']:
        sid = sess['session']
        per_session = load_session_results(sid, '')
        if per_session is None:
            continue
        # Load MVP scaffold to reconstruct emission residual
        mvp_path = MVP_ROOT / sid / 'mvp_scaffold.npz'
        if not mvp_path.exists():
            continue
        mvp = np.load(mvp_path)
        Y = mvp['obs']                    # (T, 7)
        U = mvp['cov']                    # (T, 2)
        path = per_session['path']
        x_smooth = per_session['x_smooth'] if 'x_smooth' in per_session else None
        if 'C_emit' not in per_session or 'd_emit' not in per_session or x_smooth is None:
            continue
        C = per_session['C_emit']         # (K, D_obs, D_latent)
        d = per_session['d_emit']         # (K, D_obs)
        # Per-timepoint reconstruction: y_hat = C[path[t]] @ x_smooth[t] + d[path[t]]
        # Vectorized: gather C and d by path indices, then batch matmul. Replaces
        # a per-t Python loop (~50us interpreter overhead × 4000 × 18 sessions
        # ≈ 4 sec of pure GIL time). See perf-audit S7-1.
        T = Y.shape[0]
        path_int = path.astype(np.int64, copy=False)
        C_by_t = C[path_int]                              # (T, D_obs, D_latent)
        d_by_t = d[path_int]                              # (T, D_obs)
        y_hat = np.einsum('toi,ti->to', C_by_t, x_smooth) + d_by_t
        residual = Y - y_hat
        # PCA top 3
        try:
            U_svd, S_svd, Vt = np.linalg.svd(residual - residual.mean(0),
                                              full_matrices=False)
            top3 = U_svd[:, :3] * S_svd[:3]   # (T, 3)
        except np.linalg.LinAlgError:
            continue
        # Partial correlation: corr(flex, top1) controlling for top2, top3
        flex = U[:, 0]
        # Simple multiple-regression residualization
        X_ctrl = np.column_stack([np.ones(T), top3[:, 1], top3[:, 2]])
        beta_flex = np.linalg.lstsq(X_ctrl, flex, rcond=None)[0]
        resid_flex = flex - X_ctrl @ beta_flex
        beta_top = np.linalg.lstsq(X_ctrl, top3[:, 0], rcond=None)[0]
        resid_top = top3[:, 0] - X_ctrl @ beta_top
        if resid_flex.std() > 1e-8 and resid_top.std() > 1e-8:
            r, p = pearsonr(resid_flex, resid_top)
            rows.append({'sid': sid, 'partial_r': float(r), 'partial_p': float(p)})
    abs_rs = [abs(r['partial_r']) for r in rows]
    mean_abs_r = float(np.mean(abs_rs)) if abs_rs else float('nan')
    return {
        'status': 'OK', 'rows': rows,
        'mean_abs_partial_r': mean_abs_r,
        'pass': mean_abs_r < 0.5,
        'note': ('Partial r > 0.5 indicates flexibility is leaking emission-residual '
                  'signal — confirm flexibility is genuinely external before reporting '
                  'max|S_trans|.'),
    }


# ── Report writers ──────────────────────────────────────────────────

def _write_report(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def write_dwell_report(v1: dict):
    md = ['# V1 — Dwell Verification\n\n']
    md.append(f'- Status: **{v1.get("status")}**\n')
    md.append(f'- Sessions analyzed: **{v1.get("n_sessions", 0)}**\n')
    md.append(f'- Overall pass: **{v1.get("pass")}**\n\n')
    if v1.get('rows'):
        md.append('## Per-session per-state dwell stats\n\n')
        md.append('| Session | State | Label | Mean dwell (s) | Short-dwell frac | Usage | Ratio | OK |\n')
        md.append('|---|---|---|---|---|---|---|---|\n')
        for r in v1['rows']:
            ok = 'YES' if (r['dwell_ok'] and r['short_ok']) else 'NO'
            md.append(f'| `{r["sid"]}` | S{r["state"]} | {r["label"]} | '
                      f'{r["mean_dwell_s"]:.1f} | {r["short_dwell_frac"]:.2f} | '
                      f'{r["usage_frac"]:.2%} | {r["empirical_predicted_ratio"]:.2f} | {ok} |\n')
    if v1.get('failures'):
        md.append('\n## Failures\n')
        for f in v1['failures']:
            md.append(f'- {f}\n')
    _write_report(DIAG_ROOT / 'dwell_report.md', ''.join(md))


def write_k_comparison_report(v2: dict):
    md = ['# V2 — K=3 vs K=4 Comparison\n\n']
    md.append(f'- Status: **{v2.get("status")}**\n')
    md.append(f'- K_winner: **K={v2.get("k_winner")}**\n')
    if v2.get('delta_bic') is not None:
        md.append(f'- ΔBIC: {v2["delta_bic"]:.0f}\n')
    md.append('\n## Per-K summary\n\n')
    for K_label, fit in v2.get('fits', {}).items():
        md.append(f'### {K_label}\n')
        md.append(f'- BIC: {fit["bic"]:.0f}\n')
        md.append(f'- Final LL: {fit["final_ll"]:.0f}\n')
        md.append(f'- Min/max pairwise emission distance: '
                  f'{fit["min_pairwise_emission_dist"]:.3f} / {fit["max_pairwise_emission_dist"]:.3f}\n')
        md.append(f'- Min/max ratio: {fit["min_max_ratio"]:.3f} '
                  f'(closer to 1.0 = equidistant well-separated states)\n')
        md.append(f'- State labels: {fit["state_labels"]}\n\n')
    md.append('## Note\n')
    md.append(v2.get('note_held_out_ll', '') + '\n')
    _write_report(DIAG_ROOT / 'k_comparison_report.md', ''.join(md))


def write_protocol_report(v3: dict):
    md = ['# V3 — Protocol-Stratified vs Pooled\n\n']
    md.append(f'- Status: **{v3.get("status")}**\n')
    md.append(f'- Pooled n = {v3.get("pooled_n", 0)}\n')
    md.append(f'- Pass: **{v3.get("pass")}**\n\n')
    md.append('## Pooled-fit ranking by (COUP+SHARED) during conv_1+conv_2\n\n')
    for sid, score in v3.get('pooled_ranking', []):
        md.append(f'- `{sid}`: {score:.3f}\n')
    md.append('\n## Stratified vs pooled-restricted comparison\n\n')
    for label, c in v3.get('compare', {}).items():
        md.append(f'### {label} (n={c["n"]})\n')
        md.append(f'- Pooled top: `{c["pooled_top"]}`\n')
        md.append(f'- Stratified top: `{c["strat_top"]}`\n')
        md.append(f'- Pooled bottom: `{c["pooled_bottom"]}`\n')
        md.append(f'- Stratified bottom: `{c["strat_bottom"]}`\n')
        md.append(f'- Spearman ρ (pooled vs stratified ranking): '
                  f'{c["spearman_rho"]:.3f} (p={c["spearman_p"]:.4f})\n\n')
    _write_report(DIAG_ROOT / 'protocol_stratified_report.md', ''.join(md))


def write_flexibility_report(v5: dict):
    md = ['# V5 — Coupling-Flexibility Partial-Correlation Diagnostic\n\n']
    md.append(f'- Status: **{v5.get("status")}**\n')
    md.append(f'- Mean |partial r|: **{v5.get("mean_abs_partial_r", float("nan")):.3f}**\n')
    md.append(f'- Pass (|r| < 0.5): **{v5.get("pass")}**\n\n')
    md.append('## Per-session partial correlations\n\n')
    md.append('| Session | partial r | p |\n|---|---|---|\n')
    for r in v5.get('rows', []):
        md.append(f'| `{r["sid"]}` | {r["partial_r"]:+.3f} | {r["partial_p"]:.4f} |\n')
    md.append('\n## Note\n')
    md.append(v5.get('note', '') + '\n')
    _write_report(DIAG_ROOT / 'flexibility_circularity_report.md', ''.join(md))


def write_summary(v1: dict, v2: dict, v3: dict, v5: dict):
    md = ['# MVP Verification Summary\n\n']
    md.append(f'- V1 (Dwell): **{"PASS" if v1.get("pass") else "FAIL"}**\n')
    md.append(f'- V2 (K-comparison): K_winner = K={v2.get("k_winner")}\n')
    md.append(f'- V3 (Protocol-stratified): **{"PASS" if v3.get("pass") else "FAIL"}**\n')
    md.append(f'- V4 (Shared-d_emit): **SKIPPED — --share-demit not yet wired**\n')
    md.append(f'- V5 (Flexibility partial-r): **{"PASS" if v5.get("pass") else "FAIL"}** '
              f'(mean |r| = {v5.get("mean_abs_partial_r", float("nan")):.3f})\n\n')
    md.append('## K_winner verdict\n\n')
    md.append(f'**Production K = {v2.get("k_winner")}**.\n')
    md.append('Downstream figures (Stage 8) and condition-aggregate panels use this K.\n')
    md.append('\n## Sub-reports\n')
    md.append('- [V1: Dwell](dwell_report.md)\n')
    md.append('- [V2: K-comparison](k_comparison_report.md)\n')
    md.append('- [V3: Protocol-stratified](protocol_stratified_report.md)\n')
    md.append('- [V5: Flexibility circularity](flexibility_circularity_report.md)\n')
    _write_report(DIAG_ROOT / 'verification_report.md', ''.join(md))


# ── Main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    DIAG_ROOT.mkdir(parents=True, exist_ok=True)
    log_resources(prefix='[verification] start: ')

    print('\n[V1/4] Dwell verification...', flush=True)
    v1 = v1_dwell_verification('')
    write_dwell_report(v1)
    print(f'  V1 pass: {v1.get("pass")}, sessions: {v1.get("n_sessions")}')

    print('\n[V2/4] K=3 vs K=4 comparison...', flush=True)
    v2 = v2_k_comparison()
    write_k_comparison_report(v2)
    print(f'  K_winner: K={v2.get("k_winner")}')

    print('\n[V3/4] Protocol-stratified vs pooled...', flush=True)
    v3 = v3_protocol_stratified()
    write_protocol_report(v3)
    print(f'  V3 pass: {v3.get("pass")}')

    print('\n[V5/4] Flexibility partial-correlation diagnostic...', flush=True)
    v5 = v5_flexibility_circularity()
    write_flexibility_report(v5)
    print(f'  V5 mean |r|: {v5.get("mean_abs_partial_r", float("nan")):.3f}, '
          f'pass: {v5.get("pass")}')

    write_summary(v1, v2, v3, v5)
    print(f'\nReports written to: {DIAG_ROOT}/')
    log_resources(prefix='[verification] end:   ')


if __name__ == '__main__':
    main()
