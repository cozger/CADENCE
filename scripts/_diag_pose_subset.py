"""Step 4 diagnostic: render the first 100 frames of each digest's pose stream
through ``normalize_pose_to_mp33`` and verify the resulting 33-keypoint skeleton
is anatomically correct.

Renders, per session:
    * Frame-overlay scatter for 5 sample frames (one per 20 frames in [0, 100)).
    * MediaPipe-skeleton lines connecting the standard joint pairs.
    * Visibility heatmap for the first 100 frames (33 keypoints × 100 frames).

Also benchmarks ``normalize_pose_to_mp33`` for the full pose stream so we can
verify the < 1s/55k-frame target.

Usage:
    python scripts/_diag_pose_subset.py                       # all digested sessions
    python scripts/_diag_pose_subset.py --session y_06        # single session
    python scripts/_diag_pose_subset.py --out-dir results/audit/pose_subset
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Hoist torch BEFORE numpy/matplotlib on Windows torch 2.10+numpy 2.4 stacks.
# numpy's MKL installs a DLL-search hook that breaks torch's later shm.dll
# load; matplotlib pulls in numpy, so torch must come first.
# (Per `project_win_torch_dll_fix` memory.)
import torch  # noqa: F401

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from cadence.io.paths import DIGEST_DIR
from cadence.ingest.digest import load_digest
from cadence.preprocess.pose.pose_subset import normalize_pose_to_mp33


# MediaPipe Pose Landmarker skeleton edges (standard 33-point connection list).
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
MP_EDGES = [
    # Head
    (0, 2), (2, 7), (0, 5), (5, 8),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15),
    # Right arm
    (12, 14), (14, 16),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


def _plot_session(sid: str, p_role: str, pose_raw: np.ndarray, pose_format: str,
                  out_dir: Path) -> dict:
    """Render diagnostics for one (session, participant) and return a metrics dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Benchmark + normalize ----
    t0 = time.perf_counter()
    pose_33 = normalize_pose_to_mp33(pose_raw, pose_format)
    dt_full = time.perf_counter() - t0
    n_frames = pose_33.shape[0]
    rate = n_frames / dt_full if dt_full > 0 else float("inf")

    # Visibility coverage (mean visibility across first 100 frames per slot).
    n_show = min(100, n_frames)
    vis_chunk = pose_33[:n_show, :, 3]  # (n_show, 33)
    mean_vis = vis_chunk.mean(axis=0)
    pct_visible_per_frame = (vis_chunk > 0.5).mean(axis=1) * 100.0  # (n_show,)

    # ---- Figure: 5 frame snapshots + visibility heatmap ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(f"{sid}  ({p_role})  pose_format={pose_format}  "
                 f"shape={pose_raw.shape}", fontsize=11)

    sample_idx = np.linspace(0, max(n_show - 1, 0), 5, dtype=int)
    for k, fi in enumerate(sample_idx):
        ax = axes.flat[k]
        coords = pose_33[fi]            # (33, 4)
        x, y, vis = coords[:, 0], coords[:, 1], coords[:, 3]
        # MediaPipe convention: y grows downward.
        # Flip for visual readability.
        y_disp = -y
        # Plot edges (only between pairs both visible).
        for a, b in MP_EDGES:
            if vis[a] > 0.5 and vis[b] > 0.5:
                ax.plot([x[a], x[b]], [y_disp[a], y_disp[b]],
                        color="steelblue", lw=1.4, alpha=0.7)
        ax.scatter(x[vis > 0.5], y_disp[vis > 0.5],
                   s=24, color="crimson", edgecolor="white", linewidth=0.5,
                   label="visible")
        zero_idx = np.where(vis <= 0.5)[0]
        ax.scatter(x[zero_idx], y_disp[zero_idx],
                   s=10, color="lightgray", alpha=0.4, label="zeroed/low-vis")
        ax.set_title(f"frame {fi}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)

    # Visibility heatmap subplot (axes.flat[5]).
    ax_h = axes.flat[5]
    im = ax_h.imshow(vis_chunk.T, aspect="auto", interpolation="nearest",
                     cmap="viridis", vmin=0, vmax=1)
    ax_h.set_title("visibility (first 100 frames)")
    ax_h.set_xlabel("frame")
    ax_h.set_ylabel("keypoint slot (0..32)")
    fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)

    out_path = out_dir / f"{sid}__{p_role}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    return {
        "session_id": sid,
        "participant": p_role,
        "pose_format": pose_format,
        "n_frames": n_frames,
        "gather_seconds": dt_full,
        "frames_per_second": rate,
        "mean_visibility_per_slot": mean_vis.tolist(),
        "mean_pct_keypoints_visible": float(pct_visible_per_frame.mean()),
        "fig_path": str(out_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", action="append", default=[],
                    help="Session ID (repeatable). Default: every digest in data/digest/v1/.")
    ap.add_argument("--digest-dir", default=str(DIGEST_DIR))
    ap.add_argument("--out-dir", default="results/audit/pose_subset",
                    help="Where to write the diagnostic plots.")
    args = ap.parse_args()

    digest_dir = Path(args.digest_dir)
    out_dir = Path(args.out_dir)

    if args.session:
        sids = args.session
    else:
        sids = sorted(p.stem for p in digest_dir.glob("*.npz"))

    if not sids:
        print(f"[diag_pose_subset] No digests found in {digest_dir}", file=sys.stderr)
        return 1

    metrics: list[dict] = []
    for sid in sids:
        try:
            cs = load_digest(sid, digest_dir=digest_dir)
        except FileNotFoundError as e:
            print(f"[diag_pose_subset] SKIP {sid}: {e}", file=sys.stderr)
            continue
        for p in ("p1", "p2"):
            data_key = f"{p}_pose_full"
            if data_key not in cs.arrays:
                continue
            role = f"{p}_{getattr(cs.roles, p + '_role')}"
            m = _plot_session(sid, role, cs.arrays[data_key], cs.pose_format, out_dir)
            metrics.append(m)
            print(f"  {sid}/{role}  fmt={cs.pose_format}  n={m['n_frames']:>7d}  "
                  f"gather={m['gather_seconds']*1000:.1f}ms  "
                  f"({m['frames_per_second']/1000:.0f}k frames/s)  "
                  f"mean_vis_kp%={m['mean_pct_keypoints_visible']:.1f}  -> {m['fig_path']}")

    # Summary report
    if metrics:
        max_n = max(m["n_frames"] for m in metrics)
        slowest = max(metrics, key=lambda m: m["gather_seconds"])
        print("\n=== Summary ===")
        print(f"  Sessions x participants: {len(metrics)}")
        print(f"  Max frames in a stream:  {max_n}")
        print(f"  Slowest gather:          {slowest['session_id']}/{slowest['participant']} "
              f"{slowest['gather_seconds']*1000:.1f}ms for {slowest['n_frames']} frames "
              f"(format={slowest['pose_format']})")
        # Acceptance: gather < 1s for 55k frames.
        slowest_per_55k = slowest["gather_seconds"] / max(slowest["n_frames"], 1) * 55000
        print(f"  Projected to 55k frames:  {slowest_per_55k*1000:.1f}ms  "
              f"({'PASS' if slowest_per_55k < 1.0 else 'FAIL'} <1s target)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
