"""Step 4: fine alignment per matched cluster.

Cross-correlation of the two clusters' stem-base images (low-layer points,
5 mm cells, 5 mm smoothing), searched within +-5 cm, with a sub-cell peak.
(Point-to-point ICP was tried first and rejected: on these dense blobs it
recovered only 0-50 % of a known 1 cm test shift.)

The result is reported with the image correlation before and after; a fine
shift is accepted only if it is small (<= icp_max_shift) and improves the
correlation.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .clusters import cluster_points
from .config import ClusterConfig

CELL = 0.005
SMOOTH_CELLS = 1.0
PAD = 0.08


def _image(xy, x0, y0, nx, ny):
    g = np.zeros((nx, ny))
    ix = ((xy[:, 0] - x0) / CELL).astype(int)
    iy = ((xy[:, 1] - y0) / CELL).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    np.add.at(g, (ix[ok], iy[ok]), 1)
    return gaussian_filter(g, SMOOTH_CELLS)


def _ncc(a, b) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def xcorr_shift(ref_xy: np.ndarray, mov_xy: np.ndarray, max_shift: float) -> dict:
    """Shift (m) that moves mov onto ref, with the correlation before / after."""
    if len(ref_xy) < 20 or len(mov_xy) < 20:
        return {"dx": 0.0, "dy": 0.0, "ncc_before": np.nan, "ncc_after": np.nan, "at_limit": False,
                "reason": "too few points"}
    allp = np.vstack([ref_xy, mov_xy])
    x0, y0 = allp.min(0) - PAD
    x1, y1 = allp.max(0) + PAD
    nx, ny = int((x1 - x0) / CELL) + 1, int((y1 - y0) / CELL) + 1
    A, B = _image(ref_xy, x0, y0, nx, ny), _image(mov_xy, x0, y0, nx, ny)
    a, b = A - A.mean(), B - B.mean()
    c = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b)))))
    cy, cx = np.array(c.shape) // 2
    m = int(round(max_shift / CELL))
    w = c[cy - m:cy + m + 1, cx - m:cx + m + 1]
    i, j = np.unravel_index(int(np.argmax(w)), w.shape)

    def sub(v_m, v_0, v_p):
        d = v_m - 2 * v_0 + v_p
        return 0.5 * (v_m - v_p) / d if abs(d) > 1e-12 else 0.0
    di = sub(w[i - 1, j], w[i, j], w[i + 1, j]) if 0 < i < w.shape[0] - 1 else 0.0
    dj = sub(w[i, j - 1], w[i, j], w[i, j + 1]) if 0 < j < w.shape[1] - 1 else 0.0
    dx, dy = (i + di - m) * CELL, (j + dj - m) * CELL
    at_limit = i in (0, w.shape[0] - 1) or j in (0, w.shape[1] - 1)
    after = _ncc(A, _image(mov_xy + [dx, dy], x0, y0, nx, ny))
    return {"dx": float(dx), "dy": float(dy), "ncc_before": _ncc(A, B), "ncc_after": after,
            "at_limit": bool(at_limit), "reason": ""}


def fine_align(run, clusters: dict, links_all: dict, ref: str, cfg: ClusterConfig, ref_names: dict, log=print):
    R = [c for c in clusters[ref] if not c.clod]
    rows = []
    for d, links in links_all.items():
        Q = [c for c in clusters[d] if not c.clod]
        acc = []
        for L in links:
            if not L["ref"] or not L["date"]:
                continue
            Rs, Qs = [R[i] for i in L["ref"]], [Q[j] for j in L["date"]]
            ref_xy = np.vstack([cluster_points(run, c, cfg) for c in Rs])
            mov_xy = np.vstack([cluster_points(run, c, cfg) for c in Qs])
            res = xcorr_shift(ref_xy, mov_xy, cfg.fine_search)
            mag = float(np.hypot(res["dx"], res["dy"]))
            gain_ok = res["ncc_after"] >= res["ncc_before"] + cfg.fine_min_gain
            ok = not res["reason"] and not res["at_limit"] and mag <= cfg.fine_max_accept and gain_ok
            reason = res["reason"] or ("shift at the search limit" if res["at_limit"] else
                                       f"larger than {100 * cfg.fine_max_accept:.0f} cm - check" if mag > cfg.fine_max_accept else
                                       "" if ok else "correlation not better")
            for r in Rs:
                rows.append({"reference_cluster": r.cid, "name": ref_names[r.cid], "date": d,
                             "date_clusters": " ".join(q.cid for q in Qs),
                             "dx_cm": round(100 * res["dx"], 2), "dy_cm": round(100 * res["dy"], 2),
                             "shift_cm": round(100 * mag, 2),
                             "corr_before": round(res["ncc_before"], 3), "corr_after": round(res["ncc_after"], 3),
                             "accepted": ok, "reason": reason})
            if ok:
                acc.append((res["dx"], res["dy"]))
        if acc:
            a = 100 * np.array(acc)
            log(f"  {d}: {len(acc)} of {sum(1 for L in links if L['ref'] and L['date'])} fine shifts accepted, "
                f"median dx {np.median(a[:, 0]):+.1f} cm, dy {np.median(a[:, 1]):+.1f} cm, "
                f"median size {np.median(np.hypot(a[:, 0], a[:, 1])):.1f} cm")
    return rows
