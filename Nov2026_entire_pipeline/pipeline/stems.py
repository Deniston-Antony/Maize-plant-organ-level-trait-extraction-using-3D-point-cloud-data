"""Stage 4: stem bases, clod/weed filter and groups (geometry only).

A stem is a thin column of points rising steadily from the soil. Steps:

 1. stem-base layer: plant points 2-15 cm above ground (seedlings: up to their
    height). Density map at 5 mm -> local maxima = candidates.
 2. column test per candidate: in thin height bins from 2 cm upward, enough
    points must lie within a small radius of the (leaning) column, in most
    bins, with a small horizontal spread. Soil clods (wide, low) and leaves
    lying near the soil (one height only) fail.
 3. candidates closer than `min_stem_dist` keep the stronger one; two strong
    columns 2.5-5 cm apart are kept as two stems (plants in a group).
 4. connected plant blobs (1 cm voxels) without any stem and lower than
    `blob_keep_hag` are clods / weeds / debris and are removed.
 5. groups: stems are ordered along the row (aligned frame) and split where
    the gap between neighbours is large (threshold from the gap distribution).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


@dataclass
class StemConfig:
    cell: float = 0.005               # density map cell
    base_hag: tuple = (0.02, 0.15)    # stem-base layer
    smooth_cells: float = 1.5
    peak_window: float = 0.025        # local-maximum window (m)
    min_peak_rel: float = 0.04        # peak must reach this share of the 99th-percentile density
    col_radius: float = 0.015         # points within this of the column axis
    col_bin: float = 0.01             # height bin
    col_top: float = 0.12             # test up to this height (or 80 % of the local plant height)
    col_min_top: float = 0.04         # every column is followed at least up to here
    min_plant_height: float = 0.05    # blob must be this tall; 01.06: clods < 4 cm, seedlings 6-8 cm
    col_min_pts: int = 3              # points per bin
    col_min_frac: float = 0.7         # share of bins that must be filled
    col_max_spread: float = 0.012     # median horizontal distance of column points to the axis
    col_max_lean: float = 0.6         # max horizontal drift per metre of height (~31 deg)
    max_hidden_base: float = 0.15     # stem may first appear this high (base covered by leaves)
    tall_top: float = 0.30            # second test: the column must keep rising up to here
    tall_top_frac: float = 0.6        # ... or this share of the local plant height (small plants)
    tall_min_frac: float = 0.6        # GT 230621: stems 0.89-1.0, low leaf parts 0.32-0.43
    min_stem_dist: float = 0.025      # closer candidates are one stem
    blob_voxel: float = 0.01
    blob_keep_hag: float = 0.10       # blobs without a stem are kept only if taller (detached leaves)
    row_max_offset: float = 0.10      # stems farther from the row centre line are weeds / leaf tips
    pool_tol: float = 0.03            # stems of different dates within this are the same plant
    min_dates_seen: int = 2           # a plant must be found on at least this many dates
    group_gap: float = 0.12           # split groups where stems are this far apart along the row
                                      # (29.06: 61 of 84 gaps < 10 cm, group gaps from ~12 cm)

    def to_dict(self) -> dict:
        return asdict(self)


def _peaks(xy: np.ndarray, cfg: StemConfig):
    lo = xy.min(axis=0) - 0.05
    n = np.ceil((xy.max(axis=0) + 0.05 - lo) / cfg.cell).astype(int) + 1
    g = np.zeros(n, np.float32)
    ij = ((xy - lo) / cfg.cell).astype(int)
    np.add.at(g, (ij[:, 0], ij[:, 1]), 1)
    g = ndi.gaussian_filter(g, cfg.smooth_cells)
    w = max(3, int(round(cfg.peak_window / cfg.cell)) | 1)
    thr = cfg.min_peak_rel * np.percentile(g[g > 0], 99) if (g > 0).any() else 0
    pk = (g == ndi.maximum_filter(g, size=w)) & (g > thr)
    i, j = np.where(pk)
    return lo + (np.column_stack([i, j]) + 0.5) * cfg.cell, g[i, j]


def column_test(c: np.ndarray, tree: cKDTree, pts: np.ndarray, hag: np.ndarray, top: float, cfg: StemConfig) -> dict:
    """Follow the column upward bin by bin (re-centring on each bin, limited lean).

    Hidden base: when leaves of neighbours cover the lowest part of the stem
    (RL_plant01 on 21.06: nothing below 12 cm), the column may start where the
    first points are, up to `max_hidden_base`; the test height moves up by the
    same amount."""
    idx = tree.query_ball_point(c, cfg.col_radius * 2.5)
    if not idx:
        return {"ok": False, "frac": 0.0}
    P, H = pts[idx], hag[idx]
    near0 = np.linalg.norm(P - c, axis=1) <= cfg.col_radius
    z_start = cfg.base_hag[0]
    if near0.sum() >= cfg.col_min_pts:
        lowest = float(np.percentile(H[near0], 2))
        if lowest > cfg.base_hag[0] + 0.02:
            if lowest > cfg.max_hidden_base:
                return {"ok": False, "frac": 0.0}
            z_start = lowest
    nb = max(1, int(round((top - cfg.base_hag[0]) / cfg.col_bin)))
    centre = c.copy()
    filled, spreads, centres = 0, [], []
    for b in range(nb):
        z0 = z_start + b * cfg.col_bin
        m = (H >= z0) & (H < z0 + cfg.col_bin)
        if m.any():
            d = np.linalg.norm(P[m] - centre, axis=1)
            near = d <= cfg.col_radius
            if near.sum() >= cfg.col_min_pts:
                filled += 1
                newc = np.median(P[m][near], axis=0)
                step = newc - centre
                lim = cfg.col_max_lean * cfg.col_bin
                if np.linalg.norm(step) > lim:
                    newc = centre + step / np.linalg.norm(step) * lim
                spreads.append(float(np.median(np.linalg.norm(P[m][near] - newc, axis=1))))
                centre = newc
        centres.append(centre.copy())
    frac = filled / nb
    spread = float(np.median(spreads)) if spreads else 1.0
    return {"ok": frac >= cfg.col_min_frac and spread <= cfg.col_max_spread, "frac": frac, "spread": spread,
            "bins": nb, "base_xy": centres[0] if centres else c, "hidden_base_m": z_start if z_start > cfg.base_hag[0] + 0.02 else 0.0}


def find_stems(plants: dict, cfg: StemConfig | None = None, log=print) -> tuple[list[dict], dict]:
    cfg = cfg or StemConfig()
    xyz, hag = plants["xyz"].astype(np.float64), plants["hag"].astype(np.float64)

    # connected blobs (1 cm voxels) and their height -> local top for the column test
    v = cfg.blob_voxel
    lo = xyz.min(axis=0)
    k = np.floor((xyz - lo) / v).astype(np.int64)
    occ = np.zeros(k.max(axis=0) + 1, bool)
    occ[k[:, 0], k[:, 1], k[:, 2]] = True
    lab, n_blob = ndi.label(occ, structure=np.ones((3, 3, 3)))
    blob = lab[k[:, 0], k[:, 1], k[:, 2]]
    blob_h = np.zeros(n_blob + 1)
    np.maximum.at(blob_h, blob, hag)

    base = (hag >= cfg.base_hag[0]) & (hag <= cfg.base_hag[1])
    bxy, bh, bblob = xyz[base, :2], hag[base], blob[base]
    cand, strength = _peaks(bxy, cfg)
    tree = cKDTree(bxy)
    tall = (hag >= cfg.base_hag[0]) & (hag <= cfg.tall_top + cfg.max_hidden_base + 0.02)
    txy, th = xyz[tall, :2], hag[tall]
    ttree = cKDTree(txy)
    stems = []
    for c, s in zip(cand, strength):
        near = tree.query_ball_point(c, cfg.col_radius)
        if len(near) < cfg.col_min_pts:
            continue
        b_id = int(np.bincount(bblob[near]).argmax())
        if blob_h[b_id] < cfg.min_plant_height:
            continue
        top = float(np.clip(0.8 * blob_h[b_id], cfg.col_min_top, cfg.col_top))
        r = column_test(c, tree, bxy, bh, top, cfg)
        if not r["ok"]:
            continue
        # a stem keeps rising (leaves near the base bend away)
        ttop = float(np.clip(cfg.tall_top_frac * blob_h[b_id], cfg.col_min_top, cfg.tall_top))
        r2 = column_test(c, ttree, txy, th, ttop, cfg)
        r["frac_tall"] = r2["frac"]
        if r2["frac"] >= cfg.tall_min_frac:
            stems.append({"x": float(r["base_xy"][0]), "y": float(r["base_xy"][1]), "strength": float(s),
                          "column_frac": r["frac"], "column_frac_tall": r["frac_tall"], "column_spread_m": r["spread"],
                          "hidden_base_m": r["hidden_base_m"],
                          "tested_to_m": ttop,
                          "blob": b_id, "blob_height_m": float(blob_h[b_id])})
    # merge candidates of the same stem (keep the stronger)
    stems.sort(key=lambda d: -d["strength"])
    kept: list[dict] = []
    for s_ in stems:
        if all(np.hypot(s_["x"] - t["x"], s_["y"] - t["y"]) >= cfg.min_stem_dist for t in kept):
            kept.append(s_)

    # clod / weed filter: blobs without a stem and lower than blob_keep_hag
    has_stem = np.zeros(n_blob + 1, bool)
    has_stem[[s_["blob"] for s_ in kept]] = True
    keep_blob = has_stem | (blob_h >= cfg.blob_keep_hag)
    keep_blob[0] = False
    keep_pts = keep_blob[blob]
    stats = {"n_candidates": int(len(cand)), "n_column_ok": len(stems), "n_stems": len(kept),
             "n_blobs": int(n_blob), "n_blobs_removed": int((~keep_blob[1:]).sum()),
             "points_removed_share": float(1 - keep_pts.mean()), "points_removed": int((~keep_pts).sum())}
    log(f"  stems: {len(kept)} found ({len(cand)} candidates, {len(stems)} passed the column test); "
        f"{stats['n_blobs_removed']} low blobs without a stem removed ({stats['points_removed_share']:.1%} of plant points)")
    return kept, {"keep_points": keep_pts, "blob": blob, "blob_height": blob_h, "stats": stats}


# ---------------------------------------------------------------- guided search (occluded stems)
def guided_search(plants: dict, targets_xy: np.ndarray, found_xy: np.ndarray, cfg: StemConfig,
                  radius: float = 0.03, log=print) -> list[dict]:
    """Where the reference date has a stem but this date found none within
    `radius`, look again with looser limits (closed canopy hides stem bases).
    Evidence is still required: a column in the points of this date."""
    xyz, hag = plants["xyz"].astype(np.float64), plants["hag"].astype(np.float64)
    base = (hag >= cfg.base_hag[0]) & (hag <= cfg.tall_top + cfg.max_hidden_base + 0.02)
    bxy, bh = xyz[base, :2], hag[base]
    if not len(bxy):
        return []
    tree = cKDTree(bxy)
    loose = StemConfig(**{**cfg.to_dict(), "col_min_frac": 0.5, "col_max_spread": 0.015, "col_min_pts": 2})
    ftree = cKDTree(found_xy) if len(found_xy) else None
    out = []
    for t in targets_xy:
        if ftree is not None and ftree.query(t)[0] <= radius:
            continue
        best = None
        # try the target and a small ring around it (alignment error ~1-2 cm)
        for dx in (-0.01, 0.0, 0.01):
            for dy in (-0.01, 0.0, 0.01):
                c = t + (dx, dy)
                r = column_test(c, tree, bxy, bh, 0.10, loose)
                if r["frac"] >= loose.col_min_frac and r.get("spread", 1) <= loose.col_max_spread:
                    if best is None or r["frac"] > best[1]["frac"]:
                        best = (c, r)
        if best is not None:
            c, r = best
            out.append({"x": float(r["base_xy"][0]), "y": float(r["base_xy"][1]), "strength": 0.0,
                        "column_frac": r["frac"], "column_frac_tall": float("nan"), "column_spread_m": r["spread"],
                        "tested_to_m": 0.10, "blob": -1, "blob_height_m": float("nan"), "guided": True})
    log(f"  guided search: {len(out)} hidden stems recovered at reference positions")
    return out


# ---------------------------------------------------------------- alignment check from stem positions
def stem_vote_shift(src_xy: np.ndarray, dst_xy: np.ndarray, max_shift: float = 0.25, tol: float = 0.02):
    """Translation that maps most `src` stems onto `dst` stems (vote over all
    pair differences). Returns (shift, n_matched)."""
    if len(src_xy) < 3 or len(dst_xy) < 3:
        return np.zeros(2), 0
    d = (dst_xy[None, :, :] - src_xy[:, None, :]).reshape(-1, 2)
    d = d[np.linalg.norm(d, axis=1) <= max_shift]
    if not len(d):
        return np.zeros(2), 0
    tree = cKDTree(d)
    counts = np.array([len(x) for x in tree.query_ball_point(d, tol)])
    best = d[int(np.argmax(counts))]
    near = d[np.linalg.norm(d - best, axis=1) <= tol]
    shift = np.median(near, axis=0)
    n = int((cKDTree(dst_xy).query(src_xy + shift)[0] <= tol).sum())
    return shift, n


# ---------------------------------------------------------------- groups
def make_groups(xy: np.ndarray, origin: np.ndarray, along: np.ndarray, cfg: StemConfig,
                gap: float | None = None) -> tuple[np.ndarray, float]:
    """Group index (0..) per stem: order along the row, split at gaps >= gap.
    Groups are numbered from the start of the row (south, since `along` points north)."""
    s = (xy - origin) @ along
    order = np.argsort(s)
    gaps = np.diff(s[order])
    if gap is None:
        gap = cfg.group_gap
    gid = np.zeros(len(xy), int)
    gid[order] = np.concatenate([[0], np.cumsum(gaps >= gap)])
    return gid, gap


def plant_order_in_group(xy: np.ndarray, origin: np.ndarray, along: np.ndarray) -> np.ndarray:
    """Left to right when looking along the row (north): by the 'right' coordinate, then along."""
    right = np.array([along[1], -along[0]])
    t, s = (xy - origin) @ right, (xy - origin) @ along
    return np.lexsort((s, t))


# ---------------------------------------------------------------- row centre line
def fit_row_centre(xy: np.ndarray, w: np.ndarray, origin: np.ndarray, along: np.ndarray,
                   bin_len: float = 0.5, max_off: float = 0.10, iters: int = 4):
    """Smooth centre line of the row (across-row offset as a function of the
    along-row position), fitted to the stems with outliers ignored. Returns a
    function off(xy) = distance of points from the line (m) and the knots."""
    right = np.array([along[1], -along[0]])
    s, t = (xy - origin) @ along, (xy - origin) @ right
    keep = np.ones(len(xy), bool)
    knots_s = np.arange(s.min(), s.max() + bin_len, bin_len) if len(s) else np.zeros(1)
    knots_t = np.zeros(len(knots_s))
    for it in range(iters):
        lim = max_off * (3.0 if it == 0 else (2.0 if it == 1 else 1.0))
        for i, k in enumerate(knots_s):
            m = keep & (np.abs(s - k) <= bin_len)
            if m.sum() >= 2:
                knots_t[i] = np.average(t[m], weights=w[m] + 1e-6) if it == 0 else np.median(t[m])
            elif i > 0:
                knots_t[i] = knots_t[i - 1]
        # smooth the knots (3-point running median then mean)
        if len(knots_t) >= 3:
            kt = knots_t.copy()
            kt[1:-1] = np.median(np.vstack([knots_t[:-2], knots_t[1:-1], knots_t[2:]]), axis=0)
            knots_t = np.convolve(np.r_[kt[0], kt, kt[-1]], np.ones(3) / 3, "valid")
        keep = np.abs(t - np.interp(s, knots_s, knots_t)) <= lim

    def off(p):
        p = np.atleast_2d(p)
        sp, tp = (p - origin) @ along, (p - origin) @ right
        return np.abs(tp - np.interp(sp, knots_s, knots_t))
    return off, np.column_stack([knots_s, knots_t])
