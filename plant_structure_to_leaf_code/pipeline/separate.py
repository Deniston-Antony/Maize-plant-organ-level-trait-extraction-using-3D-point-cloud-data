"""Stage 5: split the plant points of one date into single plants.

Seeds = the stem base of every plant of the register (its position mapped
into this date's frame). Plant points are thinned to 1 cm voxels and joined
to their 26 neighbours; a multi-source shortest-path search (Dijkstra) from
all seeds gives every voxel the plant it is reached from first along the
plant surface. Pieces that are not connected to any seed (leaves cut off by
occlusion) go to the nearest assigned plant if they are close.

Presence on this date: the plant needs points at its stem base and enough
points reached from it. This replaces "stem visible" as the presence test
(05.07 / 11.07 hide many stem bases).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


@dataclass
class SeparateConfig:
    voxel: float = 0.01
    seed_radius: float = 0.015        # seed voxels within this of the stem position ...
    seed_hag: tuple = (0.0, 0.15)     # ... and in this height band
    seed_search: float = 0.03         # if none: look a bit wider (alignment error)
    max_geodesic: float = 1.5         # metres along the plant surface
    orphan_max_dist: float = 0.06     # unconnected pieces closer than this join the nearest plant
    orphan_min_points: int = 30
    min_points_present: int = 1500    # fewer points -> plant counted as missing on this date
    # stem following (all plants at once, each in its own zone)
    track_slice: float = 0.03
    track_step: float = 0.01
    track_sigma: float = 0.008        # weight around the predicted axis (as in stem_tracking)
    track_max_move: float = 0.005     # per 1 cm of height
    track_search: float = 0.03
    track_max_gap: int = 15           # stop after this many empty slices (cm); a stem hidden by leaves
                                      # over 10 cm is followed across (29.06 group03_plant01: gap 20-30 cm)
    track_min_weight: float = 3.0     # points in the slice near the axis
    axis_radius: float = 0.012        # voxels this close to the followed axis become seeds
    track_merge_dist: float = 0.012   # two axes closer than this: the weaker stops
    hidden_min_points: int = 200      # points above the position needed to call a plant 'covered by canopy'
    hidden_low: float = 0.20          # ... with (almost) nothing below this height
    hidden_max_low: int = 20
    axis_seed_frac: float = 0.75      # only the lower 75 % of a followed stem seeds the plant ...
                                      # (0.5 lost hidden upper stems to neighbours; 1.0: LtR_12th IoU 0.72 -> 0.49)
    axis_seed_top: float = 9.0        # ... and never above this height
    whole_leaf_share: float = 0.7     # a piece touching one plant's stem this much goes to it entirely
    whole_leaf_min_voxels: int = 15

    def to_dict(self) -> dict:
        return asdict(self)


_OFFS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                  if (i, j, k) > (0, 0, 0)])   # 13 half-neighbourhood offsets


def _voxelize(xyz: np.ndarray, v: float):
    lo = xyz.min(axis=0) - v
    k = np.floor((xyz - lo) / v).astype(np.int64)
    key = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)
    cen = np.zeros((len(uk), 3))
    for d in range(3):
        cen[:, d] = np.bincount(inv, weights=xyz[:, d]) / cnt
    ijk = np.column_stack([uk >> 42, (uk >> 21) & 0x1FFFFF, uk & 0x1FFFFF])
    return uk, inv, cen, ijk


def _graph(uk: np.ndarray, ijk: np.ndarray, cen: np.ndarray):
    rows, cols = [], []
    for o in _OFFS:
        nk = ((ijk[:, 0] + o[0]) << 42) | ((ijk[:, 1] + o[1]) << 21) | (ijk[:, 2] + o[2])
        pos = np.searchsorted(uk, nk)
        pos = np.clip(pos, 0, len(uk) - 1)
        ok = uk[pos] == nk
        rows.append(np.where(ok)[0])
        cols.append(pos[ok])
    r, c = np.concatenate(rows), np.concatenate(cols)
    w = np.linalg.norm(cen[r] - cen[c], axis=1) + 1e-6
    n = len(uk)
    g = coo_matrix((np.r_[w, w], (np.r_[r, c], np.r_[c, r])), shape=(n, n)).tocsr()
    return g


def follow_stems(cen: np.ndarray, vh: np.ndarray, w: np.ndarray, plant_xy: np.ndarray, cfg: SeparateConfig):
    """Follow every stem upward at the same time. In each height slice a
    plant only sees voxels nearer to its own axis than to any other axis, and
    its axis moves to the weighted centre of those near the predicted axis
    (limited move). Returns per plant the axis (list of (x, y, h)) and the
    voxel indices along it (seeds)."""
    P = len(plant_xy)
    c = plant_xy.astype(float).copy()
    vel = np.zeros_like(c)
    active = np.ones(P, bool)
    gaps = np.zeros(P, int)
    axes = [[] for _ in range(P)]
    seeds = [dict() for _ in range(P)]      # voxel -> height where the axis passed it
    order = np.argsort(vh)
    hs = vh[order]
    h = cfg.seed_hag[0] + 0.005
    top = float(vh.max()) if len(vh) else 0.0
    while h < top and active.any():
        lo, hi = np.searchsorted(hs, [h, h + cfg.track_slice])
        idx = order[lo:hi]
        if len(idx):
            act = np.where(active)[0]
            pred = c[act] + vel[act]
            d_all, owner = cKDTree(pred).query(cen[idx, :2], k=1, distance_upper_bound=cfg.track_search)
            for a_i, p in enumerate(act):
                m = owner == a_i
                if not m.any():
                    gaps[p] += 1
                    continue
                vi = idx[m]
                dd = np.linalg.norm(cen[vi, :2] - pred[a_i], axis=1)
                ww = w[vi] * np.exp(-dd ** 2 / (2 * cfg.track_sigma ** 2))
                if ww.sum() < cfg.track_min_weight * np.exp(-0.5):
                    gaps[p] += 1
                    continue
                gaps[p] = 0
                new = (cen[vi, :2] * ww[:, None]).sum(axis=0) / ww.sum()
                step = new - c[p]
                n = np.linalg.norm(step)
                lim = cfg.track_max_move * cfg.track_step / 0.01
                if n > lim:
                    step *= lim / n
                vel[p] = 0.7 * vel[p] + 0.3 * step
                c[p] = c[p] + step
                axes[p].append((c[p, 0], c[p, 1], h + cfg.track_slice / 2))
                near = vi[np.linalg.norm(cen[vi, :2] - c[p], axis=1) <= cfg.axis_radius]
                for v_ in near.tolist():
                    seeds[p].setdefault(v_, h)
            # two axes on the same column: keep the one with the longer axis
            act = np.where(active)[0]
            if len(act) > 1:
                pairs = cKDTree(c[act]).query_pairs(cfg.track_merge_dist)
                for i, j in pairs:
                    a, b = act[i], act[j]
                    loser = a if len(axes[a]) < len(axes[b]) else b
                    active[loser] = False
            active &= gaps <= cfg.track_max_gap
        h += cfg.track_step
    # only the lower part of each followed stem seeds the plant: higher up a
    # stem can run onto a neighbour's leaf (GT 21.06 LtR_11th: 51 of 106 seed
    # voxels of the neighbour lay on its leaves; lower half only: IoU 0.43 -> 0.63)
    out = []
    for p in range(P):
        top_h = axes[p][-1][2] if axes[p] else 0.0
        lim = min(cfg.axis_seed_top, max(cfg.seed_hag[1], cfg.axis_seed_frac * top_h))
        out.append(np.array(sorted(v_ for v_, hh in seeds[p].items() if hh <= lim), int))
    return axes, out


def separate(xyz: np.ndarray, hag: np.ndarray, plant_xy: np.ndarray, cfg: SeparateConfig | None = None, log=print):
    """xyz/hag: plant points of one date (native frame); plant_xy: (P, 2)
    stem positions in the same frame. Returns (labels per point: plant index
    or -1, info dict per plant)."""
    cfg = cfg or SeparateConfig()
    uk, inv, cen, ijk = _voxelize(xyz, cfg.voxel)
    vcnt = np.bincount(inv).astype(float)
    vh = np.bincount(inv, weights=hag) / vcnt
    g = _graph(uk, ijk, cen)
    axes, axis_seeds = follow_stems(cen, vh, vcnt, plant_xy, cfg)
    vtree = cKDTree(cen[:, :2])
    seeds, seed_owner, seed_info = [], [], []
    for p, xy in enumerate(plant_xy):
        cand = np.array(vtree.query_ball_point(xy, cfg.seed_radius), int)
        cand = cand[(vh[cand] >= cfg.seed_hag[0]) & (vh[cand] <= cfg.seed_hag[1])] if len(cand) else cand
        wide = False
        if not len(cand):
            cand = np.array(vtree.query_ball_point(xy, cfg.seed_search), int)
            cand = cand[(vh[cand] >= cfg.seed_hag[0]) & (vh[cand] <= cfg.seed_hag[1])] if len(cand) else cand
            wide = True
        base_ok = len(cand) > 0
        allc = np.union1d(cand, axis_seeds[p]) if base_ok else cand
        seeds.extend(allc.tolist())
        seed_owner.extend([p] * len(allc))
        ax = np.array(axes[p]) if axes[p] else np.zeros((0, 3))
        seed_info.append({"seed_voxels": int(len(allc)), "seed_wide_search": wide and base_ok,
                          "stem_followed_to_m": float(ax[-1, 2]) if len(ax) else 0.0,
                          "axis": ax.tolist()})
    lab_v = np.full(len(uk), -1, int)
    dist_v = np.full(len(uk), np.inf)
    if seeds:
        seeds_a = np.array(seeds)
        owner = np.array(seed_owner)
        # a voxel can be a seed of two plants (stems 2 cm apart): keep the nearer plant
        uniq, first = np.unique(seeds_a, return_index=True)
        if len(uniq) < len(seeds_a):
            best = {}
            for s_, o_ in zip(seeds_a, owner):
                d = np.linalg.norm(cen[s_, :2] - plant_xy[o_])
                if s_ not in best or d < best[s_][1]:
                    best[s_] = (o_, d)
            seeds_a = np.array(list(best))
            owner = np.array([best[s_][0] for s_ in seeds_a])
        dist_v, _, src = dijkstra(g, directed=False, indices=seeds_a, min_only=True,
                                  return_predecessors=True, limit=cfg.max_geodesic)
        reach = np.isfinite(dist_v) & (src >= 0)
        pos = {s_: o_ for s_, o_ in zip(seeds_a, owner)}
        lab_v[reach] = [pos[s_] for s_ in src[reach]]
    # whole leaves: remove the stem axes; every remaining connected piece that
    # touches mainly ONE plant's stem goes to that plant as a whole (a leaf
    # belongs to the stem it grows from, even where it touches a neighbour)
    n_whole = 0
    if seeds:
        from scipy.sparse.csgraph import connected_components
        is_seed = np.zeros(len(uk), bool)
        is_seed[seeds_a] = True
        seed_plant = np.full(len(uk), -1)
        seed_plant[seeds_a] = owner
        nonseed = np.where(~is_seed)[0]
        sub = g[nonseed][:, nonseed]
        nc, comp = connected_components(sub, directed=False)
        # contacts: edges between a non-seed voxel and a seed voxel
        gc = g.tocoo()
        e = (~is_seed[gc.row]) & is_seed[gc.col]
        pos_in_sub = np.full(len(uk), -1)
        pos_in_sub[nonseed] = np.arange(len(nonseed))
        ec, ep = comp[pos_in_sub[gc.row[e]]], seed_plant[gc.col[e]]
        if len(ec):
            P_ = len(plant_xy)
            M = coo_matrix((np.ones(len(ec)), (ec, ep)), shape=(nc, P_)).tocsr()
            tot = np.asarray(M.sum(axis=1)).ravel()
            best = np.asarray(M.argmax(axis=1)).ravel()
            share = np.where(tot > 0, np.asarray(M.max(axis=1).todense()).ravel() / np.maximum(tot, 1), 0)
            sizes = np.bincount(comp, minlength=nc)
            good = (share >= cfg.whole_leaf_share) & (sizes >= cfg.whole_leaf_min_voxels)
            take = good[comp]
            newlab = best[comp]
            changed = take & (lab_v[nonseed] != newlab)
            n_whole = int(changed.sum())
            lab_v[nonseed[take]] = newlab[take]

    # unconnected pieces: whole components to the nearest assigned plant if close
    un = lab_v < 0
    n_orphan_join = 0
    if un.any() and (~un).any():
        sub = g[un][:, un]
        from scipy.sparse.csgraph import connected_components
        nc, comp = connected_components(sub, directed=False)
        ui = np.where(un)[0]
        atree = cKDTree(cen[~un])
        alab = lab_v[~un]
        sizes = np.bincount(comp, minlength=nc)
        d, nn = atree.query(cen[ui], k=1)
        for c in range(nc):
            m = comp == c
            if sizes[c] < 2:
                continue
            j = np.argmin(d[m])
            if d[m][j] <= cfg.orphan_max_dist:
                lab_v[ui[m]] = alab[nn[m][j]]
                n_orphan_join += int(m.sum())
    labels = lab_v[inv]
    counts = np.bincount(labels[labels >= 0], minlength=len(plant_xy))
    info = []
    ptree = cKDTree(xyz[:, :2])
    for p in range(len(plant_xy)):
        present = bool(seed_info[p]["seed_voxels"] > 0 and counts[p] >= cfg.min_points_present)
        vis = "visible"
        if not present:
            # covered by the canopy (points only high above the position) or really nothing there?
            near = ptree.query_ball_point(plant_xy[p], cfg.seed_search)
            hh = hag[near] if near else np.zeros(0)
            if len(hh) >= cfg.hidden_min_points and (hh < cfg.hidden_low).sum() < cfg.hidden_max_low:
                vis = "covered_by_canopy"
            elif len(hh) < cfg.hidden_min_points:
                vis = "nothing_there"
            else:
                vis = "too_small"
        info.append({**seed_info[p], "n_points": int(counts[p]), "present": present, "visibility": vis})
    log(f"  separation: {sum(i['present'] for i in info)} of {len(plant_xy)} plants present; "
        f"{(labels >= 0).mean():.1%} of points assigned ({n_orphan_join} voxels of loose pieces joined, "
        f"{n_whole} voxels moved to the stem their leaf grows from)")
    return labels, info
