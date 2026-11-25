"""Step 7: turn the structure code's leaf pieces into real leaves.

Rule (all geometry):
  1. a piece is a real leaf if it is at least `min_length` long OR at least
     `min_area` large;
  2. a smaller piece is joined to the real leaf it touches (gap <= `join_dist`),
     optionally only if it runs in the same direction as that leaf there
     (angle <= `max_angle`); joined pieces can carry further pieces (2 passes);
  3. optionally, a real piece whose nearest point is farther than `max_stem_gap`
     from the plant's own stem points does not belong to this plant (a piece of a
     neighbour's leaf) and is dropped together with what joined it;
  4. what is left is a 'fragment' (label -3), not counted.
The structure code's labels are not changed on disk; this is a separate,
switchable step whose result is used for leaf matching.

The settings are chosen on the hand-labelled 21.06. plants (`tune`): on the
ground-truth clouds themselves and on the pipeline's separated plants that
overlap them (ground-truth labels moved across point by point).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

FRAGMENT = -3


@dataclass
class LeafRule:
    min_length: float = 0.08
    min_area: float = 0.002
    join_dist: float = 0.02
    max_angle: float | None = None
    max_stem_gap: float | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class PieceGeom:
    """Pre-computed piece-to-piece geometry of one plant (independent of the rule)."""
    labels: list            # piece labels (>= 1)
    length: np.ndarray
    area: np.ndarray
    gap: np.ndarray         # (n, n) smallest point distance between pieces
    angle: np.ndarray       # (n, n) angle (deg) between piece i's main axis and piece j's midrib near i
    stem_gap: np.ndarray    # (n,) smallest distance from the piece to the plant's stem points (inf = no stem)


def _axis(pts: np.ndarray):
    if len(pts) < 5:
        return None
    c = pts - pts.mean(0)
    w, v = np.linalg.eigh(np.cov(c.T))
    return v[:, -1] if w[-1] > 4 * max(w[-2], 1e-12) else None     # only for elongated pieces


def piece_geometry(xyz: np.ndarray, labels: np.ndarray, info: dict, voxel: float = 0.005) -> PieceGeom:
    """info: {label: (length_m, area_m2, midrib path (k,3) junction->tip)}."""
    labs = sorted(l for l in info if (labels == l).any())
    pts = {}
    for l in labs:
        p = xyz[labels == l]
        k = np.floor(p / voxel).astype(np.int64)
        _, i = np.unique(k, axis=0, return_index=True)
        pts[l] = p[i]
    n = len(labs)
    gap = np.full((n, n), np.inf)
    ang = np.full((n, n), 0.0)
    trees = {l: cKDTree(pts[l]) for l in labs}
    axes = {l: _axis(pts[l]) for l in labs}
    for a, b in itertools.combinations(range(n), 2):
        la, lb = labs[a], labs[b]
        # cheap reject on bounding boxes
        lo = np.maximum(pts[la].min(0), pts[lb].min(0))
        hi = np.minimum(pts[la].max(0), pts[lb].max(0))
        if np.any(lo - hi > 0.05):
            continue
        small, big = (la, lb) if len(pts[la]) <= len(pts[lb]) else (lb, la)
        d, _ = trees[big].query(pts[small], distance_upper_bound=0.05)
        g = float(d.min()) if np.isfinite(d).any() else np.inf
        gap[a, b] = gap[b, a] = g
    for a in range(n):
        ax = axes[labs[a]]
        if ax is None:
            continue
        cen = pts[labs[a]].mean(0)
        for b in range(n):
            if a == b or not np.isfinite(gap[a, b]):
                continue
            path = info[labs[b]][2]
            if len(path) < 2:
                continue
            k = int(np.argmin(np.linalg.norm(path - cen, axis=1)))
            k0, k1 = max(0, k - 2), min(len(path) - 1, k + 2)
            dvec = path[k1] - path[k0]
            if np.linalg.norm(dvec) < 1e-6:
                continue
            c = abs(float(np.dot(ax, dvec / np.linalg.norm(dvec))))
            ang[a, b] = float(np.degrees(np.arccos(min(1.0, c))))
    sg = np.full(n, np.inf)
    stem = xyz[labels == 0]
    if len(stem):
        k = np.floor(stem / voxel).astype(np.int64)
        _, i = np.unique(k, axis=0, return_index=True)
        st = cKDTree(stem[i])
        for a, l in enumerate(labs):
            d, _ = st.query(pts[l], distance_upper_bound=0.2)
            sg[a] = float(d.min()) if np.isfinite(d).any() else np.inf
    return PieceGeom(labs, np.array([info[l][0] for l in labs]), np.array([info[l][1] for l in labs]), gap, ang, sg)


def apply_rule(g: PieceGeom, rule: LeafRule) -> dict:
    """{piece label: leaf label (a real piece's label) or FRAGMENT}."""
    n = len(g.labels)
    real = (g.length >= rule.min_length) | (g.area >= rule.min_area)
    if rule.max_stem_gap is not None:
        real &= g.stem_gap <= rule.max_stem_gap
    owner = np.where(real, np.arange(n), -1)
    order = np.argsort(-g.area)
    for _ in range(2):
        changed = False
        for i in order:
            if owner[i] >= 0:
                continue
            best, bg = -1, np.inf
            for j in range(n):
                if owner[j] < 0 or g.gap[i, j] > rule.join_dist or g.gap[i, j] >= bg:
                    continue
                if rule.max_angle is not None and g.angle[i, owner[j]] > rule.max_angle:
                    continue
                best, bg = owner[j], g.gap[i, j]
            if best >= 0:
                owner[i] = best
                changed = True
        if not changed:
            break
    return {g.labels[i]: (g.labels[owner[i]] if owner[i] >= 0 else FRAGMENT) for i in range(n)}


def relabel(labels: np.ndarray, mapping: dict) -> np.ndarray:
    out = labels.copy()
    for k, v in mapping.items():
        out[labels == k] = v
    return out


def leaf_scores(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Per-point comparison of leaf labels (stem = 0 and unlabelled ignored)."""
    gl = [l for l in np.unique(gt) if l >= 1]
    pl = [l for l in np.unique(pred) if l >= 1]
    if not gl:
        return {"mean_iou": np.nan, "n_true": 0, "n_pred": len(pl)}
    I = np.zeros((len(gl), max(1, len(pl))))
    for a, g in enumerate(gl):
        mg = gt == g
        for b, p in enumerate(pl):
            mp = pred == p
            inter = np.count_nonzero(mg & mp)
            if inter:
                I[a, b] = inter / np.count_nonzero(mg | mp)
    r, c = linear_sum_assignment(-I)
    ious = np.zeros(len(gl))
    ious[r] = I[r, c]
    return {"mean_iou": float(ious.mean()), "n_true": len(gl), "n_pred": len(pl),
            "leaf_point_share_fragment": float(np.mean(pred[gt >= 1] == FRAGMENT))}


def grid():
    yield "no joining (structure code as is)", LeafRule(0.0, 0.0, 0.0, None, None)
    for L, A, D, ang, sg in itertools.product([0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
                                              [0.0005, 0.001, 0.002, 0.003, 0.004],
                                              [0.01, 0.02, 0.03], [None, 60.0], [None, 0.02, 0.04, 0.06]):
        yield (f"L{100 * L:.0f}cm A{1e4 * A:.0f}cm2 D{100 * D:.0f}cm" + (f" angle{ang:.0f}" if ang else "")
               + (f" stem{100 * sg:.0f}cm" if sg else "")), LeafRule(L, A, D, ang, sg)


COUNT_WEIGHT = 0.02     # one leaf too many / too few costs as much as 0.02 mean IoU


def score_case(c: dict, rule: LeafRule) -> dict:
    mp = apply_rule(c["geom"], rule)
    sc = leaf_scores(c["gt"], relabel(c["labels"], mp))
    if "labels_full" in c:      # pipeline plant: count every leaf it keeps, also neighbour pieces
        full = relabel(c["labels_full"], mp)
        sc["n_pred"] = int(len([l for l in np.unique(full) if l >= 1]))
    return sc


def tune(cases: list[dict], log=print) -> tuple[LeafRule, list[dict]]:
    """cases: dicts with name, geom (PieceGeom), labels and gt per evaluated point.
    Score = mean leaf IoU - COUNT_WEIGHT * mean leaf-count error."""
    rows, best = [], None
    for name, rule in grid():
        s = []
        for c in cases:
            sc = score_case(c, rule)
            s.append(sc)
            rows.append({"rule": name, **rule.to_dict(), "case": c["name"], **sc})
        miou = float(np.nanmean([x["mean_iou"] for x in s]))
        cerr = float(np.mean([abs(x["n_pred"] - x["n_true"]) for x in s]))
        score = miou - COUNT_WEIGHT * cerr
        if best is None or score > best[0] + 1e-9:
            best = (score, name, rule, miou, cerr)
    log(f"  best rule: {best[1]}  (mean leaf IoU {best[3]:.3f}, leaf count off by {best[4]:.1f})")
    return best[2], rows
