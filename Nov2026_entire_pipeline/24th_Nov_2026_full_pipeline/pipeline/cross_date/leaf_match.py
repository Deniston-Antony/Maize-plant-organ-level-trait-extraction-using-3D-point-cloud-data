"""Step 8: match the leaves of each plant across dates; leaf IDs from 29.06.

Leaves = the structure code's pieces after the step-7 rule. Features of a leaf
(reference frame): junction height above ground, azimuth (direction from the
junction to the midrib point ~8 cm along it), base shape (midrib points at
2, 4, 6, 8, 10 cm from the junction, relative to the junction), length, area.

Pair cost (earlier leaf a, later leaf b):
    azimuth difference / 30 deg + |k * height(a) - height(b)| / 6 cm
  + base-shape distance / 4 cm + shrink penalty (later < 80 % of earlier length)
gated at 75 deg / 20 cm. k = stem stretch between the two dates (internodes
elongate: on 21.06. -> 29.06. leaves sit ~1.8x higher); one k per plant and
date pair, chosen from 0.8-2.5 as the one with the lowest total alignment
cost. Order-preserving sequence alignment (bottom -> top) with a gap cost: a
gap = a new leaf (top) or a lost / hidden leaf (bottom).

IDs: leaves of 29.06. are L01.. from the bottom. Later dates: direct match to
29.06., checked against the chain through the previous date (05.07. -> 11.07.);
disagreement = flagged. New leaves get the next numbers (later leaves chained
to them keep them). 21.06. leaves with no 29.06. partner get the next numbers,
marked 'not seen on 29.06.'.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline import names

from .leaf_pieces import FRAGMENT, LeafRule, apply_rule, piece_geometry

AZ_SCALE, H_SCALE, SHAPE_SCALE = 30.0, 0.06, 0.04
AZ_GATE, H_GATE = 75.0, 0.20
GAP_COST = 1.5
TRACK_MIN_LENGTH = 0.10     # only leaves this long are tracked across dates (plus the top leaf of a plant:
                            # the newest leaf is small); smaller ones are listed as 'small, not tracked'
SHRINK_LIMIT, SHRINK_W = 0.8, 5.0
BASE_ARCS = np.array([0.02, 0.04, 0.06, 0.08, 0.10])


@dataclass
class Leaf:
    date: str
    pid: str
    label: int                  # the main piece's label in the structure output
    pieces: list
    junction: np.ndarray
    junction_hag: float
    insertion_h: float
    length: float
    area: float
    azimuth: float              # deg, 0 = +x (east), counter-clockwise
    base_shape: np.ndarray      # (5, 3) relative to the junction
    path: np.ndarray
    leaf_id: str = ""
    tracked: bool = True

    def row(self) -> dict:
        return {"date": self.date, "plant": names.nice(self.pid), "leaf_label": self.label,
                "pieces": " ".join(map(str, self.pieces)), "leaf_id": self.leaf_id,
                "junction_height_m": round(self.junction_hag, 3), "insertion_height_m": round(self.insertion_h, 3),
                "length_m": round(self.length, 3), "area_cm2": round(1e4 * self.area, 1),
                "azimuth_deg": round(self.azimuth, 1), "tracked": self.tracked}


def _arc_points(path: np.ndarray, arcs: np.ndarray) -> np.ndarray:
    if len(path) < 2:
        return np.repeat(path[:1], len(arcs), 0)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    a = np.minimum(arcs, s[-1])
    return np.column_stack([np.interp(a, s, path[:, k]) for k in range(3)])


def build_leaves(plants: dict, rule: LeafRule) -> dict:
    out = {}
    for d, ps in plants.items():
        out[d] = {}
        for pid, p in ps.items():
            info = {pc.label: (pc.length, pc.area, pc.path) for pc in p.pieces}
            if not info:
                out[d][pid] = []
                continue
            g = piece_geometry(p.xyz, p.labels, info)
            mp = apply_rule(g, rule)
            byl = {pc.label: pc for pc in p.pieces}
            groups = {}
            for k, v in mp.items():
                if v != FRAGMENT:
                    groups.setdefault(v, []).append(k)
            leaves = []
            for main, members in groups.items():
                pc = byl[main]
                tip_dir = _arc_points(pc.path, np.array([0.08]))[0] - pc.junction
                az = float(np.degrees(np.arctan2(tip_dir[1], tip_dir[0])))
                leaves.append(Leaf(d, pid, main, sorted(members), pc.junction, pc.junction_hag, pc.insertion_h,
                                   pc.length, float(sum(byl[m].area for m in members)), az,
                                   _arc_points(pc.path, BASE_ARCS) - pc.junction, pc.path))
            leaves = sorted(leaves, key=lambda l: l.junction_hag)
            for k, l in enumerate(leaves):
                l.tracked = l.length >= TRACK_MIN_LENGTH or k == len(leaves) - 1
            out[d][pid] = leaves
    return out


def _angdiff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


STRETCH = np.round(np.arange(0.8, 2.51, 0.05), 2)


def pair_cost(a: Leaf, b: Leaf, k: float = 1.0) -> tuple[float, dict]:
    """a = earlier, b = later; k = stem stretch (earlier heights x k)."""
    az = _angdiff(a.azimuth, b.azimuth)
    dh = abs(k * a.junction_hag - b.junction_hag)
    if az > AZ_GATE or dh > H_GATE:
        return np.inf, {"az_diff_deg": az, "height_diff_cm": 100 * dh}
    shape = float(np.linalg.norm(a.base_shape - b.base_shape, axis=1).mean())
    ratio = b.length / a.length if a.length > 0 else 1.0
    shrink = SHRINK_W * max(0.0, SHRINK_LIMIT - ratio)
    c = az / AZ_SCALE + dh / H_SCALE + shape / SHAPE_SCALE + shrink
    return c, {"az_diff_deg": az, "height_diff_cm": 100 * dh, "shape_dist_cm": 100 * shape,
               "length_ratio": ratio}


def align_leaves(A: list, B: list) -> dict:
    """Order-preserving alignment (both bottom -> top), best stem stretch k.
    Returns {index in B: (index in A, cost, details)}; details carry k."""
    best = None
    for k in STRETCH:
        res, total = _align_k(A, B, k)
        if best is None or total < best[1] - 1e-9:
            best = (res, total)
    return best[0]


def _align_k(A: list, B: list, k: float):
    n, m = len(A), len(B)
    early_first = (A[0].date < B[0].date) if (A and B) else True
    C = np.full((n, m), np.inf)
    det = {}
    for i in range(n):
        for j in range(m):
            c, dd = pair_cost(A[i], B[j], k) if early_first else pair_cost(B[j], A[i], k)
            dd["stretch"] = float(k)
            C[i, j], det[(i, j)] = c, dd
    D = np.full((n + 1, m + 1), np.inf)
    D[0, :] = GAP_COST * np.arange(m + 1)
    D[:, 0] = GAP_COST * np.arange(n + 1)
    bt = np.zeros((n + 1, m + 1), int)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opts = (D[i - 1, j - 1] + C[i - 1, j - 1], D[i - 1, j] + GAP_COST, D[i, j - 1] + GAP_COST)
            k = int(np.argmin(opts))
            D[i, j], bt[i, j] = opts[k], k
    res, i, j = {}, n, m
    while i > 0 and j > 0:
        k = bt[i, j]
        if k == 0:
            if np.isfinite(C[i - 1, j - 1]):
                res[j - 1] = (i - 1, float(C[i - 1, j - 1]), det[(i - 1, j - 1)])
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return res, float(D[n, m])


def alternation(leaves: list) -> float:
    """Share of consecutive leaves (bottom -> top) pointing to opposite sides (> 120 deg apart)."""
    if len(leaves) < 2:
        return np.nan
    d = [_angdiff(a.azimuth, b.azimuth) for a, b in zip(leaves, leaves[1:])]
    return float(np.mean([x > 120 for x in d]))


def match_leaves(run, plants: dict, leaves: dict, assign: dict, log=print):
    ref = run.reference
    dates = [d for d in run.dates if leaves.get(d)]
    before = [d for d in dates if d < ref][::-1]       # nearest first
    after = [d for d in dates if d > ref]
    rows, tracks, pairs = [], [], {}
    counts = {}
    def tr(ls):
        return [l for l in ls if l.tracked]

    for rp, A_all in leaves.get(ref, {}).items():
        A = tr(A_all)
        for l in A_all:
            if not l.tracked:
                l.leaf_id = "small"
        if not A:
            continue
        nice = names.nice(rp)
        for k, l in enumerate(A):
            l.leaf_id = f"L{k + 1:02d}"
            rows.append({"plant": nice, "date": ref, "date_plant": nice, "leaf_label": l.label, "leaf_id": l.leaf_id,
                         "status": "reference", "alternating_share": round(alternation(A), 2) if k == 0 else ""})
        next_id = len(A) + 1
        found_on = {l.leaf_id: {ref} for l in A}

        def take(B, d, qp, direct, chain, chain_from):
            nonlocal next_id
            used = set()          # one leaf per ID per date
            for j, b in enumerate(B):
                dm, cm = direct.get(j), chain.get(j) if chain is not None else None
                did = A[dm[0]].leaf_id if dm else None
                cid = chain_from[cm[0]].leaf_id if (cm and chain_from[cm[0]].leaf_id) else None
                if did in used:
                    did, dm = None, None
                if cid in used or cid == "small":
                    cid, cm = None, None
                if did and (cid is None or cid == did):
                    b.leaf_id, st = did, ("matched" if cid is None else "matched (chain agrees)")
                elif did and cid and cid != did:
                    b.leaf_id, st = did, f"flagged (chain through the previous date says {cid})"
                elif cid:
                    b.leaf_id, st = cid, "matched via the previous date"
                else:
                    b.leaf_id = f"L{next_id:02d}"
                    next_id += 1
                    st = "new leaf" if d > ref else "not seen on 29.06."
                used.add(b.leaf_id)
                found_on.setdefault(b.leaf_id, set()).add(d)
                det = (dm or cm or (None, None, {}))[2] if (dm or cm) else {}
                rows.append({"plant": nice, "date": d, "date_plant": names.nice(qp), "leaf_label": b.label,
                             "leaf_id": b.leaf_id, "status": st,
                             "cost": round((dm or cm)[1], 3) if (dm or cm) else "",
                             **{k: round(v, 2) for k, v in det.items()}})
                counts[st.split(" (")[0]] = counts.get(st.split(" (")[0], 0) + 1

        prev_B = A
        for d in after:
            qp = assign.get(d, {}).get(rp)
            if qp is None or not leaves[d].get(qp):
                continue
            B = tr(leaves[d][qp])
            for l in leaves[d][qp]:
                if not l.tracked:
                    l.leaf_id = "small"
            if not B:
                continue
            direct = align_leaves(A, B)
            chain = align_leaves(prev_B, B) if prev_B is not A else None
            take(B, d, qp, direct, chain, prev_B)
            pairs[(rp, d)] = (A, B, direct)
            prev_B = B
        for d in before:
            qp = assign.get(d, {}).get(rp)
            if qp is None or not leaves[d].get(qp):
                continue
            B = tr(leaves[d][qp])
            for l in leaves[d][qp]:
                if not l.tracked:
                    l.leaf_id = "small"
            if not B:
                continue
            direct = align_leaves(A, B)
            take(B, d, qp, direct, None, None)
            pairs[(rp, d)] = (A, B, direct)
        for d in after + before:
            qp = assign.get(d, {}).get(rp)
            if qp is None or not tr(leaves[d].get(qp, [])):
                continue
            for l in A:
                if d not in found_on.get(l.leaf_id, set()):
                    rows.append({"plant": nice, "date": d, "date_plant": names.nice(qp), "leaf_label": "",
                                 "leaf_id": l.leaf_id, "status": "not found on this date (lost or hidden)"})
                    counts["not found"] = counts.get("not found", 0) + 1
        for d in dates:
            qp = rp if d == ref else assign.get(d, {}).get(rp)
            for l in (leaves[d].get(qp, []) if qp else []):
                tracks.append({**l.row(), "plant": nice, "leaf_id": l.leaf_id, "date_plant": names.nice(qp)})
    log("  " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return rows, tracks, pairs
