"""Step 5: check the plants of every date against the reference-date plants.

Every separated plant carries the register name given by the pipeline, so the
name is the claim and this step tests it from the plants' own geometry:
  - stem base distance (lowest point of the structure code's stem centre line,
    NOT the register seed, which would be the same by construction);
  - lean of the lower stem;
  - containment: the earlier plant's lower stem region should lie inside the
    later plant (a plant grows, it does not move).
The search is limited to the matched clusters of step 3 (plants outside every
cluster: by distance only). Assignment = Hungarian method (one-to-one).

Status per reference plant and date: confirmed (assigned plant carries the
same name) / swapped (carries another name while both names exist on both
dates - a real conflict) / other name (carries another name, but each name
exists on only one of the two dates: most likely one plant with two register
entries) / ambiguous (second-best within the margin) / missing (no candidate).
The base distance is noisy at the few-cm scale (the structure code's stem line
starts a median 3-5 cm from the register seed), so containment carries most
weight. Plants of a date with no reference
partner: 'not on the reference date'.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from pipeline import names

from .config import PlantConfig, ClusterConfig


def _cluster_of(s: float, clusters: list, margin: float):
    for c in clusters:
        if not c.clod and c.lo - margin <= s <= c.hi + margin:
            return c.cid
    return None


def apply_fine_shifts(run, plants: dict, clusters: dict, fine_rows: list, pcfg: PlantConfig) -> None:
    """Shift each plant of a date by the accepted fine shift of the cluster it lies in."""
    acc = {}
    for r in fine_rows:
        if str(r["accepted"]) == "True" or r["accepted"] is True:
            for q in r["date_clusters"].split():
                acc[q] = (r["dx_cm"] / 100.0, r["dy_cm"] / 100.0)
    for d, ps in plants.items():
        for p in ps.values():
            s = run.frame.st(p.base[None, :2])[0, 0]
            cid = _cluster_of(s, clusters.get(d, []), pcfg.cluster_margin)
            p.cluster = cid
            if cid in acc:
                p.shift(*acc[cid])


def _low_region(p, pcfg: PlantConfig) -> np.ndarray:
    hmax = min(pcfg.low_region_h, pcfg.low_region_frac * (p.height if np.isfinite(p.height) else 0.3))
    lean = p.lean()
    m = p.hag <= max(hmax, 0.05)
    pts, h = p.xyz[m], p.hag[m]
    axis = p.base[:2] + h[:, None] * lean[None, :]
    keep = np.hypot(*(pts[:, :2] - axis).T) <= pcfg.low_region_r
    pts = pts[keep]
    if len(pts) > 3000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 3000, replace=False)]
    return pts


def containment(earlier, later, pcfg: PlantConfig, cache: dict) -> float:
    key = (earlier.date, earlier.pid)
    if key not in cache:
        cache[key] = _low_region(earlier, pcfg)
    pts = cache[key]
    if len(pts) == 0:
        return np.nan
    tkey = ("tree", later.date, later.pid)
    if tkey not in cache:
        cache[tkey] = cKDTree(later.xyz)
    d, _ = cache[tkey].query(pts, distance_upper_bound=pcfg.contain_tol)
    return float(np.isfinite(d).mean())


def pair_features(P, Q, pcfg: PlantConfig, cache: dict) -> dict:
    bd = float(np.hypot(*(P.base[:2] - Q.base[:2])))
    ld = float(np.hypot(*(P.lean() - Q.lean())))
    early, late = (P, Q) if P.date < Q.date else (Q, P)
    ct = containment(early, late, pcfg, cache)
    c = bd / pcfg.base_scale + ld / pcfg.lean_scale + pcfg.contain_weight * (1 - (ct if np.isfinite(ct) else 0.0))
    return {"cost": c, "base_dist_cm": 100 * bd, "lean_diff_deg": float(np.degrees(np.arctan(ld))),
            "containment": ct}


def check_plants(run, clusters: dict, links_all: dict, plants: dict, ccfg: ClusterConfig,
                 pcfg: PlantConfig, log=print) -> tuple[list[dict], dict]:
    """Returns rows and {date: {ref pid: date pid}} (the assignment)."""
    ref = run.reference
    R = plants.get(ref, {})
    Rc = [c for c in clusters[ref] if not c.clod]
    rows, assign = [], {}
    cache = {}
    for d in run.dates:
        if d == ref or not plants.get(d):
            continue
        Qd = plants[d]
        Q = [c for c in clusters[d] if not c.clod]
        allowed = {}       # ref cluster id -> set of date cluster ids it was matched to
        for L in links_all.get(d, []):
            for i in L["ref"]:
                allowed.setdefault(Rc[i].cid, set()).update(Q[j].cid for j in L["date"])
        rp, qp = list(R.values()), list(Qd.values())
        C = np.full((len(rp), len(qp)), np.inf)
        feats = {}
        for i, P in enumerate(rp):
            pc = getattr(P, "cluster", None)
            for j, Qp in enumerate(qp):
                if np.hypot(*(P.base[:2] - Qp.base[:2])) > pcfg.search_radius:
                    continue
                qc = getattr(Qp, "cluster", None)
                if pc is not None and pc in allowed and qc is not None and qc not in allowed[pc]:
                    continue        # outside the clusters this reference cluster was matched to
                f = pair_features(P, Qp, pcfg, cache)
                feats[(i, j)] = f
                C[i, j] = f["cost"]
        big = 1e6
        Cs = np.where(np.isfinite(C) & (C <= pcfg.max_cost), C, big)
        ri, ci = linear_sum_assignment(Cs) if Cs.size else (np.array([], int), np.array([], int))
        got = {i: j for i, j in zip(ri, ci) if Cs[i, j] < big}
        assign[d] = {}
        used = set()
        for i, P in enumerate(rp):
            j = got.get(i)
            cand = sorted(C[i][np.isfinite(C[i])])
            row = {"date": d, "reference_plant": names.nice(P.pid), "reference_cluster": getattr(P, "cluster", "") or "",
                   "n_candidates": len(cand)}
            if j is None:
                row.update(status="missing", matched_plant="", note="no candidate within "
                           f"{100 * pcfg.search_radius:.0f} cm in the matched clusters" if not cand else
                           "candidates exist but cost too high")
            else:
                Qp = qp[j]
                used.add(j)
                f = feats[(i, j)]
                others = [C[i, k] for k in range(len(qp)) if k != j and np.isfinite(C[i, k])]
                others += [C[k, j] for k in range(len(rp)) if k != i and np.isfinite(C[k, j])]
                second = min(others) if others else np.inf
                if Qp.pid != P.pid:
                    st = "swapped" if (P.pid in Qd or Qp.pid in R) else "other name"
                elif second - f["cost"] < pcfg.ambiguous_margin:
                    st = "ambiguous"
                else:
                    st = "confirmed"
                assign[d][P.pid] = Qp.pid
                row.update(status=st, matched_plant=names.nice(Qp.pid), cost=round(f["cost"], 3),
                           second_best_cost=round(second, 3) if np.isfinite(second) else "",
                           base_dist_cm=round(f["base_dist_cm"], 2), lean_diff_deg=round(f["lean_diff_deg"], 1),
                           containment=round(f["containment"], 3) if np.isfinite(f["containment"]) else "",
                           fine_shift_cm=round(100 * float(np.hypot(*Qp.fine)), 2),
                           note={"swapped": f"both names exist on both dates; geometry prefers {names.nice(Qp.pid)}",
                                 "other name": f"{names.nice(P.pid)} not separated on this date and "
                                               f"{names.nice(Qp.pid)} not on 29.06.: likely one plant, two register entries",
                                 }.get(st, ""))
            rows.append(row)
        for j, Qp in enumerate(qp):
            if j not in used:
                rows.append({"date": d, "reference_plant": "", "matched_plant": names.nice(Qp.pid),
                             "status": "not on the reference date",
                             "note": "no separated plant of 29.06. matches it" if Qp.pid not in R else
                             "its namesake on the reference date was assigned elsewhere"})
        cnt = {}
        for r in rows:
            if r["date"] == d:
                cnt[r["status"]] = cnt.get(r["status"], 0) + 1
        log(f"  {d}: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())))
    return rows, assign
