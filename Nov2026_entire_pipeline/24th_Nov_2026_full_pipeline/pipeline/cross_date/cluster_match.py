"""Step 3: match the clusters of every date to the reference-date clusters.

Method A - order-preserving sequence alignment along the row (dynamic
programming). Clusters cannot swap order along a single row. Pair cost =
1 - Bhattacharyya overlap of the two Gaussians; one cluster may match up to
`max_split` neighbouring clusters on the other date (split / merge); a cluster
can stay unmatched (missing / extra) at `gap_cost`.

Method B - DBSCAN on the cluster centres (along, across the row) of all dates
pooled. A group = one physical cluster seen on several dates.

Where A and B agree the match is 'confirmed' (or 'split' / 'merged' /
'missing'); where they disagree it is 'flagged'.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

from pipeline import names

from .clusters import Cluster
from .config import ClusterConfig


def bc(mu1, s1, mu2, s2) -> float:
    """Bhattacharyya coefficient of two 1-D Gaussians (1 = identical, 0 = no overlap)."""
    v = s1 * s1 + s2 * s2
    return float(np.sqrt(2 * s1 * s2 / v) * np.exp(-((mu1 - mu2) ** 2) / (4 * v)))


def merge_gauss(cs: list[Cluster]) -> tuple[float, float]:
    m = np.array([c.mass for c in cs])
    mu = np.array([c.mu for c in cs])
    sg = np.array([c.sigma for c in cs])
    M = (m * mu).sum() / m.sum()
    var = (m * (sg ** 2 + mu ** 2)).sum() / m.sum() - M * M
    return float(M), float(np.sqrt(max(var, 1e-8)))


def _link_cost(R: list[Cluster], Q: list[Cluster], cfg: ClusterConfig) -> tuple[float, float]:
    mr, sr = merge_gauss(R)
    mq, sq = merge_gauss(Q)
    if abs(mr - mq) > cfg.max_centre_dist:
        return np.inf, 0.0
    lo, hi = min(c.lo for c in R) - 0.03, max(c.hi for c in R) + 0.03
    if any(not (lo <= c.mu <= hi) for c in Q):
        return np.inf, 0.0
    lo, hi = min(c.lo for c in Q) - 0.03, max(c.hi for c in Q) + 0.03
    if any(not (lo <= c.mu <= hi) for c in R):
        return np.inf, 0.0
    b = bc(mr, sr, mq, sq)
    if b < cfg.bc_min:
        return np.inf, b
    return 1.0 - b + cfg.split_penalty * (len(R) + len(Q) - 2), b


def sequence_align(R: list[Cluster], Q: list[Cluster], cfg: ClusterConfig) -> list[dict]:
    """Order-preserving alignment of reference clusters R and date clusters Q
    (both sorted along the row). Returns links: {ref: [i..], date: [j..], cost, bc}."""
    n, m = len(R), len(Q)
    moves = [(1, 0), (0, 1)] + [(1, k) for k in range(1, cfg.max_split + 1)] + \
            [(k, 1) for k in range(2, cfg.max_split + 1)]
    D = np.full((n + 1, m + 1), np.inf)
    back = {}
    D[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if not np.isfinite(D[i, j]):
                continue
            for a, b in moves:
                ii, jj = i + a, j + b
                if ii > n or jj > m:
                    continue
                if a == 0 or b == 0:
                    c, bb = cfg.gap_cost, 0.0
                else:
                    c, bb = _link_cost(R[i:ii], Q[j:jj], cfg)
                if D[i, j] + c < D[ii, jj] - 1e-12:
                    D[ii, jj] = D[i, j] + c
                    back[(ii, jj)] = (i, j, c, bb)
    links, i, j = [], n, m
    while (i, j) != (0, 0):
        pi, pj, c, bb = back[(i, j)]
        links.append({"ref": list(range(pi, i)), "date": list(range(pj, j)), "cost": c, "bc": bb})
        i, j = pi, pj
    return links[::-1]


def _kind(link) -> str:
    nr, nq = len(link["ref"]), len(link["date"])
    if nq == 0:
        return "missing"
    if nr == 0:
        return "extra"
    if nr == 1 and nq == 1:
        return "confirmed"
    return "split" if nq > 1 else "merged"


def run_dbscan(clusters: dict, dates: list, cfg: ClusterConfig) -> dict:
    """{cid: group label (-1 = alone)} for all non-clod clusters of all dates."""
    items = [c for d in dates for c in clusters[d] if not c.clod]
    X = np.array([[c.mu, c.t] for c in items])
    lab = DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples).fit_predict(X)
    return {c.cid: int(l) for c, l in zip(items, lab)}


def name_reference_clusters(ref_clusters: list[Cluster], register: list) -> dict:
    """Name each reference cluster after the register cluster it corresponds to
    one-to-one (plant_cluster_NN); otherwise ref_cluster_KK. Also returns the
    register comparison rows."""
    groups = {}
    for r in register:
        groups.setdefault(r["group"], []).append(r)
    inside = {}   # ref cid -> set of register groups with a plant inside
    where = {}    # group -> set of ref cids holding its plants (None = outside every cluster)
    for g, plants in groups.items():
        for p in plants:
            hit = [c.cid for c in ref_clusters if c.lo - 0.01 <= p["s"] <= c.hi + 0.01 and not c.clod]
            where.setdefault(g, set()).update(hit or [None])
            for h in hit:
                inside.setdefault(h, set()).add(g)
    out, comp = {}, []
    for k, c in enumerate(ref_clusters):
        gs = sorted(inside.get(c.cid, set()))
        one_to_one = len(gs) == 1 and where[gs[0]] == {c.cid}
        nm = names.cluster(gs[0]) if one_to_one else f"ref_cluster_{k + 1:02d}"
        out[c.cid] = nm
        comp.append({"reference_cluster": c.cid, "name": nm,
                     "register_clusters": " + ".join(names.cluster(g) for g in gs),
                     "relation": ("same" if one_to_one else
                                  "no register plant" if not gs else
                                  "holds several register clusters" if len(gs) > 1 else
                                  "register cluster spread over several clusters")})
    for g, w in sorted(where.items()):
        if None in w:
            comp.append({"reference_cluster": "", "name": "", "register_clusters": names.cluster(g),
                         "relation": "register plant(s) outside every reference cluster: " +
                         " ".join(names.nice(p["plant_id"]) for p in groups[g]
                                  if not any(c.lo - 0.01 <= p["s"] <= c.hi + 0.01 for c in ref_clusters if not c.clod))})
    return out, comp


def match_all(clusters: dict, dates: list, ref: str, cfg: ClusterConfig, ref_names: dict, log=print):
    R = [c for c in clusters[ref] if not c.clod]
    lab = run_dbscan(clusters, dates, cfg)
    by_label = {}
    for d in dates:
        for c in clusters[d]:
            if c.cid in lab and lab[c.cid] >= 0:
                by_label.setdefault(lab[c.cid], []).append(c)

    # DBSCAN: more than one cluster of one date in a group?
    crowded = {}
    for l, cs in by_label.items():
        ds = [c.date for c in cs]
        dup = sorted({d for d in ds if ds.count(d) > 1})
        if dup:
            crowded[l] = dup

    rows, links_all = [], {}
    for d in dates:
        if d == ref:
            for c in R:
                rows.append({"reference_cluster": c.cid, "name": ref_names[c.cid], "date": d,
                             "date_clusters": c.cid, "method_A": "reference", "method_B": "reference",
                             "status": "reference", "overlap_bc": 1.0, "centre_diff_cm": 0.0, "note": ""})
            continue
        Q = [c for c in clusters[d] if not c.clod]
        links = sequence_align(R, Q, cfg)
        links_all[d] = links
        for L in links:
            kind = _kind(L)
            Rs, Qs = [R[i] for i in L["ref"]], [Q[j] for j in L["date"]]
            qset = {q.cid for q in Qs}
            # what DBSCAN says for the clusters of this link:
            #   support       = a reference cluster of the link is grouped with a date cluster of the link
            #   contradiction = a link member is grouped with a cluster OUTSIDE the link
            rset = {r.cid for r in Rs}
            b_dates, outside_ref = set(), set()
            for r in Rs:
                l = lab.get(r.cid, -1)
                if l >= 0:
                    b_dates |= {c.cid for c in by_label[l] if c.date == d}
            for q in Qs:
                l = lab.get(q.cid, -1)
                if l >= 0:
                    outside_ref |= {c.cid for c in by_label[l] if c.date == ref and c.cid not in rset}
            support = bool(b_dates & qset)
            contradiction = bool(b_dates - qset) or bool(outside_ref)
            if kind in ("extra", "missing"):
                agree, verdict = not contradiction, ("agrees" if not contradiction else "disagrees")
            elif kind == "confirmed":
                agree = support and not contradiction
                verdict = "agrees" if agree else "disagrees"
            else:   # split / merged: a centre between two clusters is never grouped -> may be untestable
                agree = not contradiction
                verdict = "disagrees" if contradiction else ("agrees" if support else "cannot test")
            notes = []
            if not agree:
                if contradiction:
                    notes.append("DBSCAN groups " + ", ".join(sorted((b_dates - qset) | outside_ref)) + " with this link")
                else:
                    notes.append("DBSCAN: centres farther apart than eps")
            for q in Qs + Rs:
                l = lab.get(q.cid, -1)
                if l in crowded:
                    notes.append(f"DBSCAN group holds 2+ clusters of {'/'.join(crowded[l])}")
                    break
            status = kind if agree else "flagged"
            if Rs:
                mr, _ = merge_gauss(Rs)
            if Qs and Rs:
                mq, _ = merge_gauss(Qs)
                cd = round(100 * (mq - mr), 2)
            else:
                cd = ""
            for r in Rs or [None]:
                rows.append({"reference_cluster": r.cid if r else "", "name": ref_names[r.cid] if r else "",
                             "date": d, "date_clusters": " ".join(sorted(qset)), "method_A": kind,
                             "method_B": verdict, "status": status,
                             "overlap_bc": round(L["bc"], 3) if Qs and Rs else "", "centre_diff_cm": cd,
                             "note": "; ".join(dict.fromkeys(notes))})
        for c in clusters[d]:
            if c.clod:
                rows.append({"reference_cluster": "", "name": "", "date": d, "date_clusters": c.cid,
                             "method_A": "not matched", "method_B": "not matched", "status": "clod",
                             "overlap_bc": "", "centre_diff_cm": "",
                             "note": f"plant points never above {100 * c.top_hag:.1f} cm"})
        n = {}
        for r in rows:
            if r["date"] == d:
                n[r["status"]] = n.get(r["status"], 0) + 1
        log(f"  {d}: " + ", ".join(f"{k} {v}" for k, v in sorted(n.items())))
    return rows, links_all, lab
