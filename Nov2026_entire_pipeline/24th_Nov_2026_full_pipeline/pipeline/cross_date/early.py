"""Parts of stage 9 that run earlier in the pipeline, so later stages use them.

3b (inside stage 4, before the plant register): clusters per date are found on
   the stem-base layer, matched to the reference date, and each matched cluster
   gets a fine shift (image cross-correlation, accepted <= 3 cm). The shifts are
   stored in the date's Alignment (local_shifts), so the register pooling, the
   plant separation (register -> date) and everything after use them.
6b (after the plant structure, before tracking): plant check + stem matching
   (stage 9 steps 1-6, saved in <run>/cross_date_matching and reused by stage 9).
   Where one plant carries two register names - each separated on only one of
   two dates, and the geometry says it is the same plant - tracking and growth
   use the reference date's name.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .config import ClusterConfig


def _write_csv(path: Path, rows: list[dict]) -> None:
    from .stage import _write_csv as w
    w(path, rows)


def cluster_fine_shifts(run_root: Path, dates: list[str], ref: str, als: dict, log=print) -> dict:
    """{date: local_shifts dict for Alignment} (reference date: none). Writes
    <run>/alignment/cluster_fine_shift/ (clusters, matches, fine shifts)."""
    from .inputs import load_run
    from .clusters import find_clusters
    from .cluster_match import match_all, name_reference_clusters
    from .fine_align import fine_align

    cfg = ClusterConfig()
    out = Path(run_root) / "alignment" / "cluster_fine_shift"
    out.mkdir(parents=True, exist_ok=True)
    quiet = lambda *a, **k: None  # noqa: E731
    run = load_run(Path(run_root), cfg, quiet, alignments=als, display=False)
    clusters, _ = find_clusters(run, cfg, quiet)
    names, _ = name_reference_clusters(clusters[ref], [])
    rows, links, _ = match_all(clusters, run.dates, ref, cfg, names, quiet)
    fine = fine_align(run, clusters, links, ref, cfg, names, quiet)
    _write_csv(out / "clusters_per_date.csv", [c.row() for d in run.dates for c in clusters[d]])
    _write_csv(out / "cluster_match.csv", rows)
    _write_csv(out / "fine_alignment.csv", fine)
    by_id = {c.cid: c for d in run.dates for c in clusters[d]}
    result = {}
    for d in dates:
        if d == ref:
            continue
        items, seen = [], set()
        for r in fine:
            if r["date"] != d or not r["accepted"]:
                continue
            for q in r["date_clusters"].split():
                if q in seen or q not in by_id:
                    continue
                seen.add(q)
                c = by_id[q]
                items.append([round(c.lo, 4), round(c.hi, 4), r["dx_cm"] / 100.0, r["dy_cm"] / 100.0])
        if items:
            result[d] = {"origin": run.frame.origin.tolist(), "along": run.frame.along.tolist(),
                         "margin": 0.03, "items": sorted(items)}
            sz = [100 * np.hypot(i[2], i[3]) for i in items]
            log(f"  cluster fine shift {d}: {len(items)} clusters, median {np.median(sz):.1f} cm, max {max(sz):.1f} cm")
        else:
            log(f"  cluster fine shift {d}: no cluster shift accepted")
    (out / "local_shifts.json").write_text(json.dumps(result, indent=1))
    return result


def identity_check(run_root: Path, register: list[dict], per_date: dict, log=print) -> dict:
    """Stage 9 steps 1-6 now; then merge 'one plant, two register names' for tracking.
    Changes per_date in place; returns {date: {name on that date: reference name}}."""
    from ..config import (IDENTITY_MERGE_MAX_STEM_DEV_CM, IDENTITY_MERGE_MIN_COMMON_CM,
                          IDENTITY_MERGE_MIN_CONTAINMENT)
    from .stage import run_cross_date
    from pipeline import names

    out = run_cross_date(Path(run_root), until="stems", figures=True, log=log)
    rows = list(csv.DictReader(open(out / "02_plants_stems" / "plant_check.csv")))
    stems = {(r["date"], r["reference_plant"], r["matched_plant"]): r
             for r in csv.DictReader(open(out / "02_plants_stems" / "stem_match.csv"))}
    merges, table = {}, []
    for r in rows:
        if r.get("status") != "other name":
            continue
        ref_pid, d_pid, d = names.raw(r["reference_plant"]), names.raw(r["matched_plant"]), r["date"]
        try:
            cont, base = float(r["containment"]), float(r["base_dist_cm"])
        except (TypeError, ValueError):
            continue
        s = stems.get((d, r["reference_plant"], r["matched_plant"]), {})
        try:
            dev, common = float(s["mean_deviation_cm"]), float(s["common_height_cm"])
        except (KeyError, TypeError, ValueError):
            dev, common = float("nan"), 0.0
        why = []
        if cont < IDENTITY_MERGE_MIN_CONTAINMENT:
            why.append(f"containment {cont:.2f} < {IDENTITY_MERGE_MIN_CONTAINMENT}")
        if not dev <= IDENTITY_MERGE_MAX_STEM_DEV_CM:
            why.append(f"stems {dev:.1f} cm apart at the same heights (limit {IDENTITY_MERGE_MAX_STEM_DEV_CM:.0f} cm)")
        if common < IDENTITY_MERGE_MIN_COMMON_CM:
            why.append(f"only {common:.0f} cm of common stem height")
        ok = not why
        table.append({"date": d, "name_on_date": r["matched_plant"], "reference_name": r["reference_plant"],
                      "containment": cont, "stem_deviation_cm": dev, "common_stem_height_cm": common,
                      "lowest_points_dist_cm": base, "merged": ok, "reason": "; ".join(why)})
        if not ok:
            continue
        row = per_date.get(d, {}).get(d_pid)
        if not row or (per_date[d].get(ref_pid) or {}).get("presence_present"):
            continue
        merges.setdefault(d, {})[d_pid] = ref_pid
        moved = dict(row)
        moved["plant_id"] = ref_pid
        moved["identity_note"] = f"separated as {names.nice(d_pid)} on this date (same plant, stage 6b)"
        per_date[d][ref_pid] = moved
        per_date[d][d_pid] = {"plant_id": d_pid, "date": d, "status": f"merged into {ref_pid}",
                              "presence_present": False, "identity_note": f"same plant as {names.nice(ref_pid)}"}
    trk = Path(run_root) / "tracking"
    trk.mkdir(parents=True, exist_ok=True)
    _write_csv(trk / "identity_check.csv", table)
    if table:
        try:
            from .figures_identity import identity_figure
            identity_figure(Path(run_root))           # tracking/identity_check.png
        except Exception as e:  # noqa: BLE001
            log(f"  identity check figure failed: {e}")
    n = sum(len(v) for v in merges.values())
    log(f"  identity check: {len(table)} plants with two register names, {n} merged for tracking"
        + "".join(f"; {d}: " + ", ".join(f"{names.nice(a)} -> {names.nice(b)}" for a, b in m.items())
                  for d, m in merges.items()))
    return merges
