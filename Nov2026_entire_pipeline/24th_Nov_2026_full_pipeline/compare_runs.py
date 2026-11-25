"""Compare two pipeline runs (e.g. without / with the cluster fine shift and the
identity check) on everything that can change.

    ../../venv/bin/python compare_runs.py outputs/run_..._base outputs/run_..._new

Writes <run B>/comparison_with_<run A>.md and prints it.
"""
from __future__ import annotations

import collections
import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np


def rows(p: Path) -> list[dict]:
    return list(csv.DictReader(open(p))) if p.exists() and p.stat().st_size else []


def f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def fmt(x, nd=2):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def metrics(run: Path) -> dict:
    m = {}
    cfg = json.loads((run / "run_config.json").read_text())
    dates = [d for _, d in cfg["scans"]]
    ref = cfg["reference_date"]
    m["switches"] = f"cluster fine shift {'on' if cfg.get('cluster_fine_shift') else 'off'}, " \
                    f"identity check {'on' if cfg.get('identity_check') else 'off'}"
    reg = rows(run / "tracking" / "plant_register.csv")
    m["register plants"] = len(reg)
    m["plant clusters"] = len({r["group"] for r in reg})
    sizes = collections.Counter(collections.Counter(r["group"] for r in reg).values())
    m["single-plant clusters"] = sizes.get(1, 0)
    m["plants seen on 2 dates only"] = sum(1 for r in reg if int(r["n_dates_seen"]) == 2)
    tr = rows(run / "tracking" / "traits_by_date.csv")
    for d in dates:
        m[f"present {d}"] = sum(1 for r in tr if r["date"] == d and r["present"] == "True")
    m["height drops (separation doubtful)"] = sum(1 for r in tr if r.get("check", "").startswith("height dropped"))
    for d in dates:
        h = [f(r.get("plant_height_m")) for r in tr if r["date"] == d and r["present"] == "True"]
        m[f"median height {d} (m)"] = med(h)
    gt = rows(run / "ground_truth" / "gt_summary.csv")
    for r in gt:
        n = r["gt_name"].split("_Plant_")[0]
        m[f"GT {n}: plant IoU"] = f(r.get("plant_iou"))
        m[f"GT {n}: matched plant"] = r.get("matched_plant", "")
    idc = rows(run / "tracking" / "identity_check.csv")
    m["identity check: names merged"] = sum(1 for r in idc if r.get("merged") == "True") if idc else "–"
    cd = run / "cross_date_matching"
    cm = rows(cd / "01_clusters" / "cluster_match.csv")
    for k in ("confirmed", "flagged"):
        m[f"clusters {k} (all dates)"] = sum(1 for r in cm if r["status"] == k)
    fa = rows(cd / "01_clusters" / "fine_alignment.csv")
    m["remaining fine shift, median (cm)"] = med([f(r["shift_cm"]) for r in fa if r["accepted"] == "True"])
    m["remaining fine shifts accepted"] = sum(1 for r in fa if r["accepted"] == "True")
    rc = rows(cd / "01_clusters" / "register_comparison.csv")
    m["29.06. clusters = register clusters"] = sum(1 for r in rc if r["relation"] == "same")
    pc = rows(cd / "02_plants_stems" / "plant_check.csv")
    for k in ("confirmed", "swapped", "other name", "ambiguous", "missing"):
        m[f"plant check {k}"] = sum(1 for r in pc if r["status"] == k)
    sm = rows(cd / "02_plants_stems" / "stem_match.csv")
    m["stem deviation, median (cm)"] = med([f(r.get("mean_deviation_cm")) for r in sm])
    m["stems > 3 cm apart"] = sum(1 for r in sm if (f(r.get("mean_deviation_cm")) or 0) > 3)
    lm = rows(cd / "03_leaves" / "leaf_match.csv")
    m["leaves matched"] = sum(1 for r in lm if r["status"].startswith("matched"))
    m["leaf matches flagged"] = sum(1 for r in lm if r["status"].startswith("flagged"))
    t = rows(cd / "03_leaves" / "leaf_tracks.csv")
    ids = collections.defaultdict(set)
    for r in t:
        if r["leaf_id"].startswith("L"):
            ids[(r["plant"], r["leaf_id"])].add(r["date"])
    m["leaves on >= 2 dates"] = sum(len(v) >= 2 for v in ids.values())
    m["leaves on >= 3 dates"] = sum(len(v) >= 3 for v in ids.values())
    return m, reg, ref


def register_diff(ra: list, rb: list, tol: float = 0.03) -> dict:
    A = np.array([[float(r["x_utm"]), float(r["y_utm"])] for r in ra])
    B = np.array([[float(r["x_utm"]), float(r["y_utm"])] for r in rb])
    if not len(A) or not len(B):
        return {}
    from scipy.spatial import cKDTree
    da, ja = cKDTree(B).query(A)
    same = da <= tol
    renamed = sum(1 for i in np.where(same)[0] if ra[i]["plant_id"] != rb[ja[i]]["plant_id"])
    db, _ = cKDTree(A).query(B)
    return {"same position (<= 3 cm)": int(same.sum()), "  ... with another name": renamed,
            "only in A": int((~same).sum()), "only in B": int((db > tol).sum()),
            "median position change (cm)": float(np.median(100 * da[same])) if same.any() else None}


def main():
    a, b = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    ma, ra, _ = metrics(a)
    mb, rb, _ = metrics(b)
    keys = list(dict.fromkeys([*ma, *mb]))
    lines = [f"# {b.name} compared with {a.name}", "", f"A = `{a.name}`  \nB = `{b.name}`", "",
             "| | A | B | change |", "|---|---|---|---|"]
    for k in keys:
        va, vb = ma.get(k), mb.get(k)
        ch = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
            dv = vb - va
            ch = "" if abs(dv) < 1e-9 else (f"{dv:+.2f}" if isinstance(dv, float) else f"{dv:+d}")
        lines.append(f"| {k} | {fmt(va)} | {fmt(vb)} | {ch} |")
    lines += ["", "## Register A -> B (plants matched by position)", ""]
    for k, v in register_diff(ra, rb).items():
        lines.append(f"- {k}: {fmt(v)}")
    text = "\n".join(lines) + "\n"
    out = b / f"comparison_with_{a.name}.md"
    out.write_text(text)
    print(text)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
