"""Step 7, tuning: choose the leaf-piece rule on the hand-labelled 21.06. plants.

Two kinds of evaluation case per ground-truth plant:
  A. the ground-truth cloud itself, run through the user's structure code
     (pipeline.cross_date.gt_structure) - same points, so labels compare point by point;
  B. the pipeline's separated 21.06. plant that holds most of the ground-truth
     points (the cloud the rule is really used on, with neighbour pieces);
     ground-truth labels are compared on the ground-truth points, each taking
     the label of the pipeline point within 5 mm.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from pipeline import names
from pipeline.config import PROJECT_ORIGIN

from .gt_structure import GT_NAMES
from .leaf_pieces import piece_geometry

O = np.asarray(PROJECT_ORIGIN)


def _info_from(leaves: list, labels_order, paths) -> dict:
    by = {int(l["leaf_label"]): l for l in leaves}
    info = {}
    for lab, p in zip(labels_order, paths):
        l = by.get(int(lab))
        if l is not None:
            info[int(lab)] = (float(l["leaf_length_m"]), float(l["leaf_area_m2"]), np.asarray(p))
    return info


def gt_cases(gt_struct_dir: Path, run_dir: Path, gt_date: str = "230621", log=print) -> list[dict]:
    cases = []
    # pipeline plants of the ground-truth date (full resolution, UTM)
    pdir = run_dir / gt_date / "03_plants"
    plist = []
    if (pdir / "plants_index.csv").exists():
        for r in csv.DictReader(open(pdir / "plants_index.csv")):
            stem = pdir / r["file"].replace("_labelled.npz", "")
            if Path(str(stem) + "_skeleton.npz").exists():
                plist.append((r["plant_id"], stem))
    loaded = {}
    for n in GT_NAMES:
        f = gt_struct_dir / f"{n}.npz"
        if not f.exists():
            log(f"  {n}: no structure result (run python -m pipeline.cross_date.gt_structure first)")
            continue
        z = np.load(f)
        meta = json.loads((gt_struct_dir / f"{n}.json").read_text())
        paths = np.split(z["leaf_points"], np.cumsum(z["leaf_sizes"])[:-1]) if len(z["leaf_sizes"]) else []
        info = _info_from(meta["leaves"], z["leaf_labels"], paths)
        short = n.split("_Plant_")[0] if "_Plant_" in n else n
        cases.append({"name": f"{short} (ground-truth cloud)", "kind": "A",
                      "geom": piece_geometry(z["xyz"], z["pred_labels"], info),
                      "labels": z["pred_labels"], "gt": z["gt_labels"]})
        # B: the pipeline plant that holds most of this ground-truth plant
        gxyz = z["xyz"] + O
        best = None
        for pid, stem in plist:
            if pid not in loaded:
                L = np.load(str(stem) + "_labelled.npz")
                loaded[pid] = (L["xyz"], L["labels"], cKDTree(L["xyz"]))
            xyz, lab, tree = loaded[pid]
            lo, hi = xyz.min(0), xyz.max(0)
            if np.any(gxyz.min(0) > hi + 0.05) or np.any(gxyz.max(0) < lo - 0.05):
                continue
            d, j = tree.query(gxyz, distance_upper_bound=0.005)
            cov = float(np.isfinite(d).mean())
            if best is None or cov > best[1]:
                best = (pid, cov, j, d)
        if best is None or best[1] < 0.3:
            log(f"  {short}: no pipeline plant holds >= 30 % of it" + (f" (best {best[1]:.0%})" if best else ""))
            continue
        pid, cov, j, d = best
        xyz, lab, _ = loaded[pid]
        stem = dict(plist)[pid]
        S = np.load(str(stem) + "_skeleton.npz")
        tj = json.loads(Path(str(stem) + "_traits.json").read_text())
        ppaths = np.split(np.asarray(S["leaf_points"]), np.cumsum(S["leaf_sizes"])[:-1]) if len(S["leaf_sizes"]) else []
        pinfo = _info_from(tj.get("leaves", []), S["leaf_labels"], ppaths)
        ev = np.full(len(gxyz), -9, np.int32)
        ok = np.isfinite(d)
        ev[ok] = lab[j[ok]]
        cases.append({"name": f"{short} in {names.nice(pid)} (pipeline, {cov:.0%} covered)", "kind": "B",
                      "geom": piece_geometry(xyz, lab, pinfo), "labels_full": lab,
                      "labels": ev, "gt": z["gt_labels"], "map_idx": j, "map_ok": ok})
        log(f"  {short}: ground truth {len(info)} pieces; pipeline plant {names.nice(pid)} covers {cov:.0%}, "
            f"{len(pinfo)} pieces")
    return cases
