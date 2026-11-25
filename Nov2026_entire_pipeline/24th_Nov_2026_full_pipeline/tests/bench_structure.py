"""Compare stem/leaf segmentation methods against the labelled plants.

    python tests/bench_structure.py [method ...]

For every GT plant and method: stem IoU, leaf IoU (fragments joined),
leaf count, and the trait errors (same trait code on GT labels and on the
predicted labels, so the difference comes from the segmentation only).
Writes outputs/_bench_structure/results.csv and prints a summary.
"""

from __future__ import annotations

import contextlib
import csv
import io
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "tests"))

from pipeline import stem_tracking as st  # noqa: E402
from pipeline.config import OUTPUT_ROOT, GroundTruthConfig  # noqa: E402
from pipeline.ground_truth import PredPlant, compare_traits, evaluate_segmentation, load_ground_truth  # noqa: E402
from pipeline.structure import structure_from_labels  # noqa: E402
from test_ground_truth import GT_SRC, PLANTS  # noqa: E402

METHODS = {
    "reference": st.segment_reference,
    "reference+measure": st.with_measuring_cloud(st.segment_reference),
    "tracked+measure": st.with_measuring_cloud(st.segment_tracked),
    "v2+measure (default)": st.segment_default,
}
TRAITS = ["stem_length_m", "stem_radius_m", "n_leaves", "total_leaf_area_m2",
          "mean_leaf_length_m", "mean_leaf_angle_to_stem_deg", "plant_height_m"]
SHORT = {"stem_length_m": "stemL", "stem_radius_m": "stemR", "n_leaves": "nLeaf", "total_leaf_area_m2": "area",
         "mean_leaf_length_m": "leafL", "mean_leaf_angle_to_stem_deg": "angle", "plant_height_m": "height"}

_GT_CACHE: dict = {}


def gt_data(name):
    if name not in _GT_CACHE:
        gt = load_ground_truth(GT_SRC / f"{name}_asc", date="230621", log=lambda *_: None)
        with contextlib.redirect_stdout(io.StringIO()):
            gts = structure_from_labels(gt.xyz, gt.labels, merge_fragments=False)
        _GT_CACHE[name] = (gt, gts)
    return _GT_CACHE[name]


def score(name, method_name, fn, **kw) -> dict:
    gt, gts = gt_data(name)
    gcfg = GroundTruthConfig()
    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        xyz, lab = fn(gt.xyz, **kw)
        ps = structure_from_labels(xyz, lab)
    dt = time.time() - t
    pred = PredPlant("p", xyz, lab, np.array([ps.traits[c] for c in ("base_x", "base_y", "base_z")]),
                     ps.traits, ps.leaves, ps.merges)
    seg = evaluate_segmentation(gt.xyz, gt.labels, pred, gcfg)
    rows, leaf_rows = compare_traits(gts, pred, seg)
    # per matched leaf (not affected by extra small pieces counted as leaves):
    # median relative error of length / area, median absolute angle error
    leaf_err = {}
    for t, k in (("leaf_length_m", "lfL1"), ("leaf_area_m2", "lfA1"), ("leaf_angle_to_stem_deg", "lfAng1")):
        e = [r["abs_error"] if k == "lfAng1" else r["rel_error_pct"] for r in leaf_rows
             if r["trait"] == t and np.isfinite(r["abs_error"])]
        leaf_err[k] = float(np.median(e)) if e else float("nan")
    out = {"method": method_name, "plant": name.split("_Plant")[0], "time_s": round(dt, 1),
           "stem_iou": seg.get("stem_iou"), "leaf_iou": seg.get("mean_leaf_iou_with_fragments"),
           "gt_leaves": seg.get("n_gt_leaves"), "pred_leaves": ps.traits["n_leaves"]}
    out.update(leaf_err)
    for r in rows:
        if r["trait"] in TRAITS:
            g, p = r["ground_truth"], r["predicted"]
            out[f"gt_{r['trait']}"], out[f"pred_{r['trait']}"] = g, p
            out[f"err_{SHORT[r['trait']]}"] = (p - g) / g * 100 if (g and np.isfinite(g) and np.isfinite(p)
                                                                       and r["trait"] != "mean_leaf_angle_to_stem_deg") \
                else (p - g if r["trait"] == "mean_leaf_angle_to_stem_deg" else float("nan"))
    return out


def summarise(rows: list[dict]) -> None:
    keys = ["stem_iou", "leaf_iou"] + [f"err_{SHORT[t]}" for t in TRAITS] + ["lfL1", "lfA1", "lfAng1"]
    print(f"\n{'method':<16}{'plant':<12}{'stemIoU':>8}{'leafIoU':>8}{'leaves':>8}"
          + "".join(f"{SHORT[t]:>8}" for t in TRAITS if t != "n_leaves") + f"{'time':>6}")
    for r in rows:
        print(f"{r['method']:<16}{r['plant']:<12}{r['stem_iou']:>8.2f}{r['leaf_iou']:>8.2f}"
              f"{str(r['pred_leaves']) + '/' + str(r['gt_leaves']):>8}"
              + "".join(f"{r.get('err_' + SHORT[t], float('nan')):>+8.0f}" for t in TRAITS if t != "n_leaves")
              + f"{r['time_s']:>6}")
    print("\nMean over plants (errors: mean absolute; % except angle = degrees):")
    for m in dict.fromkeys(r["method"] for r in rows):
        rs = [r for r in rows if r["method"] == m]
        parts = []
        for k in keys:
            v = np.array([r.get(k, np.nan) for r in rs], float)
            v = v[np.isfinite(v)]
            parts.append(f"{k.replace('err_', '')}={np.mean(np.abs(v)) if ('err' in k or k.startswith('lf')) else np.mean(v):.2f}" if len(v) else f"{k}=nan")
        nl = np.mean([abs(r["pred_leaves"] - r["gt_leaves"]) for r in rs])
        print(f"  {m:<16} " + "  ".join(parts) + f"  leafcount_err={nl:.2f}")


def main(names: list[str]) -> list[dict]:
    rows = []
    for m in names:
        for p in PLANTS:
            rows.append(score(p, m, METHODS[m]))
    out = OUTPUT_ROOT / "_bench_structure"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k)))
        w.writeheader()
        w.writerows(rows)
    summarise(rows)
    return rows


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main(sys.argv[1:] or list(METHODS))
