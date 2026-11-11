"""Measurement tolerances (expected error, +-) shown in the report and the app.

Traits: mean absolute error against hand-labelled ground truth. A run with its
own ground-truth check uses its own errors; otherwise the errors measured on
the 3 matched ground-truth plants of 21.06 (LtR_10th / 11th / 12th) with the
user's structure code (copy_of_New_optimized_plant_structure.py, no extra
leaf check) are used.

Alignment: x/y = scatter (RMS) of the agreeing 1.5 m row sections around the
final shift; z = scatter of the soil height difference between the two dates
over the same sections, after the shift.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# column -> (ground-truth summary key, factor to the column's unit)
TRAIT_KEYS = {
    "plant_height_m": ("err_plant_height_m", 1.0),
    "stem_length_m": ("err_stem_length_m", 1.0),
    "stem_diameter_mm": ("err_stem_radius_m", 2000.0),
    "n_leaves": ("err_n_leaves", 1.0),
    "total_leaf_area_m2": ("err_total_leaf_area_m2", 1.0),
    "mean_leaf_length_m": ("err_mean_leaf_length_m", 1.0),
    "mean_leaf_width_m": ("mae_leaf_max_width_m", 1.0),
    "max_leaf_width_m": ("mae_leaf_max_width_m", 1.0),
    "mean_leaf_angle_to_stem_deg": ("err_mean_leaf_angle_to_stem_deg", 1.0),
}
# measured 2026-11-03, outputs/_test_user_code_gt_raw (user's structure code), mean of 3 plants
DEFAULT_TRAIT_TOL = {
    "plant_height_m": 0.032, "stem_length_m": 0.096, "stem_diameter_mm": 9.4, "n_leaves": 5.7,
    "total_leaf_area_m2": 0.0139, "mean_leaf_length_m": 0.123, "mean_leaf_width_m": 0.016,
    "max_leaf_width_m": 0.016, "mean_leaf_angle_to_stem_deg": 28.5,
}
DEFAULT_SOURCE = "ground-truth check of 21.06 (3 hand-labelled plants), your structure code"


def trait_tolerance(run_root: Path) -> tuple[dict, str]:
    """{column: +- value in the column's unit}, and where the values come from."""
    f = Path(run_root) / "ground_truth" / "gt_summary.csv"
    if f.exists():
        rows = [r for r in csv.DictReader(open(f)) if r.get("matched_plant") and r.get("plant_iou")
                and float(r["plant_iou"] or 0) > 0.3]
        if rows:
            out = {}
            for col, (k, fac) in TRAIT_KEYS.items():
                v = [abs(float(r[k])) * fac for r in rows if r.get(k) not in ("", None, "nan")]
                v = [x for x in v if np.isfinite(x)]
                out[col] = float(np.mean(v)) if v else DEFAULT_TRAIT_TOL[col]
            return out, f"ground-truth check of this run ({len(rows)} plants)"
    return dict(DEFAULT_TRAIT_TOL), DEFAULT_SOURCE


def alignment_tolerance(run_root: Path) -> dict:
    """{date: {"xy_cm", "z_cm", "n_sections"}} (nan when it cannot be estimated)."""
    root = Path(run_root)
    af = root / "alignment" / "alignment.json"
    if not af.exists():
        return {}
    d = json.loads(af.read_text())
    dates = d.get("dates", {})
    ref = next((a["reference"] for a in dates.values() if a.get("reference")), None)
    out = {}
    g_ref = _ground(root, ref) if ref else None
    for date, a in dates.items():
        if date == ref:
            out[date] = {"xy_cm": 0.0, "z_cm": 0.0, "n_sections": 0}
            continue
        used = [s for s in a.get("sections", []) if s.get("used")]
        k = np.asarray(a.get("dx_profile", [[0, 0, 0]]), float)
        if len(used) >= 2:
            dx = np.array([s["dx"] for s in used]) - (np.interp([s["s_mid"] for s in used], k[:, 0], k[:, 1]) if len(k) > 1 else k[0, 1])
            dy = np.array([s["dy"] for s in used]) - (np.interp([s["s_mid"] for s in used], k[:, 0], k[:, 2]) if len(k) > 1 else k[0, 2])
            xy = float(np.sqrt(np.mean(dx ** 2 + dy ** 2)) * 100)
        else:
            xy = float("nan")
        z = _z_scatter(_ground(root, date), g_ref, a) if g_ref is not None else float("nan")
        out[date] = {"xy_cm": xy, "z_cm": z, "n_sections": len(used)}
    return out


def _ground(root: Path, date: str):
    f = root / date / "01_soil" / "ground_1cm.npz"
    if not f.exists():
        return None
    g = np.load(f)
    return {"ix": g["ix"].astype(np.int64), "iy": g["iy"].astype(np.int64), "z": g["z"].astype(float),
            "cell": float(g["cell"])}


def _z_scatter(g, g_ref, a) -> float:
    """RMS over 1.5 m row sections of the median soil height difference after the shift (cm)."""
    if g is None or g_ref is None:
        return float("nan")
    c = g["cell"]
    k = np.asarray(a["dx_profile"], float)
    dx, dy = (np.median(k[:, 1]), np.median(k[:, 2]))
    ix = g["ix"] + int(round(dx / c))
    iy = g["iy"] + int(round(dy / c))
    key = (ix << 32) + iy
    rkey = (g_ref["ix"] << 32) + g_ref["iy"]
    common, i1, i2 = np.intersect1d(key, rkey, return_indices=True)
    if len(common) < 1000:
        return float("nan")
    diff = g_ref["z"][i2] - (g["z"][i1] + a["dz"])
    x = (ix[i1] + 0.5) * c
    y = (iy[i1] + 0.5) * c
    o, al = np.asarray(a["origin_xy"]), np.asarray(a["along"])
    s = (x - o[0]) * al[0] + (y - o[1]) * al[1]
    med = []
    for lo in np.arange(s.min(), s.max(), 1.5):
        m = (s >= lo) & (s < lo + 1.5)
        if m.sum() >= 200:
            med.append(np.median(diff[m]))
    if len(med) < 2:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(med))) * 100)


def fmt_tol(col: str, v: float | None) -> str:
    """'+- value' in the unit of a report column."""
    if v is None or not np.isfinite(v):
        return ""
    if col.endswith("_m2"):
        return f"±{v * 1e4:.0f} cm²"
    if col.endswith("_m"):
        return f"±{v * 100:.1f} cm"
    if col.endswith("_mm"):
        return f"±{v:.1f} mm"
    if col.endswith("_deg"):
        return f"±{v:.0f}°"
    return f"±{v:.1f}"
