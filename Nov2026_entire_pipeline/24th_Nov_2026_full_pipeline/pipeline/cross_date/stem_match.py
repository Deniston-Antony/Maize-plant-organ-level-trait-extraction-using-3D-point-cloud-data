"""Step 6: match the stems of each assigned plant pair (reference <-> date).

A plant grows, so the earlier stem centre line should be the lower part of the
later one. Both centre lines (structure code, reference frame) are sampled
every 1 cm of height above ground over their common height range and compared
point by point in xy.
"""
from __future__ import annotations

import numpy as np

from pipeline import names

from .config import PlantConfig


def _xy_at(p, h: np.ndarray) -> np.ndarray:
    o = np.argsort(p.centerline_hag)
    hh, cl = p.centerline_hag[o], p.centerline[o]
    return np.column_stack([np.interp(h, hh, cl[:, 0]), np.interp(h, hh, cl[:, 1])])


def match_stems(run, plants: dict, assign: dict, pcfg: PlantConfig, log=print) -> list[dict]:
    ref = run.reference
    R = plants.get(ref, {})
    rows = []
    for d, pairs in assign.items():
        devs = []
        for rp, qp in pairs.items():
            P, Q = R[rp], plants[d][qp]
            early, late = (P, Q) if P.date < Q.date else (Q, P)
            lo = max(pcfg.stem_min_h, early.centerline_hag.min(), late.centerline_hag.min())
            hi = min(early.centerline_hag.max(), late.centerline_hag.max())
            row = {"date": d, "reference_plant": names.nice(rp), "matched_plant": names.nice(qp)}
            if hi - lo < 0.03:
                row.update(status="too short to compare", common_height_cm=round(100 * max(0.0, hi - lo), 1))
                rows.append(row)
                continue
            h = np.arange(lo, hi + 1e-9, pcfg.stem_step)
            dv = np.hypot(*(_xy_at(early, h) - _xy_at(late, h)).T)
            le, ll = early.lean(), late.lean()
            lean_e = float(np.degrees(np.arctan(np.hypot(*le))))
            lean_l = float(np.degrees(np.arctan(np.hypot(*ll))))
            mean_dev = float(100 * dv.mean())
            devs.append(mean_dev)
            row.update(status="flagged" if mean_dev > pcfg.stem_flag_cm else "matched",
                       common_height_cm=round(100 * (hi - lo), 1),
                       mean_deviation_cm=round(mean_dev, 2), max_deviation_cm=round(float(100 * dv.max()), 2),
                       deviation_at_bottom_cm=round(float(100 * dv[0]), 2),
                       lean_earlier_deg=round(lean_e, 1), lean_later_deg=round(lean_l, 1),
                       lean_diff_deg=round(float(np.degrees(np.arctan(np.hypot(*(le - ll))))), 1),
                       stem_length_earlier_m=round(early.stem_length, 3), stem_length_later_m=round(late.stem_length, 3),
                       stem_growth_m=round(late.stem_length - early.stem_length, 3),
                       note="" if mean_dev <= pcfg.stem_flag_cm else
                       f"stems more than {pcfg.stem_flag_cm:.0f} cm apart on average - check the plant match")
            rows.append(row)
        if devs:
            log(f"  {d}: {len(devs)} stem pairs, median mean deviation {np.median(devs):.1f} cm, "
                f"{sum(x > pcfg.stem_flag_cm for x in devs)} flagged")
    return rows
