"""Stage 7 + 8: plants across dates, availability, growth curves, growth stage.

Plant identity comes from the plant register (stems pooled over all aligned
dates), so tracking = joining every date's plants by plant_id. A plant can be
present without a visible stem (closed canopy): presence is decided in the
separation step (points at the stem base + enough points reached from it).

Growth stage: a V-stage ESTIMATE from the number of visible leaves, kept
non-decreasing over time (lower leaves die and drop in July, and the scanner
cannot see leaf collars). It is reported as +-1-2 stages; plant height is the
more reliable growth measure.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import pretty_date  # noqa: E402
from .names import cluster, instance, nice  # noqa: E402,F401

GROWTH_TRAITS = [
    ("plant_height_m", "plant height (m)", 1.0),
    ("stem_length_m", "stem length to whorl top (m)", 1.0),
    ("stem_diameter_mm", "stem diameter (mm)", 1.0),
    ("n_leaves", "visible leaves", 1.0),
    ("total_leaf_area_m2", "total leaf area (cm2)", 1e4),
    ("mean_leaf_length_m", "mean leaf length (m)", 1.0),
    ("mean_leaf_angle_to_stem_deg", "mean leaf angle to stem (deg)", 1.0),
]


def _day(d: str) -> datetime:
    return datetime.strptime(d, "%y%m%d")


def _f(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def write_tracking(run_root: Path, dates: list[str], register: list[dict], per_date: dict, log=print) -> list[dict]:
    """per_date[date][plant_id] = row (traits + presence + status). Writes
    tracking/traits_by_date.csv, tracking/availability.csv and the growth folder."""
    trk = Path(run_root) / "tracking"
    grw = Path(run_root) / "growth"
    trk.mkdir(parents=True, exist_ok=True)
    grw.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in register:
        pid = r["plant_id"]
        for d in dates:
            x = per_date.get(d, {}).get(pid)
            rows.append({"plant_id": pid, "group": r["group"], "date": d,
                         "present": bool(x and x.get("presence_present")),
                         "stem_seen": bool(x and x.get("presence_stem_seen")),
                         **({k: v for k, v in x.items() if k not in ("plant_id", "date")} if x else {})})
    # growth stage estimate (non-decreasing visible leaf count)
    for pid in [r["plant_id"] for r in register]:
        best = 0.0
        for row in (rw for rw in rows if rw["plant_id"] == pid):
            n = _f(row.get("n_leaves"))
            if row["present"] and np.isfinite(n):
                best = max(best, n)
            row["v_stage_estimate"] = f"V{int(best)}" if row["present"] and best > 0 else ""
            row["growth_stage_note"] = ("estimate from visible leaves (+-1-2 stages), never decreasing"
                                        if row["v_stage_estimate"] else "")
    # plausibility: plants do not shrink. A drop in height of more than 20 %
    # between dates means the plant was not separated correctly on that date.
    for pid in [r["plant_id"] for r in register]:
        prev = None
        for row in (rw for rw in rows if rw["plant_id"] == pid):
            h = _f(row.get("plant_height_m"))
            row["check"] = ""
            if row["present"] and np.isfinite(h):
                if prev is not None and h < 0.8 * prev:
                    row["check"] = f"height dropped {prev:.2f} -> {h:.2f} m: separation on this date doubtful"
                prev = h if prev is None else max(prev, h)
    keys = []
    for rw in rows:
        for k in rw:
            if k not in keys:
                keys.append(k)
    with open(trk / "traits_by_date.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # availability: one row per plant
    with open(trk / "availability.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plant_id", "group", *[pretty_date(d) for d in dates], "missing_on", "n_dates_measured"])
        for r in register:
            st = []
            for d in dates:
                x = per_date.get(d, {}).get(r["plant_id"])
                if not x or not x.get("presence_present"):
                    vis = (x or {}).get("presence_visibility", "")
                    st.append("not visible (covered by canopy)" if vis == "covered_by_canopy"
                              else ("too small to measure" if vis == "too_small" else "missing"))
                elif x.get("status") != "ok":
                    st.append("present (no structure)")
                else:
                    st.append("present" if x.get("presence_stem_seen") else "present (stem hidden)")
            w.writerow([r["plant_id"], r["group"], *st,
                        " ".join(pretty_date(d) for d, s_ in zip(dates, st) if s_ == "missing"),
                        sum(s_.startswith("present") for s_ in st)])

    # wide tables + growth curves
    groups = sorted({r["group"] for r in register})
    for key, label, scale in GROWTH_TRAITS:
        with open(grw / f"{key}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["plant_id", "group", *dates])
            for r in register:
                vals = [_f(next((rw.get(key) for rw in rows if rw["plant_id"] == r["plant_id"] and rw["date"] == d
                                 and rw["present"]), None)) for d in dates]
                w.writerow([r["plant_id"], r["group"], *["" if not np.isfinite(v) else round(v * scale, 4) for v in vals]])
        nc = 5
        nr = int(np.ceil(len(groups) / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.6 * nr), sharex=True, sharey=True, squeeze=False)
        days = np.array([(_day(d) - _day(dates[0])).days for d in dates])
        vmax = 0.0
        for gi, g in enumerate(groups):
            ax = axs[gi // nc, gi % nc]
            for r in [r_ for r_ in register if r_["group"] == g]:
                v = np.array([_f(next((rw.get(key) for rw in rows if rw["plant_id"] == r["plant_id"] and rw["date"] == d
                                       and rw["present"]), None)) * scale for d in dates])
                ok = np.isfinite(v)
                if ok.any():
                    ax.plot(days[ok], v[ok], "o-", ms=3, lw=1, label=instance(r["plant_id"]))
                    vmax = max(vmax, np.nanmax(v))
            ax.set_title(cluster(g), fontsize=9)
            ax.legend(fontsize=6, loc="upper left")
        for k in range(len(groups), nr * nc):
            axs[k // nc, k % nc].axis("off")
        for ax in axs[-1]:
            ax.set_xticks(days)
            ax.set_xticklabels([pretty_date(d)[:5] for d in dates], fontsize=7, rotation=45)
        fig.suptitle(f"{label} per plant instance (gaps = plant instance missing or not measured on that date)", fontsize=11)
        fig.tight_layout()
        fig.savefig(grw / f"{key}.png", dpi=100)
        plt.close(fig)
    n_present = {d: sum(1 for r in register if per_date.get(d, {}).get(r["plant_id"], {}).get("presence_present")) for d in dates}
    log("  tracking: plants present per date " + ", ".join(f"{d}: {n}" for d, n in n_present.items()))
    return rows
