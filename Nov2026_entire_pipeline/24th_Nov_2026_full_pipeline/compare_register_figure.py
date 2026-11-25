"""Figure: register of two runs around the places where it changed - every
date's stems (aligned) and the register plants of run A and run B.

    ../../venv/bin/python compare_register_figure.py outputs/run_A outputs/run_B
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline.config import PROJECT_ORIGIN  # noqa: E402
from pipeline.cross_date.config import DATE_COLOURS, DATE_MARKERS, date_label  # noqa: E402
from pipeline.names import nice  # noqa: E402

O = np.asarray(PROJECT_ORIGIN)


def reg(run):
    return list(csv.DictReader(open(run / "tracking" / "plant_register.csv")))


def stems(run, dates):
    out = {}
    for d in dates:
        f = run / d / "02_stems_groups" / "stems.csv"
        rs = list(csv.DictReader(open(f))) if f.exists() else []
        out[d] = np.array([[float(r["x_aligned"]) - O[0], float(r["y_aligned"]) - O[1]] for r in rs
                           if r.get("off_row_m") in (None, "") or float(r["off_row_m"]) <= 0.10]).reshape(-1, 2)
    return out


def short(p):
    return nice(p).replace("plant_cluster_", "PC").replace("_plant_instance_", "/")


def main():
    A, B = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    ra, rb = reg(A), reg(B)
    dates = sorted({d for r in ra + rb for d in r["dates_seen"].split()})
    xa = np.array([[float(r["x_utm"]), float(r["y_utm"])] for r in ra]) - O[:2]
    xb = np.array([[float(r["x_utm"]), float(r["y_utm"])] for r in rb]) - O[:2]
    # places where the register changed: around a register plant, the two runs have a different number
    # of register plants within 3 cm (a plant missing, added, merged or split)
    ta, tb = cKDTree(xa), cKDTree(xb)
    spots = [p for p in np.vstack([xa, xb])
             if len(ta.query_ball_point(p, 0.03)) != len(tb.query_ball_point(p, 0.03))]
    centres = []
    for s in spots:
        if all(np.hypot(*(s - c)) > 0.15 for c in centres):
            centres.append(s)
    if not centres:
        print("registers are the same")
        return
    sa, sb = stems(A, dates), stems(B, dates)
    print(f"{len(centres)} places where the registers differ")
    fig, axes = plt.subplots(len(centres), 2, figsize=(12, 5.6 * len(centres)), squeeze=False)
    for k, c in enumerate(centres):
        for j, (run, st_, xr, rr, title) in enumerate(((A, sa, xa, ra, "A (" + A.name.split("_")[-1] + ")"),
                                                       (B, sb, xb, rb, "B (" + B.name.split("_")[-1] + ")"))):
            ax = axes[k, j]
            for d in dates:
                p = st_.get(d, np.zeros((0, 2)))
                m = np.hypot(*(p - c).T) < 0.14 if len(p) else np.zeros(0, bool)
                ax.scatter(100 * (p[m, 0] - c[0]), 100 * (p[m, 1] - c[1]), s=60, marker=DATE_MARKERS[d],
                           color=DATE_COLOURS[d], edgecolor="white", lw=0.8, zorder=3)
            m = np.hypot(*(xr - c).T) < 0.14
            for i in np.where(m)[0]:
                x, y = 100 * (xr[i] - c)
                ax.add_patch(plt.Circle((x, y), 3.0, fill=False, ls=(0, (3, 2)), color="#6b6b66", lw=0.8))
                ax.scatter([x], [y], marker="+", s=160, color="#0b0b0b", lw=1.6, zorder=4)
                ax.text(x + 0.8, y + 0.8, f"{short(rr[i]['plant_id'])}\n{len(rr[i]['dates_seen'].split())} dates",
                        fontsize=8, color="#0b0b0b", zorder=5)
            ax.set_xlim(-14, 14)
            ax.set_ylim(-14, 14)
            ax.set_aspect("equal")
            ax.grid(color="#e1e0d9", lw=0.5)
            ax.set_xlabel("x (cm)")
            ax.set_ylabel("y (cm)")
            ax.set_title(f"{title}: {int(m.sum())} register plants here", fontsize=10)
    fig.legend(handles=[Line2D([], [], marker=DATE_MARKERS[d], ls="", color=DATE_COLOURS[d], label=f"stem {date_label(d)}")
                        for d in dates] +
               [Line2D([], [], marker="+", ls="", color="#0b0b0b", markersize=10, label="register plant (mean of its stems)"),
                Line2D([], [], ls=(0, (3, 2)), color="#6b6b66", label="3 cm pooling distance")],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Plant register where the two runs differ: stems of every date (aligned) and register plants",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    out = B / f"register_changes_vs_{A.name}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
