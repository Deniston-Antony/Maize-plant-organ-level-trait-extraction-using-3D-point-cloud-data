"""Figure of the identity check (stage 6b): every 'one plant, two register names'
case - the reference plant and the plant of the other date, overlaid.

Per case (one row): top view, side view (along the row vs height) and the
containment test: the earlier plant's lower-stem points, green where the later
plant has a point within 2 cm, red where not.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial import cKDTree

from pipeline import names

from .config import DATE_COLOURS, PlantConfig, date_label
from .plant_check import _low_region

INK, MUTED, GRID = "#0b0b0b", "#6b6b66", "#e1e0d9"


def _short(n: str) -> str:
    return names.nice(n).replace("plant_cluster_", "PC").replace("_plant_instance_", "/")


def _plants(run_dir: Path, out: Path, dates: set) -> dict:
    from pipeline.align import load_alignments, Alignment
    from .plants import load_plants
    import json
    ref = json.loads((out / "01_clusters" / "summary.json").read_text())["reference_date"]
    als = load_alignments(run_dir / "alignment" / "alignment.json")
    als.setdefault(ref, Alignment(ref, ref, [0.0, 0.0], [0.0, 1.0]))
    sh = {(r["date"], r["plant_id"]): r for r in csv.DictReader(open(out / "02_plants_stems" / "plant_shifts.csv"))}
    ps = {}
    for d in dates | {ref}:
        ps[d] = load_plants(run_dir, d, als[d], log=lambda *a: None)
        for pid, p in ps[d].items():
            r = sh.get((d, pid))
            if r:
                p.shift(float(r["fine_dx_m"]), float(r["fine_dy_m"]))
    return ps, ref


def identity_figure(run_dir: Path, path: Path | None = None) -> Path | None:
    run_dir = Path(run_dir)
    out = run_dir / "cross_date_matching"
    table = run_dir / "tracking" / "identity_check.csv"
    cases = list(csv.DictReader(open(table))) if table.exists() and table.stat().st_size else []
    if not cases:
        return None
    ps, ref = _plants(run_dir, out, {c["date"] for c in cases})
    pcfg = PlantConfig()
    n = len(cases)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.6 * n), squeeze=False)
    for k, c in enumerate(cases):
        d = c["date"]
        P = ps[ref].get(names.raw(c["reference_name"]))
        Q = ps[d].get(names.raw(c["name_on_date"]))
        ax_t, ax_s, ax_c = axes[k]
        if P is None or Q is None:
            ax_t.text(0.5, 0.5, "plant files not found", transform=ax_t.transAxes, ha="center")
            continue
        o = P.base[:2]
        ss = [(P, DATE_COLOURS[ref], f"{_short(P.pid)} on {date_label(ref)} (reference name)"),
              (Q, DATE_COLOURS[d], f"{_short(Q.pid)} on {date_label(d)} (name that day)")]
        # neighbours of the date plant (grey) for context
        for pid, R in ps[d].items():
            if pid != Q.pid and np.hypot(*(R.base[:2] - o)) < 0.25:
                pts = R.xyz[::6]
                ax_t.scatter(100 * (pts[:, 0] - o[0]), 100 * (pts[:, 1] - o[1]), s=0.4, color="#d8d7d0", lw=0)
                ax_t.text(100 * (R.base[0] - o[0]), 100 * (R.base[1] - o[1]), _short(pid), fontsize=7, color=MUTED)
        for p, col, lab in ss:
            pts = p.xyz[::3]
            ax_t.scatter(100 * (pts[:, 0] - o[0]), 100 * (pts[:, 1] - o[1]), s=0.6, color=col, lw=0, alpha=0.55)
            ax_s.scatter(100 * (pts[:, 0] - o[0]), 100 * p.hag[::3], s=0.6, color=col, lw=0, alpha=0.55)
        for p, col, lab in ss:
            cl = p.centerline
            ax_t.plot(100 * (cl[:, 0] - o[0]), 100 * (cl[:, 1] - o[1]), color=INK, lw=1.2)
            ax_t.scatter([100 * (p.base[0] - o[0])], [100 * (p.base[1] - o[1])], s=90, marker="X", color=col,
                         edgecolor=INK, lw=1, zorder=5)
            ax_s.plot(100 * (cl[:, 0] - o[0]), 100 * p.centerline_hag, color=col, lw=2.2)
        bd = float(np.hypot(*(P.base[:2] - Q.base[:2])))
        ax_t.annotate("", xy=(100 * (Q.base[0] - o[0]), 100 * (Q.base[1] - o[1])), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
        ax_t.set_title(f"{date_label(d)}: top view (lowest stem points {100 * bd:.1f} cm apart)", fontsize=9.5)
        ax_t.set_aspect("equal")
        ax_t.set_xlim(-30, 30)
        ax_t.set_ylim(-30, 30)
        ax_t.set_xlabel("x (cm)")
        ax_t.set_ylabel("y (cm)")
        ax_s.set_title("side view: points and stem lines", fontsize=9.5)
        ax_s.set_xlim(-30, 30)
        ax_s.set_xlabel("x (cm)")
        ax_s.set_ylabel("height above ground (cm)")
        # containment test
        early, late = (P, Q) if P.date < Q.date else (Q, P)
        low = _low_region(early, pcfg)
        if len(low):
            dd, _ = cKDTree(late.xyz).query(low, distance_upper_bound=pcfg.contain_tol)
            inside = np.isfinite(dd)
            lp = late.xyz[::3]
            ax_c.scatter(100 * (lp[:, 0] - o[0]), 100 * late.hag[::3], s=0.4, color="#d8d7d0", lw=0)
            hz = low[:, 2] - early.xyz[:, 2].min() + early.hag.min()
            ax_c.scatter(100 * (low[inside, 0] - o[0]), 100 * hz[inside], s=5, color="#0ca30c", lw=0)
            ax_c.scatter(100 * (low[~inside, 0] - o[0]), 100 * hz[~inside], s=5, color="#d03b3b", lw=0)
            ax_c.set_title(f"containment: {inside.mean():.0%} of the earlier plant's lower stem ({date_label(early.date)}) "
                           f"lies inside the later plant ({date_label(late.date)})", fontsize=9)
        ax_c.set_xlim(-30, 30)
        ax_c.set_ylim(0, 40)
        ax_c.set_xlabel("x (cm)")
        ax_c.set_ylabel("height above ground (cm)")
        verdict = "MERGED for tracking" if c.get("merged") == "True" else "not merged: " + c.get("reason", "")
        if c.get("stem_deviation_cm") not in (None, "", "nan"):
            verdict += f"\nstem lines {float(c['stem_deviation_cm']):.1f} cm apart at the same heights"
        ax_t.text(-29, 27, f"{_short(Q.pid)} ({date_label(d)}) = {_short(P.pid)} (29.06.)?\n{verdict}",
                  fontsize=8.5, va="top", bbox=dict(fc="white", ec=GRID))
        for ax in axes[k]:
            ax.grid(color=GRID, lw=0.5)
        ax_t.legend(handles=[Line2D([], [], marker="o", ls="", color=col, label=lab) for _, col, lab in ss] +
                    [Line2D([], [], marker="X", ls="", color=MUTED, label="stem base (structure code)")],
                    loc="lower left", fontsize=7.5, frameon=True)
    from ..config import IDENTITY_MERGE_MAX_STEM_DEV_CM, IDENTITY_MERGE_MIN_CONTAINMENT
    fig.suptitle("Identity check (stage 6b): one plant with two register names? Merge needs containment >= "
                 f"{IDENTITY_MERGE_MIN_CONTAINMENT} and stem lines <= {IDENTITY_MERGE_MAX_STEM_DEV_CM:.0f} cm apart "
                 "at the same heights", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = Path(path or run_dir / "tracking" / "identity_check.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
