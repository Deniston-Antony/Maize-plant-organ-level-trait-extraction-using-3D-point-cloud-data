"""Figures for steps 5-6 (plants, stems)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from pipeline import names

from .config import DATE_COLOURS, DATE_MARKERS, date_label
from .figures import INK, MUTED, GRID

STATUS = {   # colour, short text in the cell
    "confirmed": ("#0ca30c", "ok"), "ambiguous": ("#fab219", "amb."), "other name": ("#cdbcf2", "name"),
    "swapped": ("#ec835a", "swap"), "missing": ("#e1e0d9", "–"),
}
LABELS = {"confirmed": "confirmed (same name, geometry agrees)", "ambiguous": "ambiguous (two candidates)",
          "other name": "other name (one plant, two register entries?)", "swapped": "swapped (geometry prefers another name)",
          "missing": "missing (no separated plant there)"}


def fig_plant_check(run, plants, prow, out: Path):
    ref = run.reference
    dates = [d for d in run.dates if d != ref and plants.get(d)]
    R = sorted(plants[ref].values(), key=lambda p: run.frame.st(p.base[None, :2])[0, 0])
    stat = {(r["reference_plant"], r["date"]): r for r in prow if r["reference_plant"]}
    fig, ax = plt.subplots(figsize=(2.2 + 1.3 * len(dates), 0.28 * len(R) + 1.6))
    for i, p in enumerate(R):
        nm = names.nice(p.pid)
        for j, d in enumerate(dates):
            r = stat.get((nm, d))
            st = r["status"] if r else "missing"
            col, txt = STATUS.get(st, ("#e1e0d9", st))
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, color=col, lw=0))
            if st in ("swapped", "other name") and r:
                txt = r["matched_plant"].replace("plant_cluster_", "PC").replace("_plant_instance_", "/")
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=7,
                    color="#0b0b0b" if st in ("ambiguous", "other name", "missing") else "white")
    ax.set_xlim(0, len(dates))
    ax.set_ylim(len(R), 0)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(dates)) + 0.5)
    ax.set_xticklabels([])
    for j, d in enumerate(dates):
        ax.text(j + 0.5, -0.4, date_label(d), ha="center", va="bottom", color=DATE_COLOURS[d],
                fontweight="bold", fontsize=10)
    ax.set_yticks(np.arange(len(R)) + 0.5)
    ax.set_yticklabels([names.nice(p.pid).replace("plant_cluster_", "PC ").replace("_plant_instance_", " / ") for p in R],
                       fontsize=7.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    nmiss = {d: sum(1 for r in prow if r["date"] == d and r["status"] == "not on the reference date") for d in dates}
    ax.set_xlabel("not matched to a 29.06. plant: " + ", ".join(f"{date_label(d)} {n}" for d, n in nmiss.items()),
                  fontsize=8, color=MUTED)
    fig.legend(handles=[Patch(color=STATUS[k][0], label=LABELS[k]) for k in STATUS], loc="lower center",
               ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Plant check: each 29.06. plant (row, south to north) on every other date\n"
                 "(PC = plant_cluster, swap / name cells show the plant geometry prefers)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig5_plant_check.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_stems(run, srow, pcfg_flag: float, out: Path):
    ref = run.reference
    dates = [d for d in run.dates if d != ref and any(r["date"] == d for r in srow)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    rng = np.random.default_rng(0)
    for k, (key, lab) in enumerate([("mean_deviation_cm", "mean stem deviation (cm)"),
                                     ("lean_diff_deg", "lean difference (deg)")]):
        ax = axes[k]
        for j, d in enumerate(dates):
            v = np.array([float(r[key]) for r in srow if r["date"] == d and r.get(key) not in (None, "")])
            if not len(v):
                continue
            x = j + rng.uniform(-0.18, 0.18, len(v))
            ax.scatter(x, v, s=28, color=DATE_COLOURS[d], marker=DATE_MARKERS[d], edgecolor="white", lw=0.8, zorder=3)
            ax.plot([j - 0.3, j + 0.3], [np.median(v)] * 2, color=INK, lw=1.5, zorder=4)
            ax.text(j + 0.32, np.median(v), f"median {np.median(v):.1f}", va="center", fontsize=7.5, color=INK)
        if key == "mean_deviation_cm":
            ax.axhline(pcfg_flag, color=MUTED, ls=(0, (3, 3)), lw=0.8)
            ax.text(len(dates) - 0.5, pcfg_flag, f" flag above {pcfg_flag:.0f} cm", va="bottom", ha="right",
                    fontsize=7.5, color=MUTED)
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels([date_label(d) for d in dates])
        ax.set_xlim(-0.6, len(dates) - 0.2)
        ax.set_ylabel(lab)
        ax.grid(axis="y", color=GRID, lw=0.5)
    fig.suptitle("Stem matching: each assigned plant's stem centre line against its 29.06. stem "
                 "(common height range, every 1 cm)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig6_stem_match.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plant_stem_figures(run, plants, prow, srow, out: Path, flag_cm: float = 3.0):
    fig_plant_check(run, plants, prow, out)
    fig_stems(run, srow, flag_cm, out)
