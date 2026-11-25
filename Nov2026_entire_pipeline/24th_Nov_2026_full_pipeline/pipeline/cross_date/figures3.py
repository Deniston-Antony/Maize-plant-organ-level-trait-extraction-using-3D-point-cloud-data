"""Figures for steps 7-8 (leaves)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from pipeline import names

from .config import DATE_COLOURS, date_label
from .figures import INK, MUTED, GRID

LEAF_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7b5cd6", "#3fb8c9", "#a8641f",
                "#5a8f29", "#c2410c", "#6b7280", "#b8336a"]


def _colour(leaf_id: str) -> str:
    if not leaf_id.startswith("L"):
        return "#c3c2b7"
    return LEAF_COLOURS[(int(leaf_id[1:]) - 1) % len(LEAF_COLOURS)]


def fig_rule(rule_json: Path, out: Path):
    d = json.loads(rule_json.read_text())
    res = d.get("case_results")
    if not res:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.5 * len(res) + 1.6), sharey=True)
    y = np.arange(len(res))
    for r, yy in zip(res, y):
        axes[0].plot([r["before"]["n_pred"], r["after"]["n_pred"]], [yy, yy], color=GRID, lw=2, zorder=1)
        axes[1].plot([r["before"]["mean_iou"], r["after"]["mean_iou"]], [yy, yy], color=GRID, lw=2, zorder=1)
    axes[0].scatter([r["n_true"] for r in res], y, marker="|", s=220, color=INK, lw=2, zorder=3, label="true leaves")
    axes[0].scatter([r["before"]["n_pred"] for r in res], y, s=40, facecolor="white", edgecolor=MUTED, lw=1.5,
                    zorder=3, label="structure code as is")
    axes[0].scatter([r["after"]["n_pred"] for r in res], y, s=40, color="#2a78d6", zorder=4, label="after the leaf rule")
    axes[1].scatter([r["before"]["mean_iou"] for r in res], y, s=40, facecolor="white", edgecolor=MUTED, lw=1.5, zorder=3)
    axes[1].scatter([r["after"]["mean_iou"] for r in res], y, s=40, color="#2a78d6", zorder=4)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r["case"] for r in res], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("number of leaves")
    axes[1].set_xlabel("mean leaf IoU with the ground truth")
    axes[1].set_xlim(0, 1)
    for ax in axes:
        ax.grid(axis="x", color=GRID, lw=0.5)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    r = d["rule"]
    fig.suptitle(f"Step 7 on the 21.06. ground truth: leaf rule = length >= {100 * r['min_length']:.0f} cm or area >= "
                 f"{1e4 * r['min_area']:.0f} cm2, within {100 * (r['max_stem_gap'] or 0):.0f} cm of its own stem; "
                 f"pieces joined within {100 * r['join_dist']:.0f} cm" + (f" and {r['max_angle']:.0f} deg" if r['max_angle'] else ""),
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / "fig7_leaf_rule.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plant_dates(lrow, rp_nice):
    m = {}
    for r in lrow:
        if r["plant"] == rp_nice and r.get("date_plant"):
            m[r["date"]] = names.raw(r["date_plant"])
    return m


def plant_panel(run, plants, leaves, lrow, rp: str, path: Path):
    nice = names.nice(rp)
    m = _plant_dates(lrow, nice)
    dates = [d for d in run.dates if d in m and leaves.get(d, {}).get(m[d])]
    if not dates:
        return False
    base = plants[run.reference][rp].base
    fig, axes = plt.subplots(2, len(dates), figsize=(3.2 * len(dates), 6.2), squeeze=False)
    for j, d in enumerate(dates):
        p = plants[d][m[d]]
        ls = leaves[d][m[d]]
        for row, ax in enumerate(axes[:, j]):
            pts = p.xyz[::4]
            if row == 0:
                ax.scatter(100 * (pts[:, 0] - base[0]), 100 * (pts[:, 1] - base[1]), s=0.3, color="#d8d7d0", lw=0)
            else:
                ax.scatter(100 * (pts[:, 0] - base[0]), 100 * p.hag[::4], s=0.3, color="#d8d7d0", lw=0)
            for l in ls:
                c = _colour(l.leaf_id) if l.tracked else "#9a9a92"
                lw = 1.8 if l.tracked else 0.8
                if row == 0:
                    xy = 100 * (l.path[:, :2] - base[:2])
                    ax.plot(xy[:, 0], xy[:, 1], color=c, lw=lw)
                    if l.tracked:
                        ax.text(xy[-1, 0], xy[-1, 1], l.leaf_id, fontsize=7, color=INK,
                                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=c, lw=0.8))
                else:
                    off = l.path[:, 2] - p.centerline[0, 2] + p.centerline_hag[0]
                    ax.plot(100 * (l.path[:, 0] - base[0]), 100 * off, color=c, lw=lw)
                    if l.tracked:
                        ax.text(100 * (l.path[-1, 0] - base[0]), 100 * off[-1], l.leaf_id, fontsize=7, color=INK,
                                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=c, lw=0.8))
            if row == 0:
                ax.plot(100 * (p.centerline[:, 0] - base[0]), 100 * (p.centerline[:, 1] - base[1]), color=INK, lw=1.2)
                ax.set_aspect("equal")
                ax.set_title(f"{date_label(d)}" + ("" if m[d] == rp else f"\n(as {names.nice(m[d])})"),
                             color=DATE_COLOURS[d], fontweight="bold", fontsize=9)
                ax.set_xlabel("x (cm)", fontsize=8)
                if j == 0:
                    ax.set_ylabel("top view: y (cm)", fontsize=8)
            else:
                ax.plot(100 * (p.centerline[:, 0] - base[0]), 100 * p.centerline_hag, color=INK, lw=1.2)
                ax.set_xlabel("x (cm)", fontsize=8)
                if j == 0:
                    ax.set_ylabel("side view: height above ground (cm)", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"{nice}: leaves coloured by leaf ID (IDs from 29.06.; grey = small, not tracked; black = stem)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_growth(run, tracks, example: list, out: Path):
    fig, axes = plt.subplots(1, len(example), figsize=(4.2 * len(example), 3.6), squeeze=False)
    order = {d: i for i, d in enumerate(run.dates)}
    for ax, rp in zip(axes[0], example):
        nice = names.nice(rp)
        ids = sorted({r["leaf_id"] for r in tracks if r["plant"] == nice and str(r["leaf_id"]).startswith("L")})
        for lid in ids:
            pts = sorted((order[r["date"]], 100 * float(r["length_m"])) for r in tracks
                         if r["plant"] == nice and r["leaf_id"] == lid)
            if len(pts) < 2:
                continue
            x, y = zip(*pts)
            ax.plot(x, y, color=_colour(lid), lw=1.6, marker="o", ms=4)
            ax.text(x[-1] + 0.08, y[-1], lid, fontsize=7, va="center", color=INK)
        ax.set_xticks(range(len(run.dates)))
        ax.set_xticklabels([date_label(d) for d in run.dates], fontsize=7.5)
        ax.set_title(nice.replace("plant_cluster_", "PC ").replace("_plant_instance_", " / "), fontsize=9)
        ax.set_ylabel("leaf length (cm)")
        ax.grid(axis="y", color=GRID, lw=0.5)
    fig.suptitle("Length of single leaves across dates (leaves matched on at least 2 dates)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig9_leaf_growth.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def leaf_figures(run, plants, leaves, lrow, tracks, out: Path):
    fig_rule(out / "leaf_rule.json", out)
    pdir = out / "plants"
    pdir.mkdir(exist_ok=True)
    score = {}
    for r in lrow:
        if r["status"].startswith("matched"):
            score[r["plant"]] = score.get(r["plant"], 0) + 1
    done = []
    for rp in sorted(plants.get(run.reference, {})):
        if plant_panel(run, plants, leaves, lrow, rp, pdir / f"{names.nice(rp)}.png"):
            done.append(rp)
    example = [names.raw(p) for p, _ in sorted(score.items(), key=lambda kv: -kv[1])[:3]]
    for k, rp in enumerate(example):
        src = pdir / f"{names.nice(rp)}.png"
        if src.exists():
            (out / f"fig8_leaf_ids_example{k + 1}.png").write_bytes(src.read_bytes())
    if example:
        fig_growth(run, tracks, example, out)
