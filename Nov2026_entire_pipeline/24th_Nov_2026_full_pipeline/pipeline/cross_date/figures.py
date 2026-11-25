"""Figures for checkpoint 1 (clusters)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from .config import DATE_COLOURS, DATE_MARKERS, STATUS_COLOURS, date_label

INK, MUTED, GRID = "#0b0b0b", "#6b6b66", "#e1e0d9"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})


def _parts(clusters, ref, n=3):
    s = [c.mu for cs in clusters.values() for c in cs]
    lo, hi = min(s) - 0.15, max(s) + 0.15
    e = np.linspace(lo, hi, n + 1)
    return list(zip(e[:-1], e[1:]))


def _status_of(rows):
    """{date cluster id: status} and {ref cluster id: {date: status}}."""
    qs, rs = {}, {}
    for r in rows:
        for q in r["date_clusters"].split():
            qs[q] = r["status"] if r["status"] != "reference" else "reference"
        if r["reference_cluster"]:
            rs.setdefault(r["reference_cluster"], {})[r["date"]] = r["status"]
    return qs, rs


def fig_profiles(run, clusters, profiles, ref_names, out: Path):
    parts = _parts(clusters, run.reference)
    nd = len(run.dates)
    fig = plt.figure(figsize=(16, 4.2 * len(parts)))
    gs = fig.add_gridspec(len(parts) * (nd + 1), 1, hspace=0.0, top=0.955, bottom=0.03)
    for k, (a, b) in enumerate(parts):
        for i, d in enumerate(run.dates):
            ax = fig.add_subplot(gs[k * (nd + 1) + i, 0])
            p = profiles[d]
            scale = np.percentile(p.density[p.density > 0], 99)
            m = (p.centres >= a) & (p.centres <= b)
            y = np.sqrt(np.clip(p.density[m] / scale, 0, 1.2))
            ax.fill_between(p.centres[m], 0, y, color=DATE_COLOURS[d], alpha=0.35, lw=0)
            ax.plot(p.centres[m], y, color=DATE_COLOURS[d], lw=0.8)
            ax.axhline(np.sqrt(p.threshold / scale), color=MUTED, lw=0.6, ls=(0, (3, 3)))
            for c in clusters[d]:
                if c.mu < a or c.mu > b:
                    continue
                xs = np.linspace(c.mu - 3 * c.sigma, c.mu + 3 * c.sigma, 80)
                g = np.exp(-0.5 * ((xs - c.mu) / c.sigma) ** 2) * np.sqrt(min(c.peak / scale, 1.2))
                ax.plot(xs, g, color=INK if not c.clod else MUTED, lw=0.9, ls="-" if not c.clod else ":")
                if d == run.reference:
                    ax.text(c.mu, 1.12, ref_names.get(c.cid, c.cid).replace("plant_cluster_", "PC ")
                            .replace("ref_cluster_", "RC "), ha="center", va="bottom", fontsize=7, color=INK)
            if d == run.reference:
                for r in run.register:
                    if a <= r["s"] <= b:
                        ax.plot([r["s"]], [-0.08], marker="|", color="#d03b3b", ms=6, mew=1)
            ax.set_xlim(a, b)
            ax.set_ylim(-0.15, 1.35)
            ax.set_yticks([])
            ax.set_ylabel(date_label(d), rotation=0, ha="right", va="center", color=DATE_COLOURS[d],
                          fontweight="bold")
            ax.grid(axis="x", color=GRID, lw=0.5)
            if i < nd - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(f"along the row (m), part {k + 1} of {len(parts)}")
    handles = [Line2D([], [], color=INK, lw=1, label="Gaussian of a cluster"),
               Line2D([], [], color=MUTED, lw=1, ls=":", label="Gaussian of a clod (never above 5 cm)"),
               Line2D([], [], color=MUTED, lw=0.8, ls=(0, (3, 3)), label="threshold (background + 3 sd)"),
               Line2D([], [], color="#d03b3b", marker="|", ls="", ms=8, label="plant in the register (pipeline)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Stem-base density along the row (3-15 cm above ground, upright columns only after 01.06., "
                 "sqrt scale) and the Gaussian of each cluster", y=1.0, fontsize=11)
    fig.savefig(out / "fig1_density_profiles.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_matches(run, clusters, rows, ref_names, out: Path):
    parts = _parts(clusters, run.reference)
    qstat, _ = _status_of(rows)
    ymap = {d: len(run.dates) - 1 - i for i, d in enumerate(run.dates)}
    ref_mu = {c.cid: c.mu for c in clusters[run.reference]}
    fig, axes = plt.subplots(len(parts), 1, figsize=(16, 3.3 * len(parts)))
    for ax, (a, b) in zip(np.atleast_1d(axes), parts):
        for r in rows:
            if r["date"] == run.reference or not r["reference_cluster"] or not r["date_clusters"]:
                continue
            x0 = ref_mu[r["reference_cluster"]]
            for q in r["date_clusters"].split():
                c = next(c for c in clusters[r["date"]] if c.cid == q)
                if a - 0.1 <= x0 <= b + 0.1:
                    ax.plot([x0, c.mu], [ymap[run.reference], ymap[r["date"]]], color="#9a9a92", lw=0.7, zorder=1)
        for d in run.dates:
            for c in clusters[d]:
                if not (a <= c.mu <= b):
                    continue
                st = "clod" if c.clod else qstat.get(c.cid, "extra")
                y = ymap[d]
                ax.barh(y, 2 * c.sigma, left=c.mu - c.sigma, height=0.36, color=STATUS_COLOURS[st],
                        edgecolor="white", lw=1.5, zorder=2)
                if d == run.reference:
                    ax.text(c.mu, y + 0.3, ref_names.get(c.cid, c.cid).replace("plant_cluster_", "PC ")
                            .replace("ref_cluster_", "RC "), ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_yticks(list(ymap.values()))
        ax.set_yticklabels([date_label(d) + (" (reference)" if d == run.reference else "") for d in ymap])
        for t, d in zip(ax.get_yticklabels(), ymap):
            t.set_color(DATE_COLOURS[d])
        ax.set_xlim(a, b)
        ax.set_ylim(-0.6, len(run.dates) - 0.2)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_xlabel("along the row (m); bar = cluster centre +- 1 sigma")
    order = ["reference", "confirmed", "split", "flagged", "extra", "clod"]
    labels = {"reference": "reference cluster (29.06.)", "confirmed": "confirmed (A and B agree)",
              "split": "split / merged (one-to-several)", "flagged": "flagged (A and B disagree)",
              "extra": "extra / missing (no partner)", "clod": "clod (not matched)"}
    fig.legend(handles=[Patch(color=STATUS_COLOURS[k], label=labels[k]) for k in order], loc="upper center",
               ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Cluster matching to 29.06.: grey lines join each cluster to its reference cluster "
                 "(PC = plant_cluster of the register, RC = cluster with no one-to-one register cluster)",
                 y=1.03, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig2_cluster_matches.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_fine(run, fine, cfg, out: Path):
    dates = [d for d in run.dates if d != run.reference]
    fig, axes = plt.subplots(1, len(dates), figsize=(3.6 * len(dates), 3.8), sharex=True, sharey=True)
    lim = 100 * cfg.fine_search
    for ax, d in zip(np.atleast_1d(axes), dates):
        rs = [r for r in fine if r["date"] == d]
        seen = set()
        acc = [(r["dx_cm"], r["dy_cm"]) for r in rs if r["accepted"] and not (r["date_clusters"] in seen or seen.add(r["date_clusters"]))]
        seen = set()
        rej = [(r["dx_cm"], r["dy_cm"]) for r in rs if not r["accepted"] and not (r["date_clusters"] in seen or seen.add(r["date_clusters"]))]
        ax.add_patch(plt.Circle((0, 0), 100 * cfg.fine_max_accept, fill=False, ls=(0, (3, 3)), color=MUTED, lw=0.8))
        if acc:
            A = np.array(acc)
            ax.scatter(A[:, 0], A[:, 1], s=36, color=DATE_COLOURS[d], marker=DATE_MARKERS[d],
                       edgecolor="white", lw=1, zorder=3)
            ax.text(0.03, 0.97, f"accepted {len(acc)}\nmedian {np.median(np.hypot(A[:, 0], A[:, 1])):.1f} cm",
                    transform=ax.transAxes, va="top", fontsize=8, color=INK)
        if rej:
            Rj = np.array(rej)
            ax.scatter(Rj[:, 0], Rj[:, 1], s=36, facecolor="none", edgecolor=DATE_COLOURS[d],
                       marker=DATE_MARKERS[d], lw=1.2, zorder=3)
        ax.axhline(0, color=GRID, lw=0.6)
        ax.axvline(0, color=GRID, lw=0.6)
        ax.set_xlim(-lim - 0.5, lim + 0.5)
        ax.set_ylim(-lim - 0.5, lim + 0.5)
        ax.set_aspect("equal")
        ax.set_title(date_label(d), color=DATE_COLOURS[d], fontweight="bold")
        ax.set_xlabel("fine dx (cm)")
    np.atleast_1d(axes)[0].set_ylabel("fine dy (cm)")
    fig.legend(handles=[Line2D([], [], marker="o", color=MUTED, ls="", label="accepted"),
                        Line2D([], [], marker="o", mfc="none", color=MUTED, ls="", label="not accepted"),
                        Line2D([], [], color=MUTED, ls=(0, (3, 3)), label=f"acceptance limit {100 * cfg.fine_max_accept:.0f} cm")],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Fine shift per matched cluster on top of the global alignment", y=1.14, fontsize=11)
    fig.savefig(out / "fig3_fine_alignment.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_examples(run, clusters, rows, fine, ref_names, cfg, out: Path):
    """Top view of the stem-base layer for a few reference clusters, one panel per date."""
    want = ["confirmed", "split", "merged", "flagged"]
    picks = []
    for w in want:
        for r in rows:
            if r["status"] == w and r["reference_cluster"] and r["reference_cluster"] not in [p[0] for p in picks]:
                picks.append((r["reference_cluster"], w, r["date"]))
                break
    if not picks:
        return
    fine_map = {(r["reference_cluster"], r["date"]): r for r in fine}
    ref_c = {c.cid: c for c in clusters[run.reference]}
    fig, axes = plt.subplots(len(picks), len(run.dates), figsize=(2.9 * len(run.dates), 2.25 * len(picks)),
                             squeeze=False)
    for i, (rc, why, wd) in enumerate(picks):
        c = ref_c[rc]
        a, b = c.lo - 0.08, c.hi + 0.08
        for j, d in enumerate(run.dates):
            ax = axes[i, j]
            L = run.low[d]
            m = (L.s >= a) & (L.s <= b)
            ax.scatter(100 * (L.s[m] - c.mu), 100 * L.t[m], s=1.5, color=DATE_COLOURS[d], lw=0)
            Lr = run.low[run.reference]
            mr = (Lr.s >= a) & (Lr.s <= b)
            if d != run.reference:
                ax.scatter(100 * (Lr.s[mr] - c.mu), 100 * Lr.t[mr], s=0.6, color="#9a9a92", alpha=0.35, lw=0, zorder=0)
            for q in clusters[d]:
                if q.hi >= a and q.lo <= b:
                    ax.axvspan(100 * (q.lo - c.mu), 100 * (q.hi - c.mu), color=GRID, alpha=0.5, lw=0, zorder=-1)
            st = next((r["status"] for r in rows if r["reference_cluster"] == rc and r["date"] == d), "")
            fr = fine_map.get((rc, d))
            extra = f"\nfine {fr['shift_cm']:.1f} cm{'' if fr['accepted'] else ' (not used)'}" if fr else ""
            ax.set_title(f"{date_label(d)}: {st}{extra}", fontsize=8, color=INK)
            ax.set_xlim(100 * (a - c.mu), 100 * (b - c.mu))
            ax.set_ylim(-12, 12)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(f"{ref_names[rc]}\n(example: {why})\nacross row (cm)", fontsize=8)
            if i == len(picks) - 1:
                ax.set_xlabel("along row from centre (cm)", fontsize=8)
    fig.suptitle("Examples, top view of the stem-base layer (3-15 cm, upright columns only after 01.06.): "
                 "coloured = that date, grey = 29.06., light bands = clusters found on that date", fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "fig4_examples_top_view.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def cluster_figures(run, clusters, profiles, rows, fine, ref_names, cfg, out: Path):
    fig_profiles(run, clusters, profiles, ref_names, out)
    fig_matches(run, clusters, rows, ref_names, out)
    fig_fine(run, fine, cfg, out)
    fig_examples(run, clusters, rows, fine, ref_names, cfg, out)
