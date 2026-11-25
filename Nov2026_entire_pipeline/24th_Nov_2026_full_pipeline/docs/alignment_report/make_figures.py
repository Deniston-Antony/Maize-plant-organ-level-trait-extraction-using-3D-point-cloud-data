"""Figures for the report "Aligning dates and keeping plant identity" (Nov2026 pipeline).

Reads one finished run (read only) and writes PNG figures + facts.json next to this file.
Run from the project root with the project venv:

    venv/bin/python Nov2026_entire_pipeline/docs/alignment_report/make_figures.py [run_folder]
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]                                   # Nov2026_entire_pipeline
sys.path.insert(0, str(CODE))
from pipeline.align import AlignConfig, Alignment, Grid, base_map, relief_map   # noqa: E402
from pipeline.config import PROJECT_ORIGIN                                      # noqa: E402
from pipeline.names import cluster, instance, nice                              # noqa: E402
from pipeline.stems import stem_vote_shift                                      # noqa: E402

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else CODE / "outputs" / "run_20261114_1513_230601_230621_230629_230705_230711"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
O = np.asarray(PROJECT_ORIGIN, float)
DATES = ["230601", "230621", "230629", "230705", "230711"]
LBL = {d: f"{d[4:]}.{d[2:4]}." for d in DATES}
REF = "230629"
# categorical slots 1-5 in fixed order (validated: dataviz validate_palette.js, light mode)
COL = dict(zip(DATES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MRK = dict(zip(DATES, ["o", "s", "^", "D", "v"]))
INK, MUTED, LINE, ACCENT = "#1b2430", "#5d6b7a", "#c9d1da", "#2d5bd7"
STEM_C, LEAF_C = "#8a4b26", "#4f9a3a"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": LINE, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlesize": 10, "axes.titleweight": "semibold",
                     "axes.titlecolor": INK, "legend.frameon": False, "savefig.dpi": 170,
                     "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
facts: dict = {}


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("  wrote", name)


# ------------------------------------------------------------------ data
AJ = json.loads((RUN / "alignment" / "alignment.json").read_text())
AL = {d: Alignment.from_dict(v) for d, v in AJ["dates"].items()}
CFG = json.loads((RUN / "run_config.json").read_text())
o_row = np.asarray(AL[REF].origin_xy)
along = np.asarray(AL[REF].along)
right = np.array([along[1], -along[0]])


def st(xy):                                   # local xy -> (along-row s, across-row t)
    xy = np.atleast_2d(xy)
    return (xy - o_row) @ along, (xy - o_row) @ right


def apply_xy(d, xy):
    xy = np.atleast_2d(np.asarray(xy, float))
    return AL[d].apply(np.column_stack([xy, np.zeros(len(xy))]))[:, :2]


def to_own(d, xy_ref):                        # reference frame -> this date's frame (as runner.stage5_6)
    xy_ref = np.atleast_2d(xy_ref)
    back = apply_xy(d, xy_ref) - xy_ref
    return xy_ref - back


def read_csv(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


STEMS = {d: read_csv(RUN / d / "02_stems_groups" / "stems.csv") for d in DATES}
REG = read_csv(RUN / "tracking" / "plant_register.csv")
REG_XY = np.array([[float(r["x_utm"]) - O[0], float(r["y_utm"]) - O[1]] for r in REG])
AVAIL = read_csv(RUN / "tracking" / "availability.csv")


def stems_xy(d, aligned):
    k = ("x_aligned", "y_aligned") if aligned else ("x_utm", "y_utm")
    return np.array([[float(r[k[0]]) - O[0], float(r[k[1]]) - O[1]] for r in STEMS[d]]).reshape(-1, 2)


_plants: dict = {}


def plants(d):
    if d not in _plants:
        z = np.load(RUN / d / "01_soil" / "plants.npz")
        _plants[d] = {"xyz": z["xyz"].astype(np.float64), "hag": z["hag"].astype(np.float64)}
    return _plants[d]


def ground(d):
    z = np.load(RUN / d / "01_soil" / "ground_1cm.npz")
    return {k: z[k] for k in z.files}


def base_points(d, n=120_000, lo=0.03, hi=0.15, seed=0):
    p = plants(d)
    m = np.flatnonzero((p["hag"] >= lo) & (p["hag"] <= hi))
    if len(m) > n:
        m = np.random.default_rng(seed).choice(m, n, replace=False)
    return p["xyz"][m, :2]


def window(xy, s0, s1, pad=0.0):
    s, t = st(xy)
    return (s >= s0 - pad) & (s <= s1 + pad)


# facts used in the text
facts["alignment"] = {d: {"dx_cm": round(float(np.median(np.asarray(a.dx_profile)[:, 1])) * 100, 1),
                          "dy_cm": round(float(np.median(np.asarray(a.dx_profile)[:, 2])) * 100, 1),
                          "dz_cm": round(a.dz * 100, 1), "method": a.method, "confidence": a.confidence,
                          "notes": a.notes,
                          "sections_total": len(a.sections),
                          "sections_used": sum(1 for s_ in a.sections if s_["used"])} for d, a in AL.items()}
facts["alignment_summary_csv"] = read_csv(RUN / "alignment" / "alignment_summary.csv")
facts["stems"] = {d: json.loads((RUN / d / "02_stems_groups" / "stems_summary.json").read_text()) for d in DATES}
facts["register"] = {"n_plants": len(REG), "n_groups": len({r["group"] for r in REG}),
                     "seen": {k: sum(1 for r in REG if r["n_dates_seen"] == k) for k in "2345"},
                     "group_sizes": {}}
gs = {}
for r in REG:
    gs[r["group"]] = gs.get(r["group"], 0) + 1
for v in gs.values():
    facts["register"]["group_sizes"][v] = facts["register"]["group_sizes"].get(v, 0) + 1
facts["fadly"] = {}
for d in DATES:
    s_ = json.loads((RUN / d / "03_plants" / "instances_fadly" / "summary.json").read_text())
    rows = read_csv(RUN / d / "03_plants" / "instances_fadly" / "stem_axes_auto_review.csv")
    dd = [float(r["match_dist_m"]) for r in rows if r["match_dist_m"]]
    facts["fadly"][d] = {k: s_[k] for k in ("candidate_axes", "register_plants_with_axis", "axes_not_in_register",
                                             "plants_present", "assigned_point_share", "stem_hints", "model_note")}
    facts["fadly"][d]["match_median_cm"] = round(float(np.median(dd)) * 100, 1) if dd else None
    facts["fadly"][d]["match_max_cm"] = round(float(np.max(dd)) * 100, 1) if dd else None
# independent check of alignment + pooling: each date's named stems vs their register position (reference coords)
reg_by_name = {r["plant_id"]: REG_XY[i] for i, r in enumerate(REG)}
facts["stem_to_register_cm"] = {}
for d in DATES:
    A = stems_xy(d, True)
    v = [np.linalg.norm(A[j] - reg_by_name[r["plant_id"]]) * 100 for j, r in enumerate(STEMS[d]) if r["plant_id"] in reg_by_name]
    facts["stem_to_register_cm"][d] = {"n": len(v), "median": round(float(np.median(v)), 2) if v else None,
                                       "p90": round(float(np.percentile(v, 90)), 2) if v else None,
                                       "max": round(float(np.max(v)), 2) if v else None}
date_cols = [c for c in AVAIL[0] if re.match(r"\d\d\.\d\d\.\d{4}", c)]
facts["availability"] = {c: {} for c in date_cols}
for r in AVAIL:
    for c in date_cols:
        facts["availability"][c][r[c]] = facts["availability"][c].get(r[c], 0) + 1
facts["config"] = {k: CFG[k] for k in ("alignment", "stems", "instances", "reference_date", "instance_method")}
ALONG = "along the row (m, 0 = middle of the row)"


def best_window(d="230705", width=1.6):
    """Zoom window (along-row) with the most Fadly candidate axes on date d."""
    fs = json.loads((RUN / d / "03_plants" / "instances_fadly" / "summary.json").read_text())
    org = np.asarray(fs["local_coordinate_origin_xyz_m"])[:2] - O[:2]
    rows = read_csv(RUN / d / "03_plants" / "instances_fadly" / "stem_axes_auto_review.csv")
    s, _ = st(np.array([[float(r["base_x_m"]), float(r["base_y_m"])] for r in rows]) + org)
    starts = np.arange(-4.0, 4.0 - width, 0.05)
    n = [((s >= a) & (s <= a + width)).sum() for a in starts]
    a = float(starts[int(np.argmax(n))]) - 0.05
    return round(a, 2), round(a + width + 0.1, 2)


WIN = best_window()
facts["zoom_window_m"] = WIN


# ------------------------------------------------------------------ fig 1: flow and frames
def fig_flow():
    fig, ax = plt.subplots(figsize=(10, 6.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((1, 2), 40, 92, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#f6f8fa", ec=LINE))
    ax.add_patch(FancyBboxPatch((59, 2), 40, 92, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#eef2fc", ec=ACCENT, lw=1))
    ax.text(21, 97, "Own coordinates of each date", ha="center", fontsize=11, weight="semibold", color=INK)
    ax.text(79, 97, "Reference coordinates (29.06.)", ha="center", fontsize=11, weight="semibold", color=ACCENT)

    def box(x, y, w, h, title, sub, hl=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc="#e3eafb" if hl else "white", ec=ACCENT if hl else "#9aa7b4", lw=1.3 if hl else 1))
        ax.text(x + 1.5, y + h - 3.2, title, fontsize=9.5, weight="semibold", color=INK, va="top")
        ax.text(x + 1.5, y + h - 7.6, sub, fontsize=8, color=MUTED, va="top", linespacing=1.35)

    def arrow(p, q, text="", color=MUTED, ls="-", rad=0.0, tx=None):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=1.4, ls=ls,
                                     connectionstyle=f"arc3,rad={rad}"))
        if text:
            x, y = tx if tx else ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            ax.text(x, y, text, fontsize=8, color=color, ha="center", va="center",
                    bbox=dict(fc="white", ec="none", pad=1.2))

    box(3, 77, 36, 14, "1-2  Load + soil separation", "per date: plant points, ground\nmodel, 1 cm soil raster")
    box(61, 77, 36, 14, "3  Alignment of dates", "whole scan of each date onto\n29.06.: shift dx, dy and dz", hl=True)
    box(3, 53, 36, 16, "4a  Stems per date", "thin upright columns rising\nfrom the soil (2-15 cm layer)")
    box(61, 50, 36, 19, "4b  Plant register + clusters", "stems of all dates pooled;\nsame stem on >= 2 dates =\none plant instance; clusters\nsplit at 12 cm gaps; names fixed", hl=True)
    box(3, 27, 36, 18, "5  Plant instance separation", "register positions moved back\ninto this date; points split\naround them (Fadly: 9 cm match)")
    box(3, 5, 36, 16, "6  Structure + traits", "stem, leaves, height, area ...\nclouds saved in own coordinates")
    box(61, 5, 36, 16, "7-8  Tracking + growth", "rows joined by plant name;\nno point-cloud matching here")
    arrow((39, 84), (61, 84), "soil relief +\nstem-base layer", tx=(50, 88.5))
    arrow((21, 77), (21, 69.5))
    arrow((39, 61), (61, 61), "shift(x):\nx_aligned", tx=(50, 65.5))
    arrow((79, 77), (79, 69.5), "", color=ACCENT)
    arrow((61, 53), (39, 38), "inverse shift:\nregister -> date", color=ACCENT, tx=(50, 41))
    arrow((21, 27), (21, 21.5))
    arrow((39, 13), (61, 13), "names\nonly", tx=(50, 17))
    ax.text(50, 0.2, "Alignment happens once, BEFORE plant clusters and plant instances. "
            "Plant identity across dates comes from the register, not from matching separated plants.",
            ha="center", fontsize=8.5, color=INK, style="italic")
    save(fig, "fig01_flow.png")


# ------------------------------------------------------------------ fig 2: GNSS offset before / after
def fig_offset(d="230705", s0=None, s1=None):
    s0, s1 = (s0, s1) if s0 is not None else WIN
    ref_b, own_b = base_points(REF), base_points(d, seed=1)
    al_b = apply_xy(d, own_b)
    ref_s, own_s, al_s = stems_xy(REF, True), stems_xy(d, False), stems_xy(d, True)
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
    for ax, B, S, title in [(axs[0], own_b, own_s, f"Before: {LBL[d]} in its own coordinates"),
                            (axs[1], al_b, al_s, f"After: {LBL[d]} shifted by the alignment")]:
        for P, c, lab in [(ref_b, COL[REF], f"{LBL[REF]} stem-base points (reference)"), (B, COL[d], f"{LBL[d]} stem-base points")]:
            m = window(P, s0, s1, 0.05)
            s, t = st(P[m])
            ax.scatter(s, t, s=0.6, c=c, alpha=0.45, linewidths=0, rasterized=True)
        for P, c, mk, lab in [(ref_s, COL[REF], MRK[REF], f"{LBL[REF]} stems"), (S, COL[d], MRK[d], f"{LBL[d]} stems")]:
            m = window(P, s0, s1)
            s, t = st(P[m])
            ax.scatter(s, t, s=46, marker=mk, facecolors="none", edgecolors=c, linewidths=1.6, zorder=3)
        ax.set_title(title, loc="left")
        ax.set_aspect("equal")
        ax.set_xlim(s0, s1)
        ax.set_ylim(-0.25, 0.25)
        ax.set_ylabel("across row (m)")
    axs[1].set_xlabel(ALONG)
    h = [Line2D([], [], marker="o", ls="", ms=5, color=COL[REF], label=f"{LBL[REF]} stem-base points (reference)"),
         Line2D([], [], marker="o", ls="", ms=5, color=COL[d], label=f"{LBL[d]} stem-base points"),
         Line2D([], [], marker=MRK[REF], ls="", ms=8, mfc="none", mec=COL[REF], mew=1.6, label=f"{LBL[REF]} stems"),
         Line2D([], [], marker=MRK[d], ls="", ms=8, mfc="none", mec=COL[d], mew=1.6, label=f"{LBL[d]} stems")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), fontsize=8)
    a = facts["alignment"][d]
    fig.suptitle(f"GNSS offset between dates: {LBL[d]} vs {LBL[REF]} (shift found: dx {a['dx_cm']:+.1f} cm, "
                 f"dy {a['dy_cm']:+.1f} cm, dz {a['dz_cm']:+.1f} cm)", fontsize=10.5, weight="semibold", color=INK, y=1.0)
    save(fig, "fig02_offset_before_after.png")


# ------------------------------------------------------------------ fig 3 + 4: layers and phase correlation
def correlation(ref, mov, max_cells):
    w = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
    a, b = (ref - ref.mean()) * w, (mov - mov.mean()) * w
    F = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    F /= np.abs(F) + 1e-9
    c = np.fft.fftshift(np.real(np.fft.ifft2(F)))
    cy, cx = np.array(c.shape) // 2
    return c[cy - max_cells:cy + max_cells + 1, cx - max_cells:cx + max_cells + 1]


def fig_layers(d="230705", k=1):
    cfg = AlignConfig(**{k_: (tuple(v) if isinstance(v, list) else v) for k_, v in AJ["config"].items()})
    R, P = plants(REF), plants(d)
    lo = np.floor((np.minimum(P["xyz"][:, :2].min(0), R["xyz"][:, :2].min(0)) - 0.1) / cfg.cell) * cfg.cell
    hi = np.maximum(P["xyz"][:, :2].max(0), R["xyz"][:, :2].max(0)) + 0.1
    grid = Grid(lo[0], lo[1], hi[0], hi[1], cfg.cell)
    layers = {"stem_base": (base_map(R["xyz"], R["hag"], grid, cfg), base_map(P["xyz"], P["hag"], grid, cfg)),
              "soil_relief": (relief_map(ground(REF), grid, cfg)[0], relief_map(ground(d), grid, cfg)[0])}
    s_all = (R["xyz"][:, 0] - o_row[0]) * along[0] + (R["xyz"][:, 1] - o_row[1]) * along[1]
    s_min, s_max = float(np.percentile(s_all, 0.5)), float(np.percentile(s_all, 99.5))
    n_sec = max(1, int(round((s_max - s_min) / cfg.section_len)))
    edges = np.linspace(s_min, s_max, n_sec + 1)
    X, Y = grid.centers()
    S = (X - o_row[0]) * along[0] + (Y - o_row[1]) * along[1]
    sel = (S >= edges[k]) & (S < edges[k + 1])
    ii, jj = np.where(sel)
    sl = (slice(ii.min(), ii.max() + 1), slice(jj.min(), jj.max() + 1))
    m = sel[sl]
    xs = grid.x0 + (np.arange(ii.min(), ii.max() + 1) + 0.5) * cfg.cell
    ys = grid.y0 + (np.arange(jj.min(), jj.max() + 1) + 0.5) * cfg.cell
    ext = [xs[0], xs[-1], ys[0], ys[-1]]
    maxc = int(cfg.max_shift / cfg.cell)
    sec_info = {}
    s_mid = (edges[k] + edges[k + 1]) / 2
    cx, cy = o_row + along * s_mid                     # row centre in the middle of the section
    half_len = (edges[k + 1] - edges[k]) / 2 * along[1]

    def frame(ax):
        ax.set_xlim(cx - 0.45, cx + 0.45)
        ax.set_ylim(cy - half_len, cy + half_len)
        ax.set_xlabel("x (m)")
        ax.tick_params(labelsize=7)

    # ---- fig 3: the two layers, reference and date
    fig, axs = plt.subplots(1, 4, figsize=(10, 4.9), sharey=True)
    for ax, (name, which) in zip(axs, [("stem_base", 0), ("stem_base", 1), ("soil_relief", 0), ("soil_relief", 1)]):
        img = np.where(m, layers[name][which][sl], np.nan)
        if name == "stem_base":
            ax.imshow(img.T, origin="lower", extent=ext, cmap="Greys", interpolation="nearest")
        else:
            ax.imshow(img.T * 100, origin="lower", extent=ext, cmap="BrBG_r", vmin=-1.5, vmax=1.5, interpolation="nearest")
        dd = REF if which == 0 else d
        ax.set_title(f"{'stem-base layer' if name == 'stem_base' else 'soil relief'}  {LBL[dd]}", fontsize=9)
        frame(ax)
    axs[0].set_ylabel("y (m)")
    fig.suptitle(f"The two layers the alignment compares (1 cm cells): section {k + 1}, "
                 f"{edges[k] - s_min:.1f}-{edges[k + 1] - s_min:.1f} m from the start of the row",
                 fontsize=10.5, weight="semibold", color=INK)
    fig.text(0.5, -0.02, "stem-base layer: plant points 3-15 cm above ground per cell (log, dark = many).   "
             "soil relief: ground height minus its 15 cm smoothed version, from -1.5 cm (green, hollow) to +1.5 cm (brown, clod).",
             ha="center", fontsize=7.8, color=MUTED)
    save(fig, "fig03_alignment_layers.png")
    # ---- fig 4: phase correlation surfaces + overlay before / after
    from scipy.ndimage import gaussian_filter
    fig, axs = plt.subplots(1, 4, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.25, 1.25, 0.75, 0.75]})
    rec = [r for r in AL[d].sections if abs(r["s0"] - round(float(edges[k] - s_min), 2)) < 0.02]
    for ax, name in zip(axs[:2], ["stem_base", "soil_relief"]):
        ref_m, mov_m = layers[name]
        c = gaussian_filter(correlation(np.where(m, ref_m[sl], 0), np.where(m, mov_m[sl], 0), maxc), 0.8)
        c = np.clip(c / c.max(), 0, None)
        ax.imshow(c.T, origin="lower", extent=[-maxc, maxc, -maxc, maxc], cmap="magma", interpolation="bilinear")
        r = next((r for r in rec if r["layer"] == name), None)
        if r:
            ax.add_patch(Circle((r["dx"] * 100, r["dy"] * 100), 3.2, fill=False, ec="#7fdcff", lw=1.6))
            ax.set_title(f"{name.replace('_', ' ')}\npeak dx {r['dx'] * 100:+.1f}, dy {r['dy'] * 100:+.1f} cm, "
                         f"quality {r['quality']:.1f}", fontsize=8.5)
            sec_info[name] = r
        ax.set_xlabel("dx (cm)")
        ax.set_ylabel("dy (cm)")
        ax.tick_params(labelsize=7)
    fin = np.asarray(AL[d].dx_profile)[0, 1:]
    for ax, shift, title in [(axs[2], np.zeros(2), "before"), (axs[3], fin, "after the shift")]:
        ref_m, mov_m = layers["stem_base"]
        di, dj = int(round(shift[0] / cfg.cell)), int(round(shift[1] / cfg.cell))
        mov_s = np.roll(np.roll(mov_m, di, axis=0), dj, axis=1)
        a_, b_ = ref_m[sl] / max(ref_m[sl].max(), 1e-9), mov_s[sl] / max(mov_s[sl].max(), 1e-9)
        rgb = np.stack([a_, b_, a_], -1)
        rgb[~m] = 0
        ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", extent=ext, interpolation="nearest")
        ax.set_title(f"stem-base overlay\n{title}", fontsize=8.5)
        frame(ax)
        ax.set_xlim(cx - 0.3, cx + 0.3)
        ax.set_ylim(cy - 0.5, cy + 0.5)
    fig.text(0.5, -0.03, f"Left: correlation of the two layers for every possible shift (bright = good fit; circle = peak used). "
             f"Right: magenta = {LBL[REF]}, green = {LBL[d]}, white = both; after the shift more stem bases turn white.",
             ha="center", fontsize=7.8, color=MUTED)
    fig.suptitle(f"Phase correlation, section {k + 1}: the peak is the shift that moves {LBL[d]} onto {LBL[REF]}",
                 fontsize=10.5, weight="semibold", color=INK)
    fig.tight_layout()
    save(fig, "fig04_phase_correlation.png")
    facts["layer_section"] = {"date": d, "section": k + 1, "s0": float(edges[k] - s_min), "s1": float(edges[k + 1] - s_min),
                              "records": sec_info}


# ------------------------------------------------------------------ fig 5: section shifts and consensus
def fig_sections():
    ds = ["230601", "230621", "230705", "230711"]
    fig, axs = plt.subplots(len(ds), 2, figsize=(10, 8.4), sharex=True)
    for i, d in enumerate(ds):
        a = AL[d]
        fin = np.asarray(a.dx_profile)
        for j, comp in enumerate(["dx", "dy"]):
            ax = axs[i, j]
            for layer, mk in [("stem_base", "o"), ("soil_relief", "s")]:
                for used in (True, False):
                    rs = [r for r in a.sections if r["layer"] == layer and r["used"] == used]
                    if not rs:
                        continue
                    x = [(r["s0"] + r["s1"]) / 2 for r in rs]
                    y = np.clip([r[comp] * 100 for r in rs], -31, 31)
                    ax.scatter(x, y, marker=mk, s=34, facecolors=COL[d] if used else "white",
                               edgecolors=COL[d] if used else "#9aa7b4", linewidths=1.3, zorder=3)
            ax.axhline(float(np.median(fin[:, 1 + j])) * 100, color=INK, lw=1.2, zorder=2)
            if d == "230601":
                g = re.search(r"\(([-+\d.]+) / ([-+\d.]+) cm\)", a.method)
                if g:
                    stage3 = float(np.median(fin[:, 1 + j])) * 100 - float(g.group(1 + j))
                    ax.axhline(stage3, color=INK, lw=1.0, ls="--", zorder=2)
            ax.axhline(0, color=LINE, lw=0.8, zorder=1)
            ax.set_ylim(-33, 33)
            ax.set_ylabel(f"{comp} (cm)")
            if j == 0:
                ax.text(-0.22, 0.5, f"{LBL[d]}\n{a.confidence}", transform=ax.transAxes, ha="center", va="center",
                        fontsize=9.5, weight="semibold", color=INK)
    for ax in axs[-1]:
        ax.set_xlabel("section centre, m from the start of the row")
    axs[0, 0].set_title("shift across (dx) per 1.5 m section")
    axs[0, 1].set_title("shift along (dy) per 1.5 m section")
    hl = [Line2D([], [], marker="o", ls="", mfc="#777", mec="#777", label="stem-base layer"),
          Line2D([], [], marker="s", ls="", mfc="#777", mec="#777", label="soil relief"),
          Line2D([], [], marker="o", ls="", mfc="white", mec="#9aa7b4", label="hollow = not used (weak peak or disagrees > 3 cm)"),
          Line2D([], [], color=INK, label="shift used by the pipeline"),
          Line2D([], [], color=INK, ls="--", label="01.06.: stage-3 result before stem refinement")]
    fig.legend(handles=hl, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=8)
    fig.suptitle("Every section gives its own shift; the pipeline keeps the ones that agree (values beyond ±30 cm drawn at the edge)",
                 fontsize=10.5, weight="semibold", color=INK)
    save(fig, "fig05_section_shifts.png")


# ------------------------------------------------------------------ fig 6: 01.06. refined by stems
def fig_0601():
    d, nb = "230601", "230621"
    a = AL[d]
    g = re.search(r"refined by matching (\d+) of (\d+) stems to \d+ \(([-+\d.]+) / ([-+\d.]+) cm\)", a.method)
    sh = np.array([float(g.group(3)), float(g.group(4))]) / 100
    rows = STEMS[d]
    non_guided = np.array([r["found_by"] != "guided" for r in rows])
    A_fin = stems_xy(d, True)[non_guided]
    A_before = A_fin - sh
    B = stems_xy(nb, True)
    shift, n = stem_vote_shift(A_before, B)
    facts["refine_0601"] = {"n_stems": int(len(A_before)), "n_match_reported": int(g.group(1)), "n_match_recomputed": int(n),
                            "shift_reported_cm": (sh * 100).round(1).tolist(), "shift_recomputed_cm": (shift * 100).round(1).tolist()}
    from scipy.spatial import cKDTree
    fig = plt.figure(figsize=(10, 6.0))
    gsp = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], hspace=0.45, wspace=0.25)
    axa = fig.add_subplot(gsp[0, 0])
    axc = fig.add_subplot(gsp[1, 0], sharex=axa)
    axb = fig.add_subplot(gsp[:, 1])
    tree = cKDTree(B)
    for ax, A, title in [(axa, A_before, "a) after stage 3 (confidence low)"),
                         (axc, A_fin, f"c) after the vote ({sh[0] * 100:+.1f} / {sh[1] * 100:+.1f} cm)")]:
        s, t = st(B)
        ax.scatter(s, t, s=40, marker=MRK[nb], facecolors="none", edgecolors=COL[nb], linewidths=1.3)
        s, t = st(A)
        hit = tree.query(A)[0] <= 0.02
        ax.scatter(s, t, s=20, marker=MRK[d], c=COL[d], zorder=3)
        ax.set_title(title + f": {int(hit.sum())} of {len(A)} stems within 2 cm of a 21.06. stem", loc="left", fontsize=9)
        ax.set_ylim(-0.3, 0.3)
        ax.set_ylabel("across (m)")
    axc.set_xlabel(ALONG + "; across axis stretched")
    D = (B[None, :, :] - A_before[:, None, :]).reshape(-1, 2)
    D = D[np.linalg.norm(D, axis=1) <= 0.25]
    near = np.linalg.norm(D - shift, axis=1) <= 0.02
    axb.scatter(D[~near, 0] * 100, D[~near, 1] * 100, s=10, c="#b4bdc7", linewidths=0)
    axb.scatter(D[near, 0] * 100, D[near, 1] * 100, s=16, c=ACCENT, linewidths=0)
    axb.add_patch(Circle((shift[0] * 100, shift[1] * 100), 2, fill=False, ec=ACCENT, lw=1.4))
    axb.annotate(f"densest spot {shift[0] * 100:+.1f} / {shift[1] * 100:+.1f} cm\n{n} of {len(A_before)} stems agree",
                 (shift[0] * 100, shift[1] * 100), xytext=(-23, 14), fontsize=8.5, color=ACCENT,
                 arrowprops=dict(arrowstyle="-", color=ACCENT))
    axb.set_xlim(-25, 25)
    axb.set_ylim(-25, 25)
    axb.set_aspect("equal")
    axb.set_xlabel("dx (cm)")
    axb.set_ylabel("dy (cm)")
    axb.set_title("b) vote: each 01.06./21.06. stem pair\nwithin 25 cm gives one difference", loc="left", fontsize=9)
    hl = [Line2D([], [], marker=MRK[nb], ls="", mfc="none", mec=COL[nb], mew=1.3, ms=7, label=f"{LBL[nb]} stems (aligned)"),
          Line2D([], [], marker=MRK[d], ls="", color=COL[d], ms=6, label=f"{LBL[d]} stems"),
          Line2D([], [], marker="o", ls="", color=ACCENT, ms=5, label="pair differences that agree (within 2 cm)")]
    fig.legend(handles=hl, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), fontsize=8)
    fig.suptitle("01.06. (seedlings): the stage-3 match was ~24 cm off; stage 4 corrects it with stem positions",
                 fontsize=10.5, weight="semibold", color=INK)
    save(fig, "fig06_0601_stem_refinement.png")


# ------------------------------------------------------------------ fig 7: register and clusters
def fig_register(z0=2.2, z1=4.2):
    z0, z1 = WIN
    s_reg, t_reg = st(REG_XY)
    groups = sorted({r["group"] for r in REG})
    fig = plt.figure(figsize=(10, 7.2))
    gsp = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.6], hspace=0.38)
    ax0 = fig.add_subplot(gsp[0])
    for gi, g in enumerate(groups):
        m = np.array([r["group"] == g for r in REG])
        lo_, hi_ = s_reg[m].min() - 0.03, s_reg[m].max() + 0.03
        ax0.axvspan(lo_, hi_, color="#e3eafb" if gi % 2 == 0 else "#d5dff7", lw=0)
        ax0.text((lo_ + hi_) / 2, 0.205 if gi % 2 == 0 else 0.155, cluster(g).replace("plant_cluster_", "cl."),
                 ha="center", fontsize=7, color=INK)
    sc = ax0.scatter(s_reg, t_reg, c=[int(r["n_dates_seen"]) for r in REG], cmap="Blues", vmin=1, vmax=5,
                     edgecolors=INK, linewidths=0.6, s=26, zorder=3)
    ax0.axvspan(z0, z1, fill=False, ec=ACCENT, lw=1.4, ls="--")
    ax0.text((z0 + z1) / 2, -0.23, "zoom in b)", color=ACCENT, fontsize=8, ha="center", va="bottom")
    ax0.set_ylim(-0.25, 0.25)
    ax0.set_xlim(s_reg.min() - 0.2, s_reg.max() + 0.2)
    ax0.set_ylabel("across (m)")
    ax0.set_xlabel(ALONG)
    ax0.set_title(f"a) Whole row: {len(REG)} register plants (dots, shade = dates seen) in "
                  f"{len(groups)} plant clusters (bands; split at gaps >= 12 cm)", loc="left")
    cb = fig.colorbar(sc, ax=ax0, fraction=0.025, pad=0.01)
    cb.set_label("dates seen", fontsize=8)
    ax1 = fig.add_subplot(gsp[1])
    for d in DATES:
        A = stems_xy(d, True)
        named = np.array([bool(r["plant_id"]) for r in STEMS[d]])
        on = np.array([r["group"] != "off_row" for r in STEMS[d]])
        m = window(A, z0, z1)
        s, t = st(A[m & named])
        ax1.scatter(s, t, s=30, marker=MRK[d], c=COL[d], edgecolors="white", linewidths=0.5, label=f"{LBL[d]} stem", zorder=3)
        s, t = st(A[m & ~named & on])
        ax1.scatter(s, t, s=30, marker="x", c="#8a96a3", linewidths=1.2, zorder=3,
                    label="stem on the row, not on 2 dates (no plant)" if d == DATES[-1] else None)
    m = (s_reg >= z0) & (s_reg <= z1)
    for k_, j in enumerate(np.flatnonzero(m)[np.argsort(s_reg[m])]):
        ax1.add_patch(Circle((s_reg[j], t_reg[j]), 0.03, fill=False, ec=INK, lw=1.0, ls="--", zorder=2))
        up = k_ % 2 == 0
        ax1.text(s_reg[j], t_reg[j] + (0.036 if up else -0.036), instance(REG[j]["plant_id"])[-2:], ha="center",
                 va="bottom" if up else "top", fontsize=7, color=INK, weight="semibold")
    for gi, g in enumerate(groups):
        mm = np.array([r["group"] == g for r in REG]) & m
        if mm.any():
            ax1.axvspan(s_reg[mm].min() - 0.035, s_reg[mm].max() + 0.035, color="#eef2fc", lw=0, zorder=0)
            ax1.text((s_reg[mm].min() + s_reg[mm].max()) / 2, 0.2, cluster(g), ha="center", va="top", fontsize=7.5,
                     color=ACCENT, weight="semibold")
    ax1.set_xlim(z0, z1)
    ax1.set_ylim(-0.16, 0.22)
    ax1.set_aspect("equal")
    ax1.set_xlabel(ALONG)
    ax1.set_ylabel("across (m)")
    ax1.set_title("b) Zoom: stems of all five dates after alignment; dashed circle = register plant "
                  "(3 cm), number = plant_instance", loc="left")
    ax1.legend(loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig07_register_and_clusters.png")


# ------------------------------------------------------------------ fig 8: register back into one date
def fig_mapback(d="230705", s0=None, s1=None):
    s0, s1 = (s0, s1) if s0 is not None else WIN
    own_b = base_points(d, seed=2)
    m = window(own_b, s0, s1, 0.05)
    own_reg = to_own(d, REG_XY)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    s, t = st(own_b[m])
    ax.scatter(s, t, s=0.6, c=COL[d], alpha=0.35, linewidths=0, rasterized=True)
    mm = window(REG_XY, s0, s1)
    s_r, t_r = st(REG_XY[mm])
    s_o, t_o = st(own_reg[mm])
    for a, b, c, e in zip(s_r, t_r, s_o, t_o):
        ax.add_patch(FancyArrowPatch((a, b), (c, e), arrowstyle="-|>", mutation_scale=8, color=ACCENT, lw=1))
    ax.scatter(s_r, t_r, s=40, facecolors="white", edgecolors="#8a96a3", linewidths=1.3, zorder=3,
               label="register position (reference coordinates)")
    ax.scatter(s_o, t_o, s=55, marker="+", c=INK, linewidths=1.6, zorder=4, label=f"same position moved into {LBL[d]} (seed of the plant)")
    ax.scatter([], [], s=8, c=COL[d], label=f"{LBL[d]} stem-base points, own coordinates")
    ax.set_aspect("equal")
    ax.set_xlim(s0, s1)
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlabel(ALONG)
    ax.set_ylabel("across (m)")
    ax.legend(loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.45))
    a = facts["alignment"][d]
    ax.set_title(f"Register moved back into {LBL[d]} with the inverse shift ({-a['dx_cm']:+.1f} / {-a['dy_cm']:+.1f} cm): "
                 f"each seed lands on its stem", loc="left")
    save(fig, "fig08_register_into_date.png")


# ------------------------------------------------------------------ fig 9: Fadly axes matched to the register
def fig_fadly(d="230705", s0=None, s1=None):
    s0, s1 = (s0, s1) if s0 is not None else WIN
    fs = json.loads((RUN / d / "03_plants" / "instances_fadly" / "summary.json").read_text())
    org = np.asarray(fs["local_coordinate_origin_xyz_m"])[:2] - O[:2]
    rows = read_csv(RUN / d / "03_plants" / "instances_fadly" / "stem_axes_auto_review.csv")
    ax_xy = np.array([[float(r["base_x_m"]), float(r["base_y_m"])] for r in rows]) + org
    own_reg = to_own(d, REG_XY)
    own_b = base_points(d, seed=3)
    fig = plt.figure(figsize=(10, 6.6))
    gsp = fig.add_gridspec(2, 1, height_ratios=[1, 0.9], hspace=0.75)
    ax = fig.add_subplot(gsp[0])
    m = window(own_b, s0, s1, 0.05)
    s, t = st(own_b[m])
    ax.scatter(s, t, s=0.6, c="#c9d1da", alpha=0.6, linewidths=0, rasterized=True)
    mr = window(own_reg, s0, s1)
    s, t = st(own_reg[mr])
    ax.scatter(s, t, s=55, marker="+", c=INK, linewidths=1.5, zorder=4, label="register plant (moved into this date)")
    for r, p in zip(rows, ax_xy):
        if not window(p[None], s0, s1)[0]:
            continue
        sp, tp = st(p)
        matched = r["register_index"] != ""
        ax.add_patch(Circle((sp[0], tp[0]), 0.09, fill=False, ec="#9aa7b4", lw=0.7, ls=":", zorder=1))
        ax.scatter(sp, tp, s=60, marker="^", facecolors=ACCENT if matched else "white", edgecolors=ACCENT,
                   linewidths=1.3, zorder=5)
        if matched:
            q = own_reg[int(r["register_index"])]
            sq, tq = st(q)
            ax.plot([sp[0], sq[0]], [tp[0], tq[0]], color=ACCENT, lw=1.4, zorder=4)
    ax.scatter([], [], s=60, marker="^", c=ACCENT, label="Fadly candidate stem axis (matched: line to its plant)")
    ax.plot([], [], ls=":", color="#9aa7b4", label="9 cm matching limit")
    ax.set_aspect("equal")
    ax.set_xlim(s0, s1)
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlabel(ALONG)
    ax.set_ylabel("across (m)")
    ax.legend(loc="upper center", ncol=3, fontsize=7.8, bbox_to_anchor=(0.5, -0.28))
    ax.set_title(f"a) {LBL[d]}: automatic review = one-to-one matching of Fadly axes to register plants (max. 9 cm)", loc="left")
    ax2 = fig.add_subplot(gsp[1])
    rng = np.random.default_rng(0)
    for i, dd in enumerate(DATES):
        rws = read_csv(RUN / dd / "03_plants" / "instances_fadly" / "stem_axes_auto_review.csv")
        v = np.array([float(r["match_dist_m"]) * 100 for r in rws if r["match_dist_m"]])
        un = sum(1 for r in rws if not r["match_dist_m"])
        if len(v):
            ax2.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=22, marker=MRK[dd], c=COL[dd], edgecolors="white", linewidths=0.4)
            ax2.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.6)
        ax2.text(9.3, i, f"{len(v)} matched, {un} without plant", va="center", fontsize=8, color=MUTED)
    ax2.axvline(9, color=INK, ls=":", lw=1)
    ax2.set_yticks(range(len(DATES)))
    ax2.set_yticklabels([LBL[d_] for d_ in DATES])
    ax2.set_xlim(0, 13.5)
    ax2.set_xlabel("distance Fadly axis -> register plant (cm); bar = median; dotted = 9 cm limit")
    ax2.set_title("b) All dates: matched axes sit 1-1.5 cm (median) from the register position", loc="left")
    save(fig, "fig09_fadly_matching.png")


# ------------------------------------------------------------------ fig 10: one plant over time
def pick_plant():
    cols = [c for c in AVAIL[0] if re.match(r"\d\d\.\d\d\.\d{4}", c)]
    later = cols[1:]
    cand = [r for r in AVAIL if all(r[c] == "present" for c in later)]
    return cand[0]["plant_id"] if cand else AVAIL[0]["plant_id"]


def fig_same_plant(pid=None):
    pid = pid or pick_plant()
    j = next(i for i, r in enumerate(REG) if r["plant_id"] == pid)
    reg_xy = REG_XY[j]
    ds = [d for d in DATES if (RUN / d / "03_plants" / pid.split("_")[0] / pid / f"{pid}_labelled.npz").exists()]
    fig, axs = plt.subplots(2, len(ds), figsize=(10, 5.8), gridspec_kw={"height_ratios": [1, 1]})
    axs = np.atleast_2d(axs)
    bases_own, bases_ref = {}, {}
    for i, d in enumerate(ds):
        z = np.load(RUN / d / "03_plants" / pid.split("_")[0] / pid / f"{pid}_labelled.npz")
        xyz = z["xyz"] - O
        lab = z["labels"]
        ref = AL[d].apply(xyz)
        bases_own[d] = np.asarray(z["base"])[:2] - O[:2]
        bases_ref[d] = apply_xy(d, bases_own[d])[0]
        sub = np.random.default_rng(0).choice(len(ref), min(40_000, len(ref)), replace=False)
        c = np.where(lab[sub] == 0, STEM_C, LEAF_C)
        ax = axs[0, i]
        ax.scatter(ref[sub, 0] - reg_xy[0], ref[sub, 1] - reg_xy[1], s=0.3, c=c, linewidths=0, rasterized=True)
        ax.plot(0, 0, "+", color=INK, ms=14, mew=1.8)
        ax.set_xlim(-0.45, 0.45)
        ax.set_ylim(-0.45, 0.45)
        ax.set_aspect("equal")
        ax.set_title(f"{LBL[d]}  ({len(xyz):,} points)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax = axs[1, i]
        s_ = (ref[sub, :2] - reg_xy) @ along
        ax.scatter(s_, ref[sub, 2] - (ref[sub, 2].min()), s=0.3, c=c, linewidths=0, rasterized=True)
        ax.set_xlim(-0.45, 0.45)
        ax.set_ylim(0, 1.3)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    axs[0, 0].set_ylabel("top view, y - register (m)")
    axs[1, 0].set_ylabel("side view, height (m)")
    for ax in axs[1]:
        ax.set_xlabel("along row (m)")
    hl = [Line2D([], [], marker="o", ls="", color=STEM_C, label="stem"), Line2D([], [], marker="o", ls="", color=LEAF_C, label="leaves"),
          Line2D([], [], marker="+", ls="", color=INK, ms=10, mew=1.8, label="register position (same on every date)")]
    fig.legend(handles=hl, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), fontsize=8)
    fig.suptitle(f"One plant instance, one name on every date: {nice(pid)} (all drawn in reference coordinates)",
                 fontsize=10.5, weight="semibold", color=INK)
    save(fig, "fig10_same_plant_over_time.png")
    own = np.array([bases_own[d] for d in ds])
    rf = np.array([bases_ref[d] for d in ds])
    facts["same_plant"] = {"plant": nice(pid), "dates": [LBL[d] for d in ds],
                           "base_spread_own_cm": round(float(np.max(np.linalg.norm(own - own.mean(0), axis=1))) * 100, 1),
                           "base_spread_ref_cm": round(float(np.max(np.linalg.norm(rf - rf.mean(0), axis=1))) * 100, 1),
                           "base_to_register_ref_cm": [round(float(v) * 100, 1) for v in np.linalg.norm(rf - reg_xy, axis=1)]}


# ------------------------------------------------------------------ fig 11: availability
def fig_availability():
    cols = [c for c in AVAIL[0] if re.match(r"\d\d\.\d\d\.\d{4}", c)]
    cats = [("present", "#0ca30c"), ("present (stem hidden)", "#7fd07f"), ("present (no structure)", "#b7e4b7"),
            ("not visible (covered by canopy)", "#9aa7b4"), ("no stem axis found (instance code)", "#ec835a"),
            ("too small to measure", "#fab219"), ("missing", "#d03b3b")]
    idx = {c: i for i, (c, _) in enumerate(cats)}
    M = np.array([[idx.get(r[c], len(cats) - 1) for c in cols] for r in AVAIL])
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.imshow(M.T, aspect="auto", cmap=ListedColormap([c for _, c in cats]), vmin=-0.5, vmax=len(cats) - 0.5,
              interpolation="nearest")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([c[:6] for c in cols])
    ax.set_xticks(range(len(AVAIL)))
    ax.set_xticklabels([nice(r["plant_id"]).replace("plant_cluster_", "").replace("_plant_instance_", "·") for r in AVAIL],
                       rotation=90, fontsize=6)
    ax.set_xticks(np.arange(-0.5, len(AVAIL)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cols)), minor=True)
    ax.grid(which="minor", color="white", lw=1)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("register plant (cluster·instance)")
    used = sorted(set(M.ravel()))
    ax.legend(handles=[Patch(color=cats[i][1], label=f"{cats[i][0]} ({int((M == i).sum())})") for i in used],
              loc="upper center", ncol=3, fontsize=7.8, bbox_to_anchor=(0.5, -0.3))
    ax.set_title("Status of every register plant on every date (tracking/availability.csv)", loc="left")
    save(fig, "fig11_availability.png")


if __name__ == "__main__":
    print("run:", RUN)
    fig_flow()
    fig_offset()
    fig_layers()
    fig_sections()
    fig_0601()
    fig_register()
    fig_mapback()
    fig_fadly()
    fig_same_plant()
    fig_availability()
    (HERE / "facts.json").write_text(json.dumps(facts, indent=2, default=str))
    print("  wrote facts.json")
