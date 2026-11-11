"""Check figures for the soil and alignment stages (PNG, matplotlib Agg)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import pretty_date  # noqa: E402
from .names import cluster, instance, nice  # noqa: E402,F401


def _thin(n: int, k: int, seed: int = 0) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, k, replace=False)


def soil_figure(path: Path, date: str, xyz: np.ndarray, hag: np.ndarray, plant_idx: np.ndarray, dtm, stats: dict) -> None:
    """Top view (plants coloured by height above ground, soil grey) and three
    side slices across the row (soil brown, plant green)."""
    plant = np.zeros(len(xyz), bool)
    plant[plant_idx] = True
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.1, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    si, pi = _thin(int((~plant).sum()), 300_000), _thin(len(plant_idx), 400_000, 1)
    soil_xyz = xyz[~plant][si]
    ax.scatter(soil_xyz[:, 0], soil_xyz[:, 1], s=0.05, c="0.75", rasterized=True)
    pp = xyz[plant_idx][pi]
    sc = ax.scatter(pp[:, 0], pp[:, 1], s=0.05, c=hag[plant_idx][pi], cmap="viridis", vmin=0, vmax=np.percentile(hag[plant_idx], 99),
                    rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.04, label="height above ground (m)")
    ax.set_aspect("equal")
    ax.set_title(f"{pretty_date(date)} - top view ({stats['n_plant']:,} plant points)")
    ax.set_xlabel("x (m, project frame)")
    ax.set_ylabel("y (m)")

    # three slices across the row, 4 cm thick, at 25 / 50 / 75 % of the row
    o = xyz[plant_idx, :2].mean(axis=0)
    w, v = np.linalg.eigh(np.cov((xyz[plant_idx[::50], :2] - o).T))
    along = v[:, int(np.argmax(w))]
    across = np.array([-along[1], along[0]])
    s = (xyz[:, :2] - o) @ along
    t = (xyz[:, :2] - o) @ across
    sp = s[plant_idx]
    for r, q in enumerate((25, 50, 75)):
        c = np.percentile(sp, q)
        m = np.abs(s - c) <= 0.02
        ax.plot(*(o + along * c + np.outer([-0.4, 0.4], across)).T, "r-", lw=1)
        for col, zoom in ((1, False), (2, True)):
            a2 = fig.add_subplot(gs[r, col])
            mm = m & ~plant
            a2.scatter(t[mm], xyz[mm, 2], s=0.3, c="#8d6e4a", rasterized=True, label="soil")
            mm = m & plant
            a2.scatter(t[mm], xyz[mm, 2], s=0.3, c="#2e7d32", rasterized=True, label="plant")
            X = o + along * c
            tt = np.linspace(t[m].min() if m.any() else -0.4, t[m].max() if m.any() else 0.4, 200)
            P = X + np.outer(tt, across)
            a2.plot(tt, dtm.height(P[:, 0], P[:, 1]), "k-", lw=0.7, label="ground model")
            if zoom:
                zg = np.nanmedian(dtm.height(P[:, 0], P[:, 1]))
                a2.set_ylim(zg - 0.05, zg + 0.15)
                a2.set_title(f"slice at {q} % of the row - zoom near the soil", fontsize=9)
            else:
                a2.set_title(f"slice at {q} % of the row (4 cm thick)", fontsize=9)
            a2.set_xlabel("across row (m)", fontsize=8)
            if r == 0 and col == 1:
                a2.legend(markerscale=12, fontsize=7, loc="upper right")
    fig.suptitle(f"Soil separation {pretty_date(date)}: plant share {stats['plant_share']:.1%}, "
                 f"low leaf points kept {stats['n_low_kept_as_plant']:,}, ground under plants filled in "
                 f"{stats['dtm_cells_under_plants_filled_share']:.0%} of cells (95 % within "
                 f"{stats['dtm_fill_dist_under_plants_p95_m'] * 100:.0f} cm)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def alignment_figure(path: Path, al, aux: dict) -> None:
    """Stem-base maps of the reference (magenta) and this date (green), before
    and after the shift; overlap shows white."""
    if not aux:
        return
    grid = aux["grid"]
    ref_m, mov_m = aux["layers"]["stem_base"]
    k = np.asarray(al.dx_profile)
    dx, dy = float(np.median(k[:, 1])), float(np.median(k[:, 2]))
    from scipy.ndimage import shift as nd_shift
    moved = nd_shift(mov_m, (dx / grid.cell, dy / grid.cell), order=1)

    def rgb(a, b):
        n = lambda v: np.clip(v / (np.percentile(v[v > 0], 99) if (v > 0).any() else 1), 0, 1)
        A, B = n(a), n(b)
        return np.dstack([A, B, A]).transpose(1, 0, 2)

    fig, axs = plt.subplots(1, 3, figsize=(16, 9), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    ext = [grid.x0, grid.x0 + grid.nx * grid.cell, grid.y0, grid.y0 + grid.ny * grid.cell]
    for ax, img, title in ((axs[0], rgb(ref_m, mov_m), "before"), (axs[1], rgb(ref_m, moved), "after shift")):
        ax.imshow(img, origin="lower", extent=ext)
        ax.set_title(f"{title}: magenta = {pretty_date(al.reference)}, green = {pretty_date(al.date)}, white = both", fontsize=9)
        ax.set_xlabel("x (m)")
    # zoom on the 60 cm window with the most stem-base points
    from scipy.ndimage import uniform_filter
    h = int(0.3 / grid.cell)
    i, j = np.unravel_index(int(np.argmax(uniform_filter(ref_m, size=2 * h))), ref_m.shape)
    sl = (slice(max(0, i - h), i + h), slice(max(0, j - h), j + h))
    z0 = rgb(ref_m[sl], mov_m[sl])
    z1 = rgb(ref_m[sl], moved[sl])
    axs[2].imshow(np.concatenate([z0, np.ones((z0.shape[0], 3, 3)), z1], axis=1), origin="lower")
    axs[2].set_title("zoom (60 cm): before | after", fontsize=9)
    axs[2].axis("off")
    sec = "\n".join(f"{r['layer']:<11} {r['s0']:>4.1f}-{r['s1']:<4.1f} m  dx {r['dx'] * 100:+5.1f}  dy {r['dy'] * 100:+5.1f} cm  "
                    f"q {r['quality']:.1f}{'  used' if r['used'] else ''}" for r in al.sections)
    fig.text(0.01, 0.01, sec, family="monospace", fontsize=7, va="bottom")
    fig.suptitle(f"Alignment {pretty_date(al.date)} -> {pretty_date(al.reference)}: dx {dx * 100:+.1f} cm, dy {dy * 100:+.1f} cm, "
                 f"dz {al.dz * 100:+.1f} cm - {al.method} - confidence {al.confidence}", fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def stems_figure(path: Path, date: str, plants_xyz: np.ndarray, keep: np.ndarray, stems: list[dict],
                 groups_xy: dict, title_extra: str = "") -> None:
    """Top view along the row: plant points (grey = removed clods/weeds), stems
    coloured by group, group names at the reference positions. The row is
    split into 3 panels side by side so the stems stay readable."""
    n_pan = 3
    y = plants_xyz[:, 1]
    y0, y1 = np.percentile(y, 0.2), np.percentile(y, 99.8)
    edges = np.linspace(y0, y1, n_pan + 1)
    fig, axs = plt.subplots(1, n_pan, figsize=(18, 11))
    cmap = plt.get_cmap("tab20")
    for k, ax in enumerate(axs):
        m = (y >= edges[k] - 0.05) & (y < edges[k + 1] + 0.05)
        idx = np.where(m)[0]
        idx = idx[_thin(len(idx), 250_000, k)]
        ax.scatter(plants_xyz[idx, 0], plants_xyz[idx, 1], s=0.05, c=np.where(keep[idx], "#7fa77f", "#d0d0d0"), rasterized=True)
        for s_ in stems:
            if not (edges[k] - 0.05 <= s_["y"] < edges[k + 1] + 0.05):
                continue
            g = s_.get("group", -1)
            col = cmap(g % 20) if g >= 0 else "red"
            if s_.get("guided"):
                ax.scatter([s_["x"]], [s_["y"]], s=30, marker="x", c=[col], linewidths=1.2)
            else:
                ax.scatter([s_["x"]], [s_["y"]], s=30, marker="o", facecolors="none", edgecolors=[col], linewidths=1.2)
        for name, (gx, gy) in groups_xy.items():
            if edges[k] - 0.05 <= gy < edges[k + 1] + 0.05:
                ax.text(gx + 0.12, gy, cluster(name), fontsize=8, va="center")
        ax.set_aspect("equal")
        ax.set_ylim(edges[k] - 0.05, edges[k + 1] + 0.05)
        ax.set_xlabel("x (m)")
    axs[0].set_ylabel("y (m)")
    fig.suptitle(f"Stems and plant clusters {pretty_date(date)}: {len(stems)} stems "
                 f"(o = found, x = recovered at a reference position); grey = low blobs removed (clods, weeds). {title_extra}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plants_figure(path: Path, date: str, xyz: np.ndarray, labels: np.ndarray, names: list[str],
                  pos: np.ndarray, present: list[bool]) -> None:
    """Top view coloured by plant (3 panels along the row) with plant names."""
    n_pan = 3
    y = xyz[:, 1]
    y0, y1 = np.percentile(y, 0.2), np.percentile(y, 99.8)
    edges = np.linspace(y0, y1, n_pan + 1)
    fig, axs = plt.subplots(1, n_pan, figsize=(18, 11))
    rng = np.random.default_rng(3)
    pal = plt.get_cmap("tab20")(np.arange(20))
    col_of = rng.permutation(len(names)) % 20
    for k, ax in enumerate(axs):
        m = (y >= edges[k] - 0.05) & (y < edges[k + 1] + 0.05)
        idx = np.where(m)[0]
        idx = idx[_thin(len(idx), 300_000, k)]
        lab = labels[idx]
        c = np.where(lab[:, None] >= 0, pal[col_of[np.clip(lab, 0, None)]], [[0.85, 0.85, 0.85, 1]])
        ax.scatter(xyz[idx, 0], xyz[idx, 1], s=0.05, c=c, rasterized=True)
        for nm, p_, pr in zip(names, pos, present):
            if edges[k] - 0.05 <= p_[1] < edges[k + 1] + 0.05:
                ax.plot(p_[0], p_[1], "k*" if pr else "rx", ms=6)
                ax.text(p_[0] + 0.015, p_[1], nice(nm).replace("_plant_instance", "\nplant_instance"), fontsize=5,
                        color="k" if pr else "r")
        ax.set_aspect("equal")
        ax.set_ylim(edges[k] - 0.05, edges[k + 1] + 0.05)
    fig.suptitle(f"Plant instances {pretty_date(date)}: colour = plant instance, * = present, red x = missing on this date, "
                 f"grey = not assigned", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
