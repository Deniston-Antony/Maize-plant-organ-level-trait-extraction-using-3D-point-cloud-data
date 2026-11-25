"""Stage 3: align every date onto the reference date.

GNSS error moves whole scans by several cm (measured vs 29.06: up to 10 cm in
XY, up to 3 cm in z). Plants grow and move, so the alignment uses only:

  * the stem-base layer: plant points 3-15 cm above ground, counted per 1 cm
    cell (stems stay where they are);
  * the soil relief: ground model minus its 15 cm smoothed version (clods,
    ridges, wheel marks).

The row is cut into sections along its length; in each section the shift is
found by phase correlation (FFT, sub-cell peak). Sections that agree are
combined; if they disagree, the shift is interpolated along the row. The z
offset is the median ground-height difference after the XY shift.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .soil import DTM


@dataclass
class AlignConfig:
    cell: float = 0.01
    base_hag: tuple[float, float] = (0.03, 0.15)
    section_len: float = 1.5          # m along the row
    max_shift: float = 0.30           # ignore peaks farther than this (m)
    min_quality: float = 1.15         # peak / second peak; weaker sections are ignored
    agree_tol: float = 0.03           # sections within this of the consensus are inliers
    drift_min: float = 0.015          # use a linear drift along the row if it changes the shift by more than this
    relief_sigma: float = 0.15        # m, smoothing removed from the ground model
    min_base_points_per_m: float = 2000.0   # fewer -> stem layer too weak (seedlings), soil only

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Alignment:
    date: str
    reference: str
    origin_xy: list               # row frame origin (local coords)
    along: list                   # unit vector along the row
    sections: list = field(default_factory=list)   # dicts: s0, s1, dx, dy, quality, layer, used
    dx_profile: list = field(default_factory=list) # (s, dx, dy) knots; constant if one knot
    dz: float = 0.0
    method: str = ""
    confidence: str = "high"
    notes: list = field(default_factory=list)
    # fine shifts per plant cluster (stage 4 "3b"): {"origin": [x, y], "along": [ax, ay], "margin": m,
    # "items": [[s_lo, s_hi, dx, dy], ...]} - s along the reference row frame, AFTER the shift above
    local_shifts: dict = field(default_factory=dict)

    # map points of this date into the reference frame
    def apply(self, xyz: np.ndarray) -> np.ndarray:
        out = np.array(xyz, dtype=np.float64, copy=True)
        if self.date == self.reference:
            return out
        k = np.asarray(self.dx_profile, float)
        s = (out[:, 0] - self.origin_xy[0]) * self.along[0] + (out[:, 1] - self.origin_xy[1]) * self.along[1]
        if len(k) == 1:
            dx, dy = k[0, 1], k[0, 2]
        else:
            dx, dy = np.interp(s, k[:, 0], k[:, 1]), np.interp(s, k[:, 0], k[:, 2])
        out[:, 0] += dx
        out[:, 1] += dy
        out[:, 2] += self.dz
        ls = self.local_shifts
        if ls and ls.get("items") and len(out):
            o, a, m = np.asarray(ls["origin"]), np.asarray(ls["along"]), float(ls.get("margin", 0.03))
            s2 = (out[:, 0] - o[0]) * a[0] + (out[:, 1] - o[1]) * a[1]
            done = np.zeros(len(out), bool)
            for lo, hi, ddx, ddy in ls["items"]:
                sel = ~done & (s2 >= lo - m) & (s2 <= hi + m)
                out[sel, 0] += ddx
                out[sel, 1] += ddy
                done |= sel
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Alignment":
        return cls(**d)


def row_frame(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Origin and along-row unit vector (main direction of the plant points)."""
    sub = xy[:: max(1, len(xy) // 200_000)]
    o = sub.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((sub - o).T))
    a = v[:, int(np.argmax(w))]
    if a[1] < 0:            # point "north" so that s grows with the scan
        a = -a
    return o, a


class Grid:
    def __init__(self, xmin, ymin, xmax, ymax, cell):
        self.x0, self.y0, self.cell = xmin, ymin, cell
        self.nx = int(np.ceil((xmax - xmin) / cell)) + 1
        self.ny = int(np.ceil((ymax - ymin) / cell)) + 1

    def count(self, xy: np.ndarray) -> np.ndarray:
        g = np.zeros((self.nx, self.ny), np.float32)
        ix = ((xy[:, 0] - self.x0) / self.cell).astype(np.int64)
        iy = ((xy[:, 1] - self.y0) / self.cell).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        np.add.at(g, (ix[ok], iy[ok]), 1)
        return g

    def sample_dtm(self, dtm: DTM) -> np.ndarray:
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.cell
        ys = self.y0 + (np.arange(self.ny) + 0.5) * self.cell
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        return dtm.height(X.ravel(), Y.ravel()).reshape(self.nx, self.ny)

    def centers(self):
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.cell
        ys = self.y0 + (np.arange(self.ny) + 0.5) * self.cell
        return np.meshgrid(xs, ys, indexing="ij")


def base_map(plants_xyz: np.ndarray, hag: np.ndarray, grid: Grid, cfg: AlignConfig) -> np.ndarray:
    m = (hag >= cfg.base_hag[0]) & (hag <= cfg.base_hag[1])
    g = grid.count(plants_xyz[m, :2])
    return np.log1p(gaussian_filter(g, 1.0))


def relief_map(ground: dict, grid: Grid, cfg: AlignConfig) -> tuple[np.ndarray, np.ndarray]:
    """Fine soil relief from the 1 cm soil raster (not the 3 cm ground model:
    its cell edges form a regular pattern that the matching can lock onto).
    The grid is anchored at the project origin, so cells are the same on every date."""
    z = np.full((grid.nx, grid.ny), np.nan)
    i = ground["ix"].astype(np.int64) - int(round(grid.x0 / grid.cell))
    j = ground["iy"].astype(np.int64) - int(round(grid.y0 / grid.cell))
    ok = (i >= 0) & (i < grid.nx) & (j >= 0) & (j < grid.ny)
    z[i[ok], j[ok]] = ground["z"][ok]
    have = np.isfinite(z)
    sig = cfg.relief_sigma / grid.cell
    w = gaussian_filter(have.astype(float), sig)
    smooth = gaussian_filter(np.where(have, z, 0.0), sig) / np.maximum(w, 1e-6)
    rel = np.where(have, z - smooth, 0.0)
    rel = np.clip(rel, -0.03, 0.03)                       # clods and holes only, no outliers
    return gaussian_filter(rel, 0.7), have


def phase_shift(ref: np.ndarray, mov: np.ndarray, max_cells: int) -> tuple[float, float, float]:
    """Shift (cells) that moves `mov` onto `ref`, and the peak quality."""
    if ref.std() < 1e-9 or mov.std() < 1e-9:
        return 0.0, 0.0, 0.0
    w = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
    a, b = (ref - ref.mean()) * w, (mov - mov.mean()) * w
    F = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    F /= np.abs(F) + 1e-9
    c = np.fft.fftshift(np.real(np.fft.ifft2(F)))
    cy, cx = np.array(c.shape) // 2
    win = np.zeros_like(c, bool)
    win[max(0, cy - max_cells):cy + max_cells + 1, max(0, cx - max_cells):cx + max_cells + 1] = True
    cw = np.where(win, c, -np.inf)
    i, j = np.unravel_index(int(np.argmax(cw)), c.shape)
    peak = c[i, j]
    # sub-cell refinement (parabola through the neighbours)
    def sub(v_m, v_0, v_p):
        d = v_m - 2 * v_0 + v_p
        return 0.5 * (v_m - v_p) / d if abs(d) > 1e-12 else 0.0
    di = sub(c[i - 1, j], peak, c[(i + 1) % c.shape[0], j]) if 0 < i < c.shape[0] - 1 else 0.0
    dj = sub(c[i, j - 1], peak, c[i, (j + 1) % c.shape[1]]) if 0 < j < c.shape[1] - 1 else 0.0
    # quality: peak vs best value outside a 3-cell radius
    mask = cw.copy()
    mask[max(0, i - 3):i + 4, max(0, j - 3):j + 4] = -np.inf
    second = np.max(mask[np.isfinite(mask)]) if np.isfinite(mask).any() else 0.0
    q = float(peak / second) if second > 0 else 0.0
    return float(i + di - cy), float(j + dj - cx), q


def align_date(date: str, ref_date: str, plants: dict, dtm: DTM, ref_plants: dict, ref_dtm: DTM,
               cfg: AlignConfig | None = None, log=print, ground: dict | None = None,
               ref_ground: dict | None = None) -> tuple[Alignment, dict]:
    cfg = cfg or AlignConfig()
    o, a = row_frame(ref_plants["xyz"][:, :2])
    al = Alignment(date, ref_date, o.tolist(), a.tolist())
    if date == ref_date:
        al.dx_profile, al.method = [[0.0, 0.0, 0.0]], "reference"
        return al, {}

    P, R = plants["xyz"].astype(np.float64), ref_plants["xyz"].astype(np.float64)
    lo = np.floor((np.minimum(P[:, :2].min(axis=0), R[:, :2].min(axis=0)) - 0.1) / cfg.cell) * cfg.cell
    hi = np.maximum(P[:, :2].max(axis=0), R[:, :2].max(axis=0)) + 0.1
    grid = Grid(lo[0], lo[1], hi[0], hi[1], cfg.cell)
    layers = {
        "stem_base": (base_map(R, ref_plants["hag"], grid, cfg), base_map(P, plants["hag"], grid, cfg)),
    }
    if ground is not None and ref_ground is not None:
        layers["soil_relief"] = (relief_map(ref_ground, grid, cfg)[0], relief_map(ground, grid, cfg)[0])
    else:
        al.notes.append("no 1 cm soil raster: stem-base layer only")
    n_base = int(((plants["hag"] >= cfg.base_hag[0]) & (plants["hag"] <= cfg.base_hag[1])).sum())
    s_all = (R[:, 0] - o[0]) * a[0] + (R[:, 1] - o[1]) * a[1]
    s_min, s_max = float(np.percentile(s_all, 0.5)), float(np.percentile(s_all, 99.5))
    row_len = s_max - s_min
    stem_ok = n_base / max(row_len, 1e-6) >= cfg.min_base_points_per_m
    if not stem_ok:
        al.notes.append(f"stem-base layer weak ({n_base:,} points, seedlings): soil relief only")

    X, Y = grid.centers()
    S = (X - o[0]) * a[0] + (Y - o[1]) * a[1]
    n_sec = max(1, int(round(row_len / cfg.section_len)))
    edges = np.linspace(s_min, s_max, n_sec + 1)
    maxc = int(cfg.max_shift / cfg.cell)
    for k in range(n_sec):
        sel = (S >= edges[k]) & (S < edges[k + 1])
        ii, jj = np.where(sel)
        if len(ii) < 100:
            continue
        sl = (slice(ii.min(), ii.max() + 1), slice(jj.min(), jj.max() + 1))
        for name, (ref_m, mov_m) in layers.items():
            if name == "stem_base" and not stem_ok:
                continue
            m = sel[sl]
            di, dj, q = phase_shift(np.where(m, ref_m[sl], 0), np.where(m, mov_m[sl], 0), maxc)
            al.sections.append({"s0": round(float(edges[k] - s_min), 2), "s1": round(float(edges[k + 1] - s_min), 2),
                                "s_mid": float((edges[k] + edges[k + 1]) / 2), "layer": name,
                                "dx": di * cfg.cell, "dy": dj * cfg.cell, "quality": q, "used": False})

    combine_sections(al, cfg, s_min)

    # z offset after the XY shift, from the ground models
    zr = grid.sample_dtm(ref_dtm)
    Xs, Ys = X.ravel(), Y.ravel()
    back = al.apply(np.column_stack([Xs, Ys, np.zeros_like(Xs)]))          # reference cell -> where it is in this date: inverse
    src = np.column_stack([2 * Xs - back[:, 0], 2 * Ys - back[:, 1]])       # x_date = x_ref - shift
    zd = dtm.height(src[:, 0], src[:, 1]).reshape(zr.shape)
    both = np.isfinite(zr) & np.isfinite(zd) & ref_dtm_observed(ref_dtm, grid) & ref_dtm_observed(dtm, grid, src)
    if both.sum() > 100:
        al.dz = float(np.median((zr - zd)[both]))
    else:
        al.notes.append("too little shared ground for the z offset")

    dxy = np.array(al.dx_profile)[:, 1:]
    log(f"  align {date} -> {ref_date}: shift {np.median(dxy[:, 0]) * 100:+.1f} / {np.median(dxy[:, 1]) * 100:+.1f} cm "
        f"(x / y), z {al.dz * 100:+.1f} cm; {al.method}; confidence {al.confidence}")
    return al, {"grid": grid, "layers": layers}


def combine_sections(al: Alignment, cfg: AlignConfig, s_min: float) -> None:
    """Consensus of all sections from both layers (weighted by peak quality),
    then a straight-line drift along the row if the inliers show one."""
    cand = [r for r in al.sections if r["quality"] >= cfg.min_quality]
    if not cand:
        al.dx_profile, al.method, al.confidence = [[0.0, 0.0, 0.0]], "none (no reliable peak)", "low"
        al.notes.append("no section gave a clear match")
        return
    D = np.array([[r["dx"], r["dy"]] for r in cand])
    W = np.array([r["quality"] - 1.0 for r in cand])
    S = np.array([r["s_mid"] for r in cand])
    # consensus: the candidate with the most (weighted) agreeing candidates
    support = [(W * (np.linalg.norm(D - d, axis=1) <= cfg.agree_tol)).sum() for d in D]
    c = D[int(np.argmax(support))]
    for _ in range(2):
        inl = np.linalg.norm(D - c, axis=1) <= cfg.agree_tol
        c = np.average(D[inl], axis=0, weights=W[inl])
    for r, u in zip(cand, inl):
        r["used"] = bool(u)
    Di, Wi, Si = D[inl], W[inl], S[inl]
    n_sec = len(set(Si.tolist()))
    layers = sorted({r["layer"] for r, u in zip(cand, inl) if u})
    al.dx_profile = [[0.0, float(c[0]), float(c[1])]]
    al.method = f"global shift ({'+'.join(layers)}, {int(inl.sum())} of {len(cand)} section matches agree)"
    resid = Di - c
    if n_sec >= 3 and np.ptp(Si) >= 3.0:
        A = np.column_stack([Si - Si.mean(), np.ones(len(Si))])
        sw = np.sqrt(Wi)[:, None]
        coef_x = np.linalg.lstsq(A * sw, Di[:, 0] * sw[:, 0], rcond=None)[0]
        coef_y = np.linalg.lstsq(A * sw, Di[:, 1] * sw[:, 0], rcond=None)[0]
        change = np.hypot(coef_x[0], coef_y[0]) * np.ptp(Si)
        if change >= cfg.drift_min:
            knots = [Si.min(), Si.max()]
            al.dx_profile = [[float(k), float(coef_x[1] + coef_x[0] * (k - Si.mean())),
                              float(coef_y[1] + coef_y[0] * (k - Si.mean()))] for k in knots]
            al.method += f"; linear drift along the row ({change * 100:.1f} cm over {np.ptp(Si):.1f} m)"
            resid = Di - np.column_stack([coef_x[1] + coef_x[0] * (Si - Si.mean()), coef_y[1] + coef_y[0] * (Si - Si.mean())])
    al.notes.append(f"residual of agreeing sections {np.sqrt(np.mean(np.sum(resid ** 2, axis=1))) * 100:.1f} cm (RMS)")
    rms = float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))
    al.confidence = "high" if (n_sec >= 3 and rms <= 0.012) else ("medium" if n_sec >= 2 else "low")


def compose(first: Alignment, second: Alignment) -> list:
    """Profile of 'first then second' (date -> neighbour -> reference), sampled along the row."""
    ss = sorted({k[0] for k in first.dx_profile} | {k[0] for k in second.dx_profile} | {0.0})
    out = []
    for sv in ss:
        p = np.array([[second.origin_xy[0] + second.along[0] * sv, second.origin_xy[1] + second.along[1] * sv, 0.0]])
        q = second.apply(first.apply(p))
        out.append([sv, float(q[0, 0] - p[0, 0]), float(q[0, 1] - p[0, 1])])
    return out


def ref_dtm_observed(dtm: DTM, grid: Grid, xy: np.ndarray | None = None) -> np.ndarray:
    if xy is None:
        X, Y = grid.centers()
        xy = np.column_stack([X.ravel(), Y.ravel()])
    nx, ny = dtm.z.shape
    ix = np.clip(((xy[:, 0] - dtm.x0) / dtm.cell).astype(np.int64), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - dtm.y0) / dtm.cell).astype(np.int64), 0, ny - 1)
    return dtm.observed[ix, iy].reshape(grid.nx, grid.ny)


def save_alignments(path: Path, als: list[Alignment], cfg: AlignConfig) -> None:
    path.write_text(json.dumps({"config": cfg.to_dict(), "dates": {a.date: a.to_dict() for a in als}}, indent=2))


def load_alignments(path: Path) -> dict[str, Alignment]:
    d = json.loads(Path(path).read_text())
    return {k: Alignment.from_dict(v) for k, v in d["dates"].items()}
