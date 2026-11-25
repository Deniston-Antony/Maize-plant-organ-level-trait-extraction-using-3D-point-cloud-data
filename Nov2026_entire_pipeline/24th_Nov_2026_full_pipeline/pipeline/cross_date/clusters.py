"""Step 2: plant clusters on each date from the low-layer density profile.

1. density profile along the row (points per 1 cm bin, after 5 mm thinning so
   that every date is on the same scale), lightly smoothed;
2. background level = mean density in the gaps between clusters (found
   iteratively); a cluster is a stretch that rises above it;
3. one Gaussian per cluster (density-weighted mean = centre, sd = width);
   cluster points = points inside the stretch and within +-2 sigma;
4. one Gaussian per stem inside the cluster (2-D Gaussian mixture, number
   chosen by BIC) - used later for the plant check;
5. clusters whose plant points never rise above 5 cm are marked as clods.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture

from .config import ClusterConfig
from .inputs import Run


@dataclass
class Cluster:
    date: str
    idx: int               # order along the row on this date (0..)
    cid: str               # e.g. 230705_c07
    lo: float              # stretch along the row (m)
    hi: float
    mu: float              # Gaussian centre (m along the row)
    sigma: float           # Gaussian width (m)
    t: float               # centre across the row (m)
    mass: float            # area under the profile above background
    peak: float
    n_points: int          # low-layer points inside +-2 sigma
    top_hag: float         # 95th percentile height of plant points in the stretch (m)
    clod: bool
    stems: list = field(default_factory=list)   # per-stem Gaussians: dicts s, t, sd_s, sd_t, weight

    def row(self) -> dict:
        return {"date": self.date, "cluster": self.cid, "order": self.idx + 1,
                "lo_m": round(self.lo, 4), "hi_m": round(self.hi, 4),
                "centre_m": round(self.mu, 4), "sigma_m": round(self.sigma, 4),
                "across_m": round(self.t, 4), "mass": round(self.mass, 1), "peak": round(self.peak, 1),
                "n_points": self.n_points, "top_hag_m": round(self.top_hag, 3),
                "clod": self.clod, "n_stem_gaussians": len(self.stems)}


@dataclass
class DateProfile:
    date: str
    edges: np.ndarray
    centres: np.ndarray
    density: np.ndarray    # smoothed
    background: float
    bg_sd: float
    threshold: float


def profile(s: np.ndarray, s_min: float, s_max: float, cfg: ClusterConfig):
    edges = np.arange(s_min, s_max + cfg.bin, cfg.bin)
    h, _ = np.histogram(s, edges)
    return edges, 0.5 * (edges[:-1] + edges[1:]), gaussian_filter1d(h.astype(float), cfg.smooth / cfg.bin)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index pairs of True stretches."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def threshold(dens: np.ndarray, cfg: ClusterConfig) -> tuple[float, float, float]:
    occupied = dens[dens > 0]
    floor = cfg.bg_min_frac * (np.percentile(occupied, 99) if occupied.size else 0.0)
    T = floor
    bg, sd = 0.0, 0.0
    for _ in range(5):
        gap = dens <= T
        # only gaps inside the planted part of the row (between the first and last cluster)
        on = np.flatnonzero(~gap)
        if on.size:
            gap[: on[0]] = False
            gap[on[-1] + 1:] = False
        vals = dens[gap]
        if vals.size < 5:
            break
        bg, sd = float(vals.mean()), float(vals.std())
        T_new = max(bg + cfg.bg_k * sd, floor)
        if abs(T_new - T) < 1e-6:
            break
        T = T_new
    return T, bg, sd


def _stem_gaussians(st: np.ndarray, cfg: ClusterConfig, seed: int = 0) -> list[dict]:
    if len(st) < 30:
        return [{"s": float(st[:, 0].mean()), "t": float(st[:, 1].mean()),
                 "sd_s": float(st[:, 0].std()), "sd_t": float(st[:, 1].std()), "weight": 1.0}] if len(st) else []
    rng = np.random.default_rng(seed)
    if len(st) > cfg.stem_gmm_points:
        st = st[rng.choice(len(st), cfg.stem_gmm_points, replace=False)]
    best, best_bic = None, np.inf
    for k in range(1, min(cfg.stem_gmm_max, len(st) // 30) + 1):
        g = GaussianMixture(k, covariance_type="full", reg_covar=1e-5, random_state=seed, n_init=2).fit(st)
        b = g.bic(st)
        if b < best_bic:
            best, best_bic = g, b
    out = []
    for m, c, w in zip(best.means_, best.covariances_, best.weights_):
        out.append({"s": float(m[0]), "t": float(m[1]), "sd_s": float(np.sqrt(c[0, 0])),
                    "sd_t": float(np.sqrt(c[1, 1])), "weight": float(w)})
    return sorted(out, key=lambda r: r["s"])


def find_clusters(run: Run, cfg: ClusterConfig, log=print) -> tuple[dict, dict]:
    """Returns {date: [Cluster]} and {date: DateProfile}."""
    s_all = np.concatenate([run.low[d].s for d in run.dates])
    s_min, s_max = float(np.floor(s_all.min() * 10) / 10 - 0.1), float(np.ceil(s_all.max() * 10) / 10 + 0.1)
    clusters, profiles = {}, {}
    for d in run.dates:
        L = run.low[d]
        edges, cen, dens = profile(L.s, s_min, s_max, cfg)
        T, bg, sd = threshold(dens, cfg)
        profiles[d] = DateProfile(d, edges, cen, dens, bg, sd, T)

        runs = _runs(dens > T)
        merged = []
        for a, b in runs:                      # join runs separated by a short dip
            if merged and (a - merged[-1][1]) * cfg.bin < cfg.join_gap:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        merged = [(a, b) for a, b in merged if (b - a) * cfg.bin >= cfg.min_width]

        cand = []
        for a, b in merged:
            w = np.clip(dens[a:b] - bg, 0, None)
            if w.sum() <= 0:
                continue
            mu = float((w * cen[a:b]).sum() / w.sum())
            sig = float(np.sqrt((w * (cen[a:b] - mu) ** 2).sum() / w.sum()))
            sig = max(sig, cfg.bin)
            cand.append((a, b, mu, sig, float(w.sum()), float(dens[a:b].max())))
        if not cand:
            clusters[d] = []
            continue
        med_mass = np.median([c[4] for c in cand])
        cand = [c for c in cand if c[4] >= cfg.min_mass_frac * med_mass]

        ts, th = run.tall[d]
        out = []
        for i, (a, b, mu, sig, mass, peak) in enumerate(cand):
            lo, hi = float(edges[a]), float(edges[b])
            inside = (L.s >= lo) & (L.s <= hi) & (np.abs(L.s - mu) <= cfg.sigma_window * sig)
            st = np.column_stack([L.s[inside], L.t[inside]])
            tsel = (ts >= lo) & (ts <= hi)
            top = float(np.percentile(th[tsel], 95)) if tsel.sum() else 0.0
            c = Cluster(d, i, f"{d}_c{i + 1:02d}", lo, hi, mu, sig,
                        float(np.median(st[:, 1])) if len(st) else 0.0, mass, peak, int(inside.sum()),
                        top, top < cfg.clod_top)
            c.stems = _stem_gaussians(st, cfg) if len(st) else []
            out.append(c)
        clusters[d] = out
        log(f"  {d}: background {bg:.1f} +- {sd:.1f} pts/bin, threshold {T:.1f} -> "
            f"{len(out)} clusters ({sum(c.clod for c in out)} clods)")
    return clusters, profiles


def cluster_points(run: Run, c: Cluster, cfg: ClusterConfig) -> np.ndarray:
    """Low-layer xy points of a cluster (reference frame)."""
    L = run.low[c.date]
    m = (L.s >= c.lo) & (L.s <= c.hi) & (np.abs(L.s - c.mu) <= cfg.sigma_window * c.sigma)
    return L.xyz[m, :2]
