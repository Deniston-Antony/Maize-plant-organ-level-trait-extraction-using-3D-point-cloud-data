"""Full pipeline run (called by the app and the command line).

Stages: 1 load, 2 soil separation (+ hidden ground from an earlier date),
3 alignment of all dates onto the reference date, 4 stems / clod filter /
plant register / groups, 5 plant separation, 6 structure and traits per
plant, 7 tracking and availability, 8 growth curves and stage estimate,
optional ground-truth check, report.html.

Outputs
  <run>/<date>/01_soil/   plants.npz, <date>_plants.ply, dtm.npz, soil_summary.json, soil_check.png
  <run>/alignment/        alignment.json, alignment_summary.csv, align_<date>.png
  <run>/run_config.json
"""

from __future__ import annotations

import csv
import gc
import json
import time
from pathlib import Path

import numpy as np

from .align import AlignConfig, align_date, save_alignments
from .config import REFERENCE_DATE, RunPaths, date_from_name
from .progress import Progress
from .qc_figures import alignment_figure, plants_figure, soil_figure, stems_figure
from .separate import SeparateConfig, separate
from .plants_stage import run_structures, save_plant
from .tracking import write_tracking
from .report import write_report
from .stems import (StemConfig, find_stems, fit_row_centre, guided_search, make_groups, plant_order_in_group,
                    stem_vote_shift)
from .align import row_frame
from .config import PROJECT_ORIGIN, plant_id
from scipy.spatial import cKDTree

from .soil import DTM, SoilConfig, borrow_ground, load_ground_raster, load_plants, load_scan, save_soil, separate_soil


STEPS = ("soil", "alignment", "stems", "plants", "all")


def _scan_points(path: Path) -> float:
    import laspy
    with laspy.open(str(path)) as f:
        return f.header.point_count / 1e6


def run(scans: list[tuple[str, str]], root: str | Path, reference_date: str = REFERENCE_DATE,
        gt_items=None, log=print, progress=None, soil_cfg: SoilConfig | None = None,
        align_cfg: AlignConfig | None = None, reuse: bool = True, stem_cfg: StemConfig | None = None,
        sep_cfg: SeparateConfig | None = None, stop_after: str = "all") -> dict:
    """scans: list of (path, date or ''); date is read from the file name when empty.
    stop_after: 'soil', 'alignment', 'stems', 'plants' or 'all' - stop after that step so its
    output can be checked; a later run reuses the soil results."""
    if stop_after not in STEPS:
        raise ValueError(f"stop_after must be one of {STEPS}")
    soil_cfg, align_cfg, stem_cfg = soil_cfg or SoilConfig(), align_cfg or AlignConfig(), stem_cfg or StemConfig()
    sep_cfg = sep_cfg or SeparateConfig()
    run_p = RunPaths(Path(root))
    items = []
    for path, date in scans:
        d = date or date_from_name(Path(path).name)
        if not d:
            raise ValueError(f"No date (YYMMDD) in the file name: {path}")
        items.append((Path(path), d))
    items.sort(key=lambda t: t[1])
    dates = [d for _, d in items]
    if reference_date not in dates:
        log(f"! reference date {reference_date} not among the scans; using {dates[len(dates) // 2]}")
        reference_date = dates[len(dates) // 2]
    mpts = sum(_scan_points(p) for p, _ in items)
    prog = Progress({"soil": 0.25 * len(items), "alignment": 0.05 * len(items), "stems": 0.05 * len(items),
                     "plants": 0.6 * len(items), "tracking": 0.05 * len(items)}, progress,
                    n_points_millions=mpts)
    t0 = time.time()
    run_p.root.mkdir(parents=True, exist_ok=True)
    (run_p.root / "run_config.json").write_text(json.dumps({
        "scans": [[str(p), d] for p, d in items], "reference_date": reference_date,
        "soil": soil_cfg.to_dict(), "alignment": align_cfg.to_dict(), "stems": stem_cfg.to_dict(), "separation": sep_cfg.to_dict(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    log(f"Run folder: {run_p.root}\nDates: {', '.join(dates)} (reference {reference_date}), {mpts:.0f} M points")

    # ---------------- stage 1 + 2: load and soil, one date at a time (memory)
    soil_rows = []
    for k, (path, d) in enumerate(items):
        out = run_p.stage_dir(d, "soil")
        if reuse and (out / "plants.npz").exists() and (out / "soil_summary.json").exists():
            st = json.loads((out / "soil_summary.json").read_text())
            if st.get("config") == soil_cfg.to_dict() and st.get("source") == str(path):
                log(f"  soil {d}: reusing earlier result")
                soil_rows.append(st)
                prog.update("soil", k + 1, len(items), f"soil {d} (cached)")
                continue
        prog.update("soil", k + 0.1, len(items), f"loading {d}")
        scan = load_scan(path, d)
        prog.update("soil", k + 0.4, len(items), f"soil {d}")
        res = separate_soil(scan, soil_cfg, log)
        save_soil(out, scan, res, soil_cfg)
        soil_figure(out / "soil_check.png", d, scan.xyz, res.hag, res.plant_idx, res.dtm, res.stats)
        soil_rows.append(res.stats)
        del scan, res
        gc.collect()
        prog.update("soil", k + 1, len(items), f"soil {d} done")

    def soil_csv():
        with open(run_p.root / "soil_summary.csv", "w", newline="") as f:
            keys = [k for k in soil_rows[0] if not isinstance(soil_rows[0][k], dict)]
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(soil_rows)

    def finish(step: str) -> dict:
        soil_csv()
        total = prog.finish()
        rep = write_report(run_p.root, dates, reference_date, total)
        log(f"\nStopped after step '{step}' ({total / 60:.1f} min). Check it in the Step viewer tab."
            if step != "all" else f"\nPipeline finished in {total / 60:.1f} min. Report: {rep}")
        return {"root": str(run_p.root), "dates": dates, "reference_date": reference_date, "report": str(rep),
                "stopped_after": step}

    if stop_after == "soil":
        return finish("soil")

    # ---------------- stage 3: alignment onto the reference date
    al_dir = run_p.run_dir("alignment")
    ref_out = run_p.stage_dir(reference_date, "soil")
    ref_plants, ref_dtm, ref_ground = load_plants(ref_out), DTM.load(ref_out / "dtm.npz"), load_ground_raster(ref_out)
    # reference first, then outward, so a weak date can be chained via its neighbour
    order = sorted(dates, key=lambda d: (abs(dates.index(d) - dates.index(reference_date)), d))
    done: dict = {}
    cache: dict = {reference_date: (ref_plants, ref_dtm, ref_ground)}

    def data(d):
        if d not in cache:
            o_ = run_p.stage_dir(d, "soil")
            cache[d] = (load_plants(o_), DTM.load(o_ / "dtm.npz"), load_ground_raster(o_))
        return cache[d]

    for k, d in enumerate(order):
        prog.update("alignment", k, len(order), f"aligning {d}")
        pl, dtm, gr = data(d)
        al, aux = align_date(d, reference_date, pl, dtm, ref_plants, ref_dtm, align_cfg, log, gr, ref_ground)
        if al.confidence == "low" and d != reference_date:
            i, r = dates.index(d), dates.index(reference_date)
            nb = dates[i + 1] if i < r else dates[i - 1]
            if nb != reference_date and nb in done:
                npl, ndtm, ngr = data(nb)
                first, _ = align_date(d, nb, pl, dtm, npl, ndtm, align_cfg, log, gr, ngr)
                second = done[nb]
                if first.confidence != "low":
                    from .align import compose
                    al.dx_profile = compose(first, second)
                    al.dz = first.dz + second.dz
                    al.method = f"chained via {nb}: [{first.method}] then [{second.method}]"
                    al.confidence = "medium" if "high" in (first.confidence + second.confidence) else "low"
                    al.notes.append(f"direct match to {reference_date} too weak; aligned to {nb} first")
                    k_ = np.asarray(al.dx_profile)
                    log(f"  align {d} chained via {nb}: shift {np.median(k_[:, 1]) * 100:+.1f} / "
                        f"{np.median(k_[:, 2]) * 100:+.1f} cm, z {al.dz * 100:+.1f} cm; confidence {al.confidence}")
        done[d] = al
        alignment_figure(al_dir / f"align_{d}.png", al, aux)
        # keep memory low: only the reference and the dates still needed as neighbours
        for old_d in list(cache):
            if old_d not in (reference_date, d):
                cache.pop(old_d)
    als = [done[d] for d in dates]
    save_alignments(al_dir / "alignment.json", als, align_cfg)

    # ---------------- stage 2b: ground under a closed canopy from an earlier date
    by = {a.date: a for a in als}
    stats_by = {r["date"]: r for r in soil_rows}
    src_dates = [d for d in dates if stats_by[d]["dtm_cells_under_plants_filled_share"] <= soil_cfg.borrow_src_max_filled
                 and by[d].confidence != "low"]
    for d in (dates if soil_cfg.borrow_hidden_ground else []):   # off by default: keep classify_plants.py result
        st_ = stats_by[d]
        if st_["dtm_cells_under_plants_filled_share"] < soil_cfg.borrow_when_filled or st_.get("ground_borrowed_from"):
            continue
        cands = [s_ for s_ in src_dates if s_ < d]
        if not cands or by[d].confidence == "low":
            log(f"  ground {d}: hidden under the canopy, no earlier well-seen date to borrow from")
            continue
        src_d = cands[-1]
        out, src_out = run_p.stage_dir(d, "soil"), run_p.stage_dir(src_d, "soil")
        dtm_d, dtm_s = DTM.load(out / "dtm.npz"), DTM.load(src_out / "dtm.npz")
        a_d, a_s = by[d], by[src_d]

        def to_src(xy, a_d=a_d, a_s=a_s):
            p3 = np.column_stack([xy, np.zeros(len(xy))])
            r = a_d.apply(p3)                       # this date -> reference
            back = a_s.apply(r) - r                 # source shift at that place
            return (r - back)[:, :2]                # reference -> source (shift is small)
        new_dtm, info = borrow_ground(dtm_d, dtm_s, to_src, a_s.dz - a_d.dz)
        pl = load_plants(out)
        xyz = pl["xyz"].astype(np.float64)
        hag_new = (xyz[:, 2] - new_dtm.height(xyz[:, 0], xyz[:, 1])).astype(np.float32)
        keep = hag_new >= soil_cfg.hag_plant_min
        if "low_kept" in pl and len(pl["low_kept"]) == len(xyz):
            keep |= pl["low_kept"] & (hag_new >= soil_cfg.keep_low_min)
        np.savez(out / "plants.npz", **{k: (v[keep] if (hasattr(v, "__len__") and len(v) == len(xyz)) else v)
                                        for k, v in pl.items() if k != "hag"}, hag=hag_new[keep])
        new_dtm.save(out / "dtm.npz")
        st_.update({"ground_borrowed_from": src_d, **{f"borrow_{k}": v for k, v in info.items()},
                    "n_plant_before_borrow": int(len(xyz)), "n_plant": int(keep.sum()),
                    "plant_points_removed_by_borrow": int((~keep).sum())})
        js = json.loads((out / "soil_summary.json").read_text())
        js.update(st_)
        (out / "soil_summary.json").write_text(json.dumps(js, indent=2))
        log(f"  ground {d}: hidden cells taken from {src_d} ({info['cells_borrowed']:,} cells, ground "
            f"{info['ground_change_median_cm']:+.1f} cm median vs lowest-neighbour fill); "
            f"{int((~keep).sum()):,} points were soil, not plant")
        # check figure with the new ground
        path_d = next(p_ for p_, dd in items if dd == d)
        scan = load_scan(path_d, d)
        hag_all = (scan.xyz[:, 2] - new_dtm.height(scan.xyz[:, 0], scan.xyz[:, 1])).astype(np.float32)
        mask = np.zeros(len(scan.xyz), bool)
        kx = np.round(xyz[keep] * 1000).astype(np.int64)
        sx = np.round(scan.xyz * 1000).astype(np.int64)
        kk = (kx[:, 0] << 42) ^ (kx[:, 1] << 21) ^ kx[:, 2]
        sk = (sx[:, 0] << 42) ^ (sx[:, 1] << 21) ^ sx[:, 2]
        mask[np.isin(sk, kk)] = True
        soil_figure(out / "soil_check.png", d, scan.xyz, hag_all, np.where(mask)[0], new_dtm, st_)
        del scan, hag_all, mask, sx, sk
        gc.collect()
    with open(al_dir / "alignment_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "dx_cm", "dy_cm", "dz_cm", "method", "confidence", "notes"])
        for a in als:
            k = np.asarray(a.dx_profile)
            w.writerow([a.date, round(float(np.median(k[:, 1])) * 100, 1), round(float(np.median(k[:, 2])) * 100, 1),
                        round(a.dz * 100, 1), a.method, a.confidence, "; ".join(a.notes)])
    if stop_after == "alignment":
        return finish("alignment")

    # ---------------- stage 4: stems, clod/weed filter, groups
    prog.update("stems", 0, len(dates), "stems")
    stage4(run_p, dates, reference_date, by, stem_cfg, log, lambda k, t: prog.update("stems", k, len(dates), t))
    save_alignments(al_dir / "alignment.json", [by[d] for d in dates], align_cfg)   # may be refined by stem matching

    if stop_after == "stems":
        return finish("stems")

    # ---------------- stage 5 + 6: plant separation and structure per date
    register = list(csv.DictReader(open(run_p.run_dir("tracking") / "plant_register.csv")))
    per_date: dict = {}
    for k, d in enumerate(dates):
        prog.update("plants", k, len(dates), f"plants {d}")
        per_date[d] = stage5_6(run_p, d, by[d], register, sep_cfg, log,
                               lambda i, n, d=d, k=k: prog.update("plants", k + i / max(n, 1), len(dates),
                                                                  f"plants {d}: {i}/{n}"))

    if stop_after == "plants":
        return finish("plants")

    # ---------------- stage 7 + 8: tracking, availability, growth
    prog.update("tracking", 0, 1, "tracking and growth")
    write_tracking(run_p.root, dates, register, per_date, log)

    # ---------------- ground truth (optional)
    if gt_items:
        prog.update("tracking", 0.5, 1, "ground-truth check")
        from .ground_truth import run_ground_truth_check
        run_ground_truth_check(run_p.root, gt_items, log=log)
    return finish("all")


def stage5_6(run_p: RunPaths, d: str, al, register: list[dict], cfg: SeparateConfig, log, step) -> dict:
    """Separate the plants of one date and compute their structure."""
    O = np.asarray(PROJECT_ORIGIN)
    log(f"  plants {d}:")
    soil_dir, sg_dir = run_p.stage_dir(d, "soil"), run_p.stage_dir(d, "stems_groups")
    pl = load_plants(soil_dir)
    keep = np.load(sg_dir / "keep_points.npy")
    xyz = pl["xyz"][keep].astype(np.float64)
    hag = pl["hag"][keep].astype(np.float64)
    XY = np.array([[float(r["x_utm"]) - O[0], float(r["y_utm"]) - O[1]] for r in register])
    back = al.apply(np.column_stack([XY, np.zeros(len(XY))]))[:, :2] - XY
    xy_d = XY - back                                   # register positions in this date's frame
    labels, info = separate(xyz, hag, xy_d, cfg, log)
    dtm = DTM.load(soil_dir / "dtm.npz")
    # results of an earlier run of this date are replaced completely
    import shutil
    old = run_p.root / d / "03_plants"
    if old.exists():
        shutil.rmtree(old)
    jobs, pres = [], {}
    for i, r in enumerate(register):
        pid = r["plant_id"]
        stem_seen = d in r["dates_seen"].split()
        pres[pid] = {"present": info[i]["present"], "stem_seen": stem_seen, "visibility": info[i]["visibility"],
                     "n_points_separated": info[i]["n_points"],
                     "stem_followed_to_m": info[i]["stem_followed_to_m"], "seed_wide_search": info[i]["seed_wide_search"]}
        if info[i]["present"]:
            z0 = float(dtm.height(xy_d[i:i + 1, 0], xy_d[i:i + 1, 1])[0])
            jobs.append((pid, xyz[labels == i], np.array([xy_d[i, 0], xy_d[i, 1], z0]),
                         run_p.plant_dir(d, pid) / "structure_output"))
    res = run_structures(jobs, log, on_done=step)
    out_rows = {}
    for pid, pts, base, _ in jobs:
        out_rows[pid] = save_plant(run_p, d, pid, res[pid], base, pres[pid])
    for pid, p_ in pres.items():
        if pid not in out_rows:
            out_rows[pid] = {"plant_id": pid, "date": d, "status": "missing", **{f"presence_{k}": v for k, v in p_.items()}}
        else:
            out_rows[pid].update({f"presence_{k}": v for k, v in p_.items()})
    pdir = run_p.stage_dir(d, "plants")
    rows = list(out_rows.values())
    keys = []
    for rw in rows:
        keys += [k for k in rw if k not in keys]
    with open(pdir / "plants_traits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    plants_figure(pdir / "plants_check.png", d, xyz, labels, [r["plant_id"] for r in register], xy_d,
                  [pres[r["plant_id"]]["present"] for r in register])
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    log(f"  plants {d}: {n_ok} with structure, {sum(p_['present'] for p_ in pres.values())} present")
    return out_rows


def stage4(run_p: RunPaths, dates: list[str], ref: str, als: dict, cfg: StemConfig, log, step) -> None:
    """Stems per date, clod/weed filter, groups from the reference date."""
    O = np.asarray(PROJECT_ORIGIN)
    ref_pl = load_plants(run_p.stage_dir(ref, "soil"))
    origin, along = row_frame(ref_pl["xyz"][:, :2].astype(np.float64))
    order = sorted(dates, key=lambda d: (abs(dates.index(d) - dates.index(ref)), d))
    stems_al: dict[str, np.ndarray] = {}
    per_date: dict[str, tuple] = {}
    for k, d in enumerate(order):
        step(k, f"stems {d}")
        log(f"  stems {d}:")
        pl = ref_pl if d == ref else load_plants(run_p.stage_dir(d, "soil"))
        st, aux = find_stems(pl, cfg, log)
        al = als[d]
        xy_d = np.array([[s_["x"], s_["y"]] for s_ in st]) if st else np.zeros((0, 2))
        A = al.apply(np.column_stack([xy_d, np.zeros(len(xy_d))]))[:, :2] if len(xy_d) else xy_d
        if d != ref:
            # weak alignment (01.06): match stem positions to the neighbouring date
            i, r = dates.index(d), dates.index(ref)
            nb = dates[i + 1] if i < r else dates[i - 1]
            if al.confidence == "low" and nb in stems_al and len(A) >= 5:
                sh, n = stem_vote_shift(A, stems_al[nb])
                if n >= max(5, 0.3 * len(A)):
                    al.dx_profile = [[kk[0], kk[1] + float(sh[0]), kk[2] + float(sh[1])] for kk in al.dx_profile]
                    al.method += f"; refined by matching {n} of {len(A)} stems to {nb} ({sh[0] * 100:+.1f} / {sh[1] * 100:+.1f} cm)"
                    al.confidence = "medium"
                    al.notes.append(f"stem positions matched to {nb}")
                    A = A + sh
                    log(f"  alignment {d} refined from stems: {n} of {len(A)} stems match {nb} after "
                        f"{sh[0] * 100:+.1f} / {sh[1] * 100:+.1f} cm")
                else:
                    log(f"  alignment {d}: stem matching inconclusive ({n} of {len(A)} stems)")
            # stems hidden by the canopy: look again at the reference positions
            if ref in stems_al and len(stems_al[ref]):
                R = stems_al[ref]
                back = al.apply(np.column_stack([R, np.zeros(len(R))]))[:, :2] - R
                g = guided_search(pl, R - back, xy_d, cfg, log=log)
                if g:
                    st += g
                    xy_g = np.array([[s_["x"], s_["y"]] for s_ in g])
                    A = np.vstack([A, al.apply(np.column_stack([xy_g, np.zeros(len(xy_g))]))[:, :2]])
        stems_al[d] = A
        per_date[d] = (st, aux, A)

    # row centre line from the reference stems; off-row stems (weeds, leaf tips touching the soil) are not grouped
    st_r, _, A_r_all = per_date[ref]
    w_r = np.array([s_["strength"] for s_ in st_r])
    off_fn, row_knots = fit_row_centre(A_r_all, w_r, origin, along, max_off=cfg.row_max_offset)
    for d in dates:
        st, _, A = per_date[d]
        if len(A):
            o_ = off_fn(A)
            for s_, ov in zip(st, o_):
                s_["off_row_m"] = float(ov)
                s_["on_row"] = bool(ov <= cfg.row_max_offset)
    on_r = np.array([s_["on_row"] for s_ in st_r], bool)
    idx_on = np.where(on_r)[0]
    A_r = A_r_all[idx_on]
    log(f"  row centre line: {on_r.sum()} of {len(on_r)} reference stems within {cfg.row_max_offset * 100:.0f} cm")

    # plant register: on-row stems of all dates pooled in the reference frame;
    # a plant = stems within `pool_tol` on at least `min_dates_seen` dates
    from scipy.cluster.hierarchy import fcluster, linkage
    P, owner = [], []
    for d in dates:
        st, _, A = per_date[d]
        for j, s_ in enumerate(st):
            if s_.get("on_row", True):
                P.append(A[j])
                owner.append((d, j))
    P = np.asarray(P)
    lab = fcluster(linkage(P, "average"), cfg.pool_tol, "distance") if len(P) > 1 else np.ones(len(P), int)
    reg = []
    for k in np.unique(lab):
        m = np.where(lab == k)[0]
        seen = sorted({owner[i][0] for i in m})
        if len(seen) >= min(cfg.min_dates_seen, len(dates)):
            reg.append({"xy": P[m].mean(axis=0), "members": [owner[i] for i in m], "seen": seen})
    C = np.array([r["xy"] for r in reg]) if reg else np.zeros((0, 2))
    gid, gap = make_groups(C, origin, along, cfg)
    n_g = int(gid.max()) + 1 if len(gid) else 0
    groups_xy, member_name = {}, {}
    for g in range(n_g):
        idx = np.where(gid == g)[0]
        for m_, j in enumerate(idx[plant_order_in_group(C[idx], origin, along)], start=1):
            reg[j]["name"], reg[j]["group"] = plant_id(g + 1, m_), g
            for own in reg[j]["members"]:
                member_name[own] = (reg[j]["name"], g)
        groups_xy[f"group{g + 1:02d}"] = tuple(C[idx].mean(axis=0))
    sizes = np.bincount(gid, minlength=n_g) if n_g else np.zeros(0, int)
    log(f"  plant register: {len(reg)} plants seen on >= {cfg.min_dates_seen} dates, {n_g} groups; "
        f"plants per group: {dict((int(k), int(v)) for k, v in zip(*np.unique(sizes, return_counts=True)))}")
    s_reg = (C - origin) @ along if len(C) else np.zeros(0)
    g_lo = {g: s_reg[gid == g].min() for g in range(n_g)}
    g_hi = {g: s_reg[gid == g].max() for g in range(n_g)}
    ref_tree = cKDTree(C) if len(C) else None

    # register and availability table (run level)
    trk = run_p.run_dir("tracking")
    with open(trk / "plant_register.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plant_id", "group", "x_utm", "y_utm", "along_row_m", "n_dates_seen", "dates_seen"])
        for r_ in sorted(reg, key=lambda r_: r_["name"]):
            w.writerow([r_["name"], f"group{r_['group'] + 1:02d}", round(r_["xy"][0] + O[0], 3), round(r_["xy"][1] + O[1], 3),
                        round(float((r_["xy"] - origin) @ along), 3), len(r_["seen"]), " ".join(r_["seen"])])
    with open(trk / "availability.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plant_id", "group", *dates, "missing_on"])
        for r_ in sorted(reg, key=lambda r_: r_["name"]):
            how = {}
            for d_, j in r_["members"]:
                guided = per_date[d_][0][j].get("guided", False)
                how[d_] = "recovered" if guided and how.get(d_) != "found" else "found"
            w.writerow([r_["name"], f"group{r_['group'] + 1:02d}", *[how.get(d_, "missing") for d_ in dates],
                        " ".join(d_ for d_ in dates if d_ not in how)])
    with open(run_p.root / "groups_reference.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n_plants", "x_utm", "y_utm", "along_row_m", "flag"])
        for g in range(n_g):
            nm = f"group{g + 1:02d}"
            flag = "single plant" if sizes[g] == 1 else ("6+ plants: maybe two groups" if sizes[g] >= 6 else "")
            w.writerow([nm, int(sizes[g]), round(groups_xy[nm][0] + O[0], 3), round(groups_xy[nm][1] + O[1], 3),
                        round(float((np.array(groups_xy[nm]) - origin) @ along), 3), flag])

    for d in dates:
        st, aux, A = per_date[d]
        out = run_p.stage_dir(d, "stems_groups")
        s_al = (A - origin) @ along if len(A) else np.zeros(0)
        rows = []
        dtm = DTM.load(run_p.stage_dir(d, "soil") / "dtm.npz")
        for j, s_ in enumerate(st):
            if not s_.get("on_row", True):
                g, nm = -1, ""
            elif (d, j) in member_name:
                nm, g = member_name[(d, j)]
            else:
                inside = [g for g in range(n_g) if g_lo[g] - 0.06 <= s_al[j] <= g_hi[g] + 0.06]
                g = min(inside, key=lambda g: abs(s_al[j] - (g_lo[g] + g_hi[g]) / 2)) if inside else -1
                nm = ""
            s_["group"] = g
            dist_ref = float(ref_tree.query(A[j])[0]) if ref_tree is not None else float("nan")
            z = float(dtm.height(np.array([s_["x"]]), np.array([s_["y"]]))[0])
            rows.append({"date": d, "stem_index": j,
                         "group": f"group{g + 1:02d}" if g >= 0 else ("off_row" if not s_.get("on_row", True) else "outside_groups"),
                         "plant_id": nm, "x_utm": s_["x"] + O[0], "y_utm": s_["y"] + O[1], "z_ground_utm": z + O[2],
                         "x_aligned": A[j, 0] + O[0], "y_aligned": A[j, 1] + O[1], "along_row_m": float(s_al[j]),
                         "dist_to_nearest_reference_stem_m": dist_ref, "off_row_m": s_.get("off_row_m", float("nan")), "found_by": "guided" if s_.get("guided") else "column",
                         "column_frac": s_["column_frac"], "column_frac_tall": s_.get("column_frac_tall", float("nan")),
                         "column_spread_m": s_["column_spread_m"], "plant_height_local_m": s_.get("blob_height_m", float("nan"))})
        with open(out / "stems.csv", "w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        np.save(out / "keep_points.npy", aux["keep_points"])
        stats = {**aux["stats"], "n_stems_total": len(st), "n_guided": int(sum(1 for s_ in st if s_.get("guided"))),
                 "n_outside_groups": int(sum(1 for s_ in st if s_["group"] < 0 and s_.get("on_row", True))),
                 "n_off_row": int(sum(1 for s_ in st if not s_.get("on_row", True)))}
        stats.update({"n_named": int(sum(1 for j in range(len(st)) if (d, j) in member_name)),
                      "n_groups": n_g, "group_gap_m": gap})
        (out / "stems_summary.json").write_text(json.dumps(stats, indent=2))
        pl = ref_pl if d == ref else load_plants(run_p.stage_dir(d, "soil"))
        # figure in the reference frame so the group names line up on every date
        P = als[d].apply(pl["xyz"].astype(np.float64))
        st_al = [{**s_, "x": float(A[j, 0]), "y": float(A[j, 1])} for j, s_ in enumerate(st)]
        stems_figure(out / "stems_check.png", d, P, aux["keep_points"], st_al, groups_xy,
                     f"Aligned to {ref}.")
