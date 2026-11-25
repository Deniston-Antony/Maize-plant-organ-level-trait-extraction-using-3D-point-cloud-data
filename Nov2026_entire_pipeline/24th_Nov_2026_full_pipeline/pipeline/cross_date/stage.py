"""Stage 9: cross-date matching of a finished run - clusters -> plants -> stems -> leaves.

Steps 1-4 (01_clusters/): one frame, clusters per date (stem-base density
profile + Gaussians), cluster matching across dates (sequence alignment +
DBSCAN), fine alignment per cluster.
Steps 5-6 (02_plants_stems/): plant check (Hungarian inside the matched
clusters), stem matching (centre lines).
Steps 7-8 (03_leaves/): leaf pieces -> leaves (rule tuned on the 21.06. ground
truth), leaf matching (order-preserving alignment, leaf IDs from the reference date).
Output: <run>/cross_date_matching/ with cross_date_matching_report.pdf.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from ..config import OUTPUT_ROOT, RUN_LEVEL_DIRS
from .config import ClusterConfig


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def default_out(run_dir: Path) -> Path:
    return Path(run_dir) / RUN_LEVEL_DIRS["cross_date"]


def ensure_gt_structure(gt_dir: Path, log=print) -> None:
    """The user's structure code on the ground-truth plants (cached, ~25 s once)."""
    from .gt_structure import GT_NAMES, run as run_gt
    if all((gt_dir / f"{n}.npz").exists() for n in GT_NAMES):
        return
    log("  running the structure code on the ground-truth plants (once, cached in outputs/_gt_structure)")
    run_gt(OUTPUT_ROOT / "_gt_cache", gt_dir, log)


def _steps_1_to_6(run_dir: Path, out: Path, until: str, figures: bool, log, t0: float):
    """Steps 1-6; returns (run, plants, assign) or None when stopped earlier."""
    from .inputs import load_run
    from .clusters import find_clusters
    from .cluster_match import match_all, name_reference_clusters
    from .fine_align import fine_align

    cfg = ClusterConfig()
    log("Step 1  one frame (final alignment from alignment.json, height above ground)")
    run = load_run(run_dir, cfg, log)
    log("Step 2  clusters per date (low-layer density profile + Gaussians)")
    clusters, profiles = find_clusters(run, cfg, log)
    ref = run.reference
    ref_names, reg_cmp = name_reference_clusters([c for c in clusters[ref]], run.register)
    log("Step 3  match clusters to the reference date (A: sequence alignment, B: DBSCAN)")
    rows, links_all, lab = match_all(clusters, run.dates, ref, cfg, ref_names, log)
    log("Step 4  fine alignment per matched cluster (image cross-correlation of the stem-base layer)")
    fine = fine_align(run, clusters, links_all, ref, cfg, ref_names, log)

    o = out / "01_clusters"
    crow = []
    for d in run.dates:
        for c in clusters[d]:
            r = c.row()
            r["name"] = ref_names.get(c.cid, "") if d == ref else ""
            r["dbscan_group"] = lab.get(c.cid, "")
            crow.append(r)
    _write_csv(o / "clusters_per_date.csv", crow)
    _write_csv(o / "stem_gaussians.csv", [{"date": c.date, "cluster": c.cid, "stem": k + 1,
                                           **{kk: round(v, 4) for kk, v in s.items()}}
                                          for d in run.dates for c in clusters[d] for k, s in enumerate(c.stems)])
    _write_csv(o / "cluster_match.csv", rows)
    import numpy as np
    for d, (xyz, hag) in (run.display or {}).items():       # for the app (Step viewer 8)
        np.savez_compressed(o / f"row_points_{d}.npz", xyz=xyz, hag=hag,
                            s=run.frame.st(xyz[:, :2].astype(np.float64))[:, 0].astype(np.float32))
    _write_csv(o / "fine_alignment.csv", fine)
    _write_csv(o / "register_comparison.csv", reg_cmp)

    summary = {"run": str(run_dir), "reference_date": ref, "dates": run.dates, "config": cfg.to_dict(),
               "row_frame": {"origin_local_xy": run.frame.origin.tolist(), "along": run.frame.along.tolist()},
               "profiles": {d: {"background": p.background, "background_sd": p.bg_sd, "threshold": p.threshold}
                            for d, p in profiles.items()},
               "clusters_per_date": {d: len(clusters[d]) for d in run.dates},
               "clods_per_date": {d: sum(c.clod for c in clusters[d]) for d in run.dates},
               "status_counts": {}, "seconds": None}
    for r in rows:
        summary["status_counts"].setdefault(r["date"], {}).setdefault(r["status"], 0)
        summary["status_counts"][r["date"]][r["status"]] += 1
    summary["register_relation_counts"] = {}
    for r in reg_cmp:
        k = r["relation"].split(":")[0]
        summary["register_relation_counts"][k] = summary["register_relation_counts"].get(k, 0) + 1

    if not not figures:
        log("Figures")
        from . import figures
        figures.cluster_figures(run, clusters, profiles, rows, fine, ref_names, cfg, o)

    (o / "summary.json").write_text(json.dumps(summary, indent=2))
    if until == "clusters":
        log(f"done in {time.time() - t0:.1f} s -> {out}")
        return None

    # ---------------------------------------------------------------- steps 5-6
    from .config import PlantConfig
    from .plants import load_plants
    from .plant_check import apply_fine_shifts, check_plants
    from .stem_match import match_stems

    pcfg = PlantConfig()
    o2 = out / "02_plants_stems"
    o2.mkdir(parents=True, exist_ok=True)
    log("Step 5  plant check against the reference plants (Hungarian, inside the matched clusters)")
    plants = {d: load_plants(run_dir, d, run.alignments[d], log=log) for d in run.dates}
    apply_fine_shifts(run, plants, clusters, fine, pcfg)
    prow, assign = check_plants(run, clusters, links_all, plants, cfg, pcfg, log)
    log("Step 6  stem matching (centre lines compared every 1 cm of height)")
    srow = match_stems(run, plants, assign, pcfg, log)
    _write_csv(o2 / "plant_check.csv", prow)
    _write_csv(o2 / "plant_shifts.csv", [{"date": d, "plant_id": pid, "fine_dx_m": round(p.fine[0], 5),
                                          "fine_dy_m": round(p.fine[1], 5), "cluster": getattr(p, "cluster", "") or ""}
                                         for d, ps in plants.items() for pid, p in ps.items()])
    _write_csv(o2 / "stem_match.csv", srow)
    (o2 / "assignment.json").write_text(json.dumps(assign, indent=1))
    (o2 / "settings.json").write_text(json.dumps(pcfg.to_dict(), indent=2))
    if not not figures:
        from . import figures2
        figures2.plant_stem_figures(run, plants, prow, srow, o2, pcfg.stem_flag_cm)
    if until == "stems":
        log(f"done in {time.time() - t0:.1f} s -> {out}")
        return None

    return run, plants, assign


def _reuse_1_to_6(run_dir: Path, out: Path, log):
    """Steps 1-6 were done earlier in this run (stage 6b): read them back."""
    from .inputs import light_run
    from .plants import load_plants
    run = light_run(run_dir, out)
    shifts = {(r["date"], r["plant_id"]): r for r in csv.DictReader(open(out / "02_plants_stems" / "plant_shifts.csv"))}
    plants = {d: load_plants(run_dir, d, run.alignments[d], log=log) for d in run.dates}
    for d, ps in plants.items():
        for pid, p in ps.items():
            r = shifts.get((d, pid))
            if r:
                p.shift(float(r["fine_dx_m"]), float(r["fine_dy_m"]))
                p.cluster = r.get("cluster") or None
    assign = json.loads((out / "02_plants_stems" / "assignment.json").read_text())
    log("  steps 1-6 reused from stage 6b (02_plants_stems)")
    return run, plants, assign


def run_cross_date(run_dir: Path, out: Path | None = None, until: str = "all", retune: bool = False,
                   figures: bool = True, log=print, reuse: bool = False) -> Path:
    """reuse: if steps 1-6 already ran in this run (stage 6b), only do steps 7-8 + report."""
    run_dir = Path(run_dir)
    out = Path(out or default_out(run_dir))
    (out / "01_clusters").mkdir(parents=True, exist_ok=True)
    gt_dir = OUTPUT_ROOT / "_gt_structure"
    log(f"cross-date matching: {run_dir.name} -> {out}")
    t0 = time.time()
    if reuse and until == "all" and (out / "02_plants_stems" / "assignment.json").exists():
        run, plants, assign = _reuse_1_to_6(run_dir, out, log)
    else:
        res = _steps_1_to_6(run_dir, out, until, figures, log, t0)
        if res is None:
            return out
        run, plants, assign = res

    # ---------------------------------------------------------------- steps 7-8
    from .leaf_tuning import gt_cases
    from .leaf_pieces import tune, LeafRule
    from .leaf_match import build_leaves, match_leaves

    o3 = out / "03_leaves"
    o3.mkdir(parents=True, exist_ok=True)
    log("Step 7  leaf pieces -> leaves (rule tuned on the 21.06. ground truth)")
    rule_file = o3 / "leaf_rule.json"
    if rule_file.exists() and not retune:
        rule = LeafRule(**json.loads(rule_file.read_text())["rule"])
        log(f"  rule from {rule_file.name}: {rule}")
    else:
        ensure_gt_structure(gt_dir, log)
        cases = gt_cases(gt_dir, run_dir, log=log)
        if not cases:
            raise RuntimeError("no ground-truth structure results for the leaf rule")
        rule, trows = tune(cases, log)
        _write_csv(o3 / "leaf_rule_tuning.csv", trows)
        from .leaf_pieces import score_case
        raw = LeafRule(0.0, 0.0, 0.0, None, None)
        res = [{"case": c["name"], "n_true": score_case(c, rule)["n_true"],
                "before": score_case(c, raw), "after": score_case(c, rule)} for c in cases]
        rule_file.write_text(json.dumps({"rule": rule.to_dict(), "cases": [c["name"] for c in cases],
                                         "case_results": res}, indent=2, default=float))
    leaves = build_leaves(plants, rule)
    log("Step 8  leaf matching (order-preserving alignment; IDs from 29.06.)")
    lrow, tracks, pairs = match_leaves(run, plants, leaves, assign, log)
    _write_csv(o3 / "leaf_match.csv", lrow)
    _write_csv(o3 / "leaf_tracks.csv", tracks)
    _write_csv(o3 / "leaves_per_plant.csv", [l.row() for d in leaves for ls in leaves[d].values() for l in ls])
    if not not figures:
        from . import figures3
        figures3.leaf_figures(run, plants, leaves, lrow, tracks, o3)
        from .report import build
        log(f"report: {build(out)}")
    log(f"done in {time.time() - t0:.1f} s -> {out}")
    return out

