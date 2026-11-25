"""Run report: one self-contained HTML page at <run>/report.html linking the
tables and check figures of every stage."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from .config import pretty_date
from .names import column, nice


def _table(path: Path, max_rows: int = 200, cols: list[str] | None = None, tol: dict | None = None) -> str:
    if not path.exists():
        return f"<p class='muted'>{html.escape(path.name)} not written.</p>"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "<p class='muted'>empty</p>"
    cols = [c for c in (cols or list(rows[0])) if c in rows[0]]

    def cell(v):
        try:
            x = float(v)
            return f"{x:.3f}" if abs(x) < 1e4 and x != int(x) else html.escape(v)
        except (TypeError, ValueError):
            cls = " class='bad'" if v == "missing" else ""
            return f"<span{cls}>{html.escape(nice(v))}</span>"
    head = "".join(f"<th>{html.escape(column(c))}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{cell(r.get(c, ''))}</td>" for c in cols) + "</tr>" for r in rows[:max_rows])
    if tol:
        body = ("<tr class='tol'>" + "".join(f"<td>{html.escape(tol.get(c, ''))}</td>" for c in cols) + "</tr>") + body
    more = f"<p class='muted'>{len(rows) - max_rows} more rows in {path.name}</p>" if len(rows) > max_rows else ""
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{more}"


def _img(root: Path, rel: str, caption: str) -> str:
    if not (root / rel).exists():
        return ""
    return f"<figure><a href='{rel}'><img src='{rel}' loading='lazy'></a><figcaption>{html.escape(caption)}</figcaption></figure>"


def _leaf_widths(root: Path, date: str, pid: str) -> tuple[float, float]:
    """(mean, max) of the per-leaf maximum widths of one plant (m), from its traits file."""
    f = root / date / "03_plants" / pid.split("_")[0] / pid / f"{pid}_traits.json"
    try:
        w = [float(lf["leaf_max_width_m"]) for lf in json.loads(f.read_text()).get("leaves", [])
             if lf.get("leaf_max_width_m") is not None]
    except (OSError, ValueError, KeyError):
        w = []
    return (sum(w) / len(w), max(w)) if w else (float("nan"), float("nan"))


def _with_leaf_width(root: Path) -> Path:
    """traits_by_date.csv with mean / max leaf width added where missing (written next to it)."""
    src = root / "tracking" / "traits_by_date.csv"
    out = root / "tracking" / "traits_by_date_report.csv"
    if not src.exists():
        return src
    with open(src) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return src
    for r in rows:
        if r.get("status") == "ok" and not r.get("mean_leaf_width_m"):
            a, b = _leaf_widths(root, r["date"], r["plant_id"])
            r["mean_leaf_width_m"], r["max_leaf_width_m"] = f"{a:.4f}", f"{b:.4f}"
    keys = list(rows[0]) + [k for k in ("mean_leaf_width_m", "max_leaf_width_m") if k not in rows[0]]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return out


def _f(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def summary_rows(root: Path, dates: list[str]) -> list[dict]:
    """One row per date: plant clusters, plant instances, total leaf area, mean height / stem diameter / leaf angle / leaf width."""
    src = _with_leaf_width(root)
    rows = []
    if src.exists():
        with open(src) as f:
            rows = [r for r in csv.DictReader(f) if r.get("status") == "ok"]
    out = []
    for d in dates:
        rr = [r for r in rows if r["date"] == d]

        def mean(k, sc=1.0):
            v = [_f(r.get(k)) for r in rr]
            v = [x for x in v if x is not None]
            return round(sum(v) / len(v) * sc, 2) if v else ""
        area = [_f(r.get("total_leaf_area_m2")) for r in rr]
        out.append({"date": pretty_date(d), "groups": len({r["plant_id"].split("_")[0] for r in rr}),
                    "plants": len(rr), "leaves": int(sum(_f(r.get("n_leaves")) or 0 for r in rr)),
                    "total_leaf_area_cm2": round(sum(a for a in area if a is not None) * 1e4, 0),
                    "mean_plant_height_cm": mean("plant_height_m", 100),
                    "mean_stem_diameter_mm": mean("stem_diameter_mm"),
                    "mean_leaf_angle_to_stem_deg": mean("mean_leaf_angle_to_stem_deg"),
                    "mean_leaf_width_cm": mean("mean_leaf_width_m", 100)})
    return out


def _summary_table(root: Path, dates: list[str]) -> str:
    rows = summary_rows(root, dates)
    if not rows:
        return "<p class='muted'>No plants measured yet.</p>"
    heads = {"date": "Date", "groups": "Plant clusters", "plants": "Plant instances", "leaves": "Leaves (all plant instances)",
             "total_leaf_area_cm2": "Total leaf area (cm²)", "mean_plant_height_cm": "Mean plant height (cm)",
             "mean_stem_diameter_mm": "Mean stem diameter (mm)",
             "mean_leaf_angle_to_stem_deg": "Mean leaf angle at the junction (°)",
             "mean_leaf_width_cm": "Mean leaf width (cm)"}
    from .tolerance import trait_tolerance
    t, src = trait_tolerance(root)
    pm = lambda v, f, n, u="": f"\u00b1{v * f:.{n}f}{u}"                     # noqa: E731
    tol = {"date": "deviation / error (\u00b1)", "groups": "", "plants": "",
           "leaves": pm(t["n_leaves"], 1, 1, " per plant instance"),
           "total_leaf_area_cm2": pm(t["total_leaf_area_m2"], 1e4, 0, " per plant instance"),
           "mean_plant_height_cm": pm(t["plant_height_m"], 100, 1), "mean_stem_diameter_mm": pm(t["stem_diameter_mm"], 1, 1),
           "mean_leaf_angle_to_stem_deg": pm(t["mean_leaf_angle_to_stem_deg"], 1, 0),
           "mean_leaf_width_cm": pm(t["mean_leaf_width_m"], 100, 1)}
    head = "".join(f"<th>{h}</th>" for h in heads.values())
    body = "<tr class='tol'>" + "".join(f"<td>{html.escape(tol[k])}</td>" for k in heads) + "</tr>"
    body += "".join("<tr>" + "".join(f"<td>{html.escape(str(r[k]))}</td>" for k in heads) + "</tr>" for r in rows)
    return (f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
            f"<p class='muted'>Deviation / error = expected measurement error of one plant instance's value, from the {html.escape(src)}.</p>")


def _alignment_table(root: Path) -> str:
    from .tolerance import alignment_tolerance
    f = root / "alignment" / "alignment_summary.csv"
    if not f.exists():
        return "<p class='muted'>alignment_summary.csv not written.</p>"
    tol = alignment_tolerance(root)
    heads = ("Date", "Shift x cm", "Shift y cm", "Shift z cm", "Deviation / error x/y (\u00b1 cm)", "Deviation / error z (\u00b1 cm)",
             "Confidence", "Method", "Notes")
    body = ""
    for r in csv.DictReader(open(f)):
        t = tol.get(r["date"], {})
        xy, z = t.get("xy_cm", float("nan")), t.get("z_cm", float("nan"))
        xy_s = f"{xy:.1f}" if xy == xy else f"not known ({t.get('n_sections', 0)} section)"
        z_s = f"{z:.1f}" if z == z else "not known"
        vals = (r["date"], r["dx_cm"], r["dy_cm"], r["dz_cm"], xy_s, z_s, r["confidence"], r["method"], r["notes"])
        body += "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in vals) + "</tr>"
    head = "".join(f"<th>{h}</th>" for h in heads)
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def write_report(run_root: Path, dates: list[str], reference: str, timing_s: float | None = None) -> Path:
    root = Path(run_root)
    cfg = json.loads((root / "run_config.json").read_text()) if (root / "run_config.json").exists() else {}
    parts = [f"<h1>Maize field run {html.escape(root.name)}</h1>",
             f"<p>Dates: {', '.join(pretty_date(d) for d in dates)} &middot; reference (names): {pretty_date(reference)} "
             + (f"&middot; run time {timing_s / 60:.1f} min " if timing_s else "")
             + "&middot; soil and plant structure: your code (classify_plants.py, copy_of_New_optimized_plant_structure.py)</p>",
             "<h2>Summary</h2><p>Plants with a measured structure on each date. Leaf angle = angle between the stem "
             "and the start of the leaf at its junction. Leaf width = widest part of each leaf, averaged.</p>",
             _summary_table(root, dates)]
    parts.append("<h2>Plant instances and availability</h2><p>Plant instances are found from stems pooled over all aligned dates "
                 "(seen on at least 2 dates). 'present (stem hidden)' = the plant's points were found but its stem "
                 "base was covered by leaves. 'not visible (covered by canopy)' = only canopy points high above the "
                 "position: the plant may be there but cannot be measured. 'missing' = no points at the position.</p>")
    parts.append(_table(root / "tracking" / "availability.csv"))
    parts.append("<h2>Plant clusters</h2>" + _table(root / "groups_reference.csv"))
    parts.append("<h2>Traits per plant instance and date</h2>")
    from .tolerance import alignment_tolerance, fmt_tol, trait_tolerance
    ttol, tsrc = trait_tolerance(root)
    tol_row = {c: fmt_tol(c, v) for c, v in ttol.items()}
    tol_row["plant_id"] = "deviation / error (\u00b1)"
    parts.append(f"<p class='muted'>First row (italic) = deviation / error = expected measurement error per plant instance, from the {html.escape(tsrc)}.</p>")
    parts.append(_table(_with_leaf_width(root), 400, tol=tol_row, cols=
                        ["plant_id", "date", "present", "stem_status", "plant_height_m", "stem_length_m",
                         "stem_diameter_mm", "n_leaves", "total_leaf_area_m2", "mean_leaf_length_m",
                         "mean_leaf_width_m", "max_leaf_width_m",
                         "mean_leaf_angle_to_stem_deg", "v_stage_estimate"]))
    parts.append("<h2>Growth</h2><div class='grid'>" + "".join(
        _img(root, f"growth/{k}.png", k) for k in ("plant_height_m", "stem_length_m", "n_leaves", "total_leaf_area_m2",
                                                    "stem_diameter_mm", "mean_leaf_length_m", "mean_leaf_angle_to_stem_deg"))
        + "</div>")
    cd = root / "cross_date_matching"
    if (cd / "03_leaves" / "leaf_match.csv").exists():
        parts.append("<h2>Cross-date matching (stage 9)</h2><p>Clusters, plants, stems and leaves of every date matched "
                     "to the reference date; leaves carry the same leaf ID on every date. Full report: "
                     "<a href='cross_date_matching/cross_date_matching_report.pdf'>cross_date_matching_report.pdf</a>; "
                     "per-leaf growth: <a href='cross_date_matching/03_leaves/leaf_tracks.csv'>leaf_tracks.csv</a>; "
                     "3D view: <code>view_3d.py &lt;plant&gt;</code>.</p><div class='grid'>"
                     + _img(root, "cross_date_matching/01_clusters/fig2_cluster_matches.png", "cluster matches")
                     + _img(root, "cross_date_matching/02_plants_stems/fig5_plant_check.png", "plant check")
                     + _img(root, "cross_date_matching/03_leaves/fig9_leaf_growth.png", "single leaves across dates")
                     + "</div>")
    if (root / "ground_truth" / "gt_summary.csv").exists():
        parts.append("<h2>Ground-truth check</h2>" + _table(root / "ground_truth" / "gt_summary.csv", 50,
                     ["gt_name", "matched_date", "matched_plant", "match_method", "plant_iou", "stem_iou",
                      "mean_leaf_iou_with_fragments", "n_gt_leaves", "n_pred_leaves", "err_plant_height_m",
                      "err_stem_length_m", "err_stem_radius_m"]))
    parts.append("<h2>Check figures per date</h2>")
    for d in dates:
        parts.append(f"<h3>{pretty_date(d)}</h3><div class='grid'>"
                     + _img(root, f"{d}/01_soil/soil_check.png", "soil")
                     + _img(root, f"alignment/align_{d}.png", "alignment")
                     + _img(root, f"{d}/02_stems_groups/stems_check.png", "stems and plant clusters")
                     + _img(root, f"{d}/03_plants/plants_check.png", "plants") + "</div>")
    parts.append("<h2>Alignment</h2><p>How far each date was moved onto the reference date. Deviation / error x/y = "
                 "scatter of the agreeing 1.5 m row sections around the shift; deviation / error z = scatter of the soil "
                 "height difference over the same sections after the shift.</p>" + _alignment_table(root))
    parts.append("<h2>Settings</h2><pre>" + html.escape(json.dumps(cfg, indent=1)[:20000]) + "</pre>")
    css = """
:root{--bg:#f6f8f5;--fg:#1c2a21;--muted:#5d6c62;--line:#d8e0d9;--accent:#2f6b3a;--bad:#a3322b}
@media (prefers-color-scheme:dark){:root{--bg:#131a15;--fg:#e1e9e3;--muted:#9aab9f;--line:#2c3a31;--accent:#8cc98f;--bad:#ec8a80}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:24px max(16px,4vw)}
h1{font-size:1.7rem;margin:0 0 6px}h2{margin-top:36px;border-bottom:1px solid var(--line);padding-bottom:4px}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:6px;max-height:520px}
table{border-collapse:collapse;font-size:12.5px;width:100%}th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{position:sticky;top:0;background:var(--bg)}td{font-variant-numeric:tabular-nums}
.bad{color:var(--bad);font-weight:600}tr.tol td{font-style:italic;color:var(--accent);background:color-mix(in srgb,var(--accent) 8%,transparent)}.muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px}
figure{margin:0}img{width:100%;border:1px solid var(--line);border-radius:6px}figcaption{color:var(--muted);font-size:12px}
pre{font-size:11px;overflow:auto;max-height:400px;border:1px solid var(--line);padding:8px}"""
    out = root / "report.html"
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                   f"<title>Run report {html.escape(root.name)}</title><style>{css}</style></head><body>{''.join(parts)}</body></html>")
    return out
