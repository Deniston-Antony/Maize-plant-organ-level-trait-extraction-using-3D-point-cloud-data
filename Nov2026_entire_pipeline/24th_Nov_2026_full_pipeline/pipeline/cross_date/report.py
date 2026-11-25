"""Summary report (HTML + PDF) of one cross-date matching output folder.

Usage (from the pipeline folder): python -m pipeline.cross_date.report <run>/cross_date_matching
"""
from __future__ import annotations

import collections
import csv
import json
import shutil
import statistics as st
import subprocess
import sys
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DL = {"230601": "01.06.", "230621": "21.06.", "230629": "29.06.", "230705": "05.07.", "230711": "11.07."}


def _csv(p: Path) -> list[dict]:
    return list(csv.DictReader(open(p))) if p.exists() and p.stat().st_size else []


def _table(head: list, rows: list, cls: str = "") -> str:
    h = "".join(f"<th>{x}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def _fig(src: str, cap: str) -> str:
    return f'<figure><img src="{src}"><figcaption>{cap}</figcaption></figure>'


def build(out: Path) -> Path:
    c1, c2, c3 = out / "01_clusters", out / "02_plants_stems", out / "03_leaves"
    summ = json.loads((c1 / "summary.json").read_text())
    ref = summ["reference_date"]
    dates = summ["dates"]
    others = [d for d in dates if d != ref]
    cm = _csv(c1 / "cluster_match.csv")
    fa = _csv(c1 / "fine_alignment.csv")
    pc = _csv(c2 / "plant_check.csv")
    sm = _csv(c2 / "stem_match.csv")
    lm = _csv(c3 / "leaf_match.csv")
    lp = _csv(c3 / "leaves_per_plant.csv")
    tr = _csv(c3 / "leaf_tracks.csv")
    rule = json.loads((c3 / "leaf_rule.json").read_text())

    # ---- tables
    st_order = ["confirmed", "split", "merged", "missing", "extra", "flagged", "clod"]
    t1 = []
    for d in others:
        c = collections.Counter(r["status"] for r in cm if r["date"] == d)
        acc = [float(r["shift_cm"]) for r in fa if r["date"] == d and r["accepted"] == "True"]
        t1.append([DL[d], summ["clusters_per_date"][d]] + [c.get(k, 0) for k in st_order] +
                  [f"{len(acc)} / {sum(1 for r in fa if r['date'] == d)}", f"{st.median(acc):.1f}" if acc else "–"])
    reg = summ.get("register_relation_counts", {})
    p_order = ["confirmed", "ambiguous", "other name", "swapped", "missing", "not on the reference date"]
    t2 = []
    for d in others:
        c = collections.Counter(r["status"] for r in pc if r["date"] == d)
        if not c:
            continue
        v = [float(r["mean_deviation_cm"]) for r in sm if r["date"] == d and r.get("mean_deviation_cm")]
        g = [float(r["stem_growth_m"]) for r in sm if r["date"] == d and r.get("stem_growth_m")]
        t2.append([DL[d]] + [c.get(k, 0) for k in p_order] +
                  [len(v), f"{st.median(v):.1f}" if v else "–", sum(x > 3 for x in v), f"{100 * st.median(g):+.1f}" if g else "–"])
    t3 = [[r["case"], r["n_true"], r["before"]["n_pred"], r["after"]["n_pred"],
           f"{r['before']['mean_iou']:.2f}", f"{r['after']['mean_iou']:.2f}"] for r in rule.get("case_results", [])]
    l_order = ["matched", "matched via the previous date", "flagged", "new leaf", "not seen on 29.06.", "not found on this date"]
    t4 = []
    for d in others:
        rows = [r for r in lm if r["date"] == d]
        if not rows:
            continue
        c = collections.Counter(r["status"].split(" (")[0] for r in rows)
        nl = [r for r in lp if r["date"] == d]
        t4.append([DL[d], len({r["plant"] for r in nl}), sum(r["tracked"] == "True" for r in nl)] + [c.get(k, 0) for k in l_order])
    ids = collections.defaultdict(set)
    for r in tr:
        if r["leaf_id"].startswith("L"):
            ids[(r["plant"], r["leaf_id"])].add(r["date"])
    n2, n3 = sum(len(v) >= 2 for v in ids.values()), sum(len(v) >= 3 for v in ids.values())
    matched = [r for r in lm if r["status"].startswith("matched")]

    def med(k):
        v = [float(r[k]) for r in matched if r.get(k)]
        return st.median(v) if v else float("nan")
    n_ref_leaves = sum(1 for r in lp if r["date"] == ref and r["tracked"] == "True")
    rr = rule["rule"]
    examples = sorted(p.name for p in c3.glob("fig8_leaf_ids_example*.png"))

    css = """
    body{font-family:-apple-system,Helvetica,Arial,sans-serif;color:#0b0b0b;font-size:10.5pt;line-height:1.45;margin:0}
    main{max-width:180mm;margin:0 auto;padding:10mm 0}
    h1{font-size:19pt;margin:0 0 2mm} h2{font-size:13.5pt;margin:7mm 0 2mm;border-bottom:1px solid #e1e0d9;padding-bottom:1mm}
    h3{font-size:11pt;margin:4mm 0 1mm} .sub{color:#6b6b66;margin-bottom:5mm}
    .box{background:#f4f3ee;border-left:3px solid #2a78d6;padding:3mm 4mm;margin:3mm 0}
    table{border-collapse:collapse;font-size:8.6pt;margin:2mm 0 3mm;width:100%}
    th,td{border-bottom:1px solid #e1e0d9;padding:1.2mm 1.6mm;text-align:right} th{background:#f4f3ee;font-weight:600}
    td:first-child,th:first-child{text-align:left}
    figure{margin:3mm 0;break-inside:avoid;page-break-inside:avoid} img{display:block;max-width:100%;max-height:215mm;margin:0 auto} figcaption{font-size:8.6pt;color:#6b6b66;margin-top:1mm}
    .pb{break-before:page} code{font-size:8.8pt;background:#f4f3ee;padding:0 1mm}
    li{margin-bottom:1mm} h2,h3{break-after:avoid;page-break-after:avoid}
    """
    h = []
    h.append(f"<h1>Cross-date matching: plant clusters, plants, stems and leaves</h1>"
             f"<div class='sub'>Run <code>{Path(summ['run']).name}</code> · reference date {DL[ref]} · "
             f"all geometry, no trained model</div>")
    h.append(f"""<div class='box'><b>In short.</b>
    <ul>
    <li><b>Clusters:</b> {reg.get('same', 0)} of the {summ['clusters_per_date'][ref]} clusters found on {DL[ref]} are exactly a register cluster of the pipeline, found here without using the register. On {', '.join(DL[d] for d in others)}, most {DL[ref]} clusters are confirmed one-to-one by two independent methods. The rest are split, merged or missing.</li>
    <li><b>Fine alignment:</b> after the global alignment, each matched cluster still needs about 1–2 cm.</li>
    <li><b>Plants and stems:</b> where a plant was separated on both dates, the stems agree closely (median deviation 1.6–1.8 cm). The plant names hold for most plants, with two kinds of exception:
      <ul><li>some cases where the geometry prefers a neighbour's name;</li>
      <li>a few plants that seem to have two register entries.</li></ul></li>
    <li><b>Leaves:</b>
      <ul><li>the rule tuned on the 21.06. ground truth cuts the leaf-count error from 3.6 to 1.7 leaves per plant;</li>
      <li>{n_ref_leaves} tracked leaves on {DL[ref]};</li>
      <li>{n2} leaves are followed on at least 2 dates and {n3} on at least 3.</li></ul></li>
    <li><b>The main limit is the input, not the matching.</b> On 05.07. and 11.07., many separated plants have lost leaves to neighbours or carry pieces of them.</li>
    </ul></div>""")

    h.append("<h2>1. What was done</h2><ol>"
             "<li><b>One frame.</b> Every date is moved onto 29.06. with the final shifts in <code>alignment.json</code>, and heights are measured above each date's own ground.</li>"
             "<li><b>Clusters per date</b> (your Gaussian method):"
             "<ul><li>the stem-base layer (3–15 cm above ground), keeping only upright columns after the seedling date;</li>"
             "<li>its density profile along the row;</li>"
             "<li>a background level taken from the gaps between clusters;</li>"
             "<li>one Gaussian per cluster (centre and width), using the points within ±2σ.</li></ul></li>"
             "<li><b>Cluster matching</b> to 29.06., two ways:"
             "<ul><li>A: an order-preserving sequence alignment along the row, which allows splits and merges;</li>"
             "<li>B: DBSCAN on the cluster centres.</li></ul>"
             "If A and B agree, the match is confirmed; if they contradict each other, it is flagged.</li>"
             "<li><b>Fine alignment per cluster</b> by cross-correlating the stem-base images.</li>"
             "<li><b>Plant check</b> inside the matched clusters, using:"
             "<ul><li>the stem base;</li><li>the lean of the stem;</li><li>containment: the earlier plant's lower stem should lie inside the later plant.</li></ul>"
             "Plants are assigned one-to-one with the Hungarian method.</li>"
             "<li><b>Stem matching:</b> the two centre lines are compared every 1 cm of height.</li>"
             "<li><b>Leaf pieces → leaves:</b> a rule chosen on the 21.06. ground truth.</li>"
             "<li><b>Leaf matching:</b> leaves are aligned bottom to top using direction, attachment height (allowing for stem stretch) and base shape. 29.06. gives the leaf IDs. The 11.07. matches are also checked through 05.07.</li></ol>")

    h.append("<h2 class='pb'>2. Clusters (steps 2–4)</h2>")
    h.append(_table(["date", "clusters", "confirmed", "split", "merged", "missing", "extra", "flagged", "clods",
                     "fine shifts accepted", "median fine shift (cm)"], t1))
    h.append(f"<p><b>Register check:</b> {reg.get('same', 0)} clusters correspond one-to-one to a register cluster. "
             f"Another {reg.get('no register plant', 0)} contain no register plant (small groups seen as stems on only one date, or weeds). "
             f"For {reg.get('register plant(s) outside every reference cluster', 0)} register clusters, some or all plants lie outside every 29.06. cluster. "
             "Their stem bases are thin or hidden on 29.06., so they don't pass the upright-column filter.</p>")
    h.append(_fig("01_clusters/fig1_density_profiles.png", "Figure 1. Density profile of the stem-base layer and the Gaussian of each cluster. Red ticks are the pipeline's register plants."))
    h.append(_fig("01_clusters/fig2_cluster_matches.png", "Figure 2. Cluster matches to 29.06. Bars show centre ± 1σ, and grey lines join each cluster to its reference cluster."))
    h.append(_fig("01_clusters/fig3_fine_alignment.png", "Figure 3. Fine shift per matched cluster on top of the global alignment. Shifts above 3 cm are not used and are listed for review."))
    h.append(_fig("01_clusters/fig4_examples_top_view.png", "Figure 4. Top-view examples of the stem-base layer, one per status."))

    h.append("<h2 class='pb'>3. Plants and stems (steps 5–6)</h2>")
    h.append(_table(["date"] + p_order + ["stem pairs", "median stem deviation (cm)", "stems > 3 cm", "median stem growth (cm)"], t2))
    h.append("<p><b>Status meanings:</b></p><ul>"
             "<li><b>other name:</b> each of the two names was separated on only one of the two dates, and the plants are geometrically the same. This is most likely one plant with two register entries.</li>"
             "<li><b>swapped:</b> both names exist on both dates, but the geometry prefers the neighbour.</li></ul>"
             "<p>The stem base taken from the structure code is noisy by 3–5 cm, so containment carries most of the weight. Leaf matching follows this geometric assignment, not the names.</p>")
    h.append(_fig("02_plants_stems/fig5_plant_check.png", "Figure 5. Plant check: every 29.06. plant on every other date."))
    h.append(_fig("02_plants_stems/fig6_stem_match.png", "Figure 6. Stem matching: mean deviation of the centre lines and lean difference."))

    h.append("<h2 class='pb'>4. Leaves (steps 7–8)</h2><h3>Leaf pieces → leaves, tuned on the 21.06. ground truth</h3>")
    h.append(f"<p><b>The rule:</b> a piece is a leaf if it is at least {100 * rr['min_length']:.0f} cm long or at least {1e4 * rr['min_area']:.0f} cm² large, "
             f"and within {100 * (rr['max_stem_gap'] or 0):.0f} cm of its own stem. A smaller piece joins a leaf it touches (within {100 * rr['join_dist']:.0f} cm) "
             f"if it runs in the same direction (within {rr['max_angle']:.0f}°).</p>"
             "<p><b>How it was chosen:</b> 1,441 candidate settings were scored on 7 cases:</p><ul>"
             "<li>the 4 ground-truth clouds;</li>"
             "<li>the 3 pipeline plants that hold a ground-truth plant (RL_plant01 is not separated in this run).</li></ul>"
             "<p>The score is mean leaf IoU minus 0.02 per leaf of count error.</p>")
    h.append(_table(["case", "true leaves", "leaves before", "leaves after", "IoU before", "IoU after"], t3))
    h.append(_fig("03_leaves/fig7_leaf_rule.png", "Figure 7. Leaf count and leaf overlap before and after the rule."))
    h.append("<h3>Leaf matching</h3>")
    h.append(_table(["date", "plants", "tracked leaves"] + l_order, t4))
    h.append(f"<p><b>Typical accepted match:</b> direction difference {med('az_diff_deg'):.0f}°, attachment-height difference {med('height_diff_cm'):.1f} cm "
             f"after the stem stretch (median stretch {med('stretch'):.2f}), base-shape distance {med('shape_dist_cm'):.1f} cm.</p>"
             "<p>Only leaves at least 10 cm long are tracked, plus the top leaf of each plant. Smaller pieces are listed but not tracked.</p>"
             "<p>On 21.06., \"not found\" is mostly expected: leaves that only grew after 21.06. "
             "On 05.07. and 11.07., the high \"new\" and \"not found\" counts mostly come from the plant separation. Many separated plants lost leaves to neighbours or carry pieces of them (figure 8, 11.07.).</p>")
    for k, e in enumerate(examples):
        h.append(_fig(f"03_leaves/{e}", f"Figure 8{'abc'[k]}. A plant on all its dates, with leaves coloured and labelled by leaf ID (top and side views)."))
    h.append(_fig("03_leaves/fig9_leaf_growth.png", "Figure 9. Length of single leaves across dates."))

    h.append("<h2 class='pb'>5. Limits</h2><ul>"
             "<li><b>No ground truth for matching across dates.</b> Ground truth exists for 21.06. only. The leaf matches are checked only by:"
             "<ul><li>the consistency of the chain through 05.07. with the direct match;</li><li>the order and growth constraints;</li><li>the figures.</li></ul></li>"
             "<li><b>The step-7 rule rests on 7 cases from one date.</b> On later dates, larger leaves and more overlap may need other values.</li>"
             "<li><b>Leaf length can drop between dates.</b> The midrib length is that of the leaf's main piece; leaf area adds up all pieces.</li>"
             "<li><b>01.06. joins only the cluster steps,</b> because no plants were separated on it.</li>"
             "<li><b>Separation on 05.07. and 11.07. limits the leaf results</b> more than the matching does.</li></ul>")
    h.append("<h2>6. Files</h2>" + _table(["folder", "content"], [
        ["01_clusters/", "clusters_per_date, cluster_match, fine_alignment, register_comparison, stem_gaussians, figures 1–4"],
        ["02_plants_stems/", "plant_check, stem_match, assignment.json, figures 5–6"],
        ["03_leaves/", "leaf_rule.json (+ tuning table), leaves_per_plant, leaf_match, leaf_tracks (per-leaf growth), "
                       "plants/ (one figure per 29.06. plant), figures 7–9"]]))

    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Cross-date matching</title><style>{css}</style></head><body><main>{''.join(h)}</main></body></html>"
    hp = out / "cross_date_matching_report.html"
    hp.write_text(html)
    pdf = out / "cross_date_matching_report.pdf"
    if Path(CHROME).exists():
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", hp.as_uri()], check=False, capture_output=True, timeout=120)
    return pdf if pdf.exists() else hp


if __name__ == "__main__":
    print(build(Path(sys.argv[1]).resolve()))
