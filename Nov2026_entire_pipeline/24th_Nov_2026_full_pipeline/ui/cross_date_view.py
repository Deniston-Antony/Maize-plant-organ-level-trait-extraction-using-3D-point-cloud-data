"""Step 8 - cross-date matching (stage 9 results).

Two views, chosen in the bar:
  Plant cluster   one 29.06. cluster on every date: all plant points of the
                  cluster's stretch (grey) and the separated plants on top, each
                  plant in the colour of the 29.06. plant it was matched to, so
                  the same plant has the same colour on every date. Dates without
                  separated plants (01.06.) show the points only.
  Plant           one 29.06. plant on every date: stem black, each leaf in the
                  colour of its leaf ID (same ID = same leaf on every date), small
                  untracked leaves dark grey, points not counted light grey.
The first panel overlaps all selected dates (one colour per date). All panels
move together. The table under the views lists the matching result per date.
"""

from __future__ import annotations

import csv
import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np

from pipeline.names import nice, raw
from ui.offscreen_view import CameraLink, CheckMenu, OffscreenView, size_sliders

LEAF_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7b5cd6", "#3fb8c9", "#a8641f",
                "#5a8f29", "#c2410c", "#6b7280", "#b8336a"]
PLANT_COLOURS = LEAF_COLOURS
STEM, SMALL, OTHER, BACK, UNMATCHED = "#111111", "#8a8a84", "#cfcec7", "#d9d8d2", "#55554f"
MODE_CLUSTER, MODE_PLANT = "Plant cluster", "Plant: stem & leaves"


def _rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def pretty(d: str) -> str:
    return f"{d[4:6]}.{d[2:4]}." if len(d) == 6 else d


def _csv(p: Path) -> list[dict]:
    return list(csv.DictReader(open(p))) if p.exists() and p.stat().st_size else []


def _short(n: str) -> str:
    return n.replace("plant_cluster_", "PC").replace("_plant_instance_", "/")


class Panel(OffscreenView):
    def __init__(self, master, title: str):
        super().__init__(master, title)
        self.make_view()
        self.legend, self.message = [], ""

    def draw(self, layers: list, labels=None, legend=None, message=""):
        """layers: (xyz (n,3) metres, colour rgb or per-point rgb array, point size)."""
        import pyvista as pv
        pl = self.new_plotter()
        for xyz, col, size in layers:
            if xyz is None or not len(xyz):
                continue
            P = pv.PolyData(np.asarray(xyz, np.float64))
            if isinstance(col, np.ndarray):
                P["rgb"] = (col * 255).astype(np.uint8)
                pl.add_mesh(P, scalars="rgb", rgb=True, point_size=size, style="points")
            else:
                pl.add_mesh(P, color=col, point_size=size, style="points")
        if labels:
            pts, txt = labels
            if len(pts):
                pl.add_point_labels(np.asarray(pts, np.float64), txt, font_size=12, point_size=1,
                                    shape_opacity=0.7, always_visible=True, text_color="black")
        pl.add_text(self.title, font_size=11)
        self.legend = legend or []
        self.message = message
        self._render()

    def _draw_scale(self, img) -> None:
        super()._draw_scale(img)
        from PIL import ImageDraw
        try:
            from PIL import ImageFont
            font = ImageFont.load_default(size=15)
        except Exception:  # noqa: BLE001
            font = None
        dr = ImageDraw.Draw(img)
        w, h = img.size
        if self.legend:
            labs = [lab for lab, _ in self.legend]
            tw = max(dr.textlength(t, font=font) for t in labs) if hasattr(dr, "textlength") else 9 * max(map(len, labs))
            lh, sq, pad = 20, 13, 7
            bw, bh = int(tw + sq + 3 * pad), lh * len(labs) + pad
            x0, y0 = w - bw - 10, 34
            dr.rectangle([x0, y0, x0 + bw, y0 + bh], fill=(255, 255, 255), outline=(150, 150, 150))
            for i, (lab, col) in enumerate(self.legend):
                y = y0 + pad // 2 + i * lh
                dr.rectangle([x0 + pad, y + 3, x0 + pad + sq, y + 3 + sq], fill=tuple(int(255 * v) for v in col))
                dr.text((x0 + 2 * pad + sq, y), lab, fill=(30, 30, 30), font=font)
        if self.message:
            tw = dr.textlength(self.message, font=font) if hasattr(dr, "textlength") else 9 * len(self.message)
            dr.text(((w - tw) / 2, h / 2), self.message, fill=(140, 140, 140), font=font)


class CrossDateView(ttk.Frame):
    def __init__(self, master, run: Path, dates: list[str], reference: str, log=print):
        super().__init__(master)
        self.run, self.dates, self.ref, self.log = Path(run), list(dates), reference, log
        self.cd = self.run / "cross_date_matching"
        if not (self.cd / "03_leaves" / "leaf_tracks.csv").exists():
            self._no_results()
            return
        self._load_tables()
        self._build()
        self.update_views()

    # ------------------------------------------------------------------ data
    def _no_results(self):
        f = ttk.Frame(self)
        f.pack(anchor="w", padx=8, pady=8)
        ttk.Label(f, text="No cross-date matching for this run yet (stage 9).").pack(anchor="w")
        ttk.Button(f, text="Run stage 9 now (about 1-2 min)", command=self._rerun).pack(anchor="w", pady=6)

    def _rerun(self):
        def work():
            from pipeline.cross_date.stage import run_cross_date
            run_cross_date(self.run, log=lambda m: self.after(0, self.log, m))
        def done():
            self.log("Stage 9 finished - reopen step 8 to see the new results.")
        self.log("Stage 9 (cross-date matching) started ...")
        q: queue.Queue = queue.Queue()
        threading.Thread(target=lambda: (work(), q.put(True)), daemon=True).start()

        def poll():
            if q.empty():
                self.after(500, poll)
            else:
                done()
        self.after(500, poll)

    def _load_tables(self):
        c1, c2, c3 = self.cd / "01_clusters", self.cd / "02_plants_stems", self.cd / "03_leaves"
        summ = json.loads((c1 / "summary.json").read_text())
        self.ref = summ["reference_date"]
        self.dates = summ["dates"]
        self.clusters = {r["cluster"]: r for r in _csv(c1 / "clusters_per_date.csv")}
        self.cmatch = _csv(c1 / "cluster_match.csv")
        self.fine = _csv(c1 / "fine_alignment.csv")
        self.pcheck = _csv(c2 / "plant_check.csv")
        self.smatch = _csv(c2 / "stem_match.csv")
        self.assign = json.loads((c2 / "assignment.json").read_text())
        self.shifts = {(r["date"], r["plant_id"]): r for r in _csv(c2 / "plant_shifts.csv")}
        self.lmatch = _csv(c3 / "leaf_match.csv")
        self.tracks = _csv(c3 / "leaf_tracks.csv")
        ref_cl = [r for r in self.clusters.values() if r["date"] == self.ref and r["clod"] != "True"]
        ref_cl.sort(key=lambda r: float(r["centre_m"]))
        self.ref_clusters = [(r["cluster"], r["name"] or r["cluster"]) for r in ref_cl]
        self.ref_plants = sorted({r["plant"] for r in self.tracks})
        from pipeline.align import load_alignments, Alignment
        self.als = load_alignments(self.run / "alignment" / "alignment.json")
        self.als.setdefault(self.ref, Alignment(self.ref, self.ref, [0.0, 0.0], [0.0, 1.0]))
        self._cache = {}

    # ------------------------------------------------------------------ layout
    def _build(self):
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        self.mode = tk.StringVar(value=MODE_CLUSTER)
        for m in (MODE_CLUSTER, MODE_PLANT):
            ttk.Radiobutton(bar, text=m, value=m, variable=self.mode, command=self._mode_changed).pack(side="left")
        ttk.Label(bar, text="   ").pack(side="left")
        self.sel = tk.StringVar()
        self.sel_cb = ttk.Combobox(bar, textvariable=self.sel, width=40, state="readonly")
        self.sel_cb.pack(side="left")
        self.sel_cb.bind("<<ComboboxSelected>>", lambda _: self.update_views())
        ttk.Button(bar, text="◀", width=2, command=lambda: self._step(-1)).pack(side="left")
        ttk.Button(bar, text="▶", width=2, command=lambda: self._step(1)).pack(side="left")
        self.dvars = {d: tk.BooleanVar(value=True) for d in self.dates}
        CheckMenu(bar, "Dates", [(pretty(d), self.dvars[d]) for d in self.dates], self.update_views).pack(
            side="left", padx=(8, 0))
        self.show_labels, self.show_back = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        CheckMenu(bar, "Show", [("leaf ID labels", self.show_labels), ("cluster points (grey)", self.show_back)],
                  self.update_views).pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="Open report (PDF)", command=self._open_pdf).pack(side="right")
        ttk.Button(bar, text="Re-run stage 9", command=self._rerun).pack(side="right", padx=4)
        ttk.Button(bar, text="Reset view", command=lambda: self.panels and self.panels[0]._reset()).pack(side="right")
        size_sliders(bar, lambda: [p._apply_point_size() for p in self.panels])
        self.info = ttk.Label(self, foreground="#777")
        self.info.pack(anchor="w")
        pw = ttk.PanedWindow(self, orient="vertical")
        pw.pack(fill="both", expand=True)
        self.grid_frame = ttk.Frame(pw)
        tf = ttk.Frame(pw)
        pw.add(self.grid_frame, weight=5)
        pw.add(tf, weight=1)
        self.table = ttk.Treeview(tf, show="headings", height=6)
        ys = ttk.Scrollbar(tf, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=ys.set)
        self.table.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")
        self.table.tag_configure("flag", foreground="#b4441a")
        self.panels: list[Panel] = []
        self._job = 0
        self._mode_changed(update=False)

    def _mode_changed(self, update=True):
        if self.mode.get() == MODE_CLUSTER:
            vals = [f"{nm}  ({cid})" if nm != cid else cid for cid, nm in self.ref_clusters]
        else:
            vals = self.ref_plants
        self.sel_cb.configure(values=vals)
        if vals:
            self.sel.set(vals[0])
        if update:
            self.update_views()

    def _step(self, k):
        vals = list(self.sel_cb.cget("values"))
        if vals and self.sel.get() in vals:
            self.sel.set(vals[(vals.index(self.sel.get()) + k) % len(vals)])
            self.update_views()

    def _open_pdf(self):
        import subprocess
        import sys
        f = self.cd / "cross_date_matching_report.pdf"
        if f.exists():
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(f)])

    # ------------------------------------------------------------------ loading
    def _row_points(self, d):
        key = ("row", d)
        if key not in self._cache:
            f = self.cd / "01_clusters" / f"row_points_{d}.npz"
            if f.exists():
                z = np.load(f)
                self._cache[key] = (z["xyz"].astype(np.float64), z["s"].astype(np.float64))
            else:
                self._cache[key] = (np.zeros((0, 3)), np.zeros(0))
        return self._cache[key]

    def _plant_points(self, d, pid, voxel=0.004):
        key = ("plant", d, pid)
        if key not in self._cache:
            from pipeline.config import PROJECT_ORIGIN
            idx = self.run / d / "03_plants" / "plants_index.csv"
            row = next((r for r in _csv(idx) if r["plant_id"] == pid), None)
            if row is None:
                self._cache[key] = None
                return None
            L = np.load(self.run / d / "03_plants" / row["file"])
            xyz = L["xyz"] - np.asarray(PROJECT_ORIGIN)
            k = np.floor(xyz / voxel).astype(np.int64)
            _, i = np.unique(k, axis=0, return_index=True)
            i = np.sort(i)
            xyz = self.als[d].apply(xyz[i])
            sh = self.shifts.get((d, pid))
            if sh:
                xyz[:, 0] += float(sh["fine_dx_m"])
                xyz[:, 1] += float(sh["fine_dy_m"])
            self._cache[key] = (xyz, L["labels"][i])
        return self._cache[key]

    # ------------------------------------------------------------------ drawing
    def update_views(self):
        dates = [d for d in self.dates if self.dvars[d].get()]
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self.panels = []
        if not dates or not self.sel.get():
            ttk.Label(self.grid_frame, text="Select at least one date.").pack()
            return
        n = len(dates) + 1
        ncol = 3 if n > 4 else min(n, 3)
        nrow = -(-n // ncol)
        for c in range(ncol):
            self.grid_frame.columnconfigure(c, weight=1, uniform="col")
        for r in range(nrow):
            self.grid_frame.rowconfigure(r, weight=1, uniform="row")
        link = CameraLink()
        name = self.sel.get().split("  (")[0]
        titles = [f"{_short(name)}: all dates overlapped"] + [f"{_short(name)}  {pretty(d)}" for d in dates]
        for i, t in enumerate(titles):
            p = Panel(self.grid_frame, t)
            p.grid(row=i // ncol, column=i % ncol, sticky="nsew", padx=2, pady=2)
            link.add(p)
            self.panels.append(p)
        self._job += 1
        job = self._job
        cluster_mode = self.mode.get() == MODE_CLUSTER
        sel = self.sel.get()
        opts = {"labels": self.show_labels.get(), "back": self.show_back.get()}   # Tk variables: main thread only
        q: queue.Queue = queue.Queue()

        def work():
            return self._scene_cluster(sel, dates, opts) if cluster_mode else self._scene_plant(sel, dates, opts)

        threading.Thread(target=lambda: q.put(self._safe(work)), daemon=True).start()

        def poll():
            if job != self._job or not self.winfo_exists():
                return
            try:
                ok, val = q.get_nowait()
            except queue.Empty:
                self.after(80, poll)
                return
            if not ok:
                self.log(f"Cross-date view: {val}")
                return
            self._draw(dates, *val)
        self.after(80, poll)
        self._fill_table(cluster_mode, sel)

    @staticmethod
    def _safe(f):
        try:
            return True, f()
        except Exception as e:  # noqa: BLE001
            import traceback
            return False, f"{e}\n{traceback.format_exc()}"

    def _scene_cluster(self, sel, dates, opts):
        cid = sel.split("(")[-1].rstrip(")") if "(" in sel else sel
        rc = self.clusters[cid]
        ref_plants = sorted(pid for (d, pid), r in self.shifts.items() if d == self.ref and r["cluster"] == cid)
        colour = {pid: _rgb(PLANT_COLOURS[k % len(PLANT_COLOURS)]) for k, pid in enumerate(ref_plants)}
        per, overlay = {}, []
        for d in dates:
            links = [r for r in self.cmatch if r["reference_cluster"] == cid and r["date"] == d]
            dcl = links[0]["date_clusters"].split() if links and links[0]["date_clusters"] else []
            stretch = [(float(self.clusters[c]["lo_m"]), float(self.clusters[c]["hi_m"])) for c in dcl if c in self.clusters]
            if d == self.ref or not stretch:
                stretch = [(float(rc["lo_m"]), float(rc["hi_m"]))]
            lo, hi = min(a for a, _ in stretch) - 0.05, max(b for _, b in stretch) + 0.05
            xyz, s = self._row_points(d)
            back = xyz[(s >= lo) & (s <= hi)]
            layers = [(back, _rgb(BACK), 2)] if opts["back"] else []
            inv = {v: k for k, v in self.assign.get(d, {}).items()} if d != self.ref else {p: p for p in ref_plants}
            pids = {pid for (dd, pid), r in self.shifts.items() if dd == d and r["cluster"] in (dcl or [cid])}
            pids |= {self.assign.get(d, {}).get(rp) for rp in ref_plants if self.assign.get(d, {}).get(rp)}
            if d == self.ref:
                pids = set(ref_plants)
            legend = []
            for pid in sorted(pids):
                pts = self._plant_points(d, pid)
                if pts is None:
                    continue
                rp = inv.get(pid)
                col = colour.get(rp, _rgb(UNMATCHED))
                layers.append((pts[0], col, 3))
                legend.append((_short(nice(pid)) + ("" if rp is None or rp == pid else f" = {_short(nice(rp))}")
                               + ("" if rp in colour else " (no 29.06. match)"), col))
            status = links[0]["status"] if links else ("reference" if d == self.ref else "")
            per[d] = (layers, None, legend, "" if (len(back) or len(layers) > 1) else "nothing in this stretch")
            overlay.append((back, _rgb(DATE_HEX[d]) if d in DATE_HEX else _rgb(UNMATCHED), 2))
            per[d] = per[d] + (f"{pretty(d)}: {status}" + (f" ({', '.join(dcl)})" if dcl else ""),)
        leg = [(pretty(d), _rgb(DATE_HEX.get(d, UNMATCHED))) for d in dates]
        info = (f"{sel}: grey = all plant points of the cluster's stretch on that date; colours = separated plants, "
                f"coloured by the 29.06. plant they were matched to (same colour = same plant on every date).")
        return per, (overlay, leg), info

    def _scene_plant(self, sel, dates, opts):
        rp = raw(sel)
        per, overlay = {}, []
        for d in dates:
            qp = rp if d == self.ref else self.assign.get(d, {}).get(rp)
            if not qp:
                per[d] = ([], None, [], "no plant matched on this date", f"{pretty(d)}: no plant")
                continue
            pts = self._plant_points(d, qp)
            if pts is None:
                per[d] = ([], None, [], "no structure on this date", f"{pretty(d)}: -")
                continue
            xyz, lab = pts
            rows = [r for r in self.tracks if r["plant"] == sel and r["date"] == d]
            cols = np.tile(np.array(_rgb(OTHER)), (len(lab), 1))
            cols[lab == 0] = _rgb(STEM)
            lab_pts, lab_txt, legend = [], [], []
            stem_xy = np.median(xyz[lab == 0, :2], axis=0) if (lab == 0).any() else xyz[:, :2].mean(0)
            for r in sorted(rows, key=lambda r: r["leaf_id"]):
                lid = r["leaf_id"]
                c = _rgb(LEAF_COLOURS[(int(lid[1:]) - 1) % len(LEAF_COLOURS)]) if lid.startswith("L") else _rgb(SMALL)
                idx = np.flatnonzero(np.isin(lab, [int(x) for x in r["pieces"].split()]))
                if not len(idx):
                    continue
                cols[idx] = c
                if lid.startswith("L"):
                    far = idx[np.argmax(np.hypot(*(xyz[idx, :2] - stem_xy).T))]
                    lab_pts.append(xyz[far])
                    lab_txt.append(lid)
                    legend.append((f"{lid}  {100 * float(r['length_m']):.0f} cm", c))
            labels = (lab_pts, lab_txt) if opts["labels"] else None
            name = "" if qp == rp else f" as {_short(nice(qp))}"
            if len(legend) > 10:
                legend = legend[:10] + [(f"+{len(legend) - 10} more", _rgb(OTHER))]
            per[d] = ([(xyz, cols, 3)], labels, legend, "", f"{pretty(d)}{name}")
            overlay.append((xyz, _rgb(DATE_HEX.get(d, UNMATCHED)), 2))
        leg = [(pretty(d), _rgb(DATE_HEX.get(d, UNMATCHED))) for d in dates if per.get(d) and per[d][0]]
        info = (f"{sel}: black = stem, colours = leaf IDs (same ID = same leaf on every date), dark grey = small "
                f"leaves (not tracked), light grey = points not counted as a leaf.")
        return per, (overlay, leg), info

    def _draw(self, dates, per, overlay, info):
        ov_layers, ov_leg = overlay
        self.panels[0].draw(ov_layers, legend=ov_leg)
        for p, d in zip(self.panels[1:], dates):
            layers, labels, legend, msg, title = per[d]
            p.title = f"{p.title.split('  ')[0]}  {title}"
            p.draw(layers, labels=labels, legend=legend, message=msg if not layers else "")
        p0 = self.panels[0].plotter
        if p0 is not None:
            p0.view_xz()                       # side view along the row, tilted: stems stand upright
            p0.camera.Elevation(20)
            p0.camera.Azimuth(-30)
            p0.reset_camera()
            p0.camera.Zoom(1.15 if self.mode.get() == MODE_CLUSTER else 1.6)
            for p in self.panels:
                if p.plotter is not None:
                    p.plotter.camera_position = p0.camera_position
                    p.plotter.camera.SetViewAngle(p0.camera.GetViewAngle())
                    p._home = p0.camera_position
                    p._render()
        self.info.configure(text=info + "  Drag to rotate, two fingers to zoom: all panels move together.")

    # ------------------------------------------------------------------ table
    def _fill_table(self, cluster_mode, sel):
        t = self.table
        t.delete(*t.get_children())
        if cluster_mode:
            cid = sel.split("(")[-1].rstrip(")") if "(" in sel else sel
            cols = ("date", "status", "date clusters", "method A", "DBSCAN", "overlap", "centre diff (cm)",
                    "fine shift (cm)", "fine used", "note")
            rows = []
            for d in self.dates:
                for r in self.cmatch:
                    if r["reference_cluster"] == cid and r["date"] == d:
                        f = next((x for x in self.fine if x["reference_cluster"] == cid and x["date"] == d), {})
                        rows.append((pretty(d), r["status"], r["date_clusters"], r["method_A"], r["method_B"],
                                     r["overlap_bc"], r["centre_diff_cm"], f.get("shift_cm", ""),
                                     f.get("accepted", ""), r["note"]))
        else:
            cols = ("date", "plant on that date", "plant check", "base dist (cm)", "containment",
                    "stem deviation (cm)", "stem growth (cm)", "leaves tracked", "matched", "new / not seen", "not found")
            rows = []
            for d in self.dates:
                pc = next((r for r in self.pcheck if r["reference_plant"] == sel and r["date"] == d), None)
                sm = next((r for r in self.smatch if r["reference_plant"] == sel and r["date"] == d), {})
                lm = [r for r in self.lmatch if r["plant"] == sel and r["date"] == d]
                tr = [r for r in self.tracks if r["plant"] == sel and r["date"] == d and r["leaf_id"].startswith("L")]
                if d == self.ref:
                    rows.append((pretty(d), sel, "reference", "", "", "", "", len(tr), "", "", ""))
                    continue
                if pc is None:
                    continue
                g = sm.get("stem_growth_m", "")
                rows.append((pretty(d), pc.get("matched_plant", ""), pc["status"], pc.get("base_dist_cm", ""),
                             pc.get("containment", ""), sm.get("mean_deviation_cm", ""),
                             f"{100 * float(g):+.1f}" if g not in ("", None) else "", len(tr),
                             sum(r["status"].startswith("matched") for r in lm),
                             sum(r["status"] in ("new leaf", "not seen on 29.06.") for r in lm),
                             sum(r["status"].startswith("not found") for r in lm)))
        t.configure(columns=cols)
        for c in cols:
            t.heading(c, text=c)
            t.column(c, width=max(70, 9 * len(c)), anchor="w", stretch=True)
        for r in rows:
            flag = any(str(x).startswith(("flagged", "swapped")) for x in r)
            t.insert("", "end", values=r, tags=("flag",) if flag else ())


DATE_HEX = {"230601": "#2a78d6", "230621": "#eb6834", "230629": "#1baf7a", "230705": "#eda100", "230711": "#e87ba4"}
