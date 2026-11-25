"""Rotatable 3D view inside Tkinter: a PyVista scene rendered off-screen and
shown as an image (PyVista cannot be embedded in Tkinter directly).

Drag = rotate, wheel = zoom, plus buttons for rotate / zoom / front-side-top
views. Subclasses fill the plotter in build_scene() and add their own
check boxes above the view with layer_bar().
"""

from __future__ import annotations

import contextlib
import queue
import threading
import tkinter as tk
from tkinter import ttk


class CheckMenu(ttk.Menubutton):
    """Drop-down button with a check box per item (instead of a row of check boxes that can be
    pushed out of the window). The button text shows how many items are on."""

    def __init__(self, parent, title: str, items: list[tuple[str, tk.BooleanVar]], command=None):
        super().__init__(parent, direction="below")
        self.title, self.command = title, command
        self.menu = tk.Menu(self, tearoff=False)
        self["menu"] = self.menu
        self.vars = []
        for label, var in items:
            self.menu.add_checkbutton(label=label, variable=var, command=self._changed)
            self.vars.append(var)
        self.menu.add_separator()
        self.menu.add_command(label="all on", command=lambda: self._all(True))
        self.menu.add_command(label="all off", command=lambda: self._all(False))
        self._label()

    def _label(self):
        on = sum(bool(v.get()) for v in self.vars)
        self.configure(text=f"{self.title}: {on} of {len(self.vars)} \u25be")

    def _changed(self):
        self._label()
        if self.command:
            self.command()

    def _all(self, on: bool):
        for i, v in enumerate(self.vars):
            if self.menu.entrycget(i, "state") != "disabled":
                v.set(on)
        self._changed()

    def set_item(self, i: int, label: str | None = None, enabled: bool | None = None):
        if label is not None:
            self.menu.entryconfigure(i, label=label)
        if enabled is not None:
            self.menu.entryconfigure(i, state="normal" if enabled else "disabled")
        self._label()


def size_sliders(parent, on_change) -> None:
    """'point size' and 'structure size' sliders. The values are kept on the main window, so
    every 3D view (and the next plant) uses the same sizes."""
    top = parent.winfo_toplevel()
    for name in ("_point_size_var", "_line_size_var"):
        if not hasattr(top, name):
            setattr(top, name, tk.DoubleVar(value=1.0))
    f = ttk.Frame(parent)
    f.pack(side="right", padx=6)
    for text, var in (("point size", top._point_size_var), ("structure size", top._line_size_var)):
        ttk.Label(f, text=text).pack(side="left", padx=(6, 0))
        lbl = ttk.Label(f, width=4)
        ttk.Scale(f, from_=0.3, to=4.0, variable=var, length=90,
                  command=lambda _=None, var=var, lbl=lbl: (lbl.configure(text=f"x{var.get():.1f}"), on_change())
                  ).pack(side="left")
        lbl.configure(text=f"x{var.get():.1f}")
        lbl.pack(side="left")


class CameraLink:
    """Views that move together: rotating / zooming one moves all (same data frame)."""

    def __init__(self):
        self.views: list = []

    def add(self, v) -> None:
        self.views.append(v)
        v.link = self

    def sync(self, src) -> None:
        if src.plotter is None:
            return
        pos = src.plotter.camera_position
        ang = src.plotter.camera.GetViewAngle()
        for v in self.views:
            if v is not src and v.plotter is not None and v.winfo_exists():
                v.plotter.camera_position = pos
                v.plotter.camera.SetViewAngle(ang)
                v._render()


TOUCHPAD_ZOOM = 0.02   # zoom per trackpad scroll step; make it negative to swap the direction


class OffscreenView(ttk.Frame):
    loading_text = "Loading ..."

    def __init__(self, master, title: str, log=print):
        super().__init__(master)
        self.title, self.log = title, log
        self.plotter = None
        self.actors: dict = {}
        self._pending = None
        self._last = None
        self._img = None
        self._home = None
        self._bases = None
        self.link: CameraLink | None = None

    # ---------------------------------------------------------------- layout
    def layer_bar(self) -> ttk.Frame:
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Button(bar, text="Reset view", command=self._reset).pack(side="right")
        self.size_bar(bar)
        return bar

    def size_bar(self, parent) -> None:
        size_sliders(parent, lambda: self._apply_point_size())
        top = self.winfo_toplevel()
        self.psize, self.lsize = top._point_size_var, top._line_size_var

    def _apply_point_size(self, render: bool = True):
        """Point / line sizes = sizes the scene was built with x slider factors. Only properties
        change (no data is copied or rebuilt), then one redraw."""
        if self.plotter is None:
            return
        top = self.winfo_toplevel()
        pf = float(top._point_size_var.get()) if hasattr(top, "_point_size_var") else 1.0
        lf = float(top._line_size_var.get()) if hasattr(top, "_line_size_var") else 1.0
        if self._bases is None:
            self._bases = {}
            for a in self.plotter.renderer.actors.values():
                get = getattr(a, "GetProperty", None)
                if get is not None and hasattr(get(), "GetPointSize"):
                    pr = get()
                    self._bases[a] = (pr.GetPointSize(), pr.GetLineWidth())
        for a, (ps, lw) in self._bases.items():
            pr = a.GetProperty()
            pr.SetPointSize(max(1.0, ps * pf))
            pr.SetLineWidth(max(1.0, lw * lf))
        if render:
            self._render()

    def make_view(self) -> None:
        """The image area (call after layer_bar). Rotate / zoom with the mouse or trackpad only;
        a scale bar and the view width are drawn on the picture."""
        self.canvas = tk.Label(self, text=self.loading_text, background="white")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<MouseWheel>", self._wheel)
        # trackpad (Tk 9 sends two-finger scrolling as <TouchpadScroll>): up/down = zoom, sideways = rotate
        with contextlib.suppress(tk.TclError):
            self.canvas.bind("<TouchpadScroll>", self._touchpad)
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        for key, f in (("<plus>", 1.2), ("<equal>", 1.2), ("<minus>", 1 / 1.2)):
            self.canvas.bind(key, lambda e, f=f: self._zoom(f))
        self.canvas.bind("<Button-4>", lambda e: self._zoom(1.1))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(1 / 1.1))
        self.canvas.bind("<Configure>", lambda e: self._render())
        self.canvas.bind("<Double-Button-1>", lambda e: self._reset())

    def in_background(self, work, done, fail=None) -> None:
        """Run work() in a thread; done(result) / fail(exception) run in the Tk thread.
        (Tk calls must not be made from the worker thread.)"""
        q: queue.Queue = queue.Queue()

        def run():
            try:
                q.put((True, work()))
            except Exception as e:  # noqa: BLE001
                q.put((False, e))
        threading.Thread(target=run, daemon=True).start()

        def poll():
            if not self.winfo_exists():
                return
            try:
                ok, val = q.get_nowait()
            except queue.Empty:
                self.after(50, poll)
                return
            if ok:
                done(val)
            elif fail is not None:
                fail(val)
        self.after(50, poll)

    def new_plotter(self):
        import pyvista as pv
        if self.plotter is not None:
            with contextlib.suppress(Exception):
                self.plotter.close()
        w, h = max(self.canvas.winfo_width(), 400), max(self.canvas.winfo_height(), 400)
        self.plotter = pv.Plotter(off_screen=True, window_size=(w, h))
        self._bases = None
        return self.plotter

    def show_message(self, text: str) -> None:
        self.canvas.configure(text=text, image="")

    # ---------------------------------------------------------------- render
    def _render(self):
        """Ask for a redraw; several requests within 40 ms give one redraw."""
        if self.plotter is None or self._pending is not None:
            return
        self._pending = self.after(40, self._render_now)

    def _render_now(self):
        self._pending = None
        if self.plotter is None:
            return
        from PIL import Image, ImageTk
        w, h = max(self.canvas.winfo_width(), 200), max(self.canvas.winfo_height(), 200)
        if tuple(self.plotter.window_size) != (w, h):
            self.plotter.window_size = (w, h)
        if self._bases is None:
            self._apply_point_size(render=False)     # first draw of a new scene: use the slider's size
        self.plotter.render()          # screenshot() alone returns the previous frame after a camera change
        img = Image.fromarray(self.plotter.screenshot(return_img=True))
        self._draw_scale(img)
        self._img = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self._img, text="")

    def view_height_m(self) -> float:
        """Height of the visible area at the point the camera looks at (metres)."""
        import math
        cam = self.plotter.camera
        if cam.GetParallelProjection():
            return 2.0 * cam.GetParallelScale()
        return 2.0 * cam.GetDistance() * math.tan(math.radians(cam.GetViewAngle() / 2.0))

    def _draw_scale(self, img) -> None:
        """Scale bar (nice length) + view width, bottom left; mouse hint, bottom right."""
        import math
        from PIL import ImageDraw
        w, h = img.size
        m_px = self.view_height_m() / max(h, 1)
        if not (m_px > 0 and math.isfinite(m_px)):
            return
        target = 120 * m_px                      # about 120 px long
        e = 10 ** math.floor(math.log10(target))
        L = min((f * e for f in (1, 2, 5, 10)), key=lambda v: abs(v - target))
        px = int(round(L / m_px))
        dr = ImageDraw.Draw(img)
        try:
            from PIL import ImageFont
            font = ImageFont.load_default(size=15)
        except Exception:  # noqa: BLE001  (old Pillow)
            font = None
        x0, y0 = 16, h - 22
        dr.rectangle([x0, y0, x0 + px, y0 + 5], fill=(0, 0, 0))
        for xx in (x0, x0 + px):
            dr.line([xx, y0 - 5, xx, y0 + 5], fill=(0, 0, 0), width=2)
        lab = f"{L * 100:g} cm" if L >= 0.01 else f"{L * 1000:g} mm"
        dr.text((x0, y0 - 20), lab, fill=(0, 0, 0), font=font)
        dr.text((x0 + px + 14, y0 - 6), f"view {w * m_px * 100:.0f} x {h * m_px * 100:.0f} cm", fill=(0, 0, 0), font=font)
        hint = "drag: rotate   two fingers: zoom   double-click: reset"
        tw = dr.textlength(hint, font=font) if hasattr(dr, "textlength") else 7 * len(hint)
        if w - tw > x0 + px + 260:                         # only where it does not cover the scale
            dr.text((w - 10 - tw, h - 22), hint, fill=(120, 120, 120), font=font)

    # ---------------------------------------------------------------- mouse / buttons
    def _moved(self):
        if self.link is not None:
            self.link.sync(self)

    def _press(self, e):
        self._last = (e.x, e.y)

    def _drag(self, e):
        if self.plotter is None or self._last is None:
            return
        dx, dy = e.x - self._last[0], e.y - self._last[1]
        self._last = (e.x, e.y)
        self._turn(-dx * 0.5, dy * 0.5)

    def _turn(self, az, el):
        if self.plotter is None:
            return
        cam = self.plotter.camera
        cam.Azimuth(az)
        cam.Elevation(el)
        cam.OrthogonalizeViewUp()
        self._render()
        self._moved()

    def _view(self, plane):
        if self.plotter is None:
            return
        getattr(self.plotter, f"view_{plane}")()
        self.plotter.reset_camera()
        self._render()
        self._moved()

    def _touchpad(self, e):
        try:
            dx, dy = (float(v) for v in self.tk.splitlist(self.tk.call("tk::PreciseScrollDeltas", e.delta)))
        except (tk.TclError, ValueError):
            return
        if abs(dy) >= abs(dx) and dy:
            self._zoom(min(1.25, max(0.8, 1.0 + TOUCHPAD_ZOOM * dy)))
        elif dx:
            self._turn(-dx * 0.5, 0)

    def _wheel(self, e):
        self._zoom(1.1 if e.delta > 0 else 1 / 1.1)

    def _zoom(self, f):
        if self.plotter is not None:
            self.plotter.camera.Zoom(f)
            self._render()
            self._moved()

    def _reset(self):
        if self.plotter is not None and self._home is not None:
            self.plotter.camera_position = self._home
            self._render()
            self._moved()

    def destroy(self):
        if self._pending is not None:
            with contextlib.suppress(Exception):
                self.after_cancel(self._pending)
        if self.plotter is not None:
            with contextlib.suppress(Exception):
                self.plotter.close()
        super().destroy()
