"""Matplotlib toolbar whose "Configure subplots" window explains every setting.

Each slider has a "?" button: what the setting does, a good value, and what
happens when it is increased or decreased.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

# name -> (title, what it does, good value, increase, decrease)
HELP = {
    "left": ("Left margin",
             "Position of the left edge of the plots, as a fraction of the figure width (0 = window edge, 1 = right edge).",
             "0.08 - 0.12 (matplotlib default 0.125). Must stay smaller than 'right'.",
             "Plots start further right: more room for the y-axis labels, but narrower plots.",
             "Plots start closer to the left edge: wider plots, but y-axis labels can be cut off."),
    "bottom": ("Bottom margin",
               "Position of the bottom edge of the plots, as a fraction of the figure height (0 = bottom, 1 = top).",
               "0.08 - 0.12 (default 0.11). Must stay smaller than 'top'.",
               "Plots start higher: more room for the x-axis labels and dates, but shorter plots.",
               "Plots reach further down: taller plots, but x-axis labels can be cut off."),
    "right": ("Right margin",
              "Position of the right edge of the plots, as a fraction of the figure width.",
              "0.95 - 0.98 (default 0.9). Must stay larger than 'left'.",
              "Plots reach closer to the right edge: wider plots; a colour bar or legend there can be cut off.",
              "More empty space on the right (useful for a legend or colour bar), narrower plots."),
    "top": ("Top margin",
            "Position of the top edge of the plots, as a fraction of the figure height.",
            "0.88 - 0.95 (default 0.88). Must stay larger than 'bottom'.",
            "Plots reach closer to the top: taller plots, but the title can overlap the plots.",
            "More room for the figure title above the plots, shorter plots."),
    "wspace": ("Horizontal space between plots",
               "Gap between plots side by side, as a fraction of the average plot width (e.g. one 3D view per date).",
               "0.15 - 0.3 (default 0.2).",
               "Plots further apart: labels of neighbouring plots no longer overlap, but each plot gets narrower.",
               "Plots closer together and wider; axis labels of neighbours can overlap."),
    "hspace": ("Vertical space between plots",
               "Gap between plots above each other, as a fraction of the average plot height (e.g. rows of growth curves).",
               "0.25 - 0.4 (default 0.2; more when plots have titles).",
               "Rows further apart: titles and x labels no longer run into the plot above, but each plot gets shorter.",
               "Rows closer together and taller; a title can overlap the x labels of the row above."),
}
LIMITS = {"left": (0.0, 0.49), "bottom": (0.0, 0.49), "right": (0.51, 1.0), "top": (0.51, 1.0),
          "wspace": (0.0, 1.0), "hspace": (0.0, 1.0)}


def show_help(parent, key: str) -> None:
    title, what, good, up, down = HELP[key]
    messagebox.showinfo(title, f"{what}\n\nGood value: {good}\n\nIncrease: {up}\n\nDecrease: {down}", parent=parent)


class SubplotTool(tk.Toplevel):
    def __init__(self, master, canvas):
        super().__init__(master)
        self.title("Subplot configuration tool")
        self.canvas, self.fig = canvas, canvas.figure
        self.start = {k: getattr(self.fig.subplotpars, k) for k in HELP}
        self.vars = {}
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Spacing of the plots in this figure (fractions of the figure size)").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))
        for r, k in enumerate(HELP, start=1):
            ttk.Label(frm, text=HELP[k][0], width=28).grid(row=r, column=0, sticky="w")
            v = tk.DoubleVar(value=self.start[k])
            self.vars[k] = v
            lo, hi = LIMITS[k]
            ttk.Scale(frm, from_=lo, to=hi, variable=v, length=260,
                      command=lambda _=None: self._apply()).grid(row=r, column=1, padx=4)
            val = ttk.Label(frm, width=6)
            val.grid(row=r, column=2)
            v.trace_add("write", lambda *_a, v=v, val=val: val.configure(text=f"{v.get():.3f}"))
            val.configure(text=f"{v.get():.3f}")
            ttk.Button(frm, text="?", width=2, command=lambda k=k: show_help(self, k)).grid(row=r, column=3, padx=4)
        b = ttk.Frame(frm)
        b.grid(row=len(HELP) + 1, column=0, columnspan=4, pady=(8, 0), sticky="w")
        ttk.Button(b, text="Tight layout", command=self._tight).pack(side="left")
        ttk.Button(b, text="Reset", command=self._reset).pack(side="left", padx=4)
        ttk.Button(b, text="Close", command=self.destroy).pack(side="left")
        # bring the window in front of the app
        self.transient(master)
        self.lift()
        self.attributes("-topmost", True)
        self.after(400, lambda: self.attributes("-topmost", False))
        self.focus_force()

    def _apply(self):
        p = {k: v.get() for k, v in self.vars.items()}
        if p["left"] < p["right"] and p["bottom"] < p["top"]:
            self.fig.subplots_adjust(**p)
            self.canvas.draw_idle()

    def _tight(self):
        self.fig.tight_layout()
        for k, v in self.vars.items():
            v.set(getattr(self.fig.subplotpars, k))
        self.canvas.draw_idle()

    def _reset(self):
        for k, v in self.vars.items():
            v.set(self.start[k])
        self._apply()


class HelpToolbar(NavigationToolbar2Tk):
    """NavigationToolbar2Tk with the explained subplot tool."""

    def configure_subplots(self, *args):
        SubplotTool(self.canvas.get_tk_widget().winfo_toplevel(), self.canvas)
