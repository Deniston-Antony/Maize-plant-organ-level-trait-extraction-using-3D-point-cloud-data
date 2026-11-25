"""Run progress and finish-time estimate.

The pipeline calls `Progress.update(stage, done, total)`. Each stage has a
weight (its expected share of the run time). The overall fraction is the
weighted sum of finished work, and the time left is estimated from:

  * before 10 % is done: the previous runs (seconds per million input points,
    stored in outputs/_timing_history.json), or the optional budget if there
    is no history yet;
  * after that: the measured speed of this run, blended in gradually.

The callback receives a ProgressInfo; the UI turns it into the live timer.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import OUTPUT_ROOT

HISTORY_FILE = OUTPUT_ROOT / "_timing_history.json"


@dataclass
class ProgressInfo:
    fraction: float        # 0..1 of the whole run
    stage: str             # e.g. "soil 230621"
    elapsed_s: float
    remaining_s: float | None
    budget_s: float | None


class Progress:
    def __init__(self, stages: dict[str, float], callback: Callable[[ProgressInfo], None] | None = None,
                 budget_s: float | None = None, n_points_millions: float | None = None, mode: str = "default"):
        tot = sum(stages.values()) or 1.0
        self.weights = {k: v / tot for k, v in stages.items()}
        self.done = {k: 0.0 for k in stages}
        self.callback = callback
        self.budget_s = budget_s
        self.mode = mode
        self.n_points = n_points_millions
        self.t0 = time.monotonic()
        self.prior_s = self._prior_from_history()
        self._lock = threading.Lock()
        self._last_emit = 0.0
        self._emit("starting", force=True)

    # -------------------------------------------------------------- history
    def _prior_from_history(self) -> float | None:
        if not self.n_points:
            return self.budget_s
        try:
            hist = json.loads(HISTORY_FILE.read_text())
            rates = [h["s_per_mpts"] for h in hist if h.get("mode") == self.mode][-5:]
            if rates:
                rates.sort()
                return rates[len(rates) // 2] * self.n_points
        except Exception:
            pass
        return self.budget_s

    def _save_history(self, total_s: float) -> None:
        if not self.n_points:
            return
        try:
            hist = json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else []
        except Exception:
            hist = []
        hist.append({"mode": self.mode, "total_s": round(total_s, 1), "mpts": round(self.n_points, 2),
                     "s_per_mpts": total_s / self.n_points, "when": time.strftime("%Y-%m-%d %H:%M")})
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            HISTORY_FILE.write_text(json.dumps(hist[-50:], indent=1))
        except Exception:
            pass

    # -------------------------------------------------------------- updates
    @property
    def fraction(self) -> float:
        return min(1.0, sum(self.weights[k] * self.done[k] for k in self.weights))

    def update(self, stage: str, done: float, total: float, text: str = "") -> None:
        """Report `done` of `total` work units finished in `stage`."""
        with self._lock:
            key = stage if stage in self.weights else next(iter(self.weights))
            self.done[key] = max(self.done[key], min(1.0, done / total if total else 1.0))
            self._emit(text or stage)

    def finish_stage(self, stage: str) -> None:
        self.update(stage, 1, 1)

    def finish(self) -> float:
        total = time.monotonic() - self.t0
        with self._lock:
            for k in self.done:
                self.done[k] = 1.0
            self._emit("finished", force=True)
        self._save_history(total)
        return total

    def remaining(self) -> float | None:
        el = time.monotonic() - self.t0
        f = self.fraction
        measured = el * (1 - f) / f if f > 0.02 else None
        prior = max(0.0, self.prior_s - el) if self.prior_s else None
        if measured is None:
            return prior
        if prior is None:
            return measured
        w = min(1.0, max(0.0, (f - 0.10) / 0.30))   # trust the measured speed from 10 % to 40 %
        return w * measured + (1 - w) * prior

    def _emit(self, stage: str, force: bool = False) -> None:
        now = time.monotonic()
        if not self.callback or (not force and now - self._last_emit < 0.25):
            return
        self._last_emit = now
        self.callback(ProgressInfo(self.fraction, stage, now - self.t0, self.remaining(), self.budget_s))


def fmt_seconds(s: float | None) -> str:
    if s is None:
        return "--:--"
    s = int(round(max(0.0, s)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"
