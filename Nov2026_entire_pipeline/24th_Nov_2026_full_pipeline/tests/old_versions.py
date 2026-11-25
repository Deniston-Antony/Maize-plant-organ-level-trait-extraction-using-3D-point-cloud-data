"""Adapters that run older plant-structure scripts from Research_project/
(read-only, imported by path) with the same interface as the bench methods:
fn(xyz) -> (xyz_work, labels 0 stem / 1..n leaves)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

RP = next((p / "Research_project" for p in Path(__file__).resolve().parents if (p / "Research_project").exists()),
          Path(__file__).resolve().parents[2] / "Research_project")
FILES = {
    "claude_v": RP / "new_structure_output" / "claude_plant_structure.py",
    "version02": RP / "Plant structure old" / "Plant structure new" / "version02_new_structure.py",
}
_mods: dict = {}


def _load(key):
    if key not in _mods:
        spec = importlib.util.spec_from_file_location(f"old_{key}", FILES[key])
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        _mods[key] = m
    return _mods[key]


def claude_v(xyz):
    m = _load("claude_v")
    return m.classify_plant_for_iou(xyz, m.PipelineConfig())


def version02(xyz):
    m = _load("version02")
    cfg = m.PipelineConfig()
    c = xyz.mean(axis=0)
    prep = m.preprocess_point_cloud(xyz - c, cfg)
    seg = m.segment_stem_and_leaf(prep, cfg)
    sk = m.extract_stem_centerline(seg, cfg)
    leaf = m.detect_leaf_clusters(prep, seg, sk, cfg)
    w = prep.xyz_work
    lab = np.full(len(w), -1, np.int32)
    lab[seg.stem_mask] = 0
    if leaf.clusters:
        allp = np.vstack(leaf.clusters)
        cid = np.concatenate([np.full(len(p), i + 1) for i, p in enumerate(leaf.clusters)])
        li = np.where(~seg.stem_mask)[0]
        _, nn = cKDTree(allp).query(w[li], k=1)
        lab[li] = cid[nn]
    lab[lab < 0] = 0
    return w + c, lab
