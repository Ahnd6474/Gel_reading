# -*- coding: utf-8 -*-
"""Standard curve utilities for gel band analysis."""
from __future__ import annotations

import json
import os
from typing import List, Dict

import numpy as np
from scipy.stats import linregress
import dearpygui.dearpygui as dpg


def parse_mw_list(s: str) -> List[float]:
    """Parse a comma or whitespace separated list of molecular weights."""
    if not s.strip():
        return []
    import re

    toks = [t for t in re.split(r"[\s,]+", s.strip()) if t]
    out: List[float] = []
    for t in toks:
        try:
            out.append(float(t))
        except ValueError:
            continue
    return out


def fit_standard_curve(state: Dict) -> None:
    rows = state.get("bands", [])
    if not rows:
        dpg.set_value("ui_status", "Run detection first")
        return
    lane_label = dpg.get_value("ui_ladder_lane")
    if not lane_label:
        dpg.set_value("ui_status", "No lane selected")
        return
    lane_idx = int(lane_label.split()[-1])
    order = dpg.get_value("ui_match_order")
    mw_text = dpg.get_value("ui_mw_text")
    mw_list = parse_mw_list(mw_text)
    if len(mw_list) == 0:
        dpg.set_value("ui_status", "Enter MW list (kDa)")
        return

    lane_peaks = [r for r in rows if r["lane"] == lane_idx]
    lane_peaks = sorted(lane_peaks, key=lambda r: r["y"])
    ys = [r["y"] for r in lane_peaks]
    if len(ys) == 0:
        dpg.set_value("ui_status", "No bands in the ladder lane")
        return
    if order == "bottom->top":
        ys = list(reversed(ys))
    n = min(len(ys), len(mw_list))
    ys = np.array(ys[:n], dtype=float)
    mws = np.array(mw_list[:n], dtype=float)

    log_mw = np.log10(mws)
    lr = linregress(ys, log_mw)
    a, b, r = float(lr.slope), float(lr.intercept), float(lr.rvalue)
    r2 = float(r * r)
    state["curve"] = (a, b, r2)
    dpg.set_value("ui_curve_text", f"log10(MW) = {a:.6f}*y + {b:.6f}    R^2={r2:.4f}  (n={n})")
    dpg.set_value("ui_status", "Curve fitted")


def apply_curve_to_all(state: Dict) -> None:
    if state.get("curve") is None:
        dpg.set_value("ui_status", "Fit a curve first")
        return
    a, b, _ = state["curve"]
    for r in state.get("bands", []):
        r["mw_fit_kDa"] = float(10 ** (a * float(r["y"]) + b))
    dpg.set_value("ui_status", "Applied curve to all bands")


def save_curve(state: Dict) -> None:
    if state.get("curve") is None:
        dpg.set_value("ui_status", "No curve to save")
        return
    name = dpg.get_value("ui_curve_name").strip()
    if not name:
        dpg.set_value("ui_status", "Enter curve name")
        return
    a, b, r2 = state["curve"]
    path = "standard_curves.json"
    data = {}
    if os.path.exists(path):
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            data = {}
    data[name] = {"a": a, "b": b, "r2": r2}
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    refresh_curve_dropdown()
    dpg.set_value("ui_status", f"Saved curve '{name}'")


def refresh_curve_dropdown() -> None:
    path = "standard_curves.json"
    items = []
    if os.path.exists(path):
        try:
            items = list(json.load(open(path, "r", encoding="utf-8")).keys())
        except Exception:
            items = []
    dpg.configure_item("ui_curve_load", items=items)


def load_curve_selected(state: Dict, app_data) -> None:
    name = app_data
    path = "standard_curves.json"
    if not name or not os.path.exists(path):
        return
    data = json.load(open(path, "r", encoding="utf-8"))
    if name not in data:
        return
    a = float(data[name]["a"])
    b = float(data[name]["b"])
    r2 = float(data[name]["r2"])
    state["curve"] = (a, b, r2)
    state["curve_name"] = name
    dpg.set_value("ui_curve_text", f"[{name}] log10(MW) = {a:.6f}*y + {b:.6f}    R^2={r2:.4f}")
    dpg.set_value("ui_status", f"Loaded curve '{name}'")
