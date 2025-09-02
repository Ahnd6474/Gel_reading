# -*- coding: utf-8 -*-
"""
Fast desktop UI for gel band detection + standard curve (NO Streamlit).

Install deps:
    pip install -r requirements.txt

Run:
    python detect_gel.py

Features:
  - Visualize detected lanes/bands (overlay)
  - Adjust hyperparameters from UI (lane guess, prominence, width, rolling radius)
  - Fit standard curve (log10(MW)=a*y+b) from a ladder lane and MW list
  - Save/load named curves (standard_curves.json) and apply to all bands
  - Export bands (CSV) with optional fitted MW column
  - Save annotated image (overlay) and background-subtracted image from UI
"""
from __future__ import annotations
import os
from typing import List, Dict

import numpy as np
import cv2
import pandas as pd
import dearpygui.dearpygui as dpg

from gel_detection import load_gray, detect_all
from standard_curve import (
    fit_standard_curve,
    apply_curve_to_all,
    save_curve,
    refresh_curve_dropdown,
    load_curve_selected,
)

# -------------------- Defaults / State --------------------
NUM_LANES_GUESS_DEFAULT      = 10
BAND_PROMINENCE_FRAC_DEFAULT = 0.06
BAND_MIN_WIDTH_FRAC_DEFAULT  = 0.004
ROLLING_RADIUS_FRAC_DEFAULT  = 0.03

state: Dict = {
    "img_path": None,
    "gray": None,     # np.uint8 HxW
    "sub": None,      # bg-subtracted uint8 HxW (bands bright)
    "lanes": [],      # list[(x0,x1)]
    "bands": [],      # list[dict]
    "texture_id": None,
    "tex_w": 0,
    "tex_h": 0,
    "curve": None,    # (a,b,r2)
    "curve_name": None,
}

COLORS = [(0,255,0),(0,200,255),(255,0,0),(255,0,255),(255,160,0),(0,255,160)]

# -------------------- Core detection --------------------


def run_detection():
    if state["gray"] is None:
        return
    radius = dpg.get_value("ui_radius")
    numlan = dpg.get_value("ui_numlanes")
    prom = dpg.get_value("ui_prom")
    minw = dpg.get_value("ui_minw")

    sub, lanes, rows = detect_all(state["gray"], radius, numlan, prom, minw)
    state["sub"] = sub
    state["lanes"] = lanes
    state["bands"] = rows
    update_texture_and_overlay()
    update_lane_combo()

# -------------------- Visualization helpers --------------------

def np_to_rgba_tex(img_u8: np.ndarray) -> List[float]:
    if img_u8.ndim == 2:
        rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    rgba = np.concatenate([rgb, 255*np.ones((h,w,1), dtype=np.uint8)], axis=2)
    return (rgba.astype(np.float32) / 255.0).ravel().tolist()


def update_texture_and_overlay():
    sub = state.get("sub")
    if sub is None:
        return
    h, w = sub.shape[:2]

    # Create/resize dynamic texture
    if state["texture_id"] is None or w != state["tex_w"] or h != state["tex_h"]:
        if state["texture_id"] is not None:
            dpg.delete_item(state["texture_id"])
        with dpg.texture_registry():
            state["texture_id"] = dpg.add_dynamic_texture(width=w, height=h, default_value=np_to_rgba_tex(sub), tag="tex_gel")
        state["tex_w"], state["tex_h"] = w, h
        dpg.configure_item("img_widget", texture_tag=state["texture_id"], pmax=(w, h))
        dpg.configure_item("drawlist", width=w, height=h)
    else:
        dpg.set_value(state["texture_id"], np_to_rgba_tex(sub))

    # Redraw overlay
    if dpg.does_item_exist("overlay"):
        dpg.delete_item("overlay", children_only=True)
    lanes = state.get("lanes", [])
    rows  = state.get("bands", [])

    if not dpg.get_value("ui_show_overlay"):
        return

    for li,(x0,x1) in enumerate(lanes, start=1):
        col = COLORS[(li-1)%len(COLORS)]
        dpg.draw_rectangle((x0,1), (x1,h-2), color=col, thickness=1, parent="overlay")
        dpg.draw_text((x0+3, 5), f"Lane {li}", color=col, size=12, parent="overlay")

    for r in rows:
        li = r["lane"]
        col = COLORS[(li-1)%len(COLORS)]
        dpg.draw_rectangle((r["x0"], r["y0"]), (r["x1"], r["y1"]), color=col, thickness=2, parent="overlay")

    # Optional debug save – disabled by default to avoid slow disk writes
    if os.environ.get("GEL_DEBUG_SAVE"):
        save_annotated_image_core("debug/detected.png")

# -------------------- Save images --------------------

def save_annotated_image_core(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub = state.get("sub")
    if sub is None:
        return
    vis = cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR)
    rows = state.get("bands", [])
    for r in rows:
        li = r["lane"]
        col = COLORS[(li-1)%len(COLORS)]
        cv2.rectangle(vis, (r["x0"], r["y0"]), (r["x1"], r["y1"]), col, 2)
    cv2.imwrite(path, vis)


def save_annotated_image_callback():
    out = dpg.get_value("ui_save_ann_path") or "annotated.png"
    save_annotated_image_core(out)
    dpg.set_value("ui_status", f"Annotated image saved: {out}")


def save_bgsub_image_callback():
    sub = state.get("sub")
    if sub is None:
        dpg.set_value("ui_status", "Run detection first")
        return
    out = dpg.get_value("ui_save_bgsub_path") or "bgsub.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cv2.imwrite(out, sub)
    dpg.set_value("ui_status", f"BG-sub image saved: {out}")

# -------------------- Standard curve --------------------

def update_lane_combo():
    lanes = state.get("lanes", [])
    labels = [f"Lane {i+1}" for i in range(len(lanes))]
    dpg.configure_item("ui_ladder_lane", items=labels)
    if labels:
        dpg.set_value("ui_ladder_lane", labels[0])

# -------------------- Export --------------------

def export_csv_callback():
    if not state.get("bands"):
        dpg.set_value("ui_status", "No bands to export")
        return
    out = dpg.get_value("ui_export_path") or "bands.csv"
    df = pd.DataFrame(state["bands"]).sort_values(["lane","y"])
    if state.get("curve") is not None:
        a,b,_ = state["curve"]
        df["mw_fit_kDa"] = 10 ** (a*df["y"].astype(float) + b)
    df.to_csv(out, index=False)
    dpg.set_value("ui_status", f"Exported: {out} ({len(df)} rows)")

# -------------------- UI callbacks --------------------

def open_image_by_path(sender=None, app_data=None, user_data=None):
    path = (dpg.get_value("ui_imgpath") or "").strip().strip('"')
    if not path:
        dpg.set_value("ui_status", "Path is empty")
        return
    if not os.path.isfile(path):
        dpg.set_value("ui_status", f"File not found: {path}")
        return
    try:
        gray = load_gray(path)
    except Exception as e:
        dpg.set_value("ui_status", f"Load failed: {e}")
        return
    state["img_path"] = path
    state["gray"] = gray
    dpg.set_value("ui_imgpath", path)
    run_detection()
    h, w = gray.shape[:2]
    dpg.set_value("ui_status", f"Loaded: {os.path.basename(path)}  size={w}x{h}")

# -------------------- UI callbacks --------------------

def open_image_callback(sender, app_data):
    path = app_data.get("file_path_name")
    if not path:
        return
    try:
        gray = load_gray(path)
    except Exception as e:
        dpg.set_value("ui_status", f"Load failed: {e}")
        return
    state["img_path"] = path
    state["gray"] = gray
    dpg.set_value("ui_imgpath", path)
    run_detection()
    h, w = gray.shape[:2]
    dpg.set_value("ui_status", f"Loaded: {os.path.basename(path)}  size={w}x{h}")

# -------------------- Build UI --------------------

def build_ui():
    dpg.create_context()

    # placeholder texture for initial draw_image
    with dpg.texture_registry():
        dpg.add_dynamic_texture(width=2, height=2, default_value=[1.0,1.0,1.0,1.0]*4, tag="tex_placeholder")

    with dpg.window(tag="primary", label="Gel Bands UI", width=1280, height=800):
        with dpg.group(horizontal=True):
            # left pane: controls
            with dpg.child_window(width=420, autosize_y=True):
                dpg.add_text("Image & Detection")
                dpg.add_input_text(tag="ui_imgpath", label="Image path", readonly=True)
                dpg.add_button(label="Open image", callback=lambda: dpg.show_item("file_dialog"))
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Load path", callback=open_image_by_path)
                    dpg.add_text("(paste full path above and click)")
                dpg.add_separator()
                dpg.add_slider_int(tag="ui_numlanes", label="# lanes (guess)", default_value=NUM_LANES_GUESS_DEFAULT, min_value=0, max_value=60)
                dpg.add_slider_float(tag="ui_prom", label="band prominence frac", default_value=BAND_PROMINENCE_FRAC_DEFAULT, min_value=0.0, max_value=0.5, format="%.3f")
                dpg.add_slider_float(tag="ui_minw", label="band min width frac", default_value=BAND_MIN_WIDTH_FRAC_DEFAULT, min_value=0.0, max_value=0.05, format="%.3f")
                dpg.add_slider_float(tag="ui_radius", label="rolling radius frac", default_value=ROLLING_RADIUS_FRAC_DEFAULT, min_value=0.005, max_value=0.1, format="%.3f")
                dpg.add_checkbox(tag="ui_show_overlay", label="show overlay", default_value=True, callback=lambda s,a: update_texture_and_overlay())
                dpg.add_button(label="Run detection", callback=lambda: run_detection())
                dpg.add_separator()
                dpg.add_input_text(tag="ui_export_path", label="Export CSV", default_value="bands.csv")
                dpg.add_button(label="Export bands", callback=lambda: export_csv_callback())
                dpg.add_separator()
                dpg.add_text("Standard Curve")
                dpg.add_combo(tag="ui_ladder_lane", label="ladder lane", items=[])
                dpg.add_radio_button(tag="ui_match_order", items=["top->bottom","bottom->top"], default_value="top->bottom", horizontal=True)
                dpg.add_input_text(tag="ui_mw_text", label="MW list (kDa)", multiline=True, height=80, hint="e.g. 10, 15, 25, 50, 75, 100")
                dpg.add_button(label="Fit curve", callback=lambda: fit_standard_curve(state))
                dpg.add_text(tag="ui_curve_text", default_value="(no curve)")
                dpg.add_input_text(tag="ui_curve_name", label="curve name")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save curve", callback=lambda: save_curve(state))
                    dpg.add_combo(tag="ui_curve_load", width=180, callback=lambda s,a: load_curve_selected(state, a), items=[])
                    dpg.add_button(label="Apply to all", callback=lambda: apply_curve_to_all(state))
                dpg.add_separator()
                dpg.add_text("Save Images")
                dpg.add_input_text(tag="ui_save_ann_path", label="Annotated PNG", default_value="annotated.png")
                dpg.add_button(label="Save annotated", callback=lambda: save_annotated_image_callback())
                dpg.add_input_text(tag="ui_save_bgsub_path", label="BG-sub PNG", default_value="bgsub.png")
                dpg.add_button(label="Save bg-sub", callback=lambda: save_bgsub_image_callback())
                dpg.add_separator()
                dpg.add_text(tag="ui_status", default_value="ready")

            # right pane: drawlist viewer
            with dpg.child_window(autosize_x=True, autosize_y=True):
                with dpg.drawlist(tag="drawlist", width=800, height=600):
                    dpg.draw_image("tex_placeholder", (0,0), (800,600), tag="img_widget")
                    with dpg.draw_layer(tag="overlay"):
                        pass

    with dpg.file_dialog(tag="file_dialog", directory_selector=False, show=False, callback=open_image_callback, width=900, height=500):
        # Put "All files" FIRST so it's the default filter (otherwise only the selected extension is shown)
        dpg.add_file_extension(".*", color=(128,128,128,255), custom_text="All files (*.*)")
        for ext in (".png",".PNG",".jpg",".JPG",".jpeg",".JPEG",".tif",".TIF",".tiff",".TIFF",".bmp",".BMP",".gif",".GIF"):
            dpg.add_file_extension(ext, color=(0, 255, 255, 255))

    dpg.create_viewport(title="Gel Bands UI", width=1280, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)
    refresh_curve_dropdown()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    build_ui()
