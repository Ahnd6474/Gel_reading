# -*- coding: utf-8 -*-
"""
Fast desktop UI for gel band detection + standard curve (NO Streamlit).

Install deps:
    pip install dearpygui opencv-python scikit-image scipy numpy pandas

Run:
    python gel_gui.py

Features:
  - Visualize detected lanes/bands (overlay)
  - Adjust hyperparameters from UI (lane guess, prominence, width, rolling radius)
  - Fit standard curve (log10(MW)=a*y+b) from a ladder lane and MW list
  - Save/load named curves (standard_curves.json) and apply to all bands
  - Export bands (CSV) with optional fitted MW column
  - Save annotated image (overlay) and background-subtracted image from UI
"""
from __future__ import annotations
import os, json
from typing import List, Tuple, Dict

import numpy as np
import cv2
import pandas as pd
from scipy.stats import linregress

try:
    from skimage.restoration import rolling_ball
    HAS_RB = True
except Exception:
    HAS_RB = False

import dearpygui.dearpygui as dpg

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

def load_gray(path: str) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return g


def background_subtract(gray: np.ndarray, radius_frac: float) -> np.ndarray:
    g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    diag = float(np.hypot(*g.shape))
    if HAS_RB:
        rad = max(20, int(radius_frac * diag))
        bg = rolling_ball(g, radius=rad)
        sub = cv2.subtract(g, np.asarray(bg, dtype=np.uint8))
    else:
        k = max(31, int(radius_frac * diag) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bg = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel)
        sub = cv2.subtract(g, bg)
    # ensure bands are bright
    if sub.mean() < 128:
        sub = cv2.bitwise_not(sub)
    sub = cv2.GaussianBlur(sub, (0,0), 1.0)
    return sub


def detect_lanes(sub: np.ndarray, num_guess: int) -> Tuple[List[Tuple[int,int]], np.ndarray]:
    from scipy.signal import find_peaks
    H, W = sub.shape
    col_sum = sub.sum(axis=0).astype(np.float32)
    col_sum = cv2.GaussianBlur(col_sum.reshape(1,-1), (1,0), 5).ravel()
    distance = max(5, W // max(1, (num_guess * 2))) if num_guess else None
    peaks, _ = find_peaks(col_sum, distance=distance, prominence=max(10.0, col_sum.max()*0.02))
    centers = np.sort(peaks)
    if len(centers) == 0:
        return [(0, W-1)], np.array([], dtype=int)
    mids = ((centers[:-1] + centers[1:]) // 2).astype(int) if len(centers) > 1 else np.array([W//2])
    lefts  = np.concatenate([[0], mids])
    rights = np.concatenate([mids, [W-1]])
    bounds = list(zip(lefts, rights))
    return bounds, centers


def lane_profile(sub: np.ndarray, x0: int, x1: int) -> np.ndarray:
    lane = sub[:, x0:x1+1]
    prof = lane.mean(axis=1).astype(np.float32)
    base = cv2.GaussianBlur(prof.reshape(-1,1), (0,0), sigmaX=15).ravel()
    prof2 = np.clip(prof - base, 0, None)
    prof2 = cv2.GaussianBlur(prof2.reshape(-1,1), (0,0), sigmaX=2).ravel()
    return prof2


def detect_bands_from_profile(prof: np.ndarray, img_h: int, prom_frac: float, min_w_frac: float):
    from scipy.signal import find_peaks
    min_prom  = max(5.0, float(prof.max()) * prom_frac)
    min_width = max(2, int(img_h * min_w_frac))
    peaks, props = find_peaks(prof, prominence=min_prom, width=min_width)
    bands = []
    for i, p in enumerate(peaks):
        left  = int(props["left_ips"][i])
        right = int(props["right_ips"][i])
        area  = float(prof[left:right+1].sum())
        bands.append({
            "y": int(p), "y0": left, "y1": right,
            "prom": float(props["prominences"][i]),
            "width": float(props["widths"][i]),
            "area1d": area
        })
    return bands


def quantify_lane(sub: np.ndarray, x0: int, x1: int, bands: List[dict]):
    lane = sub[:, x0:x1+1].astype(np.float32)
    rows = []
    for b in bands:
        sl = lane[b["y0"]:b["y1"]+1, :]
        area2d = float(sl.sum())
        rows.append({**b, "area2d": area2d, "x0": x0, "x1": x1})
    return rows


def run_detection():
    if state["gray"] is None:
        return
    radius = dpg.get_value("ui_radius")
    numlan = dpg.get_value("ui_numlanes")
    prom   = dpg.get_value("ui_prom")
    minw   = dpg.get_value("ui_minw")

    sub = background_subtract(state["gray"], radius)
    lanes, centers = detect_lanes(sub, numlan)
    H, W = sub.shape

    rows = []
    for li,(x0,x1) in enumerate(lanes, start=1):
        prof = lane_profile(sub, x0, x1)
        bands = detect_bands_from_profile(prof, H, prom, minw)
        qrows = quantify_lane(sub, x0, x1, bands)
        for r in qrows:
            r.update({"lane": li})
        rows.extend(qrows)

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

    # Save debug image
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

def parse_mw_list(s: str) -> List[float]:
    if not s.strip():
        return []
    toks = [t.strip() for t in s.replace("", ",").replace("	", ",").split(",")]
    out = []
    for t in toks:
        if not t:
            continue
        try:
            out.append(float(t))
        except Exception:
            pass
    return out


def update_lane_combo():
    lanes = state.get("lanes", [])
    labels = [f"Lane {i+1}" for i in range(len(lanes))]
    dpg.configure_item("ui_ladder_lane", items=labels)
    if labels:
        dpg.set_value("ui_ladder_lane", labels[0])


def fit_standard_curve():
    rows = state.get("bands", [])
    if not rows:
        dpg.set_value("ui_status", "Run detection first")
        return
    lane_label = dpg.get_value("ui_ladder_lane")
    if not lane_label:
        dpg.set_value("ui_status", "No lane selected")
        return
    lane_idx = int(lane_label.split()[-1])  # 1-based
    order = dpg.get_value("ui_match_order")  # "top->bottom" or "bottom->top"
    mw_text = dpg.get_value("ui_mw_text")
    mw_list = parse_mw_list(mw_text)
    if len(mw_list) == 0:
        dpg.set_value("ui_status", "Enter MW list (kDa)")
        return

    lane_peaks = [r for r in rows if r["lane"] == lane_idx]
    lane_peaks = sorted(lane_peaks, key=lambda r: r["y"])  # top->bottom (y increasing)
    ys = [r["y"] for r in lane_peaks]
    if len(ys) == 0:
        dpg.set_value("ui_status", "No bands in the ladder lane")
        return
    if order == "bottom->top":
        ys = list(reversed(ys))
    n = min(len(ys), len(mw_list))
    ys  = np.array(ys[:n], dtype=float)
    mws = np.array(mw_list[:n], dtype=float)

    log_mw = np.log10(mws)
    lr = linregress(ys, log_mw)
    a, b, r = float(lr.slope), float(lr.intercept), float(lr.rvalue)
    r2 = float(r*r)
    state["curve"] = (a, b, r2)
    dpg.set_value("ui_curve_text", f"log10(MW) = {a:.6f}*y + {b:.6f}    R^2={r2:.4f}  (n={n})")
    dpg.set_value("ui_status", "Curve fitted")


def apply_curve_to_all():
    if state.get("curve") is None:
        dpg.set_value("ui_status", "Fit a curve first")
        return
    a,b,_ = state["curve"]
    for r in state.get("bands", []):
        r["mw_fit_kDa"] = float(10 ** (a*float(r["y"]) + b))
    dpg.set_value("ui_status", "Applied curve to all bands")


def save_curve():
    if state.get("curve") is None:
        dpg.set_value("ui_status", "No curve to save")
        return
    name = dpg.get_value("ui_curve_name").strip()
    if not name:
        dpg.set_value("ui_status", "Enter curve name")
        return
    a,b,r2 = state["curve"]
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


def refresh_curve_dropdown():
    path = "standard_curves.json"
    items = []
    if os.path.exists(path):
        try:
            items = list(json.load(open(path, "r", encoding="utf-8")).keys())
        except Exception:
            items = []
    dpg.configure_item("ui_curve_load", items=items)


def load_curve_selected(sender, app_data):
    name = app_data
    path = "standard_curves.json"
    if not name or not os.path.exists(path):
        return
    data = json.load(open(path, "r", encoding="utf-8"))
    if name not in data:
        return
    a = float(data[name]["a"]) ; b = float(data[name]["b"]) ; r2 = float(data[name]["r2"])
    state["curve"] = (a,b,r2)
    state["curve_name"] = name
    dpg.set_value("ui_curve_text", f"[{name}] log10(MW) = {a:.6f}*y + {b:.6f}    R^2={r2:.4f}")
    dpg.set_value("ui_status", f"Loaded curve '{name}'")

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
                dpg.add_button(label="Fit curve", callback=lambda: fit_standard_curve())
                dpg.add_text(tag="ui_curve_text", default_value="(no curve)")
                dpg.add_input_text(tag="ui_curve_name", label="curve name")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save curve", callback=lambda: save_curve())
                    dpg.add_combo(tag="ui_curve_load", width=180, callback=load_curve_selected, items=[])
                    dpg.add_button(label="Apply to all", callback=lambda: apply_curve_to_all())
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
