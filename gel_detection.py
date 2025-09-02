# -*- coding: utf-8 -*-
"""Core image processing routines for gel band detection.

These functions operate directly on numpy arrays and do not depend on the
user interface.  They can be imported and used separately from the GUI to
process gel images programmatically.
"""
from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np
import cv2

try:
    from skimage.restoration import rolling_ball
    HAS_RB = True
except Exception:
    HAS_RB = False


def load_gray(path: str) -> np.ndarray:
    """Load an image as grayscale using a Unicode-safe method."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except Exception as e:
        raise RuntimeError(f"Failed to read image bytes: {path}: {e}")

    g = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise RuntimeError(f"Failed to decode image: {path}")
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
    if sub.mean() < 128:
        sub = cv2.bitwise_not(sub)
    sub = cv2.GaussianBlur(sub, (0, 0), 1.0)
    return sub


def detect_lanes(sub: np.ndarray, num_guess: int) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    from scipy.signal import find_peaks

    H, W = sub.shape
    col_sum = sub.sum(axis=0).astype(np.float32)
    col_sum = cv2.GaussianBlur(col_sum.reshape(1, -1), (1, 0), 5).ravel()
    distance = max(5, W // max(1, (num_guess * 2))) if num_guess else None
    peaks, _ = find_peaks(col_sum, distance=distance, prominence=max(10.0, col_sum.max() * 0.02))
    centers = np.sort(peaks)
    if len(centers) == 0:
        return [(0, W - 1)], np.array([], dtype=int)
    mids = ((centers[:-1] + centers[1:]) // 2).astype(int) if len(centers) > 1 else np.array([W // 2])
    lefts = np.concatenate([[0], mids])
    rights = np.concatenate([mids, [W - 1]])
    bounds = list(zip(lefts, rights))
    return bounds, centers


def lane_profile(sub: np.ndarray, x0: int, x1: int) -> np.ndarray:
    lane = sub[:, x0 : x1 + 1]
    prof = lane.mean(axis=1).astype(np.float32)
    base = cv2.GaussianBlur(prof.reshape(-1, 1), (0, 0), sigmaX=15).ravel()
    prof2 = np.clip(prof - base, 0, None)
    prof2 = cv2.GaussianBlur(prof2.reshape(-1, 1), (0, 0), sigmaX=2).ravel()
    return prof2


def detect_bands_from_profile(
    prof: np.ndarray, img_h: int, prom_frac: float, min_w_frac: float
):
    from scipy.signal import find_peaks

    min_prom = max(5.0, float(prof.max()) * prom_frac)
    min_width = max(2, int(img_h * min_w_frac))
    peaks, props = find_peaks(prof, prominence=min_prom, width=min_width)
    bands = []
    for i, p in enumerate(peaks):
        left = int(props["left_ips"][i])
        right = int(props["right_ips"][i])
        area = float(prof[left : right + 1].sum())
        bands.append(
            {
                "y": int(p),
                "y0": left,
                "y1": right,
                "prom": float(props["prominences"][i]),
                "width": float(props["widths"][i]),
                "area1d": area,
            }
        )
    return bands


def quantify_lane(sub: np.ndarray, x0: int, x1: int, bands: List[dict]):
    lane = sub[:, x0 : x1 + 1].astype(np.float32)
    rows = []
    for b in bands:
        sl = lane[b["y0"] : b["y1"] + 1, :]
        area2d = float(sl.sum())
        rows.append({**b, "area2d": area2d, "x0": x0, "x1": x1})
    return rows


def detect_all(
    gray: np.ndarray,
    radius: float,
    numlan: int,
    prom: float,
    minw: float,
):
    """Full detection pipeline returning background-subtracted image,
    lane bounds and band information."""
    sub = background_subtract(gray, radius)
    lanes, _centers = detect_lanes(sub, numlan)
    H, W = sub.shape
    rows: List[Dict] = []
    for li, (x0, x1) in enumerate(lanes, start=1):
        prof = lane_profile(sub, x0, x1)
        bands = detect_bands_from_profile(prof, H, prom, minw)
        qrows = quantify_lane(sub, x0, x1, bands)
        for r in qrows:
            r.update({"lane": li})
        rows.extend(qrows)
    return sub, lanes, rows
