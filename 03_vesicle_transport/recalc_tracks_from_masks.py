#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Recompute kymograph tracks strictly inside the saved masks (AVRG and MAX reslices).

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 5E–G; Fig. S16A–C
What it does  : For one kymograph folder it reuses the region masks saved by
                kymograph_track_multi3.py and, on the CLAHE-enhanced 'Reslice (AVRG) of ...'
                and 'Reslice (MAX) of ...' kymographs, takes the brightest pixel of every
                masked row (ties: centre of a single run, otherwise the pixel closest to
                the mask centre), giving one-pixel track images and x,y coordinate CSVs.
Inputs        : <kymo_root>/Reslice (AVRG) of*.tif, <kymo_root>/Reslice (MAX) of*.tif and
                <kymo_root>/segmented_tracks/masks/*
Outputs       : <kymo_root>/tracks_from_masks_AVRG/ and tracks_from_masks_MAX/
                (tracks/track_NNN.png, coords/track_NNN.csv, overlays/all_tracks_overlay.png)
Environment   : environment-analysis.yml (Python 3.7)
Run           : python recalc_tracks_from_masks.py   (set kymo_root to one track_<k>/kymograph folder)
"""
import os
import csv
import glob
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage import exposure

from scipy.ndimage import label  # for 1D connectivity of brightest runs
# --- USER PATHS -------------------------------------------------------------
# Point OSSICLE_DATA / OSSICLE_OUT at your copies (see data/README.md).
import os
from pathlib import Path
try:
    _HERE = Path(__file__).resolve()
except NameError:  # interactive (e.g. Spyder cell) execution
    _HERE = Path.cwd().resolve() / "_"
REPO_ROOT = next((p for p in _HERE.parents if (p / "CITATION.cff").exists()), _HERE.parent)
DATA_ROOT = Path(os.environ.get("OSSICLE_DATA", REPO_ROOT / "data"))
OUT_ROOT = Path(os.environ.get("OSSICLE_OUT", REPO_ROOT / "outputs"))
# ----------------------------------------------------------------------------

# =====================================================================
# CONFIG — set your kymograph folder here
# =====================================================================
kymo_root = Path(
    str(DATA_ROOT / "vesicles" / "confocal_2025" / "7 Sep 2025 membrane test" / "4th animal" / "good" / "Image 21_after_75min" / "track_1" / "kymograph")
).expanduser().resolve()

# =====================================================================
# UTILITIES
# =====================================================================

def load_grayscale_array(path: Path) -> np.ndarray:
    """
    Load an image as grayscale float array in [0,1]. Robust for 8/16-bit.
    NOTE: For multi-page TIFFs, this reads the first page.
    """
    img = Image.open(path).convert('L')
    arr = np.asarray(img, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"Empty image: {path}")
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr


def load_mask_array(path: Path, target_shape) -> np.ndarray:
    """
    Load a mask as a strict binary array (True/False). Resizes to target_shape if needed.
    """
    m = Image.open(path).convert('L')
    if m.size[::-1] != target_shape:
        m = m.resize((target_shape[1], target_shape[0]), Image.NEAREST)
    marr = np.asarray(m, dtype=np.uint8)
    return marr > 0


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_track_points_csv(csv_path: Path, points):
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x', 'y'])
        w.writerows(points)


def color_overlay(gray_img_path: Path, list_of_tracks, colors=None) -> np.ndarray:
    """
    Make an RGB overlay on top of 0..1 grayscale image using colored single-pixel tracks.
    list_of_tracks: list of (track_img_uint8, color_tuple_or_None).
    """
    #load image without rescaling
    gray_img_01 = cv2.imread(str(gray_img_path), cv2.IMREAD_GRAYSCALE)
    if gray_img_01 is None:
        raise ValueError(f"Could not read image: {gray_img_path}")  

    H, W = gray_img_01.shape
    # base = (gray_img_01 * 255.0).astype(np.uint8)
    base = gray_img_01.astype(np.uint8)
    overlay = np.stack([base, base, base], axis=-1)  # HxWx3

    if colors is None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                  (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for i, (track_img, color) in enumerate(list_of_tracks):
        c = color if color is not None else colors[i % len(colors)]
        m = (track_img == 255)
        overlay[m] = c

    return overlay

# =====================================================================
# TRACK EXTRACTION (STRICT, MASK-ONLY)
# =====================================================================

def brightest_centerline_strict(enhanced_img: np.ndarray, mask_bool: np.ndarray):
    """
    STRICT rule inside the mask ONLY, row by row:

      - Take all pixels within mask on row y.
      - Choose the brightest pixel(s).
      - If 1 brightest pixel -> pick it.
      - If multiple brightest pixels:
          (A) If they form ONE continuous run -> choose the CENTER of that run.
          (B) If they form MULTIPLE runs -> choose the pixel closest to the MASK-ROW CENTER
              (midpoint of masked span); tie-break by smaller x.

    Rows with no mask pixels are SKIPPED (no fabricated points).
    Returns: points list of (x,y), and a uint8 track image (single pixel per chosen row).
    """
    H, W = enhanced_img.shape
    track_img = np.zeros((H, W), dtype=np.uint8)
    points = []

    for y in range(H):
        row_mask = mask_bool[y, :]
        if not np.any(row_mask):
            continue  # strict: do nothing on this row

        xs = np.flatnonzero(row_mask)
        row_vals = np.nan_to_num(enhanced_img[y, xs], nan=0.0)
        max_val = np.max(row_vals)
        xs_bright = xs[np.isclose(row_vals, max_val)]

        if xs_bright.size == 1:
            x_c = int(xs_bright[0])

        else:
            # Build compact span to label 1D connectivity among brightest pixels
            span_min, span_max = xs.min(), xs.max()
            span_len = span_max - span_min + 1
            span_bright = np.zeros(span_len, dtype=bool)
            span_bright[(xs_bright - span_min)] = True

            lab_arr, num_labels = label(span_bright.astype(np.uint8))
            if num_labels == 1:
                # one continuous region → choose center of that region
                run_idx = np.flatnonzero(span_bright)
                center_in_span = (run_idx[0] + run_idx[-1]) / 2.0
                x_c = int(round(span_min + center_in_span))
            else:
                # multiple distinct brightest regions → choose closest to mask-row center
                mask_center = 0.5 * (xs.min() + xs.max())  # deterministic midpoint of mask span
                dists = np.abs(xs_bright - mask_center)
                # tie-break deterministically: smallest distance, then smaller x
                x_c = int(xs_bright[np.where(dists == dists.min())[0]].min())

        # write result strictly within mask row
        track_img[y, x_c] = 255
        points.append((x_c, y))

    return points, track_img

# =====================================================================
# CORE PIPELINE (ONE MASK AT A TIME)
# =====================================================================

def process_one_image_with_masks(img_path: Path, masks_dir: Path, out_root: Path, img_name_start: str):
    """
    Mask-strict recomputation for a single reslice image:
      - Enhance image once (CLAHE).
      - Iterate masks ONE BY ONE.
      - For each mask, compute track ONLY inside that mask (strict).
      - Save per-mask track (PNG), per-mask coords (CSV), per-mask overlay (PNG).
      - Also save a combined overlay at the end (for visualization only).
    """
    #check if there is an image that exists starting with img_name_start
    img_files = sorted(glob.glob(str(img_path.parent / f"{img_name_start}*.tif")))
    if not img_files:
        print(f"[WARN] No image found starting with: {img_name_start} in {img_path.parent}")
        return

    img_path = img_files[0]  # take the first match
    img_path = Path(img_path)

    if not img_path.exists():
        print(f"[WARN] Missing image: {img_path}")
        return

    # Only image-like files as masks
    mask_files = sorted(
        p for p in masks_dir.glob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
    )
    if not mask_files:
        print(f"[WARN] No masks found in: {masks_dir}")
        return

    # Load + enhance once
    img = load_grayscale_array(img_path)
    enh = exposure.equalize_adapthist(img, clip_limit=0.01)

    # Outputs
    tracks_dir   = out_root / 'tracks'     # per-mask single-pixel tracks
    overlays_dir = out_root / 'overlays'   # per-mask and combined overlays
    coords_dir = out_root / 'coords'
    ensure_dir(tracks_dir)
    ensure_dir(overlays_dir)
    ensure_dir(coords_dir)

    colored_tracks = []
    for idx, mpath in enumerate(mask_files, start=1):
        mask_bool = load_mask_array(mpath, img.shape)
        if not np.any(mask_bool):
            print(f"[INFO] Empty mask skipped: {mpath}")
            continue

        # points, track_img = skeleton_centerline(enh, mask_bool,
        #                                 min_object_size=20,
        #                                 opening_radius=1,
        #                                 percentile_fallback=70)

        points, track_img = brightest_centerline_strict(enh, mask_bool)

        # Save track image
        timg = Image.fromarray(track_img)  # single-pixel path
        timg.save(tracks_dir / f"track_{idx:03d}.png")

        # Save coordinates
        save_track_points_csv(coords_dir / f"track_{idx:03d}.csv", points)

        colored_tracks.append((track_img, None))  # color assigned later

    # Save overlay
    if colored_tracks:
        overlay = color_overlay(img_path, colored_tracks)
        Image.fromarray(overlay).save(overlays_dir / "all_tracks_overlay.png")
        print(f"[OK] Saved overlay: {overlays_dir / 'all_tracks_overlay.png'}")
    else:
        print("[WARN] No non-empty masks to overlay.")




def recalc_tracks_for_kymo_folder(kymo_folder: Path):
    """
    For a single kymograph folder:
      - Finds 'Reslice (AVRG) of images.tif' and 'Reslice (MAX) of images.tif'
      - Uses masks in 'segmented_tracks/masks/'
      - Writes results to 'tracks_from_masks_AVRG' and 'tracks_from_masks_MAX'
    """
    avrg_name = "Reslice (AVRG) of images.tif"
    max_name = "Reslice (MAX) of images.tif"

    avrg_name_start = "Reslice (AVRG) of"
    max_name_start = "Reslice (MAX) of"

    avrg_path = kymo_folder / avrg_name
    max_path = kymo_folder / max_name

    masks_dir = kymo_folder / "segmented_tracks" / "masks"
    if not masks_dir.exists():
        print(f"[WARN] Masks directory not found: {masks_dir}")
        return

    out_avrg = kymo_folder / "tracks_from_masks_AVRG"
    out_max  = kymo_folder / "tracks_from_masks_MAX"

    ensure_dir(out_avrg)
    ensure_dir(out_max)

    print(f"\n=== Processing: {kymo_folder} ===")
    process_one_image_with_masks(avrg_path, masks_dir, out_avrg, avrg_name_start)
    process_one_image_with_masks(max_path, masks_dir, out_max, max_name_start)

if __name__ == "__main__":
    # Directly process the given folder (no user input)
    if not kymo_root.exists():
        print(f"[ERROR] Folder not found: {kymo_root}")
    else:
        recalc_tracks_for_kymo_folder(kymo_root)
        print("\n✅ Done recalculating all tracks.")
