# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Pool the track images of one time-lapse into pooled_AVRG / pooled_MAX.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 5E–G; Fig. S16A–C
What it does  : For every track_* folder of one image it copies the one-pixel track
                images from kymograph/tracks_from_masks_{AVRG,MAX}/tracks (or the older
                kymograph/segmented_tracks_{AVRG,MAX}/tracks) into pooled_{AVRG,MAX}/tracks,
                prefixing each file name with the track folder name.
Inputs        : <root_dir>/track_*/kymograph/tracks_from_masks_{AVRG,MAX}/tracks/*.png
Outputs       : <root_dir>/pooled_AVRG/tracks/ and <root_dir>/pooled_MAX/tracks/
Environment   : environment-analysis.yml (Python 3.7)
Run           : python pool_tracks_v2.py   (set root_dir to one Image folder)
"""

import os
import shutil
from pathlib import Path
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

# Root directory where your "track_XX" folders are located
root_dir = str(DATA_ROOT / "vesicles" / "confocal_2025" / "7 Sep 2025 membrane test" / "4th animal" / "good" / "Image 20")  # Change this to your base directory if needed

# Output folders
pooled_avrg_dir = os.path.join(root_dir, "pooled_AVRG")
pooled_max_dir  = os.path.join(root_dir, "pooled_MAX")

# (keep a 'tracks' subfolder to mirror your original structure)
pooled_avrg_tracks = os.path.join(pooled_avrg_dir, "tracks")
pooled_max_tracks  = os.path.join(pooled_max_dir,  "tracks")

# Create the pooled directories if they don't exist
os.makedirs(pooled_avrg_tracks, exist_ok=True)
os.makedirs(pooled_max_tracks,  exist_ok=True)

# Possible source subpaths for AVRG and MAX (add more variants if needed)
SOURCE_SUBPATHS = {
    "AVRG": [
        os.path.join("kymograph", "tracks_from_masks_AVRG", "tracks"),
        os.path.join("kymograph", "segmented_tracks_AVRG", "tracks"),
    ],
    "MAX": [
        os.path.join("kymograph", "tracks_from_masks_MAX", "tracks"),
        os.path.join("kymograph", "segmented_tracks_MAX", "tracks"),
    ],
}

VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

def copy_tracks_for_mode(track_folder_path: str, track_folder_name: str, mode: str, dest_dir: str):
    """
    Try each candidate source path for the given mode (AVRG or MAX), and copy images to dest_dir.
    """
    found_any = False
    for rel_sub in SOURCE_SUBPATHS[mode]:
        src_path = os.path.join(track_folder_path, rel_sub)
        if os.path.exists(src_path):
            for file in os.scandir(src_path):
                if file.is_file() and file.name.lower().endswith(VALID_EXTS):
                    dest_file = os.path.join(dest_dir, f"{track_folder_name}_{file.name}")
                    shutil.copy(file.path, dest_file)
                    print(f"Copied {file.path} -> {dest_file}")
                    found_any = True
            # If one variant existed, we don't need to try others for this mode.
            break
    if not found_any:
        # Report all tried paths for easier debugging
        tried = [os.path.join(track_folder_path, p) for p in SOURCE_SUBPATHS[mode]]
        print(f"[{mode}] No tracks found for {track_folder_name}. Tried:\n  - " + "\n  - ".join(tried))

# Loop through all subdirectories in the root directory
for track_folder in os.scandir(root_dir):
    if track_folder.is_dir() and track_folder.name.startswith("track_"):
        # AVRG
        copy_tracks_for_mode(track_folder.path, track_folder.name, mode="AVRG", dest_dir=pooled_avrg_tracks)
        # MAX
        copy_tracks_for_mode(track_folder.path, track_folder.name, mode="MAX",  dest_dir=pooled_max_tracks)

print("✅ Pooling complete.")
print("   AVRG pooled into:", pooled_avrg_dir)
print("   MAX  pooled into:", pooled_max_dir)
