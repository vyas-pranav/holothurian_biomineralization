# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Apply one painted path mask to every frame of a time-lapse.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 5D–G; Fig. S16A–E (kymograph preparation)
What it does  : Reads the binary mask made with draw_track.py (resized to the frame size
                if needed) and writes every frame (8-bit grey) multiplied by the mask, so
                that only that path is left for the kymograph step in Fiji.
Inputs        : <Image folder>/aligned_images/* (frames) and <Image folder>/track_<k>.png
                (here: DATA_ROOT/vesicles/confocal_2025/5 Sep 2025 membrane test/session2/good/Image 17)
Outputs       : <Image folder>/track_<k>/images/<same file names> (masked frames)
Environment   : environment-analysis.yml (Python 3.7)
Run           : python apply_mask.py   (set input_folder, output_folder and mask_path per path)
"""

import cv2
import numpy as np
import os
import glob
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

# Specify the paths using raw strings or forward slashes
input_folder = str(DATA_ROOT / "vesicles" / "confocal_2025" / "5 Sep 2025 membrane test" / "session2" / "good" / "Image 17" / "aligned_images")
output_folder = str(DATA_ROOT / "vesicles" / "confocal_2025" / "5 Sep 2025 membrane test" / "session2" / "good" / "Image 17" / "track_3" / "images")
mask_path = str(DATA_ROOT / "vesicles" / "confocal_2025" / "5 Sep 2025 membrane test" / "session2" / "good" / "Image 17" / "track_3.png")

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the binary mask image in grayscale
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    print(f"Error: Mask image not found at {mask_path}")
    exit()

# Get a list of all image files in the input folder
image_files = glob.glob(os.path.join(input_folder, '*.*'))
supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

for img_file in image_files:
    # Check if the file is an image
    ext = os.path.splitext(img_file)[1].lower()
    if ext not in supported_formats:
        continue  # Skip non-image files

    # Load the image in grayscale
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image {img_file}")
        continue

    # Ensure the mask and image are the same size
    if img.shape != mask.shape:
        resized_mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        resized_mask = mask

    # Apply threshold to ensure binary mask (if not already binary)
    _, binary_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)

    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img, img, mask=binary_mask)

    # Save the masked image in grayscale
    output_path = os.path.join(output_folder, os.path.basename(img_file))
    cv2.imwrite(output_path, masked_img)
    print(f"Processed and saved: {output_path}")
