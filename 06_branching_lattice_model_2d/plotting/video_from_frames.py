# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Assemble numbered PNG frames into an AVI video.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Video S8
What it does  : Collects the *.png files in img_folder, orders them by the first number
                in each file name (e.g. ossicle_time_<ii>.png) and writes them to an
                XVID-encoded AVI at 25 frames/s.
Inputs        : img_folder. As found it points at the run D2 = 0.055, k4 = 0.01,
                vm = 0.5 of ../../05_branch_transport_model_1p5d/para_sweep_run.py (the
                advection-reaction-diffusion clip in Video S8); for the 2D model point it
                at a plots/ folder written by ../sweeps/seed_edge_sweep3.py (a frame at
                every step).
Outputs       : video_name (output_video.avi in the frame folder)
Environment   : environment-analysis.yml (Python 3.7, OpenCV 4.5)
Run           : python video_from_frames.py
"""

import cv2
import os
import glob
import re

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

# Specify the folder containing your PNG frames
img_folder = str(OUT_ROOT / "05_branch_transport_model_1p5d" / "sweep_study_dimensional" / "phase_space_study_2025_02_19_14_02_25" / "D2_0.05500000000000001_k4_0.01_vm_0.5")  # Replace with your folder path
video_name = str(OUT_ROOT / "05_branch_transport_model_1p5d" / "sweep_study_dimensional" / "phase_space_study_2025_02_19_14_02_25" / "D2_0.05500000000000001_k4_0.01_vm_0.5" / "output_video.avi")     # Output video file name

# Get a list of all PNG files in the folder
images = glob.glob(os.path.join(img_folder, '*.png'))


# Function to extract a number from the filename
def extract_number(filename):
    # Extract numbers from the base name of the file
    numbers = re.findall(r'\d+', os.path.basename(filename))
    if numbers:
        return int(numbers[0])  # Assumes the first number is the desired sort key
    return -1  # Return a default value if no number is found

# Sort images based on the number in the filename
images = sorted(images, key=extract_number)


# Check if any images were found
if not images:
    raise ValueError("No PNG images found in the specified folder.")

# Read the first image to get the dimensions (assumes all images are the same size)
frame = cv2.imread(images[0])
height, width, layers = frame.shape
size = (width, height)

# Define the codec and create VideoWriter object
fps = 25 # Change this value to set the frames per second
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec if needed
video = cv2.VideoWriter(video_name, fourcc, fps, size)

# Loop over all images and write them to the video
for image in images:
    frame = cv2.imread(image)
    if frame is None:
        print(f"Warning: Skipping file {image}, could not read as an image.")
        continue
    video.write(frame)

# Release the VideoWriter object
video.release()

print("Video created successfully!")
