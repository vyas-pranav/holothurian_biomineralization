# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Montage of final simulated ossicles for a one-parameter sweep.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. S26 (PANEL = "Fig. S26")
What it does  : Reads final_ossicle_pruned.png from every <param>_<value> run folder of
                a one-parameter sweep study, crops 3000 x 3000 px around the ossicle,
                tiles the crops in a near-square grid ordered by the parameter value,
                labels each tile "<param> = <value>" and saves the montage. The study
                folder and the parameter name are chosen with PANEL from SETTINGS:
                "release_rate_sweep" (default) reproduces the values active in the
                original (a morphogen release-rate sweep whose script is not part of
                this release); "Fig. S26" is the angle-noise study.
Inputs        : for Fig. S26, outputs of ../sweeps/angle_noise_sweep3.py: OUT_ROOT/
                06_branching_lattice_model_2d/Network simulations/angle_noise_sweep_study/
                20250402_231059/
Outputs       : <study>/combined_grid_2.png
Environment   : environment-analysis.yml (Python 3.7); needs Pillow < 10
                (uses ImageDraw.textsize)
Run           : python plot_grid_single_param.py
"""

import os
import math
import re
from PIL import Image, ImageDraw, ImageFont

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

# --- Function to crop and keep only central region of an image ---
def keep_central_region(img, width, height):
    """
    Crop and keep only the central region of the image.
    
    Parameters:
        img (PIL.Image): The input image.
        width (int): The width of the central region to keep.
        height (int): The height of the central region to keep.
        
    Returns:
        PIL.Image: The cropped image.
    """
    img_width, img_height = img.size
    left = (img_width - width) // 2
    top = (img_height - height) // 2
    right = left + width
    bottom = top + height
    return img.crop((left, top, right, bottom))

#crop images based on new center  and width and height
def crop_from_center(img, width, height, center_x, center_y):
    """
    Crop an image from the center to the specified width and height.
    
    Parameters:
        img (PIL.Image): The input image.
        width (int): The width of the cropped region.
        height (int): The height of the cropped region.
        center_x (int): The x-coordinate of the center of the crop region.
        center_y (int): The y-coordinate of the center of the crop region.
        
    Returns:
        PIL.Image: The cropped image.
    """
    img_width, img_height = img.size
    left = center_x - width // 2
    top = center_y - height // 2
    right = left + width
    bottom = top + height
    return img.crop((left, top, right, bottom))



# --- Generic parser for a single parameter from a folder name ---
def parse_single_parameter(folder_name, parameter_name):
    """
    Parse the given folder name for a parameter value associated with parameter_name.
    
    Expects folder names containing a substring of the form:
         "<parameter_name>_<value>"
    For example, if parameter_name is "A" and the folder name is "A_3.0000",
    the function returns 3.0000 as a float.
    
    Parameters:
        folder_name (str): The name of the folder.
        parameter_name (str): The parameter to parse.
        
    Returns:
        float: The parsed parameter value if found; otherwise, None.
    """
    pattern = rf"{parameter_name}_([\d\.eE+-]+)"
    match = re.search(pattern, folder_name)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            print(f"Could not convert {match.group(1)} to float for parameter {parameter_name}.")
    else:
        print(f"Parameter {parameter_name} not found in folder name: {folder_name}")
    return None

# --- Configuration ---
# Study folders recorded in the original, selected with PANEL; "release_rate_sweep"
# reproduces the values that were active in the original.
SETTINGS = {
    "Fig. S26": {
        "base_folder": str(OUT_ROOT / "06_branching_lattice_model_2d" / "Network simulations" / "angle_noise_sweep_study" / "20250402_231059"),
        "sweep_param": "angle_noise",
    },
    "release_rate_sweep": {
        "base_folder": str(OUT_ROOT / "06_branching_lattice_model_2d" / "Network simulations" / "morphogen_release_rate_sweep_study" / "20251106_174443"),
        "sweep_param": "release_rate",
    },
}
PANEL = "release_rate_sweep"
base_folder = SETTINGS[PANEL]["base_folder"]
sweep_param = SETTINGS[PANEL]["sweep_param"]
output_image_path = os.path.join(base_folder, "combined_grid_2.png")

# Grid settings
margin = 1          # margin between images in the grid (in pixels) 
font_size = 200      # desired font size for the overlay text

# --- Load images and corresponding parameter values ---
data = []  # List of tuples: (parameter_value, folder_label, image)
subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]

for subfolder in subfolders:
    image_path = os.path.join(subfolder, "final_ossicle_pruned.png")
    if os.path.exists(image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            # Crop and keep only central region (adjust size as needed)
            # img = keep_central_region(img, 3600, 3600)
            img = crop_from_center(img, 3000, 3000, 1955, 2273)

            folder_label = os.path.basename(subfolder)
            param_value = parse_single_parameter(folder_label, sweep_param)
            if param_value is not None:
                data.append((param_value, folder_label, img))
        except Exception as e:
            print(f"Error loading image from {image_path}: {e}")

if not data:
    print("No images found in the subfolders!")
    exit()

# --- Sort the data by the parameter value ---
data.sort(key=lambda x: x[0])

# --- Compute grid dimensions ---
n_images = len(data)
cols = math.ceil(math.sqrt(n_images))
rows = math.ceil(n_images / cols)
print(f"Found {n_images} images. Arranging in a grid with {rows} rows and {cols} columns.")

# Assume all images are the same size.
sample_img = data[0][2]
img_width, img_height = sample_img.size

# Calculate combined image size
grid_width = cols * img_width + (cols + 1) * margin
grid_height = rows * img_height + (rows + 1) * margin

combined_img = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))
draw = ImageDraw.Draw(combined_img)

# --- Load a scalable TrueType font ---
# Try a few common fonts; fall back to default only if none found.
font = None
for font_name in ["arial.ttf", "DejaVuSans.ttf"]:
    try:
        font = ImageFont.truetype(font_name, font_size)
        break
    except Exception:
        continue
if font is None:
    print("Custom TrueType fonts not found, using default font (font size may not change).")
    font = ImageFont.load_default()

# --- Paste images in grid and annotate with parameter values ---
# We want the text to appear in the top right corner of each image,
# with an offset equal to 1/10th of the image width from the top and right edges.
offset_x = img_width / 16
offset_y = img_height / 16

for idx, (param_value, label, img) in enumerate(data):
    row = idx // cols
    col = idx % cols
    x = margin + col * (img_width + margin)
    y = margin + row * (img_height + margin)
    
    # Paste the image into the grid.
    combined_img.paste(img, (x, y))
    
    # Create text annotation from the parameter value.
    text = f"{sweep_param} = {param_value}"
    text_width, text_height = draw.textsize(text, font=font)
    
    # Position the text so that its top right corner is offset by offset_x and offset_y from the image's top right.
    text_x = x + offset_x 
    text_y = y + offset_y
    
    # Optionally, draw a semi-transparent background for better text visibility.
    background_padding = 5
    background_box = [
        text_x - background_padding,
        text_y - background_padding,
        text_x + text_width + background_padding,
        text_y + text_height + background_padding,
    ]
    # draw.rectangle(background_box, fill=(255, 255, 255, 128))
    
    draw.text((text_x, text_y), text, fill="red", font=font)

# --- Save the combined grid image ---
combined_img.save(output_image_path)
print(f"Combined grid image saved to: {output_image_path}")
