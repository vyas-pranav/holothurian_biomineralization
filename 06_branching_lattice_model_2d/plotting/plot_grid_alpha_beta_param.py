# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Montage of final simulated ossicles over an (alpha, beta) grid.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Figs. S21-S23
What it does  : Reads final_ossicle_pruned.png from every alpha_<a>_beta_<b> run folder
                of one cell_pair_force_sweep3.py study, crops a 3000 x 3000 px window
                around the ossicle, tiles the crops (rows: alpha, columns: beta, both
                increasing), writes "alpha=<a>, beta=<b>" (Greek letters) on each tile
                and saves the montage resized to 12000 x 12000 px.
Inputs        : outputs of ../sweeps/cell_pair_force_sweep3.py: OUT_ROOT/
                06_branching_lattice_model_2d/Network simulations/
                alpha_beta_phase_space_study/<study>/ (base_folder; the original lists
                the studies 20250401_052125, 20250401_051905, 20251106_104314 and
                20251106_114503, the last one active)
Outputs       : <study>/combined_grid_replot_12000_resized.png
Environment   : environment-analysis.yml (Python 3.7); needs Pillow < 10
                (uses ImageDraw.textsize and Image.ANTIALIAS)
Run           : python plot_grid_alpha_beta_param.py
"""

import os
import math
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

# --- Configuration ---
# Base folder that contains subfolders from the simulation study.
# Each subfolder should contain a file named "final_ossicle.png".
# base_folder = str(OUT_ROOT / "06_branching_lattice_model_2d" / "Network simulations" / "alpha_beta_phase_space_study" / "20250401_052125")
# base_folder = str(OUT_ROOT / "06_branching_lattice_model_2d" / "Network simulations" / "alpha_beta_phase_space_study" / "20250401_051905")
# base_folder = str(OUT_ROOT / "06_branching_lattice_model_2d" / "Network simulations" / "alpha_beta_phase_space_study" / "20251106_104314")
base_folder = str(OUT_ROOT / "06_branching_lattice_model_2d" / "Network simulations" / "alpha_beta_phase_space_study" / "20251106_114503")

# Output file for the combined grid image.
output_image_path = os.path.join(base_folder, "combined_grid_2.png")

# Grid settings
margin = 1          # margin between images in the grid (in pixels)
text_padding = 5     # padding from the top of each image for the text
font_size = 250       # font size for the overlay text


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

# --- Helper function to parse parameters from subfolder name ---
def parse_parameters(folder_name):
    """
    Assumes folder names of the form:
         "alpha_<alpha_value>_beta_<beta_value>"
    Returns a tuple of floats: (alpha_value, beta_value)
    """
    try:
        parts = folder_name.split('_')
        # Expecting parts like ["alpha", "<alpha_val>", "beta", "<beta_val>"]
        alpha_val = float(parts[1])
        beta_val  = float(parts[3])
        return alpha_val, beta_val
    except Exception as e:
        print(f"Error parsing parameters from folder name '{folder_name}': {e}")
        return None, None



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



# --- Load images and their parameter values ---
data = []  # list of tuples: (alpha, beta, label, image)
subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]

for subfolder in subfolders:
    final_img_path = os.path.join(subfolder, "final_ossicle_pruned.png")
    if os.path.exists(final_img_path):
        try:
            img = Image.open(final_img_path).convert("RGB")
            # img = keep_central_region(img, 3600, 3600)
            img = crop_from_center(img, 3000, 3000, 1955, 2273)

            label = os.path.basename(subfolder)  # label is the folder name
            alpha_val, beta_val = parse_parameters(label)
            if alpha_val is not None and beta_val is not None:
                data.append((alpha_val, beta_val, label, img))
        except Exception as e:
            print(f"Error loading image from {final_img_path}: {e}")

if not data:
    print("No final plot images found in the subfolders!")
    exit()

# --- Sort the data by alpha and then by beta ---
data.sort(key=lambda x: (x[0], x[1]))

# --- Determine unique parameter values ---
unique_alpha = sorted({entry[0] for entry in data})
unique_beta  = sorted({entry[1] for entry in data})
n_rows = len(unique_alpha)
n_cols = len(unique_beta)
print(f"Found {n_rows} unique alpha values and {n_cols} unique beta values.")

# --- Create a dictionary mapping (alpha, beta) to (label, image) ---
grid_dict = {}
for alpha, beta, label, img in data:
    grid_dict[(alpha, beta)] = (label, img)

# --- Assume all images are the same size (or resize them as needed) ---
# We use the size of the first image in our sorted list.
sample_img = data[0][3]
img_width, img_height = sample_img.size

# Calculate size of the combined image.
grid_width = n_cols * img_width + (n_cols + 1) * margin
grid_height = n_rows * img_height + (n_rows + 1) * margin

combined_img = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))

# Try to load a TrueType font; if not available, use default.
try:
    font = ImageFont.truetype("arial.ttf", font_size)
except Exception as e:
    print("TrueType font not found, using default font.")
    font = ImageFont.load_default()

draw = ImageDraw.Draw(combined_img)


offset_x = img_width / 16
offset_y = img_height / 16

# --- Paste images and annotate with labels ---
for row_idx, alpha in enumerate(unique_alpha):
    for col_idx, beta in enumerate(unique_beta):
        x = margin + col_idx * (img_width + margin)
        y = margin + row_idx * (img_height + margin)
        
        # Look up the image for (alpha, beta)x
        if (alpha, beta) in grid_dict:
            label, img = grid_dict[(alpha, beta)]
        else:
            # If missing, create a blank image
            img = Image.new("RGB", (img_width, img_height), color=(200, 200, 200))
            label = f"alpha_{alpha:.4e}_beta_{beta:.4e}"
        
        # Paste the image
        combined_img.paste(img, (x, y))
        
        # Create label text from alpha and beta values
        text = f"α={alpha}, β={beta}"
        text_width, text_height = draw.textsize(text, font=font)

        # Position the text so that its top right corner is offset by offset_x and offset_y from the image's top right.
        text_x = x + offset_x 
        text_y = y + offset_y
        
        # Draw a semi-transparent rectangle behind text for contrast
        rect_margin = 2
        rect_coords = [x, y, x + img_width, y + text_height + 2 * rect_margin]
        # draw.rectangle(rect_coords, fill=(255, 255, 255))
        
        # Draw the text over the rectangle
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

# --- Save the combined grid image ---
# combined_img.save(output_image_path)
# print(f"Combined grid image saved to: {output_image_path}")

#resize the saved image to 3000x3000
resized_image_path = os.path.join(base_folder, "combined_grid_replot_12000_resized.png")
resized_img = combined_img.resize((12000, 12000), Image.ANTIALIAS)
resized_img.save(resized_image_path, format="PNG", dpi=(100, 100))
print(f"Resized image saved to: {resized_image_path}")  