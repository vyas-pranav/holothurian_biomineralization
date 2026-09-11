# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Montage of final simulated ossicles over a (k, D) grid.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. S24 (its tiles are also the images of Figs. 6L-M)
What it does  : Reads final_ossicle_pruned_new.png (written by replot_final_pruned.py)
                from every lambda_<k>_D_<D> run folder of one lambda_D_sweep3.py study,
                crops 3000 x 3000 px around the ossicle, tiles the crops (rows: k,
                columns: D, both increasing), labels each tile "k=<k>, D=<D>" and saves
                the montage resized to 12000 x 12000 px.
Inputs        : OUT_ROOT/06_branching_lattice_model_2d/Network simulations/
                lambda_D_phase_space_study/20250325_220918/ (the study recorded in the
                original; outputs of ../sweeps/lambda_D_sweep3.py after
                replot_final_pruned.py)
Outputs       : <study>/combined_grid_replot_12000_resized.png
Environment   : environment-analysis.yml (Python 3.7); needs Pillow < 10
                (uses ImageDraw.textsize and Image.ANTIALIAS)
Run           : python replot_final_pruned.py, then python plot_grid_lambda_D_param.py
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
# Each subfolder should contain a file named "final_ossicle_pruned.png".
base_folder = str(OUT_ROOT / "06_branching_lattice_model_2d" / "Network simulations" / "lambda_D_phase_space_study" / "20250325_220918")

# Output file for the combined grid image.
output_image_path = os.path.join(base_folder, "combined_grid_replot_dpi10.png")

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


# --- Helper function to parse parameters from subfolder name ---
def parse_parameters(folder_name):
    """
    Assumes folder names of the form:
         "lambda_<lambda_value>_D_<D_value>"
    Returns a tuple of floats: (lambda_value, D_value)
    """
    try:
        parts = folder_name.split('_')
        # Expecting parts like: ["lambda", "<lambda_val>", "D", "<D_val>"]
        lambda_val = float(parts[1])
        D_val = float(parts[3])
        return lambda_val, D_val
    except Exception as e:
        print(f"Error parsing parameters from folder name '{folder_name}': {e}")
        return None, None

# --- Load images and their parameter values ---
data = []  # list of tuples: (lambda, D, label, image)
subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]

for subfolder in subfolders:
    final_img_path = os.path.join(subfolder, "final_ossicle_pruned_new.png")
    if os.path.exists(final_img_path):
        try:
            img = Image.open(final_img_path).convert("RGB")
            # Crop image to its central 3600x3600 region
            # img = keep_central_region(img, 3600, 3600)
            img = crop_from_center(img, 3000, 3000, 1955, 2273)
            label = os.path.basename(subfolder)  # label is the folder name
            lambda_val, D_val = parse_parameters(label)
            if lambda_val is not None and D_val is not None:
                data.append((lambda_val, D_val, label, img))
        except Exception as e:
            print(f"Error loading image from {final_img_path}: {e}")

if not data:
    print("No final plot images found in the subfolders!")
    exit()

# --- Sort the data by lambda and then by D ---
data.sort(key=lambda x: (x[0], x[1]))

# --- Determine unique parameter values ---
unique_lambda = sorted({entry[0] for entry in data})
unique_D = sorted({entry[1] for entry in data})
n_rows = len(unique_lambda)
n_cols = len(unique_D)
print(f"Found {n_rows} unique lambda values and {n_cols} unique D values.")

# --- Create a dictionary mapping (lambda, D) to (label, image) ---
grid_dict = {}
for lambda_val, D_val, label, img in data:
    grid_dict[(lambda_val, D_val)] = (label, img)

# --- Assume all images are the same size (or resize them as needed) ---
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

# Offsets for text placement relative to each image's top-right corner.
offset_x = img_width / 16
offset_y = img_height / 16

# --- Paste images and annotate with labels ---
for row_idx, lambda_val in enumerate(unique_lambda):
    for col_idx, D_val in enumerate(unique_D):
        x = margin + col_idx * (img_width + margin)
        y = margin + row_idx * (img_height + margin)
        
        # Look up the image for (lambda, D)
        if (lambda_val, D_val) in grid_dict:
            label, img = grid_dict[(lambda_val, D_val)]
        else:
            # If missing, create a blank image
            img = Image.new("RGB", (img_width, img_height), color=(200, 200, 200))
            label = f"lambda_{lambda_val:.4e}_D_{D_val:.4e}"
        
        # Paste the image onto the combined grid
        combined_img.paste(img, (x, y))
        
        # Create label text from lambda and D values.
        text = f"k={lambda_val}, D={D_val}"
        text_width, text_height = draw.textsize(text, font=font)
        
        # Position the text with a slight offset from the top-right of the image.
        text_x = x + offset_x 
        text_y = y + offset_y
        
        # Optionally, draw a background rectangle behind the text for better contrast.
        rect_margin = 2
        rect_coords = [x, y, x + img_width, y + text_height + 2 * rect_margin]
        # Uncomment the next line if you wish to have a white rectangle behind the text.
        # draw.rectangle(rect_coords, fill=(255, 255, 255))
        
        # Draw the text over the image.
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

# --- Save the combined grid image ---
# combined_img.save(output_image_path, format="PNG", dpi =(10, 10))
# print(f"Combined grid image saved to: {output_image_path}")


#resize the saved image to 3000x3000
resized_image_path = os.path.join(base_folder, "combined_grid_replot_12000_resized.png")
resized_img = combined_img.resize((12000, 12000), Image.ANTIALIAS)
resized_img.save(resized_image_path, format="PNG", dpi=(100, 100))
print(f"Resized image saved to: {resized_image_path}")  