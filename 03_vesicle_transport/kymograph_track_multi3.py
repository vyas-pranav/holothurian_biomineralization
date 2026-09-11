# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Draw regions on a kymograph and extract one vesicle track per region (Tk GUI).

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 5D–G; Fig. S16A–E
What it does  : Open a kymograph (Fiji KymoResliceWide output: x = position along the
                path, y = time), paint one region per vesicle trace ('Add Region') or
                import saved masks, then 'Process All Tracks': in every time row of a
                region the track point is the intensity-weighted centre of the
                CLAHE-enhanced kymograph.
Inputs        : a kymograph chosen in the file dialog (e.g. track_<k>/kymograph/Reslice (AVRG) of images.tif)
Outputs       : <kymograph folder>/segmented_tracks/masks/mask_<i>.png, tracks/track_<i>.png
                and all_tracks_overlay.png
Environment   : environment-analysis.yml (Python 3.7)
Run           : python kymograph_track_multi3.py   (interactive Tk window; never run unattended)
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from skimage import exposure, measure
import os
from scipy.ndimage import label

# Define the KymographAnnotator class
class KymographAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Kymograph Annotator")

        # Initialize variables
        self.kymograph_path = None
        self.original_image = None
        self.display_image = None
        self.masks = []  # List to store multiple masks
        self.current_mask = None  # Current mask for drawing
        self.zoom_level = 0.08  # Initial zoom level at 8%
        self.brush_size = 5
        self.eraser_mode = False
        self.drawing = False
        self.last_x, self.last_y = None, None

        # Directory for saving results
        self.output_dir = None

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        # Frame for buttons and sliders
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Open Kymograph Button
        open_image_button = tk.Button(control_frame, text="Open Kymograph", command=self.open_kymograph)
        open_image_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Import Mask Button
        import_mask_button = tk.Button(control_frame, text="Import Mask", command=self.import_mask)
        import_mask_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Pen Button
        self.pen_button = tk.Button(control_frame, text="Pen", command=self.select_pen, relief=tk.SUNKEN)
        self.pen_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Eraser Button
        self.eraser_button = tk.Button(control_frame, text="Eraser", command=self.select_eraser)
        self.eraser_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Brush Size Slider
        self.brush_size_slider = tk.Scale(control_frame, from_=1, to=50, orient=tk.HORIZONTAL, label="Brush Size")
        self.brush_size_slider.set(self.brush_size)
        self.brush_size_slider.pack(side=tk.LEFT, padx=5, pady=5)

        # Zoom Slider
        self.zoom_slider = tk.Scale(control_frame, from_=8, to=500, orient=tk.HORIZONTAL, label="Zoom (%)")
        self.zoom_slider.set(int(self.zoom_level * 100))
        self.zoom_slider.pack(side=tk.LEFT, padx=5, pady=5)
        self.zoom_slider.bind("<B1-Motion>", self.update_zoom)
        self.zoom_slider.bind("<ButtonRelease-1>", self.update_zoom)

        # Add Region Button
        add_region_button = tk.Button(control_frame, text="Add Region", command=self.add_new_mask)
        add_region_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Process All Tracks Button
        process_all_button = tk.Button(control_frame, text="Process All Tracks", command=self.process_all_tracks)
        process_all_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Canvas for image display
        self.canvas = tk.Canvas(self.root, cursor="cross", bg='grey')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Scrollbars for canvas
        hbar = tk.Scrollbar(self.canvas, orient=tk.HORIZONTAL)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        hbar.config(command=self.canvas.xview)
        vbar = tk.Scrollbar(self.canvas, orient=tk.VERTICAL)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        vbar.config(command=self.canvas.yview)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        # Bindings for drawing
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

    def open_kymograph(self):
        file_types = [("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif")]
        kymograph_path = filedialog.askopenfilename(title='Select a kymograph', filetypes=file_types)
        if not kymograph_path:
            return

        try:
            # Open the kymograph using PIL
            self.original_image = Image.open(kymograph_path).convert('L')
            self.display_image = self.original_image.copy()
            self.kymograph_path = kymograph_path

            # Initialize the first mask for painting
            self.current_mask = Image.new('L', self.original_image.size, 0)
            self.masks = [self.current_mask]  # Store the first mask

            # Create output directory
            self.output_dir = os.path.join(os.path.dirname(kymograph_path), 'segmented_tracks')
            os.makedirs(self.output_dir, exist_ok=True)

            self.update_canvas()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open kymograph.\n{e}")

    def import_mask(self):
        """Import an existing mask image and add it to the mask list."""
        file_types = [("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif")]
        mask_path = filedialog.askopenfilename(title='Select a mask image', filetypes=file_types)
        if not mask_path:
            return

        try:
            # Load the mask and resize it to match the kymograph size
            imported_mask = Image.open(mask_path).convert('L')
            if imported_mask.size != self.original_image.size:
                imported_mask = imported_mask.resize(self.original_image.size, Image.NEAREST)

            self.masks.append(imported_mask)
            self.current_mask = imported_mask
            self.update_canvas()
            messagebox.showinfo("Mask Imported", "Mask successfully imported and added.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import mask.\n{e}")

    def update_canvas(self):
        if self.display_image is None:
            return

        # Resize the image according to the zoom level
        zoom_percentage = self.zoom_slider.get()
        self.zoom_level = zoom_percentage / 100.0
        width, height = self.display_image.size
        resized_image = self.display_image.resize((int(width * self.zoom_level), int(height * self.zoom_level)), Image.LANCZOS)

        # Composite the mask overlays on the resized display image
        overlay = resized_image.convert("RGBA")
        for mask in self.masks:
            if mask == self.current_mask:
                mask_overlay = mask.resize((int(width * self.zoom_level), int(height * self.zoom_level)), Image.NEAREST).point(lambda p: 255 if p > 0 else 0)
                overlay.paste((255, 0, 0, 128), mask=mask_overlay)  # Red overlay for active mask
            else:
                mask_overlay = mask.resize((int(width * self.zoom_level), int(height * self.zoom_level)), Image.NEAREST).point(lambda p: 128 if p > 0 else 0)
                overlay.paste((255, 255, 0, 80), mask=mask_overlay)  # Yellow overlay for previous masks

        # Convert overlay image to Tkinter format and display on the canvas
        self.tk_image = ImageTk.PhotoImage(overlay)
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)
        self.canvas.image = self.tk_image  # Keep a reference to avoid garbage collection
        self.canvas.config(scrollregion=(0, 0, resized_image.width, resized_image.height))

    def add_new_mask(self):
        """Add a new blank mask layer."""
        new_mask = Image.new('L', self.original_image.size, 0)
        self.masks.append(new_mask)
        self.current_mask = new_mask  # Set the new mask as the current mask
        self.update_canvas()

    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
        self.draw_at_point(event.x, event.y)

    def draw_at_point(self, x, y):
        """Draw a point at the specified location."""
        self.draw_line(x, y, x, y)

    def draw(self, event):
        if not self.drawing:
            return
        current_x, current_y = event.x, event.y
        self.draw_line(self.last_x, self.last_y, current_x, current_y)
        self.last_x, self.last_y = current_x, current_y

    def end_draw(self, event):
        self.drawing = False

    def draw_line(self, x1, y1, x2, y2):
        # Adjust coordinates for zoom level
        x1, y1 = int(x1 / self.zoom_level), int(y1 / self.zoom_level)
        x2, y2 = int(x2 / self.zoom_level), int(y2 / self.zoom_level)

        draw_mask = ImageDraw.Draw(self.current_mask)
        brush_size = self.brush_size_slider.get()
        fill_value = 0 if self.eraser_mode else 255
        draw_mask.line([x1, y1, x2, y2], fill=fill_value, width=brush_size)
        draw_mask.ellipse([x2 - brush_size//2, y2 - brush_size//2, x2 + brush_size//2, y2 + brush_size//2], fill=fill_value)
        self.update_canvas()

    def select_pen(self):
        self.eraser_mode = False
        self.pen_button.config(relief=tk.SUNKEN)
        self.eraser_button.config(relief=tk.RAISED)

    def select_eraser(self):
        self.eraser_mode = True
        self.pen_button.config(relief=tk.RAISED)
        self.eraser_button.config(relief=tk.SUNKEN)

    def process_track(self, mask, track_id):
        """Process a single track, saving the mask and track images."""
        # Convert mask and kymograph to numpy arrays
        mask_array = np.array(mask)
        kymograph_array = np.array(self.original_image)

        # Extract masked region and enhance contrast
        masked_region = kymograph_array * (mask_array > 0)
        enhanced_region = exposure.equalize_adapthist(masked_region)

        # Identify the midpoint of the brightest region in each row (robust version)
        track_points = []
        for y in range(masked_region.shape[0]):
            row_mask = (mask_array[y, :] > 0)
            if not np.any(row_mask):
                continue  # nothing painted in this row

            intens = enhanced_region[y, row_mask]

            # Skip if the masked row has no intensity after enhancement
            if intens.size == 0 or np.all(intens <= 0):
                continue

            x_coords = np.flatnonzero(row_mask)
    
            # Option A: intensity-weighted center (smooth, robust)
            x_center = int(round(np.average(x_coords, weights=intens)))
            track_points.append((x_center, y))

            # If you prefer the single brightest pixel (argmax), swap the above with:
            # x_center = x_coords[np.argmax(intens)]
            # track_points.append((x_center, y))


        # Create a binary image for the track
        track_image = np.zeros_like(mask_array, dtype=np.uint8)
        for x, y in track_points:
            track_image[y, x] = 255  # Mark the track point

        # Save the mask and track images
        mask_image = Image.fromarray(mask_array)
        track_image_pil = Image.fromarray(track_image)
        mask_image.save(os.path.join(self.output_dir, "masks", f"mask_{track_id}.png"))
        track_image_pil.save(os.path.join(self.output_dir, "tracks", f"track_{track_id}.png"))

        return track_image  # Return the track image array for overlay purposes

    def process_all_tracks(self):
        """Process all tracks and generate an overlay image."""
        if not self.kymograph_path:
            messagebox.showwarning("Warning", "Please load a kymograph and draw at least one track.")
            return

        # Set up output directories for individual masks and tracks
        mask_dir = os.path.join(self.output_dir, "masks")
        track_dir = os.path.join(self.output_dir, "tracks")
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(track_dir, exist_ok=True)

        # Prepare an RGB overlay image based on the kymograph
        overlay_image = np.array(self.original_image.convert('RGB'))
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]  # Color choices

        # Process each mask, saving results and building an overlay
        for track_id, mask in enumerate(self.masks, start=1):
            mask_array = np.array(mask)

            # Skip processing if the mask is empty (all zeros)
            if not np.any(mask_array):
                continue

            # Process each track and get the track image array for overlaying
            track_image = self.process_track(mask, track_id)
            color = colors[(track_id - 1) % len(colors)]

            # Overlay the single-pixel track on the original image in a specific color
            for y in range(track_image.shape[0]):
                for x in range(track_image.shape[1]):
                    if track_image[y, x] == 255:  # Only overlay track pixels
                        overlay_image[y, x] = color

        # Save the overlay image with all tracks
        overlay_image_pil = Image.fromarray(overlay_image)
        overlay_image_pil.save(os.path.join(self.output_dir, "all_tracks_overlay.png"))

        messagebox.showinfo("Success", "All tracks processed, individual images saved, and overlay generated as 'all_tracks_overlay.png'.")


    def update_zoom(self, event=None):
        """Update the zoom level and refresh the canvas."""
        self.update_canvas()

if __name__ == "__main__":
    root = tk.Tk()
    app = KymographAnnotator(root)
    root.mainloop()


    


