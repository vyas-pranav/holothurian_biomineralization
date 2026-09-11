# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Paint masks along vesicle paths on a maximum projection (Tk GUI).

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 5D–G; Fig. S15C–I; Fig. S16A–E (choice of the paths that are tracked)
What it does  : Interactive annotator. Open the maximum-intensity projection of a
                time-lapse, paint over each streak along which vesicles move (the paper
                workflow uses a 15-px brush) and save each painted region as a binary
                mask PNG (255 = painted, 0 = background).
Inputs        : an image chosen in the file dialog (e.g. <Image folder>/MAX_<image>.tif)
Outputs       : the mask PNG named in the save dialog (e.g. <Image folder>/track_<k>.png)
Environment   : environment-analysis.yml (Python 3.7)
Run           : python draw_track.py   (interactive Tk window; never run unattended)
"""

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw

class ImageAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Annotator")

        # Initialize variables
        self.image_path = None
        self.original_image = None  # PIL Image
        self.display_image = None   # PIL Image for display
        self.zoom_level = 1.0
        self.brush_size = 5
        self.eraser_mode = False
        self.drawing = False
        self.last_x, self.last_y = None, None

        # Binary mask for painted pixels
        self.mask = None  # PIL Image (mode 'L')

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        # Frame for buttons and sliders
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Open Image Button
        open_button = tk.Button(control_frame, text="Open Image", command=self.open_image)
        open_button.pack(side=tk.LEFT, padx=5, pady=5)

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

        # Zoom Slider (up to 500%)
        self.zoom_slider = tk.Scale(control_frame, from_=10, to=500, orient=tk.HORIZONTAL, label="Zoom (%)")
        self.zoom_slider.set(int(self.zoom_level * 100))
        self.zoom_slider.pack(side=tk.LEFT, padx=5, pady=5)
        self.zoom_slider.bind("<B1-Motion>", self.update_zoom)
        self.zoom_slider.bind("<ButtonRelease-1>", self.update_zoom)

        # Save Button
        save_button = tk.Button(control_frame, text="Save Mask", command=self.save_mask)
        save_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Canvas for image display and drawing
        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

        # Initialize canvas image item
        self.canvas_image = None

    def open_image(self):
        file_types = [("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif")]
        self.image_path = filedialog.askopenfilename(title='Select an image', filetypes=file_types)
        if not self.image_path:
            return

        # Open the image using PIL
        self.original_image = Image.open(self.image_path).convert('RGB')
        self.display_image = self.original_image.copy()

        # Create a binary mask image (mode 'L' for grayscale)
        self.mask = Image.new('L', self.original_image.size, 0)

        # Update the canvas display
        self.update_canvas()

    def update_canvas(self, event=None):
        if self.display_image is None:
            return

        # Resize image according to zoom level
        zoom_percentage = self.zoom_slider.get()
        self.zoom_level = zoom_percentage / 100.0
        width, height = self.display_image.size
        resized_image = self.display_image.resize(
            (int(width * self.zoom_level), int(height * self.zoom_level)),
            Image.ANTIALIAS
        )

        # Convert image to Tkinter format
        self.tk_image = ImageTk.PhotoImage(resized_image)

        # Update canvas size
        self.canvas.config(width=resized_image.width, height=resized_image.height)
        if self.canvas_image is None:
            self.canvas_image = self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)
        else:
            self.canvas.itemconfig(self.canvas_image, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y

        # Draw initial point
        self.draw_at_point(event.x, event.y)

    def draw(self, event):
        if not self.drawing or self.display_image is None:
            return

        # Draw line from last position to current position
        self.draw_line(self.last_x, self.last_y, event.x, event.y)

        self.last_x, self.last_y = event.x, event.y

    def end_draw(self, event):
        self.drawing = False

    def draw_line(self, x1, y1, x2, y2):
        # Calculate the actual positions on the original image
        x1_orig = int(x1 / self.zoom_level)
        y1_orig = int(y1 / self.zoom_level)
        x2_orig = int(x2 / self.zoom_level)
        y2_orig = int(y2 / self.zoom_level)

        # Update brush size
        self.brush_size = self.brush_size_slider.get()

        # Draw on the mask
        draw_mask = ImageDraw.Draw(self.mask)
        if self.eraser_mode:
            # Eraser mode: remove pixels from the mask
            draw_mask.line([x1_orig, y1_orig, x2_orig, y2_orig], fill=0, width=self.brush_size)
            # To make smooth edges, draw circles at the endpoints
            draw_mask.ellipse([x2_orig - self.brush_size//2, y2_orig - self.brush_size//2,
                               x2_orig + self.brush_size//2, y2_orig + self.brush_size//2], fill=0)
        else:
            # Pen mode: add pixels to the mask
            draw_mask.line([x1_orig, y1_orig, x2_orig, y2_orig], fill=255, width=self.brush_size)
            # To make smooth edges, draw circles at the endpoints
            draw_mask.ellipse([x2_orig - self.brush_size//2, y2_orig - self.brush_size//2,
                               x2_orig + self.brush_size//2, y2_orig + self.brush_size//2], fill=255)

        # Create an overlay image to visualize the drawing
        overlay = Image.new('RGBA', self.original_image.size, (0, 0, 0, 0))
        # Draw the mask onto the overlay image with a semi-transparent color
        overlay_mask = self.mask.point(lambda p: 0 if p == 0 else 255)
        overlay.paste((255, 0, 0, 128), (0, 0), overlay_mask)

        # Composite the overlay onto the original image
        self.display_image = Image.alpha_composite(self.original_image.convert('RGBA'), overlay)

        # Update the canvas display
        self.update_canvas()

    def draw_at_point(self, x, y):
        self.draw_line(x, y, x, y)

    def select_pen(self):
        self.eraser_mode = False
        self.pen_button.config(relief=tk.SUNKEN)
        self.eraser_button.config(relief=tk.RAISED)

    def select_eraser(self):
        self.eraser_mode = True
        self.pen_button.config(relief=tk.RAISED)
        self.eraser_button.config(relief=tk.SUNKEN)

    def update_zoom(self, event=None):
        self.update_canvas()

    def save_mask(self):
        if self.mask is None or self.original_image is None:
            return

        # Save the mask as a binary image at the same resolution as the original
        save_path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Save Mask"
        )
        if save_path:
            # Ensure the mask is binary
            binary_mask = self.mask.point(lambda p: 255 if p > 0 else 0)
            binary_mask.save(save_path)
            print(f"Mask saved as '{save_path}'")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotator(root)
    root.mainloop()
