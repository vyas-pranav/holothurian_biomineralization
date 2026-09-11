# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Fraction of ossicles that grew under cytoskeletal drug treatments.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 5C
What it does  : Horizontal bar chart of the fraction of ossicles with new growth (calcein
                signal) for Ctrl, DMSO (vehicle), Lat (latrunculin) and Noc (nocodazole):
                one bar per animal (1, 2) and one for both animals pooled (Total), each
                labelled "grown/total". The counts are typed in below.
Inputs        : none (counts are hard-coded)
Outputs       : fraction_values_plot_hori.png (1200 dpi) in the current directory
Environment   : environment-analysis.yml (Python 3.7)
Run           : python plot_growth_fraction.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.font_manager as fm

# Data setup
categories = ['Ctrl', 'DMSO', 'Lat', 'Noc']
fractions = [
    (23/29, 28/37),  # Ctrl
    (31/34, 31/51),  # DMSO
    (23/41, 52/68),  # Lat
    (18/97, 6/39)    # Noc
]
numerators = [
    (23, 28),  # Ctrl
    (31, 31),  # DMSO
    (23, 52),  # Lat
    (18, 6)    # Noc
]
denominators = [
    (29, 37),  # Ctrl
    (34, 51),  # DMSO
    (41, 68),  # Lat
    (97, 39)   # Noc
]

# Colors from Blues colormap
cmap = get_cmap("Blues")
colors = [cmap(0.25), cmap(0.50), cmap(0.75)]  # Lighter to darker shades

# Calculate total fractions by summing numerators and denominators
total_numerators = [num1 + num2 for (num1, num2) in numerators]
total_denominators = [den1 + den2 for (den1, den2) in denominators]
total_fraction_values = [num / den if den != 0 else 0 for num, den in zip(total_numerators, total_denominators)]

# Setting up plot dimensions and font
fig, ax = plt.subplots(figsize=(3, 4))

# set font as helvetica
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# For horizontal bars, we use y positions instead of x positions.
y_positions = np.arange(len(categories))
bar_height = 0.3  # equivalent to the bar width in the vertical chart

# Plotting horizontal bars
for i, (category, (num1, num2), (den1, den2), (frac1, frac2), total_num, total_den, total_frac) in enumerate(
    zip(categories, numerators, denominators, fractions, total_numerators, total_denominators, total_fraction_values)
):
    # For horizontal bar charts, the bar's length is the fraction value (x direction)
    # and we offset the bars along the y axis.
    y_base = y_positions[i]
    # Plot Animal 1, Animal 2, and Total bars for each category
    ax.barh(y_base - bar_height, frac1, height=bar_height, color=colors[0],
            label="1" if i == 0 else "")
    ax.barh(y_base, frac2, height=bar_height, color=colors[1],
            label="2" if i == 0 else "")
    ax.barh(y_base + bar_height, total_frac, height=bar_height, color=colors[2],
            label="Total" if i == 0 else "")
    
    text_size = 13
    # For categories other than 'Noc', place the fraction text centered in the bar;
    # for 'Noc' (i==3), place the text just to the right of the bar.
    if i != 3:
        ax.text(frac1 / 2, y_base - bar_height, f"{num1}/{den1}", ha='center', va='center', fontsize=text_size)
        ax.text(frac2 / 2, y_base, f"{num2}/{den2}", ha='center', va='center', fontsize=text_size)
        ax.text(total_frac / 2, y_base + bar_height, f"{total_num}/{total_den}", ha='center', va='center', fontsize=text_size)
    else:
        ax.text(frac1 + 0.01, y_base - bar_height, f"{num1}/{den1}", ha='left', va='center', fontsize=text_size)
        ax.text(frac2 + 0.01, y_base, f"{num2}/{den2}", ha='left', va='center', fontsize=text_size)
        ax.text(total_frac + 0.01, y_base + bar_height, f"{total_num}/{total_den}", ha='left', va='center', fontsize=text_size)

# Customizing plot
ax.set_yticks(y_positions)
ax.set_yticklabels(categories)
ax.set_ylabel("Categories", fontsize=15)
ax.set_xlabel("Fraction Values", fontsize=15)
ax.set_title("Fraction of ossicles with growth", fontsize=16)
ax.legend(fontsize=12) #loc="upper right"

# Set tick label sizes
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

#tilt y-axis labels for better readability
plt.yticks(rotation=45)

plt.tight_layout()

# Save figure
plt.savefig("fraction_values_plot_hori.png", dpi=1200)

plt.show()
