# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Number of holes vs ossicle surface area and volume.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. S3F
What it does  : Reads the per-ossicle table (type id, surface area, volume, number of
                holes), remaps the type ids, drops the single ossicle of remapped type
                1, and plots surface area vs number of holes, volume vs number of holes
                and number of holes vs type id (colour = type id).
Inputs        : tables/hole_area_vol/hole_sa_vol_stat.xlsx
Outputs       : OUT_ROOT/01_microct_morphometrics/{area_vs_number_of_holes,
                volume_vs_number_of_holes, number_of_holes_vs_type_id}.{png,svg}
Environment   : environment-analysis.yml (Python 3.7)
Run           : python plot_data.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

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
os.makedirs(OUT_ROOT / "01_microct_morphometrics", exist_ok=True)
# ----------------------------------------------------------------------------


#read data from xlsx file
folder = str(DATA_ROOT / "tables" / "hole_area_vol")
file = "hole_sa_vol_stat.xlsx"
data = pd.read_excel(folder + "\\" + file, sheet_name = 'Sheet1')

#read data into lists
#first column is ossicle id, second column is type id, third column is symmetry, forth column is curved tag, fifth column is notes, sixth column is hole area, seventh column is hole volume, eigth column is number of holes
ossicle_id = data.iloc[:, 0].tolist()
type_id = data.iloc[:, 1].tolist()
surface_area = data.iloc[:, 5].tolist()
volume = data.iloc[:, 6].tolist()
number_of_holes = data.iloc[:, 7].tolist()

#swap id values based on a map. 1 to 5, 5 to 6, 6 to 7, 7 to 8, 8 to 10, 10 to 1
type_id_new = type_id.copy()
for i in range(len(type_id)):
    if type_id[i] == 1:
        type_id_new[i] = 5
    elif type_id[i] == 5:
        type_id_new[i] = 6
    elif type_id[i] == 6:
        type_id_new[i] = 7
    elif type_id[i] == 7:
        type_id_new[i] = 8
    elif type_id[i] == 8:
        type_id_new[i] = 10
    elif type_id[i] == 10:
        type_id_new[i] = 1

type_id = type_id_new.copy()


#find index with type_id value 1 and remove that index from all lists
index = type_id.index(1)
del type_id[index]
del surface_area[index]
del volume[index]
del number_of_holes[index]
del ossicle_id[index]


color_list = "#CB8D00,#B1C100,#00C300,#C30000,#008400,#00563C,#007590,#25007C,#0033A4,#000000"


#plot data
fig, ax = plt.subplots(figsize=(5, 5))
#plot area vs number of holes and color with type id
plot = ax.scatter(number_of_holes, surface_area, c = type_id, cmap = 'Paired', s = 75, alpha = 0.5, edgecolors='none')
ax.set_xlabel('Number of holes')
ax.set_ylabel(r'Surface area ($mm^2$)')
#increase font size
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

#enforce scientific notation    
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(20)

#change label size
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)

#save figure
fig.savefig(str(OUT_ROOT / "01_microct_morphometrics" / "area_vs_number_of_holes.png"), dpi = 600, bbox_inches = 'tight')
fig.savefig(str(OUT_ROOT / "01_microct_morphometrics" / "area_vs_number_of_holes.svg"), dpi = 600, bbox_inches = 'tight')


#plot volume vs number of holes and color with type id
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(number_of_holes, volume, c = type_id, cmap = 'Paired', s = 75, alpha = 0.5, edgecolors='none')
ax.set_xlabel('Number of holes')
ax.set_ylabel(r'Volume ($mm^3$)')
#increase font size
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)
#enforce scientific notation    
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(20)

#change label size
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)

#save figure
fig.savefig(str(OUT_ROOT / "01_microct_morphometrics" / "volume_vs_number_of_holes.png"), dpi = 600, bbox_inches = 'tight')
fig.savefig(str(OUT_ROOT / "01_microct_morphometrics" / "volume_vs_number_of_holes.svg"), dpi = 600, bbox_inches = 'tight')


#plot number of holes vs type id
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(type_id, number_of_holes, c = type_id, cmap = 'Paired', s = 75, alpha = 0.5, edgecolors='none')
ax.set_xlabel('Ossicle type id')
ax.set_ylabel('Number of holes')

#increase font size
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)
#enforce scientific notation    
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(20)

#specify x axis ticks
ax.set_xticks([2, 4, 6, 8, 10])


#change label size
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)

#save figure
fig.savefig(str(OUT_ROOT / "01_microct_morphometrics" / "number_of_holes_vs_type_id.png"), dpi = 600, bbox_inches = 'tight')
fig.savefig(str(OUT_ROOT / "01_microct_morphometrics" / "number_of_holes_vs_type_id.svg"), dpi = 600, bbox_inches = 'tight')


# #create a plot with just the labels with colors and their names with type ids
# fig, ax = plt.subplots(figsize=(5, 5))
# #Create legend with type id and color
# type_id_unique = np.unique(type_id)

# #plot type id vs type id and color with type id
# ax.scatter(type_id_unique, type_id_unique, c = type_id_unique, cmap = 'Paired', s = 75)
# ax.set_xlabel('Ossicle type id')
# ax.set_ylabel('Ossicle type id')

# # create an id chart with color from cmap and type id
# cmap = plt.cm.Paired
# norm = plt.Normalize(vmin=0, vmax=type_id_unique[-1])
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm, ticks=type_id_unique)
# cbar.ax.tick_params(labelsize=20)
# cbar.set_label('Ossicle type id', rotation=270, fontsize = 20, labelpad = 20)

# #increase font size
# ax.tick_params(axis='both', which='major', labelsize=20)
# ax.tick_params(axis='both', which='minor', labelsize=20)

# #change label size
# ax.xaxis.label.set_size(20)
# ax.yaxis.label.set_size(20)

# #save figure
# fig.savefig(str(OUT_ROOT / "01_microct_morphometrics" / "type_id_color_chart.png"), dpi = 600, bbox_inches = 'tight')
# fig.savefig(str(OUT_ROOT / "01_microct_morphometrics" / "type_id_color_chart.svg"), dpi = 600, bbox_inches = 'tight')



plt.show()
