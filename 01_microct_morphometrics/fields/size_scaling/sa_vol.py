# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Surface-area / volume scaling of ossicles from seven animals.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. S5I
What it does  : Pools per-ossicle surface area and volume (sa_vol_data.pkl) of seven
                micro-CT data sets and draws a seaborn joint regression plot of
                log10(volume) vs log10(surface area), with the fitted slope and
                reference lines of slope 3/2 (sphere) and 1 (plate). The commented cell
                records how sa_vol_data.pkl, sa_list.txt and vol_list.txt were computed
                with vedo (mesh area and volume) for each data set.
Inputs        : microct/{animal_1/extracted_ossicles,
                mid_01_20220822/ossicle_extraction, mid_02_20220823/extracted_ossicles,
                relaxed_animal_20220612/extracted, animal_1_20220825, animal_3_20220825,
                animal_5}/sa_vol_data.pkl; file list of microct/animal_5/meshes/*.stl
Outputs       : sa_vol.png in the working directory
Environment   : environment-analysis.yml (Python 3.7)
Run           : python sa_vol.py
"""

from vedo import *
import os
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import re
import pickle

import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

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



#%%
def get_ids_for_selected(ossicle_type_list):
    n_files = len(file_names)

    #Pillared Tables	1
    # Head Plates	2
    # Rear Plates	3
    # Foot Base Plates	4
    # Foot Side Plates	5
    # Elongated Plates	6
    # C-rods	7
    # Blob Ossicle	8
    # Basket Cluster	9
    # Internal Calcareous Ring	10
    # Very Small 	11

    ctr = 1
    selected_list = []
    for type_id in ossicle_type_list:
        # if type_id == 1 or type_id == 2 or type_id == 3 or type_id == 4 or type_id == 5 :
        #allow all types
        selected_list.append(ctr)
        ctr = ctr+1

    file_id_list = []
    for ii in range(n_files):
        txt = file_names[ii]
        num = [int(s) for s in re.findall(r'\d+',txt)]
        file_id_list.append(num[0])

    loc_id = []
    for jj in selected_list:
        loc_id.append(file_id_list.index(jj))


    return loc_id


def linear_regression_slope(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    reg = LinearRegression().fit(x, y)
    return reg.coef_[0]


#%%
# type_dir =str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles")
# indir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "vtk")
# skeldir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "skeleton")
# datadir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "Data Calculated")

# type_dir =str(DATA_ROOT / "microct" / "mid_01_20220822" / "ossicle_extraction")
# indir = str(DATA_ROOT / "microct" / "mid_01_20220822" / "ossicle_extraction" / "meshes")

# type_dir =str(DATA_ROOT / "microct" / "mid_02_20220823" / "extracted_ossicles")
# indir = str(DATA_ROOT / "microct" / "mid_02_20220823" / "extracted_ossicles" / "meshes")

# type_dir =str(DATA_ROOT / "microct" / "relaxed_animal_20220612" / "extracted")
# indir = str(DATA_ROOT / "microct" / "relaxed_animal_20220612" / "extracted" / "meshes")

# type_dir =str(DATA_ROOT / "microct" / "animal_1_20220825")
# indir = str(DATA_ROOT / "microct" / "animal_1_20220825" / "meshes")

# type_dir =str(DATA_ROOT / "microct" / "animal_3_20220825")
# indir = str(DATA_ROOT / "microct" / "animal_3_20220825" / "meshes")

type_dir =str(DATA_ROOT / "microct" / "animal_5")
indir = str(DATA_ROOT / "microct" / "animal_5" / "meshes")

plt3d = Plotter(bg2='gray2')#, interactive=True) # screen size
all_file_names = os.listdir(indir)
file_names = [f for f in all_file_names if f.endswith('.stl')]
file_names.sort()
files = [os.path.join(indir,f) for f in file_names]
# skel_paths = [os.path.join(skeldir,f) for f in file_names]

n_files = len(file_names)

# with open(os.path.join(datadir,'ossicle_type_list.txt')) as txtfile:
#     ossicle_type_list = list(map(int, txtfile.read().split('\n')))

# loc_file_id = get_ids_for_selected(ossicle_type_list)

#%%  to calculate sa and vol and store it in files

# sa_vol_dict = {}
# sa_list = []
# vol_list = []

# for ii in range(n_files): 
#     ossicle = load(files[ii])
#     sa = ossicle.area()
#     vol = ossicle.volume()   

#     sa_list.append(sa)
#     vol_list.append(vol)
#     sa_vol_dict[file_names[ii]] = [sa, vol]

# with open(os.path.join(type_dir, 'sa_vol_data.pkl'), 'wb') as f:
#     pickle.dump(sa_vol_dict, f)

# file = open(os.path.join(type_dir,'sa_list.txt'),'w')
# for item in sa_list:
#  	file.write(str(item)+"\n")
# file.close()

# file = open(os.path.join(type_dir,'vol_list.txt'),'w')
# for item in vol_list:
#  	file.write(str(item)+"\n")
# file.close()

#%%

sa_all = []
vol_all = []
sa_pool = []
vol_pool = []
for kk in range(7):
    if kk == 0: 
        type_dir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles")
    elif kk == 1:
        type_dir = str(DATA_ROOT / "microct" / "mid_01_20220822" / "ossicle_extraction")
    elif kk == 2:
        type_dir = str(DATA_ROOT / "microct" / "mid_02_20220823" / "extracted_ossicles")
    elif kk == 3:
        type_dir =str(DATA_ROOT / "microct" / "relaxed_animal_20220612" / "extracted")
    elif kk == 4:
        type_dir =str(DATA_ROOT / "microct" / "animal_1_20220825")
    elif kk == 5:
        type_dir =str(DATA_ROOT / "microct" / "animal_3_20220825")
    elif kk == 6:
        type_dir =str(DATA_ROOT / "microct" / "animal_5")


    with open(os.path.join(type_dir, 'sa_vol_data.pkl'), 'rb') as f:
        loaded_dict = pickle.load(f)

    sa_list = []
    vol_list = []
    for key in loaded_dict:
        sa_list.append(loaded_dict[key][0])
        sa_pool.append(loaded_dict[key][0])

    for key in loaded_dict:
        vol_list.append(loaded_dict[key][1])
        vol_pool.append(loaded_dict[key][1])

    sa_all.append(sa_list)
    vol_all.append(vol_list)

# fig1,ax1 = plt.subplots(figsize=(20,20))


# Create a dataframe from the two lists
data = {'surface area': np.log10(sa_pool), 'volume': np.log10(vol_pool)}
df = pd.DataFrame(data)

# Create the seaborn dataset
sns.set(style="darkgrid")
sns.set(font_scale=2)
# sns.regplot(x="surface area", y = "volume", data = df, ax = ax1, line_kws=({"color": "C1"} ))
g = sns.jointplot(x="surface area", y = "volume", data = df, kind = 'reg', line_kws=({"color": "grey", "linewidth": 5}), height = 10, color = 'k')  #g.ax_joint.set_xlabel('log(surface area ($mm^2$))')
g.ax_joint.set_ylabel('log(volume ($mm^3$))')
g.ax_joint.set_xlabel('log(surface area ($mm^2$))')
g.ax_joint.set_ylabel('log(volume ($mm^3$))')

slope = np.round((linear_regression_slope(np.log10(sa_pool), np.log10(vol_pool))),3)
        # ax.scatter(np.log(sa_list),np.log(vol_list))
    # ax.scatter(sa_list,vol_list)
x = np.linspace(0.0001,0.02,1000)
g.ax_joint.plot(np.log10(x), np.log10(1/6*(x**1.5)/(np.pi**0.5)), linewidth = 5, linestyle =':', c='dimgrey')
g.ax_joint.plot(np.log10(x), np.log10(x/2*0.001), linewidth = 5, linestyle ='--', c = 'dimgrey')
g.ax_joint.text(-2.5, -3.6,'slope = 3/2', fontsize=20) #add text
g.ax_joint.text(-2, -5.5,'slope = 1', fontsize=20) #add text
g.ax_joint.text(-1.9, -4,'slope =' + str(slope)[:-1], fontsize=20) #add text

plt.savefig('sa_vol.png', dpi=600)

# ax.add_legen()

#%%
#What size vesicle 
