# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Skeletonize ossicle meshes (ITK thinning, Coral3D-based).

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Pipeline step (upstream of the micro-CT panels Fig. 1C-D, 2A, 2C-H,
                S2C-D, S3)
What it does  : Smooths and voxelizes each ossicle mesh (voxel size 0.0004 mm), thins
                the voxel image with the ITK executable skel_itk, converts the voxel
                skeleton into VTK lines carrying the medial thickness, and
                post-processes it: terminal branches shorter than 15 voxels are removed
                (post15_skel_line.vtk) and all open branches are removed
                (clean_skel_line.vtk). Active loop: file index 139 only. DEPENDS on the
                Coral3D-based modules skelCodes/* and helpers/* and on the compiled
                skel_itk executable (Ramirez-Portilla et al. 2022, Front. Mar. Sci.),
                which are not distributed here (no licence); our changes to Coral3D are
                in coral3d_modifications.patch. Not runnable as shipped.
Inputs        : microct/animal_1/extracted_ossicles/aligned/vtk/*.vtk (commented
                setting: microct/animal_1/extracted_ossicles/vtk)
Outputs       : microct/animal_1/extracted_ossicles/aligned/skeleton/<ossicle>.vtk/{vox.mhd,
                vox.zraw, skel_line.vtk, post15_skel_line.vtk, clean_skel_line.vtk}
                (data tree)
Environment   : environment-analysis.yml (Python 3.7)
Run           : python skeletonize_code.py (from a folder containing the Coral3D
                skelCodes/ and helpers/ packages, with skel_itk on the executable search
                path)
"""

import vtk
import pyvista as pv
from vedo import *
import os
from matplotlib import pyplot as plt
import numpy as np

# from skelCodes import pre_skel
from skelCodes.pre_skel import *
from skelCodes.skeletonization import *
from skelCodes.post_skel import *

from skelCodes.extract_lines import *
from skelCodes.basic_skeleton_measures import *

from helpers import load_data, local_directories 

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


#%% Load mesh as vtk files and convert to volume and then to line skeleton. Store line skel as vtk files

# indir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "vtk")
# skeldir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "skeleton")

indir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned" / "vtk")
skeldir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned" / "skeleton")

vox_space = 0.0004   #voxel length dimensions in mm   usually used 0.0002 for all

plt3d = Plotter(bg2='gray2')#, interactive=True) # screen size

all_file_names = os.listdir(indir)
file_names = [f for f in all_file_names if f.endswith('.vtk')]
file_names.sort()
files = [os.path.join(indir,f) for f in file_names]
skel_paths = [os.path.join(skeldir,f) for f in file_names]

n_files = len(file_names)
print("number of files: ", n_files)

for ii in range(139,140): #range(24,24):   
    print("ossicle number: ", ii, files[ii])
    ossicle_poly = load(files[ii]).polydata()
    os.mkdir(skel_paths[ii])
    skel_line = skeletonize(ossicle_poly, skel_paths[ii], vox_space)
    write(skel_line, os.path.join(skel_paths[ii],"skel_line.vtk"))
    os.remove(os.path.join(skel_paths[ii], "vox_skel.mhd"))
    os.remove(os.path.join(skel_paths[ii], "vox_skel.raw"))


#%%
# post skeletonization corrections to the line paths

for ii in range(139,140): #range(213):   
    print("ossicle number: ", ii)
    ossicle_poly = load(files[ii]).polydata()
    skel_path = os.path.join(skel_paths[ii],"skel_line.vtk")    

    # create a vtkPolyDataReader object
    reader = vtk.vtkPolyDataReader()

    # set the file name
    reader.SetFileName(skel_path)
    # update the reader
    reader.Update()
    # get the output of the reader
    skel_line = reader.GetOutput()

    # post skeletonization corrections to the line paths - choose segments only upto 15 in length, ignore smaller than that
    post_skel_line_poly = vtk.vtkPolyData()#load(os.path.join(skel_path,"skel_line.vtk")).polydata()
    post_skel_line_poly.DeepCopy(skel_line)
    postProcessSkeleton(post_skel_line_poly, ossicle_poly, 15)   #remove branches smaller than 15
    write(post_skel_line_poly, os.path.join(skel_paths[ii],"post15_skel_line.vtk"))

    # remove all open branches, only choose closed loops
    cl_skel_line_poly = vtk.vtkPolyData()#load(os.path.join(skel_path,"skel_line.vtk")).polydata()
    cl_skel_line_poly.DeepCopy(skel_line)
    postProcessSkeleton(cl_skel_line_poly, ossicle_poly, 10000000000)   #remove all open branches
    postProcessSkeleton(cl_skel_line_poly, ossicle_poly, 10000000000)   #remove all open branches
    postProcessSkeleton(cl_skel_line_poly, ossicle_poly, 10000000000)   #remove all open branches
    write(cl_skel_line_poly, os.path.join(skel_paths[ii],"clean_skel_line.vtk"))
