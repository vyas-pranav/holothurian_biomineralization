# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Batch-convert ossicle STL meshes to VTK polydata.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Pipeline step (upstream of the micro-CT panels Fig. 1C-D, 2A, 2C-H,
                S2C-D, S3)
What it does  : Converts every .stl file of a folder into a legacy .vtk polydata file
                (VTK STL reader + polydata writer). Active setting: aligned meshes
                (aligned/stl -> aligned/vtk). The commented setting records the earlier
                conversion of the unaligned Dragonfly meshes (mesh -> vtk).
Inputs        : microct/animal_1/extracted_ossicles/aligned/stl/*.stl
Outputs       : microct/animal_1/extracted_ossicles/aligned/vtk/*.vtk (data tree; read
                by skeletonize_code.py and later scripts)
Environment   : environment-analysis.yml (Python 3.7)
Run           : python stl2vtk.py
"""

import os
import vtk
import argparse

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

def convertFile(filepath, outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if os.path.isfile(filepath):
        basename = os.path.basename(filepath)
        print("Copying file:", basename)
        basename = os.path.splitext(basename)[0]
        outfile = os.path.join(outdir, basename+".vtk")

        reader = vtk.vtkSTLReader()
        reader.SetFileName(filepath)
        reader.Update()
        # reader = reader.GetOutput()

        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(outfile)
        # writer.SetInputData(reader)
        # writer.Update()        
        writer.SetInputConnection(reader.GetOutputPort())

        return writer.Write()==1

    return False

def convertFiles(indir, outdir):    
    files = os.listdir(indir)
    files = [ os.path.join(indir,f) for f in files if f.endswith('.stl') ]
    ret = 0
    print("In:", indir)
    print("Out:", outdir)
    for f in files:
        ret += convertFile(f, outdir)
        print("Successfully converted %d out of %d files." % (ret, len(files)))

# indir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "mesh")
# outdir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "vtk")
indir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned" / "stl")
outdir = str(DATA_ROOT / "microct" / "animal_1" / "extracted_ossicles" / "aligned" / "vtk")

convertFiles(indir, outdir)    
