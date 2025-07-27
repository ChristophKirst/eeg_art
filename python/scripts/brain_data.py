# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Code based on: https://github.com/theunissenlab/soundsig
"""
#%% brain points

import numpy as np
import colorcet as cc
import vtk
import vtk.util.numpy_support
from visualization.plotting import pv


brain_data_file = './data/brain/brain_human.vtk'

reader = vtk.vtkPolyDataReader()
reader.SetFileName(brain_data_file)
reader.Update()
data = reader.GetOutput()

points = data.GetPoints()
num_points = points.GetNumberOfPoints()
point_coords = vtk.util.numpy_support.vtk_to_numpy(points.GetData())

point_data = data.GetPointData()
arrays = {}
if point_data.GetNumberOfArrays() > 0:
    point_data_arr_idx = list(range(point_data.GetNumberOfArrays()))
    for idx in point_data_arr_idx:
        array = point_data.GetArray(idx)
        arrays[array.GetName()] = array

scalars = vtk.util.numpy_support.vtk_to_numpy(arrays['scalars'])

down_sample = 1
pv.plot(point_coords[::down_sample], scalars=scalars[::down_sample], cmap=cc.glasbey)


#%% write jfx for max msp

down_sample = 10

from utils.jxf import write_jxf
write_jxf('./data/brain/brain_coords.jxf', point_coords[::down_sample].T, plane_count=True)
write_jxf('./data/brain/brain_scalars.jxf', np.asarray(scalars[::down_sample], dtype='int16'), plane_count=False)


#%% brain mesh
import colorcet as cc
from visualization.plotting import pv

brain_data_file = './data/brain/brain_human.vtk'
mesh = pv.read(brain_data_file)
mesh.plot(cpos='xz', cmap=cc.glasbey_warm)

