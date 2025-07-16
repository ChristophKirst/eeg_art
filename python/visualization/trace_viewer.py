# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst
"""
import utils.initialize_pycharm_gui

import pyvista as pv
import pyvistaqt as pvq


plotter = pvq.BackgroundPlotter()  # or True if you're in a Jupyter notebook

# Initial position of the moving point
moving_point_coords = np.array([[0.0, 0.0, 0.0]])
moving_point_mesh = pv.PolyData(moving_point_coords)

# Initialize the trace
trace_points = pv.PolyData()
