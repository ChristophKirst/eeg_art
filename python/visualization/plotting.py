# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Examples
>>> from visualization.plotting import pg
>>> pg.plot([1, 2, 3, 4, 5])
"""

# initialize plotting for pycharm
try:
    import IPython
    IPython.get_ipython().run_line_magic("gui", "qt")  # qt for interactive figures in pycharm
except ModuleNotFoundError:
    pass

try:
    import matplotlib as mpl
    mpl.use('Qt5Agg')  # qt for interactive figures in pycharm
except (ModuleNotFoundError, ValueError):
    pass

import matplotlib.pyplot as plt

import pyqtgraph as pg

import pyvista as pv
import pyvistaqt as pvq

from matplotlib.colors import Colormap

import pyqtgraph.opengl as gl

app = pg.mkQApp() if not pg.QAPP else pg.QAPP


def make_colormap(name: str, colors):
    import matplotlib.colors as mpc  # noqa
    return mpc.LinearSegmentedColormap.from_list(name, colors, N=len(colors))


def plot_colormap(cmap: Colormap, figure: int | None = None):
    plt.figure(figure, figsize=(8, 2))
    if figure is not None:
        plt.clf()
    import numpy as np

    plt.imshow(np.linspace(0, 1, cmap.N).reshape(1, -1), aspect='auto', cmap=cmap)
    plt.colorbar(orientation='horizontal', label='colors')
    plt.axis('off')
    plt.show()
