
def initialize_pycharm_gui():
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


initialize_pycharm_gui()


import matplotlib.pyplot as plt

import pyqtgraph as pg

import pyvista as pv
import pyvistaqt as pvq

from matplotlib.colors import Colormap


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
