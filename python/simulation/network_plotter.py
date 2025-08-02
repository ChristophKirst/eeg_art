# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Example
>>> import numpy as np
>>> from simulation.network_plotter import NetworkPlotter
>>> n = 1000
>>> positions = np.random.rand(n, 3)
>>> values = np.random.rand(n)
>>> post = np.random.randint(0, n, (n, 3))
>>> import matplotlib.pylab as plt
>>> import matplotlib.colors as mcolots
>>> colors = ['yellow', 'orange', 'white', 'lightblue']
>>> cmap = mcolots.LinearSegmentedColormap.from_list('electricity', colors)
>>> pl = NetworkPlotter(positions=positions, post=post, rotation_speed=0.03, points=dict(cmap=cmap, value_range=(0, 1.4)), lines=dict(cmap=cmap, value_range=(0, 1.4)))
>>> _ = [pl.update(np.random.rand(n) < 0.002) for _ in range(1000)]


"""



from abc import ABC, abstractmethod
import numpy as np
import functools as ft

from visualization.plotting import pg, gl
pg.CONFIG_OPTIONS['useOpenGL'] = True  # set to False if trouble seeing data.

from pyqtgraph.Qt import QtGui, QtWidgets
from pyqtgraph.Qt.QtGui import QFont
from pyqtgraph.Qt.QtWidgets import QWidget

from visualization.plotting import pv, pvq




class PlotData(ABC):

    @abstractmethod
    def update(self, *args, **kwargs):
        ...


class PointData(PlotData):
    def __init__(
            self,
            plotter,
            positions: np.ndarray,
            values: np.ndarray | None = None,
            cmap='Blues',
            opacity='linear',
            render_as_spheres=False,
            point_size=15,
            value_range=(0.0, 2.0),
            decay: float = 0.99,
            **kwargs
    ):
        if values is None:
            values = np.ones(len(positions))

        self.data = pv.PolyData(positions)
        self.data['intensity'] = values

        self.actor = plotter.add_mesh(
            self.data,
            scalars='intensity',
            cmap=cmap,
            clim=value_range,
            opacity=opacity,
            render_points_as_spheres=render_as_spheres,
            point_size=point_size,
            **kwargs
        )

        self.decay = decay

    def update(self, spiking: np.ndarray, spike_height: float = 1.0):
        values = self.data['intensity']
        values *= self.decay
        values[spiking] += spike_height


class LineData(PlotData):
    def __init__(
            self,
            plotter,
            positions: np.ndarray,
            post: np.ndarray | None = None,
            values: np.ndarray | None = None,
            cmap='Grays',
            opacity='linear',
            line_width=1,
            value_range=(0.0, 3.0),
            decay: float = 0.99,
            **kwargs
    ):
        lines = self.lines_from_post(post)
        values = values if values is not None else np.ones(post.size)

        self.data = pv.PolyData(positions, lines=lines)
        self.data['intensity'] = values

        self.actor = plotter.add_mesh(
            self.data,
            scalars='intensity',
            line_width=line_width,
            cmap=cmap,
            clim=value_range,
            opacity=opacity,
            **kwargs
        )

        self.m = post.shape[1]
        self.decay = decay

    @staticmethod
    def lines_from_post(post: np.ndarray):
        lines = []
        for i, post_i in enumerate(post):
            lines.extend([[2, i, p] for p in post_i])
        lines = np.array(lines).flatten()
        return lines

    def update(self, spiking: np.ndarray, spike_height: float = 1.0):
        values = self.data['intensity']
        values *= self.decay
        values[np.repeat(spiking, self.m)] += spike_height
        self.actor.mapper.Modified()


class NetworkPlotter(QWidget):
    def __init__(
            self,
            positions: np.ndarray,
            post: np.ndarray | None = None,
            points: bool | dict | None = True,
            lines: bool | dict | None = True,
            rotation_speed: float = 0.0,
            title: str | None = None,
            window_size: tuple[int, int] = (1200, 1600),
            parent: object = None
    ):
        QWidget.__init__(self, parent)

        self.rotation_speed = rotation_speed
        self.plotter = pvq.QtInteractor(self)

        if lines is True and positions is not None and post is not None:
            lines = dict(positions=positions, post=post)
        if isinstance(lines, dict):
            lines.update(dict(positions=positions, post=post))
            lines = LineData(self.plotter, **lines)
        else:
            lines = None
        self.lines = lines

        if points is True and positions is not None:
            points = dict(positions=positions)
        if isinstance(points, dict):
            points.update(dict(positions=positions))
            points = PointData(self.plotter, **points)
        else:
            points = None
        self.points = points

        if title is None:
            title = 'NetworkPlotter'

        # Gui Construction
        self.setWindowTitle(title)
        self.resize(window_size[0], window_size[1])

        self.layout = pg.QtWidgets.QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.layout.addWidget(self.plotter.interactor)

        self.show()
        self.plotter.render()

    def update_camera(self):
        self.plotter.camera.azimuth += self.rotation_speed

    def update(
            self,
            spiking: np.ndarray,
            **kwargs  # noqa
    ):
        if self.points is not None:
            self.points.update(spiking)
        if self.lines is not None:
            self.lines.update(spiking)
        self.update_camera()
        self.plotter.render()

        QtWidgets.QApplication.processEvents()
