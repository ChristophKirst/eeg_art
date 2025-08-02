# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Example
>>> import numpy as np
>>> import simulation.network_viewer as nv
>>> raster = np.zeros((100, 2))
>>> raster[:, 0] = np.random.randint(0, 1000, size=100)
>>> raster[:, 1] = np.random.randint(0, 30, size=100)
>>> variables = np.vstack([np.cos(np.arange(1000) * 0.1), np.sin(np.arange(1000) * 0.1)]).T
>>> densities = np.random.rand(1000, 100)
>>> v = nv.NetworkViewer(rasters=raster, variables=variables, densities=densities, time_window=(0, 1000))

>>> raster_new = np.zeros((50, 2))
>>> raster_new[:, 1] = np.random.randint(0, 30, size=50)
>>> raster_new[:, 0] = np.random.randint(1000, 1500, size=50)
>>> variables_new = np.vstack([np.cos(np.arange(1000, 1500) * 0.1), np.sin(np.arange(1000, 1500) * 0.1)])
>>> densities_new = np.random.rand(500, 100)
>>> v.update_data(rasters=raster_new, variables=variables_new, densities=densities_new, shift=500)
"""

import numpy as np
import functools as ft

from visualization.plotting import pg, gl
pg.CONFIG_OPTIONS['useOpenGL'] = True  # set to False if trouble seeing data.

from pyqtgraph.Qt import QtGui, QtWidgets
from pyqtgraph.Qt.QtGui import QFont
from pyqtgraph.Qt.QtWidgets import QWidget


def custom_symbol(symbol: str, font: QFont = QFont("San Serif")):
    """Create custom symbol with font"""
    assert len(symbol) == 1
    pg_symbol = QtGui.QPainterPath()
    pg_symbol.addText(0, 0, font, symbol)

    br = pg_symbol.boundingRect()
    scale = min(1. / br.width(), 1. / br.height())
    tr = QtGui.QTransform()
    tr.scale(scale, scale)
    tr.translate(-br.x() - br.width() / 2., -br.y() - br.height() / 2.)
    return tr.map(pg_symbol)


class NetworkViewer(QWidget):
    def __init__(
            self,
            rasters: np.ndarray | list | None = None,
            variables: np.ndarray | list | None = None,
            densities: np.ndarray | list | None = None,
            neuron_window: tuple | None = None,
            time_window: tuple | None = None,
            link_time: bool = True,
            title: str | None = None,
            window_size: tuple[int, int] = (1200, 1600),
            parent: object = None,
            *args
    ):
        QWidget.__init__(self, parent, *args)

        if rasters is None and variables is None and densities is None:
            raise ValueError

        if rasters is not None:
            if not isinstance(rasters, list):
                rasters = [rasters]

        if variables is not None:
            if not isinstance(variables, list):
                variables = [variables]

        if densities is not None:
            if not isinstance(densities, list):
                densities = [densities]

        if neuron_window is None:
            if rasters is not None:
                neuron_window = (np.min(rasters[0][:, 1]), np.max(rasters[0][:, 1]))
            elif variables is not None:
                neuron_window = (0, variables[0].shape[1])
            elif densities is not None:
                neuron_window = (0, densities[0].shape[1])
        elif isinstance(neuron_window, int):
            neuron_window = (0, neuron_window)
        self.neuron_window = neuron_window

        if time_window is None:
            if rasters is not None:
                time_window = (np.min(rasters[0][:, 0]), np.max(rasters[0][:, 0]))
            elif variables is not None:
                time_window = (0, variables[0].shape[0])
            elif densities is not None:
                time_window = (0, densities[0].shape[0])
        self.time_window = time_window

        if title is None:
            title = 'NetworkViewer'

        # Gui Construction
        self.setWindowTitle(title)
        self.resize(window_size[0], window_size[1])

        self.layout = pg.QtWidgets.QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.splitter = pg.QtWidgets.QSplitter()
        self.splitter.setOrientation(pg.QtCore.Qt.Vertical)
        self.splitter.setSizePolicy(pg.QtWidgets.QSizePolicy.Expanding, pg.QtWidgets.QSizePolicy.Expanding)
        self.layout.addWidget(self.splitter)

        self.graphic_views = {}

        # rasters
        self.rasters = None
        self.raster_scatters = None
        self.raster_views = None

        if rasters is not None:
            self.rasters = rasters

            scatters = [pg.ScatterPlotItem(pos=raster, color='white', symbol=custom_symbol("|")) for raster in rasters]
            self.raster_scatters = scatters

            # views = [pg.ViewBox() for _ in rasters]
            views = [pg.PlotItem() for _ in rasters]
            self.raster_views = views

            p = 0
            for rv, rs in zip(views, scatters):
                p += 1

                rv.addItem(rs)
                rv.setXRange(self.time_window[0], self.time_window[1], padding=0)
                rv.setYRange(self.neuron_window[0], self.neuron_window[1], padding=0)

                graphics_view = pg.GraphicsView()
                graphics_view.setCentralItem(rv)
                self.graphic_views[f'raster_{p}'] = graphics_view

                self.splitter.addWidget(graphics_view)

        # variables
        self.variables = None
        self.variables_curves = None
        self.variables_views = None

        if variables is not None:
            self.variables = variables

            curves = [[pg.PlotCurveItem(y=v) for v in variable.T] for variable in variables]
            self.variables_curves = curves

            views = [pg.PlotItem() for _ in variables]
            self.variables_views = views

            p = 0
            for vv, vcs in zip(views, curves):
                p += 1
                for vc in vcs:
                    vv.addItem(vc)

                graphics_view = pg.GraphicsView()
                graphics_view.setCentralItem(vv)
                self.graphic_views[f'variables_{p}'] = graphics_view

                self.splitter.addWidget(graphics_view)

        # densities
        self.densities = None
        self.density_images = None
        self.density_views = None
        if densities is not None:
            self.densities = densities

            images = [pg.ImageItem(density) for density in densities]
            self.density_images = images

            views = [pg.PlotItem() for _ in densities]
            self.density_views = views

            p = 0
            for vv, di in zip(views, images):
                p += 1

                vv.addItem(di)

                graphics_view = pg.GraphicsView()
                graphics_view.setCentralItem(vv)
                self.graphic_views[f'densities_{p}'] = graphics_view

                self.splitter.addWidget(graphics_view)

        # link
        if link_time and len(self.graphic_views) > 1:
            graphic_views = list(self.graphic_views.values())
            gv0 = graphic_views[0]
            for gv in graphic_views[1:]:
                gv.centralWidget.setXLink(gv0.centralWidget)

        self.show()

    def update(
            self,
            rasters: np.ndarray | list | None = None,
            variables: np.ndarray | list | None = None,
            densities: np.ndarray | list | None = None,
            shift: int | None = None,
            extend: int | None = None,
            time_window: tuple[int, int] | None = None,
            neuron_window: tuple[int, int] | None = None,
            **kwargs  # noqa
    ):
        if time_window is not None:
            self.time_window = time_window
        extend = 0 if extend is None else extend
        shift = 0 if shift is None else shift
        self.time_window = (self.time_window[0] + shift, self.time_window[1] + shift + extend)

        if neuron_window is not None:
            self.neuron_window = neuron_window

        if self.rasters is not None:
            rasters = [rasters] if not isinstance(rasters, list) else rasters
            rasters = [np.concatenate([raster_prev[raster_prev[:, 0] >= self.time_window[0]], raster])
                       for raster_prev, raster in zip(self.rasters, rasters)]
            self.rasters = rasters

            for rv, rs, r in zip(self.raster_views, self.raster_scatters, self.rasters):
                rs.setData(pos=r - [self.time_window[0], 0])
                rv.setXRange(0, self.time_window[1] - self.time_window[0])
                if neuron_window is not None:
                    rv.setYRange(neuron_window[0], self.neuron_window[1])
                rv.update()

        if self.variables is not None:
            variables = [variables] if not isinstance(variables, list) else variables
            variables = [np.vstack([variable[shift:, :], variable_new])
                         for variable, variable_new in zip(self.variables, variables)]
            self.variables = variables

            for vv, vcs, vas in zip(self.variable_views, self.variable_curves, self.variables):
                for vc, va in zip(vcs, vas.T):
                    vc.setData(y=va)
                if neuron_window is not None:
                    vv.setYRange(neuron_window[0], self.neuron_window[1])
                vv.update()

        if self.densities is not None:
            densities = [densities] if not isinstance(densities, list) else densities
            densities = [np.vstack([density[shift:, :], density_new])
                         for density, density_new in zip(self.densities, densities)]

            self.densities = densities
            for dv, di, ds in zip(self.density_views, self.density_images, self.densities):
                di.updateImage(ds, levels=(np.min(ds), np.max(ds)))
                if neuron_window is not None:
                    dv.setYRange(neuron_window[0], self.neuron_window[1])
                dv.update()

        # noinspection PyArgumentList
        QtWidgets.QApplication.processEvents()
