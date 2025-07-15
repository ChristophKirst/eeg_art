# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Example
>>> from boards.virtual_board import RandomBoard
>>> from visualization.eeg_viewer import EEGViewer
>>> board = RandomBoard()
>>> board.start()
>>> viewer = EEGViewer(board=board)
>>> viewer.show()
>>> viewer.start()
"""
import numpy as np
import functools as tf

import pyqtgraph as pg
pg.CONFIG_OPTIONS['useOpenGL'] = False
app = pg.mkQApp() if not pg.QAPP else pg.QAPP

from pyqtgraph import Qt
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets

from boards.board import Board

from utils.pycharm_gui import initialize_pycharm_gui
initialize_pycharm_gui()


class EEGViewer(QtWidgets.QWidget):
    def __init__(
            self,
            board: Board,
            update_interval: int = 50,  # msec
            window_duration: float = 4,  # sec
            parent=None
    ):
        QtWidgets.QWidget.__init__(self, parent=parent)

        self.board: Board = board
        self.sampling_rate = board.sampling_rate
        self.update_interval = update_interval
        self.window_duration = window_duration
        self.n_points = int(self.window_duration * self.sampling_rate)
        self.data_buffer = np.zeros((self.board.n_channels, self.n_points))

        # initialize gui
        self.setWindowTitle('EEG Viewer')

        self.layout = QtWidgets.QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.view = pg.ViewBox()
        self.graphics_view = pg.GraphicsView()
        self.graphics_view.setCentralItem(self.view)

        self.splitter = QtWidgets.QSplitter(self)
        self.splitter.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.splitter.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.splitter)

        self.curves = []
        self.initialize_timeseries_plots()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)

    def initialize_timeseries_plots(self):
        self.curves = []
        for i in range(self.board.n_channels):
            plot = pg.PlotItem()
            curve = pg.PlotCurveItem(y=np.zeros(self.n_points), pen=(255,0,0))
            plot.addItem(curve)

            view = pg.GraphicsView()
            view.setCentralItem(plot)
            view.centralWidget.setXLink(self.graphics_view.centralWidget)
            plot.showAxis('left', False)
            plot.setMenuEnabled('left', False)
            plot.showAxis('bottom', False)
            plot.setMenuEnabled('bottom', False)
            if i == 0:
                plot.setTitle('TimeSeries Plot')

            view_box = plot.getViewBox()
            view_box.setXRange(0, self.n_points, padding=0.01)
            view_box.autoRangeEnabled()

            self.curves.append(curve)
            self.splitter.addWidget(view)

    def update(self):
        channels = np.array(self.board.channels, dtype=int)

        data = self.board.get_data(self.n_points)

        n_new_points = data.shape[1]
        if n_new_points < self.n_points:
            self.data_buffer = np.roll(self.data_buffer.roll, -n_new_points, axis=1)
            self.data_buffer[:, -n_new_points:] = data[channels]
        else:
            self.data_buffer[:] = data[channels]

        for i, curve in enumerate(self.curves):
            curve.setData(y=self.data_buffer[i])

        app.processEvents()  # noqa

    def start(self):
        self.timer.start(self.update_interval)

    def stop(self):
        self.timer.stop()
