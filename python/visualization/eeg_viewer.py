# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst
"""

import numpy as np
import functools as tf

import pyqtgraph as pg
pg.CONFIG_OPTIONS['useOpenGL'] = False
app = pg.mkQApp() if not pg.QAPP else pg.QAPP

from pyqtgraph import Qt
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets



import brainflow
from brainflow.board_shim import BoardShim #, BrainFlowInputParams, BoardIds, BrainFlowError
# from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations

#https://brainflow.readthedocs.io/en/stable/Examples.html#python

class EEGViewer(QtWidgets.QWidget):
    def __init__(self, board_shim, parent=None):
        QtWidgets.QWidget.__init__(self, parent=parent)

        self.setWindowTitle('EEG Viewer')
        self.initialize_board(board_shim)

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

        self.initialize_timeseries_plots()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)

    def start(self):
        self.timer.start(self.update_speed_ms)

    def stop(self):
        self.timer.stop()

    def initialize_board(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = board_shim.get_exg_channels(self.board_id)
        self.sampling_rate = board_shim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

    def initialize_timeseries_plots(self):
        self.curves = []
        for i in range(len(self.exg_channels)):
            plot = pg.PlotItem()
            curve = pg.PlotCurveItem(y=np.zeros(self.num_points), pen=(255,0,0))
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
            self.curves.append(curve)
            self.splitter.addWidget(view)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        #avg_bands = [0, 0, 0, 0, 0]
        print('updating: ', data.shape)
        for count, channel in enumerate(self.exg_channels):
            print(count, channel)
            # plot timeseries
            # DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            # DataFilter.perform_bandpass(data[channel], self.sampling_rate, 51.0, 100.0, 2,
            #                             FilterTypes.BUTTERWORTH.value, 0)
            # DataFilter.perform_bandpass(data[channel], self.sampling_rate, 51.0, 100.0, 2,
            #                             FilterTypes.BUTTERWORTH.value, 0)
            # DataFilter.perform_bandstop(data[channel], self.sampling_rate, 50.0, 4.0, 2,
            #                             FilterTypes.BUTTERWORTH.value, 0)
            # DataFilter.perform_bandstop(data[channel], self.sampling_rate, 60.0, 4.0, 2,
            #                             FilterTypes.BUTTERWORTH.value, 0)
            self.curves[count].setData(data[channel].tolist())

        app.processEvents()