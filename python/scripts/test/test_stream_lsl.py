# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst
"""

"""Example program to show how to read a multi-channel time series from LSL."""
import time
from pylsl import StreamInlet, resolve_stream
from time import sleep

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
duration = 10

sleep(1)


def testLSLSamplingRate():
    start = time.time()
    totalNumSamples = 0
    validSamples = 0
    numChunks = 0

    while time.time() <= start + duration:
        # get chunks of samples
        samples, timestamp = inlet.pull_chunk()
        print(len(samples), len(timestamp))
        if samples:
            numChunks += 1
            #print(timestamp, len(samples) )
            totalNumSamples += len(samples)
            # print(samples);
            for sample in samples:
                print(sample)
                validSamples += 1

    print( "Number of Chunks and Samples == {} , {}".format(numChunks, totalNumSamples) )
    print( "Valid Samples and Duration == {} / {}".format(validSamples, duration) )
    print( "Avg Sampling Rate == {}".format(validSamples / duration) )


testLSLSamplingRate()


#%%
# %gui qt5


import time
from pylsl import StreamInlet, resolve_stream
from time import sleep

import numpy as np
import pyqtgraph as pg
import matplotlib as mpl

pg.setConfigOption('background', 'w')
colors = np.array(255* mpl.cm.coolwarm(np.linspace(0,1,8)), dtype=int)

x = np.random.normal(size=10)
y = np.random.normal(size=10)
p = pg.plot(x, y)  ## setting pen=None disables line drawing
for i in range(1,8):
  x = np.random.normal(size=10)
  y = np.random.normal(size=10)
  p.plot(x,y, pen=pg.mkPen(color=colors[i,:3],style=None, symbol='o'))
curves = p.plotItem.curves


# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
duration = 10

sleep(1)


def stream_plot(duration = 10, max_points = 10000, offset = 100):
    start = time.time()
    plot_data = np.zeros((0,8))
    time_data = np.zeros((0,))
    while time.time() <= start + duration:
        # get chunks of samples
        samples, timestamp = inlet.pull_chunk()
        if samples:
          print('new data: length=%d' % len(timestamp))
          data = np.vstack(samples)
          plot_data = np.concatenate([plot_data, data], axis=0)
          #time_data = np.hstack([time_data, timestamp])
          
          if plot_data.shape[0] > max_points:
            #time_data = time_data[-max_points:];
            plot_data = plot_data[-max_points:]
          p.enableAutoRange('xy', False)
          for i in range(8):
            curves[i].setData(np.arange(plot_data[:,i].shape[0]),plot_data[:,i] + i * offset)
          p.enableAutoRange('xy', True)
          #p.plotItem.update()
          pg.QtGui.QApplication.processEvents()
            
stream_plot(10)


#%% use it to generate nice images somehow?



#%%

win = pg.GraphicsWindow(title="Sample process")
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

# Random data process
p6 = win.addPlot(title="Updating plot")
curve = p6.plot(pen='y')
data = np.random.normal(size=(10,1000)) #  If the Gaussian distribution shape is, (m, n, k), then m * n * k samples are drawn.

# plot counter
ptr = 0 

# Function for updating data display
def update():
    global curve, data, ptr, p6
    curve.setData(data[ptr%10])
    if ptr == 0:
        p6.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
    ptr += 1

# Update data display    
timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)