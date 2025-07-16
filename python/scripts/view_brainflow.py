# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst
"""
import utils.initialize_pycharm_gui

from boards import OpenBCIBoard as Board
# from boards import PlaybackBoard as Board

from visualization.eeg_viewer import EEGViewer

board = Board()
board.start(session_length=60)  # session_length in min

viewer = EEGViewer(board=board)
viewer.show()
viewer.start()
