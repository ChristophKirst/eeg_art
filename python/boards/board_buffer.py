# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Example
>>> import time
>>> from boards import RandomBoard
>>> board = RandomBoard().start()
>>> from boards.board_buffer import BoardBuffer
>>> buffer = BoardBuffer(board=board, update_interval=0.2)
>>> buffer.start()
>>> time.sleep(1.5)
>>> buffer.shape

>>> buffer.stop()

"""
from abc import ABC, abstractmethod

import time
import threading
import logging

import numpy as np

from .board import Board
from streaming.sampling import BufferedSampler


class BoardBuffer(BufferedSampler):
    def __init__(
            self,
            board: Board,
            channels: list[int] | None = None,
            buffer_length: float | None = None,  # sec
            update_interval: float = 0.2,  # sec
            buffer_mode: str = 'append'
    ):
        self.channels = channels if channels is not None else board.channels
        BufferedSampler.__init__(
            self,
            data_shape=(len(self.channels),),
            update_interval=update_interval,
            sampling_rate=board.sampling_rate,
            buffer_mode=buffer_mode
        )
        self.board: Board = board

    def get_current_data(self, n_samples: int | None = None):
        return self.board.get_current_data(n_samples)

    def get_data(self, n_samples: int | None = None):
        return self.board.get_data(n_samples)

    def __repr__(self):
        return f"{BufferedSampler.__repr__(self)[:-1]}, board={self.board}, channels={self.channels})"
