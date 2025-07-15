# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst
"""

import numpy as np


class VirtualBoard(Board):
    def __init__(self, board_id, sampling_rate=1000, *args, **kwargs):
        self._board_id = board_id
        self._sampling_rate = sampling_rate
        self._n_channels = 16

    @property
    def

    def prepare_session(self, *args, **kwargs):
        return

    def start_stream(self, *args, **kwargs):
        return

    @classmethod
    def enable_dev_board_logger(cls):
        return

    def is_prepared(self):
        return True

    def release_session(self):
        return

    def get_board_id(self):
        return self.board_id

    def get_exg_channels(self, board_id):
        return [i for i in range(self.n_channels)]

    def get_sampling_rate(self, board_id):
        return self.sampling_rate

    def get_current_board_data(self, num_points):
        return np.random.rand(*(self.n_channels, num_points))