# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Example
>>> from boards.virtual_board import RandomBoard
>>> board = RandomBoard()
>>> board
RandomBoard(n_channels=16, sample_rate=1000)

>>> board.start()
>>> data = board.get_data()
>>> data.shape
(16, 100)
"""
import numpy as np

from .board import Board


class RandomBoard(Board):
    def __init__(
            self,
            board_id: int = 0,
            n_channels: int = 16,
            sampling_rate: int = 1000,
            n_points: int = 100,
            seed: int | None = None,
            scale: float | None = None,
            *args, **kwargs):  # noqa
        self._board_id = board_id
        self._sampling_rate = sampling_rate
        self._n_channels = n_channels
        self._n_points = n_points
        self._scale = scale if scale is not None else 1.0

        if seed is None:
            seed = np.random.randint(0, 64000)
        self._seed = seed
        self._random = None

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def channels(self) -> list[int]:
        return list(range(self.n_channels))

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @property
    def board_id(self) -> int:
        return self._board_id

    def start(self):
        self._random = np.random.RandomState(self._seed)

    def stop(self):
        pass

    def get_data(self, n_points: int | None = None):
        return self._scale * self._random.rand(*(self.n_channels, n_points if n_points is not None else self._n_points))
