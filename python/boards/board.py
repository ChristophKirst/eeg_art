# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst
"""

from abc import ABC, abstractmethod


class Board(ABC):
    @property
    @abstractmethod
    def n_channels(self) -> int:
        ...

    @property
    @abstractmethod
    def channels(self) -> list[int]:
        ...

    @property
    @abstractmethod
    def sampling_rate(self) -> int:
        ...

    @abstractmethod
    def start(self):
        ...

    @abstractmethod
    def stop(self):
        ...

    @abstractmethod
    def get_data(self, n_points: int | None):
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(n_channels={self.n_channels}, sample_rate={self.sampling_rate})"
