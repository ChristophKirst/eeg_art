# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst
"""

from abc import ABC, abstractmethod
from typing import Self


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
    def start(self, *args, **kwargs) -> Self:
        ...

    @abstractmethod
    def stop(self):
        ...

    def __del__(self):
        self.stop()

    @abstractmethod
    def get_data(self, n_samples: int | None):
        ...

    @abstractmethod
    def get_current_data(self, n_samples: int | None):
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(n_channels={self.n_channels}, sample_rate={self.sampling_rate})"
