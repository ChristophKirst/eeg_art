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
(16, ...)

>>> board.get_elapsed_time()
...

>>> board.get_elapsed_time_since_access()
...

>>> board.get_elapsed_time() - board.get_elapsed_time_since_access()
...

>>> from boards.virtual_board import PlaybackBoard
>>> board = PlaybackBoard(recording='./data/example/recording.npy', loop=True)
>>> board
PlaybackBoard(n_channels=32, sample_rate=1000, file=./data/example/recording.npy)

>>> board.start()

>>> data = board.get_current_data(100)
>>> data.shape
(..., 32)

>>> board.get_n_samples_since_start()
...
"""
import time
import logging
import numpy as np

from abc import ABC
from typing import Self

from .board import Board

DEFAULT_RECORDING = './data/example/recording.npy'


class RingbufferBoard(Board, ABC):
    def __init__(
            self,
            sampling_rate: int = None,
            buffer_size: int | None = None,
    ):
        super().__init__()
        self._sampling_rate = sampling_rate
        self._start_time = None
        self._last_access = None
        self._buffer_size = buffer_size

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    def start(self) -> Self:
        self._start_time = time.perf_counter()
        self._last_access = self._start_time
        return self

    def set_last_access(self, access_time: float | None = None):
        self._last_access = access_time if access_time is not None else time.perf_counter()

    def get_start_time(self):
        return self._start_time

    def get_current_time(self, current_time: float | None = None) -> float:  # noqa
        return time.perf_counter() if current_time is None else current_time

    def get_elapsed_time(self, current_time: float | None = None) -> float:
        return self.get_current_time(current_time) - self.get_start_time()

    def get_last_access_time(self) -> float:
        return self._last_access

    def get_elapsed_time_since_access(self, current_time: float | None = None) -> float:
        return self.get_current_time(current_time) - self.get_last_access_time()

    def get_n_samples_since_start(self, current_time: float | None = None) -> int:
        return int(self.sampling_rate * self.get_elapsed_time(current_time))

    def get_n_samples_since_access(self, current_time: float | None = None) -> int:
        return int(self.sampling_rate * self.get_elapsed_time_since_access(current_time))

    def get_buffer_data_count(self, current_time: float | None = None) -> int:
        count = self.get_n_samples_since_access(current_time)
        if self._buffer_size is not None:
            if count > self._buffer_size:
                logging.warn('{self} ringbuffer overflow')
                count = self._buffer_size
        return count


class RandomBoard(RingbufferBoard):
    def __init__(
            self,
            board_id: int = 0,
            n_channels: int = 16,
            sampling_rate: int = 1000,
            seed: int | None = None,
            scale: float | None = None,
            bias: float | None = None,
            buffer_size: int | None = None,
            *args, **kwargs  # noqa
    ):
        RingbufferBoard.__init__(self, sampling_rate=sampling_rate, buffer_size=buffer_size)
        self._board_id = board_id
        self._n_channels = n_channels
        self._scale = scale if scale is not None else 1.0
        self._bias = bias if bias is not None else 0.0
        self._seed = np.random.randint(0, 64000) if seed is None else seed
        self._random = None

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def channels(self) -> list[int]:
        return list(range(self.n_channels))

    @property
    def board_id(self) -> int:
        return self._board_id

    def start(self) -> Self:
        RingbufferBoard.start(self)
        self._random = np.random.RandomState(self._seed)
        return self

    def stop(self):
        pass

    def get_data_count(self) -> int:
        return self.get_buffer_data_count()

    def get_data(self, n_samples: int | None = None):
        current_time = self.get_current_time()
        n_data = self.get_buffer_data_count(current_time=current_time)
        n_samples = min(n_data, n_samples) if n_samples is not None else n_data
        self.set_last_access(access_time=current_time)
        return self._scale * self._random.rand(*(self.n_channels, n_samples)) + self._bias

    def get_current_data(self, n_samples: int | None):
        return self.get_data(n_samples)


class PlaybackBoard(RingbufferBoard):
    def __init__(
        self,
        recording: str | np.ndarray | None = None,
        board_id: int = 0,
        sampling_rate: int = 1000,
        n_channels: int | None = None,
        channels: list[int] | None = None,
        scale: float | None = None,
        buffer_size: int | None = None,
        loop: bool = False,
        *args, **kwargs  # noqa
    ):
        RingbufferBoard.__init__(self, sampling_rate=sampling_rate, buffer_size=buffer_size)
        recording = recording if recording is not None else './data/example/recording.npy'
        self._recording_file = recording if isinstance(recording, str) else None
        self._recording = np.load(recording) if isinstance(recording, str) else recording
        self._board_id = board_id
        n_channels = self._recording.shape[1] if n_channels is None else n_channels
        self._channels = channels if channels is not None else list(range(n_channels))
        self._scale = scale if scale is not None else 1.0
        self._loop = loop

    @property
    def channels(self) -> list[int]:
        return self._channels

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_times(self) -> int:
        return self._recording.shape[0]

    def __len__(self):
        return self.n_times

    @property
    def session_length(self) -> float:
        return self.n_times * self.sampling_rate

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @property
    def recording(self) -> np.ndarray:
        return self._recording

    def start(self, *args, **kwargs) -> Self:
        return RingbufferBoard.start(self)

    def stop(self):
        pass

    def get_data_count(self) -> int:
        return self.get_buffer_data_count()

    def _get_data(self, index, n_samples):
        if n_samples == 0:
            return np.zeros((self.n_channels, 0))
        channels = self.channels
        recording = self.recording[:, channels]
        n_times = self.n_times
        if self._loop:
            n_arrays = 1
            looped_recording = recording
            if n_samples > n_times:
                n_arrays = (n_samples - 1) // n_times + 1
                looped_recording = np.tile(recording, (1, n_arrays))
            if index > n_times:
                looped_recording = np.concatenate([looped_recording, recording], axis=0)
                n_arrays += 1
                index = index % n_times + (n_arrays - 1) * n_times
            return looped_recording[index-n_samples:index]
        else:
            if index - n_samples > n_times or index > n_times:
                logging.warn('requesting recording data out of range')
            return recording[max(0, index - n_samples):min(index, n_times)]

    def get_data(self, n_samples: int | None):
        current_time = self.get_current_time()
        n_data = self.get_buffer_data_count(current_time=current_time)
        index = self.get_n_samples_since_start(current_time=current_time)
        self.set_last_access(current_time)
        return self._get_data(index, min(n_data, n_samples))

    def get_current_data(self, n_samples: int | None):
        current_time = self.get_current_time()
        n_data = self.get_buffer_data_count(current_time=current_time)
        index = self.get_n_samples_since_start(current_time=current_time)
        return self._get_data(index, min(n_data, n_samples))

    def __repr__(self):
        r = RingbufferBoard.__repr__(self)
        if self._recording_file is not None:
            r = f"{r[:-1]}, file={self._recording_file})"
        return r
