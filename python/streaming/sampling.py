# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Example
>>> import time
>>> from boards.board_buffer import SampleCounter, Sampler
>>> counter = SamplingCounter()
>>> counter
SampleCounter(sampling_rate=1000)

>>> counter.start()
>>> time.sleep(1.5)
>>> counter.get_n_samples_since_start()
1500

>>> counter.update()
>>> time.sleep(0.5)
>>> counter.get_n_samples_since_sampling()
500

>>> sampler = Sampler(update_interval=0.5, sampling_rate=1000)
>>> sampler
Sampler(sampling_rate=1000, update_interval=500)

>>> sampler.start()
>>> time.sleep(1.5)
>>> sampler.get_n_samples_since_sampling()
"""
from abc import ABC, abstractmethod

import logging

import numpy as np

from .buffer import ArrayBuffer, RingArrayBuffer



class SamplingCounter(Timer):
    def __init__(
            self,
            sampling_rate: int = 1000,
            max_sample_size: int | None = None
    ):
        Timer.__init__(self)
        self._sampling_rate = sampling_rate
        self._max_sample_size = max_sample_size
        self._last_sampling_time = None

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @property
    def max_sample_size(self) -> int | None:
        return self._max_sample_size

    def get_sampling_time(self):
        return self._last_sampling_time

    def set_sampling_time(self, current_time: float | None = None):
        self._last_sampling_time = self.get_current_time(current_time)

    def get_time_elapsed_since_sampling(self, current_time: float | None = None) -> float:
        return self.get_current_time(current_time) - self.get_sampling_time()

    def _get_n_samples(
            self,
            n_samples: int,
            max_sample_size: bool | int | None = None
    ) -> int:
        max_sample_size = True if max_sample_size is None else max_sample_size
        if max_sample_size is not False:
            max_sample_size = self._max_sample_size if max_sample_size is True else max_sample_size
            n_samples = min(n_samples, max_sample_size)
        return n_samples

    def get_n_samples_since_start(
            self,
            current_time: float | None = None,
            max_sample_size: bool | int | None = None
    ) -> int:
        n_samples = int(self._sampling_rate * self.get_time_elapsed_since_start(current_time))
        return self._get_n_samples(n_samples, max_sample_size)

    def get_n_samples_since_sampling(
            self,
            current_time: float | None = None,
            max_sample_size: bool | int | None = None
    ) -> int:
        n_samples = int(self._sampling_rate * self.get_time_elapsed_since_sampling(current_time))
        return self._get_n_samples(n_samples, max_sample_size)

    def get_current_index(self, current_time: float | None = None):
        return self.get_n_samples_since_start(current_time=current_time, max_sample_size=False)

    def get_sampling_index(self):
        return self.get_current_index(current_time=self.get_sampling_time())

    def start(self):
        Timer.start(self)

    def stop(self):
        Timer.stop(self)
        self._last_sampling_time = None

    def __repr__(self):
        return f"{self.__class__.__name__}(sampling_rate={self._sampling_rate}, max_samples={self._max_sample_size})"


class Sampler(SamplingCounter, ABC):
    def __init__(
            self,
            sampling_rate: int = 1000,
            max_sample_size: int | None = None,
            sample_shape: tuple[int, ...] | None = None,
            sample_dtype: type = float
    ):
        SamplingCounter.__init__(self, sampling_rate=sampling_rate, max_sample_size=max_sample_size)
        self._sample_shape = sample_shape
        self._sample_dtype = sample_dtype

    @property
    def sample_shape(self):
        return self._sample_shape

    @property
    def sample_dtype(self):
        return self._sample_dtype

    def get_n_samples(self, n_samples: int | None = None, max_sample_size: bool | int | None = None):
        return self.get_n_samples_since_sampling(max_sample_size=max_sample_size) if n_samples is None else n_samples

    def get_data_shape(self, n_samples: int | None = None):
        return (self.get_n_samples(n_samples),) + self.sample_shape

    @abstractmethod
    def get_data(self, n_samples: int | None = None, remove_data: bool = True, **kwargs):
        """Get recent data of up to n_samples or less."""
        ...

#
# class Generator(Sampler, ABC):
#     def __init__(
#             self,
#             sampling_rate: int = 1000,
#             max_sample_size: int | None = None,
#             sample_shape: tuple[int, ...] | None = None,
#             sample_dtype: type = float
#     ):
#         Sampler.__init__(
#             self,
#             sampling_rate=sampling_rate,
#             max_sample_size=max_sample_size,
#             sample_shape=sample_shape,
#             sample_dtype=sample_dtype
#         )
#         self._buffer = None if max_sample_size is None else \
#             RingArrayBuffer(shape=(max_sample_size,)+sample_shape, dtype=sample_dtype)
#
#     @abstractmethod
#     def generate_data(self, n_samples: int, start_index: int | None = None):
#         ...
#
#     def get_data(
#             self,
#             n_samples: int | None = None,
#             remove_data: bool = True,
#             current_time: float | None = None,
#             **kwargs
#     ):
#         current_time = self.get_current_time(current_time=current_time)
#         index = self.get_current_index(current_time=current_time)
#         n_samples = self.get_n_samples(n_samples=n_samples)
#
#         if self.max_sample_size is None:
#             data = self.generate_data(n_samples, index - n_samples)
#         else:
#             data = self.generate_data(max_smaple_size, index - max_sample_size)
#             self._buffer[:] = data
#
#         if remove_data:
#             self._buffer_index = index
#
#         if not remove_data:
#
#         last_index = self.get_sampling_index()
#
#         n_samples = self.get_n_samples(n_samples=n_samples)
#
#
#         data = self.generate_data(n_samples, start_index=index - n_samples)
#
#
#
#
#
#
#     def _get_data_since_sampling(
#             self,
#             n_samples: int | None = None,
#             current_time: float | None = None,
#             data_getter: callable = self.get_data,
#             data_buffer: np.ndarray | None = None,
#             max_sample_size: bool | int = True,
#     ):
#         current_time = self.get_current_time(current_time=current_time)
#         n_samples_since_sampling = \
#             self.get_n_samples_since_sampling(current_time=current_time, max_sample_size=max_sample_size)
#
#         data = data_getter(n_samples_since_sampling)
#
#
#
#
#     # @abstractmethod
#     # def get_data_size(self) -> int:
#     #     """Get the size of the current data available"""
#     #     ...
#
#
#     def get_full_data(
#             self,
#             n_samples: int,
#             current_time: float | None = None,
#             data_getter: callable = self.get_data,
#             data_buffer: np.ndarray | None = None,
#     ):
#
#         current_time = self.get_current_time(current_time=current_time)
#         n_samples_all = self.get_n_samples_since_sampling(current_time=current_time, use_max_size=False)
#
#
#         if n_samples n_samples > n_samples_all:
#             n_samples = n_samples_all
#
#         data = data_getter(n_samples)
#         n_samples = len(data)
#
#         if n_samples < n_samples_full:
#             if data_buffer is None:
#                 buffer = np.zeros((n_samples_full - n_samples) + data.shape[1:])
#             else:
#                 buffer = data_buffer[-n_samples_full - n_samples:]
#             data = np.concatenate([buffer, data], axis=0)
#
#         return data
#
#     def get_buffered_data(
#             self,
#             n_samples: int | None = None,
#             current_time: float | None = None,
#             buffer: np.ndarray | None = None):
#         return self._get_buffered_data(self.get_current_data, n_samples, current_time, buffer)
#
#
#
#
#
#
# class BufferedSampler(Sampler, ABC): # buffered for
#     def __init__(
#             self,
#             sampling_rate: int = 1000,
#             max_sample_size: int | None = None,
#             sample_shape: tuple[int] = (1,),
#             buffer_size: int | None = None,
#             buffer_duration: float | None = None,
#             buffer_mode: str = 'append'
#     ):
#         Sampler.__init__(self, sampling_rate=sampling_rate, max_sample_size=max_sample_size)
#         if buffer_size is not None:
#             self._buffer_size = buffer_size
#         elif buffer_duration is not None:
#             self._buffer_size = int(buffer_duration * sampling_rate)
#         else:
#             self._buffer_size = None
#
#         self._sample_shape = sample_shape
#
#         buffer_shape = (self.buffer_size if self.buffer_size is not None else 64,) + self.sample_shape
#         self.buffer = BufferedArray(shape=buffer_shape)
#
#         if buffer_mode not in ['append', 'insert']:
#             raise ValueError
#         self._buffer_mode = buffer_mode
#
#     @property
#     def sample_shape(self):
#         return self._sample_shape
#
#     @property
#     def buffer_size(self):
#         return self._buffer_size
#
#     @property
#     def buffer_duration(self):
#         return self.buffer_size / self.sampling_rate if self.buffer_size is not None else None
#
#     @property
#     def buffer_mode(self):
#         return self._buffer_mode
#
#     def get_data_to_buffer(self, n_samples: int | None = None):
#         current_time = self.get_current_time()
#
#         # get data
#         if self.buffer_mode == 'insert':
#             data = self.get_full_data(n_samples=n_samples, current_time=current_time)
#         else:
#             data = self.get_data(n_samples=n_samples)
#         data_size = len(data)
#         if data_size == 0:
#             return
#
#         # update buffer
#         buffer_size = self.buffer_size
#         if buffer_size is None:  # infinite buffer
#             self.buffer.append(data)
#         else:  # ring buffer
#             if data_size > buffer_size:
#                 data = data[-buffer_size:]
#
#             if data_size < buffer_size:
#                 self.buffer.roll(-data_size)
#                 self.buffer[-data_size:] = data
#             else:
#                 self.buffer[:] = data[:]
#
#     @property
#     def shape(self):
#         return self.buffer.shape
#
#     def __len__(self):
#         return len(self.buffer)
#
#     def __getitem__(self, item):
#         return self.buffer.__getitem__(item)
#
#     def __setitem__(self, key, value):
#         self.buffer.__setitem__(key, value)
#
#     def __repr__(self):
#         return f"{Sampler.__repr__(self)[:-1]}, buffer_size={self.buffer_size}, buffer_mode={self.buffer_mode})"
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# def rand(state: np.random.RandomState, shape: tuple[int, ...]):
#     return state.rand(*shape)
#
#
# class RandomSampler(Sampler):
#     def __init__(
#             self,
#             sampling_rate: int = 1000,
#             max_sample_size: int | None = None,
#             sample_shape: tuple[int, ...] = (1,),
#             seed: int | None = None,
#             method: callable = rand
#     ):
#         Sampler.__init__(self, sampling_rate=sampling_rate, max_sample_size=max_sample_size, sample_shape=sample_shape)
#         self._seed = np.random.randint(0, 64000) if seed==-1 else seed
#         self._random = None
#         self._method = method
#         self._buffer = None
#
#     def initialize_random(self):
#         self._seed = seed if seed is not None else np.random.randint(0, 64000)
#         self._random = np.random.RandomState(seed=seed)
#
#     def start(self):
#         self.initialize_random()
#         Sampler.start(self)
#
#     def generate_data(self, n_samples: int):
#         return method(self._random, (n_samples,) + self.sample_shape)
#
#     def get_data(self, n_samples: int | None = None):
#         n_samples_all = self.get_n_samples_since_sampling(max_sample_size=False)
#         n_samples_since_sampling = self.get_n_samples_since_sampling()
#         data = self.generate_data(n_samples_all)[-n_samples_since_sampling:]
#
#         self.set_sampling_time()
#         n_samples = n_samples_since_sampling if n_samples is None else min(n_samples, n_samples_since_sampling)
#         return
#
#     def get_current_data(self, n_samples: int):
#         current_time = self.get_current_time()
#
#         max_sample_size = self.max_sample_size
#         if max_sample_size is not None:  # ring buffer
#             data = self.get_full_current_data(n_samples=n_samples, current_time=current_time, buffer=self._buffer)
#             self._buffer = data
#         else:  # new data assuming no ring buffer
#             data = method(self._random, (n_samples_since_sampling,) + self.sample_shape)
#
#
#
#         else:
#             n_samples = min(n_samples, self.get_n_samples_since_start())
#             self.set_sampling_time()
#             return method(self._random, (n_samples,) + self.sample_shape)
#
#
# class PlaybackSampler(Sampler):
#     def __init__(
#             self,
#             sampling_rate: int = 1000,
#             max_sample_size: int | None = None,
#             sample_shape: tuple[int, ...] = (1,),
#             seed: int | None = None,
#             method: callable = rand
#     ):
#         Sampler.__init__(sampling_rate=sampling_rate, max_sample_size=max_sample_size, sample_shape=sample_shape)
#         self._seed = np.random.randint(0, 64000) if seed==-1 else seed
#         self._random = None
#         self._method = method
#
#
#     def __init__(
#         self,
#         recording: str | np.ndarray | None = None,
#         board_id: int = 0,
#         sampling_rate: int = 1000,
#         n_channels: int | None = None,
#         channels: list[int] | None = None,
#         scale: float | None = None,
#         buffer_size: int | None = None,
#         loop: bool = False,
#         *args, **kwargs  # noqa
#     ):
#         RingbufferBoard.__init__(self, sampling_rate=sampling_rate, buffer_size=buffer_size)
#         recording = recording if recording is not None else './data/example/recording.npy'
#         self._recording_file = recording if isinstance(recording, str) else None
#         self._recording = np.load(recording) if isinstance(recording, str) else recording
#         self._board_id = board_id
#         n_channels = self._recording.shape[1] if n_channels is None else n_channels
#         self._channels = channels if channels is not None else list(range(n_channels))
#         self._scale = scale if scale is not None else 1.0
#         self._loop = loop
#
#     @property
#     def channels(self) -> list[int]:
#         return self._channels
#
#     @property
#     def n_channels(self) -> int:
#         return len(self.channels)
#
#     @property
#     def n_times(self) -> int:
#         return self._recording.shape[0]
#
#     def __len__(self):
#         return self.n_times
#
#     @property
#     def session_length(self) -> float:
#         return self.n_times * self.sampling_rate
#
#     @property
#     def sampling_rate(self) -> int:
#         return self._sampling_rate
#
#     @property
#     def recording(self) -> np.ndarray:
#         return self._recording
#
#     def start(self, *args, **kwargs) -> Self:
#         return RingbufferBoard.start(self)
#
#     def stop(self):
#         pass
#
#     def get_data_count(self) -> int:
#         return self.get_buffer_data_count()
#
#     def _get_data(self, index, n_samples):
#         if n_samples == 0:
#             return np.zeros((self.n_channels, 0))
#         channels = self.channels
#         recording = self.recording[:, channels]
#         n_times = self.n_times
#         if self._loop:
#             n_arrays = 1
#             looped_recording = recording
#             if n_samples > n_times:
#                 n_arrays = (n_samples - 1) // n_times + 1
#                 looped_recording = np.tile(recording, (1, n_arrays))
#             if index > n_times:
#                 looped_recording = np.concatenate([looped_recording, recording], axis=0)
#                 n_arrays += 1
#                 index = index % n_times + (n_arrays - 1) * n_times
#             return looped_recording[index-n_samples:index]
#         else:
#             if index - n_samples > n_times or index > n_times:
#                 logging.warn('requesting recording data out of range')
#             return recording[max(0, index - n_samples):min(index, n_times)]
#
#     def get_data(self, n_samples: int | None):
#         current_time = self.get_current_time()
#         n_data = self.get_buffer_data_count(current_time=current_time)
#         index = self.get_n_samples_since_start(current_time=current_time)
#         self.set_last_access(current_time)
#         return self._get_data(index, min(n_data, n_samples))
#
#     def get_current_data(self, n_samples: int | None):
#         current_time = self.get_current_time()
#         n_data = self.get_buffer_data_count(current_time=current_time)
#         index = self.get_n_samples_since_start(current_time=current_time)
#         return self._get_data(index, min(n_data, n_samples))
#
#     def __repr__(self):
#         r = RingbufferBoard.__repr__(self)
#         if self._recording_file is not None:
#             r = f"{r[:-1]}, file={self._recording_file})"
#         return r