# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Examples
--------
>>> from streaming.buffer.ring_buffer import RingBuffer
>>> r = RingBuffer(max_length=3)
>>> r
RingBuffer(shape=(0,), dtype=float, max_length=3)

>>> r.update([1, 2])
>>> r.update([3, 4, 5])
>>> r
RingBuffer(shape=(2,), dtype=float, max_length=3)

>>> r.array_shape
(5,)

>>> r.to_array()
array([1, 2, 3, 4, 5])

>>> r.update([6, 6, 6, 6])
>>> r.update([10, 10])
>>> r.to_array()
array([ 3,  4,  5,  6,  6,  6,  6, 10, 10])

>>> r.array_shape
(9,)

>>> r.read_array(6)
array([3, 4, 5, 6, 6, 6])

>>> r.to_array()
array([ 6, 10, 10])

>>> r.clear()
>>> r.shape
(0,)
"""
from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from .buffer import Buffer


class RingBuffer(Buffer):
    def __init__(
            self,
            sample_shape: tuple[int, ...] | None = None,
            dtype: type = float,
            max_length: int | None = None,
            data: deque | None = None
    ):
        if data is not None and len(data) > 0:
            sample_shape = data[0].shape[1:]
            dtype = data[0].dtype
            max_length = data.maxlen
        else:
            data = deque(iterable=(), maxlen=max_length)
            sample_shape = () if sample_shape is None else sample_shape
        Buffer.__init__(self, sample_shape=sample_shape, dtype=dtype, max_length=max_length)
        self.data: deque = data
        self.zero_index = 0

    def __len__(self):
        return len(self.data)

    @property
    def shape(self) -> tuple[int, ...]:
        return len(self),

    def update(self, value):
        self.data.append(value)

    def to_array(self) -> np.ndarray:
        return np.concatenate(self.data, axis=0)

    @property
    def array_len(self):
        return sum([len(d) for d in self.data])

    @property
    def array_shape(self):
        return (self.array_len,) + self.sample_shape

    def read(self, length: int | None = None, delete_data: bool = True, reduce_length: bool = False):
        if length is None:
            data = list(self.data)
            if delete_data:
                self.clear()
        else:
            if length > len(self):
                if reduce_length:
                    length = len(self)
                else:
                    raise ValueError
            if delete_data:
                data = [self.data.popleft() for _ in range(length)]
            else:
                data = [self.data[i] for i in range(length)]
        return data

    def clear(self):
        self.data.clear()

    def read_array(self, length: int | None, reduce_length: bool = False):
        if length is None:
            data = self.to_array()
            self.clear()
        else:
            lengths = [len(d) for d in self.data]
            if length > sum(lengths):
                if reduce_length:
                    length = sum(lengths)
                else:
                    raise ValueError
            e = np.where(length <= np.cumsum(lengths))[0][0] + 1
            data = [self.data.popleft() for _ in range(e - 1)]
            final_block = self.data.popleft()
            n = sum(lengths[:e-1])
            i = length-n
            data.append(final_block[:i])
            if i < len(final_block):
                self.data.appendleft(final_block[i:])
            data = np.concatenate(data, axis=0)
        return data

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def max(self):
        return max([max(d) for d in self.data])

    def min(self):
        return min([min(d) for d in self.data])
