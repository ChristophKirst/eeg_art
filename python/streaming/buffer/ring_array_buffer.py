# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Examples
--------
>>> from streaming.buffer.array_buffer import RingArrayBuffer
>>> r = RingArrayBuffer(sample_shape=(), max_length=5, dtype=int)
>>> r
RingArrayBuffer(shape=(0,), dtype=int, max_length=5)

>>> r.update([1, 1])
>>> r[:]
array([1, 1])

>>> r.update((2,))
>>> r[:]
array([1, 1, 2])

>>> r[-2:]
array([1, 2])

>>> r.data
array([1, 1, 2, 0, 0])

>>> from streaming.buffer.array_buffer import RingArrayBuffer
>>> r = RingArrayBuffer(sample_shape=(3,), max_length=5, dtype=int)
>>> r
RingArrayBuffer(shape=(0, 3), dtype=int, max_length=5)

>>> r.update([[1, 2, 3]])
>>> r[:]
 array([[1, 2, 3]])

>>> r.update([(4, 5, 6), (7, 8, 9)])
>>> r[:]
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])

>>> r.update([(-1, -1, -1), (-2, -2, -2), (-3, -3, -3)])
>>> r[:]
array([[ 4,  5,  6],
       [ 7,  8,  9],
       [-1, -1, -1],
       [-2, -2, -2],
       [-3, -3, -3]])
"""
from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from .buffer import Buffer
from .array_buffer import _index_for_length


class RingArrayBuffer(Buffer):
    """"""
    def __init__(
            self,
            sample_shape: tuple[int, ...] | None = None,
            dtype: type = float,
            max_length: int | None = None,
            data: np.ndarray | None = None,
    ):
        if data is not None:
            max_length = len(data)
            sample_shape = data.shape[1:]
            dtype = data.dtype
            length = max_length
        else:
            if max_length is None or sample_shape is None:
                raise ValueError
            shape = (max_length,) + sample_shape
            data = np.zeros(shape, dtype=dtype)
            length = 0

        Buffer.__init__(self, sample_shape=sample_shape, dtype=dtype, max_length=max_length)
        self.data: np.ndarray = data
        self.zero_index: int = 0
        self.length = length

    def __len__(self):
        return self.length

    def update(self, value):
        self.extend(value)

    def to_array(self) -> np.ndarray:
        return self[:]

    def _index(self, index):
        index = _index_for_length(index, self.length, replace_slice=True)
        return ((index[0] + self.zero_index) % self.max_length,) + index[1:]

    def __getitem__(self, index):
        return self.data[self._index(index)]

    def __setitem__(self, index, value):
        self.data[self._index(index)] = value

    def read(self, length: int | None = None, delete_data: bool = True, reduce_length: bool = False):
        if length is None:
            data = self[:]
            if delete_data:
                self.clear()
        else:
            if length > self.length:
                if reduce_length:
                    length = self.length
                else:
                    raise ValueError
            data = self[:length]
            if delete_data:
                self.delete(slice(None, length))
        return data

    def clear(self):
        self.zero_index = 0
        self.length = 0

    def append(self, value):
        self.extend([value])

    def extend(self, value):
        n = self.max_length
        l = len(value)

        if l > n:
            raise ValueError

        length = self.length
        s = (self.zero_index + length) % n
        e = s + l
        if e <= n:
            self.data[s:e] = value
        else:
            self.data[s:] = value[:n-s]
            self.data[:e-n] = value[n-s:]

        if length < n:
            self.length = min(length + l, n)

        self.zero_index = (e - self.length) % n

    def delete(self, indices):
        data = np.delete(self[:], indices, axis=0)
        self.clear()
        length = len(data)
        self.data[:length] = data
        self.length = length
        self.zero_index = 0

    def resize(self, size: int, fill_value=0):
        n = len(self)
        if size < n:
            self.delete(slice(size, n))
        else:
            new_shape = (size - n,) + self.shape[1:]
            self.data = np.concatenate(
                [self.data, np.full(new_shape, dtype=self.data.dtype, fill_value=fill_value)],
                axis=0
            )
        self._max_length = len(data)

    def roll(self, shift: int):
        self.zero_index = (self.zero_index + shift) % len(self)

    def max(self):
        return np.max(self.data)

    def min(self):
        return np.min(self.data)
