# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Examples
--------
>>> from streaming.buffer.array_buffer import ArrayBuffer
>>> a = ArrayBuffer(sample_shape=(4,), dtype=int)
>>> a
ArrayBuffer(shape=(0, 4), dtype=int, max_length=None)

>>> a.capacity
16

>>> a.append([1,2,3,4])
>>> a
ArrayBuffer(shape=(1, 4), dtype=int, max_length=None)

>>> a.append([5,6,7,8])
>>> a[1]  # noqa
array([5, 6, 7, 8])

>>> a[:]
array([[1, 2, 3, 4],
       [5, 6, 7, 8]])

>>> a.capacity
16

>>> a.extend([[5,6,7,8]] * 30)  # noqa
>>> a.capacity
32

>>> len(a)
32

>>> a.shape
(32, 4)

>>> a.delete([3,4,5])
>>> a.shape
(29, 4)
"""
from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from .buffer import Buffer


def _index_for_length(index, length: int, replace_slice: bool = False):
    if not isinstance(index, tuple):
        index = (index,)

    index0 = index[0]
    if isinstance(index0, range):
        index0 = np.array(index0)

    if isinstance(index0, slice):
        if replace_slice:
            index0 = np.arange(length)[index0]
        else:
            index0 = slice(*index0.indices(length))

    if isinstance(index0, (int, range, tuple, list, np.ndarray)):
        index0 = np.array(index0)
        if not np.all(np.logical_and(-length <= index0, index0 < length)):
            raise IndexError(f"index {index0} along axis 0 out of bounds")
        negative = index0 < 0
        index0[negative] = length + index0[negative]

    if not isinstance(index0, (slice, np.ndarray)):
        raise IndexError(f"index {index0} not supported")

    return (index0,) + index[1:]


class ArrayBuffer(Buffer):
    """Simple numpy array buffer."""

    default_capacity: int = 16

    def __init__(
            self,
            sample_shape: tuple[int, ...] | None = None,
            dtype: type | str = float,
            max_length: int | None = None,
            capacity: int | None = None,
            data: np.ndarray | None = None,
    ):
        if data is not None:
            sample_shape = data.shape[1:]
            dtype = data.dtype
            length = len(data)
        else:
            length = 0
        Buffer.__init__(self, max_length=max_length, sample_shape=sample_shape, dtype=dtype)
        data, capacity = self._initialize_buffer(data, capacity, dtype)
        self.data: np.ndarray = data
        self.capacity: int = capacity
        self.length: int = length

    def _initialize_buffer(
            self,
            data: np.ndarray | None = None,
            capacity: int | None = None,
            dtype: type | str = float
    ):
        capacity = capacity if capacity is not None else self.default_capacity
        if data is None:
            capacity = capacity if capacity is not None else self.default_capacity
            shape = (capacity,) + self.sample_shape
            data = np.empty(shape, dtype=dtype)
        else:
            capacity = max(len(data), capacity)
        return data, capacity

    def __len__(self):
        return self.length

    def update(self, value):
        self.extend(value)

    def to_array(self) -> np.ndarray:
        return self.data[:self.length]

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

    def _index(self, index: int | range | slice | tuple | list | np.ndarray):
        return _index_for_length(index, length=self.length)

    def __getitem__(self, index):
        return self.data[self._index(index)]

    def __setitem__(self, index, value):
        self.data[self._index(index)] = value

    def append(self, value):
        if self.length == self.capacity:
            length = 2 * self.capacity
            if self.max_length is not None:
                length = min(length, self.max_length)
                if self.capacity == length:
                    raise ValueError
            self.resize(length)
        self.data[self.length] = value
        self.length += 1

    def extend(self, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        if self.shape[1:] != value.shape[1:]:
            raise ValueError(f"cannot concatenate arrays of shape {self.shape} and {value.shape} along axis=0")
        size = self.length
        capacity = self.capacity
        new_size = size + len(value)
        if new_size > capacity:
            while new_size > capacity:
                capacity *= 2
            self.resize(capacity)
        self.data[size: new_size] = value
        self.length = new_size

    def delete(self, index):
        index = self._index(index)
        if len(index) > 1:
            raise ValueError
        index = index[0]
        if isinstance(index, slice):
            index = range(self.length)[index]
        n = len(index)
        if n == 0:
            return
        self.data[:-n] = np.delete(self.data, index, axis=0)
        self.length -= n

    def clear(self):
        self.length = 0

    def resize(self, capacity: int):
        if self.max_length is not None:
            if capacity > self.max_length:
                raise ValueError
        add_capacity = capacity - self.capacity
        if add_capacity < 0:
            raise RuntimeError("can only increase capacity.")
        shape = self.shape
        new_shape = (add_capacity,) + shape[1:]
        self.data = np.concatenate(
            [self.data, np.empty(new_shape, dtype=self.data.dtype)], axis=0)
        self.capacity += add_capacity

    def roll(self, shift: int):
        self.data[:self.length] = np.roll(self.data[:self.length], shift, axis=0)

    def max(self):
        return np.max(self.data)

    def min(self):
        return np.min(self.data)
