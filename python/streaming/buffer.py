# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Examples
--------
>>> from streaming.buffer import ArrayBuffer
>>> a = ArrayBuffer(shape=(1,4), dtype=int)
>>> a
array([], shape=(0, 4), dtype=int64)

>>> a.capacity
1

>>> a.append([1,2,3,4])
>>> a
ArrayBuffer(shape=(1, 4), dtype=int64)

>>> a.append([5,6,7,8])
>>> a[1]  # noqa
array([5, 6, 7, 8])

>>> a[:]
array([[1, 2, 3, 4],
       [5, 6, 7, 8]])

>>> a.capacity
2

>>> a.append([5,6,7,8])
>>> a.capacity
4

>>> len(a)
3

>>> a.shape
(3, 4)

>>> import numpy as np
>>> a.extend(np.ones((50,4)))
>>> a.shape
(53, 4)

>>> a.capacity
64

>>> a.delete([3,4,5])
>>> a.shape
(50,4)

>>> from streaming.buffer import RingArrayBuffer
>>> r = RingArrayBuffer(sample_shape=(), max_length=5, dtype=int)
>>> r
RingArrayBuffer(shape=(5,), dtype=int)

>>> r.update([1, 1])
>>> r[:]
array([0, 0, 0, 1, 1])

>>> r.update((2,))
>>> r[:]
array([0, 0, 1, 1, 2])

>>> r.data
array([1, 1, 2, 0, 0])

>>> from streaming.buffer import RingArrayBuffer
>>> r = RingArrayBuffer(sample_shape=(3,), max_length=5, dtype=int)
>>> r
RingArrayBuffer(shape=(5, 3), dtype=int)

>>> r.update([[1, 2, 3]])
>>> r[:]
array([[0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [1, 2, 3]])

>>> r.update([(4, 5, 6), (7, 8, 9)])
>>> r[:]
array([[0, 0, 0],
       [0, 0, 0],
       [1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])

>>> r.update([(-1, -1, -1), (-2, -2, -2), (-3, -3, -3)])
>>> r[:]
array([[ 4,  5,  6],
       [ 7,  8,  9],
       [-1, -1, -1],
       [-2, -2, -2],
       [-3, -3, -3]])

>>> from streaming.buffer import RingBuffer
>>> r = RingBuffer(capacity=3)
>>> r
RingBuffer( length=0, capacity=3)

>>> r.update([1, 2])
>>> r.update([3, 4])
>>> r
RingBuffer( length=2, capacity=3)

>>> r.to_array()
array([1, 2, 3, 4])

>>> r.update([5, 5, 5, 5])
>>> r.update([10, 10])
[-      ]

>>> r.to_array()
array([ 3,  4,  5,  5,  5,  5, 10, 10])
"""
from abc import ABC, abstractmethod
from collections import deque

import numpy as np


class Buffer:
    def __init__(
            self,
            sample_shape: tuple[int, ...] = (),
            dtype: type = float,
            max_length: int | None = None,
    ):
        self._sample_shape = sample_shape
        self._dtype = dtype
        self._max_length = max_length

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def update(self, value):
        ...

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self),) + self.sample_shape

    @property
    def dtype(self) -> type:
        return self._dtype

    @property
    def max_length(self) -> int | None:
        return self._max_length

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return self._sample_shape

    def __repr__(self):
        dtype = self.dtype.__name__ if hasattr(self.dtype, '__name__') else self.dtype
        return f"{self.__class__.__name__}(shape={self.shape}, dtype={dtype}, max_length={self.max_length})"


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
        Buffer.__init__(self, max_length=max_length, sample_shape=sample_shape, dtype=dtype)
        data, capacity = self._initialize_buffer(data, capacity)
        self.data: np.ndarray = data
        self.capacity: int = capacity
        self.length: int = len(data) if data is not None else 0

    def _initialize_buffer(
            self,
            data: np.ndarray | None = None,
            capacity: int | None = None
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
        self.append(value)

    def to_numpy(self) -> np.ndarray:
        return self.data[:self.length]

    def _index(self, index: int | tuple | list | np.ndarray | slice):
        if not isinstance(index, tuple):
            index = (index,)

        index0 = index[0]
        if isinstance(index0, np.ndarray):
            if not index0.ndim == 1 or not np.all(np.logical_and(-self.length <= index0, index0 < self.length)):
                raise IndexError(f"index {index0} along axis 0 out of bounds")
        elif isinstance(index0, slice) and index0 == slice(None):
            index0 = slice(None, self.length, None)
        elif isinstance(index0, slice) or (isinstance(index0, int) and index0 < 0):
            index0 = range(self.length)[index0]
        if isinstance(index0, (int, tuple, list, np.ndarray)):
            check = np.array(index0, ndmin=1)
            if not np.all(np.logical_and(0 <= check, check < self.length)):
                raise IndexError(f"index {index0} along axis 0 out of bounds")

        return (index0,) + index[1:]

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

    def extend(self, value: np.ndarray):
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

    def delete(self, indices):
        n = len(indices)
        if n == 0:
            return
        self.data[:-n] = np.delete(self.data, indices, axis=0)
        self.length -= n

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
            [self.data, np.full(new_shape, dtype=self.data.dtype, fill_value=self.fill_value)], axis=0)
        self.capacity += add_capacity

    def roll(self, shift: int):
        self.data[:self.length] = np.roll(self.data[:self.length], shift, axis=0)

    def max(self):
        return np.max(self.data)

    def min(self):
        return np.min(self.data)


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
        else:
            if max_length is None or sample_shape is None:
                raise ValueError
            shape = (max_length,) + sample_shape
            data = np.zeros(shape, dtype=dtype)
        Buffer.__init__(self, sample_shape=sample_shape, dtype=dtype, max_length=max_length)
        self.data: np.ndarray = data
        self.zero_index: int = 0

    def __len__(self):
        return len(self.data)

    def update(self, value):
        self.append(value)

    def to_numpy(self) -> np.ndarray:
        return self[:]

    def _index(self, index):
        n = len(self)
        i = self.zero_index

        if not isinstance(index, tuple):
            index = (index,)

        index0 = index[0]
        if isinstance(index0, slice):
            index0 = np.arange(n)[index0]
        if not isinstance(index0, (int, tuple, list, np.ndarray)):
            raise KeyError
        index0 = (np.array(index0) + i) % n

        return (index0,) + index[1:]

    def __getitem__(self, index):
        return self.data[self._index(index)]

    def __setitem__(self, index, value):
        self.data[self._index(index)] = value

    def append(self, value, index: int = 0, roll: bool = True):
        n = len(self)
        l = len(value)
        if l > n:
            raise ValueError

        i = self.zero_index
        s = (index + i) % n
        e = s + l
        if e <= n:
            self.data[s:e] = value
        else:
            self.data[s:] = value[:n-s]
            self.data[:e-n] = value[n-s:]

        if roll:
            self.zero_index = e % n

    def delete(self, indices):
        indices = self._index(indices)
        self.data = np.delete(self.data, indices, axis=0)
        self.zero_index = min(self.zero_index, len(self) - 1)
        self._max_length = len(data)

    def resize(self, size: int, fill_value=0):
        n = len(self)
        if size < n:
            self.delete(slice(size, n))
        else:
            new_shape = (size - n,) + self.shape[1:]
            self.data = np.concatenate(
                [self.data, np.full(new_shape, dtype=self.data.dtype, fill_value=self.fill_value)],
                axis=0
            )
        self._max_length = len(data)

    def roll(self, shift: int):
        self.zero_index = (self.zero_index + shift) % len(self)

    def max(self):
        return np.max(self.data)

    def min(self):
        return np.min(self.data)


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
        Buffer.__init__(sample_shape=sample_shape, dtype=dtype, max_length=max_length)
        self.data: deque = data
        self.zero_index = 0

    def __len__(self):
        return sum([len(d) for d in self.data])

    def update(self, value):
        self.data.append(value)

    def to_numpy(self) -> np.ndarray:
        return np.concatenate(self.data, axis=0)

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def max(self):
        return max([max(d) for d in self.data])

    def min(self):
        return min([min(d) for d in self.data])



# class RingBuffer:
#     def __init__(self, capacity: int = 32):
#         self.data = np.zeros(capacity, dtype=object)
#         self.capacity = capacity
#         self.zero_index = 0
#         self.length = 0
#
#     def __len__(self):
#         return self.length
#
#     def update(self, value):
#         n = self.length
#         c = self.capacity
#         if n < c:
#             self.data[n] = value
#             self.length = n + 1
#         else:
#             i = self.zero_index
#             self.data[i] = value
#             self.zero_index = (i + 1) % c
#
#     def _index(self, index):
#         n = self.length
#         i = self.zero_index
#
#         if isinstance(index, slice):
#             index = np.arange(n)[index]
#         if not isinstance(index, (int, tuple, list, np.ndarray)):
#             raise KeyError
#         index = (np.array(index) + i) % n
#         return index
#
#     def __getitem__(self, item):
#         index = self._index(item)
#         return self.data[index]
#
#     def __setitem__(self, key, value):
#         index = self._index(key)
#         self.data[index] = value
#
#     def to_array(self):
#         return np.concatenate(self[:], axis=0)
#
#     def array_shape(self):
#         return self.to_array().shape
#
#     def array_length(self):
#         return len(self.to_array())
#
#     def max(self):
#         return np.max(self.to_array())
#
#     def min(self):
#         return np.min(self.to_array())
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}( length={len(self)}, capacity={self.capacity})"
