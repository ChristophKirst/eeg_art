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
            dtype: str | type = float,
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
    def to_array(self) -> np.ndarray:
        ...

    @abstractmethod
    def read(self, length: int | None = None, delete_data: bool = True, reduce_length: bool = False):
        ...

    @abstractmethod
    def clear(self):
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self),) + self.sample_shape

    @property
    def dtype(self) -> str | type:
        return self._dtype

    @property
    def max_length(self) -> int | None:
        return self._max_length

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return self._sample_shape

    @property
    def array_len(self) -> int:
        return len(self)

    @property
    def array_shape(self) -> tuple[int, ...]:
        return self.shape

    def array_available(self, length: int | None) -> bool:
        return self.array_len > (length - 1 if length is not None else 0)

    def __repr__(self):
        dtype = self.dtype.__name__ if hasattr(self.dtype, '__name__') else self.dtype
        return f"{self.__class__.__name__}(shape={self.shape}, dtype={dtype}, max_length={self.max_length})"
