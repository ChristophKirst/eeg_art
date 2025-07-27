# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

JXF binary file io

Based on file specifications:
https://sdk.cdn.cycling74.com/max-sdk-8.0.3/chapter_jit_jxf.html

groupID	JIT_BIN_CHUNK_CONTAINER ('FORM')
File size	32-bit int
IFF Type	JIT_BIN_FORMAT ('JIT!')
Format Chunk
chunkID	JIT_BIN_CHUNK_FORMAT_VERSION ('FVER')
Chunk size	12 bytes
Version	JIT_BIN_VERSION_1 (0x3C93DC80)
Matrix
chunk ID	JIT_BIN_CHUNK_MATRIX ('MTRX')
chunk size	32-bit int
offset	32-bit int
type	4-char
planecount	32-bit int
dimcount	32-bit int
dim	Array of 32-bit ints that contain the dimensions
data

Example
>>> from utils.jxf import read_jxf
>>> matrix = read_jxf('./data/matrix.jxf', verbose=True)

>>> import numpy as np
>>> from utils.jxf import write_jxf, read_jxf
>>> matrix = np.random.rand(*(5, 10, 20))
>>> write_jxf('./data/test.jxf', matrix, plane_count=True)
>>> matrix_read = read_jxf('./data/test.jxf', verbose=True)
>>> np.allclose(matrix_read, matrix_read)

>>> import numpy as np
>>> from utils.jxf import write_jxf
>>> matrix = np.arange(10, dtype='int32')
>>> write_jxf('./data/test_int.jxf', matrix, plane_count=False)

>>> import numpy as np
>>> from utils.jxf import write_jxf, read_jxf
>>> matrix = np.arange(10, dtype='float32')
>>> write_jxf('./data/test_float.jxf', matrix, plane_count=False)

>>> read_jxf('./data/test_float.jxf', verbose=True)
"""
import io
import struct
import numpy as np


JIT_BIN_CHUNK_CONTAINER = b'FORM'

JIT_BIN_FORMAT = b'JIT!'

JIT_BIN_CHUNK_FORMAT_VERSION = b'FVER'  # noqa

JIT_BIN_CHUNK_SIZE = 12

JIT_BIN_VERSION_1 = int(0x3C93DC80)

JIT_BIN_CHUNK_MATRIX = b'MTRX'

NUMPY_TO_JIT = {
    np.dtype('int16'): b'CHAR',
    np.dtype('int32'): b'LONG',
    np.dtype('float32'): b'FL32',
    np.dtype('float64'): b'FL64'
}

JIT_TO_NUMPY = {v: k for k, v in NUMPY_TO_JIT.items()}

INT = '>I'


def read_jxf(file: io.FileIO | str, verbose: bool = False):

    opened = False
    if isinstance(file, str):
        file = open(file, 'rb')
        opened = True

    # group id

    header = file.read(4)
    if header != JIT_BIN_CHUNK_CONTAINER:
        raise ValueError(f'{JIT_BIN_CHUNK_CONTAINER=} != {header}')
    if verbose:
        print(f"{header=}")

    file_size = struct.unpack(INT, file.read(4))[0]
    if verbose:
        print(f"{file_size=}")

    jit_format = file.read(4)
    if jit_format != JIT_BIN_FORMAT:
        raise ValueError(f'{JIT_BIN_FORMAT=} != {jit_format}')
    if verbose:
        print(f"{jit_format=}")

    jit_chunk_format = file.read(4)
    if jit_chunk_format != JIT_BIN_CHUNK_FORMAT_VERSION:
        raise ValueError(f'{JIT_BIN_CHUNK_FORMAT_VERSION=} != {jit_chunk_format}')
    if verbose:
        print(f"{jit_chunk_format=}")

    chunk_size = struct.unpack(INT, file.read(4))[0]
    if chunk_size != JIT_BIN_CHUNK_SIZE:
        raise ValueError(f'{JIT_BIN_CHUNK_SIZE=} != {chunk_size}')
    if verbose:
        print(f"{chunk_size=}")

    jit_version = file.read(4)
    if jit_version != JIT_BIN_VERSION_1.to_bytes(4, byteorder='big'):
        raise ValueError(f'{JIT_BIN_VERSION_1=} != {jit_version}')
    if verbose:
        print(f"{jit_version=}")

    jit_matrix = file.read(4)
    if jit_matrix != JIT_BIN_CHUNK_MATRIX:
        raise ValueError(f'{JIT_BIN_CHUNK_MATRIX=} != {jit_matrix}')
    if verbose:
        print(f"{jit_matrix=}")

    chunk_size = struct.unpack(INT, file.read(4))[0]
    if verbose:
        print(f"{chunk_size=}")
        # 24 + (4 * dim_count) + (item_size * plane_count * total_points))

    offset = struct.unpack(INT, file.read(4))[0]
    if verbose:
        print(f"{offset=}")
        # 24 + (4 * dim_count)

    jtype = file.read(4)  # noqa
    dtype = JIT_TO_NUMPY[jtype]
    if verbose:
        print(f"{jtype=}, {dtype=}")

    plane_count = struct.unpack(INT, file.read(4))[0]
    if verbose:
        print(f"{plane_count=}")

    dim_count = struct.unpack(INT, file.read(4))[0]
    if verbose:
        print(f"{dim_count=}")

    dim = tuple(struct.unpack(INT, file.read(4))[0] for _ in range(dim_count))
    if verbose:
        print(f"{dim=}")

    data_size = plane_count * np.prod(dim) * dtype.itemsize
    data = file.read(data_size)

    matrix = np.frombuffer(data, dtype=dtype).byteswap().reshape((plane_count,) + dim, order='F')  # noqa

    if opened:
        file.close()

    return matrix


def write_jxf(file: io.FileIO | str, data: np.ndarray, plane_count: bool = True):
    if data.dtype not in NUMPY_TO_JIT:
        raise ValueError

    if plane_count is True:
        plane_count = data.shape[0]
        dim_count = len(data.shape[1:])
        dim = data.shape[1:]
    else:
        plane_count = 1
        dim_count = len(data.shape)
        dim = data.shape
    item_size = data.itemsize

    opened = False
    if isinstance(file, str):
        file = open(file, 'wb')
        opened = True

    file.write(JIT_BIN_CHUNK_CONTAINER)

    file_size = 4 + (4 + 12 + 4) + (6 * 4 + 4 * dim_count) + item_size * data.size  # starts after file size
    file.write(struct.pack(INT, file_size))

    file.write(JIT_BIN_FORMAT)

    file.write(JIT_BIN_CHUNK_FORMAT_VERSION)

    file.write(struct.pack(INT, JIT_BIN_CHUNK_SIZE))

    file.write(struct.pack(INT, JIT_BIN_VERSION_1))

    file.write(JIT_BIN_CHUNK_MATRIX)

    chunk_size = 24 + (4 * dim_count) + (item_size * data.size)
    file.write(struct.pack(INT, chunk_size))

    offset = 24 + (4 * dim_count)
    file.write(struct.pack(INT, offset))

    jtype = NUMPY_TO_JIT[data.dtype]  # noqa
    file.write(jtype)

    file.write(struct.pack(INT, plane_count))

    file.write(struct.pack(INT, dim_count))

    for d in dim:
        file.write(struct.pack(INT, d))

    file.write(np.asarray(data, order='F').byteswap().tobytes())  # noqa

    if opened:
        file.close()
