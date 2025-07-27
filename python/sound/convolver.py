# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Examples
>>> import numpy as np
>>> from sound.convolver import Convolver
>>> kernel = [0.2, 0.2, 0.2, 0.2, 0.2]  # Length 5
>>> block_size = 10  # Process data in blocks of 10 samples
>>> convolver = Convolver(kernel, block_size)
>>> input_stream = np.random.rand(2 * block_size)  # 100 random samples
>>> output_stream = []
>>> for i in range(0, len(input_stream), block_size):
        current_block = input_stream[i: i + block_size]
        output_block = convolver.process_block(current_block)
        output_stream.extend(output_block)

>>> output_stream = np.array(output_stream)
>>> print("Input Stream:")
>>> print(input_stream)
>>> print("Convolved Output Stream (Online):")
>>> print(output_stream)

>>> from scipy.signal import fftconvolve
>>> full_convolution_output = fftconvolve(input_stream, kernel, mode='full')
>>> print(full_convolution_output.shape, output_stream.shape)

>>> print("Convolved Output Stream (Full Convolution for Comparison):")
>>> print(full_convolution_output)
>>> print(np.allclose(full_convolution_output[:-convolver.overlap], output_stream))
"""

import numpy as np


class Convolver:
    def __init__(self, kernel, block_size):
        self.kernel = np.array(kernel)
        self.block_size = block_size
        self.kernel_len = len(self.kernel)
        self.overlap = self.kernel_len - 1

        self.n_fft = self.block_size + self.overlap
        self.kernel_fft = np.fft.rfft(self.kernel, self.n_fft)
        self.overlap_buffer = np.zeros(self.overlap)

    def process_block(self, input_block):
        n = len(input_block)
        if n != self.block_size:
            raise ValueError

        input_fft = np.fft.rfft(input_block, self.n_fft)
        result_fft = input_fft * self.kernel_fft

        convolution = np.fft.irfft(result_fft)

        out = convolution[:self.block_size]
        out[:self.overlap] += self.overlap_buffer
        self.overlap_buffer = convolution[self.block_size:]

        return out
