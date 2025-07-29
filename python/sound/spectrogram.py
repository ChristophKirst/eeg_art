# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Examples
--------
>>> import numpy as np
>>> from sound.spectrogram import Spectrogram
>>> s = Spectrogram(size=1024 * 16, chunk_size=1024 * 8, hop=1024//4, sample_rate=10)
>>> s
Spectrogram(size=10240, sample_rate=1, chunk_size=4096, hop=256, max_frequency=5.0)

>>> s.max_frequency
5.0

>>> s.update(np.sin(2 * np.pi * 0.1 * np.linspace(0, s.max_time, s.size + 1)[:-1]))
>>> s.shape
(40, 513)

>>> s.dt
25.6

>>> from visualization.plotting import pg
>>> img = pg.image(s.to_array())
"""
from typing import Literal

import numpy as np
from scipy.signal import spectrogram, ShortTimeFFT
from scipy.signal.windows import gaussian

from streaming.buffer import ArrayBuffer


class Spectrogram:
    def __init__(
            self,
            size: int,
            chunk_size: int = 8 * 1024,
            sample_rate: int = 48000,
            hop: int | None = None,
            win: int | np.ndarray | None = None,
            scale_to: Literal['magnitude', 'psd'] | None = 'psd',
            max_frequency: int | float | None = None
    ):
        if size < chunk_size:
            raise ValueError

        self.size = size
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.hop = hop if hop is not None else chunk_size // 8
        self.win = win if win is not None else np.full(chunk_size, fill_value=1.0/chunk_size)
        self.max_frequency = max_frequency

        if self.size % self.hop != 0:
            raise ValueError

        self.stft = ShortTimeFFT(win=self.win, hop=self.hop, fs=self.sample_rate, scale_to=scale_to)

        self.dt = self.hop / self.sample_rate  # the spectrogram sample rate
        self.df = self.stft.f[1]

        self.frequencies = self.stft.f
        self.n_frequencies = len(self.frequencies)
        if self.max_frequency is not None:
            self.frequencies = self.frequencies[self.frequencies <= self.max_frequency]
            self.n_frequencies = len(self.frequencies)
        else:
            self.max_frequency = np.max(self.frequencies)

        self.n_times = self.size // self.hop
        self.times = np.linspace(0, self.n_times * self.dt, self.n_times + 1)[:-1]
        self.max_time = self.times[-1]

        self.overlap = self.stft.lower_border_end[1] - self.stft.p_min

        self.wave_buffer = ArrayBuffer(sample_shape=(), dtype=float, max_length=size)
        self.buffer = ArrayBuffer(sample_shape=(self.n_frequencies,), dtype=float, max_length=size)

    def __len__(self):
        return len(self.buffer)

    @property
    def shape(self):
        return self.buffer.shape

    def to_array(self):
        return self.buffer.to_array()

    def update(self, wave):
        w = len(wave)
        b = len(self.wave_buffer)
        n = w + b
        k = self.stft.nearest_k_p(n)
        if k == 0:
            self.wave_buffer.update(wave)
            return

        values = self.wave_buffer.read(length=k, reduce_length=True)

        d = k - b
        if d > 0:
            values = np.concatenate([values, wave[:d]], axis=0)
            self.wave_buffer.update(wave[d:])

        Sxx = np.abs(self.stft.stft(values))[:self.n_frequencies].T  # (T, F)
        # Sxx = 10 * np.log10(Sxx + 1e-10)

        overlap = self.overlap
        if len(self.buffer) > overlap:
            self.buffer[-overlap:] += Sxx[:overlap]
        self.buffer.update(Sxx[overlap:])

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, sample_rate={self.sample_rate}, " \
               f"chunk_size={self.chunk_size}, hop={self.hop}, max_frequency={self.max_frequency})"
