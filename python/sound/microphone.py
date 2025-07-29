# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Examples
>>> from sound.microphone import MicrophoneInput
>>> mic = MicrophoneInput(device=7)
>>> mic
MicrophoneInput(sample_rate=48000, block_size=None, channels=1, dtype='int16', device=7)

>>> mic.start()
>>> data = mic.read(frames=1024)
>>> mic.stop()
>>> data.shape
(1024, ...)

>>> mic.close()

>>> import time
>>> from sound.microphone import Microphone
>>> mic = Microphone(device=7, block_size=4800)
>>> mic
Microphone(sample_rate=48000, block_size=None, channels=1, dtype='int16', device=7, shape=(0,), dtype=int16, ...)

>>> mic.start()
>>> time.sleep(1)
>>> mic.stop()
>>> mic
Microphone(sample_rate=48000, block_size=4800, channels=1, dtype='int16', device=7, shape=(7,), ...)
"""
from typing import Callable

import logging
import sounddevice as sd
import numpy as np

from streaming.buffer import Buffer, RingBuffer


class MicrophoneInput:
    def __init__(
            self,
            sample_rate: int = 48000,
            block_size: int | None = None,
            channels: int = 1,
            dtype: str = 'int16',
            device: int | None = 4,
            # stream_callback: Callable | None = None
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels
        self.dtype = dtype
        self.device = device

        self.stream = None
        self.data = None
        self.initialize()

    def initialize(self, stream_callback: Callable | None = None):
        if self.stream is not None:
            self.stream.close()

        stream_callback = stream_callback if stream_callback is not None else\
            self.stream_callback if hasattr(self, 'stream_callback') \
            else None

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=self.channels,
                dtype=self.dtype,
                callback=stream_callback,
                device=self.device
            )
            logging.info(f"{self.__class__.__name__}: stream initialized.")
        except Exception as e:
            logging.warning(f"{self.__class__.__name__}: error opening audio stream: {e}")
            logging.warning(f"{self.__class__.__name__}: available audio devices:")
            logging.warning(f"{self.__class__.__name__}: {sd.query_devices()}")
            self.stream = None

    def start(self):
        self.stream.start()
        logging.info(f"{self.__class__.__name__}: stream started.")

    def is_running(self) -> bool:
        return self.stream is not None and self.stream.active

    def stop(self):
        if self.is_running():  # Ensure stream is active before stopping
            self.stream.stop()
            logging.info(f"{self.__class__.__name__}: stream stopped.")

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
            logging.info(f"{self.__class__.__name__}: stream closed.")

    def read(self, frames: int | None = None, return_overflow: bool = False):
        frames = frames if frames is not None else self.block_size if self.block_size is not None else 1024
        data, overflow = self.stream.read(frames)
        return (data, overflow) if return_overflow else data

    def __del__(self):
        self.stop()
        self.close()

    def __repr__(self):
        sample_rate = self.sample_rate
        block_size = self.block_size
        channels = self.channels
        dtype = self.dtype
        device = self.device
        return f"{self.__class__.__name__}({sample_rate=}, {block_size=}, {channels=}, {dtype=}, {device=})"


class Microphone(MicrophoneInput):
    def __init__(
            self,
            sample_rate: int = 48000,
            block_size: int | None = None,
            channels: int = 1,
            dtype: str = 'int16',
            device: int | None = 4,
            max_length: int | None = None,
            buffer: Buffer | None = None,
            buffer_type: type = RingBuffer
    ):
        MicrophoneInput.__init__(
            self,
            sample_rate=sample_rate,
            block_size=block_size,
            channels=channels,
            dtype=dtype,
            device=device
        )

        sample_shape = () if channels == 1 else (channels,)
        self.buffer = buffer if buffer is not None else \
            buffer_type(sample_shape=sample_shape, dtype=self.dtype, max_length=max_length)

    def stream_callback(self, indata, frames, time, status):
        if status:
            logging.info(f"{self.__class__.__name__}: sounddevice status: {status}")
        print('callback', indata.shape)
        self.buffer.update(indata[:, 0] if self.channels == 1 else indata)

    def __len__(self):
        return len(self.buffer)

    def update(self, value):
        self.buffer.update(value)

    def to_array(self) -> np.ndarray:
        return self.buffer.to_array()

    @property
    def array_shape(self):
        return self.buffer.array_shape

    def __getitem__(self, index):
        return self.buffer.__getitem__(index)

    def __setitem__(self, index, value):
        self.buffer.__setitem__(index, value)

    def __repr__(self):
        buffer_repr = self.buffer.__repr__()[len(self.buffer.__class__.__name__)+1:-1]
        microphone_repr = MicrophoneInput.__repr__(self)[:-1]
        return f"{microphone_repr}, {buffer_repr})"
