# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Examples
>>> import time
>>> from sound.microphone import Microphone
>>> mic = Microphone()
>>> mic.start()
>>> time.sleep(1)
>>> mic.stop()
>>> mic.shape
"""
import logging
import sounddevice as sd
import numpy as np

from streaming.streaming import Stream, BufferedStream
from streaming.buffer import Buffer, ArrayBuffer


class MicrophoneStream(Stream):

    def __init__(
            self,
            sample_rate: int = 48000,
            block_size: int | None = None,
            channels: int = 1,
            dtype: str = 'int16',
            device: int | None = 4
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels
        self.dtype = dtype
        self.device = device

        self.stream = None
        self.data = None
        self.open()

    def open(self):
        if self.stream is not None:
            self.stream.close()

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=self.channels,
                dtype=self.dtype,
                callback=self.stream_callback if hasattr(self, 'stream_callback') else None,
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

    def stream_callback(self, indata, frames, time, status):
        if status:
            logging.info(f"{self.__class__.__name__}: sounddevice status: {status}")
        self.data = indata

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


class Microphone(MicrophoneStream, Buffer):
    def __init__(
            self,
            sample_rate: int = 48000,
            block_size: int | None = None,
            channels: int = 1,
            dtype: str = 'int16',
            device: int | None = 4,
            max_length: int | None = None,
            buffer: Buffer | None = None,
            buffer_type: type = ArrayBuffer
    ):
        MicrophoneStream.__init__(
            self,
            sample_rate=sample_rate,
            block_size=block_size,
            channels=channels,
            dtype=dtype,
            device=device
        )
        sample_shape = () if channels == 1 else (channels,)
        Buffer.__init__(
            self,
            sample_shape=sample_shape,
            dtype=dtype,
            max_length=max_length,
        )

    def __len__(self):
        return len(self.buffer)

    def update(self, value):
        self.buffer.update(value)

    def to_numpy(self) -> np.ndarray:
        return self.buffer.to_numpy()

    def __getitem__(self, index):
        return self.buffer.__getitem__(index)

    def __setitem__(self, index, value):
        self.buffer.__setitem__(index, value)

    def callback(self, indata, frames, time, status):
        if status:
            logging.info(f"{self.__class__.__name__}: sounddevice status: {status}")

        self.buffer.update(indata[:, 0] if self.channels == 1 else indata)

    def __repr__(self):
        buffer_repr = Buffer.__repr__(self)[len(Buffer.__class__.__name__):-1]
        microphone_repr = MicrophoneStream.__repr__(self)[len(MicrophoneStream.__class__.__name__):-1]
        return f"{self.__class__.__name__}{microphone_repr}{buffer_repr}"
