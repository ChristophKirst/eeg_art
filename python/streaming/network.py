# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Example
>>> import logging
>>> logging.basicConfig(level=logging.DEBUG)
>>> from boards import RandomBoard
>>> from network.streamer import UDPBoardStreamer
>>> board = RandomBoard().start()
>>> streamer = UDPBoardStreamer(board=board, ip='127.0.0.1', port=8200, interval=0.1)
>>> streamer
UDPBoardStreamer(ip=127.0.0.1, port=8200, interval=0.1, board=RandomBoard(n_channels=16, sample_rate=1000))

>>> streamer.start()
...

>>> streamer.stop()
...
"""
from abc import ABC, abstractmethod

import logging
import threading
import socket
import time

try:
    from pythonosc import udp_client, osc_message_builder
    HAS_OSC = True
except ModuleNotFoundError:
    HAS_OSC = False

from boards import Board

import socketserver


def get_free_port():
    with socketserver.TCPServer(("localhost", 0), None) as s:  # noqa
        free_port = s.server_address[1]
    return free_port


# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Example
>>> import time
>>> from boards.board_buffer import SampleCounter, Sampler
>>> counter = SampleCounter()
>>> counter
SampleCounter(sampling_rate=1000)

>>> counter.start()
>>> time.sleep(1.5)
>>> counter.get_n_samples_since_start()
1500

>>> counter.update()
>>> time.sleep(0.5)
>>> counter.get_n_samples_since_update()
500

>>> sampler = Sampler(update_interval=0.5, sampling_rate=1000)
>>> sampler
Sampler(sampling_rate=1000, update_interval=500)

>>> sampler.start()
>>> time.sleep(1.5)
>>> sampler.get_n_samples_since_update()
"""
from abc import ABC, abstractmethod

import time
import threading
import logging

import numpy as np

from .board import Board
from utils.buffered_array import BufferedArray




class Streamer(ABC):
    def __init__(
            self,
            ip: str = '127.0.0.1',
            port: int = 8200,
            interval: float = 0.1,  # seconds
    ):

        self.ip = ip
        self.port = port
        self.interval = interval

        self.stop_event = threading.Event()  # Event to signal the thread to stop
        self.thread = None

    @abstractmethod
    def get_data(self):
        ...

    @abstractmethod
    def _send_data_loop(self):
        ...

    def is_stopped(self) -> bool:
        return self.stop_event.is_set()

    def start(self):
        if self.thread and self.thread.is_alive():
            logging.warning(f"{self.__class__.__name__} is already running.")
            return

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._send_data_loop, daemon=True)
        self.thread.start()
        logging.info(f"{self.__class__.__name__} started.")

    def stop(self):
        if not self.thread or not self.thread.is_alive():
            logging.warning(f"{self.__class__.__name__} is not running.")
            return

        self.stop_event.set()
        self.thread.join()
        logging.info(f"{self.__class__.__name__} stopped.")

    def __repr__(self):
        return f"{self.__class__.__name__}(ip={self.ip}, port={self.port}, interval={self.interval})"









class UDPStreamer(Streamer, ABC):
    def __init__(
            self,
            ip: str = '127.0.0.1',
            port: int = 8200,
            interval: float = 0.1,
    ):
        Streamer.__init__(self, ip=ip, port=port, interval=interval)

    def _send_data_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Create a UDP socket

        while not self.is_stopped():  # Keep running until stop_event is set
            try:
                sock.sendto(self.get_data(), (self.ip, self.port))  # Send the data
                logging.info(f"{self.__class__.__name__}: send data to {self.ip}:{self.port}")
            except Exception as e:
                logging.warning(f"{self.__class__.__name__}: Error sending UDP data: {e}")

            self.stop_event.wait(self.interval)  # Wait for the specified interval or until stop_event is set

        sock.close()  # Close the socket when the loop finishes
        logging.info(f"{self.__class__.__name__}: Sending loop stopped.")


class OSCStreamer(Streamer, ABC):
    def __init__(
            self,
            ip: str = '127.0.0.1',
            port: int = 8200,
            interval: float = 0.1,
            message: str = '/max'
    ):
        Streamer.__init__(self, ip=ip, port=port, interval=interval)
        self.message = message

    def _send_data_loop(self):
        while not self.is_stopped():
            try:
                self.client.send_mesaage(self.message, data)
                logging.info(f"{self.__class__.__name__}: sent {data.shape} to {self.ip}:{self.port}:{self.message}")
            except Exception as e:
                logging.warning(f"{self.__class__.__name__}: error sending message: {e}")

            self.stop_event.wait(self.interval_seconds)

        logging.info(f"{self.__class__.__name__}: sending loop stopped.")

    def __repr__(self):
        return f"{Streamer.__repr__(self)[:-1]}, message={self.message})"


class BoardStreamerMixin:
    def __init__(
            self,
            board: Board
    ):
        self.board = board
        self.n_samples = int(self.interval * self.board._sampling_rate)

    def get_data(self):
        return self.board.get_data(self.n_samples)

    def __repr__(self):
        return f"{super().__repr__()[:-1]}, board={self.board})"


class UDPBoardStreamer(BoardStreamerMixin, UDPStreamer):
    def __init__(
            self,
            board: Board,
            ip: str = '127.0.0.1',
            port: int = 8200,
            interval: float = 0.1,
    ):
        UDPStreamer.__init__(self, ip=ip, port=port, interval=interval)
        BoardStreamerMixin.__init__(self, board=board)


class OSCBoardStreamer(BoardStreamerMixin, OSCStreamer):
    def __init__(
            self,
            board: Board,
            ip: str = '127.0.0.1',
            port: int = 8200,
            interval: float = 0.1,
            message: str = '/max'
    ):
        OSCStreamer.__init__(self, ip=ip, port=port, interval=interval, message=message)
        BoardStreamerMixin.__init__(self, board=board)



#
# class SampleCounter:
#     def __init__(
#             self,
#             sampling_rate: int = 1000,
#     ):
#         self.sampling_rate = sampling_rate
#         self.last_update_time = None
#         self.start_time = None
#
#     def get_current_time(self, current_time: float | None = None) -> float:  # noqa
#         return current_time if current_time is not None else time.perf_counter()
#
#     def get_start_time(self):
#         return self.start_time
#
#     def set_start_time(self, current_time: float | None = None):
#         self.start_time = self.get_current_time(current_time)
#
#     def get_time_elapsed_since_start(self, current_time: float | None = None) -> float:
#         return self.get_current_time(current_time) - self.get_start_time()
#
#     def get_update_time(self):
#         return self.last_update_time
#
#     def set_update_time(self, current_time: float | None = None):
#         self.last_update_time = self.get_current_time(current_time)
#
#     def get_time_elapsed_since_update(self, current_time: float | None = None) -> float:
#         return self.get_current_time(current_time) - self.get_update_time()
#
#     def get_n_samples_since_start(self, current_time: float | None = None) -> int:
#         return int(self.sampling_rate * self.get_time_elapsed_since_start(current_time))
#
#     def get_n_samples_since_update(self, current_time: float | None = None) -> int:
#         return int(self.sampling_rate * self.get_time_elapsed_since_update(current_time))
#
#     def start(self):
#         self.set_start_time()
#
#     def stop(self):
#         self.timer.stop()
#
#     def update(self):
#         self.set_update_time()
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}(sampling_rate={self.sampling_rate})"
#
#
# class Sampler(SampleCounter):
#     def __init__(
#             self,
#             sampling_rate: int = 1000,
#             update_interval: float = 0.2,  # sec
#     ):
#         SampleCounter.__init__(self, sampling_rate=sampling_rate)
#         self.update_interval = int(update_interval * 1000)  # msec
#
#         self.timer = QtCore.QTimer(None)
#         self.timer.timeout.connect(self.update)
#
#     def start(self):
#         self.set_start_time()
#         self.timer.start(self.update_interval)
#
#     def stop(self):
#         self.timer.stop()
#
#
# class Sampler(SampleCounter):
#     def __init__(
#             self,
#             sampling_rate: int = 1000,
#             update_interval: float = 0.2,  # sec
#     ):
#         SampleCounter.__init__(self, sampling_rate=sampling_rate)
#         self.update_interval = int(update_interval * 1000)  # msec
#
#         self._timer = None
#         self._running = False
#
#     def _start_timer(self):
#         self._timer = threading.Timer(self.interval, self._update_wrapper)
#         self._timer.start()
#
#     def _update_wrapper(self):
#         if self._running:
#             self.update()  # Call the user-defined update method
#             self._start_timer() # Reschedule the timer for the next interval
#
#     def start(self):
#         if not self._running:
#             self._running = True
#             self._start_timer()
#             logging.info(f"{self.__class__.__name__} periodic updating started: {self.update_interval} seconds.")
#
#     def stop(self):
#         if self._running:
#             self._running = False
#             if self._timer:
#                 self._timer.cancel()  # Stop the currently running timer
#             logging.info(f"{self.__class__.__name__} periodic updating stopped.")
#
#     def update(self):
#         SampleCounter.update(self)
#
#     def __repr__(self):
#         return f"{SampleCounter.__repr__(self)[:-1]}, update_interval={self.update_interval})"
#
#
# class BoardBuffer(Sampler):
#     def __init__(
#             self,
#             board: Board,
#             update_interval: float = 0.2,  # sec
#             buffer_interval: float | None = None,  # sec
#             channels: list[int] | None = None
#     ):
#         Sampler.__init__(self, sampling_rate=board.sampling_rate, update_interval=update_interval)
#         self.board: Board = board
#         self.channels = channels if channels is not None else board.channels
#         self.buffer_interval = buffer_interval
#         self.buffer_size = None if self.buffer_interval is None else int(self.buffer_interval * board.sampling_rate)
#         self.buffer = BufferedArray(shape=(self.buffer_size if self.buffer_size is not None else 64, self.channels))
#
#     @property
#     def sampling_rate(self):
#         return self.board.sampling_rate
#
#     def update(self):
#         n_samples = self.get_n_samples_since_update()
#         data = self.board.get_current_data(n_samples)
#         n_new_points = data.shape[0]
#         if n_new_points == 0:
#             return
#         if self.buffer_size is None:
#             self.buffer.append(data)
#         else:
#             if n_new_points < self.buffer_size:
#                 self.buffer.roll(-n_new_points)
#                 self.buffer[-n_new_points:] = data
#             else:
#                 self.buffer[:] = data[self.channels]
#
#     def __repr__(self):
#         return f"{SampleCounter.__repr__(self)[:-1]}, channels={self.channels})"
#
#
