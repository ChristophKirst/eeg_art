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
        self.n_samples = int(self.interval * self.board.sampling_rate)

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
