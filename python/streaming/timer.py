# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Example
>>> from streaming.timer import Timer
>>> timer = Timer()
>>> timer
Timer()

>>> timer.start()
>>> timer.get_elapsed_time()
...

>>> timer.is_running()
True

>>> timer.stop()
>>> timer.is_running()
False

>>> timer.get_elapsed_time()
"""
import time
import logging


class Timer:
    def __init__(self):
        self._start_time = None
        self._stop_time = None
        self._running = False

    def get_current_time(self, current_time: float | None = None) -> float:  # noqa
        return current_time if current_time is not None else time.perf_counter()

    def get_start_time(self):
        return self._start_time

    def get_stop_time(self):
        return self._stop_time

    def set_start_time(self, current_time: float | None = None):
        self._start_time = self.get_current_time(current_time)

    def set_stop_time(self, current_time: float | None = None):
        self._stop_time = self.get_current_time(current_time)

    def get_elapsed_time(self, current_time: float | None = None) -> float:
        if not self.is_running():
            start_time = self.get_start_time()
            start_time = start_time if start_time is not None else 0
            stop_time = self.get_stop_time()
            stop_time = stop_time if stop_time is not None else 0
        else:
            start_time = self.get_start_time()
            stop_time = self.get_current_time(current_time)

        return stop_time - start_time

    def start(self):
        logging.info(f"{self.__class__.__name__}: started.")
        self._running = True
        self.set_start_time()
        self._stop_time = None

    def stop(self):
        logging.info(f"{self.__class__.__name__}: stopped.")
        self._running = False
        self._start_time = None
        self.set_stop_time()

    def is_running(self):
        return self._running

    def __repr__(self):
        return f"{self.__class__.__name__}()"



