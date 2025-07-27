# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Example
>>> from streaming.updating import Updater
>>> class Test(Updater):
       def update(self):
           print('updating')
>>> updater = Test(update_interval=0.2, session_length=1.0)
>>> updater.start()
"""

from abc import ABC, abstractmethod

import time
import logging

from threading import Thread, Event

from .timer import Timer
from .sampling import SamplingCounter


class Updater(Thread, Timer):
    def __init__(
            self,
            update_interval: float = 0.2,  # sec
            session_length: float | None = None  # sec
    ):
        Thread.__init__(self)
        Timer.__init__(self)
        self._finished = Event()

    def cancel(self):
        """Stop the timer if it hasn't finished yet."""
        self._finished.set()

    def run(self):
        self._finished.wait(self.interval)
        if not self._finished.is_set():
            self.function(*self.args, **self.kwargs)
        self._finished.set()



class Updater(Timer, ABC):
    def __init__(
            self,
            update_interval: float = 0.2,  # sec
            session_length: float | None = None  # sec
    ):
        Timer.__init__(self)
        self._update_interval = update_interval
        self._session_length = session_length
        self._timer = None

    @property
    def update_interval(self):
        return self._update_interval

    @property
    def session_length(self):
        return self._session_length

    def _update(self):
        if self.is_running():
            self.update()  # Call the user-defined update method
            if self.session_length is None or (self.get_time_elapsed_since_start() < self.session_length):
                self._start_timer() # Reschedule the timer for the next interval

    def _

    def start(self, session_length: float | None = None):
        Timer.start(self)
        self._session_length = session_length if session_length is not None else self.session_length
        if not self.is_running():
            self._running = True
            self._timer = threading.Timer(self.update_interval, self._update)
            self._timer.start()
            logging.info(f"{self.__class__.__name__} started: updating every {self.update_interval} seconds.")

    def stop(self):
        Timer.stop(self)
        if self.is_running():
            self._running = False
            if self._timer:
                self._timer.cancel()
            logging.info(f"{self.__class__.__name__} stopped.")

    @abstractmethod
    def update(self):
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}, update_interval={self.update_interval}," \
               f" session_length={self.session_length})"

class SamplingUpdater(SamplingCounter, Updater, ABC):
    def __init__(
            self,
            sampling_rate: int = 1000,
            max_sample_size: int | None = None,
            update_interval: float = 0.2,  # sec
            session_length: float | None = None  # sec
    ):
        SamplingCounter.__init__(self, sampling_rate=sampling_rate, max_sample_size=max_sample_size)
        Updater.__init__(update_interval=update_interval, session_length=session_length)

    def start(self, session_length: float | None = None):
        SamplingCounter.start(self)
        Updater.start(self, session_length=session_length)

    def stop(self):
        SamplingCounter.stop(self)
        Updater.stop(self)

    def _update(self):
        SamplingCounter.set_sampling_time(self)
        Updater._update(self)

    def __repr__(self):
        return f"{SamplingCounter.__repr__(self)[:-1]}, {Updater.__repr__(self)[len(Updater.__class__.__name__) + 1:]}"
