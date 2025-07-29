# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst
"""
from abc import ABC, abstractmethod


class Stream(ABC):

    @property
    @abstractmethod
    def is_running(self) -> bool:
        ...

    @abstractmethod
    def open(self):
        ...

    @abstractmethod
    def start(self):
        ...

    @abstractmethod
    def stop(self):
        ...

    @abstractmethod
    def close(self):
        ...

    @abstractmethod
    def data_available(self):
        ...

    @abstractmethod
    def read(self):
        ...
