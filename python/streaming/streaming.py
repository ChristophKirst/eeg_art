from abc import ABC, abstractmethod

from streaming.buffer import Buffer, RingBuffer


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
