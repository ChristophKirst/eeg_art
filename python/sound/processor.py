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
    def register_callback(self, callback: callable):
        ...


class BufferedStream(Stream):
    buffer_type: type = Buf

    def __ini__(self, buffer: ):




class Processor(ABC):

    def __init__(self, stream: Stream | None = None):
        self.attach_to_stream()

    def attach_to_stream(self):
        self._stream = stream
        if self.stream is not None:
            self.stream.register_callback(self.process)

    @property
    def stream(self) -> Stream | None:
        return self._stream

    def has_stream(self) -> bool:
        return self.stream is not None

    @abstractmethod
    def process(self, *args, **kwargs):
        ...

    def __call__(self, *args, **kwargs):
         return self.process(*args, **kwargs)


