# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst
"""
import logging

from typing import Self

from .board import Board
from streaming.utils import get_free_port

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BrainFlowError
from brainflow.exit_codes import BrainFlowExitCodes, BrainFlowError
from brainflow.board_shim import BrainFlowInputParams


class OpenBCIBoard(Board):
    def __init__(self):
        super().__init__()
        self.board_id = None
        self.board_shim: BoardShim | None = None
        self.initialize()

    def initialize(self):
        self.initialize_logging()
        logging.info('Initializing openbci board')

        self.board_id = brainflow.BoardIds.CYTON_DAISY_WIFI_BOARD

        params = BrainFlowInputParams()
        params.ip_port = get_free_port()
        params.serial_port = ''
        params.mac_address = ''
        params.other_info = ''
        params.serial_number = ''
        params.ip_address = '192.168.4.1'
        params.ip_protocol = 0
        params.timeout = 0
        params.file = ''
        self.board_shim = BoardShim(self.board_id, params)

        logging.info('Initializing openbci board done!')

    def initialize_logging(self):
        self.enable_dev_board_logger()
        logging.basicConfig(level=logging.DEBUG)

    def is_prepared(self) -> bool:
        return self.board_shim.is_prepared()

    def start(self, session_length: float = 60) -> Self:
        logging.info('Starting session')
        self.board_shim.prepare_session()

        streamer_params = ''
        n_points = int(session_length * self.sampling_rate)
        self.board_shim.start_stream(n_points, streamer_params)

        logging.info('Session started')
        return self

    def stop(self):
        logging.info('Session stopped')
        self.board_shim.release_session()

    @property
    def channels(self):
        return self.board_shim.get_exg_channels(self.board_id)

    @property
    def eeg_channels(self):
        return self.board_shim.get_eeg_channels(self.board_id)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def sampling_rate(self) -> int:
        return self.board_shim.get_sampling_rate(self.board_id)

    def get_data_count(self) -> int:
        return self.board_shim.get_board_data_count()

    def get_data(self, n_samples: int | None):
        try:
            data = self.board_shim.get_board_data(n_samples)
        except BrainFlowError as e:
            data = np.zeros((0, self.n_channels))
            logging.warning(f'Exception: {e}', exc_info=True)
        return data

    def get_current_data(self, n_samples: int | None):
        try:
            data = self.board_shim.get_current_board_data(n_samples)
        except BrainFlowError as e:
            data = np.zeros((0, self.n_channels))
            logging.warning(f'Exception: {e}', exc_info=True)
        return data

    def __repr__(self):
        return f"{Board.__repr__(self)[:-1]}, id={self.board_id}, is_prepared={self.is_prepared()})"
