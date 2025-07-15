# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst
"""
import logging

from .board import Board
from .utils.utils import get_free_port

import brainflow
from brainflow.board_shim import BoardShim
from brainflow.board_shim import BrainFlowInputParams #, BoardIds, BrainFlowError
# from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations


logging.basicConfig(level=logging.DEBUG)


class OpenBCIBoard(Board):

    def __init__(self):
        super().__init__()
        BoardShim.enable_dev_board_logger()

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
        streamer_params = ''

from pythonosc.udp_client import SimpleUDPClient

ip = '127.0.0.1'
port = 8200
client = SimpleUDPClient(ip, port)

record = []

try:
    board_shim = BoardShim(board_id, params)
    board_shim.prepare_session()
    logging.info('Starting session')

    board_shim.start_stream(450000, streamer_params)
    logging.info('Starting session')

    # viewer = EEGViewer(board_shim)
    # viewer.show()
    # viewer.start()
    # app.exec()
    # logging.info('Viewer started')
    2 * 60 * 1000
    :
        data = board_shim.get_current_board_data(100)
        print(data)
        client.send_message('/pitch', data[0, 0])

    # num_points = 10
    # for i in range(20):
    #     data = board_shim.get_current_board_data(num_points)
    #     logging.info(f'{data=}')
    # logging.info('Releasing session')
    # board_shim.release_session()
except BaseException as e:
    logging.warning('Exception', exc_info=True)
finally:
    logging.info('End')
    if board_shim.is_prepared():
        logging.info('Releasing session')
        board_shim.release_session()


#%%


import numpy as np
path = '/Users/jennychai/Documents/EEG_Recordings/BrainFlow-RAW_EEG_Recordings_0.csv'
data = np.loadtxt(path, max_rows=53600)

np.save('/Users/jennychai/Documents/EEG_Recordings/recording.npy', data)




BoardShim.enable_dev_board_logger()
logging.basicConfig(level=logging.DEBUG)


#board_id = brainflow.BoardIds.CYTON_BOARD;
board_id = brainflow.BoardIds.CYTON_WIFI_BOARD.value
board_id = brainflow.BoardIds.CYTON_DAISY_WIFI_BOARD;

params = BrainFlowInputParams()
params.ip_port = free_port
params.serial_port = ''
params.mac_address = ''
params.other_info = ''
params.serial_number = ''
params.ip_address = '192.168.4.1'
params.ip_protocol = 0
params.timeout = 0
params.file = ''
streamer_params = ''
