# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst
"""

from abc import ABC, abstract

class Board(ABC):

    @ab
    def n_channels(self):
        return 0

    @property
    def sample_rate(self):
        raise NotImplemented

    def start(self):
        raise NotImplemented

    def stop(self):
        raise NotImplemented

    def get_data(self, num_points):
        raise NotImplemented
