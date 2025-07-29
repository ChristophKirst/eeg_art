# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Driving a network by audio stimuli.
"""

from simulation.spiking_network import SpikingNetwork, NetworkTopology, NetworkPlotter
from sound.microphone import Microphone


mic = Microphone(device=7)
