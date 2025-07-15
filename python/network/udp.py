# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst
"""

from pythonosc.udp_client import SimpleUDPClient

ip = '127.0.0.1'
port = 8200
client = SimpleUDPClient(ip, port)

record = []