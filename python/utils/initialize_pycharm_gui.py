# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst
"""


def initialize_pycharm_gui():
    from IPython import get_ipython  # noqa
    get_ipython().run_line_magic('gui', 'qt')


initialize_pycharm_gui()