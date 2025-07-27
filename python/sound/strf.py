# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Code based on: https://github.com/theunissenlab/soundsig

Examples
>>> import numpy as np
>>> from sound.strf import onset_strf, plot_strf
>>> t, f = np.linspace(0, 0.05, 30), np.linspace(300, 8000, 20)
>>> t_freq=20; t_c = 0.5 * (1.0 / t_freq) - 0.010
>>> t, f = np.meshgrid(t, f)
>>> strf = onset_strf(t, f, f_c=4000, t_c=t_c, t_freq=t_freq, t_sigma=0.005, t_phase=np.pi)
>>> plot_strf(strf, extent=[np.min(t), np.max(t), np.min(f), np.max(f)], figure=1)
"""
import numpy as np

import utils.initialize_pycharm_gui
import matplotlib.pyplot as plt


def onset_strf(
        t, f,
        t_c=0.150, t_freq=10.0, t_phase=0.0, t_sigma=0.250,
        f_c=3000.0, f_sigma=500.0
):
    f_part = np.exp(-(f - f_c)**2 / (2 * f_sigma**2))
    t_part = np.sin(2*np.pi * t_freq * (t - t_c) + t_phase)
    exp_part = np.exp((-(t - t_c)**2 / (2 * t_sigma**2)))

    strf = t_part * f_part * exp_part
    return strf


def checkerboard_strf(
        t, f,
        t_c=0.150, t_freq=10.0, t_phase=0.0, t_sigma=500.0,
        f_c=3000.0, f_freq=1e-6, f_phase=0.0, f_sigma=500.0,
        harmonic=False
):
    t_part = np.cos(2*np.pi * t_freq * t + t_phase)
    f_part = np.cos(2*np.pi * f_freq * f + f_phase)
    exp_part = np.exp((-(t-t_c)**2 / (2 * t_sigma**2)) - ((f - f_c)**2 / (2 * f_sigma**2)))

    if harmonic:
        f_part = np.abs(f_part)

    strf = t_part*f_part*exp_part
    return strf


def sweep_strf(
        t, f,
        t_c=0.0, f_c=5000.0,
        t_max=1.0, f_max=1.0,
        theta=0.0, aspect_ratio=1.0, phase=0.0, wavelength=0.5, spread=1.0
):
    t = (t - t_c) / t_max
    f = (f - f_c) / f_max

    tp = t * np.cos(theta) + f * np.sin(theta)
    fp = -t * np.sin(theta) + f * np.cos(theta)

    exp_part = np.exp(-(tp**2 + (aspect_ratio**2 * fp**2)) / (2 * spread**2))
    cos_part = np.cos((2*np.pi*tp / wavelength) + phase)

    return exp_part*cos_part


# def strf_kernel(ts: np.ndarray, tf: np.ndarray, f: np.ndarray, strf: callable, window: callable):
#
#     ts_, tf_ = np.meshgrid(ts, tf)
#     qs = np.unique(ts_ + tf_)
#
#     k = np.zeros(len(qs))
#     for i, q in enumerate(qs):
#         k[i] = 0
#         for ts_ in ts:
#             for tf_ in tf:
#                 for f_ in f:
#                     k[i] +=
#
#
#
#     nf = len(f)
#     nt = len(t)
#     nw = len(fourier_window)
#     q = np.




def plot_strf(strf, figure: int | None = None, title: str | None = None, extent=None):
    abs_max = np.abs(strf).max()
    plt.figure(figure)
    plt.clf()
    plt.imshow(
        strf,
        interpolation='nearest',
        aspect='auto',
        origin='lower',
        vmin=-abs_max,
        vmax=abs_max,
        cmap=plt.cm.seismic,
        extent=extent
    )
    if title is not None:
        plt.title(title)
