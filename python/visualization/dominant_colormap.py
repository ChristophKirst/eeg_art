# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Warhol like color maps

Examples
>>> from visualization.dominant_colormap import get_colorma, pplot_colormap


>>> import numpy as np
>>> from visualization.warhol_colormap import create_warhol_colors
>>> colors = create_warhol_colors(n_colors=256)
>>> np.save('./data/images/warhol_colors.npy', colors)

>>> from visualization.warhol_colormap import get_warhol_colormap, plot_colormap
>>> cmap = get_warhol_colormap(n_colors=32)
>>> plot_colormap(cmap)
"""
import numpy as np

from sklearn.cluster import k_means

from visualization.plotting import pvq, plot_colormap, make_colormap
from visualization.colors import sort_colors
import matplotlib.colors as mcolors  # noqa

from torchvision import transforms
from PIL import Image


def create_dominant_colors(image_paths: list, n_colors: int = 256, sort: callable = sort_colors, verbose: bool = False):
    images = [Image.open(path).convert('RGB') for path in image_paths]

    preprocess = transforms.Resize((256, 256))

    colors = np.zeros((0, 3), dtype=int)
    for image in images:
        image = preprocess(image)
        data = np.array(image).reshape((-1, 3))  # noqa

        dtype = data.dtype.descr * 3
        struct = data.view(dtype)

        colors_, color_counts_ = np.unique(struct, return_counts=True)
        colors_ = colors_.view(data.dtype).reshape(-1, 3)
        colors = np.concatenate([colors, colors_], axis=0)

    if verbose:
        pl = pvq.BackgroundPlotter()
        pl.add_mesh(np.asarray(colors, dtype=float), scalars=colors, rgb=True)  # noqa

    color_centers, _, _ = k_means(colors, n_clusters=n_colors, random_state=0)

    if verbose:
        pl.add_mesh(color_centers, color='black', point_size=20, render_points_as_spheres=True)  # noqa

    if sort is not None:
        color_centers = np.array(sort(color_centers))

    return color_centers / 255


def get_dominant_colormap(image_paths: list, n_colors: int = 256, sort: callable = sort_colors, name: str = 'dcmap', verbose: bool = False):
    colors = create_dominant_colors(image_paths=image_paths, n_colors=n_colors, verbose=verbose)

    if sort is not None:
        colors = sort(colors)

    cmap = make_colormap(name, colors)

    if verbose:
        plot_colormap(cmap)

    return cmap
