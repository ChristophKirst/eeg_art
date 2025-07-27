"""
>>> from visualization.plotting import make_colormap, plot_colormap
>>> from visualization.colors import sort_colors_nn
>>> from visualization.warhol_colormap import create_warhol_colors
>>> colors = create_warhol_colors(n_colors=512)
>>> colors_sorted = sort_colors_nn(colors)
>>> plot_colormap(make_colormap('sorted', colors_sorted))


>>> from visualization.colors import sort_colors
>>> colors_sorted = sort_colors(colors)
>>> plot_colormap(make_colormap('sorted', colors_sorted))
"""
import numpy as np
import copy

import colorsys

from scipy.spatial import distance


def sort_colors_nn(colors, distance=distance.euclidean):
    n_colors = len(colors)

    A = np.zeros([n_colors, n_colors])
    for x in range(0, n_colors-1):
        for y in range(0, n_colors-1):
            A[x, y] = distance(colors[x], colors[y])

    # Nearest neighbour algorithm
    path, _ = NN(A, 0)

    colors_sorted = []
    for i in path:
        colors_sorted.append(colors[i])

    return colors_sorted


def NN(A, start):
    start = start-1
    n = len(A)
    path = [start]
    costList = []
    tmp = copy.deepcopy(start)
    B = copy.deepcopy(A)

    for h in range(n):
        B[h][start] = np.inf

    for i in range(n):
        for j in range(n):
            if B[tmp][j] == min(B[tmp]):
                costList.append(B[tmp][j])
                path.append(j)
                tmp = j
                break

        for k in range(n):
            B[k][tmp] = np.inf

    cost = sum([i for i in costList if i < np.inf]) + A[path[len(path)-2]][start]

    path.pop(n)
    path.insert(n, start)
    path = [i+1 for i in path]

    return path, cost


def hsv(rgb):
    return colorsys.rgb_to_hsv(*rgb)


def hls(rgb):
    return colorsys.rgb_to_hls(*rgb)


def luminosity(rgb):
    r, g, b = rgb
    return np.sqrt(.241 * r + .691 * g + .068 * b)


def hsv_r(rgb):
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return h, -s, v


def hls_r(rgb):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return h, -l, s


def luminosity_r(rgb):
    r, g, b = rgb
    return - np.sqrt(.241 * r + .691 * g + .068 * b)


def step(rgb, repetitions=1):
    r, g, b = rgb
    lum = np.sqrt(.241 * r + .691 * g + .068 * b)
    h, s, v = colorsys.rgb_to_hsv(r,g,b)
    h2 = int(h * repetitions)
    lum2 = int(lum * repetitions)
    v2 = int(v * repetitions)
    return h2, lum2, v2


def sort_colors(colors, key=luminosity):
    colors = list(colors)
    colors.sort(key=key)
    return colors
