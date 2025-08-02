# -*- coding: utf-8 -*-
"""
EEG Art Project

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Examples
>>> import simulation.spiking_network as net
>>> topology = net.NetworkTopology(topology_type='spatial', weights=30)
>>> snn = net.SpikingNetwork(topology=topology, par_i=7.5, par_d=(3, 3))

>>> viewer = snn.simulate(5000, verbose=True)
>>> snn.simulate(500, verbose=viewer)

>>> import numpy as np
>>> plotter = net.
PAram(positions=topology.positions, points=True, lines=True, rotation_speed=0.1)
>>> snn.simulate(5000, verbose=plotter)

>>> topology.plot_adjacency()
>>> pl = net.NetworkPlotter(positions=topology.positions, post=topology.post)
"""
from typing import Callable, Literal, get_args

import numpy as np
import functools as ft
import tqdm

from scipy.spatial.distance import cdist

from visualization.plotting import plt
from .network_viewer import NetworkViewer
from .network_plotter import NetworkPlotter

TopologyType = Literal['random', 'spatial']


class NetworkTopology:
    def __init__(
            self,
            n_neurons: int | tuple[int, int] = (80, 20),
            n_synapses_per_neuron: int = 10,
            topology_type: TopologyType = 'random',
            weights: float | tuple | np.ndarray = (6, -5),
            positions: np.ndarray | str | None = None
    ):
        if isinstance(n_neurons, int):
            n_neurons = tuple(int(f * n_neurons) for f in (0.8, 0.2))
        self.n_neurons = n_neurons
        self.n_synapses_per_neuron = n_synapses_per_neuron

        self.positions = self.get_positions(positions)

        if topology_type not in get_args(TopologyType):
            raise ValueError
        self.topology_type = topology_type

        self.pre, self.post = self.get_topology()
        self.weights = self.get_weights(weights)

    @property
    def n_excitatory_neurons(self) -> int:
        return self.n_neurons[0]

    @property
    def n_inhibitory_neurons(self) -> int:
        return self.n_neurons[1]

    @property
    def n_total_neurons(self) -> int:
        return sum(self.n_neurons)

    @classmethod
    def get_random_positions(cls, n_neurons: int, dim: int = 3):
        return np.random.rand(n_neurons, dim)

    def get_positions(self, positions: np.ndarray | Callable | None = None, dim: int = 3):
        positions = positions if positions is not None else ft.partial(self.get_random_positions, dim=dim)
        if isinstance(positions, Callable):  # noqa
            positions = positions(n_neurons=self.n_total_neurons, dim=dim)
        return positions

    def get_distance_matrix(self):
        return cdist(self.positions, self.positions, metric='euclidean')

    @ft.cached_property
    def distances(self):
        return self.get_distance_matrix()

    def get_post_synapses(self, i):
        N, Ne, Ni = self.n_total_neurons, self.n_excitatory_neurons, self.n_inhibitory_neurons
        M = self.n_synapses_per_neuron

        n = N if i < Ne else Ne
        if self.topology_type == 'random':
            return np.random.permutation(n)[:M]
        if self.topology_type == 'spatial':
            return np.argsort(self.distances[i, :n])[:M]

    def get_topology(self) -> tuple:
        N, M = self.n_total_neurons, self.n_synapses_per_neuron

        post = np.zeros((N, M), dtype=int)

        for i in range(N):
            post[i, :] = self.get_post_synapses(i)

        pre = [[] for _ in range(N)]
        for i in range(N):
            for m in range(M):
                pre[post[i, m]].append([i, m])  # flat index into synaptic connectivity array
        pre = [np.array(p).T for p in pre]

        return pre, post

    def get_weights(self, weights):
        N, Ne, Ni = self.n_total_neurons, self.n_excitatory_neurons, self.n_inhibitory_neurons
        M = self.n_synapses_per_neuron

        if weights is None:
            weights = (6, -5)
        if isinstance(weights, float | int):
            weights = (weights, weights)
        if isinstance(weights, tuple):
            weights = np.array([weights[i >= Ne] * (np.random.rand(M) * 0.5 + 0.5) for i in range(N)])

        if not isinstance(weights, np.ndarray) or weights.shape != (N, M):
            raise ValueError

        return weights

    def adjacency(self, weighted: bool = True):
        N, M, post, weights = self.n_total_neurons, self.n_synapses_per_neuron, self.post, self.weights

        A = np.zeros((N, N))
        for i in range(N):
            for m in range(M):
                A[i, post[i, m]] = weights[i, m] if weighted else 1

        return A

    def plot_adjacency(self, ax: plt.Axes = None):
        if ax is None:
            plt.figure()
            ax = plt.subplot(1, 1, 1)

        weights_norm = np.max(np.abs(self.weights))

        im = ax.imshow(self.adjacency(), origin='lower', cmap='seismic', clim=(-weights_norm, weights_norm))
        ax.figure.colorbar(im)

    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n_total_neurons}, " \
               f"ne={self.n_excitatory_neurons}, ni={self.n_inhibitory_neurons})"


class SpikingNetwork:
    """Simple spiking neuronal network for live art."""
    def __init__(
            self,
            v: np.ndarray | None = None,
            u: np.ndarray | None = None,
            topology: NetworkTopology | None = None,
            par_a: float | tuple | np.ndarray | None = None,
            par_d: float | tuple | np.ndarray | None = None,
            par_i: float | tuple | np.ndarray | None = None,
            threshold: float = 30,
            dt: float = 0.1,
    ):
        self.topology = topology if topology is not None else NetworkTopology()
        self.v = v if v is not None else -65 * np.ones(self.n_total_neurons)
        self.u = u if u is not None else 0.2 * self.v

        self.par_a = self._initialize_parameter(par_a, default=(0.02, 0.1))
        self.par_d = self._initialize_parameter(par_d, default=(8, 2))
        self.par_i = self._initialize_parameter(par_i, default=(0, 0))

        self.threshold = threshold
        self.dt = dt

    @property
    def n_excitatory_neurons(self) -> int:
        return self.topology.n_excitatory_neurons

    @property
    def n_inhibitory_neurons(self) -> int:
        return self.topology.n_inhibitory_neurons

    @property
    def n_total_neurons(self) -> int:
        return self.topology.n_total_neurons

    def _initialize_parameter(self, par, default):
        if par is None:
            par = default
        if isinstance(par, float):
            par = (par, par)
        if isinstance(par, tuple) and len(par) == 2:
            par = np.append(
                par[0] * np.ones(self.n_excitatory_neurons),
                par[1] * np.ones(self.n_inhibitory_neurons)
            )
        par = np.array(par, dtype=float)
        if len(par) != self.n_total_neurons:
            raise ValueError
        return par

    def simulate(
            self,
            steps: int,
            verbose: bool | NetworkViewer | NetworkPlotter = False
    ):
        n = self.n_total_neurons
        v, u, post, weights = self.v, self.u, self.topology.post, self.topology.weights

        # parameter
        par_a, par_d, par_i, threshold, dt = self.par_a, self.par_d, self.par_i, self.threshold, self.dt

        current = np.ones(n) * par_i

        viewer = None
        if verbose is True:
            # viewer = NetworkPlotter(
            #     positions=self.topology.positions,
            #     points=dict(values=np.zeros(self.n_total_neurons))
            # )
            viewer = NetworkViewer(
                rasters=np.zeros((0, 2)),
                time_window=(-dt * steps, 0),
                neuron_window=(0, self.n_total_neurons)
            )
        elif isinstance(verbose, NetworkPlotter | NetworkViewer):
            viewer = verbose

        t = 0
        for _ in tqdm.tqdm(range(steps)):
            spiking = v >= threshold  # neurons reaching firing threshold
            # spiking = np.where(spiking)[0]
            # print('spikes: ', spiking.sum())

            v[spiking] = -65  # reset v for spiking neurons
            u[spiking] = u[spiking] + par_d[spiking]  # update u for spiking neurons

            # synaptic currents & stdp depression
            current[:] = par_i
            current[np.random.randint(n)] = 20  # random thalamic input

            for i in np.where(spiking)[0]:
                current[post[i]] += weights[i]

            # integrate variables
            v = v + dt * 0.5 * ((0.04 * v + 5) * v + 140 - u + current)
            v = v + dt * 0.5 * ((0.04 * v + 5) * v + 140 - u + current)
            u = u + dt * par_a * (0.2 * v - u)

            t += dt

            if viewer is not None:
                if isinstance(viewer, NetworkPlotter):
                    viewer.update(spiking=spiking)
                else:
                    spiking = np.where(spiking)[0]
                    rasters = np.full((len(spiking), 2), fill_value=t)
                    rasters[:, 1] = spiking
                    viewer.update(rasters=rasters, variables=v, shift=dt)

        return viewer

    def __repr__(self):
        topology = self.topology.__repr__()[len(self.topology.__class__.__name__):]
        return f"{self.__class__.__name__}({topology=})"
