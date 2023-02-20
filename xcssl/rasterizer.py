import abc
import itertools

import numpy as np

from .dimension import IntegerDimension, RealDimension
from .obs_space import IntegerObsSpace, RealObsSpace

_MIN_NUM_GRID_DIMS = 1


class ObsSpaceRasterizerABC(metaclass=abc.ABCMeta):
    def __init__(self, obs_space, num_grid_dims, num_bins_per_grid_dim, seed):
        self._obs_space = obs_space
        self._d = len(self._obs_space)

        assert _MIN_NUM_GRID_DIMS <= num_grid_dims <= self._d
        self._k = num_grid_dims
        self._b = num_bins_per_grid_dim

        self._b_pow_vec = self._gen_b_pow_vec(self._b, self._k)

        self._num_grid_cells = (self._b**self._k)

        self._grid_dim_idxs = self._init_grid_dim_idxs(self._d, self._k, seed)

    def _gen_b_pow_vec(self, b, k):
        b_pow = 1
        res = [b_pow]

        for _ in range(k - 1):
            b_pow *= b
            res.append(b_pow)

        assert len(res) == k

        return tuple(reversed(res))

    def _init_grid_dim_idxs(self, d, k, seed):
        rng = np.random.RandomState(int(seed))

        # d C k
        grid_dim_idxs = rng.choice(a=range(d), size=k, replace=False)
        grid_dim_idxs = tuple(sorted(grid_dim_idxs))

        return grid_dim_idxs

    def rasterize_phenotype(self, encoding, phenotype):

        phenotype_bounding_intervals = \
            encoding.calc_phenotype_bounding_intervals_on_dims(
                phenotype, self._grid_dim_idxs)

        grid_cell_bins_covered_each_dim = \
            self._rasterize_phenotype_bounding_intervals_each_dim(
                phenotype_bounding_intervals)

        grid_cells_covered = []
        for grid_cell_bin_combo in itertools.product(
                *grid_cell_bins_covered_each_dim):

            grid_cells_covered.append(
                self._convert_grid_cell_bin_combo_to_dec(grid_cell_bin_combo))

        return tuple(grid_cells_covered)

    @abc.abstractmethod
    def _rasterize_phenotype_bounding_intervals_each_dim(
            self, phenotype_bounding_intervals):
        raise NotImplementedError

    def rasterize_obs(self, obs):
        grid_cell_bin_combo = self._rasterize_obs_val_each_dim(obs)
        return self._convert_grid_cell_bin_combo_to_dec(grid_cell_bin_combo)

    @abc.abstractmethod
    def _rasterize_obs_val_each_dim(self, obs):
        raise NotImplementedError

    def _convert_grid_cell_bin_combo_to_dec(self, grid_cell_bin_combo):
        return sum(e * b_pow
                   for (e, b_pow) in zip(grid_cell_bin_combo, self._b_pow_vec))

    @property
    def num_grid_cells(self):
        return self._num_grid_cells


class IntegerObsSpaceRasterizer(ObsSpaceRasterizerABC):
    def __init__(self, obs_space, num_grid_dims, seed):
        assert isinstance(obs_space, IntegerObsSpace)

        dim_spans = [dim.span for dim in obs_space]
        # enforce all dims must have the same span
        assert len(set(dim_spans)) == 1
        num_bins_per_grid_dim = dim_spans[0]

        super().__init__(obs_space, num_grid_dims, num_bins_per_grid_dim, seed)

    def _rasterize_phenotype_bounding_intervals_each_dim(
            self, phenotype_bounding_intervals):

        grid_cell_bins_covered_each_dim = []

        for bounding_interval in phenotype_bounding_intervals:
            # go up in +1 increments from lo to hi, since integer space
            lo = bounding_interval[0]
            hi = bounding_interval[1]
            grid_cell_bins_covered_each_dim.append(
                tuple(range(lo, (hi + 1), 1)))

        return grid_cell_bins_covered_each_dim

    def _rasterize_obs_val_each_dim(self, obs):
        return tuple(obs[idx] for idx in self._grid_dim_idxs)


class RealObsSpaceRasterizer(ObsSpaceRasterizerABC):
    pass
