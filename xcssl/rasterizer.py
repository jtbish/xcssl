import abc
import itertools

import numpy as np

from .dimension import IntegerDimension, RealDimension
from .obs_space import IntegerObsSpace, RealObsSpace

_MIN_NUM_GRID_DIMS = 1


class ObsSpaceRasterizerABC(metaclass=abc.ABCMeta):
    def __init__(self, obs_space, num_grid_dims, seed):
        self._obs_space = obs_space
        self._d = len(self._obs_space)

        assert _MIN_NUM_GRID_DIMS <= num_grid_dims <= self._d
        self._k = num_grid_dims

        self._grid_dim_idxs = self._init_grid_dim_idxs(self._d, self._k, seed)

        self._num_bins_each_dim = self._calc_num_bins_each_dim(
            self._obs_space, self._grid_dim_idxs)

        self._num_grid_cells = np.product(self._num_bins_each_dim)

    def _init_grid_dim_idxs(self, d, k, seed):
        rng = np.random.RandomState(int(seed))

        grid_dim_idxs = rng.choice(a=range(d), size=k, replace=False)
        grid_dim_idxs = tuple(sorted(grid_dim_idxs))

        return grid_dim_idxs

    @abc.abstractmethod
    def _calc_num_bins_each_dim(self, obs_space, grid_dim_idxs):
        raise NotImplementedError

    def rasterize_phenotype(self, encoding, phenotype):

        phenotype_bounding_intervals = \
            encoding.calc_phenotype_bounding_intervals_on_dims(
                phenotype, self._grid_dim_idxs)

        grid_cell_bins_covered_each_dim = \
            self._rasterize_phenotype_bounding_intervals_each_dim(
                phenotype_bounding_intervals)

        assert len(grid_cell_bins_covered_each_dim) == self._k

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
        # thanks: https://math.stackexchange.com/questions/2008367/how-to-convert-an-index-into-n-coordinates
        # (column-major i.e. same as nested for loops)

        N = len(grid_cell_bin_combo)

        res = 0

        for n in range(1, N + 1):
            i_n = grid_cell_bin_combo[n - 1]

            prod = 1

            for m in range(n + 1, N + 1):
                s_m = self._num_bins_each_dim[m - 1]
                prod *= s_m

            res += (i_n * prod)

        return res

    @property
    def num_grid_cells(self):
        return self._num_grid_cells


class IntegerObsSpaceRasterizer(ObsSpaceRasterizerABC):
    def __init__(self, obs_space, num_grid_dims, seed):
        assert isinstance(obs_space, IntegerObsSpace)
        super().__init__(obs_space, num_grid_dims, seed)

    def _calc_num_bins_each_dim(self, obs_space, grid_dim_idxs):
        res = []

        for dim_idx in grid_dim_idxs:
            dim = obs_space[dim_idx]
            assert isinstance(dim, IntegerDimension)
            res.append(dim.span)

        return res

    def _rasterize_phenotype_bounding_intervals_each_dim(
            self, phenotype_bounding_intervals):

        grid_cell_bins_covered_each_dim = []

        for bounding_interval in phenotype_bounding_intervals:
            # go up in +1 increments from lo to hi, since integer space
            lo = bounding_interval[0]
            hi = bounding_interval[1]
            assert lo <= hi
            grid_cell_bins_covered_each_dim.append(list(range(lo, hi + 1, 1)))

        return grid_cell_bins_covered_each_dim

    def _rasterize_obs_val_each_dim(self, obs):
        return tuple(obs[idx] for idx in self._grid_dim_idxs)


class RealObsSpaceRasterizer(ObsSpaceRasterizerABC):
    pass
