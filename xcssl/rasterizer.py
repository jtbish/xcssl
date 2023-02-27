import abc
import itertools

import numpy as np

from .dimension import IntegerDimension, RealDimension
from .obs_space import IntegerObsSpace, RealObsSpace

_MIN_NUM_GRID_DIMS = 1


def make_rasterizer(obs_space, rasterizer_kwargs):

    if isinstance(obs_space, IntegerObsSpace):
        cls = IntegerObsSpaceRasterizer

    elif isinstance(obs_space, RealObsSpace):
        cls = RealObsSpaceRasterizer

    else:
        assert False

    return cls(obs_space, **rasterizer_kwargs)


class ObsSpaceRasterizerABC(metaclass=abc.ABCMeta):
    def __init__(self, obs_space, seed, num_grid_dims, num_bins_per_grid_dim):
        self._obs_space = obs_space
        self._d = len(self._obs_space)

        assert _MIN_NUM_GRID_DIMS <= num_grid_dims <= self._d
        self._k = num_grid_dims
        self._b = num_bins_per_grid_dim

        self._b_pow_vec = self._gen_b_pow_vec(self._b, self._k)

        self._num_grid_cells = (self._b**self._k)

        self._grid_dim_idxs = self._init_grid_dim_idxs(self._d, self._k, seed)

        self._anti_grid_dim_idxs = self._calc_anti_grid_dim_idxs(
            self._d, self._k, self._grid_dim_idxs)

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

    def _calc_anti_grid_dim_idxs(self, d, k, grid_dim_idxs):
        all_grid_dims_set = set(range(0, d))

        anti_grid_dim_idxs = tuple(
            sorted(all_grid_dims_set - set(grid_dim_idxs)))
        assert len(anti_grid_dim_idxs) == (d - k)

        return anti_grid_dim_idxs

    @property
    def num_grid_cells(self):
        return self._num_grid_cells

    def rasterize_aabb(self, aabb):
        bins_covered_on_grid_dims = self._rasterize_aabb_on_grid_dims(aabb)
        return itertools.product(*bins_covered_on_grid_dims)

    @abc.abstractmethod
    def _rasterize_aabb_on_grid_dims(self, aabb):
        raise NotImplementedError

    def rasterize_obs(self, obs):
        return self.convert_grid_cell_bin_combo_to_dec(
            self._rasterize_obs_on_grid_dims(obs))

    @abc.abstractmethod
    def _rasterize_obs_on_grid_dims(self, obs):
        raise NotImplementedError

    def convert_grid_cell_bin_combo_to_dec(self, grid_cell_bin_combo):
        return sum(e * b_pow
                   for (e, b_pow) in zip(grid_cell_bin_combo, self._b_pow_vec))

    @abc.abstractmethod
    def match_idxd_aabb(self, aabb, obs):
        raise NotImplementedError


class IntegerObsSpaceRasterizer(ObsSpaceRasterizerABC):
    def __init__(self,
                 obs_space,
                 seed,
                 num_grid_dims,
                 num_bins_per_grid_dim=None):

        assert isinstance(obs_space, IntegerObsSpace)
        assert num_bins_per_grid_dim is None

        dim_spans = [dim.span for dim in obs_space]
        # enforce all dims must have the same span, and that b is equal to this
        # common span
        # TODO could relax this
        assert len(set(dim_spans)) == 1
        num_bins_per_grid_dim = dim_spans[0]

        super().__init__(obs_space, seed, num_grid_dims, num_bins_per_grid_dim)

    def _rasterize_aabb_on_grid_dims(self, aabb):

        bins_covered_on_grid_dims = []

        for dim_idx in self._grid_dim_idxs:
            interval = aabb[dim_idx]
            # go up in +1 increments from lower to upper, since integer space
            # where all vals on each dim are included as bins in the grid
            bins_covered_on_grid_dims.append(
                tuple(range(interval.lower, (interval.upper + 1), 1)))

        return bins_covered_on_grid_dims

    def _rasterize_obs_on_grid_dims(self, obs):
        return tuple(obs[idx] for idx in self._grid_dim_idxs)

    def match_idxd_aabb(self, aabb, obs):
        return aabb.contains_obs_given_dims(obs, self._anti_grid_dim_idxs)


class RealObsSpaceRasterizer(ObsSpaceRasterizerABC):
    pass
