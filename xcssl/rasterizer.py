import abc
import math
import itertools

import numpy as np

from .obs_space import IntegerObsSpace, RealObsSpace

_MIN_NUM_GRID_DIMS = 1
_MIN_NUM_BINS_PER_GRID_DIM = 2


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

        self._grid_dim_idxs = self._init_grid_dim_idxs(self._d, self._k, seed)

        self._anti_grid_dim_idxs = self._calc_anti_grid_dim_idxs(
            self._d, self._k, self._grid_dim_idxs)

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

    def rasterize_aabb(self, aabb):
        bins_covered_on_grid_dims = self._rasterize_aabb_on_grid_dims(aabb)
        return itertools.product(*bins_covered_on_grid_dims)

    @property
    def num_grid_dims(self):
        return self._k

    @property
    def num_bins_per_grid_dim(self):
        return self._b

    @abc.abstractmethod
    def _rasterize_aabb_on_grid_dims(self, aabb):
        raise NotImplementedError

    @abc.abstractmethod
    def rasterize_obs(self, obs):
        """Return grid cell bin combo tup for obs."""
        raise NotImplementedError

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

    def rasterize_obs(self, obs):
        return [obs[idx] for idx in self._grid_dim_idxs]

    def match_idxd_aabb(self, aabb, obs):
        # logic here is that, since all possible vals on each of the grid dims
        # are being indexed, the only thing needed to check if aabb matches is
        # to check the anti grid dims
        return aabb.contains_obs_given_dims(obs, self._anti_grid_dim_idxs)


class RealObsSpaceRasterizer(ObsSpaceRasterizerABC):
    def __init__(self, obs_space, seed, num_grid_dims, num_bins_per_grid_dim):

        num_bins_per_grid_dim = int(num_bins_per_grid_dim)
        assert num_bins_per_grid_dim >= _MIN_NUM_BINS_PER_GRID_DIM

        super().__init__(obs_space, seed, num_grid_dims, num_bins_per_grid_dim)

        # enforce that obs space must be min-max scaled to occupy unit
        # hypercube (this makes rasterization logic easier)
        for dim in obs_space:
            assert dim.lower == 0.0
            assert dim.upper == 1.0

        self._max_bin_idx = (self._b - 1)

    def _rasterize_aabb_on_grid_dims(self, aabb):
        bins_covered_on_grid_dims = []

        for dim_idx in self._grid_dim_idxs:
            interval = aabb[dim_idx]
            # calc the bins that lower/upper of the interval occupies
            lower_bin_idx = self._calc_bin_idx(interval.lower)
            upper_bin_idx = self._calc_bin_idx(interval.upper)
            # then say that the interval covers all the in-between bins as well
            # (if any)
            bins_covered_on_grid_dims.append(
                tuple(range(lower_bin_idx, (upper_bin_idx + 1), 1)))

        return bins_covered_on_grid_dims

    def rasterize_obs(self, obs):
        return [self._calc_bin_idx(obs[idx]) for idx in self._grid_dim_idxs]

    def _calc_bin_idx(self, val):
        # first determine integer bin idx along range of [0.0, 1.0] that val
        # belongs to.
        # then handle the edge case of one over the max bin idx by truncating
        # with min()
        return min(math.floor(val * self._b), self._max_bin_idx)

    def match_idxd_aabb(self, aabb, obs):
        # logic here is that, if obs not contained in anti grid dim AABB
        # intervals, not possible for it to match.
        # However, if the obs *is contained* in the anti grid dim intervals,
        # still possible that the whole AABB could not match, due to the
        # discretisation of the real space applied on the grid dim idxs,
        # so need to check the grid dim intervals as well in that case.
        if not aabb.contains_obs_given_dims(obs, self._anti_grid_dim_idxs):
            return False
        else:
            return aabb.contains_obs_given_dims(obs, self._grid_dim_idxs)
