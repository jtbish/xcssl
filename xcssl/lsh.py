import abc

import numpy as np

_MIN_NUM_PROJS = 1


class LSHKey:
    """Wrapper over tuple for LSH keys to cache the hash value."""
    def __init__(self, key_tup):
        self._key_tup = key_tup
        self._hash = hash(self._key_tup)

    def __eq__(self, other):
        return self._key_tup == other._key_tup

    def __hash__(self):
        return self._hash


class LocalitySensitiveHasherABC(metaclass=abc.ABCMeta):
    def __init__(self, num_dims, num_projs, seed):
        self._num_dims = num_dims
        # num_projs should be less than num_dims otherwise projecting into an
        # equal or higher dimensional space than what was started with, which
        # is weird
        assert _MIN_NUM_PROJS <= num_projs < num_dims
        self._num_projs = num_projs

        self._rng = np.random.RandomState(int(seed))

        self._projector = self._init_projector(self._num_dims, self._num_projs,
                                               self._rng)

    @property
    def num_dims(self):
        return self._num_dims

    @property
    def num_projs(self):
        return self._num_projs

    @abc.abstractmethod
    def _init_projector(self, num_dims, num_projs, rng):
        raise NotImplementedError

    @abc.abstractmethod
    def hash(self, vec):
        raise NotImplementedError


class HammingLSH(LocalitySensitiveHasherABC):
    def _init_projector(self, num_dims, num_projs, rng):
        """For Hamming LSH, the projector is equal to the hash function, which
        is simply a vector of length num_projs, specifying the (ordered)
        indexes of the input vector to consider."""

        to_sample = range(num_dims)
        projector = rng.choice(a=to_sample, size=num_projs, replace=False)
        projector = tuple(sorted(projector))

        return projector

    def hash(self, vec):
        return LSHKey(tuple(vec[idx] for idx in self._projector))


class EuclideanLSH(LocalitySensitiveHasherABC):
    def _init_projector(self, num_dims, num_projs, rng):
        raise NotImplementedError

    def hash(self, vec):
        raise NotImplementedError
