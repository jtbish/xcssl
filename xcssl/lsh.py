import abc
from math import factorial as fact

import numpy as np

from .rng import get_rng


class LocalitySensitiveHasherABC(metaclass=abc.ABCMeta):
    def __init__(self, d, p, b):
        self._d = d
        assert p <= self._d
        self._p = p
        self._b = b

        self._bands = self._init_bands(self._d, self._p, self._b)

    @abc.abstractmethod
    def _init_bands(self, d, p, b):
        raise NotImplementedError

    @abc.abstractmethod
    def hash(self, vec, band_idx):
        raise NotImplementedError


class HammingLSH(LocalitySensitiveHasherABC):
    def __init__(self, d, p, b):
        def _n_choose_k(n, k):
            return int(fact(n) / (fact(k) * fact(n - k)))

        assert p <= d
        # a hash func. chooses p dims from possible d
        num_possible_hash_funcs = _n_choose_k(n=d, k=p)

        if num_possible_hash_funcs <= b:
            raise ValueError("Num possible hash functions for Hamming LSH is "
                             "<= the number of bands b specified.")

        super().__init__(d, p, b)

    def _init_bands(self, d, p, b):
        """For Hamming LSH, each 'hash function' in a band is simply a vector
        of length p (the number of projs), specifying the (ordered) indexes of
        the input vector to consider."""
        def _is_dup_hash_func(new_hash_func, bands):
            for existing_hash_func in bands:
                if np.array_equal(new_hash_func, existing_hash_func):
                    return True
            return False

        bands = []
        to_sample = range(d)

        for _ in range(b):

            is_valid_hash_func = False
            while not is_valid_hash_func:
                # sample p random idxs from possible range of [0, d-1]
                hash_func = get_rng().choice(a=to_sample,
                                             size=p,
                                             replace=False)
                # then order idxs
                hash_func = np.sort(hash_func)

                is_valid_hash_func = \
                    not _is_dup_hash_func(hash_func, bands)

            bands.append(hash_func)

        return bands

    def hash(self, vec, band_idx):
        hash_func = self._bands[band_idx]
        return tuple(vec[idx] for idx in hash_func)


class EuclideanLSH(LocalitySensitiveHasherABC):
    def _init_bands(self, d, p, b):
        raise NotImplementedError

    def hash(self, vec, band_idx):
        raise NotImplementedError
