import abc

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
    def _init_bands(self, d, p, b):
        """For Hamming LSH, each 'hash function' in a band is simply a vector
        of length p (the number of projs), specifying the indexes of the input
        vector to consider."""

        bands = []
        to_sample = range(d)
        for _ in range(b):
            hash_function = get_rng().choice(a=to_sample,
                                             size=p,
                                             replace=False)
            bands.append(hash_function)

        return bands

    def hash(self, vec, band_idx):
        hash_function = self._bands[band_idx]
        return tuple([vec[idx] for idx in hash_function])


class EuclideanLSH(LocalitySensitiveHasherABC):
    def _init_bands(self, d, p, b):
        raise NotImplementedError

    def hash(self, vec, band_idx):
        raise NotImplementedError
