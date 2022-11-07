import abc
import numpy as np


class PhenotypeABC(metaclass=abc.ABCMeta):
    def __init__(self, elems):
        self._elems = elems

    @property
    def elems(self):
        return self._elems

    def __eq__(self, other):
        for (my_elem, other_elem) in zip(self._elems, other._elems):
            if my_elem != other_elem:
                return False
        return True

    def __iter__(self):
        return iter(self._elems)

    def __len__(self):
        return len(self._elems)


class VanillaPhenotype(PhenotypeABC):
    pass


class VectorisedPhenotype(PhenotypeABC):
    """Phenotype for LSH with additional vectorised repr."""
    def __init__(self, elems, vec):
        super().__init__(elems)
        self._vec = vec
        # cache the hash value for faster set/dict operations
        self._hash = self._calc_hash(self._vec)

    def _calc_hash(self, vec):
        return hash(tuple(vec))

    @property
    def vec(self):
        return self._vec

    def __eq__(self, other):
        """Equality can be done faster for VectorisedPhenotype by comparing
        vecs."""
        return np.array_equal(self._vec, other._vec)

    def __hash__(self):
        return self._hash

    def __str__(self):
        return str(self._vec)
