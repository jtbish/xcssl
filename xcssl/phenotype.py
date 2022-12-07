import abc


class PhenotypeABC(metaclass=abc.ABCMeta):
    def __init__(self, elems):
        self._elems = tuple(elems)

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

    def __getitem__(self, idx):
        return self._elems[idx]

    def __len__(self):
        return len(self._elems)


class VanillaPhenotype(PhenotypeABC):
    pass


class VectorisedPhenotype(PhenotypeABC):
    """Phenotype for LSH, imbued with additional vectorised repr."""
    def __init__(self, elems, vec):
        super().__init__(elems)
        self._vec = vec
        # cache the hash value for faster set/dict operations
        self._hash = hash(self._elems)

    @property
    def vec(self):
        return self._vec

    def __hash__(self):
        return self._hash

    def __str__(self):
        return str(self._vec)
