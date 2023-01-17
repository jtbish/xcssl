class Phenotype:
    def __init__(self, elems, generality):
        self._elems = tuple(elems)
        self._generality = generality

        # cache the hash value for faster set/dict + eq operations
        self._hash = hash(self._elems)

    @property
    def elems(self):
        return self._elems

    @property
    def generality(self):
        return self._generality

    def __eq__(self, other):
        if self._hash != other._hash:
            return False
        else:
            # only compare all elems on hash collision
            for (my_elem, other_elem) in zip(self._elems, other._elems):
                if my_elem != other_elem:
                    return False
            return True

    def __getitem__(self, idx):
        return self._elems[idx]

    def __hash__(self):
        return self._hash

    def __iter__(self):
        return iter(self._elems)

    def __len__(self):
        return len(self._elems)
