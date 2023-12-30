class Phenotype:
    def __init__(self, elems):
        self._elems = tuple(elems)
        self._aabb = None

    @property
    def elems(self):
        return self._elems

    @property
    def aabb(self):
        return self._aabb

    def __eq__(self, other):
        for (my_elem, other_elem) in zip(self._elems, other._elems):
            if my_elem != other_elem:
                return False
        return True

    def monkey_patch_and_return_aabb(self, encoding):
        aabb = encoding.make_phenotype_aabb(self)
        self._aabb = aabb
        return aabb

    def __getitem__(self, idx):
        return self._elems[idx]

    def __iter__(self):
        return iter(self._elems)

    def __len__(self):
        return len(self._elems)
