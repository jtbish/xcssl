import abc

from .dimension import IntegerDimension, RealDimension


class ObsSpaceBase:
    def __init__(self, dims):
        self._dims = tuple(dims)

    @property
    def dims(self):
        return self._dims

    def __iter__(self):
        return iter(self._dims)

    def __len__(self):
        return len(self._dims)


class IntegerObsSpace(ObsSpaceBase):
    pass


class RealObsSpace(ObsSpaceBase):
    pass


class ObsSpaceBuilderABC(metaclass=abc.ABCMeta):
    """Convenience class to make syntax of building obs space nicer."""
    def __init__(self):
        self._dims = []

    def add_dim(self, dim):
        self._check_dim(dim)
        self._dims.append(dim)

    @abc.abstractmethod
    def _check_dim(self, dim):
        raise NotImplementedError

    @abc.abstractmethod
    def create_space(self):
        raise NotImplementedError


class IntegerObsSpaceBuilder(ObsSpaceBuilderABC):
    def _check_dim(self, dim):
        assert isinstance(dim, IntegerDimension)

    def create_space(self):
        return IntegerObsSpace(self._dims)


class RealObsSpaceBuilder(ObsSpaceBuilderABC):
    def _check_dim(self, dim):
        assert isinstance(dim, RealDimension)

    def create_space(self):
        return RealObsSpace(self._dims)


def make_binary_obs_space(num_dims):
    """Binary obs space is special case of integer one where each dim only
    0/1"""
    builder = IntegerObsSpaceBuilder()
    for i in range(num_dims):
        builder.add_dim(IntegerDimension(lower=0, upper=1, name=f"bit_{i}"))
    return builder.create_space()
