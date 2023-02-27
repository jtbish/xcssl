import abc
import time

from .index import CoverageMap


def make_vanilla_population(do_timing=False):
    if not do_timing:
        return VanillaPopulation()
    else:
        return TimedPopulationWrapper.as_vanilla_pop()


def make_fm_population(vanilla_pop,
                       encoding,
                       rasterizer_kwargs,
                       do_timing=False):
    if not do_timing:
        return FastMatchingPopulation(vanilla_pop, encoding, rasterizer_kwargs)
    else:
        timed_vanilla_pop = vanilla_pop
        return TimedPopulationWrapper.as_fm_pop_from_timed_vanilla_pop(
            timed_vanilla_pop, encoding, rasterizer_kwargs)


class TimedPopulationWrapper:
    def __init__(self, wrapped_pop, timers):
        self._wrapped_pop = wrapped_pop
        self._timers = timers

    @classmethod
    def as_vanilla_pop(cls):
        # timed vanilla pop gets zeroed timers
        timers = {
            "__init__": 0.0,
            "add_new": 0.0,
            "remove": 0.0,
            "gen_match_set": 0.0
        }

        tick = time.perf_counter()
        wrapped_pop = VanillaPopulation()
        tock = time.perf_counter()

        timers["__init__"] += (tock - tick)

        return cls(wrapped_pop, timers)

    @classmethod
    def as_fm_pop_from_timed_vanilla_pop(cls, timed_vanilla_pop, encoding,
                                         rasterizer_kwargs):
        assert isinstance(timed_vanilla_pop, TimedPopulationWrapper)
        vanilla_pop = timed_vanilla_pop._wrapped_pop

        tick = time.perf_counter()
        wrapped_pop = FastMatchingPopulation(vanilla_pop, encoding,
                                             rasterizer_kwargs)
        tock = time.perf_counter()

        # timed FM pop inherits timers of timed vanilla pop
        timers = timed_vanilla_pop._timers
        timers["__init__"] += (tock - tick)

        return cls(wrapped_pop, timers)

    @property
    def timers(self):
        return self._timers

    @property
    def num_macros(self):
        return self._wrapped_pop.num_macros

    @property
    def num_micros(self):
        return self._wrapped_pop.num_micros

    @property
    def ops_history(self):
        return self._wrapped_pop.ops_history

    def alter_numerosity(self, clfr, delta, op):
        self._wrapped_pop.alter_numerosity(clfr, delta, op)

    def add_new(self, clfr, op, time_step=None):
        tick = time.perf_counter()
        self._wrapped_pop.add_new(clfr, op, time_step)
        tock = time.perf_counter()

        self._timers["add_new"] += (tock - tick)

    def remove(self, clfr, op=None):
        tick = time.perf_counter()
        self._wrapped_pop.remove(clfr, op)
        tock = time.perf_counter()

        self._timers["remove"] += (tock - tick)

    def gen_match_set(self, obs):
        tick = time.perf_counter()
        self._wrapped_pop.gen_match_set(obs)
        tock = time.perf_counter()

        self._timers["gen_match_set"] += (tock - tick)

    def __getitem__(self, idx):
        return self._wrapped_pop[idx]

    def __iter__(self):
        return iter(self._wrapped_pop)


class PopulationABC(metaclass=abc.ABCMeta):
    """Population is just a list of macroclassifiers with tracking of the number of
    microclassifiers, in order to avoid having to calculate this number on the
    fly repeatedly and waste time."""
    @property
    def num_macros(self):
        return len(self._clfrs)

    @property
    def num_micros(self):
        return self._num_micros

    @property
    def ops_history(self):
        return self._ops_history

    def alter_numerosity(self, clfr, delta, op):
        clfr.numerosity += delta
        self._num_micros += delta
        assert op in ("absorption", "deletion", "ga_subsumption",
                      "as_subsumption")
        # delta can be neg. (obviously), but op. counts are pos.
        self._ops_history[op] += abs(delta)

    @abc.abstractmethod
    def add_new(self, clfr, op, time_step=None):
        raise NotImplementedError

    @abc.abstractmethod
    def remove(self, clfr, op=None):
        raise NotImplementedError

    @abc.abstractmethod
    def gen_match_set(self, obs):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self._clfrs[idx]

    def __iter__(self):
        return iter(self._clfrs)


class VanillaPopulation(PopulationABC):
    """Default-style population that does not use phenotype clustering and
    perfoms fully accurate and exhuastive matching."""
    def __init__(self):
        self._clfrs = []
        self._num_micros = 0
        self._ops_history = {
            "covering": 0,
            "absorption": 0,
            "insertion": 0,
            "deletion": 0,
            "ga_subsumption": 0,
            "as_subsumption": 0
        }

    def add_new(self, clfr, op, time_step=None):
        self._clfrs.append(clfr)
        self._num_micros += clfr.numerosity
        assert op in ("covering", "insertion")
        self._ops_history[op] += clfr.numerosity

    def remove(self, clfr, op=None):
        self._clfrs.remove(clfr)
        self._num_micros -= clfr.numerosity
        if op is not None:
            assert op == "deletion"
            self._ops_history[op] += clfr.numerosity

    def gen_match_set(self, obs):
        """Full and exhaustive matching procedure: match each
        macroclassifier."""
        return [clfr for clfr in self._clfrs if clfr.does_match(obs)]


class FastMatchingPopulation(PopulationABC):
    """Population that uses an index to perform fast matching."""
    def __init__(self, vanilla_pop, encoding, rasterizer_kwargs):
        """FastMatchingPopulation needs to be inited from existing
        VanillaPopulation."""

        assert isinstance(vanilla_pop, VanillaPopulation)

        self._clfrs = vanilla_pop._clfrs
        self._num_micros = vanilla_pop._num_micros
        self._ops_history = vanilla_pop._ops_history

        self._index = CoverageMap(
            encoding=encoding,
            rasterizer_kwargs=rasterizer_kwargs,
            phenotypes=[clfr.condition.phenotype for clfr in self._clfrs])

    def add_new(self, clfr, op, time_step=None):
        self._clfrs.append(clfr)
        self._num_micros += clfr.numerosity
        assert op in ("covering", "insertion")
        self._ops_history[op] += clfr.numerosity

        self._index.try_add_phenotype(clfr.condition.phenotype)

    def remove(self, clfr, op=None):
        self._clfrs.remove(clfr)
        self._num_micros -= clfr.numerosity
        if op is not None:
            assert op == "deletion"
            self._ops_history[op] += clfr.numerosity

        self._index.try_remove_phenotype(clfr.condition.phenotype)

    def gen_match_set(self, obs):
        sparse_phenotype_matching_map = \
            self._index.gen_sparse_phenotype_matching_map(obs)

        return [
            clfr for clfr in self._clfrs if sparse_phenotype_matching_map.get(
                clfr.condition.phenotype, False)
        ]
