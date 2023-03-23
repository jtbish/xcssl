import abc
import time

from .index import CoverageMap


def make_vanilla_population(do_timing=False):
    pop = VanillaPopulation()

    if not do_timing:
        return pop
    else:
        return TimedPopulationWrapper(pop)


def make_fm_population(encoding, rasterizer_kwargs, do_timing=False):
    pop = FastMatchingPopulation(encoding, rasterizer_kwargs)

    if not do_timing:
        return pop
    else:
        return TimedPopulationWrapper(pop)


class TimedPopulationWrapper:
    def __init__(self, wrapped_pop):
        self._wrapped_pop = wrapped_pop
        self._timers = {"add_new": 0.0, "remove": 0.0, "gen_match_set": 0.0}

    @property
    def timers(self):
        return self._timers

    @property
    def clfrs(self):
        return self._wrapped_pop.clfrs

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

    def add_new(self, clfr, op):
        tick = time.perf_counter()
        self._wrapped_pop.add_new(clfr, op)
        tock = time.perf_counter()

        self._timers["add_new"] += (tock - tick)

    def remove(self, clfr, op=None):
        tick = time.perf_counter()
        self._wrapped_pop.remove(clfr, op)
        tock = time.perf_counter()

        self._timers["remove"] += (tock - tick)

    def gen_match_set(self, obs):
        tick = time.perf_counter()
        match_set = self._wrapped_pop.gen_match_set(obs)
        tock = time.perf_counter()

        self._timers["gen_match_set"] += (tock - tick)

        return match_set

    def __getitem__(self, idx):
        return self._wrapped_pop[idx]

    def __iter__(self):
        return iter(self._wrapped_pop)


class PopulationABC(metaclass=abc.ABCMeta):
    """Population is just a list of macroclassifiers with tracking of the number of
    microclassifiers, in order to avoid having to calculate this number on the
    fly repeatedly and waste time."""
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

    @property
    def clfrs(self):
        return self._clfrs

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
    def add_new(self, clfr, op):
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
    """Default-style population that does not use an index and
    perfoms fully accurate and exhuastive matching."""
    def add_new(self, clfr, op):
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
    def __init__(self, encoding, rasterizer_kwargs):
        super().__init__()
        self._index = CoverageMap.from_scratch(encoding, rasterizer_kwargs)

    def add_new(self, clfr, op):
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
