import abc
import time
from collections import OrderedDict

from .index import RandomIndex


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

    def remove(self, clfr_idx, op=None):
        tick = time.perf_counter()
        self._wrapped_pop.remove(clfr_idx, op)
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

    @property
    @abc.abstractmethod
    def clfrs(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_macros(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add_new(self, clfr, op):
        raise NotImplementedError

    @abc.abstractmethod
    def remove(self, clfr_idx, op=None):
        raise NotImplementedError

    @abc.abstractmethod
    def gen_match_set(self, obs):
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError


class VanillaPopulation(PopulationABC):
    """Default-style population that does not use an index and
    perfoms exhuastive matching."""
    def __init__(self):
        super().__init__()
        # use list to store clfrs
        self._clfrs = []

    @property
    def clfrs(self):
        return self._clfrs

    @property
    def num_macros(self):
        return len(self._clfrs)

    def add_new(self, clfr, op):
        self._num_micros += clfr.numerosity
        assert op in ("covering", "insertion")
        self._ops_history[op] += clfr.numerosity

        self._clfrs.append(clfr)

    def remove(self, clfr_idx, op=None):
        clfr = self._clfrs.pop(clfr_idx)

        self._num_micros -= clfr.numerosity
        if op is not None:
            assert op == "deletion"
            self._ops_history[op] += clfr.numerosity

    def gen_match_set(self, obs):
        """Exhaustive matching procedure: match each
        macroclassifier individually."""
        return [clfr for clfr in self._clfrs if clfr.does_match(obs)]

    def __getitem__(self, idx):
        return self._clfrs[idx]

    def __iter__(self):
        return iter(self._clfrs)


class FastMatchingPopulation(PopulationABC):
    """Population that uses an index to perform fast matching."""
    def __init__(self, encoding, rasterizer_kwargs):
        super().__init__()
        # use map from int_id to clfr to store clfrs (as need the int ids to
        # efficiently add to/query from the index). Needs to be a map to deal
        # with "holes" introduced in the ids via deletion: if just used a list
        # and used the list idxs as ids then would have to readjust the index
        # everytime deleting something not at the end of the list
        self._int_id_clfr_map = OrderedDict()
        self._next_int_id = 0
        self._index = RandomIndex.from_scratch(encoding, rasterizer_kwargs)

    @property
    def clfrs(self):
        return list(self._int_id_clfr_map.keys())

    @property
    def num_macros(self):
        return len(self._int_id_clfr_map)

    def add_new(self, clfr, op):
        self._num_micros += clfr.numerosity
        assert op in ("covering", "insertion")
        self._ops_history[op] += clfr.numerosity

        int_id = self._get_next_int_id()
        self._int_id_clfr_map[int_id] = clfr
        self._index.add(int_id, clfr.condition.phenotype)

    def remove(self, clfr_idx, op=None):
        int_id = self._find_int_id_of_clfr_idx(clfr_idx)
        clfr_numerosity = (self._int_id_clfr_map[int_id]).numerosity

        self._num_micros -= clfr_numerosity
        if op is not None:
            assert op == "deletion"
            self._ops_history[op] += clfr_numerosity
        
        del self._int_id_clfr_map[int_id]
        self._index.remove(int_id)

    def gen_match_set(self, obs):
        """Lookup in index to gen match set."""
        return self._index.gen_match_set(self._int_id_clfr_map, obs)

    def _get_next_int_id(self):
        res = self._next_int_id
        self._next_int_id += 1
        return res

    def __getitem__(self, idx):
        int_id = self._find_int_id_of_clfr_idx(idx)
        return self._int_id_clfr_map[int_id]

    def __iter__(self):
        return iter(self._int_id_clfr_map.values())

    def _find_int_id_of_clfr_idx(self, clfr_idx):
        return list(self._int_id_clfr_map.keys())[clfr_idx]
