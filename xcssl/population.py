import abc
import logging

from .clustering import ConditionClustering
from .constants import MIN_TIME_STEP
from .hyperparams import get_hyperparam as get_hp


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

    def _gen_match_set_exhaustive(self, obs):
        """Match all macroclassifiers in fully accurate & exhaustive fashion"""
        return [clfr for clfr in self._clfrs if clfr.does_match(obs)]

    def __iter__(self):
        return iter(self._clfrs)

    def __getitem__(self, idx):
        return self._clfrs[idx]

    @abc.abstractmethod
    def add_new(self, clfr, op, time_step=None):
        raise NotImplementedError

    @abc.abstractmethod
    def remove(self, clfr, op=None):
        raise NotImplementedError

    @abc.abstractmethod
    def gen_match_set_train(self, obs, time_step):
        raise NotImplementedError

    @abc.abstractmethod
    def gen_match_set_test(self, obs):
        raise NotImplementedError


class VanillaPopulation(PopulationABC):
    """Default style population that does not use condition clustering and
    perfoms fully accurate and exhuastive matching."""
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

    def gen_match_set_train(self, obs, time_step=None):
        return self._gen_match_set_exhaustive(obs)

    def gen_match_set_test(self, obs):
        return self._gen_match_set_exhaustive(obs)


def FastApproxMatchingPopulation(PopulationABC):
    def __init__(self):
        # init new (empty) condition clustering
        self._condition_clustering = ConditionClustering()
        self._last_cover_time_step = MIN_TIME_STEP
        super().__init__()

    def add_new(self, clfr, op, time_step):
        # do the addition
        self._clfrs.append(clfr)
        self._num_micros += clfr.numerosity
        assert op in ("covering", "insertion")
        self._ops_history[op] += clfr.numerosity

        # record covering time step if applicable
        if op == "covering":
            self._last_cover_time_step = time_step

        # update clustering if possible
        if not self._condition_clustering.is_empty:
            self._condition_clustering.add_condition(clfr.condition)

    def remove(self, clfr, op=None):
        self._clfrs.remove(clfr)
        self._num_micros -= clfr.numerosity
        if op is not None:
            assert op == "deletion"
            self._ops_history[op] += clfr.numerosity

        # update clustering if possible
        if not self._condition_clustering.is_empty:
            self._condition_clustering.remove_condition(clfr.condition)

    def gen_match_set_train(self, obs, time_step):
        if self._condition_clustering.is_empty:
            self._try_init_condition_clustering(time_step)

        if self._condition_clustering.is_empty:
            # use full exhaustive matching
            return self._gen_match_set_exhaustive(obs)
        else:
            # use fast approx matching
            return self._gen_match_set_approx(obs)

    def _try_init_condition_clustering(self, time_step):
        time_steps_since_last_cover = \
            (time_step - self._last_cover_time_step)
        should_init_clustering = \
            time_steps_since_last_cover > get_hp("theta_cc")
        if should_init_clustering:
            logging.info(f"Initing cond clustering at time step "
                         f"{time_step}")
            macroclfr_conditions = [clfr.condition for clfr in self._clfrs]
            self._condition_clustering.init_clustering(
                macroclfr_conditions)

    def _gen_match_set_approx(self, obs):
        condition_match_map = \
            self._condition_clustering.gen_condition_match_map(obs)
        match_set = []
        # this loop also has the (desired) side effect of validating that the
        # cond. of each macroclfr is contained in the clustering since it
        # performs dict lookup into the cond match map for each one.
        # Thus it is implicitly checking the invariant that the state
        # of the clustering reflects the conds contained in pop at any given
        # time step when matching is executed.
        for clfr in self._clfrs:
            if condition_match_map[clfr.condition]:
                match_set.append(clfr)
        return match_set

    def gen_match_set_test(self, obs):
        if self._condition_clustering.is_empty:
            return self._gen_match_set_exhaustive(obs)
        else:
            return self._gen_match_set_approx(obs)
