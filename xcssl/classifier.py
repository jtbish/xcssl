import numpy as np

from .hyperparams import get_hyperparam as get_hp

np.seterr(divide="raise", over="raise", invalid="raise")

_EXPERIENCE_MIN = 0
_ACTION_SET_SIZE_MIN = 1
_NUMEROSITY_MIN = 1
_TIME_STAMP_MIN = 0
_ATTR_EQ_REL_TOL = 1e-10


class Classifier:
    def __init__(self, condition, action, time_step):
        """Only used by covering."""
        self._condition = condition
        self._action = action
        self._prediction = get_hp("pred_I")
        self._error = get_hp("epsilon_I")
        self._fitness = get_hp("fitness_I")
        self._experience = 0
        self._time_stamp = time_step
        self._action_set_size = 1
        self._numerosity = 1

        # "reactive"/calculated params for deletion
        self._deletion_vote = self._calc_deletion_vote(self._action_set_size,
                                                       self._numerosity)
        self._deletion_has_sufficient_exp = \
            self._calc_deletion_has_sufficient_exp(self._experience)
        self._numerosity_scaled_fitness = \
            self._calc_numerosity_scaled_fitness(self._fitness,
                                                 self._numerosity)

    @property
    def condition(self):
        return self._condition

    @condition.setter
    def condition(self, val):
        self._condition = val

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, val):
        self._action = val

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self, val):
        self._prediction = val

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, val):
        self._error = val

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, val):
        self._fitness = val
        self._numerosity_scaled_fitness = \
            self._calc_numerosity_scaled_fitness(self._fitness,
                                                 self._numerosity)

    @property
    def experience(self):
        return self._experience

    @experience.setter
    def experience(self, val):
        assert val >= _EXPERIENCE_MIN
        self._experience = val
        self._deletion_has_sufficient_exp = \
            self._calc_deletion_has_sufficient_exp(self._experience)

    @property
    def time_stamp(self):
        return self._time_stamp

    @time_stamp.setter
    def time_stamp(self, val):
        assert val >= _TIME_STAMP_MIN
        self._time_stamp = val

    @property
    def action_set_size(self):
        return self._action_set_size

    @action_set_size.setter
    def action_set_size(self, val):
        assert val >= _ACTION_SET_SIZE_MIN
        self._action_set_size = val
        self._deletion_vote = self._calc_deletion_vote(self._action_set_size,
                                                       self._numerosity)

    @property
    def numerosity(self):
        return self._numerosity

    @numerosity.setter
    def numerosity(self, val):
        assert val >= _NUMEROSITY_MIN
        self._numerosity = val
        self._deletion_vote = self._calc_deletion_vote(self._action_set_size,
                                                       self._numerosity)
        self._numerosity_scaled_fitness = \
            self._calc_numerosity_scaled_fitness(self._fitness,
                                                 self._numerosity)

    @property
    def deletion_vote(self):
        return self._deletion_vote

    @property
    def deletion_has_sufficient_exp(self):
        return self._deletion_has_sufficient_exp

    @property
    def numerosity_scaled_fitness(self):
        return self._numerosity_scaled_fitness

    def _calc_deletion_vote(self, action_set_size, numerosity):
        return action_set_size * numerosity

    def _calc_deletion_has_sufficient_exp(self, experience):
        return experience > get_hp("theta_del")

    def _calc_numerosity_scaled_fitness(self, fitness, numerosity):
        return fitness / numerosity

    def does_match(self, obs):
        return self._condition.does_match(obs)

    def is_more_general(self, other):
        return (self._condition.generality > other._condition.generality) \
                and self.does_subsume(other)

    def does_subsume(self, other):
        return self._condition.does_subsume(other._condition)

    def __eq__(self, other):
        # Fast version of eq: (condition, action) pair must be unique for all
        # macroclassifiers. Sufficient for removal checks.
        # Check action first since might short circuit and less expensive than
        # cond. check.
        return (self._action == other._action) and (self._condition
                                                    == other._condition)

    def __hash__(self):
        return hash((self._condition, self._action))

    def full_eq(self, other):
        # full version of eq: check all non-calculated params
        return (self._condition == other._condition
                and self._action == other._action and np.isclose(
                    self._prediction, other._prediction, rtol=_ATTR_EQ_REL_TOL)
                and np.isclose(
                    self._error, other._error, rtol=_ATTR_EQ_REL_TOL)
                and np.isclose(
                    self._fitness, other._fitness, rtol=_ATTR_EQ_REL_TOL)
                and self._experience == other._experience
                and self._time_stamp == other._time_stamp
                and np.isclose(self._action_set_size,
                               other._action_set_size,
                               rtol=_ATTR_EQ_REL_TOL)
                and self._numerosity == other._numerosity)
