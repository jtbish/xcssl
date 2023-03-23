from collections import OrderedDict, deque

from .action_selection import (NULL_ACTION, ActionSelectionModes,
                               choose_action_selection_mode,
                               greedy_action_selection,
                               random_action_selection)
from .constants import MIN_TIME_STEP
from .covering import calc_num_unique_actions, gen_covering_classifier
from .deletion import deletion
from .environment import ClassificationStreamEnvironment
from .ga import run_ga
from .hyperparams import get_hyperparam as get_hp
from .hyperparams import register_hyperparams
from .param_update import update_action_set
from .population import make_fm_population, make_vanilla_population
from .rng import seed_rng

_EXPLOIT_CORRECT_HISTORY_MAXLEN = 100


class XCS:
    def __init__(self,
                 env,
                 encoding,
                 hyperparams_dict,
                 use_fm=False,
                 do_pop_timing=False):

        assert isinstance(env, ClassificationStreamEnvironment)
        self._env = env
        self._encoding = encoding
        self._hyperparams_dict = hyperparams_dict
        register_hyperparams(self._hyperparams_dict)
        seed_rng(get_hp("seed"))

        self._action_selection_mode = None
        self._time_step = MIN_TIME_STEP
        self._num_steps_done = 0
        self._exploit_correct_history = \
            deque(maxlen=_EXPLOIT_CORRECT_HISTORY_MAXLEN)

        # always cover all actions
        self._theta_mna = len(self._env.action_space)

        if not use_fm:
            self._pop = make_vanilla_population(do_pop_timing)

        else:
            # collect rasterizer kwargs for index
            seed = get_hp("seed")
            rngd = get_hp("rstr_num_grid_dims")
            try:
                rnbpgd = get_hp("rstr_num_bins_per_grid_dim")
            except KeyError:
                rnbpgd = None

            rasterizer_kwargs = {
                "seed": seed,
                "num_grid_dims": rngd,
                "num_bins_per_grid_dim": rnbpgd
            }

            self._pop = make_fm_population(self._encoding, rasterizer_kwargs,
                                           do_pop_timing)

    @property
    def pop(self):
        return self._pop

    def calc_exploit_perf(self):
        if len(self._exploit_correct_history) < \
                _EXPLOIT_CORRECT_HISTORY_MAXLEN:
            raise ValueError("Not enough data")
        else:
            return list(self._exploit_correct_history).count(True) / \
                _EXPLOIT_CORRECT_HISTORY_MAXLEN

    def train_for_steps(self, num_steps):
        for _ in range(num_steps):
            self._run_step()
        self._num_steps_done += num_steps

    def _run_step(self):
        obs = self._env.curr_obs
        match_set = self._gen_match_set_and_cover(obs)
        prediction_arr = self._gen_prediction_arr(match_set)
        action = self._select_action(prediction_arr)
        action_set = self._gen_action_set(match_set, action)

        # single-step only, no previous action sets or discounting
        env_response = self._env.step(action)

        if self._action_selection_mode == ActionSelectionModes.exploit:
            self._exploit_correct_history.append(env_response.correct)

        payoff = env_response.reward

        update_action_set(action_set, payoff, self._pop)
        self._try_run_ga(action_set, self._pop, self._time_step,
                         self._encoding, obs, self._env.action_space)
        self._time_step += 1

    def _gen_match_set_and_cover(self, obs):
        match_set = self._gen_match_set(obs)

        while (calc_num_unique_actions(match_set) < self._theta_mna):
            clfr = gen_covering_classifier(obs, self._encoding, match_set,
                                           self._env.action_space,
                                           self._time_step)
            self._pop.add_new(clfr, op="covering")
            deletion(self._pop)
            match_set.append(clfr)

        return match_set

    def _gen_match_set(self, obs):
        return self._pop.gen_match_set(obs)

    def _gen_prediction_arr(self, match_set):

        prediction_arr = OrderedDict(
            {action: None
             for action in self._env.action_space})

        actions_reprd_in_m = set(clfr.action for clfr in match_set)
        for a in actions_reprd_in_m:
            # to bootstap sum below
            prediction_arr[a] = 0

        fitness_sum_arr = OrderedDict(
            {action: 0
             for action in self._env.action_space})

        for clfr in match_set:
            a = clfr.action
            f = clfr.fitness
            prediction_arr[a] += (clfr.prediction * f)
            fitness_sum_arr[a] += f

        for a in self._env.action_space:
            if fitness_sum_arr[a] != 0:
                prediction_arr[a] /= fitness_sum_arr[a]

        return prediction_arr

    def _select_action(self, prediction_arr):
        # action selection mode chosen at each individual time step / example
        self._action_selection_mode = choose_action_selection_mode()

        if self._action_selection_mode == ActionSelectionModes.explore:
            return random_action_selection(prediction_arr)
        elif self._action_selection_mode == ActionSelectionModes.exploit:
            return greedy_action_selection(prediction_arr)
        else:
            assert False

    def _gen_action_set(self, match_set, action):
        return [clfr for clfr in match_set if clfr.action == action]

    def _try_run_ga(self, action_set, pop, time_step, encoding, obs,
                    action_space):
        # GA can only be active on exploration steps
        if self._action_selection_mode == ActionSelectionModes.explore:

            avg_time_stamp_in_as = self._calc_avg_time_stamp_in_as(action_set)
            should_apply_ga = ((time_step - avg_time_stamp_in_as) >
                               get_hp("theta_ga"))

            if should_apply_ga:
                run_ga(action_set, pop, time_step, encoding, obs, action_space)

    def _calc_avg_time_stamp_in_as(self, action_set):
        numer = 0
        denom = 0

        for clfr in action_set:
            n = clfr.numerosity
            numer += (clfr.time_stamp * n)
            denom += n

        return (numer / denom)

    def select_action(self, obs):
        """Action selection for outside testing - always exploit"""
        match_set = self._gen_match_set(obs)
        if len(match_set) > 0:
            prediction_arr = self._gen_prediction_arr(match_set)
            return greedy_action_selection(prediction_arr)
        else:
            return NULL_ACTION

    def gen_prediction_arr(self, obs):
        """For outside testing."""
        match_set = self._gen_match_set(obs)
        return self._gen_prediction_arr(match_set)
