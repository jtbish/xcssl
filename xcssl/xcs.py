import logging
from collections import OrderedDict, deque
from enum import Enum

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
from .population import FastMatchingPopulation, VanillaPopulation
from .rng import seed_rng
from .util import calc_num_micros

MatchingModes = Enum("MatchingModes", ["full", "fast"])

_EXPLOIT_CORRECT_HISTORY_MAXLEN = 100


class XCS:
    def __init__(self, env, encoding, hyperparams_dict, use_fm):

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

        # always starts out with vanilla pop, switches later if using FM
        self._pop = VanillaPopulation()
        self._use_fm = use_fm
        self._match_mode = MatchingModes.full
        self._last_cover_time_step = MIN_TIME_STEP

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
        prediction_arr = self._gen_prediction_arr(match_set, obs)
        action = self._select_action(prediction_arr)
        action_set = self._gen_action_set(match_set, action)

        # single-step only, no previous action sets or discounting
        env_response = self._env.step(action)

        if self._action_selection_mode == ActionSelectionModes.exploit:
            self._exploit_correct_history.append(env_response.correct)

        payoff = env_response.reward

        update_action_set(action_set, payoff, obs, self._pop)
        self._try_run_ga(action_set, self._pop, self._time_step,
                         self._encoding, obs, self._env.action_space)
        self._time_step += 1

        self._try_switch_match_mode()

    def _gen_match_set_and_cover(self, obs):
        match_set = self._gen_match_set(obs)
        # always cover all actions
        theta_mna = len(self._env.action_space)
        while (calc_num_unique_actions(match_set) < theta_mna):
            clfr = gen_covering_classifier(obs, self._encoding, match_set,
                                           self._env.action_space,
                                           self._time_step)
            self._pop.add_new(clfr, op="covering", time_step=self._time_step)
            deletion(self._pop)
            match_set.append(clfr)
            self._last_cover_time_step = self._time_step

        return match_set

    def _gen_match_set(self, obs):
        return self._pop.gen_match_set(obs)

    def _gen_prediction_arr(self, match_set, obs):
        prediction_arr = OrderedDict(
            {action: None
             for action in self._env.action_space})
        actions_reprd_in_m = set([clfr.action for clfr in match_set])
        for a in actions_reprd_in_m:
            # to bootstap sum below
            prediction_arr[a] = 0

        fitness_sum_arr = OrderedDict(
            {action: 0
             for action in self._env.action_space})

        for clfr in match_set:
            a = clfr.action
            prediction_arr[a] += (clfr.prediction * clfr.fitness)
            fitness_sum_arr[a] += clfr.fitness

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

            avg_time_stamp_in_as = sum(
                [clfr.time_stamp * clfr.numerosity
                 for clfr in action_set]) / calc_num_micros(action_set)

            should_apply_ga = ((time_step - avg_time_stamp_in_as) >
                               get_hp("theta_ga"))

            if should_apply_ga:
                run_ga(action_set, pop, time_step, encoding, obs, action_space)

    def _try_switch_match_mode(self):
        if (self._use_fm and self._match_mode == MatchingModes.full):

            time_steps_since_last_cover = (self._time_step -
                                           self._last_cover_time_step)

            should_switch_mode = (time_steps_since_last_cover >
                                  get_hp("theta_fm"))
            if should_switch_mode:

                self._match_mode = MatchingModes.fast
                # all phenotypes generated by the encoding should be vectorised
                # from now on
                self._encoding.enable_phenotype_vectorisation()
                lsh = self._encoding.make_lsh()

                # replace vanilla pop with FM pop
                self._pop = FastMatchingPopulation(vanilla_pop=self._pop,
                                                   encoding=self._encoding,
                                                   lsh=lsh)

    def select_action(self, obs):
        """Action selection for outside testing - always exploit"""
        match_set = self._gen_match_set(obs)
        if len(match_set) > 0:
            prediction_arr = self._gen_prediction_arr(match_set, obs)
            return greedy_action_selection(prediction_arr)
        else:
            return NULL_ACTION

    def gen_prediction_arr(self, obs):
        """For outside testing."""
        match_set = self._gen_match_set(obs)
        return self._gen_prediction_arr(match_set, obs)
