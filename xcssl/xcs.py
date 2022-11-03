import logging
import sys
from collections import OrderedDict

import numpy as np

from .action_selection import (NULL_ACTION, ActionSelectionModes,
                               choose_action_selection_mode,
                               greedy_action_selection)
from .constants import MIN_TIME_STEP
from .covering import calc_num_unique_actions, gen_covering_classifier
from .deletion import deletion
from .ga import run_ga
from .hyperparams import get_hyperparam as get_hp
from .hyperparams import register_hyperparams
from .param_update import update_action_set
from .population import FastApproxMatchingPopulation, VanillaPopulation
from .rng import seed_rng
from .util import calc_num_micros


class XCS:
    def __init__(self, env, encoding, action_selection_strat, hyperparams_dict,
                 use_fast_approx_matching):
        self._env = env
        self._encoding = encoding
        self._action_selection_strat = action_selection_strat
        self._hyperparams_dict = hyperparams_dict
        register_hyperparams(self._hyperparams_dict)
        seed_rng(get_hp("seed"))

        if use_fast_approx_matching:
            self._pop = FastApproxMatchingPopulation()
        else:
            self._pop = VanillaPopulation()

        self._action_selection_mode = None
        self._time_step = MIN_TIME_STEP
        self._num_epochs_done = 0

    @property
    def pop(self):
        return self._pop

    def train_for_epoch(self):
        assert self._env.is_terminal
        self._env.init_epoch()
        while not self._env.is_terminal:
            self._run_step()
        self._num_epochs_done += 1

    def _run_step(self):
        obs = self._env.curr_obs
        match_set = self._gen_match_set_and_cover(obs)
        prediction_arr = self._gen_prediction_arr(match_set, obs)
        action = self._select_action(prediction_arr)
        action_set = self._gen_action_set(match_set, action)
        # single-step only, no previous action sets or discounting
        reward = self._env.step(action)
        payoff = reward
        update_action_set(action_set, payoff, obs, self._pop)
        self._try_run_ga(action_set, self._pop, self._time_step,
                         self._encoding, obs, self._env.action_space)
        self._time_step += 1

    def _gen_match_set_and_cover(self, obs):
        match_set = self._gen_match_set(obs)
        theta_mna = len(self._env.action_space)  # always cover all actions
        while (calc_num_unique_actions(match_set) < theta_mna):
            clfr = gen_covering_classifier(obs, self._encoding, match_set,
                                           self._env.action_space,
                                           self._time_step)
            self._pop.add_new(clfr, op="covering", time_step=self._time_step)
            deletion(self._pop)
            match_set.append(clfr)
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
        # action selection mode chosen at each individual step / example
        self._action_selection_mode = choose_action_selection_mode()
        if self._action_selection_mode == ActionSelectionModes.explore:
            return self._action_selection_strat(prediction_arr,
                                                self._time_step)
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
