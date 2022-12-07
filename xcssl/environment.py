import functools
from collections import namedtuple

import numpy as np
import pandas as pd

_DEFAULT_SEED = 0

REWARD_INCORRECT = 0
REWARD_CORRECT = 1000

EnvironmentResponse = namedtuple("EnvironmentResponse", ["reward", "correct"])
StreamDataInstance = namedtuple("StreamDataInstance", ["obs", "label"])


def check_terminal(public_method):
    """Decorator to check if environment is terminal before performing
    operations on it."""
    @functools.wraps(public_method)
    def decorator(self, *args, **kwargs):
        if self.is_terminal:
            assert self._curr_obs_idx is None
            raise OutOfDataError("Environment is out of data (epoch is "
                                 "finished). Call env.init_epoch() to "
                                 "reinitialise for next epoch.")
        else:
            assert self._curr_obs_idx is not None
        return public_method(self, *args, **kwargs)

    return decorator


class OutOfDataError(Exception):
    """Indicates need to call init_epoch() again, occuring either immediately
    after environment obj. init, or after epoch has elapsed."""
    pass


def _validate_dataset(dataset):
    assert isinstance(dataset, pd.DataFrame)
    assert len(dataset.columns) == 2
    assert "obs" in dataset.columns
    assert "label" in dataset.columns


class ClassificationEpochEnvironment:
    def __init__(self, obs_space, action_space, dataset, shuffle_seed=None):
        self._obs_space = obs_space
        self._action_space = action_space

        _validate_dataset(dataset)
        self._dataset = dataset

        self._curr_obs_idx = None
        self._is_terminal = True

        if shuffle_seed is None:
            shuffle_seed = _DEFAULT_SEED

        self._shuffle_rng = np.random.RandomState(seed=int(shuffle_seed))
        self._num_examples = len(dataset)
        self._epoch_idx_iter = None

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def dataset(self):
        return self._dataset

    @property
    @check_terminal
    def curr_obs(self):
        return self._dataset["obs"][self._curr_obs_idx]

    @property
    def is_terminal(self):
        return self._is_terminal

    def init_epoch(self):
        self._is_terminal = False
        epoch_idx_order = self._shuffle_rng.permutation(
            range(0, self._num_examples))
        self._epoch_idx_iter = iter(epoch_idx_order)
        self._curr_obs_idx = self._get_next_obs_idx()

    @check_terminal
    def step(self, action):
        assert action in self._action_space

        # calc reward
        label = self._dataset["label"][self._curr_obs_idx]
        correct = (action == label)

        if correct:
            reward = REWARD_CORRECT
        else:
            reward = REWARD_INCORRECT

        # update obs idx
        self._curr_obs_idx = self._get_next_obs_idx()

        return EnvironmentResponse(reward, correct)

    def _get_next_obs_idx(self):
        try:
            return next(self._epoch_idx_iter)
        except StopIteration:
            self._is_terminal = True
            return None


class ClassificationStreamEnvironment:
    OpModes = namedtuple("OpModes", ["dataset", "gen"])

    def __init__(self,
                 obs_space,
                 action_space,
                 dataset=None,
                 instance_gen_func=None,
                 seed=None):

        self._obs_space = obs_space
        self._action_space = action_space

        # for stream env, either a dataset is provided, or an instance gen func
        # is provided, but not both!
        assert (dataset is not None and instance_gen_func is None) or \
               (dataset is None and instance_gen_func is not None)

        if dataset is not None:

            self._op_mode = self.OpModes.dataset
            _validate_dataset(dataset)
            self._dataset = dataset
            self._instance_gen_func = None

        elif instance_gen_func is not None:

            self._op_mode = self.OpModes.gen
            self._dataset = None
            self._instance_gen_func = instance_gen_func

        if seed is None:
            seed = _DEFAULT_SEED

        self._rng = np.random.RandomState(seed=int(seed))

        self._curr_data_instance = self._get_data_instance(
            self._op_mode, self._dataset, self._instance_gen_func, self._rng)

    @classmethod
    def from_dataset(cls, obs_space, action_space, dataset, seed=None):
        return cls(obs_space,
                   action_space,
                   dataset=dataset,
                   instance_gen_func=None,
                   seed=seed)

    @classmethod
    def from_instance_gen_func(cls,
                               obs_space,
                               action_space,
                               instance_gen_func,
                               seed=None):
        return cls(obs_space,
                   action_space,
                   dataset=None,
                   instance_gen_func=instance_gen_func,
                   seed=seed)

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def curr_obs(self):
        return self._curr_data_instance.obs

    def _get_data_instance(self, op_mode, dataset, instance_gen_func, rng):
        if op_mode == self.OpModes.dataset:

            # pick a random row from the dataset
            idx = rng.randint(low=0, high=len(dataset))
            return StreamDataInstance(obs=dataset["obs"][idx],
                                      label=dataset["label"][idx])

        elif op_mode == self.OpModes.gen:
            return instance_gen_func(rng)
        else:
            assert False

    def step(self, action):
        assert action in self._action_space

        # calc reward
        label = self._curr_data_instance.label
        correct = (action == label)

        if correct:
            reward = REWARD_CORRECT
        else:
            reward = REWARD_INCORRECT

        self._curr_data_instance = self._get_data_instance(
            self._op_mode, self._dataset, self._instance_gen_func, self._rng)

        return EnvironmentResponse(reward, correct)
