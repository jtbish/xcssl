import abc
import functools

import numpy as np
import pandas as pd

_DEFAULT_SHUFFLE_SEED = 0

REWARD_INCORRECT = 0
REWARD_CORRECT = 1000


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


class ClassificationEnvironmentBase(metaclass=abc.ABCMeta):
    """Supervised learning, classification environment base class."""
    def __init__(self, dataset, obs_space, action_space, shuffle_seed=None):
        self._validate_dataset(dataset)
        self._dataset = dataset
        self._obs_space = obs_space
        self._action_space = action_space

        self._curr_obs_idx = None
        self._is_terminal = True

        if shuffle_seed is None:
            shuffle_seed = _DEFAULT_SHUFFLE_SEED

        self._shuffle_rng = np.random.RandomState(seed=int(shuffle_seed))
        self._num_examples = len(dataset)
        self._epoch_idx_iter = None

    def _validate_dataset(self, dataset):
        assert isinstance(dataset, pd.DataFrame)
        assert len(dataset.columns) == 2
        assert "examples" in dataset.columns
        assert "labels" in dataset.columns

    @property
    def dataset(self):
        return self._dataset

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    @property
    @check_terminal
    def curr_obs(self):
        return self._dataset["examples"][self._curr_obs_idx]

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
        label = self._dataset["labels"][self._curr_obs_idx]
        correct = (action == label)
        if correct:
            reward = REWARD_CORRECT
        else:
            reward = REWARD_INCORRECT

        # update obs idx
        self._curr_obs_idx = self._get_next_obs_idx()

        return reward

    def _get_next_obs_idx(self):
        try:
            return next(self._epoch_idx_iter)
        except StopIteration:
            self._is_terminal = True
            return None
