from collections import OrderedDict
from enum import Enum

from .hyperparams import get_hyperparam as get_hp
from .rng import get_rng

NULL_ACTION = -1

ActionSelectionModes = Enum("ActionSelectionModes", ["explore", "exploit"])


def choose_action_selection_mode():
    if get_rng().random() < get_hp("p_exp"):
        return ActionSelectionModes.explore
    else:
        return ActionSelectionModes.exploit


def random_action_selection(prediction_arr):
    # random action from non-null actions reprd in pred arr
    prediction_arr = filter_null_prediction_arr_entries(prediction_arr)
    return get_rng().choice(list(prediction_arr.keys()))


def greedy_action_selection(prediction_arr):
    prediction_arr = filter_null_prediction_arr_entries(prediction_arr)
    return max(prediction_arr, key=prediction_arr.get)


def filter_null_prediction_arr_entries(prediction_arr):
    return OrderedDict(
        {a: p
         for (a, p) in prediction_arr.items() if p is not None})
