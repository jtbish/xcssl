import numpy as np

from .hyperparams import get_hyperparam as get_hp
from .subsumption import action_set_subsumption
from .util import calc_num_micros

np.seterr(divide="raise", over="raise", invalid="raise")

_MAX_ACC = 1.0


def update_action_set(action_set, payoff, obs, pop):
    as_num_micros = calc_num_micros(action_set)
    for clfr in action_set:
        _update_experience(clfr)
        payoff_diff = (payoff - clfr.prediction)
        _update_error(clfr, payoff_diff)
        _update_prediction(clfr, payoff_diff)
        _update_action_set_size(clfr, as_num_micros)
    _update_fitness(action_set)

    if get_hp("do_as_subsumption"):
        action_set_subsumption(action_set, pop)


def _update_experience(clfr):
    clfr.experience += 1


def _update_error(clfr, payoff_diff):
    beta = get_hp("beta")
    error_target = (abs(payoff_diff) - clfr.error)
    if clfr.experience < (1 / beta):
        clfr.error += (error_target / clfr.experience)
    else:
        clfr.error += (beta * error_target)


def _update_prediction(clfr, payoff_diff):
    beta = get_hp("beta")
    if clfr.experience < (1 / beta):
        clfr.prediction += (payoff_diff / clfr.experience)
    else:
        clfr.prediction += (beta * payoff_diff)


def _update_action_set_size(clfr, as_num_micros):
    beta = get_hp("beta")
    as_size_diff = (as_num_micros - clfr.action_set_size)
    if clfr.experience < (1 / beta):
        clfr.action_set_size += (as_size_diff / clfr.experience)
    else:
        clfr.action_set_size += (beta * as_size_diff)


def _update_fitness(action_set):
    acc_sum = 0
    acc_vec = []
    e_nought = get_hp("epsilon_nought")
    for clfr in action_set:
        if clfr.error < e_nought:
            acc = _MAX_ACC
        else:
            acc = (get_hp("alpha") *
                   (clfr.error / e_nought)**(-1 * get_hp("nu")))
        acc_vec.append(acc)
        acc_sum += (acc * clfr.numerosity)

    for (clfr, acc) in zip(action_set, acc_vec):
        relative_acc = (acc * clfr.numerosity / acc_sum)
        clfr.fitness += (get_hp("beta") * (relative_acc - clfr.fitness))
