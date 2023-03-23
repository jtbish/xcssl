import numpy as np

from .hyperparams import get_hyperparam as get_hp
from .subsumption import action_set_subsumption
from .util import calc_num_micros

np.seterr(divide="raise", over="raise", invalid="raise")

_MAX_ACC = 1.0


def update_action_set(action_set, payoff, pop):
    as_num_micros = calc_num_micros(action_set)

    beta = get_hp("beta")
    beta_inv = (1 / beta)

    for clfr in action_set:
        _update_experience(clfr)

        payoff_diff = (payoff - clfr.prediction)

        _update_error(clfr, payoff_diff, beta, beta_inv)
        _update_prediction(clfr, payoff_diff, beta, beta_inv)
        _update_action_set_size(clfr, as_num_micros, beta, beta_inv)

    _update_fitness(action_set, beta)

    if get_hp("do_as_subsumption"):
        action_set_subsumption(action_set, pop)


def _update_experience(clfr):
    clfr.experience += 1


def _update_error(clfr, payoff_diff, beta, beta_inv):
    error_target = (abs(payoff_diff) - clfr.error)

    if clfr.experience < beta_inv:
        clfr.error += (error_target / clfr.experience)
    else:
        clfr.error += (beta * error_target)


def _update_prediction(clfr, payoff_diff, beta, beta_inv):
    if clfr.experience < beta_inv:
        clfr.prediction += (payoff_diff / clfr.experience)
    else:
        clfr.prediction += (beta * payoff_diff)


def _update_action_set_size(clfr, as_num_micros, beta, beta_inv):
    as_size_diff = (as_num_micros - clfr.action_set_size)

    if clfr.experience < beta_inv:
        clfr.action_set_size += (as_size_diff / clfr.experience)
    else:
        clfr.action_set_size += (beta * as_size_diff)


def _update_fitness(action_set, beta):
    e_nought = get_hp("epsilon_nought")
    alpha = get_hp("alpha")
    neg_nu = (-1 * get_hp("nu"))

    acc_sum = 0
    acc_vec = []

    for clfr in action_set:
        if clfr.error < e_nought:
            acc = _MAX_ACC
        else:
            acc = (alpha * (clfr.error / e_nought)**(neg_nu))
        acc_vec.append(acc)
        acc_sum += (acc * clfr.numerosity)

    for (clfr, acc) in zip(action_set, acc_vec):
        relative_acc = (acc * clfr.numerosity / acc_sum)
        clfr.fitness += (beta * (relative_acc - clfr.fitness))
