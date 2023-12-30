import copy

from .hyperparams import get_hyperparam as get_hp


def action_set_subsumption(action_set, pop):
    raise NotImplementedError


def does_subsume(subsumer, subsumee):
    """Determines if subsumer clfr really does subsume subsumee clfr."""
    return (could_subsume(subsumer) and subsumer.action == subsumee.action
            and subsumer.does_subsume(subsumee))


def could_subsume(clfr):
    return (clfr.experience > get_hp("theta_sub")
            and clfr.error < get_hp("epsilon_nought"))
