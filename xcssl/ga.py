from .classifier import Classifier
from .deletion import deletion
from .hyperparams import get_hyperparam as get_hp
from .rng import get_rng
from .subsumption import does_subsume

_ERROR_CUTDOWN = 0.25
_FITNESS_CUTDOWN = 0.1


def run_ga(action_set, pop, time_step, encoding, obs, action_space):
    for clfr in action_set:
        clfr.time_stamp = time_step

    parent_a = _tournament_selection(action_set)
    parent_b = _tournament_selection(action_set)

    # Build up the children data as dicts while doing crossover + mutation.
    # Only make the actual Classifier objs. for the children after both phases
    # are complete.
    children_data = {
        "a": {
            "cond_alleles": list(parent_a.condition.alleles),
            "action": parent_a.action,
            "prediction": parent_a.prediction,
            "error": parent_a.error,
            "fitness": parent_a.fitness,
            "time_step": time_step,
            "action_set_size": parent_a.action_set_size
        },
        "b": {
            "cond_alleles": list(parent_b.condition.alleles),
            "action": parent_b.action,
            "prediction": parent_b.prediction,
            "error": parent_b.error,
            "fitness": parent_b.fitness,
            "time_step": time_step,
            "action_set_size": parent_b.action_set_size
        },
    }

    do_crossover = get_rng().random() < get_hp("chi")
    if do_crossover:

        child_a_cond_alleles = children_data["a"]["cond_alleles"]
        child_b_cond_alleles = children_data["b"]["cond_alleles"]
        # Crossover child cond alleles *IN-PLACE*
        _uniform_crossover(child_a_cond_alleles, child_b_cond_alleles)

        avg_parent_pred = (parent_a.prediction + parent_b.prediction) / 2
        children_data["a"]["prediction"] = avg_parent_pred
        children_data["b"]["prediction"] = avg_parent_pred

        avg_parent_error = (parent_a.error + parent_b.error) / 2
        children_data["a"]["error"] = avg_parent_error
        children_data["b"]["error"] = avg_parent_error

        avg_parent_fitness = (parent_a.fitness + parent_b.fitness) / 2
        children_data["a"]["fitness"] = avg_parent_fitness
        children_data["b"]["fitness"] = avg_parent_fitness

    for child_label in ("a", "b"):
        child_data = children_data[child_label]

        child_data["error"] *= _ERROR_CUTDOWN
        child_data["fitness"] *= _FITNESS_CUTDOWN

        # Mutate child cond alleles and action *NOT IN-PLACE*, hence
        # assignments
        child_cond_alleles = child_data["cond_alleles"]
        child_cond_alleles = encoding.mutate_condition_alleles(
            child_cond_alleles, obs)
        child_condition = encoding.make_condition(child_cond_alleles)
        del child_data["cond_alleles"]

        child_data["action"] = _mutate_action(child_data["action"],
                                              action_space)

        # Now, actually make the child Classifier obj.
        # (feed the constructor the Condition obj. plus all the stuff in the
        # child_data dict as kwargs)
        child = Classifier.from_ga(condition=child_condition, **child_data)

        if get_hp("do_ga_subsumption"):
            if does_subsume(parent_a, child):
                pop.alter_numerosity(parent_a, delta=1, op="ga_subsumption")
            elif does_subsume(parent_b, child):
                pop.alter_numerosity(parent_b, delta=1, op="ga_subsumption")
            else:
                _insert_in_pop(pop, child)
        else:
            _insert_in_pop(pop, child)
        deletion(pop)


def _tournament_selection(action_set):
    """From Butz book 'Rule Based Evolutionary Online Learning Systems' SELECT
    OFFSPRING function in Appendix B."""
    tau = get_hp("tau")
    best = None

    while best is None:
        max_fitness = 0
        for clfr in action_set:
            if clfr.numerosity_scaled_fitness > max_fitness:
                for _ in range(clfr.numerosity):
                    if get_rng().random() < tau:
                        best = clfr
                        max_fitness = clfr.numerosity_scaled_fitness
                        break
    return best


def _uniform_crossover(child_a_cond_alleles, child_b_cond_alleles):
    """Uniform crossover on children condition allele lists, *in-place*."""
    assert len(child_a_cond_alleles) == len(child_b_cond_alleles)
    n = len(child_a_cond_alleles)

    def _swap(ls_a, ls_b, idx):
        ls_a[idx], ls_b[idx] = ls_b[idx], ls_a[idx]

    upsilon = get_hp("upsilon")
    for idx in range(0, n):
        if get_rng().random() < upsilon:
            _swap(child_a_cond_alleles, child_b_cond_alleles, idx)


def _mutate_action(child_action, action_space):
    if get_rng().random() < get_hp("mu"):
        other_actions = tuple(set(action_space) - {child_action})
        return get_rng().choice(other_actions)
    else:
        return child_action


def _insert_in_pop(pop, child):
    for clfr in pop:
        # check action first to potentially short circuit more expensive
        # condition check
        if (clfr.action == child.action and clfr.condition == child.condition):
            pop.alter_numerosity(clfr, delta=1, op="absorption")
            return
    pop.add_new(child, op="insertion")
