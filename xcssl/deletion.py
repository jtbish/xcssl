from .hyperparams import get_hyperparam as get_hp
from .rng import get_rng

_MIN_NUM_MACROS = 1


def deletion(pop):
    max_pop_size = get_hp("N")
    pop_size = pop.num_micros
    num_to_delete = max(0, (pop_size - max_pop_size))

    if num_to_delete > 0:

        for _ in range(num_to_delete):
            _delete_single_microclfr(pop)

        assert pop.num_macros >= _MIN_NUM_MACROS
        assert pop.num_micros <= max_pop_size


def _delete_single_microclfr(pop):
    avg_fitness_in_pop = sum(clfr.fitness for clfr in pop) / pop.num_micros
    vote_increase_threshold = (get_hp("delta") * avg_fitness_in_pop)

    votes = []
    max_vote = None

    for clfr in pop:
        vote = _deletion_vote(clfr, avg_fitness_in_pop,
                              vote_increase_threshold)
        votes.append(vote)

        if max_vote is None or vote > max_vote:
            max_vote = vote

    assert max_vote is not None

    # roulette wheel selection via stochastic acceptance
    clfr_idx_to_remove = None
    accepted = False

    num_macros = pop.num_macros

    while not accepted:

        idx = get_rng().randint(0, num_macros)
        (clfr, vote) = (pop[idx], votes[idx])  # since pop ordered this is ok
        p_accept = (vote / max_vote)

        if get_rng().random() < p_accept:
            accepted = True

            if clfr.numerosity > 1:
                pop.alter_numerosity(clfr, delta=-1, op="deletion")
            elif clfr.numerosity == 1:
                clfr_idx_to_remove = clfr
            else:
                # not possible
                assert False

    if clfr_idx_to_remove is not None:
        pop.remove(clfr_idx_to_remove, op="deletion")


def _deletion_vote(clfr, avg_fitness_in_pop, vote_increase_threshold):
    vote = clfr.deletion_vote

    if clfr.deletion_has_sufficient_exp:

        scaled_fitness = clfr.numerosity_scaled_fitness
        should_increase_vote = (scaled_fitness < vote_increase_threshold)

        if should_increase_vote:
            vote *= (avg_fitness_in_pop / scaled_fitness)

    return vote
