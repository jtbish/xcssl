import logging

import numpy as np

from .hyperparams import get_hyperparam as get_hp
from .rng import get_rng

_MIN_K = 2
_MIN_DIST = 0


class ConditionClustering:
    """k-medoids clustering of (macro)classifier conditions."""
    def __init__(self):
        k = get_hp("cc_k")
        assert k >= _MIN_K
        self._k = k

        self._unique_conditions = None
        self._dist_matrix = None
        self._clustering = None
        self._cost = None

        self._is_empty = True

    @property
    def is_empty(self):
        return self._is_empty

    def init_clustering(self, macroclfr_conditions):
        assert self._is_empty
        # can have dup. conds since might have two macroclfrs with same
        # cond. but diff. action, only do the clustering on unique ones
        self._unique_conditions = set(macroclfr_conditions)
        self._dist_matrix = self._init_dist_matrix(self._unique_conditions)
        (self._clustering,
         self._cost) = self._init_clustering(self._unique_conditions,
                                             self._dist_matrix, self._k)
        self._is_empty = False

    def _init_dist_matrix(self, unique_conditions):
        m = len(unique_conditions)
        logging.info(f"Num unique conds (m) = {m}")

        dist_matrix = {}
        num_dist_calcs = 0
        for (idx_a, cond_a) in enumerate(unique_conditions):
            dist_matrix[cond_a] = {}
            for (idx_b, cond_b) in enumerate(unique_conditions):
                if idx_a != idx_b:
                    try:
                        # see if already computed for transpose elem
                        dist = dist_matrix[cond_b][cond_a]
                    except KeyError:
                        # not already computed, so do it now
                        dist = cond_a.distance_from(cond_b)
                        num_dist_calcs += 1
                    dist_matrix[cond_a][cond_b] = dist
                else:
                    assert cond_a == cond_b
                    # dist between identical conds on diag.
                    dist_matrix[cond_a][cond_b] = _MIN_DIST

            assert len(dist_matrix[cond_a]) == m

        assert len(dist_matrix) == m
        # should have only computed a single off-diagonal since symmetrical
        expected_num_dist_calcs = sum([i for i in range(1, m)])
        assert num_dist_calcs == expected_num_dist_calcs
        logging.info(f"Inited cond. clustering dist matrix of size "
                     f"{m}*{m} = {m**2} using {num_dist_calcs} dist. calcs")

        return dist_matrix

    def _init_clustering(self, unique_conditions, dist_matrix, k):
        # 0. sample medoids uniformly at random
        assert k <= len(unique_conditions)
        medoids = set(get_rng().choice(list(unique_conditions),
                                       size=k,
                                       replace=False))

        # 1. associate each other condition with closest medoid to build
        # intial clustering
        (clustering,
         cost) = self._assoc_others_to_medoids(medoids, unique_conditions,
                                               dist_matrix)

        # 2. optimise clustering to minimise cost function for current objs
        (clustering, cost) = self._optimise_clustering(clustering, cost,
                                                       dist_matrix,
                                                       unique_conditions)
        return (clustering, cost)

    def _assoc_others_to_medoids(self, medoids, unique_conditions,
                                 dist_matrix):
        """Build clustering by associating each obj in set (unique_conditions -
        medoids) to its closest medoid according to dist_matrix."""
        # clustering is a mapping from medoid condition to conditions in its
        # cluster. Also maintain an "inverse" clustering to make inverse
        # lookups fast
        clustering = {medoid: set() for medoid in medoids}
        others = (unique_conditions - medoids)
        inv_clustering = {other: None for other in others}
        cost = 0

        for other in others:
            medoids_with_dists = {}
            for medoid in medoids:
                medoids_with_dists[medoid] = dist_matrix[other][medoid]
            closest_medoid = min(medoids_with_dists,
                                 key=medoids_with_dists.get)
            dist_to_closest = medoids_with_dists[closest_medoid]
            cost += dist_to_closest
            assert other not in clustering[closest_medoid]
            # assoc other with medoid
            clustering[closest_medoid].add(other)

        # validate clustering: check all objs accounted for
        num_objs_in_clustering = 0
        for medoid in medoids:
            # for medoid itself
            num_objs_in_clustering += 1
            # for others
            num_objs_in_clustering += len(clustering[medoid])
        assert num_objs_in_clustering == len(dist_matrix)

        for (i, medoid) in enumerate(medoids):
            cost_for_medoid = 0
            for other in clustering[medoid]:
                cost_for_medoid += dist_matrix[other][medoid]
            logging.info(f"Medoid {i}: {len(clustering[medoid])} objs, "
                         f"with cost {cost_for_medoid}")

        return (clustering, cost)

    def _optimise_clustering(self, clustering, cost, dist_matrix,
                             unique_conditions):
        """Optimise clustering with given cost via application of k-means style
        Voronoi iteration algorithm."""

        medoids = clustering.keys()
        prev_cost = cost
        converged = False
        iter_num = 0

        while not converged:
            logging.info("\n")

            # a) in each cluster, make the point that minimises the sum of
            # distances within the cluster the new medoid
            new_medoids = set()
            for medoid in medoids:
                cluster = {medoid}.union(clustering[medoid])

                min_dist_sum = None
                new_medoid = None
                for (idx_a, cond_a) in enumerate(cluster):
                    dist_sum = 0
                    for (idx_b, cond_b) in enumerate(cluster):
                        if idx_a != idx_b:  # idx equality is fast
                            dist_sum += dist_matrix[cond_a][cond_b]
                    if min_dist_sum is None or dist_sum < min_dist_sum:
                        min_dist_sum = dist_sum
                        new_medoid = cond_a

                assert min_dist_sum is not None
                assert new_medoid is not None
                if new_medoid != medoid:
                    logging.info("Medoid changed")
                new_medoids.add(new_medoid)

            assert len(new_medoids) == len(medoids)

            # b) update clustering by doing assoc. step with new medoids
            medoids = new_medoids
            (clustering, curr_cost) = \
                self._assoc_others_to_medoids(medoids, unique_conditions,
                                              dist_matrix)

            iter_num += 1
            logging.info(f"Iter {iter_num}: prev cost = {prev_cost}, "
                         f"curr cost = {curr_cost}")
            cost_decreased = curr_cost < prev_cost
            converged = not cost_decreased
            prev_cost = curr_cost

        return (clustering, curr_cost)

    def gen_condition_match_map(self, obs):
        condition_match_map = {}
        # match each medoid, and others in its cluster inherit its matching
        # status
        for medoid in self._clustering.keys():
            does_match = medoid.does_match(obs)
            condition_match_map[medoid] = does_match
            for other in self._clustering[medoid]:
                condition_match_map[other] = does_match

        return condition_match_map

    def add_condition(self, condition):
        # condition being added *may not* be unique. If it's a dup, there is
        # nothing to do.
        is_dup = (condition in self._unique_conditions)
        if not is_dup:
            # update the dist matrix
            self._dist_matrix = self._add_new_to_dist_matrix(
                self._dist_matrix, self._unique_conditions, condition)
            # add it to unique set *afterwards*
            self._unique_conditions.add(condition)
            # do single assoc
            (self._clustering, self._cost) = \
                self._update_clustering_new_condition(self._clustering,
                                                      self._cost,
                                                      self._dist_matrix,
                                                      condition)
            # iterate till converge
            (self._clustering, self._cost) = \
                self._optimise_clustering(self._clustering, self._cost,
                                          self._dist_matrix,
                                          self._unique_conditions)

    def _add_new_to_dist_matrix(self, dist_matrix, unique_conditions,
                                condition):
        self._dist_matrix[condition] = {}
        # dist for self
        self._dist_matrix[condition][condition] = _MIN_DIST
        # dist for others (symmetrical)
        for other in unique_conditions:
            dist = condition.distance_from(other)
            self._dist_matrix[condition][other] = dist
            self._dist_matrix[other][condition] = dist

        return self._dist_matrix

    def _update_clustering_new_condition(self, clustering, cost, dist_matrix,
                                         condition):
        # Update the clustering, i.e. do an assoc step but only with
        # the single new condition
        medoids = clustering.keys()
        min_dist = None
        assoc_medoid = None
        for medoid in medoids:
            dist = dist_matrix[condition][medoid]
            if min_dist is None or dist < min_dist:
                min_dist = dist
                assoc_medoid = medoid
        assert min_dist is not None
        assert assoc_medoid is not None

        assert condition not in clustering[assoc_medoid]
        clustering[assoc_medoid].add(condition)
        cost += min_dist

        return (clustering, cost)

    def remove_condition(self, condition):
        assert condition in self._unique_conditions

        # 0. check if condition is a medoid
        medoids = set(self._clustering.keys())
        if condition in medoids:
            # TODO deal with this later
            raise Exception

        # 1. it's not a medoid, so figure out which medoid it belongs to
        # remove it from the clustering and update the cost
        assoc_medoid = None
        for medoid in medoids:
            if condition in self._clustering[medoid]:
                assoc_medoid = medoid
        assert assoc_medoid is not None
        dist_to_assoc_medoid = self._dist_matrix[condition][assoc_medoid]
        self._clustering[assoc_medoid].remove(condition)
        self._cost -= dist_to_assoc_medoid

        # 2. remove from dist matrix
        self._dist_matrix = \
            self._remove_existing_from_dist_matrix(self._dist_matrix,
                                                   self._unique_conditions,
                                                   condition)
        # 3. remove from unique conditions
        self._unique_conditions.remove(condition)

        # 4. iterate till converge
        (self._clustering, self._cost) = \
            self._optimise_clustering(self._clustering, self._cost,
                                      self._dist_matrix,
                                      self._unique_conditions)

    def _remove_existing_from_dist_matrix(self, dist_matrix, unique_conditions,
                                          condition):
        # delete own row
        del (dist_matrix[condition])
        # delete column in other rows
        others = (unique_conditions - {condition})
        for other in others:
            del (dist_matrix[other][condition])

        return dist_matrix
