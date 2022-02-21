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
        """Build clustering by associating each obj in set unique_conditions -
        medoids to its closest medoid according to dist_matrix."""
        # clustering is a mapping from medoid condition to conditions in its
        # cluster
        clustering = {medoid: set() for medoid in medoids}
        others = (unique_conditions - medoids)
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

        medoids = set(clustering.keys())
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
                for obj_a in cluster:
                    dist_sum = 0
                    for obj_b in cluster:
                        if obj_a != obj_b:
                            dist_sum += dist_matrix[obj_a][obj_b]
                    if min_dist_sum is None or dist_sum < min_dist_sum:
                        min_dist_sum = dist_sum
                        new_medoid = obj_a

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
        pass

    def remove_condition(self, condition):
        pass
