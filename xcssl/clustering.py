import logging

import numpy as np

from .rng import get_rng

_MIN_K = 2
_MIN_DIST = 0


class ConditionClustering:
    """k-medoids clustering of (macro)classifier conditions."""
    def __init__(self, conditions, k):
        assert k >= _MIN_K
        self._k = k
        # can have dup. conds since might have two macroclfrs with same
        # cond. but diff. action
        self._unique_conditions = set(conditions)
        self._dist_matrix = self._init_dist_matrix(self._unique_conditions)
        self._clustering = self._init_clustering(self._unique_conditions,
                                                 self._dist_matrix, self._k)

    def _init_dist_matrix(self, unique_conditions):
        m = len(unique_conditions)
        logging.info(f"Num uniques conds (m) = {m}")

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
                        dist = cond_a.distance_between(cond_b)
                        num_dist_calcs += 1
                    dist_matrix[cond_a][cond_b] = dist
                else:
                    assert cond_a == cond_b
                    # dist between identical conds on diag.
                    dist_matrix[cond_a][cond_b] = _MIN_DIST

            assert len(dist_matrix[cond_a]) == m

        assert len(dist_matrix) == m
        # single off-diagonal
        expected_num_dist_calcs = sum([i for i in range(1, m)])
        assert num_dist_calcs == expected_num_dist_calcs
        logging.info(f"Inited cond. clustering dist matrix of size "
                     f"{m}*{m} = {m**2} using {num_dist_calcs} dist. calcs")

        return dist_matrix

    def _init_clustering(self, unique_conditions, dist_matrix, k):
        # clusetring is a mapping from medoid condition to conditions in its
        # cluster

        # 0. sample medoids and init clustering
        assert k <= len(unique_conditions)
        medoids = set(get_rng().choice(list(unique_conditions),
                                       size=k,
                                       replace=False))
        clustering = {medoid: set() for medoid in medoids}

        # 1. associate each other condition with closest medoid
        others = (unique_conditions - medoids)
        curr_cost = 0
        for other in others:
            medoids_with_dists = {}
            for medoid in medoids:
                medoids_with_dists[medoid] = dist_matrix[other][medoid]
            closest_medoid = min(medoids_with_dists,
                                 key=medoids_with_dists.get)
            dist_to_closest = medoids_with_dists[closest_medoid]
            curr_cost += dist_to_closest
            assert other not in clustering[closest_medoid]
            clustering[closest_medoid].add(other)

        for (i, medoid) in enumerate(medoids):
            cost_for_medoid = 0
            for other in clustering[medoid]:
                cost_for_medoid += dist_matrix[other][medoid]
            logging.info(f"Medoid {i}: {len(clustering[medoid])} objs, "
                         f"with cost {cost_for_medoid}")

        prev_cost = curr_cost
        converged = False
        iter_num = 0
        while not converged:
            logging.info("\n")
            # 2. in each cluster, make the point that minimises the sum of
            # distances within the cluster the new medoid: "Voronoi iteration"
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

            # 3. update clustering by repeating step 1 after updating medoids
            medoids = new_medoids
            clustering = {medoid: set() for medoid in medoids}
            others = (unique_conditions - medoids)
            curr_cost = 0
            for other in others:
                medoids_with_dists = {}
                for medoid in medoids:
                    medoids_with_dists[medoid] = dist_matrix[other][medoid]
                closest_medoid = min(medoids_with_dists,
                                     key=medoids_with_dists.get)
                dist_to_closest = medoids_with_dists[closest_medoid]
                curr_cost += dist_to_closest
                assert other not in clustering[closest_medoid]
                clustering[closest_medoid].add(other)

            for (i, medoid) in enumerate(medoids):
                cost_for_medoid = 0
                for other in clustering[medoid]:
                    cost_for_medoid += dist_matrix[other][medoid]
                logging.info(f"Medoid {i}: {len(clustering[medoid])} objs, "
                             f"with cost {cost_for_medoid}")

            iter_num += 1
            logging.info(
                f"Iter {iter_num}: prev cost = {prev_cost}, curr cost = {curr_cost}"
            )
            cost_decreased = curr_cost < prev_cost
            converged = not cost_decreased
            prev_cost = curr_cost

        return clustering
