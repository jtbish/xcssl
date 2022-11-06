from .encoding import (IntegerUnorderedBoundEncoding,
                       RealUnorderedBoundEncoding, TernaryEncoding)
from .lsh import EuclideanLSH, HammingLSH

_MIN_NUM_PROJS_PER_BAND = 1
_MIN_NUM_BANDS = 1


class ConditionClustering:
    def __init__(self, encoding, phenotypes, num_projs_per_band, num_bands):
        self._encoding = encoding

        assert num_projs_per_band >= _MIN_NUM_PROJS_PER_BAND
        self._p = num_projs_per_band
        assert num_bands >= _MIN_NUM_BANDS
        self._b = num_bands

        self._lsh = self._init_lsh(self._encoding, self._p, self._b)
        self._C = self._init_phenotype_count_map(phenotypes)

        self._lsh_key_maps = self._gen_lsh_key_maps(self._lsh, self._C,
                                                    self._b)
        self._dist_mat = DistMat()
        self._clusterings = self._form_clusterings(self._lsh_key_maps, self._b,
                                                   self._encoding,
                                                   self._dist_mat)

        self._M = self._init_medoid_count_map(self._clusterings)

        self._N = self._init_predictee_nearest_medoid_map(
            self._C, self._lsh_key_maps, self._dist_mat, self._clusterings,
            self._M)

        assert len(self._M) + len(self._N) == len(self._C)

    def _init_lsh(self, encoding, p, b):
        # TODO make polymorphic?
        lsh_cls = None

        if isinstance(encoding, TernaryEncoding):
            lsh_cls = HammingLSH
        elif isinstance(encoding, IntegerUnorderedBoundEncoding):
            lsh_cls = HammingLSH
        elif isinstance(encoding, RealUnorderedBoundEncoding):
            lsh_cls = EuclideanLSH
        else:
            assert False

        assert lsh_cls is not None

        d = encoding.calc_num_phenotype_vec_dims()
        return lsh_cls(d, p, b)

    def _init_phenotype_count_map(self, phenotypes):
        phenotype_count_map = {}

        for phenotype in phenotypes:
            try:
                phenotype_count_map[phenotype] += 1
            except KeyError:
                phenotype_count_map[phenotype] = 1

        return phenotype_count_map

    def _gen_lsh_key_maps(self, lsh, C, b):
        """Generates the LSH hash/key for each phenotype in C, for all b bands
        used by the hasher."""
        # one map for each band, each map storing phenotype -> lsh key
        key_maps = [{} for _ in range(b)]

        for phenotype in C.keys():
            vec = phenotype.vec

            for band_idx in range(b):
                key = lsh.hash(vec, band_idx)
                key_maps[band_idx][phenotype] = key

        return key_maps

    def _form_clusterings(self, lsh_key_maps, b, encoding, dist_mat):
        """Forms the clusters for each of the b bands used by the hasher, for
        all the pheontypes in C.
        This is the inverse mapping of the lsh key maps.

        As part of constructing the Cluster objs, dist_mat gets updated in
        place to contain all the needed distance pairs."""

        clusterings = [{} for _ in range(b)]

        # first just collate all the phenotypes into lists for each cluster
        for band_idx in range(b):
            key_map = lsh_key_maps[band_idx]

            for (phenotype, lsh_key) in key_map.items():
                # try add phenotype to existing cluster, else make new cluster
                try:
                    clusterings[band_idx][lsh_key].append(phenotype)
                except KeyError:
                    clusterings[band_idx][lsh_key] = [phenotype]

        # then make actual Cluster objs from these lists
        for band in clusterings:
            for (lsh_key, phenotypes) in band.items():
                band[lsh_key] = Cluster(phenotypes, encoding, dist_mat)

        return clusterings

    def _init_medoid_count_map(self, clusterings):
        medoid_count_map = {}

        for band in clusterings:
            for cluster in band.values():
                medoid = cluster.medoid

                try:
                    medoid_count_map[medoid] += 1
                except KeyError:
                    medoid_count_map[medoid] = 1

        return medoid_count_map

    def _init_predictee_nearest_medoid_map(self, C, lsh_key_maps, dist_mat,
                                           clusterings, M):
        predictee_nm_map = {}
        predictees = set(C.keys()) - set(M.keys())

        b = len(clusterings)

        for predictee in predictees:
            medoid_dist_tuples = []

            for band_idx in range(b):

                lsh_key = lsh_key_maps[band_idx][predictee]
                cluster = clusterings[band_idx][lsh_key]

                medoid = cluster.medoid
                dist = dist_mat.query(predictee, medoid)

                medoid_dist_tuples.append((medoid, dist))

            # sort by dist. ascending
            sorted_medoid_dist_tuples = sorted(medoid_dist_tuples,
                                               key=lambda tup: tup[1],
                                               reverse=False)

            assert len(sorted_medoid_dist_tuples) == b
            predictee_nm_map[predictee] = sorted_medoid_dist_tuples

        return predictee_nm_map


class DistMat:
    """Wrapper class over double-nested dicts to make syntax + logic for
    querying/updating/removing entries nicer."""
    def __init__(self):
        self._mat = {}
        self._num_entries = 0

    @property
    def size(self):
        return self._num_entries

    def query(self, phenotype_a, phenotype_b):
        return self._mat[phenotype_a][phenotype_b]

    def add_pair(self, phenotype_a, phenotype_b, dist):
        """Add new pair to the matrix, meaning both combinations of (p_a, p_b),
        i.e. both off-diagonals."""

        for (first, second) in [(phenotype_a, phenotype_b),
                                (phenotype_b, phenotype_a)]:
            try:
                first_entries = self._mat[first]
            except KeyError:
                first_entries = {}
                self._mat[first] = first_entries

            first_entries[second] = dist

        self._num_entries += 2


class Cluster:
    def __init__(self, phenotypes, encoding, dist_mat):
        self._phenotypes = list(phenotypes)
        (self._dist_sums, self._medoid) = \
            self._calc_dist_sums_and_medoid(self._phenotypes, encoding,
                                            dist_mat)

    def _calc_dist_sums_and_medoid(self, phenotypes, encoding, dist_mat):
        dist_sums = {}

        medoid_dist_sum = None
        medoid = None

        for phenotype_a in phenotypes:
            dist_sum = 0

            for phenotype_b in phenotypes:
                if phenotype_a != phenotype_b:

                    try:
                        dist = dist_mat.query(phenotype_a, phenotype_b)
                    except KeyError:
                        dist = encoding.distance_between(
                            phenotype_a.vec, phenotype_b.vec)
                        dist_mat.add_pair(phenotype_a, phenotype_b, dist)

                    dist_sum += dist

            dist_sums[phenotype_a] = dist_sum

            if medoid is None:
                medoid = phenotype_a
                medoid_dist_sum = dist_sum
            else:
                assert medoid_dist_sum is not None
                if dist_sum < medoid_dist_sum:
                    medoid = phenotype_a
                    medoid_dist_sum = dist_sum

        assert medoid is not None

        return (dist_sums, medoid)

    @property
    def phenotypes(self):
        return self._phenotypes

    @property
    def medoid(self):
        return self._medoid

    def __len__(self):
        return len(self._phenotypes)
