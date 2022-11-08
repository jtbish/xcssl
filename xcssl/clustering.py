from collections import namedtuple

MedoidChange = namedtuple("MedoidChange", ["old", "new"])

_MIN_NUM_PROJS_PER_BAND = 1
_MIN_NUM_BANDS = 1


class ConditionClustering:
    def __init__(self, encoding, phenotypes, lsh):
        self._encoding = encoding

        self._lsh = lsh
        self._num_bands = lsh.num_bands

        self._phenotype_count_map = self._init_phenotype_count_map(phenotypes)

        self._lsh_key_maps = self._gen_lsh_key_maps(self._lsh,
                                                    self._phenotype_count_map,
                                                    self._num_bands)
        self._dist_mat = DistMat()
        self._clusterings = self._form_clusterings(self._lsh_key_maps,
                                                   self._num_bands,
                                                   self._encoding,
                                                   self._dist_mat)

        self._medoid_count_map = self._init_medoid_count_map(self._clusterings)

        self._predictee_set = \
            self._init_predictee_set(self._phenotype_count_map,
                                     self._medoid_count_map)

        assert (len(self._medoid_count_map) + len(self._predictee_set)) == len(
            self._phenotype_count_map)

    def _init_phenotype_count_map(self, phenotypes):
        phenotype_count_map = {}

        for phenotype in phenotypes:
            try:
                phenotype_count_map[phenotype] += 1
            except KeyError:
                phenotype_count_map[phenotype] = 1

        return phenotype_count_map

    def _gen_lsh_key_maps(self, lsh, phenotype_count_map, num_bands):
        """Generates the LSH hash/key for each phenotype, for all bands
        used by the hasher."""
        # one map for each band, each map storing phenotype -> lsh key
        lsh_key_maps = [{} for _ in range(num_bands)]

        for phenotype in phenotype_count_map.keys():
            vec = phenotype.vec

            for band_idx in range(num_bands):
                lsh_key = lsh.hash(vec, band_idx)
                lsh_key_maps[band_idx][phenotype] = lsh_key

        return lsh_key_maps

    def _form_clusterings(self, lsh_key_maps, num_bands, encoding, dist_mat):
        """Forms the clusters for each of the bands used by the hasher, for
        all the pheontypes.
        This is the inverse mapping of lsh_key_maps.

        As part of constructing the Cluster objs, dist_mat gets updated in
        place to contain all the needed distance pairs."""

        clusterings = [{} for _ in range(num_bands)]

        # first just collate all the phenotypes into lists for each cluster
        for band_idx in range(num_bands):
            lsh_key_map = lsh_key_maps[band_idx]

            for (phenotype, lsh_key) in lsh_key_map.items():
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

    def _init_predictee_set(self, phenotype_count_map, medoid_count_map):
        return set(phenotype_count_map.keys()) - set(medoid_count_map.keys())

    def gen_phenotype_matching_map(self, obs):
        # first, exhaustively match all the medoids
        medoid_matching_map = {
            medoid: self._encoding.does_phenotype_match(medoid, obs)
            for medoid in self._medoid_count_map.keys()
        }

        # then, do NN (nearest medoid) prediction for other predictee
        # phenotypes
        predictee_matching_map = {}
        for predictee in self._predictee_set:
            nearest_medoid = self._find_nearest_medoid(predictee)
            predictee_matching_map[predictee] = \
                medoid_matching_map[nearest_medoid]

        # merge both maps to make the overall result
        return {**medoid_matching_map, **predictee_matching_map}

    def _find_nearest_medoid(self, phenotype):
        nearest_medoid = None
        nearest_medoid_dist = None

        for band_idx in range(self._num_bands):
            lsh_key = self._lsh_key_maps[band_idx][phenotype]
            medoid = (self._clusterings[band_idx][lsh_key]).medoid
            dist = self._dist_mat.query(phenotype, medoid)

            if nearest_medoid is None:
                nearest_medoid = medoid
                nearest_medoid_dist = dist
            else:
                if dist < nearest_medoid_dist:
                    nearest_medoid = medoid
                    nearest_medoid_dist = dist

        return nearest_medoid

    def try_add_phenotype(self, phenotype):
        # first, determine if phenotype is actually new
        try:
            self._phenotype_count_map[phenotype] += 1
        except KeyError:
            self._phenotype_count_map[phenotype] = 1
            is_new = True
        else:
            is_new = False

        if is_new:
            self._add_phenotype(phenotype)

    def _add_phenotype(self, phenotype):
        # 1: calc and store lsh keys for phenotype
        vec = phenotype.vec
        lsh_keys = []
        for band_idx in range(self._num_bands):
            lsh_key = self._lsh.hash(vec, band_idx)
            self._lsh_key_maps[band_idx][phenotype] = lsh_key
            lsh_keys.append(lsh_key)

        # 2: add the phenotype to a cluster in each band,
        # recording the medoid changes that happen as a result
        medoid_changes = []

        for (band_idx, lsh_key) in enumerate(lsh_keys):
            try:
                # try access existing cluster
                cluster = self._clusterings[band_idx][lsh_key]
            except KeyError:
                # new cluster needed, consisting solely of new phenotype
                cluster = Cluster(phenotypes=[phenotype],
                                  encoding=self._encoding,
                                  dist_mat=self._dist_mat)
                # no old medoid, new medoid is sole phenotype in cluster
                medoid_changes.append(MedoidChange(old=None, new=phenotype))
            else:
                # add to existing cluster
                medoid_change = cluster.add(phenotype, self._encoding,
                                            self._dist_mat)
                medoid_changes.append(medoid_change)

        assert len(medoid_changes) == self._num_bands

        # 3: apply the accrued medoid changes to the medoid count map
        for (old_medoid, new_medoid) in medoid_changes:

            if old_medoid is None:

                try:
                    self._medoid_count_map[new_medoid] += 1
                except KeyError:
                    self._medoid_count_map[new_medoid] = 1

            # note: don't need to handle case of old == new since no change
            elif old_medoid != new_medoid:

                # old medoid has to be in the medoid count map already
                self._medoid_count_map[old_medoid] -= 1
                # but new one might not be
                try:
                    self._medoid_count_map[new_medoid] += 1
                except KeyError:
                    self._medoid_count_map[new_medoid] = 1

        # 4: if any medoids now have count == 0, then they need to be removed
        # from medoid count map and added as predictees
        to_remove_as_medoids = [
            phenotype for (phenotype, count) in self._medoid_count_map.items()
            if count == 0
        ]
        for phenotype in to_remove_as_medoids:
            del self._medoid_count_map[phenotype]
            self._predictee_set.add(phenotype)

        assert (len(self._medoid_count_map) + len(self._predictee_set)) == len(
            self._phenotype_count_map)

    def try_remove_phenotype(self, phenotype):
        # first, determine if removal actually necessary
        count = self._phenotype_count_map[phenotype]
        assert count >= 1

        if count == 1:
            del self._phenotype_count_map[phenotype]
            do_removal = True
        else:
            self._phenotype_count_map[phenotype] = (count - 1)
            do_removal = False

        if do_removal:
            self._remove_phenotype(phenotype)

    def _remove_phenotype(self, phenotype):
        # TODO
        raise NotImplementedError


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
        self._phenotypes = phenotypes
        (self._dist_sums, self._medoid) = \
            self._calc_init_dist_sums_and_medoid(self._phenotypes, encoding,
                                                 dist_mat)

    def _calc_init_dist_sums_and_medoid(self, phenotypes, encoding, dist_mat):
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

    def add(self, new_phenotype, encoding, dist_mat):
        old_medoid = self._medoid

        new_medoid = None
        new_medoid_dist_sum = None

        new_phenotype_dist_sum = 0

        for existing_phenotype in self._phenotypes:
            # calc dist between existing and new phenotype
            try:
                dist = dist_mat.query(new_phenotype, existing_phenotype)
            except KeyError:
                dist = encoding.distance_between(new_phenotype.vec,
                                                 existing_phenotype.vec)
                dist_mat.add_pair(new_phenotype, existing_phenotype, dist)

            # update the dist_sum for this existing phenotype
            updated_dist_sum = (self._dist_sums[existing_phenotype] + dist)
            self._dist_sums[existing_phenotype] = updated_dist_sum

            # consider if (now updated) existing phenotype should be new medoid
            if new_medoid is None:
                new_medoid = existing_phenotype
                new_medoid_dist_sum = updated_dist_sum
            else:
                assert new_medoid_dist_sum is not None
                if updated_dist_sum < new_medoid_dist_sum:
                    new_medoid = existing_phenotype
                    new_medoid_dist_sum = updated_dist_sum

            # accumulate dist_sum for new phenotype
            new_phenotype_dist_sum += dist

        assert new_medoid is not None

        # store the new phenotype and its dist sum
        self._phenotypes.append(new_phenotype)
        self._dist_sums[new_phenotype] = new_phenotype_dist_sum
        assert len(self._phenotypes) == len(self._dist_sums)

        # lastly, check if (now complete) dist_sum for new phenotype makes it
        # the new medoid
        if new_phenotype_dist_sum < new_medoid_dist_sum:
            new_medoid = new_phenotype

        # update the stored medoid, and return the change to the caller
        self._medoid = new_medoid

        return MedoidChange(old=old_medoid, new=new_medoid)

    def remove(self, phenotype, encoding, dist_mat):
        pass

    def __len__(self):
        return len(self._phenotypes)
