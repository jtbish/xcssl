from collections import namedtuple

MedoidChange = namedtuple("MedoidChange", ["old", "new"])


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

    def try_add_phenotype(self, addee):
        # first, determine if addee is actually new
        try:
            self._phenotype_count_map[addee] += 1
        except KeyError:
            self._phenotype_count_map[addee] = 1
            is_new = True
        else:
            is_new = False

        if is_new:
            self._add_phenotype(addee)

    def _add_phenotype(self, addee):
        # 1: calc and store lsh keys for addee
        vec = addee.vec
        lsh_keys = []
        for band_idx in range(self._num_bands):
            lsh_key = self._lsh.hash(vec, band_idx)
            self._lsh_key_maps[band_idx][addee] = lsh_key
            lsh_keys.append(lsh_key)

        # 2: add the addee to a cluster in each band,
        # recording the medoid changes that happen as a result
        medoid_changes = []

        for (band_idx, lsh_key) in enumerate(lsh_keys):
            try:
                # try access existing cluster
                cluster = self._clusterings[band_idx][lsh_key]
            except KeyError:
                # new cluster needed, consisting solely of addee
                cluster = Cluster(phenotypes=[addee],
                                  encoding=self._encoding,
                                  dist_mat=self._dist_mat)
                # no old medoid, new medoid is sole phenotype in cluster
                medoid_changes.append(MedoidChange(old=None, new=addee))
            else:
                # add to existing cluster
                medoid_change = cluster.add(addee, self._encoding,
                                            self._dist_mat)
                medoid_changes.append(medoid_change)

        assert len(medoid_changes) == self._num_bands

        # 3: apply the accrued medoid changes to the medoid count map
        for (old_medoid, new_medoid) in medoid_changes:
            assert new_medoid is not None

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

        # 5: if addee was not determined to be a medoid, it needs to be
        # a predictee
        if addee not in self._medoid_count_map:
            self._predictee_set.add(addee)

        assert (len(self._medoid_count_map) + len(self._predictee_set)) == len(
            self._phenotype_count_map)

    def try_remove_phenotype(self, removee):
        # first, determine if removal actually necessary
        count = self._phenotype_count_map[removee]
        assert count >= 1

        if count == 1:
            del self._phenotype_count_map[removee]
            do_removal = True
        else:
            self._phenotype_count_map[removee] = (count - 1)
            do_removal = False

        if do_removal:
            self._remove_phenotype(removee)

    def _remove_phenotype(self, removee):
        # 1: remove the removee from each cluster in each band,
        # recording the medoid changes that happen as a result
        medoid_changes = []

        for band_idx in range(self._num_bands):
            lsh_key = self._lsh_key_maps[band_idx][removee]
            cluster = self._clusterings[band_idx][lsh_key]

            assert removee in cluster.phenotype_set

            assert len(cluster) >= 1
            if len(cluster) == 1:

                # remove the cluster entirely since the removee is
                # the only member
                del self._clusterings[band_idx][lsh_key]
                medoid_changes.append(MedoidChange(old=removee, new=None))

            else:

                medoid_change = cluster.remove(removee, self._dist_mat)
                medoid_changes.append(medoid_change)

        assert len(medoid_changes) == self._num_bands

        # 2: apply the accrued medoid changes to the medoid count map
        for (old_medoid, new_medoid) in medoid_changes:
            assert old_medoid is not None

            if new_medoid is None:

                self._medoid_count_map[old_medoid] -= 1

            # note: don't need to handle case of old == new since no change
            elif old_medoid != new_medoid:

                # old medoid has to be in the medoid count map already
                self._medoid_count_map[old_medoid] -= 1
                # but new one might not be
                try:
                    self._medoid_count_map[new_medoid] += 1
                except KeyError:
                    self._medoid_count_map[new_medoid] = 1

        # 3: remove all other traces of the phenotype
        # NOTE: phenotype count map already handled in caller
        # i) predictee set
        try:
            self._predictee_set.remove(removee)
        except KeyError:
            pass

        # ii) medoid count map
        try:
            del self._medoid_count_map[removee]
        except KeyError:
            pass

        # iii) dist mat entries
        for phenotype in self._phenotype_count_map.keys():
            self._dist_mat.remove_single_entry(phenotype, removee)
        self._dist_mat.remove_entire_row(removee)
        assert (self._dist_mat.size % 2 == 0)

        # iv) lsh key maps
        for band_idx in range(self._num_bands):
            del self._lsh_key_maps[band_idx][removee]

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

    def remove_single_entry(self, phenotype_a, phenotype_b):
        del self._mat[phenotype_a][phenotype_b]
        self._num_entries -= 1

    def remove_entire_row(self, phenotype):
        num_entries_in_row = len(self._mat[phenotype])
        del self._mat[phenotype]
        self._num_entries -= num_entries_in_row


class Cluster:
    def __init__(self, phenotypes, encoding, dist_mat):
        self._phenotype_set = set(phenotypes)
        (self._dist_sums, self._medoid) = \
            self._calc_init_dist_sums_and_medoid(self._phenotype_set, encoding,
                                                 dist_mat)

    def _calc_init_dist_sums_and_medoid(self, phenotype_set, encoding,
                                        dist_mat):
        dist_sums = {}

        medoid_dist_sum = None
        medoid = None

        for phenotype_a in phenotype_set:
            dist_sum = 0

            for phenotype_b in phenotype_set:
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
    def phenotype_set(self):
        return self._phenotype_set

    @property
    def medoid(self):
        return self._medoid

    def add(self, addee, encoding, dist_mat):
        old_medoid = self._medoid

        new_medoid = None
        new_medoid_dist_sum = None

        addee_dist_sum = 0

        for existing_phenotype in self._phenotype_set:
            # calc dist between existing and new phenotype
            try:
                dist = dist_mat.query(addee, existing_phenotype)
            except KeyError:
                dist = encoding.distance_between(addee.vec,
                                                 existing_phenotype.vec)
                dist_mat.add_pair(addee, existing_phenotype, dist)

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
            addee_dist_sum += dist

        assert new_medoid is not None

        # store the new phenotype and its dist sum
        self._phenotype_set.add(addee)
        self._dist_sums[addee] = addee_dist_sum
        assert len(self._phenotype_set) == len(self._dist_sums)

        # lastly, check if (now complete) dist_sum for new phenotype makes it
        # the new medoid
        if addee_dist_sum < new_medoid_dist_sum:
            new_medoid = addee

        # update the stored medoid, and return the change to the caller
        self._medoid = new_medoid

        return MedoidChange(old=old_medoid, new=new_medoid)

    def remove(self, removee, dist_mat):
        old_medoid = self._medoid

        new_medoid = None
        new_medoid_dist_sum = None

        other_phenotypes = (self._phenotype_set - {removee})

        for other_phenotype in other_phenotypes:

            dist = dist_mat.query(other_phenotype, removee)

            updated_dist_sum = (self._dist_sums[other_phenotype] - dist)
            self._dist_sums[other_phenotype] = updated_dist_sum

            # consider if (now updated) other phenotype should be new medoid
            if new_medoid is None:
                new_medoid = other_phenotype
                new_medoid_dist_sum = updated_dist_sum
            else:
                assert new_medoid_dist_sum is not None
                if updated_dist_sum < new_medoid_dist_sum:
                    new_medoid = other_phenotype
                    new_medoid_dist_sum = updated_dist_sum

        assert new_medoid is not None

        # remove the removee from stored attrs
        self._phenotype_set.remove(removee)
        del self._dist_sums[removee]
        assert len(self._phenotype_set) == len(self._dist_sums)

        # update the stored medoid, and return the change to the caller
        self._medoid = new_medoid

        return MedoidChange(old=old_medoid, new=new_medoid)

    def __len__(self):
        return len(self._phenotype_set)
