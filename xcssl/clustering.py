class PhenotypeClustering:
    def __init__(self, encoding, lsh, phenotypes):
        self._encoding = encoding

        self._lsh = lsh

        self._phenotype_count_map = self._init_phenotype_count_map(phenotypes)

        self._lsh_key_map = self._gen_lsh_key_map(self._lsh,
                                                  self._phenotype_count_map)

        self._clustering = self._form_clustering(self._lsh_key_map,
                                                 self._encoding)

        self._num_additions = 0
        self._num_removals = 0

    def _init_phenotype_count_map(self, phenotypes):
        phenotype_count_map = {}

        for phenotype in phenotypes:
            try:
                phenotype_count_map[phenotype] += 1
            except KeyError:
                phenotype_count_map[phenotype] = 1

        return phenotype_count_map

    def _gen_lsh_key_map(self, lsh, phenotype_count_map):
        """Generates the LSH hash/key for each unique phenotype
        (assumes hasher is using a single band)."""

        return {
            phenotype: lsh.hash(phenotype.vec)
            for phenotype in phenotype_count_map.keys()
        }

    def _form_clustering(self, lsh_key_map, encoding):
        clustering = {}

        for (phenotype, lsh_key) in lsh_key_map.items():
            # try add phenotype to existing cluster, else make new collection
            try:
                clustering[lsh_key].append(phenotype)
            except KeyError:
                clustering[lsh_key] = [phenotype]

        # then make actual Cluster objs from these collections
        for (lsh_key, phenotypes) in clustering.items():
            clustering[lsh_key] = Cluster(phenotypes, encoding)

        return clustering

    def gen_phenotype_matching_map(self, obs):

        res = {}

        for cluster in self._clustering.values():
            res.update(cluster.gen_phenotype_matching_map(self._encoding, obs))

        return res

    def try_add_phenotype(self, phenotype):

        try:
            self._phenotype_count_map[phenotype] += 1
        except KeyError:
            self._phenotype_count_map[phenotype] = 1
            do_add = True
        else:
            do_add = False

        if do_add:

            self._add_phenotype(phenotype)
            self._num_additions += 1

    def _add_phenotype(self, phenotype):

        lsh_key = self._lsh.hash(phenotype.vec)
        self._lsh_key_map[phenotype] = lsh_key

        # try add to existing cluster, else make new cluster with phenotype as
        # sole member
        try:
            (self._clustering[lsh_key]).add(phenotype, self._encoding)
        except KeyError:
            self._clustering[lsh_key] = Cluster(phenotypes=[phenotype],
                                                encoding=self._encoding)

    def try_remove_phenotype(self, phenotype):

        count = self._phenotype_count_map[phenotype]
        count -= 1
        self._phenotype_count_map[phenotype] = count

        if count == 0:

            self._remove_phenotype(phenotype)
            self._num_removals += 1

    def _remove_phenotype(self, phenotype):

        lsh_key = self._lsh_key_map[phenotype]
        cluster = self._clustering[lsh_key]

        if cluster.size == 1:
            # remove the cluster entirely
            del self._clustering[lsh_key]

        else:
            # remove this single phenotype from the cluster
            cluster.remove(phenotype, self._encoding)

        # remove other traces of the phenotype
        del self._lsh_key_map[phenotype]
        del self._phenotype_count_map[phenotype]


class Cluster:
    def __init__(self, phenotypes, encoding):

        self._phenotypes = phenotypes
        self._num_phenotypes = len(self._phenotypes)

        if self._num_phenotypes == 1:
            self._subsumer_phenotype = None
        else:
            self._subsumer_phenotype = encoding.make_subsumer_phenotype(
                self._phenotypes)

    @property
    def phenotypes(self):
        return self._phenotypes

    @property
    def size(self):
        return self._num_phenotypes

    @property
    def subsumer_phenotype(self):
        return self._subsumer_phenotype

    def gen_phenotype_matching_map(self, encoding, obs):

        if self._subsumer_phenotype is not None:

            if (not encoding.does_phenotype_match(self._subsumer_phenotype,
                                                  obs)):

                # impossible for any cluster members to match since the
                # subsumer does not
                return {phenotype: False for phenotype in self._phenotypes}

            else:

                # possible that cluster members could match, so fully match
                # all of them
                return {
                    phenotype: encoding.does_phenotype_match(phenotype, obs)
                    for phenotype in self._phenotypes
                }

        else:

            # no subsumer phenotype for singular member of the cluster
            phenotype = self._phenotypes[0]
            return {phenotype: encoding.does_phenotype_match(phenotype, obs)}

    def add(self, addee, encoding):

        self._phenotypes.append(addee)
        self._num_phenotypes += 1

        if self._subsumer_phenotype is not None:

            # check to see if existing subsumer phenotype needs to be altered
            # (this is only necessary if it does not subsume the addee)
            if (not encoding.does_subsume(self._subsumer_phenotype, addee)):

                self._subsumer_phenotype = \
                    encoding.expand_subsumer_phenotype(
                        self._subsumer_phenotype, addee)

        else:

            # before addee the cluster size was 1
            self._subsumer_phenotype = encoding.make_subsumer_phenotype(
                self._phenotypes)

    def remove(self, removee, encoding):
        self._phenotypes.remove(removee)
        self._num_phenotypes -= 1

        if self._num_phenotypes == 1:
            self._subsumer_phenotype = None
        else:
            # re-calc the subsumer to make it as "tight" as possible
            self._subsumer_phenotype = encoding.make_subsumer_phenotype(
                self._phenotypes)
