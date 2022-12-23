class LSHPartitioning:
    def __init__(self, encoding, lsh, phenotypes):
        self._encoding = encoding
        self._lsh = lsh

        self._phenotype_count_map = self._init_phenotype_count_map(phenotypes)

        self._phenotype_lsh_key_map = self._gen_phenotype_lsh_key_map(
            self._encoding,
            self._lsh,
            phenotype_set=self._phenotype_count_map.keys())

        self._lsh_key_partition_map = self._gen_lsh_key_partition_map(
            self._phenotype_lsh_key_map, self._encoding)

    def _init_phenotype_count_map(self, phenotypes):
        phenotype_count_map = {}

        for phenotype in phenotypes:
            try:
                phenotype_count_map[phenotype] += 1
            except KeyError:
                phenotype_count_map[phenotype] = 1

        return phenotype_count_map

    def _gen_phenotype_lsh_key_map(self, encoding, lsh, phenotype_set):
        """Generates the LSH hash/key for each phenotype
        (assumes hasher is using a single band)."""

        phenotype_lsh_key_map = {}

        for phenotype in phenotype_set:
            # first, make vectorised repr of phenotype so it can be hashed by
            # LSH
            vec = encoding.gen_phenotype_vec(phenotype.elems)
            phenotype.vec = vec
            phenotype_lsh_key_map[phenotype] = lsh.hash(vec)

        return phenotype_lsh_key_map

    def _gen_lsh_key_partition_map(self, phenotype_lsh_key_map, encoding):
        """Generates the mapping between LSH keys and Partition objs."""
        # first partition the phenotypes into LSH key buckets as sets
        partitions = {}

        for (phenotype, lsh_key) in phenotype_lsh_key_map.items():
            # try add phenotype to existing partition, else make new partition
            try:
                partitions[lsh_key].add(phenotype)
            except KeyError:
                partitions[lsh_key] = {phenotype}

        # then make the actual Partition objs. from these
        lsh_key_partition_map = {}
        for (lsh_key, phenotype_set) in partitions.items():
            lsh_key_partition_map[lsh_key] = Partition(phenotype_set, encoding)

        return lsh_key_partition_map

    def gen_sparse_phenotype_matching_map(self, obs):

        sparse_phenotype_matching_map = {}

        for partition in self._lsh_key_partition_map.values():

            subsumer = partition.subsumer_phenotype

            subsumer_does_match = self._encoding.does_phenotype_match(
                subsumer, obs)

            if subsumer_does_match:

                if partition.size == 1:
                    # don't need to check the sole member of the partition
                    # since it is identical to the subsumer

                    sparse_phenotype_matching_map[subsumer] = True

                else:
                    # otherwise, need to check all members of the partition

                    sparse_phenotype_matching_map.update(
                        partition.gen_phenotype_matching_map(
                            self._encoding, obs))

        return sparse_phenotype_matching_map

    def gen_matching_trace(self, obs):

        sparse_phenotype_matching_map = {}
        num_matching_ops_done = 0

        for partition in self._lsh_key_partition_map.values():

            subsumer = partition.subsumer_phenotype

            subsumer_does_match = self._encoding.does_phenotype_match(
                subsumer, obs)
            num_matching_ops_done += 1

            if subsumer_does_match:

                if partition.size == 1:
                    # don't need to check the sole member of the partition
                    # since it is identical to the subsumer

                    sparse_phenotype_matching_map[subsumer] = True

                else:
                    # otherwise, need to check all members of the partition

                    sparse_phenotype_matching_map.update(
                        partition.gen_phenotype_matching_map(
                            self._encoding, obs))

                    num_matching_ops_done += partition.size

        return (sparse_phenotype_matching_map, num_matching_ops_done)

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

    def _add_phenotype(self, addee):
        # calc and set vec for addee
        vec = self._encoding.gen_phenotype_vec(addee.elems)
        addee.vec = vec

        # then calc lsh key
        lsh_key = self._lsh.hash(vec)
        self._phenotype_lsh_key_map[addee] = lsh_key

        # try to add to existing partition, else make new partition
        try:
            (self._lsh_key_partition_map[lsh_key]).add(addee, self._encoding)
        except KeyError:
            partition = Partition(phenotype_set={addee},
                                  encoding=self._encoding)
            self._lsh_key_partition_map[lsh_key] = partition

    def try_remove_phenotype(self, phenotype):
        count = self._phenotype_count_map[phenotype]
        count -= 1

        do_remove = (count == 0)

        if do_remove:
            del self._phenotype_count_map[phenotype]
            self._remove_phenotype(phenotype)

        else:
            self._phenotype_count_map[phenotype] = count

    def _remove_phenotype(self, removee):
        lsh_key = self._phenotype_lsh_key_map[removee]
        del self._phenotype_lsh_key_map[removee]

        partition = self._lsh_key_partition_map[lsh_key]

        if partition.size == 1:
            # delete the partition entirely since removing removee will cause
            # size to be zero
            del self._lsh_key_partition_map[lsh_key]
        else:
            partition.remove(removee, self._encoding)


class Partition:
    def __init__(self, phenotype_set, encoding):
        self._phenotype_set = phenotype_set

        self._num_phenotypes = len(self._phenotype_set)

        if self._num_phenotypes == 1:
            # sole member of the partition is the subsumer
            self._subsumer_phenotype = (list(self._phenotype_set))[0]

        elif self._num_phenotypes > 1:
            self._subsumer_phenotype = \
                encoding.make_subsumer_phenotype(self._phenotype_set)

        else:
            assert False

    @property
    def phenotype_set(self):
        return self._phenotype_set

    @property
    def size(self):
        return self._num_phenotypes

    @property
    def subsumer_phenotype(self):
        return self._subsumer_phenotype

    def gen_phenotype_matching_map(self, encoding, obs):
        return {
            phenotype: encoding.does_phenotype_match(phenotype, obs)
            for phenotype in self._phenotype_set
        }

    def add(self, addee, encoding):
        self._phenotype_set.add(addee)
        self._num_phenotypes += 1

        addee_is_subsumed = encoding.does_subsume(
            phenotype_a=self._subsumer_phenotype, phenotype_b=addee)

        if not addee_is_subsumed:
            # expand the subsumer to fit the addee
            new_subsumer = encoding.expand_subsumer_phenotype(
                subsumer_phenotype=self._subsumer_phenotype,
                new_phenotype=addee)

            self._subsumer_phenotype = new_subsumer

    def remove(self, removee, encoding):
        self._phenotype_set.remove(removee)
        self._num_phenotypes -= 1

        if self._num_phenotypes == 1:
            # shrink subsumer to be equal to the (now) sole member of the
            # partition
            self._subsumer_phenotype = (list(self._phenotype_set))[0]

        elif self._num_phenotypes > 1:
            # shrink subsumer to fit the remaining members
            self._subsumer_phenotype = \
                encoding.make_subsumer_phenotype(self._phenotype_set)
