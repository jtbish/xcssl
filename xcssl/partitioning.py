class LSHPartitioning:
    def __init__(self, encoding, lsh, phenotypes):
        self._encoding = encoding
        self._lsh = lsh

        self._phenotype_count_map = self._init_phenotype_count_map(phenotypes)

        self._phenotype_lsh_key_map = self._gen_phenotype_lsh_key_map(
            self._lsh, phenotype_set=self._phenotype_count_map.keys())

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

    def _gen_phenotype_lsh_key_map(self, lsh, phenotype_set):
        """Generates the LSH hash/key for each phenotype
        (assumes hasher is using a single band)."""

        return {
            phenotype: lsh.hash(phenotype.vec)
            for phenotype in phenotype_set
        }

    def _gen_lsh_key_partition_map(self, phenotype_lsh_key_map, encoding):
        """Generates the mapping between LSH keys and Partition objs."""
        # first partition the phenotypes into LSH key buckets as lists
        partitions = {}

        for (phenotype, lsh_key) in phenotype_lsh_key_map.items():
            # try add phenotype to existing partition, else make new partition
            try:
                partitions[lsh_key].append(phenotype)
            except KeyError:
                partitions[lsh_key] = [phenotype]

        # then make the actual Partition objs. from these
        lsh_key_partition_map = {}
        for (lsh_key, phenotypes) in partitions.items():
            lsh_key_partition_map[lsh_key] = Partition(phenotypes, encoding)

        return lsh_key_partition_map

    def gen_sparse_phenotype_matching_map(self, obs):

        sparse_phenotype_matching_map = {}
        num_matching_ops_done = 0

        for partition in self._lsh_key_partition_map.values():

            subsumer_does_match = self._encoding.does_phenotype_match(
                partition.subsumer_phenotype, obs)

            num_matching_ops_done += 1

            if subsumer_does_match:

                if partition.size == 1:
                    # don't need to check the sole member of the partition
                    # since it will be equal to the subsumer

                    sparse_phenotype_matching_map[
                        partition.subsumer_phenotype] = subsumer_does_match

                else:

                    (partition_matching_map, partition_num_matching_ops
                     ) = partition.gen_phenotype_matching_map(
                         self._encoding, obs)

                    sparse_phenotype_matching_map.update(
                        partition_matching_map)
                    num_matching_ops_done += partition_num_matching_ops

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
        lsh_key = self._lsh.hash(addee.vec)
        self._phenotype_lsh_key_map[addee] = lsh_key

        # try to add to existing partition, else make new partition
        try:
            (self._lsh_key_partition_map[lsh_key]).add(addee, self._encoding)
        except KeyError:
            partition = Partition(phenotypes=[addee], encoding=self._encoding)
            self._lsh_key_partition_map[lsh_key] = partition

    def try_remove_phenotype(self, phenotype):
        count = self._phenotype_count_map[phenotype]
        count -= 1

        assert count >= 0

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
        partition.remove(removee, self._encoding)

        if partition.size == 0:
            del self._lsh_key_partition_map[lsh_key]


class Partition:
    def __init__(self, phenotypes, encoding):
        self._phenotypes = phenotypes

        self._num_phenotypes = len(self._phenotypes)

        if self._num_phenotypes == 1:
            # sole member of the partition is the subsumer
            self._subsumer_phenotype = self._phenotypes[0]

        elif self._num_phenotypes > 1:
            self._subsumer_phenotype = \
                encoding.make_subsumer_phenotype(self._phenotypes)

        else:
            assert False

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
        phenotype_matching_map = {
            phenotype: encoding.does_phenotype_match(phenotype, obs)
            for phenotype in self._phenotypes
        }
        num_matching_ops_done = len(phenotype_matching_map)

        return (phenotype_matching_map, num_matching_ops_done)

    def add(self, addee, encoding):
        self._phenotypes.append(addee)
        self._num_phenotypes += 1
        # TODO remove
        assert len(self._phenotypes) == self._num_phenotypes

        assert self._num_phenotypes >= 1

        addee_is_subsumed = encoding.does_subsume(
            phenotype_a=self._subsumer_phenotype, phenotype_b=addee)

        if not addee_is_subsumed:

            new_subsumer = encoding.expand_subsumer_phenotype(
                subsumer_phenotype=self._subsumer_phenotype,
                new_phenotype=addee)

            self._subsumer_phenotype = new_subsumer

    def remove(self, removee, encoding):
        self._phenotypes.remove(removee)
        self._num_phenotypes -= 1
        # TODO remove
        assert len(self._phenotypes) == self._num_phenotypes

        assert self._num_phenotypes >= 0

        if self._num_phenotypes == 1:
            # shrink subsumer to be the sole member
            self._subsumer_phenotype = self._phenotypes[0]

        elif self._num_phenotypes > 1:
            # shrink subsumer to fit the remaining members
            self._subsumer_phenotype = \
                encoding.make_subsumer_phenotype(self._phenotypes)
