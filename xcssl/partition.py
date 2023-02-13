class SubsumptionPartition:
    def __init__(self, encoding, lsh, phenotypes):
        self._encoding = encoding
        self._lsh = lsh

        self._phenotype_count_map = self._init_phenotype_count_map(phenotypes)

        self._phenotype_lsh_key_map = self._gen_phenotype_lsh_key_map(
            self._encoding,
            self._lsh,
            phenotype_set=self._phenotype_count_map.keys())

        self._lsh_key_cell_map = self._gen_lsh_key_cell_map(
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
            # make vectorised repr of phenotype so it can be hashed by
            # LSH
            phenotype_lsh_key_map[phenotype] = \
                lsh.hash(encoding.gen_phenotype_vec(phenotype.elems))

        return phenotype_lsh_key_map

    def _gen_lsh_key_cell_map(self, phenotype_lsh_key_map, encoding):
        """Generates the mapping between LSH keys and Cell objs."""
        # first partition the phenotypes into subsets based on lsh key
        partition = {}

        for (phenotype, lsh_key) in phenotype_lsh_key_map.items():
            # try add phenotype to existing subset, else make new subset
            try:
                partition[lsh_key].add(phenotype)
            except KeyError:
                partition[lsh_key] = {phenotype}

        # then make the actual Cell objs. from these subsets
        lsh_key_cell_map = {}

        for (lsh_key, phenotype_set) in partition.items():

            if len(phenotype_set) == 1:
                cell = Cell.from_single_phenotype((tuple(phenotype_set))[0])
            else:
                cell = Cell.from_phenotype_set(phenotype_set, encoding)

            lsh_key_cell_map[lsh_key] = cell

        return lsh_key_cell_map

    def gen_sparse_phenotype_matching_map(self, obs):

        sparse_phenotype_matching_map = {}

        for cell in self._lsh_key_cell_map.values():

            subsumer = cell.subsumer_phenotype

            subsumer_does_match = self._encoding.does_phenotype_match(
                subsumer, obs)

            if subsumer_does_match:

                if cell.size == 1:
                    # don't need to check the sole member of the cell
                    # since it is identical to the subsumer

                    sparse_phenotype_matching_map[subsumer] = True

                else:
                    # otherwise, need to check all members of the cell

                    sparse_phenotype_matching_map.update(
                        cell.gen_phenotype_matching_map(self._encoding, obs))

        return sparse_phenotype_matching_map

    def gen_matching_trace(self, obs):

        sparse_phenotype_matching_map = {}
        num_matching_ops_done = 0

        for cell in self._lsh_key_cell_map.values():

            subsumer = cell.subsumer_phenotype

            subsumer_does_match = self._encoding.does_phenotype_match(
                subsumer, obs)
            num_matching_ops_done += 1

            if subsumer_does_match:

                if cell.size == 1:
                    # don't need to check the sole member of the cell
                    # since it is identical to the subsumer

                    sparse_phenotype_matching_map[subsumer] = True

                else:
                    # otherwise, need to check all members of the cell

                    sparse_phenotype_matching_map.update(
                        cell.gen_phenotype_matching_map(self._encoding, obs))

                    num_matching_ops_done += cell.size

        return (sparse_phenotype_matching_map, num_matching_ops_done)

    def gen_partial_matching_trace(self, obs):

        # always need to match the subsumer for all cells
        num_matching_ops_done = len(self._lsh_key_cell_map)

        for cell in self._lsh_key_cell_map.values():

            # if subsumer not equal to sole member of cell and it
            # matches, would also need to match all members of the cell
            if (cell.size > 1) and self._encoding.does_phenotype_match(
                    cell.subsumer_phenotype, obs):

                num_matching_ops_done += cell.size

        return num_matching_ops_done

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
        # calc lsh key given addee vec
        lsh_key = self._lsh.hash(self._encoding.gen_phenotype_vec(addee.elems))
        self._phenotype_lsh_key_map[addee] = lsh_key

        # try to add to existing cell, else make new cell
        try:
            (self._lsh_key_cell_map[lsh_key]).add(addee, self._encoding)
        except KeyError:
            self._lsh_key_cell_map[lsh_key] = Cell.from_single_phenotype(addee)

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

        cell = self._lsh_key_cell_map[lsh_key]

        if cell.size == 1:
            # delete the cell entirely since removing removee will cause
            # its size to be zero
            del self._lsh_key_cell_map[lsh_key]
        else:
            cell.remove(removee, self._encoding)


class Cell:
    def __init__(self, phenotype_set, encoding=None, subsumer_phenotype=None):

        assert (encoding is None and subsumer_phenotype is not None) \
            or (encoding is not None and subsumer_phenotype is None)

        self._phenotype_set = phenotype_set

        self._num_phenotypes = len(self._phenotype_set)

        if subsumer_phenotype is None:
            # none provided, need to calc
            self._subsumer_phenotype = \
                encoding.make_subsumer_phenotype(self._phenotype_set)
        else:
            self._subsumer_phenotype = subsumer_phenotype

    @classmethod
    def from_single_phenotype(cls, phenotype):
        # subsumer is identical to the sole phenotype
        return cls(phenotype_set={phenotype},
                   encoding=None,
                   subsumer_phenotype=phenotype)

    @classmethod
    def from_phenotype_set(cls, phenotype_set, encoding):
        # subsumer needs to be calced via encoding for multiple phenotypes in
        # the set
        return cls(phenotype_set=phenotype_set,
                   encoding=encoding,
                   subsumer_phenotype=None)

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
            # cell
            self._subsumer_phenotype = (tuple(self._phenotype_set))[0]

        elif self._num_phenotypes > 1:
            # shrink subsumer to fit the remaining members
            self._subsumer_phenotype = \
                encoding.make_subsumer_phenotype(self._phenotype_set)
