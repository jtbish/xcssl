from .lsh import distance_between_lsh_keys


class SubsumptionTree:
    def __init__(self, encoding, lsh, phenotypes):

        self._encoding = encoding
        self._lsh = lsh

        self._phenotype_count_map = self._init_phenotype_count_map(phenotypes)

        self._phenotype_lsh_key_map = self._gen_phenotype_lsh_key_map(
            self._lsh, phenotype_set=self._phenotype_count_map.keys())

        (self._lsh_key_leaf_node_map,
         self._root_node) = self._make_tree(self._encoding, self._lsh,
                                            self._phenotype_lsh_key_map)

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

    def _make_tree(self, encoding, lsh, phenotype_lsh_key_map):
        # first, make leaf nodes at bottom of the tree, each identified by
        # their LSH key
        lsh_key_leaf_node_map = \
            self._gen_lsh_key_leaf_node_map(phenotype_lsh_key_map, encoding)

        # next, do HAC (complete linkage) on the leaf nodes to build the
        # internal nodes and root node of the tree.
        root_node = self._build_internal_and_root_nodes(
            lsh_key_leaf_node_map, encoding)

        return (lsh_key_leaf_node_map, root_node)

    def _gen_lsh_key_leaf_node_map(self, phenotype_lsh_key_map, encoding):
        """Generates the mapping between LSH keys and the leaf nodes at the
        bottom of the tree."""
        # first partition the phenotypes into LSH key buckets
        partitions = {}

        for (phenotype, lsh_key) in phenotype_lsh_key_map.items():
            # try add phenotype to existing partition, else make new partition
            try:
                partitions[lsh_key].append(phenotype)
            except KeyError:
                partitions[lsh_key] = [phenotype]

        # then make the actual LeafNode objs. from these partitions
        lsh_key_leaf_node_map = {}
        for (lsh_key, phenotypes) in partitions.items():
            lsh_key_leaf_node_map[lsh_key] = LeafNode(phenotypes, encoding)

        return lsh_key_leaf_node_map

    def _build_internal_and_root_nodes(self, lsh_key_leaf_node_map, encoding):

        # make node list by firstly
        # inducing ordering on leaf nodes so they can be referred to by integer
        # id (== idx in the list)
        node_ls = list(lsh_key_leaf_node_map.values())

        n = len(node_ls)
        subsumer_cost_mat = {node_id: {} for node_id in range(n)}

        # iter over right off-diag of mat, fill it symmetrically
        # can use ranges of ids in these loops since initially the ids form
        # seq. of len n with no gaps
        for row_node_id in range(0, (n - 1)):
            for col_node_id in range((row_node_id + 1), n):

                (subsumer,
                 dist) = encoding.make_subsumer_phenotype_and_calc_dist(
                     (node_ls[row_node_id]).subsumer_phenotype,
                     (node_ls[col_node_id]).subsumer_phenotype)

                cost = (subsumer.generality, dist)

                subsumer_cost_mat[row_node_id][col_node_id] = (subsumer, cost)
                subsumer_cost_mat[col_node_id][row_node_id] = (subsumer, cost)

        assert len(subsumer_cost_mat) == n
        for row in subsumer_cost_mat.values():
            # excludes diag.
            assert len(row) == (n - 1)

        # id of the next node that results from a merge
        next_node_id = n

        num_merges_done = 0
        root_node = None
        done = False
        while not done:

            self._pretty_print_subsumer_cost_mat(subsumer_cost_mat)

            n = len(subsumer_cost_mat)
            make_internal_node = (n > 2)

            if make_internal_node:

                # find min cost pair to do merge for
                min_cost = None
                merge_pair = None

                rod_iter = self._make_right_off_diag_iter(
                    node_id_set=subsumer_cost_mat.keys())

                for (node_id_a, node_id_b) in rod_iter:
                    (_, cost) = subsumer_cost_mat[node_id_a][node_id_b]

                    # use tuple ordering here since cost is 2-tuple of (genr,
                    # dist), but < operator will correctly compute
                    # lexicographic ordering (minimise genr first, then dist)
                    if min_cost is None or cost < min_cost:
                        min_cost = cost
                        merge_pair = (node_id_a, node_id_b)

                # do the merge
                assert merge_pair is not None
                (left_child_node_id, right_child_node_id) = merge_pair

                print(f"{next_node_id} = merge({left_child_node_id}, "
                      f"{right_child_node_id}) @ cost {min_cost}")

                # make new internal node and store it in the node list
                left_child_node = node_ls[left_child_node_id]
                right_child_node = node_ls[right_child_node_id]

                (internal_node_subsumer_phenotype, _) = \
                    subsumer_cost_mat[left_child_node_id][right_child_node_id]

                internal_node = InternalNode(left_child_node, right_child_node,
                                             internal_node_subsumer_phenotype)
                node_ls.append(internal_node)

                # update parent pointers for both children
                for child_node in (left_child_node, right_child_node):
                    child_node.parent_node = internal_node

                # update the subsumer mat for next iter
                # first, copy over all the entries for the non merged
                # node ids
                non_merged_node_id_set = (
                    set(subsumer_cost_mat.keys()) -
                    {left_child_node_id, right_child_node_id})

                next_subsumer_cost_mat = {
                    node_id: {}
                    for node_id in non_merged_node_id_set
                }

                rod_iter = self._make_right_off_diag_iter(
                    node_id_set=non_merged_node_id_set)

                for (node_id_a, node_id_b) in rod_iter:
                    (subsumer, cost) = subsumer_cost_mat[node_id_a][node_id_b]

                    next_subsumer_cost_mat[node_id_a][node_id_b] = (subsumer,
                                                                    cost)
                    next_subsumer_cost_mat[node_id_b][node_id_a] = (subsumer,
                                                                    cost)

                # then, make a row for the next node id (result of the merge)
                # and populate the matrix for this next node id
                next_subsumer_cost_mat[next_node_id] = {}

                for non_merged_node_id in non_merged_node_id_set:

                    (subsumer,
                     dist) = encoding.make_subsumer_phenotype_and_calc_dist(
                         (node_ls[non_merged_node_id]).subsumer_phenotype,
                         internal_node_subsumer_phenotype)

                    cost = (subsumer.generality, dist)

                    next_subsumer_cost_mat[next_node_id][
                        non_merged_node_id] = (subsumer, cost)
                    next_subsumer_cost_mat[non_merged_node_id][
                        next_node_id] = (subsumer, cost)

                # mat should have shrunk by one row and one column in
                # each row
                assert len(next_subsumer_cost_mat) == (n - 1)
                for row in next_subsumer_cost_mat.values():
                    # excludes diag.
                    assert len(row) == (n - 2)

                subsumer_cost_mat = next_subsumer_cost_mat
                next_node_id += 1

            else:
                # make root node
                node_ids = list(subsumer_cost_mat.keys())
                assert len(node_ids) == 2

                left_child_node_id = node_ids[0]
                right_child_node_id = node_ids[1]

                print(f"Root = merge({left_child_node_id}, "
                      f"{right_child_node_id})")

                left_child_node = node_ls[left_child_node_id]
                right_child_node = node_ls[right_child_node_id]

                # make root node and update parent pointers for children
                root_node = RootNode(left_child_node, right_child_node)

                for child_node in (left_child_node, right_child_node):
                    child_node.parent_node = root_node

                done = True

            num_merges_done += 1

        assert root_node is not None

        print("\n")
        print(f"Num merges done = {num_merges_done}")

        num_leaf_nodes = len(lsh_key_leaf_node_map)
        num_internal_nodes = (len(node_ls) - num_leaf_nodes)
        total_num_nodes = (len(node_ls) + 1)
        print(f"Num leaf nodes = {num_leaf_nodes}")
        print(f"Num internal nodes = {num_internal_nodes}")
        print(f"Total num nodes = {total_num_nodes}")

        print("\n")
        for (id_, node) in enumerate(node_ls):
            print(id_, str(node.subsumer_phenotype.elems))

        return root_node

    def _pretty_print_subsumer_cost_mat(self, subsumer_cost_mat):
        print("\n")

        sorted_node_ids = sorted(subsumer_cost_mat.keys())
        header = "     \t".join([str(id_) for id_ in sorted_node_ids])
        print(f"\t\t{header}")
        print("-" * 80)

        for id_ in sorted_node_ids:
            dict_ = dict(subsumer_cost_mat[id_])
            dict_[id_] = ("-", "-" * 6)
            costs = [v[1] for (k, v) in sorted(dict_.items())]
            costs_str = "\t".join([str(c) for c in costs])
            print(f"{id_}\t|\t{costs_str}")

        print("\n")

    def _make_right_off_diag_iter(self, node_id_set):
        rod_iter = []

        outer_id_set = node_id_set
        inner_id_set = set(node_id_set)

        for outer_id in outer_id_set:
            inner_id_set.remove(outer_id)

            for inner_id in inner_id_set:

                rod_iter.append((outer_id, inner_id))

        return rod_iter

    def gen_phenotype_matching_map(self, obs):
        raise NotImplementedError

    def try_add_phenotype(self, phenotype):
        raise NotImplementedError

    def _add_phenotype(self, addee):
        raise NotImplementedError

    def try_remove_phenotype(self, phenotype):
        raise NotImplementedError

    def _remove_phenotype(self, removee):
        raise NotImplementedError


class RootNode:
    """RootNode is the top of the tree.
    No bounding volume, just left and right pointers to InternalNodes i.e.
    subtrees."""
    def __init__(self, left_child_node, right_child_node):

        for node in (left_child_node, right_child_node):
            assert (isinstance(node, LeafNode)
                    or isinstance(node, InternalNode))

        self._left_child_node = left_child_node
        self._right_child_node = right_child_node

    @property
    def left_child_node(self):
        return self._left_child_node

    @property
    def right_child_node(self):
        return self._right_child_node


class InternalNode:
    """InternalNode is a bounding volume with left and right pointers
    to either other InternalNode or LeafNode objs."""
    def __init__(self, left_child_node, right_child_node, subsumer_phenotype):

        for node in (left_child_node, right_child_node):
            assert (isinstance(node, LeafNode)
                    or isinstance(node, InternalNode))

        self._left_child_node = left_child_node
        self._right_child_node = right_child_node

        self._subsumer_phenotype = subsumer_phenotype

        # at init, parent node is unknown, will be updated when building higher
        # levels of the tree
        self._parent_node = None

    @property
    def left_child_node(self):
        return self._left_child_node

    @property
    def right_child_node(self):
        return self._right_child_node

    @property
    def subsumer_phenotype(self):
        return self._subsumer_phenotype

    @property
    def parent_node(self):
        return self._parent_node

    @parent_node.setter
    def parent_node(self, node):
        # should only be used a single time when building tree
        # hence when setting parent node it should be null
        assert self._parent_node is None

        assert (isinstance(node, InternalNode) or isinstance(node, RootNode))
        self._parent_node = node


class LeafNode:
    """LeafNode is a partition of phenotypes that has associated subsumer
    phenotype (i.e. bounding volume)."""
    def __init__(self, phenotypes, encoding):
        self._phenotypes = phenotypes

        self._num_phenotypes = len(self._phenotypes)

        if self._num_phenotypes == 1:
            # sole member of cluster is the subsumer
            self._subsumer_phenotype = self._phenotypes[0]

        elif self._num_phenotypes > 1:
            self._subsumer_phenotype = \
                encoding.make_subsumer_phenotype(self._phenotypes)

        else:
            assert False

        # at init, parent node is unknown, will be updated when building tree
        self._parent_node = None

    @property
    def phenotypes(self):
        return self._phenotypes

    @property
    def size(self):
        return self._num_phenotypes

    @property
    def subsumer_phenotype(self):
        return self._subsumer_phenotype

    @property
    def parent_node(self):
        return self._parent_node

    @parent_node.setter
    def parent_node(self, node):
        # should only be used a single time when building tree
        # hence when setting parent node it should be null
        assert self._parent_node is None

        assert (isinstance(node, InternalNode) or isinstance(node, RootNode))
        self._parent_node = node
