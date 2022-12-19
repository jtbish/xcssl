from .lsh import distance_between_lsh_keys

_MIN_GENR_CUTOFF_EXCL = 0.0
_MAX_GENR_CUTOFF_INCL = 1.0


class SubsumptionForest:
    def __init__(self, encoding, lsh, phenotypes, genr_cutoff):
        self._encoding = encoding
        self._lsh = lsh

        self._phenotype_count_map = self._init_phenotype_count_map(phenotypes)

        self._phenotype_lsh_key_map = self._gen_phenotype_lsh_key_map(
            self._lsh, phenotype_set=self._phenotype_count_map.keys())

        assert _MIN_GENR_CUTOFF_EXCL < genr_cutoff <= _MAX_GENR_CUTOFF_INCL
        self._genr_cutoff = genr_cutoff

        (self._lsh_key_leaf_node_map,
         self._root_nodes) = self._make_forest(self._encoding, self._lsh,
                                               self._phenotype_lsh_key_map,
                                               self._genr_cutoff)

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

    def _make_forest(self, encoding, lsh, phenotype_lsh_key_map, genr_cutoff):
        # first, make leaf nodes at bottom of the tree, each identified by
        # their LSH key
        lsh_key_leaf_node_map = \
            self._gen_lsh_key_leaf_node_map(phenotype_lsh_key_map, encoding)

        # next, do HAC on the leaf nodes to build the
        # tree(s) in the forest, each tree reprd. by its root node
        root_nodes = self._build_trees(lsh_key_leaf_node_map, encoding,
                                       genr_cutoff)

        return (lsh_key_leaf_node_map, root_nodes)

    def _gen_lsh_key_leaf_node_map(self, phenotype_lsh_key_map, encoding):
        """Generates the mapping between LSH keys and the leaf nodes at the
        bottom of a tree."""
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
        for (node_id, (lsh_key, phenotypes)) in enumerate(partitions.items()):
            lsh_key_leaf_node_map[lsh_key] = LeafNode(node_id, phenotypes,
                                                      encoding)

        return lsh_key_leaf_node_map

    def _build_trees(self, lsh_key_leaf_node_map, encoding, genr_cutoff):

        # make node list by firstly
        # inducing ordering on leaf nodes so they can be referred to by integer
        # node id (== idx in the list)
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
        done = False

        while not done:

            self._pretty_print_subsumer_cost_mat(subsumer_cost_mat)

            n = len(subsumer_cost_mat)
            make_internal_node = (n > 2)

            if make_internal_node:

                # find min valid cost pair to do merge for
                min_valid_cost = None
                merge_pair = None

                rod_iter = self._make_right_off_diag_iter(
                    node_id_set=subsumer_cost_mat.keys())

                for (node_id_a, node_id_b) in rod_iter:

                    (_, cost) = subsumer_cost_mat[node_id_a][node_id_b]
                    (genr, _) = cost

                    is_valid = (genr <= genr_cutoff)

                    if is_valid:
                        # use tuple ordering here since cost is 2-tuple of
                        # (genr, dist), but < operator will correctly compute
                        # lexicographic ordering (minimise genr first, then
                        # dist)
                        if min_valid_cost is None or cost < min_valid_cost:
                            min_valid_cost = cost
                            merge_pair = (node_id_a, node_id_b)

                # do the merge if possible
                merge_possible = (merge_pair is not None)

                if not merge_possible:
                    # terminate early
                    done = True

                else:
                    # make the node for the merge and update the subsumer cost
                    # mat for next iter

                    (left_child_node_id, right_child_node_id) = merge_pair

                    print(f"{next_node_id} = merge({left_child_node_id}, "
                          f"{right_child_node_id}) @ cost "
                          f"({min_valid_cost[0]:.2f}, {min_valid_cost[1]})")

                    # make new internal node and store it in the node list
                    left_child_node = node_ls[left_child_node_id]
                    right_child_node = node_ls[right_child_node_id]

                    (internal_node_subsumer_phenotype, _) = subsumer_cost_mat[
                        left_child_node_id][right_child_node_id]
                    internal_node_height = self._calc_node_height(
                        left_child_node, right_child_node)

                    internal_node = InternalNode(
                        next_node_id, internal_node_height, left_child_node,
                        right_child_node, internal_node_subsumer_phenotype)
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
                        (subsumer,
                         cost) = subsumer_cost_mat[node_id_a][node_id_b]

                        next_subsumer_cost_mat[node_id_a][node_id_b] = (
                            subsumer, cost)
                        next_subsumer_cost_mat[node_id_b][node_id_a] = (
                            subsumer, cost)

                    # then, make a row for the next node id
                    # (result of the merge)
                    # and populate the matrix for this next node id
                    next_subsumer_cost_mat[next_node_id] = {}

                    for non_merged_node_id in non_merged_node_id_set:

                        (subsumer, dist
                         ) = encoding.make_subsumer_phenotype_and_calc_dist(
                             (node_ls[non_merged_node_id]).subsumer_phenotype,
                             internal_node_subsumer_phenotype)

                        cost = (subsumer.generality, dist)

                        next_subsumer_cost_mat[next_node_id][
                            non_merged_node_id] = (subsumer, cost)
                        next_subsumer_cost_mat[non_merged_node_id][
                            next_node_id] = (subsumer, cost)

                    # mat should have shrunk by one row and one column in
                    # each row compared to before merge
                    assert len(next_subsumer_cost_mat) == (n - 1)
                    for row in next_subsumer_cost_mat.values():
                        # excludes diag.
                        assert len(row) == (n - 2)

                    num_merges_done += 1
                    subsumer_cost_mat = next_subsumer_cost_mat
                    next_node_id += 1

            else:
                # only two nodes left to merge, make root node for them
                # don't need to check genr cutoff since root node is just a
                # token node with no subsumer phenotype
                node_ids = list(subsumer_cost_mat.keys())
                assert len(node_ids) == 2

                left_child_node_id = node_ids[0]
                right_child_node_id = node_ids[1]

                print(f"Root {next_node_id} = merge({left_child_node_id}, "
                      f"{right_child_node_id})")

                left_child_node = node_ls[left_child_node_id]
                right_child_node = node_ls[right_child_node_id]

                # make root node and store it in the node list
                root_node_height = self._calc_node_height(
                    left_child_node, right_child_node)
                root_node = RootNode(next_node_id, root_node_height,
                                     left_child_node, right_child_node)
                node_ls.append(root_node)

                # update pointers for children
                for child_node in (left_child_node, right_child_node):
                    child_node.parent_node = root_node

                num_merges_done += 1
                done = True

        # now, figure out the root nodes of the tree(s) in the forest
        # to do this, simply examine all nodes in the node list
        # nodes that don't have a parent must be the root of a tree in the
        # forest
        root_nodes = [node for node in node_ls if node.parent_node is None]

        assert len(root_nodes) > 0

        if len(root_nodes) == 1:
            # single tree
            assert isinstance(root_nodes[0], RootNode)
        else:
            # multiple trees, but none of them should have a RootNode obj.
            # since this is reserved for the case where the merging procedure
            # eventually merges all starting LeafNode objs. forming a
            # forest composed of a single tree
            for node in root_nodes:
                assert (isinstance(node, LeafNode)
                        or isinstance(node, InternalNode))

        print("\n")
        print(f"Num merges done = {num_merges_done}")

        num_leaf_nodes = len(lsh_key_leaf_node_map)
        print(f"Num leaf nodes = {num_leaf_nodes}")

        num_trees = len(root_nodes)
        print(f"Num trees in forest = {num_trees}")
        root_heights = [root.height for root in root_nodes]
        print(f"Height of tree root nodes = {root_heights}")

        total_num_nodes = len(node_ls)
        print(f"Total num nodes = {total_num_nodes}")

        print("\n")
        for (id_, node) in enumerate(node_ls):
            print(id_, node.height, str(node.subsumer_phenotype.elems))

        print("\n")
        for (i, root_node) in enumerate(root_nodes):
            print(f"Tree {i+1}")
            self._pretty_print_tree(root_node)
            print("\n")

        return root_nodes

    def _pretty_print_subsumer_cost_mat(self, subsumer_cost_mat):
        print("\n")

        sorted_node_ids = sorted(subsumer_cost_mat.keys())
        header = "         \t".join([str(id_) for id_ in sorted_node_ids])
        print(f"\t\t{header}")
        print("-" * 180)

        for id_ in sorted_node_ids:
            dict_ = dict(subsumer_cost_mat[id_])
            dict_[id_] = ("-", (0, 0))
            costs = [v[1] for (k, v) in sorted(dict_.items())]
            costs_str = "\t".join([f"({c[0]:.2f}, {c[1]})" for c in costs])
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

    def _calc_node_height(self, left_child_node, right_child_node):
        return min(left_child_node.height, right_child_node.height) + 1

    def _pretty_print_tree(self, root_node):
        # post-order traversal
        print(f"{type(root_node)}\t\t{root_node.node_id}\t{root_node.height}\t"
              f"{str(root_node.subsumer_phenotype.elems)}")

        stack = []
        if not isinstance(root_node, LeafNode):
            stack.append(root_node.right_child_node)
            stack.append(root_node.left_child_node)

        while len(stack) > 0:
            node = stack.pop()
            print(f"{type(node)}\t\t{node.node_id}\t{node.height}\t"
                  f"{str(node.subsumer_phenotype.elems)}")

            if not isinstance(node, LeafNode):
                stack.append(node.right_child_node)
                stack.append(node.left_child_node)

    def gen_sparse_phenotype_matching_map(self, obs):

        res = {}

        num_matching_ops_done = 0

        # stack based pre-order traversal of tree(s) in the forest
        stack = []
        stack.append(self._root_node.right_child_node)
        stack.append(self._root_node.left_child_node)

        while len(stack) > 0:

            node = stack.pop()
            is_internal = isinstance(node, InternalNode)

            does_match = self._encoding.does_phenotype_match(
                node.subsumer_phenotype, obs)

            num_matching_ops_done += 1

            if is_internal and does_match:
                # traverse the children
                stack.append(node.right_child_node)
                stack.append(node.left_child_node)

            if not is_internal:

                if does_match:
                    # matching leaf node, so need to check all phenotypes
                    # contained within it
                    res.update(
                        node.gen_phenotype_matching_map(self._encoding, obs))

                    num_matching_ops_done += node.size

                else:
                    # non-matching leaf node, so all phenotypes contained
                    # within it also do not match
                    res.update(
                        {phenotype: False
                         for phenotype in node.phenotypes})

                # in either case, mark the leaf node as checked
                checked_leaf_node_lsh_keys.add(node.lsh_key)

        # any leaf node lsh keys that weren't checked implicitly did not match
        # since traversal never reached them
        for (lsh_key, leaf_node) in self._lsh_key_leaf_node_map.items():

            if lsh_key not in checked_leaf_node_lsh_keys:

                res.update(
                    {phenotype: False
                     for phenotype in leaf_node.phenotypes})

        assert len(res) == len(self._phenotype_count_map)

        return (res, num_matching_ops_done)

    def try_add_phenotype(self, phenotype):
        raise NotImplementedError

    def _add_phenotype(self, addee):
        raise NotImplementedError

    def try_remove_phenotype(self, phenotype):
        raise NotImplementedError

    def _remove_phenotype(self, removee):
        raise NotImplementedError


class NodeBase:
    _MIN_HEIGHT_INCL = 0

    def __init__(self, node_id, height):
        self._node_id = node_id

        assert height >= self._MIN_HEIGHT_INCL
        self._height = height

        # at init, parent node is unknown, will possibly be updated later when
        # building a given tree in the forest
        self._parent_node = None

    @property
    def node_id(self):
        return self._node_id

    @property
    def height(self):
        return self._height

    @property
    def parent_node(self):
        return self._parent_node

    @parent_node.setter
    def parent_node(self, node):

        if isinstance(self, RootNode):
            raise TypeError("Cannot set parent node for RootNode (by "
                            "definition)")
        else:
            # should only be used a single time when building a tree
            # in the forest (if at all)
            # hence when setting parent node it should be null
            assert self._parent_node is None

            assert (isinstance(node, InternalNode)
                    or isinstance(node, RootNode))
            self._parent_node = node


class RootNode(NodeBase):
    """RootNode is the top of a tree that results from merging the last two
    available nodes together.
    No subsumer phenotype, just left and right pointers to other nodes, i.e.
    subtrees."""
    def __init__(self, node_id, height, left_child_node, right_child_node):
        super().__init__(node_id, height)

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


class InternalNode(NodeBase):
    """InternalNode is a bounding volume with left and right pointers
    to either other InternalNode or LeafNode objs."""
    def __init__(self, node_id, height, left_child_node, right_child_node,
                 subsumer_phenotype):

        super().__init__(node_id, height)

        for node in (left_child_node, right_child_node):
            assert (isinstance(node, LeafNode)
                    or isinstance(node, InternalNode))

        self._left_child_node = left_child_node
        self._right_child_node = right_child_node

        self._subsumer_phenotype = subsumer_phenotype

    @property
    def left_child_node(self):
        return self._left_child_node

    @property
    def right_child_node(self):
        return self._right_child_node

    @property
    def subsumer_phenotype(self):
        return self._subsumer_phenotype


class LeafNode(NodeBase):
    """LeafNode is a partition of phenotypes that has associated subsumer
    phenotype (i.e. bounding volume)."""
    _HEIGHT = 0

    def __init__(self, node_id, phenotypes, encoding):
        super().__init__(node_id, height=self._HEIGHT)

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
        return {
            phenotype: encoding.does_phenotype_match(phenotype, obs)
            for phenotype in self._phenotypes
        }
