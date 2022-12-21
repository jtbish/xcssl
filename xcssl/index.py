import abc
import logging
from collections import OrderedDict

_INIT_NODE_ID = 0
_MIN_NUM_ROOT_NODES = 2


class SubsumptionForest:
    def __init__(self, encoding, lsh, phenotypes, theta_clust):
        self._encoding = encoding
        self._lsh = lsh

        self._phenotype_count_map = self._init_phenotype_count_map(phenotypes)

        self._phenotype_lsh_key_map = self._gen_phenotype_lsh_key_map(
            self._lsh, phenotype_set=self._phenotype_count_map.keys())

        self._next_node_id = _INIT_NODE_ID

        (self._lsh_key_leaf_node_map,
         self._node_id_root_node_map) = self._build_forest(
             self._encoding, self._lsh, self._phenotype_lsh_key_map)

        self._num_adds_and_removes = 0
        self._last_forest_build_step = 0
        self._theta_clust = theta_clust

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

    def _get_next_node_id(self):
        id_ = self._next_node_id
        self._next_node_id += 1
        return id_

    def _build_forest(self, encoding, lsh, phenotype_lsh_key_map):
        # first, make leaf nodes at bottom of the tree, each identified by
        # their LSH key
        lsh_key_leaf_node_map = \
            self._gen_lsh_key_leaf_node_map(phenotype_lsh_key_map, encoding)

        # next, do HAC on the leaf nodes to build the
        # tree(s) in the forest, each tree reprd. by its root node
        node_id_root_node_map = self._build_trees(lsh_key_leaf_node_map,
                                                  encoding)

        return (lsh_key_leaf_node_map, node_id_root_node_map)

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
        # preserve ordering via ordered dict (since ordering used later when
        # constructing forest)
        lsh_key_leaf_node_map = OrderedDict()
        for (lsh_key, phenotypes) in partitions.items():
            node_id = self._get_next_node_id()
            lsh_key_leaf_node_map[lsh_key] = LeafNode(node_id, phenotypes,
                                                      encoding)

        return lsh_key_leaf_node_map

    def _build_trees(self, lsh_key_leaf_node_map, encoding):

        # make node list.
        # use ordering on leaf nodes so they can be referred to by integer
        # node id (== idx in the list)
        node_ls = list(lsh_key_leaf_node_map.values())
        for (idx, node) in enumerate(node_ls):
            assert node.node_id == idx

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

        num_merges_done = 0
        max_generality = encoding.calc_max_generality()
        done = False

        while not done:

            #self._pretty_print_subsumer_cost_mat(subsumer_cost_mat)

            n = len(subsumer_cost_mat)
            try_make_merge_node = (n > 2)

            if not try_make_merge_node:
                done = True

            else:
                # find min valid cost pair to do merge for (if any)
                min_valid_cost = None
                merge_pair = None

                rod_iter = self._make_right_off_diag_iter(
                    node_id_set=subsumer_cost_mat.keys())

                for (node_id_a, node_id_b) in rod_iter:

                    (subsumer, cost) = subsumer_cost_mat[node_id_a][node_id_b]

                    is_valid = (subsumer.generality < max_generality)

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

                    next_node_id = self._get_next_node_id()
                    #print(f"{next_node_id} = merge({left_child_node_id}, "
                    #      f"{right_child_node_id}) @ cost "
                    #      f"({min_valid_cost[0]}, {min_valid_cost[1]})")

                    # make new node and store it in the node list
                    left_child_node = node_ls[left_child_node_id]
                    right_child_node = node_ls[right_child_node_id]

                    (merge_node_subsumer_phenotype, _) = subsumer_cost_mat[
                        left_child_node_id][right_child_node_id]
                    merge_node_height = self._calc_merge_node_height(
                        left_child_node, right_child_node)

                    merge_node = MergeNode(next_node_id, merge_node_height,
                                           left_child_node, right_child_node,
                                           merge_node_subsumer_phenotype)
                    node_ls.append(merge_node)

                    # update parent pointers for both children
                    for child_node in (left_child_node, right_child_node):
                        child_node.parent_node = merge_node

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
                             merge_node_subsumer_phenotype)

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

        # now, figure out the root nodes of the tree(s) in the forest
        # to do this, simply examine all nodes in the node list
        # nodes that don't have a parent must be the root of a tree in the
        # forest
        node_id_root_node_map = {
            node.node_id: node
            for node in node_ls if node.parent_node is None
        }

        assert len(node_id_root_node_map) >= _MIN_NUM_ROOT_NODES

        #        print("\n")
        #        print(f"Num merges done = {num_merges_done}")
        #
        #        num_leaf_nodes = len(lsh_key_leaf_node_map)
        #        print(f"Num leaf nodes = {num_leaf_nodes}")
        #
        #        num_trees = len(node_id_root_node_map)
        #        print(f"Num trees in forest = {num_trees}")
        #        root_heights = [root.height for root in node_id_root_node_map.values()]
        #        print(f"Height of tree root nodes = {root_heights}")
        #
        #        total_num_nodes = len(node_ls)
        #        print(f"Total num nodes = {total_num_nodes}")
        #
        #        print("\n")
        #        for (id_, node) in enumerate(node_ls):
        #            print(id_, node.height, str(node.subsumer_phenotype.elems))
        #
        #        print("\n")
        #        for (i, root_node) in enumerate(node_id_root_node_map.values()):
        #            print(f"Tree {i+1}")
        #            self._pretty_print_tree(root_node)
        #            print("\n")

        return node_id_root_node_map

    def _pretty_print_subsumer_cost_mat(self, subsumer_cost_mat):
        print("\n")

        sorted_node_ids = sorted(subsumer_cost_mat.keys())
        header = "         \t".join([str(id_) for id_ in sorted_node_ids])
        print(f"\t\t{header}")
        print("-" * 180)

        for id_ in sorted_node_ids:
            dict_ = dict(subsumer_cost_mat[id_])
            dict_[id_] = ("-", ("-", 0))
            costs = [v[1] for (k, v) in sorted(dict_.items())]
            costs_str = "\t".join([f"({c[0]}, {c[1]})" for c in costs])
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

    def _calc_merge_node_height(self, left_child_node, right_child_node):
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

        sparse_phenotype_matching_map = {}
        total_num_matching_ops_done = 0

        # stack based pre-order traversal of each tree in the forest
        for root_node in self._node_id_root_node_map.values():

            match_cache = {}

            stack = []
            stack.append(root_node)

            while len(stack) > 0:

                node = stack.pop()

                if not node.is_empty():

                    subsumer_phenotype = node.subsumer_phenotype

                    try:
                        does_match = match_cache[subsumer_phenotype]
                    except KeyError:
                        does_match = self._encoding.does_phenotype_match(
                            subsumer_phenotype, obs)

                        total_num_matching_ops_done += 1

                        if does_match:
                            # might re-use further down in the tree, so cache
                            match_cache[subsumer_phenotype] = does_match

                    if does_match:

                        if isinstance(node, LeafNode):

                            (leaf_phenotype_matching_map,
                             leaf_num_matching_ops_done
                             ) = node.gen_phenotype_matching_map(
                                 self._encoding, obs, match_cache)

                            sparse_phenotype_matching_map.update(
                                leaf_phenotype_matching_map)
                            total_num_matching_ops_done += \
                                leaf_num_matching_ops_done

                        else:
                            stack.append(node.right_child_node)
                            stack.append(node.left_child_node)

        return (sparse_phenotype_matching_map, total_num_matching_ops_done)

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
            self._num_adds_and_removes += 1
            self._try_rebuild_forest()

    def _add_phenotype(self, addee):
        lsh_key = self._lsh.hash(addee.vec)
        self._phenotype_lsh_key_map[addee] = lsh_key

        # try add to existing leaf node, else make new leaf node
        # and set it as a root (since it is sole member of a new tree in the
        # forest, since not sure what tree if any to connect it to without
        # re-doing the clustering, so it is its own tree for now...)
        try:
            (self._lsh_key_leaf_node_map[lsh_key]).add(addee, self._encoding)
        except KeyError:
            node_id = self._get_next_node_id()
            leaf_node = LeafNode(node_id=node_id,
                                 phenotypes=[addee],
                                 encoding=self._encoding)

            self._lsh_key_leaf_node_map[lsh_key] = leaf_node
            self._node_id_root_node_map[node_id] = leaf_node

    def try_remove_phenotype(self, phenotype):
        count = self._phenotype_count_map[phenotype]
        count -= 1

        do_remove = (count == 0)

        if do_remove:
            del self._phenotype_count_map[phenotype]
            self._remove_phenotype(phenotype)
            self._num_adds_and_removes += 1
            self._try_rebuild_forest()

        else:
            self._phenotype_count_map[phenotype] = count

    def _remove_phenotype(self, removee):
        lsh_key = self._phenotype_lsh_key_map[removee]
        del self._phenotype_lsh_key_map[removee]

        leaf_node = self._lsh_key_leaf_node_map[lsh_key]
        leaf_node.remove(removee)

        # if the leaf node is also a root node and its size has decreased to
        # zero then delete it entirely.
        # otherwise, it is the child of a merge node so its parent will still
        # have a left/right reference to it, but... can't know which one it is
        # (left or right), so can't delete the ref in the parent.
        # doesn't matter anyway since checking it will be skipped
        # when gening the sparse phenotype map (as there is a check for empty
        # nodes in there).
        if (leaf_node.node_id in self._node_id_root_node_map.keys()
                and leaf_node.is_empty()):
            del self._node_id_root_node_map[leaf_node.node_id]

    def _try_rebuild_forest(self):
        should_rebuild_forest = \
            (self._num_adds_and_removes - self._last_forest_build_step) > \
            self._theta_clust

        if should_rebuild_forest:
            logging.info(f"Rebuilding subsumption forest @ "
                         f"{self._num_adds_and_removes} num adds/removes")
            self._rebuild_forest()
            self._last_forest_build_step = self._num_adds_and_removes

    def _rebuild_forest(self):
        # first, reset the node ids
        self._next_node_id = _INIT_NODE_ID

        # second, remake the lsh key leaf node map
        # this will reset the ids back to starting from zero and ensure the
        # leaf node subsumer phenotypes are as tight as possible
        self._lsh_key_leaf_node_map = self._gen_lsh_key_leaf_node_map(
            self._phenotype_lsh_key_map, self._encoding)

        # third, rebuild the trees in the forest
        self._node_id_root_node_map = \
            self._build_trees(self._lsh_key_leaf_node_map, self._encoding)


class NodeABC(metaclass=abc.ABCMeta):
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
        # should only be used a single time when building a tree
        # in the forest (if at all)
        # hence when setting parent node it should be null
        assert self._parent_node is None

        assert isinstance(node, MergeNode)
        self._parent_node = node

    @property
    def subsumer_phenotype(self):
        return self._subsumer_phenotype

    @subsumer_phenotype.setter
    def subsumer_phenotype(self, val):
        self._subsumer_phenotype = val

    @abc.abstractmethod
    def is_empty(self):
        raise NotImplementedError


class MergeNode(NodeABC):
    """MergeNode is a bounding volume with left and right pointers
    to either other MergeNode or LeafNode objs."""
    def __init__(self, node_id, height, left_child_node, right_child_node,
                 subsumer_phenotype):

        super().__init__(node_id, height)

        for node in (left_child_node, right_child_node):
            assert (isinstance(node, LeafNode) or isinstance(node, MergeNode))

        self._left_child_node = left_child_node
        self._right_child_node = right_child_node

        self._subsumer_phenotype = subsumer_phenotype

    @property
    def left_child_node(self):
        return self._left_child_node

    @property
    def right_child_node(self):
        return self._right_child_node

    def is_empty(self):
        # by definition
        return False


class LeafNode(NodeABC):
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

    def is_empty(self):
        return self._num_phenotypes == 0

    def gen_phenotype_matching_map(self, encoding, obs, match_cache=None):
        if match_cache is None:

            phenotype_matching_map = {
                phenotype: encoding.does_phenotype_match(phenotype, obs)
                for phenotype in self._phenotypes
            }
            num_matching_ops_done = len(phenotype_matching_map)

        else:

            phenotype_matching_map = {}
            num_matching_ops_done = 0

            for phenotype in self._phenotypes:
                try:
                    does_match = match_cache[phenotype]
                except KeyError:
                    does_match = encoding.does_phenotype_match(phenotype, obs)
                    num_matching_ops_done += 1

                phenotype_matching_map[phenotype] = does_match

        return (phenotype_matching_map, num_matching_ops_done)

    def add(self, addee, encoding):
        self._phenotypes.append(addee)
        self._num_phenotypes += 1

        # check if the addee is already subsumed by the subsumer
        # phenotype of the leaf node. if it is, don't need to do anything else
        addee_is_subsumed = encoding.does_subsume(
            phenotype_a=self._subsumer_phenotype, phenotype_b=addee)

        if not addee_is_subsumed:
            # expand the leaf node subsumer phenotype to accommodate the addee
            new_subsumer_phenotype = encoding.expand_subsumer_phenotype(
                subsumer_phenotype=self._subsumer_phenotype,
                new_phenotype=addee)
            self._subsumer_phenotype = new_subsumer_phenotype

            # percolate the change up the tree this leaf node belongs to
            curr_node = self
            done = False

            while not done:
                parent_node = curr_node.parent_node

                if parent_node is None:
                    done = True

                else:
                    parent_subsumer = parent_node.subsumer_phenotype
                    curr_subsumer = curr_node.subsumer_phenotype
                    parent_does_subsume = encoding.does_subsume(
                        phenotype_a=parent_subsumer, phenotype_b=curr_subsumer)

                    if parent_does_subsume:
                        done = True

                    else:
                        new_parent_subsumer = \
                            encoding.expand_subsumer_phenotype(
                                subsumer_phenotype=parent_subsumer,
                                new_phenotype=curr_subsumer)
                        parent_node.subsumer_phenotype = new_parent_subsumer

                        curr_node = parent_node

    def remove(self, removee):
        self._phenotypes.remove(removee)
        self._num_phenotypes -= 1
        assert self._num_phenotypes >= 0