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
        """Complete linkage HAC."""

        lsh_key_ls = []
        tree_node_ls = []

        # induce ordering on leaf nodes so they can be referred to by integer
        # id == idx in lists
        for (lsh_key, leaf_node) in lsh_key_leaf_node_map.items():
            lsh_key_ls.append(lsh_key)
            tree_node_ls.append(leaf_node)

        n = len(lsh_key_leaf_node_map)
        node_dist_mat = {node_id: {} for node_id in range(n)}

        # iter over right off-diag of dist mat, fill it symmetrically
        # can use ranges of ids in these loops since initially the ids form
        # seq. of len n with no gaps
        for row_node_id in range(0, (n - 1)):
            for col_node_id in range((row_node_id + 1), n):

                dist = distance_between_lsh_keys(
                    lsh_key_a=lsh_key_ls[row_node_id],
                    lsh_key_b=lsh_key_ls[col_node_id])

                node_dist_mat[row_node_id][col_node_id] = dist
                node_dist_mat[col_node_id][row_node_id] = dist

        assert len(node_dist_mat) == n
        for row in node_dist_mat.values():
            # excludes diag.
            assert len(row) == (n - 1)

        curr_node_dist_mat = node_dist_mat
        # id of the next node that results from a merge
        next_node_id = n

        num_merges_done = 0
        root_node = None
        done = False
        while not done:

            self._pretty_print_node_dist_mat(curr_node_dist_mat)

            n = len(curr_node_dist_mat)
            make_internal_node = (n > 2)

            if make_internal_node:

                # find min dist pair of node ids to merge
                min_dist = None
                merge_pair = None

                node_id_b_set = set(curr_node_dist_mat.keys())

                for node_id_a in curr_node_dist_mat.keys():
                    # skip self comparison
                    node_id_b_set.remove(node_id_a)

                    for node_id_b in node_id_b_set:

                        dist = curr_node_dist_mat[node_id_a][node_id_b]

                        if min_dist is None or dist < min_dist:
                            min_dist = dist
                            merge_pair = (node_id_a, node_id_b)

                # do the merge
                assert merge_pair is not None
                (left_child_node_id, right_child_node_id) = merge_pair

                print(f"{next_node_id} = merge({left_child_node_id}, "
                      f"{right_child_node_id}) @ {min_dist}")

                # make new internal node and store it in the node list
                left_child_node = tree_node_ls[left_child_node_id]
                right_child_node = tree_node_ls[right_child_node_id]

                internal_node = InternalNode(left_child_node, right_child_node,
                                             encoding)
                tree_node_ls.append(internal_node)

                # update parent pointers for both children
                for child_node in (left_child_node, right_child_node):
                    child_node.parent_node = internal_node

                # update the dist mat for next iter
                # first, copy over all the pairwise dists for the non merged
                # node ids
                non_merged_node_id_set = set(curr_node_dist_mat.keys()) - \
                    {left_child_node_id, right_child_node_id}

                next_node_dist_mat = {
                    node_id: {}
                    for node_id in non_merged_node_id_set
                }

                node_id_b_set = set(non_merged_node_id_set)

                for node_id_a in non_merged_node_id_set:
                    # skip self comparison
                    node_id_b_set.remove(node_id_a)

                    for node_id_b in node_id_b_set:

                        dist = curr_node_dist_mat[node_id_a][node_id_b]

                        next_node_dist_mat[node_id_a][node_id_b] = dist
                        next_node_dist_mat[node_id_b][node_id_a] = dist

                # then make a row for the next node id (result of the merge)
                next_node_dist_mat[next_node_id] = {}

                for non_merged_node_id in non_merged_node_id_set:
                    # complete linkage == max dist between non merged id and
                    # either of the merged node ids
                    max_dist = max(
                        curr_node_dist_mat[non_merged_node_id]
                        [left_child_node_id],
                        curr_node_dist_mat[non_merged_node_id]
                        [right_child_node_id])

                    next_node_dist_mat[next_node_id][non_merged_node_id] = \
                        max_dist
                    next_node_dist_mat[non_merged_node_id][next_node_id] = \
                        max_dist

                # dist mat should have shrunk by one row and one column in each
                # row
                assert len(next_node_dist_mat) == (n - 1)
                for row in next_node_dist_mat.values():
                    # excludes diag.
                    assert len(row) == (n - 2)

                curr_node_dist_mat = next_node_dist_mat
                next_node_id += 1

            else:
                # make root node
                assert len(curr_node_dist_mat) == 2

                node_ids = list(curr_node_dist_mat.keys())
                assert len(node_ids) == 2

                left_child_node_id = node_ids[0]
                right_child_node_id = node_ids[1]

                print(f"Root = merge({left_child_node_id}, "
                      f"{right_child_node_id})")

                left_child_node = tree_node_ls[left_child_node_id]
                right_child_node = tree_node_ls[right_child_node_id]

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
        num_internal_nodes = (len(tree_node_ls) - num_leaf_nodes)
        total_num_nodes = (len(tree_node_ls) + 1)
        print(f"Num leaf nodes = {num_leaf_nodes}")
        print(f"Num internal nodes = {num_internal_nodes}")
        print(f"Total num nodes = {total_num_nodes}")

        print("\n")
        for (id_, node) in enumerate(tree_node_ls):
            print(id_, str(node.susbumer_phenotype.elems))

        return root_node

    def _pretty_print_node_dist_mat(self, node_dist_mat):
        print("\n")

        sorted_node_ids = sorted(node_dist_mat.keys())
        header = "\t".join([str(id_) for id_ in sorted_node_ids])
        print(f"\t\t{header}")
        print("-"*80)

        for id_ in sorted_node_ids:
            dict_ = dict(node_dist_mat[id_])
            dict_[id_] = "-"
            dists = [e[1] for e in sorted(dict_.items())]
            dists_str = "\t".join([str(d) for d in dists])
            print(f"{id_}\t|\t{dists_str}")

        print("\n")

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
    def __init__(self, left_child_node, right_child_node, encoding):

        for node in (left_child_node, right_child_node):
            assert (isinstance(node, LeafNode)
                    or isinstance(node, InternalNode))

        self._left_child_node = left_child_node
        self._right_child_node = right_child_node

        # subsumer for this InternalNode needs to susbume both the subsumer
        # phenotypes for the child nodes
        phenotypes_to_subsume = (self._left_child_node.susbumer_phenotype,
                                 self._right_child_node.susbumer_phenotype)
        self._subsumer_phenotype = \
            encoding.make_subsumer_phenotype(phenotypes_to_subsume)

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
    def susbumer_phenotype(self):
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
    def susbumer_phenotype(self):
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
