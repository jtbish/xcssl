import copy

from lsh import distance_between_lsh_keys
from phenotype import VanillaPhenotype


class SubsumptionTree:
    def __init__(self, encoding, lsh, phenotypes):

        self._encoding = encoding
        self._lsh = lsh

        self._phenotype_count_map = self._init_phenotype_count_map(phenotypes)

        (self._lsh_key_leaf_node_map, self._root_node) = self._make_tree(
            self._encoding,
            self._lsh,
            phenotype_set=self._phenotype_count_map.keys())

    def _init_phenotype_count_map(self, phenotypes):
        phenotype_count_map = {}

        for phenotype in phenotypes:
            try:
                phenotype_count_map[phenotype] += 1
            except KeyError:
                phenotype_count_map[phenotype] = 1

        return phenotype_count_map

    def _make_tree(self, encoding, lsh, phenotype_set):
        # first, make leaf nodes at bottom of the tree, each identified by
        # their LSH key
        phenotype_lsh_key_map = self._gen_phenotype_lsh_key_map(
            lsh, phenotype_set)

        lsh_key_leaf_node_map = \
            self._gen_lsh_key_leaf_node_map(phenotype_lsh_key_map, encoding)

        # next, do HAC (complete linkage) on the leaf nodes to build the
        # internal nodes and root node of the tree.
        lsh_key_dist_map = self._gen_lsh_key_dist_map(self,
                                                      lsh_key_leaf_node_map)

        root_node = self._build_internal_and_root_nodes(
            lsh_key_leaf_node_map, lsh_key_dist_map, encoding)

        return (lsh_key_leaf_node_map, root_node)

    def _gen_phenotype_lsh_key_map(self, lsh, phenotype_set):
        """Generates the LSH hash/key for each phenotype
        (assumes hasher is using a single band)."""

        return {
            phenotype: lsh.hash(phenotype.vec)
            for phenotype in phenotype_set
        }

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

    def _gen_lsh_key_dist_map(self, lsh_key_leaf_node_map):
        # TODO change back to making a dist mat since don't necessarily konw
        # the indexing oredr (could be either way). so needs to be symmetrical
        """Will do n choose 2 dist. calcs, where n =
        len(lsh_key_leaf_node_map).
        Stores in sparse non-symmetrical double-nested dict."""

        # all keys (rows) to loop over
        full_lsh_key_set = lsh_key_leaf_node_map.keys()
        # current set of other keys to compare to (shrinks each iter)
        compare_lsh_key_set = copy.deepcopy(full_lsh_key_set)

        lsh_key_dist_map = {}

        for lsh_key_a in full_lsh_key_set:
            # don't compare to self
            compare_lsh_key_set.remove(lsh_key_a)

            if len(compare_lsh_key_set) > 0:

                local_map = {}

                for lsh_key_b in compare_lsh_key_set:
                    local_map[lsh_key_b] = \
                        distance_between_lsh_keys(lsh_key_a, lsh_key_b)

                lsh_key_dist_map[lsh_key_a] = local_map

        return lsh_key_dist_map

    def _build_internal_and_root_nodes(self, lsh_key_leaf_node_map,
                                       lsh_key_dist_map, encoding):
        """Complete linkage HAC."""
        curr_dist_map = lsh_key_dist_map

        # map from "composite" keys (tuples of merged lsh keys) to internal
        # nodes.
        # this starts out with the leaf nodes (which techincally have
        # non-composite keys), but need to be included to simplify key lookups
        # (instead of having a leaf vs. non-leaf map and logic to check which
        # one to query).
        cmpst_key_node_map = {k: v for (k, v) in lsh_key_leaf_node_map.items()}

        done = False
        while not done:

            make_internal_node = (len(curr_dist_map) > 1)

            if make_internal_node:

                # find min dist pair of keys to merge
                min_dist = None
                merge_pair = None

                for key_a in curr_dist_map.keys():
                    for (key_b, dist) in (curr_dist_map[key_a]).items():

                        if min_dist is None or dist < min_dist:
                            min_dist = dist
                            merge_pair = (key_a, key_b)

                # do the merge and make the new dist map
                assert merge_pair is not None
                # the new composite key is the merge pair
                cmpst_key = merge_pair

                # make new internal node and store it in node map
                left_child_node = cmpst_key_node_map[cmpst_key[0]]
                right_child_node = cmpst_key_node_map[cmpst_key[1]]

                internal_node = InternalNode(left_child_node, right_child_node,
                                             encoding)
                cmpst_key_node_map[cmpst_key] = internal_node

                # update parent pointers for both children
                for child_node in (left_child_node, right_child_node):
                    child_node.parent_node = internal_node

                # update the dist map for next iter
                # first, copy over entries that aren't part of the merge (i.e.
                # keys that aren't part of the new composite key)
                next_dist_map = {
                    k: v
                    for (k, v) in curr_dist_map.items() if k not in cmpst_key
                }

                # next, determine the dist val for each unchanged key for the
                # new composite key
                cmpst_key_dists = {}
                for unchanged_key in next_dist_map.keys():

                    # complete linkage == max dist.
                    max_dist = None

                    for sub_key in cmpst_key:
                        dist = curr_dist_map[sub_key][unchanged_key]

                        if max_dist is None or dist > max_dist:
                            max_dist = dist

                    # distance for merged key is equal to this max. dist
                    cmpst_key_dists[unchanged_key] = max_dist

                next_dist_map[cmpst_key] = cmpst_key_dists

                assert len(next_dist_map) == (len(curr_dist_map) - 1)

                curr_dist_map = next_dist_map

            else:
                # make root node
                assert len(curr_dist_map) == 1

                done = True

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
    No bounding volume, just left and right pointers to InternalNodes."""
    pass


class InternalNode:
    """InternalNode is a bounding volume with left and right pointers
    to either other InternalNode or LeafNode objs."""
    def __init___(self, left_child_node, right_child_node, encoding):

        for node in (left_child_node, right_child_node):
            assert (isinstance(node, LeafNode)
                    or isinstance(node, InternalNode))

        self._left_child_node = left_child_node
        self._right_child_node = right_child_node

        # subsumer for this InternalNode needs to susbume both the subsumer
        # phenotypes for the child nodes
        phenotypes_to_subsume = [
            self._left_child_node.susbumer_phenotype,
            self._right_child_node.susbumer_phenotype
        ]
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
            # sole member of cluster is the subsumer (but new obj., only needs
            # do be vanilla phenotype, not vectorised).
            self._subsumer_phenotype = \
                VanillaPhenotype(self._phenotypes[0].elems)

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
        assert self._parent_node is None

        assert isinstance(node, InternalNode)
        self._parent_node = node
