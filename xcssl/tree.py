import abc
import logging
import math
from collections import namedtuple

_MIN_DEPTH = 1
_MIN_THETA_BUILD = 1

FlatTreeNode = namedtuple("FlatTreeNode", ["node", "left_child_offset",
    "right_child_offset"])


class SubsumptionTree:
    def __init__(self,
                 encoding,
                 phenotypes,
                 max_depth=math.inf,
                 theta_build=math.inf):

        assert max_depth >= _MIN_DEPTH
        self._max_depth = max_depth
        assert theta_build >= _MIN_THETA_BUILD
        self._theta_build = theta_build

        self._encoding = encoding

        self._phenotype_count_map = self._init_phenotype_count_map(phenotypes)

        self._root_node = self._build_tree(
            self._encoding,
            phenotype_set=set(self._phenotype_count_map.keys()),
            max_depth=self._max_depth)

        (self._flat_tree, self._phenotype_node_map
         ) = self._flatten_tree_and_make_phenotype_node_map(self._root_node)

        self._num_updates = 0
        self._last_tree_build_step = 0

    def _init_phenotype_count_map(self, phenotypes):
        phenotype_count_map = {}

        for phenotype in phenotypes:
            try:
                phenotype_count_map[phenotype] += 1
            except KeyError:
                phenotype_count_map[phenotype] = 1

        return phenotype_count_map

    def _build_tree(self, encoding, phenotype_set, max_depth):
        """Recursively builds the subsumption tree in a top-down fashion.

        To do this, at each node, the current phenotype set is taken in, and a
        "split point" is chosen, to create two "split" phenotypes, given the
        subsumer phenotype of the current parent node.

        Then, the phenotype set is partitioned into three groups:

            a) Those that are subsumed by the first split / left child.

            b) Those that are subsumed by the second split / right child.

            c) Those that are neither subsumed by the first nor the second
            split (meaning they are only subsumed by the parent subsumer
            phenotype).

        The phenotypes in case c) are stored as dependents of the current
        (parent) node.

        The splitting procedure then recurses on the sets for a) and b).

        Base cases of the recursion are (checked in this order):

        1. The max depth is reached.
        2. There are either 0 or 1 phenotypes in the node's subsumed set (so no
           more splits are possible).
        """

        root_subsumer = encoding.make_maximally_general_phenotype()

        (split_phenotypes, split_subsumed_sets, split_dim_idx,
         obs_elem_decision_func) = encoding.split_phenotype_set_on_parent(
             parent_subsumer_phenotype=root_subsumer,
             phenotype_set=phenotype_set)

        (split_phenotype_a, split_phenotype_b) = split_phenotypes
        (split_a_subsumed_set, split_b_subsumed_set, root_subsumed_set) = \
            split_subsumed_sets

        depth = 0
        root_left_child_node = self._recursive_build(split_phenotype_a,
                                                     split_a_subsumed_set,
                                                     depth=(depth + 1))
        root_right_child_node = self._recursive_build(split_phenotype_b,
                                                      split_b_subsumed_set,
                                                      depth=(depth + 1))

        return InternalNode(root_subsumer, root_subsumed_set,
                            root_left_child_node, root_right_child_node,
                            split_dim_idx, obs_elem_decision_func)

    def _recursive_build(self, split_phenotype, split_subsumed_set, depth):

        if depth == self._max_depth:
            # terminate recursion: make a leaf node
            return LeafNode(subsumer_phenotype=split_phenotype,
                            subsumed_phenotype_set=split_subsumed_set)

        else:
            # keep on splitting if possible

            if len(split_subsumed_set) < 2:
                # no more splits possible: make a leaf node
                return LeafNode(subsumer_phenotype=split_phenotype,
                                subsumed_phenotype_set=split_subsumed_set)

            else:
                # split via recursion
                parent_subsumer_phenotype = split_phenotype

                (split_phenotypes, split_subsumed_sets, split_dim_idx,
                 obs_elem_decision_func
                 ) = self._encoding.split_phenotype_set_on_parent(
                     parent_subsumer_phenotype,
                     phenotype_set=split_subsumed_set)

                (split_phenotype_a, split_phenotype_b) = split_phenotypes
                (split_a_subsumed_set, split_b_subsumed_set,
                 parent_subsumed_set) = split_subsumed_sets

                left_child_node = self._recursive_build(split_phenotype_a,
                                                        split_a_subsumed_set,
                                                        depth=(depth + 1))
                right_child_node = self._recursive_build(split_phenotype_b,
                                                         split_b_subsumed_set,
                                                         depth=(depth + 1))

                return InternalNode(
                    subsumer_phenotype=parent_subsumer_phenotype,
                    subsumed_phenotype_set=parent_subsumed_set,
                    left_child_node=left_child_node,
                    right_child_node=right_child_node,
                    split_dim_idx=split_dim_idx,
                    obs_elem_decision_func=obs_elem_decision_func)

    def _flatten_tree_and_make_phenotype_node_map(self, root_node):
        flat_tree = []
        phenotype_node_map = {}

        # store flat tree in order of pre-order traversal
        stack = []
        stack.append(root_node)

        while len(stack) > 0:

            node = stack.pop()

            flat_tree.append(node)

            for phenotype in node.subsumed_phenotype_set:
                phenotype_node_map[phenotype] = node

            if not node.is_leaf:
                stack.append(node.right_child_node)
                stack.append(node.left_child_node)

        assert len(phenotype_node_map) == len(self._phenotype_count_map)
        return (flat_tree, phenotype_node_map)

    def gen_sparse_phenotype_matching_map(self, obs):

        sparse_phenotype_matching_map = {}

        curr_node = self._root_node

        if curr_node.size > 0:
            sparse_phenotype_matching_map.update(
                curr_node.gen_phenotype_matching_map(self._encoding, obs))

        # follow a path down the tree, starting at root
        while not curr_node.is_leaf:

            obs_elem = obs[curr_node.split_dim_idx]

            next_node = (curr_node.right_child_node
                         if curr_node.obs_elem_decision_func(obs_elem) else
                         curr_node.left_child_node)

            if next_node.size > 0:
                sparse_phenotype_matching_map.update(
                    next_node.gen_phenotype_matching_map(self._encoding, obs))

            curr_node = next_node

        return sparse_phenotype_matching_map

    def gen_matching_trace(self, obs):

        sparse_phenotype_matching_map = {}
        num_matching_ops_done = 0

        curr_node = self._root_node

        if curr_node.size > 0:
            sparse_phenotype_matching_map.update(
                curr_node.gen_phenotype_matching_map(self._encoding, obs))
            num_matching_ops_done += curr_node.size

        # follow a path down the tree, starting at root
        while not curr_node.is_leaf:

            obs_elem = obs[curr_node.split_dim_idx]

            next_node = (curr_node.right_child_node
                         if curr_node.obs_elem_decision_func(obs_elem) else
                         curr_node.left_child_node)

            if next_node.size > 0:
                sparse_phenotype_matching_map.update(
                    next_node.gen_phenotype_matching_map(self._encoding, obs))
                num_matching_ops_done += next_node.size

            curr_node = next_node

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
            # TODO should check rebuild here?
            self._add_phenotype(phenotype)
            self._num_updates += 1

            self._try_rebuild_tree()

    def _add_phenotype(self, addee):
        curr_node = self._root_node

        subsumer_node = None

        while True:

            if curr_node.is_leaf:
                # can't descend any further
                subsumer_node = curr_node
                break

            if self._encoding.does_subsume(
                    curr_node.left_child_node.subsumer_phenotype, addee):
                # examine left
                curr_node = curr_node.left_child_node

            elif self._encoding.does_subsume(
                    curr_node.right_child_node.subsumer_phenotype, addee):
                # examine right
                curr_node = curr_node.right_child_node

            else:
                # stop here
                subsumer_node = curr_node
                break

        assert subsumer_node is not None

        subsumer_node.add_phenotype(addee)
        self._phenotype_node_map[addee] = subsumer_node

    def try_remove_phenotype(self, phenotype):
        count = self._phenotype_count_map[phenotype]
        count -= 1

        do_remove = (count == 0)

        if do_remove:
            del self._phenotype_count_map[phenotype]
            # TODO should check rebuild here?
            self._remove_phenotype(phenotype)
            self._num_updates += 1

            self._try_rebuild_tree()

        else:
            self._phenotype_count_map[phenotype] = count

    def _remove_phenotype(self, removee):
        subsumer_node = self._phenotype_node_map[removee]
        subsumer_node.remove_phenotype(removee)
        del self._phenotype_node_map[removee]

    def _try_rebuild_tree(self):
        should_rebuild_tree = \
            (self._num_updates - self._last_tree_build_step) >= \
            self._theta_build

        if should_rebuild_tree:
            logging.info(f"Rebuilding subsumption tree @ "
                         f"{self._num_updates} num updates")
            self._last_tree_build_step = self._num_updates

    def _rebuild_tree(self):
        self._root_node = self._build_tree(
            self._encoding,
            phenotype_set=set(self._phenotype_count_map.keys()),
            max_depth=self._max_depth)


class NodeABC(metaclass=abc.ABCMeta):
    def __init__(self, subsumer_phenotype, subsumed_phenotype_set):
        self._subsumer_phenotype = subsumer_phenotype
        self._subsumed_phenotype_set = subsumed_phenotype_set

        self._size = len(self._subsumed_phenotype_set)

    @property
    def subsumer_phenotype(self):
        """The bounding volume of the node."""
        return self._subsumer_phenotype

    @property
    def subsumed_phenotype_set(self):
        """The set of phenotypes that are subsumed by this node's bounding
        volume, but not by the bounding volumes of either of its children (if
        it has children, i.e. is not a leaf node)."""
        return self._subsumed_phenotype_set

    @property
    def size(self):
        return self._size

    @property
    @abc.abstractmethod
    def is_leaf(self):
        raise NotImplementedError

    def gen_phenotype_matching_map(self, encoding, obs):
        return {
            phenotype: encoding.does_phenotype_match(phenotype, obs)
            for phenotype in self._subsumed_phenotype_set
        }

    def add_phenotype(self, addee):
        self._subsumed_phenotype_set.add(addee)
        self._size += 1

    def remove_phenotype(self, removee):
        self._subsumed_phenotype_set.remove(removee)
        self._size -= 1
        assert self._size >= 0


class InternalNode(NodeABC):
    def __init__(self, subsumer_phenotype, subsumed_phenotype_set,
                 left_child_node, right_child_node, split_dim_idx,
                 obs_elem_decision_func):

        super().__init__(subsumer_phenotype, subsumed_phenotype_set)

        self._left_child_node = left_child_node
        self._right_child_node = right_child_node
        self._split_dim_idx = split_dim_idx
        self._obs_elem_decision_func = obs_elem_decision_func

    @property
    def left_child_node(self):
        return self._left_child_node

    @property
    def right_child_node(self):
        return self._right_child_node

    @property
    def split_dim_idx(self):
        return self._split_dim_idx

    @property
    def obs_elem_decision_func(self):
        return self._obs_elem_decision_func

    @property
    def is_leaf(self):
        return False


class LeafNode(NodeABC):
    @property
    def is_leaf(self):
        return True
