import abc
import math

_MIN_DEPTH = 1


class SubsumptionTree:
    def __init__(self,
                 encoding,
                 phenotypes,
                 max_depth=math.inf,
                 theta_tree=None):
        self._encoding = encoding

        self._phenotype_count_map = self._init_phenotype_count_map(phenotypes)

        assert max_depth >= _MIN_DEPTH
        self._max_depth = max_depth

        self._root_node = self._build_tree(
            self._encoding,
            phenotype_set=set(self._phenotype_count_map.keys()),
            max_depth=self._max_depth)

        self._num_adds_and_removes = 0
        self._last_build_tree_step = 0
        self._theta_tree = theta_tree

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
            a) Those that are neither subsumed by the first nor the second
            split (meaning they are only subsumed by the parent subsumer
            phenotype).
            b) Those that are subsumed by the first split / left child.
            c) Those that are subsumed by the second split / right child.

        The phenotypes in case a) are stored as dependents of the current
        (parent) node.

        The splitting procedure then recurses on the sets for b) and c).

        Base cases of the recursion are (checked in this order):

        1. The max depth is reached.
        2. There are either 0 or 1 phenotypes in the node's subsumed set (so no
           more splits are possible).

        """
        root_subsumer = encoding.make_maximally_general_phenotype()

        (split_phenotypes,
         split_subsumed_sets) = encoding.split_phenotype_set_on_parent(
             parent_subsumer_phenotype=root_subsumer,
             phenotype_set=phenotype_set)

        (split_phenotype_a, split_phenotype_b) = split_phenotypes
        (root_subsumed_set, split_a_subsumed_set, split_b_subsumed_set) = \
            split_subsumed_sets

        depth = 0
        root_left_child_node = self._recursive_build(encoding,
                                                     split_phenotype_a,
                                                     split_a_subsumed_set,
                                                     depth=(depth + 1),
                                                     max_depth=max_depth)
        root_right_child_node = self._recursive_build(encoding,
                                                      split_phenotype_b,
                                                      split_b_subsumed_set,
                                                      depth=(depth + 1),
                                                      max_depth=max_depth)

        return RootNode(root_subsumer, root_subsumed_set, root_left_child_node,
                        root_right_child_node)

    def _recursive_build(self, encoding, split_phenotype, split_subsumed_set,
                         depth, max_depth):

        if depth == max_depth:
            # terminate recursion: make a leaf node
            return LeafNode(subsumer_phenotype=split_phenotype,
                            subsumed_phenotype_set=split_subsumed_set,
                            depth=depth)

        else:
            # keep on splitting if possible
            size = len(split_subsumed_set)

            if (size == 0 or size == 1):
                # no more splits possible: make a leaf node
                return LeafNode(subsumer_phenotype=split_phenotype,
                                subsumed_phenotype_set=split_subsumed_set,
                                depth=depth)

            else:
                # split via recursion
                parent_subsumer_phenotype = split_phenotype

                (split_phenotypes,
                 split_subsumed_sets) = encoding.split_phenotype_set_on_parent(
                     parent_subsumer_phenotype,
                     phenotype_set=split_subsumed_set)

                (split_phenotype_a, split_phenotype_b) = split_phenotypes
                (parent_subsumed_set, split_a_subsumed_set,
                 split_b_subsumed_set) = split_subsumed_sets

                left_child_node = self._recursive_build(encoding,
                                                        split_phenotype_a,
                                                        split_a_subsumed_set,
                                                        depth=(depth + 1),
                                                        max_depth=max_depth)
                right_child_node = self._recursive_build(encoding,
                                                         split_phenotype_b,
                                                         split_b_subsumed_set,
                                                         depth=(depth + 1),
                                                         max_depth=max_depth)

                return InternalNode(
                    subsumer_phenotype=parent_subsumer_phenotype,
                    subsumed_phenotype_set=parent_subsumed_set,
                    depth=depth,
                    left_child_node=left_child_node,
                    right_child_node=right_child_node)

    def gen_sparse_phenotype_matching_map(self, obs):

        sparse_phenotype_matching_map = {}

        root = self._root_node

        if root.size > 0:
            sparse_phenotype_matching_map.update(
                root.gen_phenotype_matching_map(self._encoding, obs))

        stack = []
        stack.append(root.right_child_node)
        stack.append(root.left_child_node)

        while len(stack) > 0:

            node = stack.pop()
            subsumer = node.subsumer_phenotype

            if not node.is_leaf:

                subsumer_does_match = self._encoding.does_phenotype_match(
                    subsumer, obs)

                if subsumer_does_match:

                    # match any subsumed phenotypes then add children to
                    # traversal
                    if node.size > 0:
                        sparse_phenotype_matching_map.update(
                            node.gen_phenotype_matching_map(
                                self._encoding, obs))

                    stack.append(node.right_child_node)
                    stack.append(node.left_child_node)

            else:
                size = node.size

                if size == 1:
                    # match the sole subsumed phenotype of the leaf node
                    # directly, without first checking the subsumer phenotype
                    sparse_phenotype_matching_map.update(
                        node.gen_phenotype_matching_map(self._encoding, obs))

                elif size > 1:
                    # match subsumed phenotypes only if subsumer matches
                    subsumer_does_match = self._encoding.does_phenotype_match(
                        subsumer, obs)

                    if subsumer_does_match:

                        sparse_phenotype_matching_map.update(
                            node.gen_phenotype_matching_map(
                                self._encoding, obs))

                # else if size is 0 nothing to do!

        return sparse_phenotype_matching_map

    def gen_matching_trace(self, obs):

        sparse_phenotype_matching_map = {}
        num_matching_ops_done = 0

        root = self._root_node

        if root.size > 0:
            sparse_phenotype_matching_map.update(
                root.gen_phenotype_matching_map(self._encoding, obs))
            num_matching_ops_done += root.size

        stack = []
        stack.append(root.right_child_node)
        stack.append(root.left_child_node)

        while len(stack) > 0:

            node = stack.pop()
            subsumer = node.subsumer_phenotype

            if not node.is_leaf:

                subsumer_does_match = self._encoding.does_phenotype_match(
                    subsumer, obs)
                num_matching_ops_done += 1

                if subsumer_does_match:

                    # match any subsumed phenotypes then add children to
                    # traversal
                    if node.size > 0:
                        sparse_phenotype_matching_map.update(
                            node.gen_phenotype_matching_map(
                                self._encoding, obs))
                        num_matching_ops_done += node.size

                    stack.append(node.right_child_node)
                    stack.append(node.left_child_node)

            else:
                size = node.size

                if size == 1:
                    # match the sole subsumed phenotype of the leaf node
                    # directly, without first checking the subsumer phenotype
                    sparse_phenotype_matching_map.update(
                        node.gen_phenotype_matching_map(self._encoding, obs))
                    num_matching_ops_done += 1

                elif size > 1:
                    # match subsumed phenotypes only if subsumer matches
                    subsumer_does_match = self._encoding.does_phenotype_match(
                        subsumer, obs)
                    num_matching_ops_done += 1

                    if subsumer_does_match:

                        sparse_phenotype_matching_map.update(
                            node.gen_phenotype_matching_map(
                                self._encoding, obs))
                        num_matching_ops_done += size

                # else if size is 0 nothing to do!

        return (sparse_phenotype_matching_map, num_matching_ops_done)


class NodeABC(metaclass=abc.ABCMeta):
    def __init__(self, subsumer_phenotype, subsumed_phenotype_set, depth):
        self._subsumer_phenotype = subsumer_phenotype
        self._subsumed_phenotype_set = subsumed_phenotype_set
        self._depth = depth

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
    def depth(self):
        return self._depth

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


class RootNode(NodeABC):
    _DEPTH = 0

    def __init__(self, maximally_general_phenotype, subsumed_phenotype_set,
                 left_child_node, right_child_node):

        super().__init__(subsumer_phenotype=maximally_general_phenotype,
                         subsumed_phenotype_set=subsumed_phenotype_set,
                         depth=self._DEPTH)

        self._left_child_node = left_child_node
        self._right_child_node = right_child_node

    @property
    def left_child_node(self):
        return self._left_child_node

    @property
    def right_child_node(self):
        return self._right_child_node

    @property
    def is_leaf(self):
        return False


class InternalNode(NodeABC):
    def __init__(self, subsumer_phenotype, subsumed_phenotype_set, depth,
                 left_child_node, right_child_node):

        super().__init__(subsumer_phenotype, subsumed_phenotype_set, depth)

        self._left_child_node = left_child_node
        self._right_child_node = right_child_node

    @property
    def left_child_node(self):
        return self._left_child_node

    @property
    def right_child_node(self):
        return self._right_child_node

    @property
    def is_leaf(self):
        return False


class LeafNode(NodeABC):
    @property
    def is_leaf(self):
        return True
