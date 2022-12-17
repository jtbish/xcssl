import abc

from .tree import SubsumptionTree


class PopulationABC(metaclass=abc.ABCMeta):
    """Population is just a list of macroclassifiers with tracking of the number of
    microclassifiers, in order to avoid having to calculate this number on the
    fly repeatedly and waste time."""
    @property
    def num_macros(self):
        return len(self._clfrs)

    @property
    def num_micros(self):
        return self._num_micros

    @property
    def ops_history(self):
        return self._ops_history

    def alter_numerosity(self, clfr, delta, op):
        clfr.numerosity += delta
        self._num_micros += delta
        assert op in ("absorption", "deletion", "ga_subsumption",
                      "as_subsumption")
        # delta can be neg. (obviously), but op. counts are pos.
        self._ops_history[op] += abs(delta)

    def add_new(self, clfr, op, time_step=None):
        self._clfrs.append(clfr)
        self._num_micros += clfr.numerosity
        assert op in ("covering", "insertion")
        self._ops_history[op] += clfr.numerosity

    def remove(self, clfr, op=None):
        self._clfrs.remove(clfr)
        self._num_micros -= clfr.numerosity
        if op is not None:
            assert op == "deletion"
            self._ops_history[op] += clfr.numerosity

    def __iter__(self):
        return iter(self._clfrs)

    def __getitem__(self, idx):
        return self._clfrs[idx]

    @abc.abstractmethod
    def gen_match_set(self, obs):
        raise NotImplementedError


class VanillaPopulation(PopulationABC):
    """Default-style population that does not use phenotype clustering and
    perfoms fully accurate and exhuastive matching."""
    def __init__(self):
        self._clfrs = []
        self._num_micros = 0
        self._ops_history = {
            "covering": 0,
            "absorption": 0,
            "insertion": 0,
            "deletion": 0,
            "ga_subsumption": 0,
            "as_subsumption": 0
        }

    def gen_match_set(self, obs):
        """Full and exhaustive matching procedure: match each
        macroclassifier."""
        return [clfr for clfr in self._clfrs if clfr.does_match(obs)]

    def gen_matching_trace(self, obs):
        return [clfr.does_match(obs) for clfr in self._clfrs]


class FastMatchingPopulation(PopulationABC):
    """Population that uses an index to perform fast matching."""
    def __init__(self, vanilla_pop, encoding, lsh):
        """FastMatchingPopulation needs to be inited from existing
        VanillaPopulation."""
        assert isinstance(vanilla_pop, VanillaPopulation)

        self._clfrs = vanilla_pop._clfrs
        self._num_micros = vanilla_pop._num_micros
        self._ops_history = vanilla_pop._ops_history

        # make the phenotype index
        # first, vectorise all the condition phenotype in the pop
        # so that distance calcs can be done for them
        self._vectorise_clfr_phenotypes(self._clfrs)

        self._index = SubsumptionTree(
            encoding=encoding,
            lsh=lsh,
            phenotypes=[clfr.condition.phenotype for clfr in self._clfrs])

    def _vectorise_clfr_phenotypes(self, clfrs):
        """Update phenotypes of clfr conditions in place with vectorised
        variants."""
        for clfr in clfrs:
            clfr.condition.vectorise_phenotype()

    def add_new(self, clfr, op, time_step=None):
        super().add_new(clfr, op, time_step)

        self._index.try_add_phenotype(clfr.condition.phenotype)

    def remove(self, clfr, op=None):
        super().remove(clfr, op)

        self._index.try_remove_phenotype(clfr.condition.phenotype)

    def gen_match_set(self, obs):
        (phenotype_matching_map, _) = \
            self._index.gen_phenotype_matching_map(obs)
        return [
            clfr for clfr in self._clfrs
            if phenotype_matching_map[clfr.condition.phenotype]
        ]

    def gen_matching_trace(self, obs):
        (phenotype_matching_map, num_matching_ops_done) = \
            self._index.gen_phenotype_matching_map(obs)
        trace = [
            phenotype_matching_map[clfr.condition.phenotype]
            for clfr in self._clfrs
        ]
        return (trace, num_matching_ops_done)
