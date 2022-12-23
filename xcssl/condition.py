import abc

from .phenotype import IndexablePhenotype, VanillaPhenotype

TERNARY_HASH = "#"


class ConditionABC(metaclass=abc.ABCMeta):
    def __init__(self, alleles, encoding):
        # alleles == genotype
        self._alleles = list(alleles)
        self._encoding = encoding

        self._phenotype = self._encoding.decode(self._alleles)
        self._generality = \
            self._encoding.calc_phenotype_generality(self._phenotype)

    @property
    def alleles(self):
        return self._alleles

    @property
    def phenotype(self):
        return self._phenotype

    @property
    def generality(self):
        return self._generality

    def does_match(self, obs):
        return self._encoding.does_phenotype_match(self._phenotype, obs)

    def convert_to_indexable_phenotype(self):
        """Converts the current VanillaPhenotype of this condition to an
        IndexablePhenotype."""
        assert isinstance(self._phenotype, VanillaPhenotype)

        elems = self._phenotype.elems
        self._phenotype = IndexablePhenotype(elems)

    def __eq__(self, other):
        return self.phenotype == other.phenotype

    def __len__(self):
        return len(self._phenotype)

    def does_subsume(self, other):
        """Does this condition subsume other condition?"""
        return self._encoding.does_subsume(self._phenotype, other._phenotype)

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class TernaryCondition(ConditionABC):
    def __str__(self):
        return " ".join([str(elem) for elem in self._phenotype])


class IntervalCondition(ConditionABC):
    def __str__(self):
        return " && ".join([str(interval) for interval in self._phenotype])
