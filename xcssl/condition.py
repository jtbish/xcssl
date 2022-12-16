import abc

from .phenotype import VanillaPhenotype, VectorisedPhenotype

TERNARY_HASH = "#"


class ConditionABC(metaclass=abc.ABCMeta):
    def __init__(self, alleles, encoding):
        # alleles == genotype
        self._alleles = list(alleles)
        self._encoding = encoding

        self._phenotype = self._encoding.decode(self._alleles)

    @property
    def alleles(self):
        return self._alleles

    @property
    def phenotype(self):
        return self._phenotype

    @property
    def generality(self):
        return self._phenotype.generality

    def does_match(self, obs):
        return self._encoding.does_phenotype_match(self._phenotype, obs)

    def vectorise_phenotype(self):
        """Converts the current VanillaPhenotype of this condition to a
        VectorisedPhenotype."""
        assert isinstance(self._phenotype, VanillaPhenotype)

        elems = self._phenotype.elems
        genr = self._phenotype.generality
        phenotype_vec = \
            self._encoding.gen_phenotype_vec(elems)
        self._phenotype = VectorisedPhenotype(elems, genr, phenotype_vec)

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
