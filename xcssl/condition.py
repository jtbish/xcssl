import abc

TERNARY_HASH = "#"


class ConditionABC(metaclass=abc.ABCMeta):
    def __init__(self, alleles, encoding):
        self._alleles = list(alleles)
        self._encoding = encoding
        self._phenotype = self._encoding.decode(self._alleles)
        self._generality = self._encoding.calc_condition_generality(
            self._phenotype)

    @property
    def alleles(self):
        return self._alleles

    @property
    def phenotype(self):
        return self._phenotype

    @property
    def generality(self):
        return self._generality

    def __eq__(self, other):
        return self._alleles == other._alleles

    def __len__(self):
        return len(self._phenotype)

    @abc.abstractmethod
    def does_match(self, obs):
        raise NotImplementedError

    @abc.abstractmethod
    def does_subsume(self, other):
        """Does this condition subsume other condition?"""
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class TernaryCondition(ConditionABC):
    def does_match(self, obs):
        for (obs_compt, elem) in zip(obs, self._phenotype):
            if elem != TERNARY_HASH and elem != obs_compt:
                return False
        return True

    def does_subsume(self, other):
        for (my_elem, other_elem) in zip(self._phenotype, other._phenotype):
            if my_elem != TERNARY_HASH and my_elem != other_elem:
                return False
        return True

    def __str__(self):
        return " ".join([str(elem) for elem in self._phenotype])


class IntervalCondition(ConditionABC):
    def does_match(self, obs):
        for (obs_compt, interval) in zip(obs, self._phenotype):
            if not interval.contains_val(obs_compt):
                return False
        return True

    def does_subsume(self, other):
        for (my_interval, other_interval) in zip(self._phenotype,
                                                 other._phenotype):
            if not my_interval.does_subsume(other_interval):
                return False
        return True

    def __str__(self):
        return " && ".join([str(interval) for interval in self._phenotype])
