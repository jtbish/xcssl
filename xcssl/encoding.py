import abc

import numpy as np

from .condition import TERNARY_HASH, IntervalCondition, TernaryCondition
from .hyperparams import get_hyperparam as get_hp
from .interval import IntegerInterval, RealInterval
from .obs_space import IntegerObsSpace, RealObsSpace
from .phenotype import VanillaPhenotype, VectorisedPhenotype
from .rng import get_rng


class EncodingABC(metaclass=abc.ABCMeta):
    def __init__(self, obs_space, use_fast_approx_matching):
        self._obs_space = obs_space
        self._use_fast_approx_matching = use_fast_approx_matching

    @property
    def obs_space(self):
        return self._obs_space

    def make_condition(self, cond_alleles):
        return self._COND_CLS(cond_alleles, self)

    def decode(self, cond_alleles):
        phenotype_elems = self._decode(cond_alleles)
        if not self._use_fast_approx_matching:
            return VanillaPhenotype(phenotype_elems)
        else:
            phenotype_vec = self.gen_phenotype_vec(phenotype_elems)
            return VectorisedPhenotype(phenotype_elems, phenotype_vec)

    @abc.abstractmethod
    def gen_covering_condition(self, obs):
        raise NotImplementedError

    @abc.abstractmethod
    def _decode(self, cond_alleles):
        """Convert genotype to phenotype."""
        raise NotImplementedError

    @abc.abstractmethod
    def calc_condition_generality(self, phenotype):
        raise NotImplementedError

    @abc.abstractmethod
    def mutate_condition_alleles(self, cond_alleles, obs=None):
        raise NotImplementedError

    @abc.abstractmethod
    def does_phenotype_match(self, phenotype, obs):
        raise NotImplementedError

    @abc.abstractmethod
    def calc_num_phenotype_vec_dims(self):
        raise NotImplementedError

    @abc.abstractmethod
    def gen_phenotype_vec(self, phenotype_elems):
        """Generate vectorised repr of phenotype."""
        raise NotImplementedError

    @abc.abstractmethod
    def distance_between(self, phenotype_vec_a, phenotype_vec_b):
        raise NotImplementedError


class TernaryEncoding(EncodingABC):
    _COND_CLS = TernaryCondition

    def __init__(self, obs_space, use_fast_approx_matching):
        assert isinstance(obs_space, IntegerObsSpace)
        # check is actually binary
        for dim in obs_space.dims:
            assert dim.lower == 0
            assert dim.upper == 1
        super().__init__(obs_space, use_fast_approx_matching)

    def gen_covering_condition(self, obs):
        num_alleles = len(self._obs_space)
        assert len(obs) == num_alleles

        cond_alleles = []
        for obs_compt in obs:
            if get_rng().random() < get_hp("p_hash"):
                cond_alleles.append(TERNARY_HASH)
            else:
                cond_alleles.append(obs_compt)

        assert len(cond_alleles) == num_alleles
        return self.make_condition(cond_alleles)

    def _decode(self, cond_alleles):
        # genotype == phenotype
        return tuple(cond_alleles)

    def calc_condition_generality(self, phenotype):
        """Number of don't care elems."""
        return phenotype.elems.count(TERNARY_HASH)

    def mutate_condition_alleles(self, cond_alleles, obs):
        mut_alleles = []
        for (allele, obs_compt) in zip(cond_alleles, obs):
            if get_rng().random() < get_hp("mu"):
                if allele == TERNARY_HASH:
                    mut_alleles.append(obs_compt)
                else:
                    mut_alleles.append(TERNARY_HASH)
            else:
                mut_alleles.append(allele)
        assert len(mut_alleles) == len(cond_alleles)
        return mut_alleles

    def does_phenotype_match(self, phenotype, obs):
        for (obs_compt, elem) in zip(obs, phenotype):
            if (elem != TERNARY_HASH and elem != obs_compt):
                return False
        return True

    def calc_num_phenotype_vec_dims(self):
        return (2 * len(self._obs_space))

    def gen_phenotype_vec(self, phenotype_elems):
        vec = []

        for elem in phenotype_elems:
            # one-hot encoding of elem vals, hash counts as both 0 and 1
            if elem == 0:
                val = [1, 0]
            elif elem == 1:
                val = [0, 1]
            elif elem == TERNARY_HASH:
                val = [1, 1]
            else:
                assert False

            vec.extend(val)

        return np.asarray(vec, dtype=np.uint8)

    def distance_between(self, phenotype_vec_a, phenotype_vec_b):
        """Hamming dist."""
        return sum(e_a != e_b
                   for (e_a, e_b) in zip(phenotype_vec_a, phenotype_vec_b))


class UnorderedBoundEncodingABC(EncodingABC, metaclass=abc.ABCMeta):
    _COND_CLS = IntervalCondition
    _GENERALITY_UB_INCL = 1.0

    def gen_covering_condition(self, obs):
        num_alleles = len(self._obs_space) * 2
        cond_alleles = []
        assert len(obs) == len(self._obs_space)
        for (obs_compt, dim) in zip(obs, self._obs_space):
            (lower, upper) = self._gen_covering_alleles(obs_compt, dim)
            cover_alleles = [lower, upper]
            # to avoid bias, insert alleles into genotype in random order
            get_rng().shuffle(cover_alleles)
            for allele in cover_alleles:
                cond_alleles.append(allele)
        assert len(cond_alleles) == num_alleles
        return self.make_condition(cond_alleles)

    @abc.abstractmethod
    def _gen_covering_alleles(self, obs_compt, dim):
        """Return (lower, upper) covering alleles, with lower <= upper."""
        raise NotImplementedError

    def _decode(self, cond_alleles):
        phenotype = []
        assert len(cond_alleles) % 2 == 0
        for i in range(0, len(cond_alleles), 2):
            first_allele = cond_alleles[i]
            second_allele = cond_alleles[i + 1]
            lower = min(first_allele, second_allele)
            upper = max(first_allele, second_allele)
            phenotype.append(self._INTERVAL_CLS(lower, upper))
        assert len(phenotype) == len(cond_alleles) // 2
        return phenotype

    @abc.abstractmethod
    def calc_condition_generality(self, cond_intervals):
        raise NotImplementedError

    def mutate_condition_alleles(self, cond_alleles, obs=None):
        alleles = cond_alleles
        assert len(alleles) % 2 == 0
        allele_pairs = [(alleles[i], alleles[i + 1])
                        for i in range(0, len(alleles), 2)]
        mut_alleles = []
        for (allele_pair, dim) in zip(allele_pairs, self._obs_space):
            for allele in allele_pair:
                if get_rng().random() < get_hp("mu"):
                    noise = self._gen_mutation_noise(dim)
                    sign = get_rng().choice([-1, 1])
                    mut_allele = allele + (sign * noise)
                    mut_allele = max(mut_allele, dim.lower)
                    mut_allele = min(mut_allele, dim.upper)
                    mut_alleles.append(mut_allele)
                else:
                    mut_alleles.append(allele)
        assert len(mut_alleles) == len(alleles)
        return mut_alleles

    @abc.abstractmethod
    def _gen_mutation_noise(self, dim=None):
        raise NotImplementedError

    def does_phenotype_match(self, phenotype, obs):
        for (obs_compt, interval) in zip(obs, phenotype):
            if not interval.contains_val(obs_compt):
                return False
        return True

    def calc_num_phenotype_vec_dims(self):
        raise NotImplementedError

    def gen_phenotype_vec(self, phenotype_elems):
        raise NotImplementedError

    def distance_between(self, phenotype_vec_a, phenotype_vec_b):
        raise NotImplementedError


class IntegerUnorderedBoundEncoding(UnorderedBoundEncodingABC):
    _GENERALITY_LB_EXCL = 0
    _INTERVAL_CLS = IntegerInterval

    def __init__(self, obs_space, use_fast_approx_matching):
        assert isinstance(obs_space, IntegerObsSpace)
        super().__init__(obs_space, use_fast_approx_matching)

    def _gen_covering_alleles(self, obs_compt, dim):
        r_nought = get_hp("r_nought")
        # rand integer ~ [0, r_nought]
        lower = obs_compt - get_rng().randint(low=0, high=(r_nought + 1))
        upper = obs_compt + get_rng().randint(low=0, high=(r_nought + 1))
        lower = max(lower, dim.lower)
        upper = min(upper, dim.upper)
        return (lower, upper)

    def calc_condition_generality(self, phenotype):
        # condition generality calc as in
        # Wilson '00 Mining Oblique Data with XCS
        cond_intervals = phenotype
        numer = sum([interval.span for interval in cond_intervals])
        denom = sum([dim.span for dim in self._obs_space])
        generality = numer / denom
        # b.c. of +1s in numer, gen cannot be 0
        assert self._GENERALITY_LB_EXCL < generality <= \
            self._GENERALITY_UB_INCL
        return generality

    def _gen_mutation_noise(self, dim=None):
        # integer ~ [1, m_0]
        return get_rng().randint(low=1, high=(get_hp("m_nought") + 1))


class RealUnorderedBoundEncoding(UnorderedBoundEncodingABC):
    _GENERALITY_LB_INCL = 0
    _INTERVAL_CLS = RealInterval

    def __init__(self, obs_space, use_fast_approx_matching):
        assert isinstance(obs_space, RealObsSpace)
        super().__init__(obs_space, use_fast_approx_matching)

    def _gen_covering_alleles(self, obs_compt, dim):
        # r_0 interpreted as fraction of dim span to draw uniform random noise
        # from
        r_nought = get_hp("r_nought")
        assert 0.0 < r_nought <= 1.0
        cover_high = (r_nought * dim.span)
        lower = obs_compt - get_rng().uniform(low=0, high=cover_high)
        upper = obs_compt + get_rng().uniform(low=0, high=cover_high)
        lower = max(lower, dim.lower)
        upper = min(upper, dim.upper)
        return (lower, upper)

    def calc_condition_generality(self, phenotype):
        cond_intervals = phenotype
        numer = sum([interval.span for interval in cond_intervals])
        denom = sum([dim for dim in self._obs_space])
        generality = numer / denom
        # gen could be 0 if all intervals in numer collapse to single point
        assert self._GENERALITY_LB_INCL <= generality <= \
            self._GENERALITY_UB_INCL
        return generality

    def _gen_mutation_noise(self, dim):
        # m_0 interpreted as fraction of dim span to draw uniform random
        # noise from
        m_nought = get_hp("m_nought")
        assert 0.0 < m_nought <= 1.0
        mut_high = (m_nought * dim.span)
        return get_rng().uniform(low=0, high=mut_high)
