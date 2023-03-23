import abc

from .aabb import AxisAlignedBoundingBox
from .condition import TERNARY_HASH, IntervalCondition, TernaryCondition
from .hyperparams import get_hyperparam as get_hp
from .interval import IntegerInterval, RealInterval
from .obs_space import IntegerObsSpace, RealObsSpace
from .phenotype import Phenotype
from .rng import get_rng


class EncodingABC(metaclass=abc.ABCMeta):
    def __init__(self, obs_space):
        self._obs_space = obs_space
        self._num_obs_dims = len(self._obs_space)

    @property
    def obs_space(self):
        return self._obs_space

    def make_condition(self, cond_alleles):
        return self._COND_CLS(cond_alleles, self)

    def decode(self, cond_alleles):
        phenotype_elems = self._decode(cond_alleles)
        return Phenotype(phenotype_elems)

    @abc.abstractmethod
    def gen_covering_condition(self, obs):
        raise NotImplementedError

    @abc.abstractmethod
    def _decode(self, cond_alleles):
        """Convert genotype to phenotype."""
        raise NotImplementedError

    @abc.abstractmethod
    def calc_phenotype_generality(self, phenotype):
        raise NotImplementedError

    @abc.abstractmethod
    def mutate_condition_alleles(self, cond_alleles, obs=None):
        raise NotImplementedError

    @abc.abstractmethod
    def does_phenotype_match(self, phenotype, obs):
        raise NotImplementedError

    @abc.abstractmethod
    def does_subsume(self, phenotype_a, phenotype_b):
        """Does phenotype_a subsume phenotype_b?"""
        raise NotImplementedError

    @abc.abstractmethod
    def make_phenotype_aabb(self, phenotype):
        raise NotImplementedError


class TernaryEncoding(EncodingABC):
    _COND_CLS = TernaryCondition

    _PHENOTYPE_ELEM_BOUNDING_INTERVALS = {
        0: IntegerInterval(0, 0),
        1: IntegerInterval(1, 1),
        TERNARY_HASH: IntegerInterval(0, 1)
    }

    def __init__(self, obs_space):
        assert isinstance(obs_space, IntegerObsSpace)
        # check is actually binary
        for dim in obs_space.dims:
            assert dim.lower == 0
            assert dim.upper == 1
        super().__init__(obs_space)

    def gen_covering_condition(self, obs):
        assert len(obs) == self._num_obs_dims

        p_hash = get_hp("p_hash")
        cond_alleles = []

        for obs_compt in obs:
            if get_rng().random() < p_hash:
                cond_alleles.append(TERNARY_HASH)
            else:
                cond_alleles.append(obs_compt)

        assert len(cond_alleles) == self._num_obs_dims
        return self.make_condition(cond_alleles)

    def _decode(self, cond_alleles):
        # phenotype == genotype
        return tuple(cond_alleles)

    def calc_phenotype_generality(self, phenotype):
        """Number of don't care elems."""
        return phenotype.elems.count(TERNARY_HASH)

    def mutate_condition_alleles(self, cond_alleles, obs):
        assert len(cond_alleles) == self._num_obs_dims

        mu = get_hp("mu")
        mut_alleles = []

        for (allele, obs_compt) in zip(cond_alleles, obs):
            if get_rng().random() < mu:
                if allele == TERNARY_HASH:
                    mut_alleles.append(obs_compt)
                else:
                    mut_alleles.append(TERNARY_HASH)
            else:
                mut_alleles.append(allele)

        assert len(mut_alleles) == self._num_obs_dims
        return mut_alleles

    def does_phenotype_match(self, phenotype, obs):
        for (obs_compt, elem) in zip(obs, phenotype):
            if (elem != TERNARY_HASH and elem != obs_compt):
                return False
        return True

    def does_subsume(self, phenotype_a, phenotype_b):
        for (a_elem, b_elem) in zip(phenotype_a, phenotype_b):
            if (a_elem != TERNARY_HASH and a_elem != b_elem):
                return False
        return True

    def make_phenotype_aabb(self, phenotype):
        return AxisAlignedBoundingBox([
            self._PHENOTYPE_ELEM_BOUNDING_INTERVALS[elem] for elem in phenotype
        ])


class UnorderedBoundEncodingABC(EncodingABC, metaclass=abc.ABCMeta):
    _COND_CLS = IntervalCondition
    _GENERALITY_UB_INCL = 1.0

    def __init__(self, obs_space):
        super().__init__(obs_space)
        self._num_cond_alleles = (self._num_obs_dims * 2)
        self._phenotype_generality_denom = sum(dim.span
                                               for dim in self._obs_space)

    def gen_covering_condition(self, obs):
        assert len(obs) == self._num_obs_dims

        cond_alleles = []
        for (obs_compt, dim) in zip(obs, self._obs_space):
            (lower, upper) = self._gen_covering_alleles(obs_compt, dim)
            cover_alleles = [lower, upper]
            # to avoid bias, insert alleles into genotype in random order
            get_rng().shuffle(cover_alleles)
            cond_alleles.extend(cover_alleles)

        assert len(cond_alleles) == self._num_cond_alleles
        return self.make_condition(cond_alleles)

    @abc.abstractmethod
    def _gen_covering_alleles(self, obs_compt, dim):
        """Return (lower, upper) covering alleles, with lower <= upper."""
        raise NotImplementedError

    def _decode(self, cond_alleles):
        assert len(cond_alleles) == self._num_cond_alleles

        phenotype = []
        for i in range(0, len(cond_alleles), 2):
            first_allele = cond_alleles[i]
            second_allele = cond_alleles[i + 1]
            lower = min(first_allele, second_allele)
            upper = max(first_allele, second_allele)
            phenotype.append(self._INTERVAL_CLS(lower, upper))

        assert len(phenotype) == self._num_obs_dims
        return phenotype

    @abc.abstractmethod
    def calc_phenotype_generality(self, phenotype):
        raise NotImplementedError

    def mutate_condition_alleles(self, cond_alleles, obs=None):
        assert len(cond_alleles) == self._num_cond_alleles

        mu = get_hp("mu")
        mut_alleles = []
        i = 0
        for dim in self._obs_space:

            # independently mutate each of the 2 alleles
            # that pertain to this dim
            for allele in (cond_alleles[i], cond_alleles[i + 1]):

                if get_rng().random() < mu:
                    noise = self._gen_mutation_noise(dim)
                    sign = get_rng().choice([-1, 1])
                    mut_allele = allele + (sign * noise)
                    mut_allele = max(mut_allele, dim.lower)
                    mut_allele = min(mut_allele, dim.upper)
                    mut_alleles.append(mut_allele)
                else:
                    mut_alleles.append(allele)

            i += 2

        assert len(mut_alleles) == self._num_cond_alleles
        return mut_alleles

    @abc.abstractmethod
    def _gen_mutation_noise(self, dim=None):
        raise NotImplementedError

    def does_phenotype_match(self, phenotype, obs):
        for (obs_compt, interval) in zip(obs, phenotype):
            if not interval.contains_val(obs_compt):
                return False
        return True

    def does_subsume(self, phenotype_a, phenotype_b):
        for (a_interval, b_interval) in zip(phenotype_a, phenotype_b):
            if not a_interval.does_subsume(b_interval):
                return False
        return True

    def make_phenotype_aabb(self, phenotype):
        # phenotype is already an AABB!
        return AxisAlignedBoundingBox([interval for interval in phenotype])


class IntegerUnorderedBoundEncoding(UnorderedBoundEncodingABC):
    _GENERALITY_LB_EXCL = 0
    _INTERVAL_CLS = IntegerInterval

    def __init__(self, obs_space):
        assert isinstance(obs_space, IntegerObsSpace)
        super().__init__(obs_space)

    def _gen_covering_alleles(self, obs_compt, dim):
        r_nought = get_hp("r_nought")
        # rand integer ~ [0, r_nought]
        lower = obs_compt - get_rng().randint(low=0, high=(r_nought + 1))
        upper = obs_compt + get_rng().randint(low=0, high=(r_nought + 1))
        lower = max(lower, dim.lower)
        upper = min(upper, dim.upper)
        return (lower, upper)

    def calc_phenotype_generality(self, phenotype):
        # condition generality calc as in
        # Wilson '00 Mining Oblique Data with XCS
        numer = sum(interval.calc_span() for interval in phenotype.elems)
        generality = (numer / self._phenotype_generality_denom)
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
    _MUT_MEAN = 0.0

    def __init__(self, obs_space):
        assert isinstance(obs_space, RealObsSpace)
        super().__init__(obs_space)

    def _gen_covering_alleles(self, obs_compt, dim):
        # r_0 interpreted as fraction of dim span to draw uniform random noise
        # from
        r_nought = get_hp("r_nought")
        cover_high = (r_nought * dim.span)
        lower = obs_compt - get_rng().uniform(low=0, high=cover_high)
        upper = obs_compt + get_rng().uniform(low=0, high=cover_high)
        lower = max(lower, dim.lower)
        upper = min(upper, dim.upper)
        return (lower, upper)

    def calc_phenotype_generality(self, phenotype):
        numer = sum(interval.calc_span() for interval in phenotype.elems)
        generality = (numer / self._phenotype_generality_denom)
        # gen could be 0 if all intervals in numer collapse to single point
        assert self._GENERALITY_LB_INCL <= generality <= \
            self._GENERALITY_UB_INCL
        return generality

    def _gen_mutation_noise(self, dim):
        # m_0 interpreted as fraction of dim span to use for stdev of gaussian
        # random noise
        return get_rng().normal(loc=self._MUT_MEAN,
                                scale=(get_hp("m_nought") * dim.span))
