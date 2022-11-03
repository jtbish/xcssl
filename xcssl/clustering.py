from .encoding import (IntegerUnorderedBoundEncoding,
                       RealUnorderedBoundEncoding, TernaryEncoding)
from .lsh import EuclideanLSH, HammingLSH

_MIN_NUM_PROJS_PER_BAND = 1
_MIN_NUM_BANDS = 1


class ConditionClustering:
    def __init__(self, encoding, num_projs_per_band, num_bands, phenotypes):
        self._encoding = encoding

        assert num_projs_per_band >= _MIN_NUM_PROJS_PER_BAND
        self._p = num_projs_per_band
        assert num_bands >= _MIN_NUM_BANDS
        self._b = num_bands

        self._lsh = self._init_lsh(self._encoding, self._p, self._b)
        self._C = self._init_phenotype_set(phenotypes)

        self._lsh_key_maps = self._gen_lsh_key_maps(self._lsh, self._C,
                                                    self._b)
        self._clusterings = self._form_clusterings(self._C, self._lsh_key_maps,
                                                   self._b)

    def _init_lsh(self, encoding, p, b):
        # TODO make polymorphic?
        lsh_cls = None
        if isinstance(encoding, TernaryEncoding):
            lsh_cls = HammingLSH
        elif isinstance(encoding, IntegerUnorderedBoundEncoding):
            lsh_cls = HammingLSH
        elif isinstance(encoding, RealUnorderedBoundEncoding):
            lsh_cls = EuclideanLSH
        else:
            assert False
        assert lsh_cls is not None

        d = encoding.calc_num_phenotype_vec_dims()
        return lsh_cls(d, p, b)

    def _init_phenotype_set(self, phenotypes):
        return set(phenotypes)

    def _gen_lsh_key_maps(self, lsh, C, b):
        """Generates the LSH hash/key for each phenotype in C, for all b bands
        used by the hasher."""
        # one map (dict) for each band
        key_maps = [{} for _ in range(b)]

        for phenotype in C:
            vec = phenotype.vec

            for band_idx in range(b):
                key = lsh.hash(vec, band_idx)
                key_maps[band_idx][phenotype] = key

        return key_maps

    def _form_clusterings(self, C, lsh_key_maps, b):
        """Forms the clusters for each of the b bands used by the hasher, for
        all the pheontypes in C.
        This is the inverse mapping of the lsh key maps."""
        clusterings = [{} for _ in range(b)]

        for band_idx in range(b):
            key_map = lsh_key_maps[band_idx]

            for (phenotype, lsh_key) in key_map.items():
                # try add phenotype to existing cluster, else make new cluster
                try:
                    clusterings[band_idx][lsh_key].append(phenotype)
                except KeyError:
                    clusterings[band_idx][lsh_key] = [phenotype]

        return clusterings
