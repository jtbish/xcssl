import itertools

import numpy as np
import pandas as pd

from .environment import ClassificationEnvironmentBase
from .obs_space import make_binary_obs_space

_MIN_NUM_ADDR_BITS = 2
_ACTION_SPACE = (0, 1)
_NP_DTYPE = np.uint8  # use minimal memory


def make_mux_env(num_addr_bits, shuffle_seed=None):
    num_addr_bits = int(num_addr_bits)
    assert num_addr_bits >= _MIN_NUM_ADDR_BITS
    return MultiplexerEnvironment(num_addr_bits, shuffle_seed)


class MultiplexerEnvironment(ClassificationEnvironmentBase):
    def __init__(self, num_addr_bits, shuffle_seed):
        self._num_addr_bits = num_addr_bits
        self._num_reg_bits = 2**num_addr_bits
        self._total_num_bits = (self._num_addr_bits + self._num_reg_bits)

        dataset = self._gen_dataset(self._num_addr_bits, self._total_num_bits)
        obs_space = make_binary_obs_space(num_dims=self._total_num_bits)
        super().__init__(dataset=dataset,
                         obs_space=obs_space,
                         action_space=_ACTION_SPACE,
                         shuffle_seed=shuffle_seed)

    def _gen_dataset(self, num_addr_bits, total_num_bits):
        examples = []
        labels = []
        bitstrings = itertools.product([0, 1], repeat=total_num_bits)
        for bitstring in bitstrings:
            examples.append(np.array(bitstring, dtype=_NP_DTYPE))
            label = self._calc_label(bitstring, num_addr_bits)
            labels.append(label)
        assert len(examples) == len(labels)

        dataset = pd.DataFrame({"examples": examples, "labels": labels})
        dataset = dataset.astype({"labels": _NP_DTYPE})
        return dataset

    def _calc_label(self, bitstring, num_addr_bits):
        """Apply MUX function."""
        addr_bits = bitstring[:num_addr_bits]
        reg_bits = bitstring[num_addr_bits:]

        addr_bits_str = "".join([str(b) for b in addr_bits])
        reg_idx = int(addr_bits_str, base=2)  # bin to int
        label = reg_bits[reg_idx]

        return label
