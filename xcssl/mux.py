import itertools
from functools import partial

import numpy as np
import pandas as pd

from .environment import (ClassificationEpochEnvironment,
                          ClassificationStreamEnvironment, StreamDataInstance)
from .obs_space import make_binary_obs_space

_MIN_NUM_ADDR_BITS = 2
_ACTION_SPACE = (0, 1)
_NP_DTYPE = np.uint8  # use minimal memory


def make_mux_epoch_env(num_addr_bits, seed=None):
    num_addr_bits = int(num_addr_bits)
    assert num_addr_bits >= _MIN_NUM_ADDR_BITS

    num_reg_bits = 2**num_addr_bits
    total_num_bits = (num_addr_bits + num_reg_bits)

    obs_space = make_binary_obs_space(num_dims=total_num_bits)
    dataset = _gen_dataset(num_addr_bits, total_num_bits)

    return ClassificationEpochEnvironment(obs_space=obs_space,
                                          action_space=_ACTION_SPACE,
                                          dataset=dataset,
                                          shuffle_seed=seed)


def make_mux_stream_env(num_addr_bits, seed=None):
    num_addr_bits = int(num_addr_bits)
    assert num_addr_bits >= _MIN_NUM_ADDR_BITS

    num_reg_bits = 2**num_addr_bits
    total_num_bits = (num_addr_bits + num_reg_bits)

    obs_space = make_binary_obs_space(num_dims=total_num_bits)

    def _gen_instance(rng, num_addr_bits, total_num_bits):
        # gen random bitstring and calc label for it
        obs = rng.randint(low=0, high=(1 + 1), size=total_num_bits)
        label = _calc_label(obs, num_addr_bits)
        return StreamDataInstance(obs=obs, label=label)

    instance_gen_func = partial(_gen_instance,
                                num_addr_bits=num_addr_bits,
                                total_num_bits=total_num_bits)

    return ClassificationStreamEnvironment.from_instance_gen_func(
        obs_space=obs_space,
        action_space=_ACTION_SPACE,
        instance_gen_func=instance_gen_func,
        seed=seed)


def _gen_dataset(num_addr_bits, total_num_bits):
    examples = []
    labels = []

    bitstrings = itertools.product([0, 1], repeat=total_num_bits)

    for bitstring in bitstrings:
        examples.append(np.array(bitstring, dtype=_NP_DTYPE))
        label = _calc_label(bitstring, num_addr_bits)
        labels.append(label)

    assert len(examples) == len(labels)

    dataset = pd.DataFrame({"obs": examples, "label": labels})
    dataset = dataset.astype({"label": _NP_DTYPE})
    return dataset


def _calc_label(bitstring, num_addr_bits):
    """Apply MUX function."""
    addr_bits = bitstring[:num_addr_bits]
    reg_bits = bitstring[num_addr_bits:]

    addr_bits_str = "".join([str(b) for b in addr_bits])
    reg_idx = int(addr_bits_str, base=2)  # bin to int
    label = reg_bits[reg_idx]

    return label
