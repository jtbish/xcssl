from functools import partial

import numpy as np

from .dimension import RealDimension
from .environment import ClassificationStreamEnvironment, StreamDataInstance
from .mux import calc_mux_label
from .obs_space import RealObsSpaceBuilder

_MIN_NUM_ADDR_BITS = 2
_ACTION_SPACE = (0, 1)

_DIM_LOWER = 0.0
_DIM_UPPER = 1.0

_DEFAULT_SEED = 0


def make_rmux_stream_env(num_addr_bits, threshold_vec, seed=None):
    num_addr_bits = int(num_addr_bits)
    assert num_addr_bits >= _MIN_NUM_ADDR_BITS

    num_reg_bits = 2**num_addr_bits
    total_num_bits = (num_addr_bits + num_reg_bits)

    obs_space_builder = RealObsSpaceBuilder()
    for i in range(total_num_bits):
        obs_space_builder.add_dim(
            RealDimension(lower=_DIM_LOWER, upper=_DIM_UPPER, name=f"x_{i}"))
    obs_space = obs_space_builder.create_space()

    threshold_vec = np.asarray(threshold_vec)
    _validate_threshold_vec(threshold_vec, total_num_bits)

    def _gen_instance(rng, num_addr_bits, total_num_bits, threshold_vec):
        # gen random obs and calc label for it based on threshold vec
        obs = rng.random(size=total_num_bits)
        label = calc_rmux_label(obs, threshold_vec, num_addr_bits)
        return StreamDataInstance(obs=obs, label=label)

    instance_gen_func = partial(_gen_instance,
                                num_addr_bits=num_addr_bits,
                                total_num_bits=total_num_bits,
                                threshold_vec=threshold_vec)

    return ClassificationStreamEnvironment.from_instance_gen_func(
        obs_space=obs_space,
        action_space=_ACTION_SPACE,
        instance_gen_func=instance_gen_func,
        seed=seed)


def _validate_threshold_vec(threshold_vec, total_num_bits):
    assert len(threshold_vec) == total_num_bits
    for elem in threshold_vec:
        assert _DIM_LOWER < elem < _DIM_UPPER


def calc_rmux_label(obs, threshold_vec, num_addr_bits):
    # discretise obs based on thresholds, then apply regular MUX function
    bitstring = (obs >= threshold_vec).astype(int)
    return calc_mux_label(bitstring, num_addr_bits)
