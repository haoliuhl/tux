import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import os
import re
import json
import numpy as np
import flax
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from ml_collections.config_dict.config_dict import placeholder as config_placeholder
from .config import config_dict, update_config_dict
from typing import Any, Mapping, NamedTuple, Text, Tuple, Union
from .jax_utils import named_tree_map


class AdamConfigurator(object):
    """ AdamW optimizer with cosine schedule. """

    @staticmethod
    def get_default_config(updates=None):
        config = config_dict()
        config.init_lr = 0.0
        config.end_lr = 0.001
        config.lr = 0.01
        config.lr_warmup_steps = 2000
        config.lr_decay_steps = 500000
        config.b1 = 0.9
        config.b2 = 0.95
        config.clip_gradient = 1.0
        config.weight_decay = 1e-4
        return update_config_dict(config, updates)

    @classmethod
    def get_optimizer_and_schedule(cls, config, weight_decay_mask=None):
        config = cls.get_default_config(config)
        learning_rate_schedule = optax.warmup_cosine_decay_schedule(
            init_value=config.init_lr,
            peak_value=config.lr,
            warmup_steps=config.lr_warmup_steps,
            decay_steps=config.lr_decay_steps,
            end_value=config.end_lr,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.clip_gradient),
            optax.adamw(
                learning_rate=learning_rate_schedule,
                weight_decay=config.weight_decay,
                b1=config.b1,
                b2=config.b2,
                mask=weight_decay_mask,
            ),
        )
        return optimizer, learning_rate_schedule


class OptaxScheduledWeightDecayState(NamedTuple):
    count: jnp.array


def optax_add_scheduled_weight_decay(schedule_fn, mask=None):
    """ Apply weight decay with schedule. """

    def init_fn(params):
        del params
        return OptaxScheduledWeightDecayState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError('Params cannot be None for weight decay!')

        weight_decay = schedule_fn(state.count)
        updates = jax.tree_util.tree_map(
            lambda g, p: g + weight_decay * p, updates, params
        )
        return updates, OptaxScheduledWeightDecayState(
            count=optax.safe_int32_increment(state.count)
        )

    if mask is not None:
        return optax.masked(optax.GradientTransformation(init_fn, update_fn), mask)
    return optax.GradientTransformation(init_fn, update_fn)


def get_mask(exclusions, tf_map=None):
    """ Return a mask function that computes the pytree masks
        according to the given exclusion rules.
    """
    if tf_map is None:
        tf_map = {True: True, False: False}
    else:
        assert len(tf_map) == 2 and True in tf_map and False in tf_map

    def to_keep(name, _):
        for rule in exclusions:
            if re.search(rule, name) is not None:
                return False
        return True

    def mask_fn(params):
        return named_tree_map(lambda *args: tf_map[to_keep(*args)], params, sep='/')

    return mask_fn


# For backwards compatibility
get_weight_decay_mask = get_mask
