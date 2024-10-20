import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import os
import re
import json
import tux
import numpy as np
import flax
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from ml_collections.config_dict.config_dict import placeholder as config_placeholder


class JaxDistributedConfigurator(object):
    """ Utility class for initializing JAX distributed. """

    @staticmethod
    def get_default_config(updates=None):
        config = tux.config_dict()
        config.initialize_jax_distributed = False
        config.coordinator_address = config_placeholder(str)
        config.num_processes = config_placeholder(int)
        config.process_id = config_placeholder(int)
        config.local_device_ids = config_placeholder(str)
        return tux.update_config_dict(config, updates)


    @classmethod
    def initialize(cls, config):
        config = cls.get_default_config(config)
        if config.initialize_jax_distributed:
            if config.local_device_ids is not None:
                local_device_ids = [int(x) for x in config.local_device_ids.split(',')]
            else:
                local_device_ids = None

            jax.distributed.initialize(
                coordinator_address=config.coordinator_address,
                num_processes=config.num_processes,
                process_id=config.process_id,
                local_device_ids=local_device_ids,
            )
