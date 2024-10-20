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


class Checkpointer(object):
    """ A simple wrapper for orbax checkpointing. """

    def __init__(self, path):
        self.path = path
        self.checkpointer = ocp.StandardCheckpointer()
        if self.path != '':
            tux.makedirs(self.path)

    def save_pytree(self, pytree, prefix=None):
        """ Save pytree of JAX arrays. """
        if self.path == '':
            return
        if prefix is None:
            path = self.path
        else:
            path = os.path.join(self.path, prefix)

        self.checkpointer.save(path, pytree, force=True)
        # Create a commit_success.txt file to indicate that the checkpoint is
        # saved successfully. This is a workaround for orbax so that locally
        # saved checkpoint can be restored when copied to Google cloud storage.
        tux.open_file(os.path.join(path, 'commit_success.txt'), 'w').close()

    @classmethod
    def restore_pytree(cls, path, item):
        return ocp.StandardCheckpointer().restore(
            path, args=ocp.args.StandardRestore(item)
        )

    def save_json(self, data, name):
        """ Save dictionary as JSON. """
        if self.path == '':
            return
        path = os.path.join(self.path, name)
        tux.makedirs(self.path)
        with tux.open_file(path, 'w') as f:
            f.write(json.dumps(data, indent=4))

    @classmethod
    def load_json(cls, path):
        with tux.open_file(path, 'r') as f:
            return json.loads(f.read())

    @classmethod
    def get_shape_dtype_struct(cls, tree):
        return jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, tree)
