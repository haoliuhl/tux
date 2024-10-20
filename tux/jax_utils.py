import dataclasses
import random

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from flax.training.train_state import TrainState


class JaxRNG(object):
    """ A convenient stateful Jax RNG wrapper. Can be used to wrap RNG inside
        pure function.
    """
    global_rng_generator = None

    @classmethod
    def from_seed(cls, seed):
        return cls(jax.random.PRNGKey(seed))

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, keys=None):
        if keys is None:
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng
        elif isinstance(keys, int):
            split_rngs = jax.random.split(self.rng, num=keys + 1)
            self.rng = split_rngs[0]
            return tuple(split_rngs[1:])
        else:
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)
            self.rng = split_rngs[0]
            return {key: val for key, val in zip(keys, split_rngs[1:])}

    @classmethod
    def init_global_rng(cls, seed):
        cls.global_rng_generator = cls.from_seed(seed)

    @classmethod
    def next_rng(cls, *args, **kwargs):
        assert cls.global_rng_generator is not None, 'Global RNG not initialized.'
        return cls.global_rng_generator(*args, **kwargs)


def wrap_function_with_rng(rng):
    """ To be used as decorator, automatically bookkeep a RNG for the wrapped function. """
    def wrap_function(function):
        def wrapped(*args, **kwargs):
            nonlocal rng
            rng, split_rng = jax.random.split(rng)
            return function(split_rng, *args, **kwargs)
        return wrapped
    return wrap_function


def init_rng(seed):
    global jax_utils_rng
    jax_utils_rng = JaxRNG.from_seed(seed)


def next_rng(*args, **kwargs):
    global jax_utils_rng
    return jax_utils_rng(*args, **kwargs)


def flatten_tree(xs, is_leaf=None, sep=None):
    """ A stronger version of flax.traverse_util.flatten_dict, supports
        dict, tuple, list and TrainState. Tuple and list indices will be
        converted to strings.
    """
    tree_node_classes = (FrozenDict, dict, tuple, list, TrainState)
    if not isinstance(xs, tree_node_classes):
        ValueError('fUnsupported node type: {type(xs)}')

    def _is_leaf(prefix, fx):
        if is_leaf is not None:
            return is_leaf(prefix, xs)
        return False

    def _key(path):
        if sep is None:
            return path
        return sep.join(path)

    def _convert_to_dict(xs):
        if isinstance(xs, (FrozenDict, dict)):
            return xs
        elif isinstance(xs, (tuple, list)):
            return {f'{i}': v for i, v in enumerate(xs)}
        elif isinstance(xs, TrainState):
            output = {}
            for field in dataclasses.fields(xs):
                if 'pytree_node' not in field.metadata or field.metadata['pytree_node']:
                    output[field.name] = getattr(xs, field.name)
            return output
        else:
            raise ValueError('fUnsupported node type: {type(xs)}')

    def _flatten(xs, prefix):
        if not isinstance(xs, tree_node_classes) or _is_leaf(prefix, xs):
            return {_key(prefix): xs}

        result = {}
        is_empty = True
        for (key, value) in _convert_to_dict(xs).items():
            is_empty = False
            path = prefix + (key, )
            result.update(_flatten(value, path))
        return result

    return _flatten(xs, ())


# def named_tree_map(f, tree, is_leaf=None, sep=None):
#     """ An extended version of jax.tree_util.tree_map, where the mapped function
#         f takes both the name (path) and the tree leaf as input.
#     """
#     flattened_tree = flatten_tree(tree, is_leaf=is_leaf, sep=sep)
#     id_to_name = {id(val): key for key, val in flattened_tree.items()}
#     def map_fn(leaf):
#         name = id_to_name[id(leaf)]
#         return f(name, leaf)
#     return jax.tree_util.tree_map(map_fn, tree)


def tree_path_to_string(path, sep=None):
    keys = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            keys.append(str(key.idx))
        elif isinstance(key, jax.tree_util.DictKey):
            keys.append(str(key.key))
        elif isinstance(key, jax.tree_util.GetAttrKey):
            keys.append(str(key.name))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            keys.append(str(key.key))
        else:
            keys.append(str(key))
    if sep is None:
        return tuple(keys)
    return sep.join(keys)


def named_tree_map(f, tree, *rest, is_leaf=None, sep=None):
    """ An extended version of jax.tree_util.tree_map, where the mapped function
        f takes both the name (path) and the tree leaf as input.
    """
    return jax.tree_util.tree_map_with_path(
        lambda path, x, *r: f(tree_path_to_string(path, sep=sep), x, *r),
        tree, *rest,
        is_leaf=is_leaf
    )


def print_pytree_structure(tree, sep='/', is_leaf=None):
    def print_fn(path, val):
        shape = f'shape: {val.shape if hasattr(val, "shape") else "none"}'
        dtype = f'dtype: {val.dtype if hasattr(val, "dtype") else "none"}'
        print(f'{path}: {shape}, {dtype}')
    named_tree_map(print_fn, tree, is_leaf=is_leaf, sep=sep)


def get_pytree_shape_info(tree):
    flattend_tree = flatten_tree(tree, sep='/')
    shapes = []
    for key in sorted(list(flattend_tree.keys())):
        val = flattend_tree[key]
        shapes.append(f'{key}: {val.dtype}, {val.shape}')
    return '\n'.join(shapes)


def collect_metrics(metrics, names, prefix=None):
    collected = {}
    for name in names:
        if name in metrics:
            collected[name] = jnp.mean(metrics[name])
    if prefix is not None:
        collected = {
            '{}/{}'.format(prefix, key): value for key, value in collected.items()
        }
    return collected


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    init_rng(seed)


def tree_apply(fns, tree):
    """ Apply a pytree of functions to the pytree. """
    return jax.tree_util.tree_map(lambda fn, x: fn(x), fns, tree)
