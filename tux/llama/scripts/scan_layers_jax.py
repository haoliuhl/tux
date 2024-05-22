from absl.app import run
import flax
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import tux
from tux import (
    define_flags_with_default, StreamingCheckpointer, JaxDistributedConfig,
    set_random_seed, get_float_dtype_by_name, JaxRNG, next_rng,
    match_partition_rules, make_shard_and_gather_fns, get_weight_decay_mask,
    OptimizerFactory, with_sharding_constraint
)
from model.minimal_model import LLaMAConfig, FlaxLLaMAForCausalLMModule

FLAGS, FLAGS_DEF = define_flags_with_default(
    output_dir='',
    mesh_dim='1,1,-1,1',
    dtype='bf16',
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfig.get_default_config(),
)


def main(argv):
    set_random_seed(42)
    if FLAGS.load_llama_config != '':
        llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
        updates = LLaMAConfig(**FLAGS.llama)
        llama_config.update(dict(
            remat_block=updates.remat_block,
            remat_attention=updates.remat_attention,
            remat_mlp=updates.remat_mlp,
            scan_attention=updates.scan_attention,
            scan_mlp=updates.scan_mlp,
            scan_query_chunk_size=updates.scan_query_chunk_size,
            scan_key_chunk_size=updates.scan_key_chunk_size,
            scan_mlp_chunk_size=updates.scan_mlp_chunk_size,
            scan_layers=updates.scan_layers,
            param_scan_axis=updates.param_scan_axis,
        ))
    else:
        llama_config = LLaMAConfig(**FLAGS.llama)

    if FLAGS.update_llama_config != '':
        llama_config.update(dict(eval(FLAGS.update_llama_config)))

    llama_config.update(dict(mesh_dim=FLAGS.mesh_dim))

    model = FlaxLLaMAForCausalLMModule(
        llama_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
    )

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(LLaMAConfig.get_weight_decay_exclusions())
    )

    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, FLAGS.output_dir,
        enable=jax.process_index() == 0,
    )

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        batch = 2
        seq_length = 32
        params = model.init(
            input_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((batch, seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state, params = checkpointer.load_trainstate_checkpoint(
        FLAGS.load_checkpoint, train_state_shapes, None, #shard_fns,
    )

    def update(params):
        params = params.unfreeze()
        trees = []
        for i in range(llama_config.num_hidden_layers):
            trees.append(params['transformer']['h'][str(i)])
        trees = jax.tree_map(lambda *xs: np.stack(xs, axis=0), *trees)
        params['transformer']['h'] = {'scan_decoder': trees}
        params = flax.core.frozen_dict.freeze(params)
        return params

    params = params['params']
    params = update(params)
    shapes = jax.tree_map(lambda x: x.shape, params)
    print(shapes)
    StreamingCheckpointer.save_train_state_to_file(params, FLAGS.output_dir, float_dtype='bf16')
    print('done')

if __name__ == "__main__":
    run(main)
