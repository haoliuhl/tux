import jax
import jax.numpy as jnp


def mse_loss(val, target, valid=None):
    if valid is None:
        valid = jnp.ones((*target.shape[:2], 1))
    valid = valid.astype(jnp.float32)
    loss = jnp.mean(
        jnp.where(
            valid > 0.0,
            jnp.square(val - target),
            0.0
        )
    )
    return loss


def cross_entropy_loss(logits, labels, smoothing_factor=0.):
    num_classes = logits.shape[-1]
    if labels.dtype == jnp.int32 or labels.dtype == jnp.int64:
        labels = jax.nn.one_hot(labels, num_classes)
    if smoothing_factor > 0.:
        labels = labels * (1. - smoothing_factor) + smoothing_factor / num_classes
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(logp * labels, axis=-1))


def cross_entropy_loss_and_accuracy(logits, tokens, valid=None):
    if valid is None:
        valid = jnp.ones(tokens.shape[:2])
    valid = valid.astype(jnp.float32)
    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)
    logits = logits.astype(jnp.float32) # for numerical stability
    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            jnp.expand_dims(tokens, -1),
            axis=-1,
        ),
        -1,
    )
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))
    loss = -jnp.mean(jnp.sum(token_log_prob, axis=-1) / valid_text_length)
    correct = jnp.where(
        valid > 0.0,
        jnp.argmax(logits, axis=-1) == tokens,
        jnp.array(False)
    )
    accuracy = jnp.mean(jnp.sum(correct, axis=-1) / valid_text_length)
    return loss, accuracy
