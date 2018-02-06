import numpy as np
import tensorflow as tf

from jack.tfutil.modular_encoder import modular_encoder


def gumbel_softmax(logits):
    step = tf.train.get_global_step()
    if step is not None:
        step = tf.to_float(step)
        logits /= tf.maximum(10.0 * tf.exp(-step / 1000), 1.0)
    dist = tf.contrib.distributions.RelaxedOneHotCategorical(0.5, logits=logits)
    return dist.sample()


def gumbel_sigmoid(logits):
    step = tf.train.get_global_step()
    if step is not None:
        step = tf.to_float(step)
        logits /= tf.maximum(10.0 * tf.exp(-step / 1000), 1.0)
    dist = tf.contrib.distributions.RelaxedBernoulli(0.5, logits=logits)
    return dist.sample()


def gumbel_logits(logits, temp=0.5):
    uniform = tf.random_uniform(
        shape=tf.shape(logits), minval=np.finfo(np.float32).tiny, maxval=1., dtype=tf.float32)
    gumbel = -tf.log(-tf.log(uniform))
    return tf.div(gumbel + logits, temp)


def horizontal_probs(logits, length, segm_probs, is_eval):
    logits = tf.cond(is_eval, lambda: logits, lambda: gumbel_logits(logits))
    exps = tf.exp(logits - tf.reduce_max(logits, axis=1, keep_dims=True))
    # probs should not be bigger than 1
    summed_exps = tf.maximum(intra_segm_sum(exps, segm_probs, length), exps)
    probs = exps / (summed_exps + 1e-8)

    return probs


def intra_segm_contributions(segm_probs, length):
    log_keep = tf.log(tf.maximum(tf.minimum(1.0 - segm_probs, 1.0), 1e-8))
    mask = tf.expand_dims(tf.sequence_mask(length, dtype=tf.float32), 2)
    log_keep *= mask
    # [B, L, 1]
    revcum_log_keep = tf.cumsum(log_keep, 1, reverse=True)
    cum_log_keep = tf.cumsum(log_keep, 1, exclusive=True)

    # [B, L, L]
    contributions_fw = cum_log_keep - tf.transpose(cum_log_keep, [0, 2, 1])
    contributions_bw = revcum_log_keep - tf.transpose(revcum_log_keep, [0, 2, 1])
    contributions = tf.exp(tf.maximum(tf.minimum(contributions_fw, contributions_bw), -20.0))
    contributions *= mask
    contributions *= tf.reshape(mask, [-1, 1, tf.shape(mask)[1]])
    return contributions


def intra_segm_sum(inputs, segm_probs, length):
    # [B, L, L] * [B, L, D] = [B, L, D]
    summ = tf.matmul(intra_segm_contributions(segm_probs, length), inputs)
    return summ

def controller(sequence, length, controller_config, repr_dim, is_eval):
    controller_out = modular_encoder(
        controller_config['encoder'], {'text': sequence}, {'text': length}, {},
        repr_dim, 0.0, is_eval=is_eval)[0]['text']
    return controller_out


def bow_segm_encoder(sequence, length, repr_dim, controller_out, is_eval):
    segm_logits = tf.layers.dense(tf.layers.dense(controller_out, repr_dim, tf.nn.relu), 1)
    segm_ends = tf.cond(is_eval,
                        lambda: tf.round(tf.sigmoid(segm_logits)),
                        lambda: gumbel_sigmoid(segm_logits))

    tf.identity(tf.nn.sigmoid(segm_logits), name='segm_probs')

    seq_as_start, seq_as_end, seq_transformed = tf.split(
        tf.layers.dense(sequence, 3 * repr_dim, tf.nn.relu), 3, 2)

    segm_contributions = intra_segm_contributions(segm_ends, length)

    bow_sum = tf.matmul(segm_contributions, seq_transformed)
    bow_num = tf.matmul(segm_contributions, tf.ones_like(segm_ends))
    bow_mean = bow_sum / (bow_num + 1e-6)

    segm_starts = tf.concat([tf.ones([tf.shape(segm_ends)[0], 1, 1]), segm_ends[:, :-1]], 1)
    seq_as_start *= segm_starts
    seq_as_end *= segm_ends

    segment_reps = bow_mean + tf.matmul(segm_contributions, seq_as_end) + tf.matmul(segm_contributions, seq_as_start)

    return segment_reps, segm_ends, segm_logits


def segmentation_encoder(sequence, length, repr_dim, controller_out, is_eval):
    segm_logits = tf.layers.dense(tf.layers.dense(controller_out, repr_dim, tf.nn.relu), 1)
    segm_probs = tf.cond(is_eval,
                         lambda: tf.round(tf.sigmoid(segm_logits)),
                         lambda: gumbel_sigmoid(segm_logits))

    tf.identity(tf.nn.sigmoid(segm_logits), name='segm_probs')

    memory_rnn = SegmentationCell(repr_dim)
    forward_memories, final_memory = tf.nn.dynamic_rnn(memory_rnn, (sequence, segm_probs), length, dtype=tf.float32)

    back_cell = PropagationCell(repr_dim)
    memories_rev = tf.reverse_sequence(forward_memories, length, 1)
    segm_rev = tf.reverse_sequence(segm_probs, length, 1)
    back_memories = tf.nn.dynamic_rnn(back_cell, (memories_rev, segm_rev), length, dtype=tf.float32)[0]

    segment_reps = tf.reverse_sequence(back_memories, length, 1)

    return segment_reps, segm_probs, segm_logits


def governor_detection_encoder(length, repr_dim, controller_out, segm_probs, segms, is_eval):
    frame_end_logits = tf.layers.dense(tf.layers.dense(controller_out, repr_dim, tf.nn.relu), 1,
                                       bias_initializer=tf.constant_initializer(0.0))
    frame_probs = tf.cond(is_eval,
                          lambda: tf.round(tf.sigmoid(frame_end_logits)),
                          lambda: gumbel_sigmoid(frame_end_logits))
    frame_probs *= segm_probs
    tf.identity(tf.sigmoid(frame_end_logits), name='frame_probs')

    governor_logits = tf.layers.dense(tf.layers.dense(segms, repr_dim, tf.nn.relu), 1)
    governor_logits = tf.cond(is_eval, lambda: governor_logits, lambda: gumbel_logits(governor_logits))
    exps = tf.exp(governor_logits - tf.reduce_max(governor_logits, axis=1, keep_dims=True))
    exps *= segm_probs
    # probs should not be bigger than 1
    summed_exps = tf.maximum(intra_segm_sum(exps, frame_probs, length), exps)
    governor_probs = exps / (summed_exps + 1e-8)
    tf.identity(governor_probs, name='governor_probs')

    govenors = intra_segm_sum(governor_probs * segms, frame_probs, length)
    return govenors, frame_probs, frame_end_logits, governor_probs, governor_logits


def assoc_memory_encoder(length, repr_dim, num_slots, governor, frame_probs, segm_probs, segms, is_eval,
                         num_iterations=1):
    inputs = tf.concat([governor, segms], 2)
    address_logits = tf.layers.dense(tf.layers.dense(inputs, repr_dim, tf.nn.relu), num_slots,
                                     bias_initializer=tf.constant_initializer(0.0))
    address_logits = tf.cond(is_eval, lambda: address_logits, lambda: gumbel_logits(address_logits))
    potentials = tf.exp(address_logits - tf.reduce_max(address_logits, axis=1, keep_dims=True))
    potentials *= segm_probs  # put zero probability on non segment ends
    original_potentials = potentials

    frame_contributions = intra_segm_contributions(frame_probs, length)

    def iteration(address_probs, x):
        potentials = original_potentials
        if address_probs is not None:
            potentials *= address_probs
        row_sum = tf.maximum(tf.matmul(frame_contributions, potentials), potentials) + 1e-8
        column_sum = tf.reduce_sum(potentials, axis=2, keep_dims=True) + 1e-8
        weights = potentials * potentials / column_sum / row_sum
        row_weight_sum = tf.matmul(frame_contributions, weights) + 1e-8
        column_weight_sum = tf.reduce_sum(weights, axis=2, keep_dims=True) + 1e-8
        address_probs = weights / tf.maximum(row_weight_sum, column_weight_sum)
        return address_probs

    address_probs = iteration(None, None)

    if num_iterations > 1:
        end = tf.cond(is_eval, lambda: num_iterations - 1,
                      lambda: tf.random_uniform([], 0, num_iterations - 1, tf.int32))
        r = tf.range(0, end)
        address_probs = tf.cond(end > 0, lambda: tf.scan(iteration, r, address_probs)[-1], lambda: address_probs)

    tf.identity(address_probs, name='address_probs')
    memory = tf.expand_dims(address_probs, 3) * tf.expand_dims(segms, 2)
    memory = tf.reshape(memory, [tf.shape(memory)[0], tf.shape(memory)[1], num_slots * segms.get_shape()[-1].value])
    memory = tf.matmul(frame_contributions, memory)

    return memory, address_probs, address_logits


def simple_assoc_memory_encoder(length, repr_dim, num_slots, governor, frame_probs, segm_probs, segms, is_eval):
    inputs = tf.concat([governor, segms], 2)
    address_logits = tf.layers.dense(tf.layers.dense(inputs, repr_dim, tf.nn.relu), num_slots,
                                     bias_initializer=tf.constant_initializer(0.0))
    address_logits = tf.cond(is_eval, lambda: address_logits, lambda: gumbel_logits(address_logits))
    potentials = tf.exp(address_logits - tf.reduce_max(address_logits, axis=1, keep_dims=True))
    potentials *= segm_probs  # put zero probability on non segment ends

    frame_contributions = intra_segm_contributions(frame_probs, length)

    row_sum = tf.maximum(tf.matmul(frame_contributions, potentials), potentials) + 1e-8
    column_sum = tf.reduce_sum(potentials, axis=2, keep_dims=True) + 1e-8

    address_probs = potentials / row_sum * potentials / column_sum

    tf.identity(address_probs, name='address_probs')
    memory = tf.expand_dims(address_probs, 3) * tf.expand_dims(segms, 2)
    memory = tf.reshape(memory, [tf.shape(memory)[0], tf.shape(memory)[1], num_slots * segms.get_shape()[-1].value])
    memory = tf.matmul(frame_contributions, memory)

    return memory, address_probs, address_logits


class PropagationCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, repr_dim):
        self._repr_dim = repr_dim

    def __call__(self, inputs, state, scope=None):
        inputs, segm_end = inputs
        new_memory = segm_end * inputs + (1.0 - segm_end) * state
        return new_memory, new_memory

    @property
    def output_size(self):
        return self._repr_dim

    @property
    def state_size(self):
        return self._repr_dim


class SegmentationCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, repr_dim):
        self._cell = tf.contrib.rnn.GRUBlockCell(repr_dim)

    def __call__(self, inputs, state, scope=None):
        inputs, segment_end = inputs
        out, new_state = self._cell(inputs, state)
        if isinstance(new_state, tf.Tensor):
            new_state *= (1.0 - segment_end)
        else:
            new_state = tf.nest.map_structure(lambda x: x * (1.0 - segment_end), new_state)
        return segment_end * out, new_state

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def state_size(self):
        return self._repr_dim, self._repr_dim, 1, 1
