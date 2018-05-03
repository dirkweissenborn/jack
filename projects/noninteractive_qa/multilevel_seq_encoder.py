import math

import numpy as np
import tensorflow as tf

from jack.tfutil.modular_encoder import modular_encoder


def gumbel_softmax(logits):
    # step = tf.train.get_global_step()
    # if step is not None:
    #    step = tf.to_float(step)
    #    logits /= tf.maximum(10.0 * tf.exp(-step / 1000), 1.0)
    dist = tf.contrib.distributions.RelaxedOneHotCategorical(0.5, logits=logits)
    return dist.sample()


def gumbel_sigmoid(logits):
    # step = tf.train.get_global_step()
    # if step is not None:
    #    step = tf.to_float(step)
    #    logits /= tf.maximum(10.0 * tf.exp(-step / 1000), 1.0)
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
    log_keep = tf.log(tf.maximum(1.0 - segm_probs, 1e-8))
    mask = tf.expand_dims(tf.sequence_mask(length, dtype=tf.float32), 2)
    log_keep *= mask
    # [B, L, 1]
    cum_log_keep = tf.cumsum(log_keep, 1, exclusive=True)

    # [B, L, L]
    contributions_fw = cum_log_keep - tf.transpose(cum_log_keep, [0, 2, 1])
    contributions_bw = tf.transpose(contributions_fw, [0, 2, 1])
    contributions = tf.exp(tf.maximum(tf.minimum(contributions_fw, contributions_bw), -20.0))
    contributions *= mask
    contributions *= tf.reshape(mask, [-1, 1, tf.shape(mask)[1]])
    return contributions


def left_segm_sum_contributions(segm_probs, length):
    log_segm_end = tf.log(tf.maximum(segm_probs, 1e-8))
    log_keep = tf.log(tf.maximum(1.0 - segm_probs, 1e-8))
    mask = tf.expand_dims(tf.sequence_mask(length, dtype=tf.float32), 2)
    log_keep *= mask
    log_segm_end *= mask
    # [B, L, 1]
    cum_log_keep = tf.cumsum(log_keep, 1, exclusive=True)

    # [B, L, L]
    contributions = cum_log_keep - tf.transpose(cum_log_keep + log_keep - log_segm_end, [0, 2, 1])
    contributions = tf.exp(tf.maximum(tf.minimum(contributions, 0.0), -20.0))
    # lower triangular matrix to consider only the "left" contributions
    contributions -= tf.matrix_band_part(contributions, 0, -1)
    contributions *= mask
    contributions *= tf.reshape(mask, [-1, 1, tf.shape(mask)[1]])
    return contributions


def right_segm_sum_contributions(segm_probs, length):
    # shift segment probs to become probability of segment starts instead of end
    segm_probs = tf.concat([tf.zeros([tf.shape(segm_probs)[0], 1, 1]), segm_probs[:, :-1]], 1)
    log_segm_end = tf.log(tf.maximum(segm_probs, 1e-8))
    log_keep = tf.log(tf.maximum(1.0 - segm_probs, 1e-8))
    mask = tf.expand_dims(tf.sequence_mask(length, dtype=tf.float32), 2)
    log_keep *= mask
    log_segm_end *= mask
    # [B, L, 1]
    revcum_log_keep = tf.cumsum(log_keep, 1, reverse=True, exclusive=True)

    # [B, L, L]
    contributions = revcum_log_keep - tf.transpose(revcum_log_keep + log_keep - log_segm_end, [0, 2, 1])
    contributions = tf.exp(tf.maximum(tf.minimum(contributions, 0.0), -20.0))
    # upper triangular matrix to consider only the "right" contributions
    contributions -= tf.matrix_band_part(contributions, -1, 0)
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


def bow_segm_encoder(sequence, length, repr_dim, segm_ends, mask=None, normalize=False, transform=True,
                     activation=tf.nn.relu):
    segm_contributions = intra_segm_contributions(segm_ends, length)
    if mask is not None:
        segm_contributions *= tf.transpose(mask, [0, 2, 1])

    sequence = tf.matmul(segm_contributions, sequence)
    if normalize:
        bow_num = tf.matmul(segm_contributions, tf.ones_like(segm_ends))
        sequence /= (bow_num + 1e-6)

    sequence = tf.layers.dense(sequence, repr_dim, activation,
                               name='bow_segm_dense') if transform else sequence

    return sequence


def weighted_bow_segm_encoder(sequence, length, repr_dim, segm_ends, mask=None):
    logits = tf.layers.dense(sequence, 1, None)
    potentials = tf.exp(logits - tf.reduce_max(logits, axis=1, keep_dims=True))
    if mask is not None:
        potentials *= mask  # put zero probability on non segment ends
    contributions = intra_segm_contributions(segm_ends, length)
    row_sum = tf.maximum(tf.matmul(contributions, potentials), potentials) + 1e-8

    probs = potentials / row_sum
    weighted_segm = probs * sequence
    weighted_segm = tf.matmul(contributions, weighted_segm)

    return weighted_segm, probs


def bow_start_end_segm_encoder(sequence, length, repr_dim, segm_ends, mask=None):
    segm_contributions = intra_segm_contributions(segm_ends, length)
    if mask is not None:
        segm_contributions *= tf.transpose(mask, [0, 2, 1])

    bow_sum = tf.matmul(segm_contributions, sequence)
    bow_num = tf.matmul(segm_contributions, tf.ones_like(segm_ends))
    bow_mean = bow_sum / (bow_num + 1e-6)

    segm_starts = tf.concat([tf.ones([tf.shape(segm_ends)[0], 1, 1]), segm_ends[:, :-1]], 1)
    seq_as_start = tf.matmul(segm_contributions, sequence * segm_starts)
    seq_as_end = tf.matmul(segm_contributions, sequence * segm_ends)

    segment_reps = tf.layers.dense(tf.concat([seq_as_start, seq_as_end, bow_mean], 2), repr_dim, tf.nn.relu,
                                   name='bow_start_end_segm_dense')

    return segment_reps


def conv_start_end_segm_encoder(sequence, length, repr_dim, segm_ends, mask=None):
    padded_sequence = tf.pad(sequence, [[0, 0], [1, 1], [0, 0]])
    padded_segm_ends = tf.pad(segm_ends, [[0, 0], [2, 0], [0, 0]], constant_values=1.0)
    right = padded_sequence[:, 2:] * (1.0 - segm_ends)
    left = (padded_sequence * (1.0 - padded_segm_ends))[:, :-2]
    conv = tf.concat([sequence, right, left], 2)
    return bow_start_end_segm_encoder(conv, length, repr_dim, segm_ends, mask)


def segmentation_encoder(sequence, length, repr_dim, segm_ends):
    memory_rnn = SegmentationCell(repr_dim)
    forward_memories, final_memory = tf.nn.dynamic_rnn(memory_rnn, (sequence, segm_ends), length, dtype=tf.float32)

    back_cell = PropagationCell(repr_dim)
    memories_rev = tf.reverse_sequence(forward_memories, length, 1)
    segm_rev = tf.reverse_sequence(segm_ends, length, 1)
    back_memories = tf.nn.dynamic_rnn(back_cell, (memories_rev, segm_rev), length, dtype=tf.float32)[0]

    segment_reps = tf.reverse_sequence(back_memories, length, 1)

    return segment_reps


def edge_detection_encoder(inputs, repr_dim, is_eval, mask=None, bias=0.0):
    edge_logits = tf.layers.dense(tf.layers.dense(inputs, repr_dim, tf.nn.relu) if repr_dim > 0 else inputs, 1,
                                  bias_initializer=tf.constant_initializer(bias))
    edge_probs = tf.cond(is_eval,
                         lambda: tf.round(tf.sigmoid(edge_logits)),
                         lambda: gumbel_sigmoid(edge_logits))
    if mask is not None:
        edge_probs *= mask
    return edge_probs, edge_logits


def segment_selection_encoder(length, repr_dim, frame_probs, segm_probs, segms, ctrl, is_eval, with_sentinel=False):
    logits = tf.layers.dense(tf.layers.dense(ctrl, repr_dim, tf.nn.relu), 1)
    logits -= tf.reduce_max(logits, axis=1, keep_dims=True)
    logits = tf.cond(is_eval, lambda: logits, lambda: gumbel_logits(logits))

    contributions = intra_segm_contributions(frame_probs, length)

    exps = tf.exp(logits)
    if segm_probs is not None:
        exps *= segm_probs
    # probs should not be bigger than 1
    summed_exps = tf.maximum(tf.matmul(contributions, exps), exps)
    if with_sentinel:
        summed_exps += tf.exp(tf.get_variable('sentinel', [], tf.float32, tf.constant_initializer(-5.0)))
    probs = exps / (summed_exps + 1e-20)

    selected = tf.matmul(contributions, probs * segms)
    return selected, probs, logits


def assoc_memory_encoder(length, repr_dim, num_slots, frame_probs, segm_probs, segms, ctrl, is_eval, num_iterations=1):
    address_logits = tf.layers.dense(tf.layers.dense(ctrl, repr_dim, tf.nn.relu), num_slots,
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
        row_sum = tf.maximum(tf.matmul(frame_contributions, potentials), potentials)
        column_sum = tf.reduce_sum(potentials, axis=2, keep_dims=True)
        weights = tf.square(potentials) / (column_sum * row_sum + 1e-20)
        row_weight_sum = tf.matmul(frame_contributions, weights) + 1e-8
        column_weight_sum = tf.reduce_sum(weights, axis=2, keep_dims=True) + 1e-20
        address_probs = weights / tf.maximum(row_weight_sum, column_weight_sum)
        return address_probs

    address_probs = iteration(None, None)

    if num_iterations > 1 and num_slots > 1:
        end = tf.cond(is_eval, lambda: num_iterations - 1,
                      lambda: tf.random_uniform([], 0, num_iterations - 1, tf.int32))
        r = tf.range(0, end)
        address_probs = tf.cond(end > 0, lambda: tf.scan(iteration, r, address_probs)[-1], lambda: address_probs)

    memory = tf.expand_dims(address_probs, 3) * tf.expand_dims(segms, 2)
    memory = tf.reshape(memory, [tf.shape(memory)[0], tf.shape(memory)[1], num_slots * segms.get_shape()[-1].value])
    memory = tf.matmul(frame_contributions, memory)

    return memory, address_probs


def softmax_assoc_memory_encoder(length, repr_dim, num_slots, frame_probs, segm_probs, segms, ctrl, is_eval):
    address_logits = tf.layers.dense(tf.layers.dense(ctrl, repr_dim, tf.nn.relu), num_slots, use_bias=False)
    address_probs = tf.cond(is_eval, lambda: tf.one_hot(tf.argmax(address_logits, -1), num_slots),
                            lambda: gumbel_softmax(address_logits))
    address_probs *= segm_probs  # put zero probability on non segment ends

    memory = tf.expand_dims(address_probs, 3) * tf.expand_dims(segms, 2)
    memory = tf.reshape(memory, [tf.shape(memory)[0], tf.shape(memory)[1], num_slots * segms.get_shape()[-1].value])

    frame_contributions = intra_segm_contributions(frame_probs, length)
    memory = tf.matmul(frame_contributions, memory)

    return memory, address_probs, address_logits


def sigmoid_assoc_memory_encoder(length, repr_dim, num_slots, frame_probs, segm_probs, segms, ctrl, is_eval):
    address_logits = tf.layers.dense(tf.layers.dense(ctrl, repr_dim, tf.nn.relu), num_slots)
    address_probs = tf.cond(is_eval, lambda: tf.round(tf.sigmoid(address_logits)),
                            lambda: gumbel_sigmoid(address_logits))
    address_probs *= segm_probs  # put zero probability on non segment ends

    memory = tf.expand_dims(address_probs, 3) * tf.expand_dims(segms, 2)
    memory = tf.reshape(memory, [tf.shape(memory)[0], tf.shape(memory)[1], num_slots * segms.get_shape()[-1].value])

    frame_contributions = intra_segm_contributions(frame_probs, length)
    memory = tf.matmul(frame_contributions, memory)

    return memory, address_probs, address_logits


def simple_assoc_memory_encoder(length, repr_dim, num_slots, frame_probs, segm_probs, segms, ctrl, is_eval):
    address_logits = tf.layers.dense(tf.layers.dense(ctrl, repr_dim, tf.nn.relu), num_slots,
                                     bias_initializer=tf.constant_initializer(0.0))
    address_logits = tf.cond(is_eval, lambda: address_logits, lambda: gumbel_logits(address_logits))
    potentials = tf.exp(address_logits - tf.reduce_max(address_logits, axis=1, keep_dims=True))
    potentials *= segm_probs  # put zero probability on non segment ends

    frame_contributions = intra_segm_contributions(frame_probs, length)

    row_sum = tf.maximum(tf.matmul(frame_contributions, potentials), potentials) + 1e-8
    column_sum = tf.reduce_sum(potentials, axis=2, keep_dims=True) + 1e-8

    address_probs = potentials / row_sum * potentials / column_sum

    memory = tf.expand_dims(address_probs, 3) * tf.expand_dims(segms, 2)
    memory = tf.reshape(memory, [tf.shape(memory)[0], tf.shape(memory)[1], num_slots * segms.get_shape()[-1].value])
    memory = tf.matmul(frame_contributions, memory)

    return memory, address_probs


def incremental_assoc_memory_encoder(length, repr_dim, num_slots, frame_probs, segm_probs, segms, ctrl, is_eval):
    allowed = segm_probs
    assoc_logits = tf.layers.dense(tf.layers.dense(ctrl, repr_dim, tf.nn.relu), num_slots)
    assoc_logits = tf.cond(is_eval, lambda: assoc_logits, lambda: gumbel_logits(assoc_logits))

    exps = tf.exp(assoc_logits - tf.reduce_max(assoc_logits, axis=1, keep_dims=True))
    exps = tf.split(exps, num_slots, 2)

    assoc_probs = []
    frame_contributions = intra_segm_contributions(frame_probs, length)
    slots = []
    for i in range(num_slots):
        potentials = exps[i] * allowed
        # probs should not be bigger than 1
        summed = tf.maximum(tf.matmul(frame_contributions, potentials), potentials)
        if i > 0:
            summed += tf.exp(tf.get_variable('sentinel_' + str(i), [], tf.float32,
                                             tf.constant_initializer(-5.0)))
        probs = potentials / (summed + 1e-20)
        selected = tf.matmul(frame_contributions, probs * segms)
        slots.append(selected)
        # assoc_ctrl = tf.concat([assoc_ctrl, selected], 2)
        allowed *= (1.0 - probs)
        assoc_probs.append(probs)

    return slots, assoc_probs


def segment_self_attention(ctrl, seq, length, is_eval, key_dim, value_dim=None, scaled=True,
                           num_heads=1, attn_probs=None):
    batch_size = tf.shape(seq)[0]
    if value_dim is None:
        value = seq
    else:
        value = tf.reshape(tf.layers.dense(seq, key_dim * num_heads, tf.nn.relu, name='value'),
                           [batch_size, -1, num_heads, value_dim])
    attn_scores = None
    edge_probs, edge_logits = None, None
    if attn_probs is None:
        attn_scores, edge_probs, edge_logits = segment_self_attention_scores(
            ctrl, seq, length, is_eval, key_dim, scaled=scaled, num_heads=num_heads)

        s = tf.get_variable('sentinel_score', [1, 1, 1, num_heads], tf.float32, tf.zeros_initializer())
        s = tf.tile(s, [tf.shape(attn_scores)[0], tf.shape(attn_scores)[1], 1, 1])
        attn_probs = tf.nn.softmax(tf.concat([s, attn_scores], 2), 2)
        attn_probs = attn_probs[:, :, 1:]

    if value_dim is None:
        attn_states = tf.einsum('abdh,adc->abhc', attn_probs, value)
    else:
        attn_states = tf.einsum('abdh,adhc->abhc', attn_probs, value)

    return attn_scores, attn_probs, attn_states, edge_probs, edge_logits


def segment_self_attention_scores(ctrl, seq, length, is_eval, key_dim, scaled=True,
                                  num_heads=1, edge_probs=None, attn_probs=None):
    edge_logits = None
    if edge_probs is None:
        # [B, L, H]
        edge_logits = tf.layers.dense(tf.layers.dense(ctrl, 16, tf.nn.relu, name='edge_logits_hidden'),
                                      num_heads, name='edge_logits')
        edge_probs = tf.cond(is_eval,
                             lambda: tf.round(tf.sigmoid(edge_logits)),
                             lambda: gumbel_sigmoid(edge_logits))

    batch_size = tf.shape(seq)[0]
    l = tf.shape(seq)[1]

    attn_scores = None
    if attn_probs is None:
        key = tf.reshape(tf.layers.dense(ctrl, key_dim * num_heads, name='key'), [batch_size, -1, num_heads, key_dim])

        query = tf.reshape(tf.layers.dense(ctrl, key_dim * num_heads, name='query'),
                           [batch_size, -1, num_heads, key_dim])

        # [B, L, L, H]
        attn_scores = tf.einsum('abhc,adhc->abdh', query, key)
        attn_scores += tf.transpose(tf.layers.dense(query, 1, use_bias=False), [0, 1, 3, 2])
        attn_scores += tf.transpose(tf.layers.dense(key, 1, use_bias=False), [0, 3, 1, 2])
        if scaled:
            attn_scores /= math.sqrt(float(query.get_shape()[-1].value))

        # [B * H, L, 1]
        edge_probs_t = tf.reshape(tf.transpose(edge_probs, [0, 2, 1]), [batch_size * num_heads, -1, 1])
        # [B * H, L, L]
        associations = intra_segm_contributions(edge_probs_t, tf.tile(length, [num_heads]))
        # [B, L, L, H]
        associations = tf.transpose(tf.reshape(associations, [batch_size, num_heads, l, l]), [0, 2, 3, 1])
        attn_scores += tf.log(associations + 1e-10)

        # exclude attending to state itself
        # attn_scores += tf.expand_dims(tf.expand_dims(tf.diag(tf.fill([tf.shape(attn_scores)[1]], -1e6)), 0), 3)

    return attn_scores, edge_probs, edge_logits


def _get_query_key_value(seq1, seq2, key_value_attn):
    if key_value_attn:
        with tf.variable_scope('key_value_projection') as vs:
            key = tf.layers.dense(seq2, seq2.get_shape()[-1].value, name='key')
            value = tf.layers.dense(seq2, seq2.get_shape()[-1].value, name='value')
            query = tf.layers.dense(seq1, seq1.get_shape()[-1].value, name='query')
        return query, key, value
    else:
        return seq1, seq2, seq2


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
