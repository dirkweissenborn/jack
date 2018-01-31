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


def gumbel_logits(logits):
    uniform = tf.random_uniform(
        shape=tf.shape(logits), minval=np.finfo(np.float32).tiny, maxval=1., dtype=tf.float32)
    gumbel = -tf.log(-tf.log(uniform))
    return tf.div(gumbel + logits, 0.5)


def horizontal_probs(logits, length, segm_probs, is_eval):
    logits = tf.cond(is_eval, lambda: logits, lambda: gumbel_logits(logits))
    exps = tf.exp(logits)
    summed_exps = tf.maximum(intra_segm_sum(exps, segm_probs, length), exps)  # probs should not be bigger than 1
    # exps = tf.Print(exps, [exps], message='exp', summarize=10)
    # summed_exps = tf.Print(exps, [summed_exps], message='sum', summarize=10)
    return exps / (summed_exps + 1e-10)


def intra_segm_sum(inputs, segm_probs, length):
    repr_dim = inputs.get_shape()[-1].value
    keep = 1.0 - segm_probs
    sum_fw, end_state = tf.nn.dynamic_rnn(_SumReset(repr_dim), (inputs, keep), length, dtype=tf.float32)
    sum_fw_rev = tf.reverse_sequence(sum_fw, length, 1)
    segm_rev = tf.reverse_sequence(segm_probs, length, 1)
    summ = tf.nn.dynamic_rnn(PropagationCell(repr_dim), (sum_fw_rev, segm_rev), length, initial_state=end_state)[0]
    summ = tf.reverse_sequence(summ, length, 1)
    return summ


class _SumReset(tf.nn.rnn_cell.RNNCell):
    def __init__(self, repr_dim):
        self._repr_dim = repr_dim

    def __call__(self, inputs, state, scope=None):
        inputs, keep = inputs
        state += inputs
        return state, state * keep

    @property
    def output_size(self):
        return self._repr_dim

    @property
    def state_size(self):
        return self._repr_dim


def controller(sequence, length, controller_config, repr_dim, is_eval):
    controller_out = modular_encoder(
        controller_config['encoder'], {'text': sequence}, {'text': length}, {},
        repr_dim, 0.0, is_eval=is_eval)[0]['text']
    return controller_out


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


def governor_detection_encoder(sequence, length, repr_dim, controller_out, segm_probs, segms, is_eval):
    frame_end_logits = tf.layers.dense(tf.layers.dense(controller_out, repr_dim, tf.nn.relu), 1,
                                       bias_initializer=tf.constant_initializer(0.0))
    frame_probs = tf.cond(is_eval,
                          lambda: tf.round(tf.sigmoid(frame_end_logits)),
                          lambda: gumbel_sigmoid(frame_end_logits))
    frame_probs *= tf.stop_gradient(segm_probs)
    tf.identity(tf.sigmoid(frame_end_logits), name='frame_probs')

    governor_logits = tf.layers.dense(tf.layers.dense(controller_out, repr_dim, tf.nn.relu), 1)
    governor_logits += (segm_probs - 1.0) * 100  # mask non segment ends
    # frame_probs = tf.Print(frame_probs, [frame_probs], message='frame_probs', summarize=10)
    governor_probs = horizontal_probs(governor_logits, length, frame_probs, is_eval)
    tf.identity(governor_probs, name='governor_probs')

    govenors = intra_segm_sum(governor_probs * segms, frame_probs, length)

    return govenors, frame_probs, frame_end_logits, governor_probs, governor_logits


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
        return self._cell.state_size
