import tensorflow as tf

from jack.tfutil import sequence_encoder


def gumbel_softmax(logits):
    dist = tf.contrib.distributions.RelaxedOneHotCategorical(1.0 / logits.get_shape()[-1].value, logits=logits)
    return dist.sample()


def gumbel_sigmoid(logits):
    dist = tf.contrib.distributions.RelaxedBernoulli(0.5, logits=logits)
    return dist.sample()


def associative_mem_net(sequence, length, key_dim, num_slots, slot_dim, controller_config, is_eval):
    controller_out = sequence_encoder.encoder(sequence, length, **controller_config)

    address_logits = tf.layers.dense(tf.layers.dense(controller_out, key_dim), num_slots, use_bias=False)
    address_probs = tf.cond(is_eval,
                            lambda: tf.one_hot(tf.argmax(address_logits, axis=-1), num_slots, axis=-1),
                            lambda: gumbel_softmax(address_logits))

    reset_logits = tf.layers.dense(tf.layers.dense(controller_out, key_dim), num_slots, use_bias=False)
    reset_probs = tf.cond(is_eval,
                          lambda: tf.round(tf.sigmoid(reset_logits)),
                          lambda: gumbel_sigmoid(reset_logits))

    memory_rnn = AssociativeMemoryCell(num_slots, slot_dim)

    forward_memories, final_memory = tf.nn.dynamic_rnn(
        memory_rnn, (sequence, address_probs, reset_probs), length, dtype=tf.float32)

    return forward_memories, final_memory, address_probs, reset_probs


def bidirectional_associative_mem_net(
        sequence, length, key_dim, num_slots, slot_dim, controller_config, is_eval):
    forward_memories, final_memory, address_probs, reset_probs = associative_mem_net(
        sequence, length, key_dim, num_slots, slot_dim, controller_config, is_eval)

    # copy memory states back through time, between memory states
    memory_rnn = BackwardAssociativeMemoryCell(num_slots, slot_dim)

    reset_probs_shifted = tf.concat([reset_probs[:, 1:, :], tf.zeros([tf.shape(reset_probs)[0], 1, num_slots])], 1)
    forward_memories_rev = tf.reverse_sequence(forward_memories, length, 1)
    reset_probs_shifted_rev = tf.reverse_sequence(reset_probs_shifted, length, 1)
    inputs_rev = (forward_memories_rev, reset_probs_shifted_rev)
    rev_memories, _ = tf.nn.dynamic_rnn(memory_rnn, inputs_rev, length - 1, initial_state=final_memory)
    memories = tf.reverse_sequence(rev_memories, length, 1)

    return memories, address_probs, reset_probs


class BackwardAssociativeMemoryCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_slots, slot_dim):
        self._num_slots = num_slots
        self._slot_dim = slot_dim

    def __call__(self, inputs, memory, scope=None):
        forward_memory, reset = inputs
        reset = tf.expand_dims(reset, 2)
        forward_memory = tf.reshape(forward_memory, [-1, self._num_slots, self._slot_dim])
        new_memory = reset * forward_memory + (1.0 - reset) * memory
        return tf.reshape(new_memory, [-1, self.output_size]), new_memory

    @property
    def output_size(self):
        return self._num_slots * self._slot_dim

    @property
    def state_size(self):
        return tf.TensorShape([self._num_slots, self._slot_dim])


class AssociativeMemoryCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_slots, slot_dim):
        self._num_slots = num_slots
        self._slot_dim = slot_dim

    def __call__(self, inputs, memory, scope=None):
        inputs, address, reset = inputs

        memory = memory * tf.expand_dims(reset, 2)

        read = tf.einsum('ab,abc->ac', address, memory)
        address_input = tf.layers.dense(address, self._slot_dim, use_bias=False)
        inputs = tf.concat([inputs, read, address_input], 1)
        new_state = tf.layers.dense(tf.layers.dense(inputs, self._slot_dim, tf.nn.relu), self._slot_dim, tf.nn.relu)
        gate = tf.layers.dense(tf.concat([read, new_state], 1), self._slot_dim, tf.sigmoid)
        new_state = gate * read + (1.0 - gate) * new_state
        address = tf.expand_dims(address, 2)
        new_memory = (1.0 - address) * memory + address * tf.expand_dims(new_state, 1)

        return tf.reshape(new_memory, [-1, self.output_size]), new_memory

    @property
    def output_size(self):
        return self._num_slots * self._slot_dim

    @property
    def state_size(self):
        return tf.TensorShape([self._num_slots, self._slot_dim])
