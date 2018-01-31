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


def bi_assoc_mem_net(sequence, length, key_dim, num_slots, slot_dim, controller_config, is_eval):
    controller_out = modular_encoder(
        controller_config, {'text': sequence}, {'text': length}, {},
        key_dim, 0.0, is_eval=is_eval)[0]['text']

    address_logits = tf.layers.dense(tf.layers.dense(controller_out, key_dim), num_slots, use_bias=False)
    access_probs = tf.cond(is_eval,
                           lambda: tf.one_hot(tf.argmax(address_logits, axis=-1), num_slots, axis=-1),
                           lambda: gumbel_softmax(address_logits))

    tf.identity(tf.nn.softmax(address_logits), name='access_probs')

    reset_logits = tf.layers.dense(tf.layers.dense(controller_out, key_dim), num_slots,
                                   bias_initializer=tf.constant_initializer(-2.0))
    reset_probs = tf.cond(is_eval,
                          lambda: tf.round(tf.sigmoid(reset_logits)),
                          lambda: gumbel_sigmoid(reset_logits))

    reset_access_probs = tf.maximum(0.0, access_probs[:, 1:] - access_probs[:, :-1])
    reset_access_probs = tf.concat([tf.zeros([tf.shape(access_probs)[0], 1, num_slots]), reset_access_probs], 1)
    reset_probs_fw = tf.maximum(tf.minimum(reset_probs, 1.0 - access_probs), reset_access_probs)

    reset_access_probs = tf.maximum(0.0, access_probs[:, :-1] - access_probs[:, 1:])
    reset_access_probs = tf.concat([reset_access_probs, tf.zeros([tf.shape(access_probs)[0], 1, num_slots])], 1)
    reset_probs_bw = tf.maximum(tf.minimum(reset_probs, 1.0 - access_probs), reset_access_probs)

    tf.identity(tf.sigmoid(reset_logits), name='reset_probs')

    memory_rnn = AssociativeMemoryCell(num_slots, slot_dim)

    fw_memories, _ = tf.nn.dynamic_rnn(
        memory_rnn, (sequence, access_probs, reset_probs_fw), length, dtype=tf.float32, scope='forward')

    bw_memories, _ = tf.nn.dynamic_rnn(
        memory_rnn,
        (tf.reverse_sequence(sequence, length, seq_dim=1),
         tf.reverse_sequence(access_probs, length, seq_dim=1),
         tf.reverse_sequence(reset_probs_bw, length, seq_dim=1)), length,
        dtype=tf.float32, scope='backward')

    memories = tf.split(tf.concat([fw_memories, tf.reverse_sequence(bw_memories, length, seq_dim=1)], 2), num_slots, 2)
    memories = tf.concat([tf.layers.dense(m, slot_dim, tf.nn.relu) for m in memories], 2)

    return tf.concat(memories, 2), access_probs


class _SumReset(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_slots):
        self._num_slots = num_slots

    def __call__(self, inputs, state, scope=None):
        scores, reset = inputs
        state *= reset
        state += scores
        return state, state

    @property
    def output_size(self):
        return self._num_slots

    @property
    def state_size(self):
        return self._num_slots


def segmentation_net(sequence, length, key_dim, num_slots, slot_dim, controller_config, is_eval):
    num_slots = 1
    controller_out = modular_encoder(
        controller_config['encoder'], {'text': sequence}, {'text': length}, {},
        key_dim, 0.0, is_eval=is_eval)[0]['text']

    write_logits = tf.layers.dense(tf.layers.dense(controller_out, key_dim), 1)
    write_probs = tf.cond(is_eval,
                          lambda: tf.round(tf.sigmoid(write_logits)),
                          lambda: gumbel_sigmoid(write_logits))

    tf.add_to_collection(
        tf.GraphKeys.LOSSES,
        controller_config.get('lambda_write', 0.0001) * tf.reduce_mean(
            tf.reduce_sum(tf.sigmoid(write_logits), reduction_indices=[1, 2])))

    tf.identity(tf.nn.sigmoid(write_logits), name='write_probs')
    tf.identity(write_probs, name='address_probs')
    reset_probs = tf.concat([tf.zeros([tf.shape(write_probs)[0], 1, 1]), write_probs[:, :-1]], 1)
    tf.identity(reset_probs, name='reset_probs')
    memory_rnn = AssociativeMemoryCell(num_slots, slot_dim)
    forward_memories, final_memory = tf.nn.dynamic_rnn(
        memory_rnn, (sequence, write_probs, reset_probs), length, dtype=tf.float32)

    return forward_memories[:, :, -slot_dim:], final_memory[0], controller_out, write_probs, reset_probs


def associative_mem_net(sequence, length, key_dim, num_slots, slot_dim, controller_config, is_eval):
    controller_out = modular_encoder(
        controller_config['encoder'], {'text': sequence}, {'text': length}, {},
        key_dim, 0.0, is_eval=is_eval)[0]['text']

    write_logits = tf.layers.dense(tf.layers.dense(controller_out, key_dim), 1,
                                   bias_initializer=tf.constant_initializer(0.0))
    write_probs = tf.cond(is_eval,
                          lambda: tf.round(tf.sigmoid(write_logits)),
                          lambda: gumbel_sigmoid(write_logits))

    tf.add_to_collection(
        tf.GraphKeys.LOSSES,
        controller_config.get('lambda_write', 0.0) * tf.reduce_mean(
            tf.reduce_sum(tf.sigmoid(write_logits), reduction_indices=[1, 2])))
    tf.identity(tf.nn.sigmoid(write_logits), name='write_probs')

    reset_logits = tf.layers.dense(tf.layers.dense(controller_out, key_dim), 1,
                                   bias_initializer=tf.constant_initializer(0.0))
    reset_probs = tf.cond(is_eval,
                          lambda: tf.round(tf.sigmoid(reset_logits)),
                          lambda: gumbel_sigmoid(reset_logits))
    buffer_reset_probs = tf.concat([tf.ones([tf.shape(write_probs)[0], 1, 1]), write_probs[:, :-1]], 1)
    reset_probs *= buffer_reset_probs
    address_logits = tf.layers.dense(tf.layers.dense(controller_out, key_dim), num_slots, use_bias=False)
    address_probs = tf.cond(is_eval,
                            lambda: tf.one_hot(tf.argmax(address_logits, axis=-1), num_slots, axis=-1),
                            lambda: gumbel_softmax(address_logits))
    address_probs *= write_probs

    tf.identity(tf.nn.softmax(address_logits) * tf.nn.sigmoid(write_logits), name='address_probs')
    tf.identity(tf.sigmoid(reset_logits), name='reset_probs')

    memory_rnn = AssociativeMemoryCell(num_slots, slot_dim)

    forward_memories, final_memory = tf.nn.dynamic_rnn(
        memory_rnn, (sequence, address_probs, reset_probs), length, dtype=tf.float32)

    reset_probs = tf.concat([tf.tile(reset_probs, [1, 1, num_slots]), buffer_reset_probs], 2)

    return forward_memories, final_memory[0], address_probs, reset_probs


def bidirectional_associative_mem_net(sequence, length, key_dim, num_slots, slot_dim, controller_config, is_eval):
    forward_memories, final_memory, address_probs, reset_probs = associative_mem_net(
        sequence, length, key_dim, num_slots, slot_dim, controller_config, is_eval)

    # copy memory states back through time, between memory states
    memory_rnn = BackwardAssociativeMemoryCell(num_slots, slot_dim)

    reset_probs_shifted = tf.concat(
        [reset_probs[:, 1:, :], tf.zeros([tf.shape(reset_probs)[0], 1, tf.shape(reset_probs)[2]])], 1)
    forward_memories_rev = tf.reverse_sequence(forward_memories, length, 1)
    reset_probs_shifted_rev = tf.reverse_sequence(reset_probs_shifted, length, 1)
    inputs_rev = (forward_memories_rev, reset_probs_shifted_rev)
    rev_memories, _ = tf.nn.dynamic_rnn(memory_rnn, inputs_rev, length - 1, initial_state=final_memory)
    memories = tf.reverse_sequence(rev_memories, length, 1)

    return memories, address_probs, reset_probs


class BackwardAssociativeMemoryCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_slots, slot_dim):
        self._num_slots = num_slots + 1
        self._slot_dim = slot_dim

    def __call__(self, inputs, memory, scope=None):
        forward_memory, reset_probs = inputs
        reset_probs = tf.expand_dims(reset_probs, 2)
        forward_memory = tf.reshape(forward_memory, [-1, self._num_slots, self._slot_dim])
        new_memory = reset_probs * forward_memory + (1.0 - reset_probs) * memory
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
        self._cell = tf.contrib.rnn.GRUBlockCell(slot_dim)

    def __call__(self, inputs, state, scope=None):
        inputs, address, reset = inputs
        memory, buffer_state = state
        memory = memory[:, :self._num_slots]
        write2memory = tf.reduce_sum(address, -1, keep_dims=True)

        memory *= (1.0 - tf.expand_dims(reset, 2))
        new_buffer_state = self._cell(inputs, buffer_state)[0]
        new_state = tf.layers.dense(new_buffer_state, self._num_slots * self._slot_dim, tf.nn.relu)
        new_state = tf.reshape(new_state, [-1, self._num_slots, self._slot_dim])
        address = tf.expand_dims(address, 2)
        new_state = address * new_state
        new_memory = (1.0 - address) * memory + new_state
        out = tf.concat([tf.reshape(new_memory, [-1, self._num_slots * self._slot_dim]), new_buffer_state], 1)
        new_buffer_state *= (1.0 - write2memory)
        return out, (tf.reshape(out, [-1, self._num_slots + 1, self._slot_dim]), new_buffer_state)

    @property
    def output_size(self):
        return (self._num_slots + 1) * self._slot_dim

    @property
    def state_size(self):
        return (tf.TensorShape([self._num_slots + 1, self._slot_dim]), self._slot_dim)


class AssociativeBoWMemoryCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_slots, slot_dim):
        self._num_slots = num_slots
        self._slot_dim = slot_dim

    def __call__(self, inputs, memory, scope=None):
        inputs, address, reset = inputs

        memory *= (1.0 - tf.expand_dims(reset, 2))
        read = tf.einsum('ab,abc->ac', address, memory)
        address_input = tf.layers.dense(address, self._slot_dim, use_bias=False)
        inputs = tf.concat([inputs, read, address_input], 1)
        new_state = tf.layers.dense(inputs, self._slot_dim, tf.nn.tanh)
        new_state = read + new_state
        address = tf.expand_dims(address, 2)
        new_memory = (1.0 - address) * memory + address * tf.expand_dims(new_state, 1)

        return tf.reshape(tf.nn.l2_normalize(new_memory, 2), [-1, self.output_size]), new_memory

    @property
    def output_size(self):
        return self._num_slots * self._slot_dim

    @property
    def state_size(self):
        return tf.TensorShape([self._num_slots, self._slot_dim])
