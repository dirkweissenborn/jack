"""
This file contains FastQA specific modules and ports
"""

from jack.core import *
from jack.readers.extractive_qa.tensorflow.abstract_model import AbstractXQAModelModule
from jack.readers.extractive_qa.tensorflow.answer_layer import compute_question_state, compute_spans, \
    compute_question_weights
from jack.tfutil import misc
from jack.tfutil.embedding import conv_char_embedding
from jack.tfutil.highway import highway_network
from jack.tfutil.modular_encoder import modular_encoder
from jack.tfutil.sequence_encoder import gated_linear_convnet
from jack.tfutil.xqa import xqa_crossentropy_loss
from projects.noninteractive_qa.multilevel_seq_encoder import assoc_memory_encoder, edge_detection_encoder, \
    left_segm_sum_contributions, \
    right_segm_sum_contributions, bow_start_end_segm_encoder


class NonInteractiveModularQAModule(AbstractXQAModelModule):
    # @property
    # def training_input_ports(self) -> Sequence[TensorPort]:
    #   return super().training_input_ports

    # @property
    # def output_ports(self) -> Sequence[TensorPort]:
    #    return super().output_ports + [question_state_start_port, question_state_end_port]

    def create_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)

        input_size = shared_resources.config["repr_dim_input"]
        repr_dim = shared_resources.config["repr_dim"]
        with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)
        dropout = shared_resources.config.get("dropout", 0.0)

        # set shapes for inputs
        tensors.emb_question.set_shape([None, None, input_size])
        tensors.emb_support.set_shape([None, None, input_size])

        emb_question = tensors.emb_question
        emb_support = tensors.emb_support
        if with_char_embeddings:
            with tf.variable_scope("char_embeddings") as vs:
                # compute combined embeddings
                [char_emb_question, char_emb_support] = conv_char_embedding(
                    len(shared_resources.char_vocab), repr_dim, tensors.word_chars, tensors.word_char_length,
                    [tensors.question_words, tensors.support_words])

                emb_question = tf.concat([emb_question, char_emb_question], 2)
                emb_support = tf.concat([emb_support, char_emb_support], 2)
                input_size += repr_dim

                # set shapes for inputs
                emb_question.set_shape([None, None, input_size])
                emb_support.set_shape([None, None, input_size])

                emb_question = tf.layers.dense(emb_question, repr_dim, name="embeddings_projection")
                emb_question = highway_network(emb_question, 1)
                vs.reuse_variables()
                emb_support = tf.layers.dense(emb_support, repr_dim, name="embeddings_projection")
                emb_support = highway_network(emb_support, 1)

        with tf.variable_scope("encoder") as vs:
            encoded_question = modular_encoder(
                shared_resources.config['encoder'],
                {'text': emb_question}, {'text': tensors.question_length}, {},
                repr_dim, dropout, tensors.is_eval)[0]['text']
            vs.reuse_variables()
            encoded_support = modular_encoder(
                shared_resources.config['encoder'],
                {'text': emb_support}, {'text': tensors.support_length}, {},
                repr_dim, dropout, tensors.is_eval)[0]['text']

        start_scores, end_scores, span = _simple_answer_layer(
            encoded_question, encoded_support, repr_dim, shared_resources, tensors)

        return TensorPort.to_mapping(self.output_ports, (start_scores, end_scores, span))


all_start_scores = TensorPort(tf.float32, [None, None, None], 'all_start_scores')
all_end_scores = TensorPort(tf.float32, [None, None, None], 'all_end_scores')


class MultilevelSequenceEncoderQAModule(AbstractXQAModelModule):
    # @property
    # def training_input_ports(self) -> Sequence[TensorPort]:
    #    return super().training_input_ports + [all_start_scores, all_end_scores]

    # @property
    # def output_ports(self) -> Sequence[TensorPort]:
    #    return super().output_ports + [all_start_scores, all_end_scores]

    def create_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)

        input_size = shared_resources.config["repr_dim_input"]
        repr_dim = shared_resources.config["repr_dim"]
        with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)
        dropout = shared_resources.config.get("dropout", 0.0)

        # set shapes for inputs
        tensors.emb_question.set_shape([None, None, input_size])
        tensors.emb_support.set_shape([None, None, input_size])

        emb_question = tensors.emb_question
        emb_support = tensors.emb_support
        if with_char_embeddings:
            with tf.variable_scope("char_embeddings") as vs:
                # compute combined embeddings
                [char_emb_question, char_emb_support] = conv_char_embedding(
                    len(shared_resources.char_vocab), repr_dim, tensors.word_chars, tensors.word_char_length,
                    [tensors.question_words, tensors.support_words])

                emb_question = tf.concat([emb_question, char_emb_question], 2)
                emb_support = tf.concat([emb_support, char_emb_support], 2)
                input_size += repr_dim

                # set shapes for inputs
                emb_question.set_shape([None, None, input_size])
                emb_support.set_shape([None, None, input_size])

                emb_question = tf.layers.dense(emb_question, repr_dim, name="embeddings_projection")
                emb_question = highway_network(emb_question, 1)
                vs.reuse_variables()
                emb_support = tf.layers.dense(emb_support, repr_dim, name="embeddings_projection")
                emb_support = highway_network(emb_support, 1)

                mask = tf.nn.dropout(tf.ones([1, 1, repr_dim]), keep_prob=1.0 - dropout)
                emb_support, emb_question = tf.cond(tensors.is_eval,
                                                    lambda: (emb_support, emb_question),
                                                    lambda: (emb_support * mask, emb_question * mask))

        step = tf.train.get_global_step() or tf.constant(20001, tf.int32)

        def encoding(inputs, length, reuse=False):
            with tf.variable_scope("encoding", reuse=reuse):
                with tf.variable_scope("controller"):
                    controller_out = modular_encoder(
                        shared_resources.config['controller']['encoder'],
                        {'text': inputs}, {'text': length}, {},
                        repr_dim, dropout, is_eval=tensors.is_eval)[0]['text']

                representations = {"ctrl": controller_out}

                segm_probs, segm_logits = edge_detection_encoder(controller_out, repr_dim, tensors.is_eval)
                frame_probs, frame_logits = edge_detection_encoder(
                    controller_out, repr_dim, tensors.is_eval, mask=segm_probs)

                tf.identity(tf.sigmoid(segm_logits), name='segm_probs')
                tf.identity(tf.sigmoid(frame_logits), name='frame_probs')

                with tf.variable_scope("representations"):
                    representations['word'] = inputs
                    segms = bow_start_end_segm_encoder(inputs, length, repr_dim, segm_probs, tensors.is_eval)
                    representations['segm'] = segms

                    left_segm_contribs = left_segm_sum_contributions(segm_probs, length)
                    right_segm_contribs = right_segm_sum_contributions(segm_probs, length)

                    left_segms = tf.matmul(left_segm_contribs, segms)
                    right_segms = tf.matmul(right_segm_contribs, segms)

                    if (shared_resources.config.get('num_slots', 0) and
                                'assoc' in shared_resources.config['prediction_levels']):
                        assoc_ctrl = tf.concat([segms, left_segms, right_segms], 2)
                        memory, _, _ = assoc_memory_encoder(
                            length, repr_dim, shared_resources.config['num_slots'], frame_probs,
                            segm_probs, segms, assoc_ctrl, tensors.is_eval)
                        if shared_resources.config.get('load_dir') is None:
                            memory = tf.cond(step >= 5000, lambda: memory, lambda: tf.stop_gradient(memory))
                        for i, m in enumerate(tf.split(memory, shared_resources.config['num_slots'], 2)):
                            representations['assoc_' + str(i)] = m

            return representations, frame_probs, frame_logits, segm_probs, segm_logits

        encoded_question, q_boundary_probs, q_boundary_logits, q_segm_probs, q_segm_logits = encoding(
            emb_question, tensors.question_length)
        encoded_support, s_boundary_probs, s_boundary_logits, s_segm_probs, s_segm_logits = encoding(
            emb_support, tensors.support_length, True)

        # enforce no boundary in question
        # q_mask = tf.expand_dims(tf.sequence_mask(tensors.question_length - 1, dtype=tf.float32), 2)
        # tf.add_to_collection(
        #   tf.GraphKeys.LOSSES,
        #   shared_resources.config.get('lambda_boundary', 0.0) *
        #   tf.reduce_mean(
        #       tf.reduce_sum(q_boundary_probs[:, 1:] * q_mask, reduction_indices=[1, 2]) -
        #       tf.reduce_sum(q_boundary_probs[:, 0], 1)
        #   ))

        all_start_scores = []
        all_end_scores = []
        # computing single time attention over question
        prediction_levels = [k for k in encoded_question if
                             any(k.startswith(k2) for k2 in shared_resources.config['prediction_levels'])]
        prefix_count = {k2: sum(k.startswith(k2) for k in prediction_levels)
                        for k2 in shared_resources.config['prediction_levels']}
        full_encoded_question = [encoded_question[k] for k in prediction_levels]
        full_encoded_question_splits = [q.get_shape()[2].value for q in full_encoded_question]
        question_attention_weights = compute_question_weights(
            tf.concat([encoded_question['word'], encoded_question['ctrl']], 2), tensors.question_length)
        question_state = tf.reduce_sum(question_attention_weights * tf.concat(full_encoded_question, 2), 1)
        question_state = tf.gather(question_state, tensors.support2question)
        question_state = tf.split(question_state, full_encoded_question_splits, 1)
        for q, k in zip(question_state, prediction_levels):
            with tf.variable_scope(k) as vs:
                question_hidden = tf.layers.dense(q, 2 * repr_dim, tf.nn.relu, name="hidden")
                question_hidden_start, question_hidden_end = tf.split(question_hidden, 2, 1)
                vs.reuse_variables()
                hidden = tf.layers.dense(encoded_support[k], 2 * repr_dim, tf.nn.relu, name="hidden")
                hidden_start, hidden_end = tf.split(hidden, 2, 2)
                support_mask = misc.mask_for_lengths(tensors.support_length)
                start_scores = tf.einsum('ik,ijk->ij', question_hidden_start, hidden_start)
                start_scores = start_scores + support_mask
                end_scores = tf.einsum('ik,ijk->ij', question_hidden_end, hidden_end)
                end_scores = end_scores + support_mask

                prefix_ct = list(prefix_count[k2] for k2 in shared_resources.config['prediction_levels']
                                 if k.startswith(k2))[0]
                all_start_scores.append(start_scores / prefix_ct)
                all_end_scores.append(end_scores / prefix_ct)

        start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
            compute_spans(tf.add_n(all_start_scores), tf.add_n(all_end_scores), tensors.answer2support, tensors.is_eval,
                          tensors.support2question, max_span_size=shared_resources.config.get('max_span_size', 16))
        span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)

        return TensorPort.to_mapping(self.output_ports, (start_scores, end_scores, span))

    def create_training_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)
        loss = xqa_crossentropy_loss(
            tensors.start_scores, tensors.end_scores, tensors.answer_span,
            tensors.answer2support, tensors.support2question,
            use_sum=shared_resources.config.get('loss', 'sum') == 'sum')
        if tf.get_collection(tf.GraphKeys.LOSSES):
            loss += tf.reduce_sum(tf.get_collection(tf.GraphKeys.LOSSES))
        return {Ports.loss: loss}


class HierarchicalSegmentQAModule(AbstractXQAModelModule):
    # @property
    # def training_input_ports(self) -> Sequence[TensorPort]:
    #    return super().training_input_ports + [all_start_scores, all_end_scores]

    # @property
    # def output_ports(self) -> Sequence[TensorPort]:
    #    return super().output_ports + [all_start_scores, all_end_scores]

    def create_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)

        input_size = shared_resources.config["repr_dim_input"]
        repr_dim = shared_resources.config["repr_dim"]
        with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)
        dropout = shared_resources.config.get("dropout", 0.0)

        # set shapes for inputs
        tensors.emb_question.set_shape([None, None, input_size])
        tensors.emb_support.set_shape([None, None, input_size])

        emb_question = tensors.emb_question
        emb_support = tensors.emb_support
        if with_char_embeddings:
            with tf.variable_scope("char_embeddings") as vs:
                # compute combined embeddings
                [char_emb_question, char_emb_support] = conv_char_embedding(
                    len(shared_resources.char_vocab), repr_dim, tensors.word_chars, tensors.word_char_length,
                    [tensors.question_words, tensors.support_words])

                emb_question = tf.concat([emb_question, char_emb_question], 2)
                emb_support = tf.concat([emb_support, char_emb_support], 2)
                input_size += repr_dim

                # set shapes for inputs
                emb_question.set_shape([None, None, input_size])
                emb_support.set_shape([None, None, input_size])

                emb_question = tf.layers.dense(emb_question, repr_dim, name="embeddings_projection")
                emb_question = highway_network(emb_question, 1)
                vs.reuse_variables()
                emb_support = tf.layers.dense(emb_support, repr_dim, name="embeddings_projection")
                emb_support = highway_network(emb_support, 1)

        step = tf.train.get_global_step() or tf.constant(10000, tf.int32)

        def encoding(inputs, length, reuse=False):
            representations = list()
            with tf.variable_scope("encoding", reuse=reuse):
                segm_probs = None
                segms = inputs
                ctrl = gated_linear_convnet(repr_dim, inputs, 2, 5)
                original_ctrl = ctrl
                representations.append(inputs)
                representations.append(ctrl)
                for i in range(shared_resources.config['num_layers']):
                    with tf.variable_scope("layer" + str(i)):
                        segm_probs, segm_logits = edge_detection_encoder(
                            ctrl, repr_dim, tensors.is_eval, mask=segm_probs)
                        segm_probs = tf.cond(step >= 1000 * i,
                                             lambda: segm_probs,
                                             lambda: tf.stop_gradient(segm_probs))
                        tf.identity(tf.sigmoid(segm_logits), name='segm_probs' + str(i))
                        segms = bow_start_end_segm_encoder(segms, length, repr_dim, segm_probs, tensors.is_eval)

                        representations.append(segms)

                        left_segm_contribs = left_segm_sum_contributions(segm_probs, length)
                        right_segm_contribs = right_segm_sum_contributions(segm_probs, length)

                        left_segms = tf.matmul(left_segm_contribs, segms)
                        right_segms = tf.matmul(right_segm_contribs, segms)

                        ctrl = tf.concat([original_ctrl, segms, left_segms, right_segms], 2)

            return representations

        encoded_question = encoding(emb_question, tensors.question_length)
        encoded_support = encoding(emb_support, tensors.support_length, True)

        all_start_scores = []
        all_end_scores = []
        # computing single time attention over question
        encoded_question = tf.concat(encoded_question, 2)
        question_attention_weights = compute_question_weights(encoded_question, tensors.question_length)
        question_state = tf.reduce_sum(question_attention_weights * encoded_question, 1)
        question_state = tf.gather(question_state, tensors.support2question)
        question_state = tf.split(question_state, shared_resources.config['num_layers'] + 2, 1)
        for i, (q, s) in enumerate(zip(question_state, encoded_support)):
            with tf.variable_scope('prediction' + str(i)) as vs:
                question_hidden = tf.layers.dense(q, 2 * repr_dim, tf.nn.relu, name="hidden")
                question_hidden_start, question_hidden_end = tf.split(question_hidden, 2, 1)
                vs.reuse_variables()
                hidden = tf.layers.dense(s, 2 * repr_dim, tf.nn.relu, name="hidden")
                hidden_start, hidden_end = tf.split(hidden, 2, 2)
                support_mask = misc.mask_for_lengths(tensors.support_length)
                start_scores = tf.einsum('ik,ijk->ij', question_hidden_start, hidden_start)
                start_scores = start_scores + support_mask
                end_scores = tf.einsum('ik,ijk->ij', question_hidden_end, hidden_end)
                end_scores = end_scores + support_mask
                all_start_scores.append(start_scores)
                all_end_scores.append(end_scores)

        start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
            compute_spans(tf.add_n(all_start_scores), tf.add_n(all_end_scores), tensors.answer2support, tensors.is_eval,
                          tensors.support2question, max_span_size=shared_resources.config.get('max_span_size', 16))
        span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)

        return TensorPort.to_mapping(self.output_ports, (start_scores, end_scores, span))

    def create_training_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)
        loss = xqa_crossentropy_loss(
            tensors.start_scores, tensors.end_scores, tensors.answer_span,
            tensors.answer2support, tensors.support2question,
            use_sum=shared_resources.config.get('loss', 'sum') == 'sum')
        if tf.get_collection(tf.GraphKeys.LOSSES):
            loss += tf.reduce_sum(tf.get_collection(tf.GraphKeys.LOSSES))
        return {Ports.loss: loss}


class HierarchicalAssocQAModule(AbstractXQAModelModule):
    # @property
    # def training_input_ports(self) -> Sequence[TensorPort]:
    #    return super().training_input_ports + [all_start_scores, all_end_scores]

    # @property
    # def output_ports(self) -> Sequence[TensorPort]:
    #    return super().output_ports + [all_start_scores, all_end_scores]

    def create_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)

        input_size = shared_resources.config["repr_dim_input"]
        repr_dim = shared_resources.config["repr_dim"]
        with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)
        dropout = shared_resources.config.get("dropout", 0.0)

        # set shapes for inputs
        tensors.emb_question.set_shape([None, None, input_size])
        tensors.emb_support.set_shape([None, None, input_size])

        emb_question = tensors.emb_question
        emb_support = tensors.emb_support
        if with_char_embeddings:
            with tf.variable_scope("char_embeddings") as vs:
                # compute combined embeddings
                [char_emb_question, char_emb_support] = conv_char_embedding(
                    len(shared_resources.char_vocab), repr_dim, tensors.word_chars, tensors.word_char_length,
                    [tensors.question_words, tensors.support_words])

                emb_question = tf.concat([emb_question, char_emb_question], 2)
                emb_support = tf.concat([emb_support, char_emb_support], 2)
                input_size += repr_dim

                # set shapes for inputs
                emb_question.set_shape([None, None, input_size])
                emb_support.set_shape([None, None, input_size])

                emb_question = tf.layers.dense(emb_question, repr_dim, name="embeddings_projection")
                emb_question = highway_network(emb_question, 1)
                vs.reuse_variables()
                emb_support = tf.layers.dense(emb_support, repr_dim, name="embeddings_projection")
                emb_support = highway_network(emb_support, 1)

        step = tf.train.get_global_step() or tf.constant(10000, tf.int32)

        def encoding(inputs, length, reuse=False):
            representations = list()
            with tf.variable_scope("encoding", reuse=reuse):
                segm_probs = None
                segms = inputs
                ctrl = gated_linear_convnet(repr_dim, inputs, 2, 5)
                first_ctrl = ctrl
                representations.append(inputs)
                representations.append(ctrl)
                for i in range(shared_resources.config['num_layers']):
                    with tf.variable_scope("layer" + str(i)):
                        prev_segm_probs = segm_probs
                        segm_probs, segm_logits = edge_detection_encoder(
                            ctrl, repr_dim, tensors.is_eval, mask=segm_probs)
                        segm_probs = tf.cond(step >= 2000 * i,
                                             lambda: segm_probs,
                                             lambda: tf.stop_gradient(segm_probs))
                        tf.identity(tf.sigmoid(segm_logits), name='segm_probs' + str(i))

                        prev_segm_probs = prev_segm_probs if i > 0 else tf.ones_like(segm_probs)
                        memory, address_probs, _ = assoc_memory_encoder(
                            length, repr_dim, shared_resources.config['num_slots'], segm_probs, prev_segm_probs, segms,
                            ctrl, tensors.is_eval)

                        representations.extend(tf.split(memory, shared_resources.config['num_slots'], 2))

                        tf.identity(address_probs, name='address_probs' + str(i))

                        segms = tf.layers.dense(tf.layers.dense(memory, 2 * repr_dim, tf.nn.relu), repr_dim, tf.nn.relu)

                        left_segm_contribs = left_segm_sum_contributions(segm_probs, length)
                        right_segm_contribs = right_segm_sum_contributions(segm_probs, length)

                        left_segms = tf.matmul(left_segm_contribs, segms)
                        right_segms = tf.matmul(right_segm_contribs, segms)

                        ctrl = tf.concat([first_ctrl, segms, left_segms, right_segms], 2)

            return representations

        encoded_question = encoding(emb_question, tensors.question_length)
        encoded_support = encoding(emb_support, tensors.support_length, True)

        all_start_scores = []
        all_end_scores = []
        # computing single time attention over question
        encoded_question = tf.concat(encoded_question, 2)
        question_attention_weights = compute_question_weights(encoded_question, tensors.question_length)
        question_state = tf.reduce_sum(question_attention_weights * encoded_question, 1)
        question_state = tf.gather(question_state, tensors.support2question)
        question_state = tf.split(question_state, len(encoded_support), 1)
        for i, (q, s) in enumerate(zip(question_state, encoded_support)):
            with tf.variable_scope('prediction' + str(i)) as vs:
                question_hidden = tf.layers.dense(q, 2 * repr_dim, tf.nn.relu, name="hidden")
                question_hidden_start, question_hidden_end = tf.split(question_hidden, 2, 1)
                vs.reuse_variables()
                hidden = tf.layers.dense(s, 2 * repr_dim, tf.nn.relu, name="hidden")
                hidden_start, hidden_end = tf.split(hidden, 2, 2)
                support_mask = misc.mask_for_lengths(tensors.support_length)
                start_scores = tf.einsum('ik,ijk->ij', question_hidden_start, hidden_start)
                start_scores = start_scores + support_mask
                end_scores = tf.einsum('ik,ijk->ij', question_hidden_end, hidden_end)
                end_scores = end_scores + support_mask
                all_start_scores.append(start_scores)
                all_end_scores.append(end_scores)

        start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
            compute_spans(tf.add_n(all_start_scores), tf.add_n(all_end_scores), tensors.answer2support, tensors.is_eval,
                          tensors.support2question, max_span_size=shared_resources.config.get('max_span_size', 16))
        span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)

        return TensorPort.to_mapping(self.output_ports, (start_scores, end_scores, span))

    def create_training_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)
        loss = xqa_crossentropy_loss(
            tensors.start_scores, tensors.end_scores, tensors.answer_span,
            tensors.answer2support, tensors.support2question,
            use_sum=shared_resources.config.get('loss', 'sum') == 'sum')
        if tf.get_collection(tf.GraphKeys.LOSSES):
            loss += tf.reduce_sum(tf.get_collection(tf.GraphKeys.LOSSES))
        return {Ports.loss: loss}


def _simple_answer_layer(encoded_question, encoded_support, repr_dim, shared_resources, tensors):
    # computing single time attention over question
    question_state = compute_question_state(encoded_question, tensors.question_length)
    question_state = tf.gather(question_state, tensors.support2question)
    question_state = tf.layers.dense(question_state, 2 * repr_dim, tf.nn.relu, name="hidden")
    question_state_start, question_state_end = tf.split(question_state, 2, 1)
    tf.get_variable_scope().reuse_variables()
    hidden = tf.layers.dense(encoded_support, 2 * repr_dim, tf.nn.relu, name="hidden")
    hidden_start, hidden_end = tf.split(hidden, 2, 2)
    support_mask = misc.mask_for_lengths(tensors.support_length)
    start_scores = tf.einsum('ik,ijk->ij', question_state_start, hidden_start)
    start_scores = start_scores + support_mask
    end_scores = tf.einsum('ik,ijk->ij', question_state_end, hidden_end)
    end_scores = end_scores + support_mask
    start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
        compute_spans(start_scores, end_scores, tensors.answer2support, tensors.is_eval, tensors.support2question,
                      max_span_size=shared_resources.config.get('max_span_size', 16))
    span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)
    return start_scores, end_scores, span
