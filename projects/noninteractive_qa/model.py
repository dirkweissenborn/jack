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
from jack.tfutil.sequence_encoder import gated_linear_convnet
from jack.tfutil.xqa import xqa_crossentropy_loss
from projects.noninteractive_qa.multilevel_seq_encoder import *


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

                emb_question = tf.layers.dense(emb_question, repr_dim, tf.tanh, name="embeddings_projection")
                emb_question = highway_network(emb_question, 1)
                vs.reuse_variables()
                emb_support = tf.layers.dense(emb_support, repr_dim, tf.tanh, name="embeddings_projection")
                emb_support = highway_network(emb_support, 1)

                all_words = tf.reshape(tf.concat([tensors.question_words, tensors.support_words], 1), [-1])
                all_embeddings = tf.unsorted_segment_max(
                    tf.reshape(tf.concat([emb_question, emb_support], 1), [-1, repr_dim]), all_words,
                    tf.reduce_max(all_words) + 1)

                mask = tf.nn.dropout(tf.ones([1, 1, repr_dim]), keep_prob=1.0 - dropout)
                emb_support, emb_question = tf.cond(tensors.is_eval,
                                                    lambda: (emb_support, emb_question),
                                                    lambda: (emb_support * mask, emb_question * mask))

        step = tf.train.get_global_step() or tf.constant(100000, tf.int32)

        def encoding(inputs, length, input_words, reuse=False, regularize=False):
            with tf.variable_scope("encoding", reuse=reuse):
                with tf.variable_scope("controller"):
                    controller_out = modular_encoder(
                        shared_resources.config['controller']['encoder'],
                        {'text': inputs}, {'text': length}, {},
                        repr_dim, dropout, is_eval=tensors.is_eval)[0]['text']

                segm_probs, segm_logits = edge_detection_encoder(controller_out, repr_dim, tensors.is_eval)
                segm_probs_stop = segm_probs  # tf.stop_gradient(segm_probs)
                tf.identity(tf.sigmoid(segm_logits), name='segm_probs')
                frame_probs, frame_logits = edge_detection_encoder(
                    controller_out, repr_dim, tensors.is_eval, mask=segm_probs)
                tf.identity(tf.sigmoid(frame_logits), name='frame_probs')

                with tf.variable_scope("representations"):
                    with tf.variable_scope("phrase") as vs:
                        segms = bow_start_end_segm_encoder(inputs, length, repr_dim, segm_probs)
                        vs.reuse_variables()
                        segms_stop = segms  # bow_start_end_segm_encoder(inputs, length, repr_dim, segm_probs_stop)

                    frame, slots = None, []

                    if 'frame' in shared_resources.config['prediction_levels']:
                        frame, probs = weighted_bow_segm_encoder(inputs, length, repr_dim, frame_probs, segm_probs_stop)
                        tf.identity(probs, 'frame_attn')

                        if 'assoc' in shared_resources.config['prediction_levels']:
                            left_segm_contribs = left_segm_sum_contributions(segm_probs_stop, length)
                            right_segm_contribs = right_segm_sum_contributions(segm_probs_stop, length)

                            left_segms = tf.matmul(left_segm_contribs, segms_stop)
                            right_segms = tf.matmul(right_segm_contribs, segms_stop)
                            left2_segms = tf.matmul(left_segm_contribs, left_segms)
                            right2_segms = tf.matmul(right_segm_contribs, right_segms)
                            frame_ctrl = tf.layers.dense(
                                tf.concat([left_segms, left2_segms, right_segms, right2_segms], 2),
                                repr_dim, tf.nn.relu)
                            ctrl = tf.concat([segms_stop, frame_ctrl, frame], 2)

                            memory, assoc_probs, address_logits = softmax_assoc_memory_encoder(
                                length, repr_dim, shared_resources.config['num_slots'], frame_probs, segm_probs_stop,
                                segms_stop, ctrl, tensors.is_eval)
                            # [B, L, N], [B, L, N * S] -> [B, L, S]
                            slots = tf.split(memory, shared_resources.config['num_slots'], 2)

                            tf.identity(tf.nn.softmax(address_logits), 'assoc_probs')
                            if regularize:
                                weights = tf.sequence_mask(length, dtype=tf.float32)
                                memory_reshaped = tf.reshape(
                                    memory, [-1, tf.shape(memory)[1], shared_resources.config['num_slots'], repr_dim])
                                selected_slot = tf.squeeze(
                                    tf.matmul(tf.expand_dims(assoc_probs, 2), memory_reshaped), 2)
                                logits = tf.matmul(
                                    tf.layers.dense(tf.reshape(selected_slot, [-1, repr_dim]), repr_dim, tf.tanh),
                                    all_embeddings, transpose_b=True)
                                loss = \
                                    tf.losses.sparse_softmax_cross_entropy(
                                        tf.reshape(input_words, [-1]), logits, weights=tf.reshape(weights, [-1]),
                                        loss_collection=None)
                                tf.add_to_collection(tf.GraphKeys.LOSSES, 0.03 * loss)

            return controller_out, segms, frame, slots, frame_probs, segm_probs

        s_ngram, s_segms, s_frames, s_slots, s_boundary_probs, s_segm_probs = encoding(
            emb_support, tensors.support_length, tensors.support_words, regularize=False)
        q_ngram, q_segms, q_frames, q_slots, q_boundary_probs, q_segm_probs = encoding(
            emb_question, tensors.question_length, tensors.question_words, True)

        # computing single time attention over question
        question_attention_weights = compute_question_weights(q_ngram, tensors.question_length)
        support_mask = misc.mask_for_lengths(tensors.support_length)

        def score(q, s, name):
            with tf.variable_scope(name) as vs:
                q = tf.reduce_sum(question_attention_weights * q, 1)
                q = tf.gather(q, tensors.support2question)
                question_hidden = tf.layers.dense(q, repr_dim, tf.nn.relu, name="hidden")
                vs.reuse_variables()
                hidden = tf.layers.dense(s, repr_dim, tf.nn.relu, name="hidden")
                scores = tf.einsum('ik,ijk->ij', question_hidden, hidden)
                tf.identity(scores, name=name)
                return scores

        all_start_scores = []
        all_end_scores = []
        q_reps = []
        s_reps = []

        if 'word' in shared_resources.config['prediction_levels']:
            all_start_scores.append(score(emb_question, emb_support, 'start_word_score'))
            all_end_scores.append(score(emb_question, emb_support, 'end_word_score'))
            q_reps.append(emb_question)
            s_reps.append(emb_support)
        if 'ngram' in shared_resources.config['prediction_levels']:
            all_start_scores.append(score(q_ngram, s_ngram, 'start_ngram_score'))
            all_end_scores.append(score(q_ngram, s_ngram, 'end_ngram_score'))
            q_reps.append(q_ngram)
            s_reps.append(s_ngram)
        if 'segm' in shared_resources.config['prediction_levels']:
            all_start_scores.append(score(q_segms, s_segms, 'start_segm_score'))
            all_end_scores.append(score(q_segms, s_segms, 'end_segm_score'))
            q_reps.append(q_segms)
            s_reps.append(s_segms)
        if 'frame' in shared_resources.config['prediction_levels']:
            frames = score(q_frames, s_frames, 'frames_score')
            all_start_scores.append(frames)
            all_end_scores.append(frames)
            q_reps.append(q_frames)
            s_reps.append(s_frames)

        if 'assoc' in shared_resources.config['prediction_levels']:
            assoc_scores = tf.add_n(
                [score(q, s, 'assoc_' + str(i)) for i, (q, s) in enumerate(zip(q_slots, s_slots))]) / len(q_slots)
            all_start_scores.append(assoc_scores)
            all_end_scores.append(assoc_scores)
            q_reps.extend(tf.unstack(q_slots))
            s_reps.extend(tf.unstack(s_slots))

        tf.concat(q_reps, 2, name='question_representation')
        tf.concat(s_reps, 2, name='support_representation')

        start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
            compute_spans(tf.add_n(all_start_scores) + support_mask, tf.add_n(all_end_scores) + support_mask,
                          tensors.answer2support, tensors.is_eval,
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

                mask = tf.nn.dropout(tf.ones([1, 1, repr_dim]), keep_prob=1.0 - dropout)
                emb_support, emb_question = tf.cond(tensors.is_eval,
                                                    lambda: (emb_support, emb_question),
                                                    lambda: (emb_support * mask, emb_question * mask))

        step = tf.train.get_global_step() or tf.constant(100000, tf.int32)

        def encoding(inputs, length, reuse=False):
            representations = list()
            with tf.variable_scope("encoding", reuse=reuse):
                ctrl = gated_linear_convnet(repr_dim, inputs, 2, 5)
                segm_probs, segm_logits = edge_detection_encoder(ctrl, repr_dim, tensors.is_eval)
                pop_probs, pop_logits = edge_detection_encoder(ctrl, repr_dim, tensors.is_eval,
                                                               mask=segm_probs)

                push_mask = tf.concat([tf.ones([tf.shape(segm_probs)[0], 1, 1]), segm_probs[:, :-1]], 1) * (
                    1.0 - pop_probs)
                push_probs, push_logits = edge_detection_encoder(ctrl, repr_dim, tensors.is_eval, mask=push_mask)

                tf.identity(tf.sigmoid(segm_logits), name='segm_probs')
                tf.identity(tf.sigmoid(push_logits), name='push_probs')
                tf.identity(tf.sigmoid(pop_logits), name='pop_probs')
                depth = shared_resources.config['depth']

                representations.append(inputs)
                representations.append(ctrl)

                # [B, L, 1] -> [B, L, D, D]
                push_matrix = tf.tile(tf.expand_dims(push_probs, 3), [1, 1, depth, depth])
                push_matrix = tf.matrix_band_part(push_matrix, 0, 1)
                pop_matrix = tf.tile(tf.expand_dims(pop_probs, 3), [1, 1, depth, depth])
                pop_matrix = tf.matrix_band_part(pop_matrix, 1, 0)
                push_pop_matrix = push_matrix + pop_matrix
                push_pop_matrix -= tf.matrix_band_part(push_pop_matrix, 0, 0)

                push_pop_diag = tf.reduce_sum(push_pop_matrix, 3)
                push_pop_matrix += tf.matrix_diag(1.0 - push_pop_diag)

                # push_pop_matrix = tf.Print(push_pop_matrix, [push_pop_matrix], summarize=30)

                def push_pop_ctrl(acc, pp_matrix):
                    acc = tf.einsum('ij,ijk->ik', acc, pp_matrix)
                    return acc

                depth_prob_init = tf.one_hot(
                    tf.zeros([tf.shape(inputs)[1], tf.shape(inputs)[0]], dtype=tf.int32), depth)

                depth_prob = tf.scan(push_pop_ctrl, tf.transpose(push_pop_matrix, [1, 0, 2, 3]),
                                     initializer=depth_prob_init[0])

                # depth_prob = tf.cond(step > 1000, lambda: depth_prob, lambda: depth_prob_init)
                # depth_prob = tf.Print(depth_prob, [depth_prob], summarize=30)

                depth_prob_shift = tf.concat([depth_prob_init[:1], depth_prob[:-1]], 0)
                depth_prob = tf.transpose(depth_prob, [1, 0, 2])
                depth_prob_shift = tf.transpose(depth_prob_shift, [1, 0, 2])

                tf.identity(depth_prob, 'depth_prob')

                frames = list()
                segms = list()
                depth_prob_split = tf.split(depth_prob, depth, 2)

                depth_prob_shift_split = tf.split(depth_prob_shift, depth, 2)

                num_slots = 1
                if shared_resources.config.get('assoc', False):
                    num_slots = shared_resources.config['num_slots']

                for i, (p, p_shift) in enumerate(zip(depth_prob_split, depth_prob_shift_split)):
                    with tf.variable_scope('segmentation', reuse=i > 0):
                        frame_probs = pop_probs * p_shift
                        if i > 0:
                            p += frame_probs  # share segment end on pop with lower layer
                        this_segm_probs = segm_probs * p
                        this_segms = bow_start_end_segm_encoder(inputs, length, repr_dim, this_segm_probs)
                        if shared_resources.config.get('assoc', False):
                            slots, assoc_probs = incremental_assoc_memory_encoder(
                                length, repr_dim, shared_resources.config['num_slots'], frame_probs,
                                this_segm_probs, this_segms, this_segms, tensors.is_eval)
                            tf.identity(tf.concat(assoc_probs, 2), 'assoc_probs_' + str(i))
                        else:
                            slots = [bow_segm_encoder(this_segms, length, repr_dim, frame_probs, this_segm_probs)]
                        frames.append(tf.stack(slots, 2))
                        segms.append(this_segms)

                representations.append(tf.reduce_sum(tf.stack(segms, 2) * tf.expand_dims(depth_prob, 3), 2))
                frames = tf.reduce_sum(tf.stack(frames, 2) * depth_prob[:, :, :, tf.newaxis, tf.newaxis], 2)
                frames = tf.cond(step >= 5000, lambda: frames, lambda: tf.stop_gradient(frames))
                if num_slots > 1:
                    representations.extend(tf.squeeze(f, 2) for f in tf.split(frames, num_slots, 2))
                else:
                    representations.append(tf.squeeze(frames, 2))

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
