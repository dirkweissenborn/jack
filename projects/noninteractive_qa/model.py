"""
This file contains FastQA specific modules and ports
"""

from jack.core import *
from jack.readers.extractive_qa.tensorflow.abstract_model import AbstractXQAModelModule
from jack.readers.extractive_qa.tensorflow.answer_layer import compute_question_state, compute_spans, \
    compute_question_weights, answer_layer
from jack.tfutil import misc
from jack.tfutil.embedding import conv_char_embedding
from jack.tfutil.highway import highway_network
from jack.tfutil.sequence_encoder import gated_linear_convnet, encoder
from jack.tfutil.xqa import xqa_crossentropy_loss
from projects.noninteractive_qa.multilevel_seq_encoder import *

_start_scores = TensorPort(tf.float32, [None, None], "additional_start_scores",
                           "Represents start scores for each support sequence",
                           "[S, max_num_tokens]")
_end_scores = TensorPort(tf.float32, [None, None], "additional_end_scores",
                         "Represents start scores for each support sequence",
                         "[S, max_num_tokens]")


class NonInteractiveQAModule(AbstractXQAModelModule):
    @property
    def training_input_ports(self) -> Sequence[TensorPort]:
        return super().training_input_ports + [_start_scores, _end_scores]

    @property
    def output_ports(self) -> Sequence[TensorPort]:
        return super().output_ports + [_start_scores, _end_scores]

    def encoder(self, shared_resources, emb, length, tensors):
        raise NotImplementedError()

    def create_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)

        input_size = shared_resources.config["repr_dim_input"]
        repr_dim = shared_resources.config["repr_dim"]
        with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)
        dropout = shared_resources.config.get("dropout", 0.0)
        with_wiq = shared_resources.config.get('with_wiq', False)

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

                emb_question = tf.layers.dense(emb_question, repr_dim - 1 if with_wiq else repr_dim,
                                               activation=tf.tanh, name="embeddings_projection")
                emb_question = highway_network(emb_question, 1)
                vs.reuse_variables()
                emb_support = tf.layers.dense(emb_support, repr_dim - 1 if with_wiq else repr_dim,
                                              activation=tf.tanh, name="embeddings_projection")
                emb_support = highway_network(emb_support, 1)

        mask = tf.nn.dropout(tf.ones([tf.shape(emb_question)[0], 1, repr_dim - 1 if with_wiq else repr_dim]),
                             keep_prob=1.0 - dropout)
        emb_support, emb_question = tf.cond(tensors.is_eval,
                                            lambda: (emb_support, emb_question),
                                            lambda: (emb_support * tf.gather(mask, tensors.support2question),
                                                     emb_question * mask))

        if shared_resources.config.get('with_wiq', False):
            batch_size, q_len, _ = tf.unstack(tf.shape(emb_question))
            emb_question = tf.concat([emb_question, tf.ones([batch_size, q_len, 1])], 2)
            emb_support = tf.concat([emb_support, tf.expand_dims(tensors.word_in_question, 2)], 2)

        with tf.variable_scope("encoder") as vs:
            encoded_question_list = self.encoder(shared_resources, emb_question, tensors.question_length, tensors)
            vs.reuse_variables()
            encoded_support_list = self.encoder(shared_resources, emb_support, tensors.support_length, tensors)

        encoded_question = tf.concat(encoded_question_list, 2, name='question_representation')
        encoded_support = tf.concat(encoded_support_list, 2, name='question_representation')

        with tf.variable_scope("answer_layer"):
            start_scores, end_scores, span = _simple_answer_layer(
                encoded_question_list, encoded_support_list, repr_dim, shared_resources, tensors)

        if shared_resources.config.get('num_interactive', 0):
            for i in range(shared_resources.config.get('num_interactive', 0)):
                with tf.variable_scope('attention', reuse=i > 0):
                    diag = tf.get_variable('attn_weight', [1, 1, encoded_question.get_shape()[-1].value], tf.float32,
                                           initializer=tf.zeros_initializer())
                    diag = tf.nn.sigmoid(diag)
                    # [B, S, Q]
                    attn_scores = tf.einsum('abc,adc->abd', encoded_support, encoded_question * diag)
                    attn_scores += tf.expand_dims(misc.mask_for_lengths(tensors.support_length), 2)
                    # attn_scores /= math.sqrt(float(encoded_question.get_shape()[-1].value))
                    attn_prob = tf.nn.softmax(attn_scores, 1)

                question2support = tf.einsum('bsq,bqr->bsr', attn_prob, emb_question)

                with tf.variable_scope("encoder", reuse=True):
                    encoded_support_list = self.encoder(shared_resources, emb_support + question2support,
                                                        tensors.support_length, tensors)

            with tf.variable_scope("answer_layer", reuse=True):
                new_start_scores, new_end_scores, span = _simple_answer_layer(
                    encoded_question_list, encoded_support_list, repr_dim, shared_resources, tensors)

            return TensorPort.to_mapping(self.output_ports,
                                         (new_start_scores, new_end_scores, span, start_scores, end_scores))

        if shared_resources.config.get('is_interactive', False):
            post_non_interactive_config = shared_resources.config['post_non_interactive']
            encoded, lengths, mappings = modular_encoder(
                post_non_interactive_config['encoder'],
                {'question': encoded_question, 'support': encoded_support},
                {'question': tensors.question_length, 'support': tensors.support_length},
                {'question': None, 'support': tensors.support2question},
                repr_dim, dropout, tensors.is_eval)
            with tf.variable_scope('answer_layer'):
                answer_layer_config = post_non_interactive_config['answer_layer']
                encoded_question = encoded[answer_layer_config.get('question', 'question')]
                encoded_support = encoded[answer_layer_config.get('support', 'support')]

                if 'repr_dim' not in answer_layer_config:
                    answer_layer_config['repr_dim'] = repr_dim
                if 'max_span_size' not in answer_layer_config:
                    answer_layer_config['max_span_size'] = shared_resources.config.get('max_span_size', 16)
                beam_size = tf.get_variable(
                    'beam_size', initializer=shared_resources.config.get('beam_size', 1), dtype=tf.int32,
                    trainable=False)
                beam_size_p = tf.placeholder(tf.int32, [], 'beam_size_setter')
                beam_size_assign = beam_size.assign(beam_size_p)
                self._beam_size_assign = lambda k: self.tf_session.run(beam_size_assign, {beam_size_p: k})

                new_start_scores, new_end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
                    answer_layer(encoded_question, lengths[answer_layer_config.get('question', 'question')],
                                 encoded_support, lengths[answer_layer_config.get('support', 'support')],
                                 mappings[answer_layer_config.get('support', 'support')],
                                 tensors.answer2support, tensors.is_eval,
                                 tensors.correct_start, beam_size=beam_size, **answer_layer_config)
                span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)
            return TensorPort.to_mapping(self.output_ports,
                                         (new_start_scores, new_end_scores, span, start_scores, end_scores))
        else:
            return TensorPort.to_mapping(self.output_ports, (start_scores, end_scores, span, start_scores, end_scores))

    def create_training_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)
        loss = xqa_crossentropy_loss(
            tensors.start_scores, tensors.end_scores, tensors.answer_span,
            tensors.answer2support, tensors.support2question,
            use_sum=shared_resources.config.get('loss', 'sum') == 'sum')

        tf.summary.scalar('actual_training_loss', loss)

        if tf.get_collection(tf.GraphKeys.LOSSES):
            loss += tf.reduce_sum(tf.get_collection(tf.GraphKeys.LOSSES))

        if shared_resources.config.get('is_interactive', False) or shared_resources.config.get('num_interactive'):
            non_interactive_loss = xqa_crossentropy_loss(
                tensors.additional_start_scores, tensors.additional_end_scores, tensors.answer_span,
                tensors.answer2support, tensors.support2question,
                use_sum=shared_resources.config.get('loss', 'sum') == 'sum')
            loss += non_interactive_loss
        return {Ports.loss: loss}


class NonInteractiveModularQAModule(NonInteractiveQAModule):
    # @property
    # def training_input_ports(self) -> Sequence[TensorPort]:
    #   return super().training_input_ports

    # @property
    # def output_ports(self) -> Sequence[TensorPort]:
    #    return super().output_ports + [question_state_start_port, question_state_end_port]

    def encoder(self, shared_resources, emb, length, tensors):
        repr_dim = shared_resources.config["repr_dim"]
        dropout = shared_resources.config.get("dropout", 0.0)
        encoded = modular_encoder(
            shared_resources.config['encoder'],
            {'text': emb}, {'text': length}, {},
            repr_dim, dropout, tensors.is_eval)[0]

        return [emb] + [encoded[k] for k in sorted(encoded.keys()) if k.startswith('output')]


class MultilevelSequenceEncoderQAModule(AbstractXQAModelModule):
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

                all_words = tf.reshape(
                    tf.concat([tf.gather(tensors.question_words, tensors.support2question),
                               tensors.support_words], 1), [-1])
                all_embeddings = tf.unsorted_segment_max(
                    tf.reshape(tf.concat([tf.gather(emb_question, tensors.support2question), emb_support], 1),
                               [-1, repr_dim]), all_words,
                    tf.reduce_max(all_words) + 1)

        mask = tf.nn.dropout(tf.ones([tf.shape(emb_question)[0], 1, repr_dim]), keep_prob=1.0 - dropout)
        emb_support, emb_question = tf.cond(tensors.is_eval,
                                            lambda: (emb_support, emb_question),
                                            lambda: (emb_support * tf.gather(mask, tensors.support2question),
                                                     emb_question * mask))

        step = tf.train.get_global_step() or tf.constant(100000, tf.int32)

        def encoding(inputs, length, input_words, reuse=False, regularize=False):
            with tf.variable_scope("encoding", reuse=reuse):
                with tf.variable_scope("controller"):
                    controller_out = modular_encoder(
                        shared_resources.config['controller']['encoder'],
                        {'text': inputs}, {'text': length}, {},
                        repr_dim, dropout, is_eval=tensors.is_eval)[0]['text']

                segm_probs, segm_logits = edge_detection_encoder(controller_out, repr_dim, tensors.is_eval, bias=-1)
                segm_probs_stop = segm_probs  # tf.stop_gradient(segm_probs)
                tf.identity(tf.sigmoid(segm_logits), name='segm_probs')
                frame_probs, frame_logits = edge_detection_encoder(
                    controller_out, repr_dim, tensors.is_eval, mask=segm_probs, bias=-1)
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
                question_hidden = tf.layers.dense(q, repr_dim, tf.nn.tanh, name="hidden")
                vs.reuse_variables()
                hidden = tf.layers.dense(s, repr_dim, tf.nn.tanh, name="hidden")
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


class HierarchicalSegmentQAModule(NonInteractiveQAModule):
    def encoder(self, shared_resources, emb, length, tensors):
        repr_dim = shared_resources.config["repr_dim"]
        dropout = shared_resources.config.get("dropout", 0.0)
        representations = list()
        segm_probs = None
        # ctrl = depthwise_separable_convolution(repr_dim, inputs, 5)
        ctrl = gated_linear_convnet(repr_dim, emb, 1, conv_width=5)
        # ctrl = convnet(repr_dim, emb, 1, conv_width=5, activation=tf.nn.tanh)
        representations.append(emb)
        representations.append(ctrl)

        step = tf.train.get_global_step() or tf.constant(10000, tf.int32)

        mask = tf.expand_dims(tf.sequence_mask(length, dtype=tf.float32), 2)
        for i in range(shared_resources.config['num_layers']):
            with tf.variable_scope("layer" + str(i)):
                prev_segm_probs = segm_probs
                segm_probs, segm_logits = edge_detection_encoder(
                    ctrl, repr_dim, tensors.is_eval)
                if prev_segm_probs is not None:
                    segm_probs = tf.maximum(prev_segm_probs, segm_probs)
                else:
                    prev_segm_probs = tf.zeros_like(segm_probs)

                segm_probs = tf.cond(step >= 1000 * i,
                                     lambda: segm_probs,
                                     lambda: tf.stop_gradient(segm_probs))

                segm_probs_cum = intra_segm_sum(segm_probs, prev_segm_probs, length)
                prev_segm_probs_cum = intra_segm_sum(prev_segm_probs, prev_segm_probs, length)
                tf.add_to_collection(tf.GraphKeys.LOSSES, tf.reduce_mean(tf.reduce_sum(tf.maximum(
                    0.0, (0.5 + prev_segm_probs_cum - segm_probs_cum) * mask), axis=[1, 2]) / tf.to_float(length)))

                tf.identity(tf.sigmoid(segm_logits), name='segm_probs' + str(i))
                segms = bow_segm_encoder(emb, length, repr_dim, segm_probs, normalize=True, activation=tf.nn.tanh)

                # segms = tf.cond(tensors.is_eval, lambda: segms, lambda: segms * get_dropout_mask(i, is_support))
                representations.append(segms)

        return representations


class HierarchicalDependencyQAModule(NonInteractiveQAModule):
    def encoder(self, shared_resources, emb, length, tensors):
        repr_dim = shared_resources.config["repr_dim"]
        dropout = shared_resources.config.get("dropout", 0.0)
        representations = list()
        segm_probs = None
        # ctrl = depthwise_separable_convolution(repr_dim, inputs, 5)
        ctrl = gated_linear_convnet(repr_dim, emb, 1, conv_width=5)
        # ctrl = convnet(repr_dim, emb, 1, conv_width=5, activation=tf.nn.tanh)
        representations.append(emb)
        representations.append(ctrl)

        step = tf.train.get_global_step() or tf.constant(10000, tf.int32)

        mask = tf.expand_dims(tf.sequence_mask(length, dtype=tf.float32), 2)
        segms = emb
        for i in range(shared_resources.config['num_layers']):
            with tf.variable_scope("layer" + str(i)):
                prev_segm_probs = segm_probs
                segm_probs, segm_logits = edge_detection_encoder(
                    ctrl, repr_dim, tensors.is_eval)
                if prev_segm_probs is not None:
                    segm_probs = tf.maximum(prev_segm_probs, segm_probs)

                # segm_probs = tf.cond(step >= 1000 * i,
                #                     lambda: segm_probs,
                #                     lambda: tf.stop_gradient(segm_probs))

                if i > 0:
                    segm_probs_cum = intra_segm_sum(segm_probs, prev_segm_probs, length)
                    prev_segm_probs_cum = intra_segm_sum(prev_segm_probs, prev_segm_probs, length)
                    tf.add_to_collection(tf.GraphKeys.LOSSES, tf.reduce_mean(tf.reduce_sum(tf.maximum(
                        0.0, (0.5 + prev_segm_probs_cum - segm_probs_cum) * mask), axis=[1, 2]) / tf.to_float(length)))

                tf.identity(tf.sigmoid(segm_logits), name='segm_probs' + str(i))
                segms, probs, logits = segment_selection_encoder(
                    length, repr_dim, segm_probs, prev_segm_probs, emb, ctrl, tensors.is_eval)
                tf.identity(probs, name='selection_probs' + str(i))

                # segms = tf.cond(tensors.is_eval, lambda: segms, lambda: segms * get_dropout_mask(i, is_support))
                representations.append(segms)

        return representations


class HierarchicalSelfAttnQAModule(NonInteractiveQAModule):
    def encoder(self, shared_resources, emb, length, tensors):
        repr_dim = shared_resources.config["repr_dim"]
        key_dim = shared_resources.config.get("key_dim", 64)
        value_dim = shared_resources.config.get("value_dim")
        num_heads = shared_resources.config['num_heads']
        dropout = shared_resources.config.get("dropout", 0.0)
        representations = list()
        segm_probs = None
        attn_probs = None
        state = encoder(emb, length, repr_dim, module='conv_glu', num_layers=1, conv_width=5, residual=True)
        # ctrl = gated_linear_convnet(repr_dim, emb, 1, conv_width=5)
        # ctrl = convnet(repr_dim, emb, 1, conv_width=5, activation=tf.nn.tanh)

        for i in range(shared_resources.config['num_layers']):
            with tf.variable_scope("self_attn", reuse=i > 0):
                scores, attn_probs, states, segm_probs, segm_logits = segment_self_attention(
                    state, state, length, tensors.is_eval, key_dim, value_dim, scaled=True, key_value_attn=True,
                    num_heads=num_heads, edge_probs=segm_probs, attn_probs=attn_probs)
                if i == 0:
                    tf.identity(tf.sigmoid(segm_logits), name='segm_probs')
                    tf.identity(attn_probs, name='selection_probs')

                s = tf.shape(states)
                new_state = tf.layers.dense(tf.reshape(states, [s[0], s[1], states.get_shape()[-1].value * num_heads]),
                                            repr_dim, tf.tanh, name='self_attn_projection')
                state += new_state

        return [state]



class HierarchicalGCNQAModule(NonInteractiveQAModule):
    def encoder(self, shared_resources, emb, length, tensors):
        repr_dim = shared_resources.config["repr_dim"]
        key_dim = shared_resources.config.get("key_dim", 64)
        value_dim = shared_resources.config.get("value_dim")
        num_heads = shared_resources.config['num_heads']
        dropout = shared_resources.config.get("dropout", 0.0)
        representations = list()
        segm_probs = None
        attn_probs = None
        state = encoder(emb, length, repr_dim, module='conv_glu', num_layers=1, conv_width=5, residual=True)
        # ctrl = gated_linear_convnet(repr_dim, emb, 1, conv_width=5)
        # ctrl = convnet(repr_dim, emb, 1, conv_width=5, activation=tf.nn.tanh)

        with tf.variable_scope('ajacency'):
            # [B, L, L, H]
            A, segm_probs, segm_logits = segment_self_attention_scores(
                state, state, length, tensors.is_eval, key_dim, num_heads=num_heads, exclude_self=True)
            # s = tf.get_variable('sentinel_score', [1, 1, 1, num_heads], tf.float32, tf.zeros_initializer())
            # s = tf.tile(s, [tf.shape(attn_scores)[0], tf.shape(attn_scores)[1], 1, 1])

            # also add backward connections
            A = tf.concat([A, tf.transpose(A, [0, 2, 1, 3])], 3)

            # only 1 edge should be active
            A = tf.nn.sigmoid(A) * tf.nn.softmax(A)

            # [B, 2H, L, L]
            A = tf.transpose(A, [0, 3, 1, 2])

            D_sqrt = tf.matrix_diag(tf.sqrt(1.0 / tf.reduce_sum(A, axis=3) + 1e-8))

            l = tf.shape(state)[1]

            #A_back = A_trans / tf.maximum(1.0, tf.reduce_sum(A_trans, axis=1, keep_dims=True))

            #A = tf.reshape(tf.transpose(A, [0,3,1,2]), [-1, l, l])

            #D_sqrt = 1.0 / tf.sqrt(tf.matrix_diag(tf.reduce_sum(A, axis=2) + 1e-8))

            A_trans = tf.matmul(tf.matmul(D_sqrt, A), D_sqrt)
            # [B, L, L, 2H]
            A_trans = tf.transpose(A_trans, [0, 2, 3, 1])

        tf.identity(tf.sigmoid(segm_logits), name='segm_probs')
        tf.identity(A, name='selection_probs')

        for i in range(shared_resources.config['num_layers']):
            with tf.variable_scope("GCN_" + str(i)):
                new_state = tf.layers.dense(state, 2 * num_heads * repr_dim, name='state_projection')
                new_state = tf.reshape(new_state, [-1, l, repr_dim, 2 * num_heads])
                new_state = tf.tanh(tf.einsum('abch,acrh->abr', A_trans, new_state))

                state += new_state

        return [state]


def _simple_answer_layer(encoded_question, encoded_support, repr_dim, shared_resources, tensors):
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
            question_hidden = tf.layers.dense(q, 2 * repr_dim, tf.nn.tanh, name="hidden")
            question_hidden_start, question_hidden_end = tf.split(question_hidden, 2, 1)
            vs.reuse_variables()
            hidden = tf.layers.dense(s, 2 * repr_dim, tf.nn.tanh, name="hidden")
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
    return start_scores, end_scores, span


def _labeling_answer_layer(encoded_question, encoded_support, repr_dim, shared_resources, tensors):
    # computing single time attention over question
    question_state = compute_question_state(encoded_question, tensors.question_length)
    question_state = tf.gather(question_state, tensors.support2question)
    question_state = tf.layers.dense(question_state, 4 * repr_dim, tf.nn.tanh, name="hidden")
    question_state = tf.reshape(question_state, [-1, repr_dim, 4])
    tf.get_variable_scope().reuse_variables()
    hidden = tf.layers.dense(encoded_support, repr_dim, tf.nn.tanh, name="hidden")
    scores = tf.einsum('ijk,ikl->ijl', hidden, question_state)

    logprobs = tf.nn.log_softmax(scores)

    b_logprobs, l_logprobs, i_logprobs, o_logprobs = tf.split(logprobs, 4, 3)
    o_logprobs *= tf.sequence_mask(tensors.support_length, dtype=tf.float32)

    # [B, L, L]
    span_logprobs = (b_logprobs + tf.transpose(l_logprobs, [0, 2, 1]) +
                     tf.cumsum(o_logprobs, axis=1, exclusive=True) +
                     tf.cumsum(tf.transpose(o_logprobs, [0, 2, 1]), axis=2, exclusive=True, reverse=True) +
                     tf.cumsum(tf.transpose(i_logprobs, [0, 2, 1]), exclusive=True) -
                     tf.cumsum(i_logprobs))

    length = tf.shape(span_logprobs)[1]
    best_span = tf.argmax(tf.reshape(span_logprobs, [tf.shape(span_logprobs)[0], -1]), output_type=tf.int32)

    best_start = best_span / length
    best_end = best_span % length

    return scores, tf.stack([tf.range(0, tf.shape(span_logprobs)[0], dtype=tf.int32), best_start, best_end], 1)
