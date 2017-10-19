import os
from typing import List

import numpy as np
import tensorflow as tf

from jack.core import TensorPort, Ports, SharedResources
from jack.core.tensorflow import TFModelModule
from jack.tf_util import misc
from jack.tf_util.rnn import birnn_with_projection
from jack.util.map import numpify
from projects.assertion_mr.assertion_generation import predicate_classification_generator
from projects.assertion_mr.multiple_choice_modules import MultipleChoiceAssertionInputModule, nli_model
from projects.assertion_mr.shared import AssertionMRPorts
from projects.assertion_mr.tf_util import embedding_refinement


class AssertionGenerationPorts:
    predicate2words = TensorPort(tf.int32, [None], 'predicate2words')
    used_assertions = TensorPort(tf.int32, [None, None], 'used_assertions')
    span_pair_scores = TensorPort(tf.int32, [None, None, None, None, None], 'span_pair_scores')
    generated_assertions = TensorPort(tf.int32, [None, None], 'generated_assertions')
    generated_assertion2question = TensorPort(tf.int32, [None], 'generated_assertion2question')


class MultipleChoiceAssertionInputModuleWithPredicates(MultipleChoiceAssertionInputModule):
    @property
    def output_ports(self):
        return super(MultipleChoiceAssertionInputModuleWithPredicates, self).output_ports + [
            AssertionGenerationPorts.predicate2words]

    def setup_from_data(self, data):
        super(MultipleChoiceAssertionInputModuleWithPredicates, self).setup_from_data(data)
        generator = self.shared_resources.config.get('generator', '')
        if generator:
            shared_resources = SharedResources()
            shared_resources.load(os.path.join(generator, "shared_resources"))
            self.shared_resources.predicate_vocab = shared_resources.predicate_vocab
            self.shared_resources.full_predicate_vocab = shared_resources.full_predicate_vocab
            self.shared_resources.max_predicate_length = shared_resources.max_predicate_length
            self.shared_resources.char_vocab = shared_resources.char_vocab
        else:
            self.shared_resources.predicate_vocab = {'not': 0, 'related': 1, 'to': 2}
            self.shared_resources.full_predicate_vocab = {'not related to': 0}
            self.shared_resources.max_predicate_length = 3

    def create_batch(self, annotations, is_eval, with_answers):
        batch = super(MultipleChoiceAssertionInputModuleWithPredicates, self).create_batch(
            annotations, is_eval, with_answers)

        vocab = batch["__vocab"]
        rev_vocab = batch["__rev_vocab"]
        lemma_vocab = batch["__lemma_vocab"]
        word_chars = batch[AssertionMRPorts.word_chars].tolist()
        word_lengths = batch[AssertionMRPorts.word_char_length].tolist()
        word2lemma = batch[AssertionMRPorts.word2lemma].tolist()
        embeddings = batch[AssertionMRPorts.word_embeddings]
        new_embeddings = []
        predicate2words = [None] * len(self.shared_resources.predicate_vocab)
        at_least_one = False
        for w, i in self.shared_resources.predicate_vocab.items():
            if w not in vocab:
                at_least_one = True
                lemma_vocab[w] = len(lemma_vocab)
                vocab[w] = len(vocab)
                word2lemma.append(lemma_vocab[w])
                rev_vocab.append(w)
                new_embeddings.append(self._get_emb(self.shared_resources.vocab(w)))
                word_lengths.append(min(len(w), 20))
                word_chars.append([self.char_vocab.get(c, 0) for c in w[:20]])
            predicate2words[i] = vocab[w]

        batch[AssertionGenerationPorts.predicate2words] = predicate2words
        if at_least_one:
            batch[AssertionMRPorts.word2lemma] = word2lemma
            batch[AssertionMRPorts.word_embeddings] = np.concatenate([embeddings, np.stack(new_embeddings)], 0)
            batch[AssertionMRPorts.word_chars] = word_chars
            batch[AssertionMRPorts.word_char_length] = word_lengths
            return numpify(batch, keys=[AssertionMRPorts.word_chars])
        else:
            return batch


class NLIAssertionSeekingAndGeneratingModel(TFModelModule):
    def setup(self, is_training=True):
        super(NLIAssertionSeekingAndGeneratingModel, self).setup(is_training)
        init_model = self.shared_resources.config.get('generator')
        if is_training and init_model is not None:
            if not init_model.endswith('model_module'):
                # path to a reader was provided
                init_model = os.path.join(init_model, 'model_module')
            from tensorflow.python import pywrap_tensorflow
            reader = pywrap_tensorflow.NewCheckpointReader(init_model)
            init_vars = dict()
            for n in reader.get_variable_to_shape_map().keys():
                found = False
                for v in self.variables:
                    if v.op.name[len(self.shared_resources.config['name']):] == n[n.index('/'):]:
                        found = True
                        init_vars[n] = v
                        break
                if not found:
                    print("WARNING: Could not find variable", n, " in computation graph.")
            saver = tf.train.Saver(init_vars)
            saver.restore(self.tf_session, init_model)

    @property
    def input_ports(self) -> List[TensorPort]:
        return [AssertionMRPorts.question_length, AssertionMRPorts.support_length,
                # char embedding inputs
                AssertionMRPorts.word_chars, AssertionMRPorts.word_char_length,
                AssertionMRPorts.question, AssertionMRPorts.support,
                # optional input, provided only during training
                AssertionMRPorts.is_eval,
                # assertion related ports
                AssertionMRPorts.word_embeddings, AssertionMRPorts.assertion_lengths,
                AssertionMRPorts.assertion2question, AssertionMRPorts.assertions,
                AssertionMRPorts.question_arg_span, AssertionMRPorts.support_arg_span,
                AssertionMRPorts.assertion2question_arg_span, AssertionMRPorts.assertion2support_arg_span,
                AssertionMRPorts.word2lemma, AssertionGenerationPorts.predicate2words]

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits, Ports.Prediction.candidate_index,
                AssertionMRPorts.logits_3D,
                AssertionMRPorts.policy_log_probs,
                AssertionMRPorts.assertion_selection_mask,
                AssertionMRPorts.selected_assertions,
                AssertionGenerationPorts.span_pair_scores,
                AssertionGenerationPorts.generated_assertions,
                AssertionGenerationPorts.generated_assertion2question]

    def create_output(self, shared_resources, question_length, support_length,
                      word2chars, word_char_length,
                      question, support, is_eval,
                      word_embeddings, assertion_length, assertion2question, assertions,
                      question_arg_span, support_arg_span,
                      assertion2question_arg_span, assertion2support_arg_span, word2lemma,
                      predicate2words):
        input_size = shared_resources.config["repr_dim_input"]
        size = shared_resources.config["repr_dim"]
        with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)
        batch_size = tf.shape(question_length)[0]
        word_embeddings.set_shape([None, input_size])

        reading_sequence = [support, question]
        reading_sequence_lengths = [support_length, question_length]
        reading_sequence_2_batch = [None, None]

        new_word_embeddings, reading_sequence_offset, _ = embedding_refinement(
            size, word_embeddings,
            reading_sequence, reading_sequence_2_batch, reading_sequence_lengths,
            word2lemma, word2chars, word_char_length, is_eval,
            keep_prob=1.0 - shared_resources.config.get('dropout', 0.0),
            with_char_embeddings=with_char_embeddings, num_chars=len(shared_resources.char_vocab))

        emb_question = tf.nn.embedding_lookup(new_word_embeddings, reading_sequence_offset[1])
        emb_support = tf.nn.embedding_lookup(new_word_embeddings, reading_sequence_offset[0])

        logits = nli_model(size, shared_resources.config["answer_size"],
                           emb_question, question_length, emb_support, support_length)

        ######################### Iterative Assertion Seeking ##########################

        ############ Embed concept spans of question and support ############
        # argument selection from policy net
        emb_question = tf.concat([
            tf.nn.embedding_lookup(word_embeddings, question), tf.stop_gradient(emb_question)], 2)
        # emb_question = tf.nn.embedding_lookup(word_embeddings, question_words2uniq)
        emb_support = tf.concat([
            tf.nn.embedding_lookup(word_embeddings, support), tf.stop_gradient(emb_support)], 2)
        # emb_support = tf.nn.embedding_lookup(word_embeddings, support_words2uniq)
        q_outputs = birnn_with_projection(
            size, tf.contrib.rnn.LSTMBlockFusedCell(size), emb_question, question_length,
            rnn_scope="q_rnn_selection", projection_scope="q_rnn_selection_projection")

        s_outputs = birnn_with_projection(
            size, tf.contrib.rnn.LSTMBlockFusedCell(size), emb_support, support_length,
            rnn_scope="s_rnn_selection", projection_scope="s_rnn_selection_projection")

        max_span = shared_resources.config.get('max_span', 5)

        def all_spans(encoded):
            encoded = tf.layers.dense(encoded, 2 * size)
            encoded1, encoded2 = tf.split(encoded, 2, -1)
            spans = []
            for m in range(max_span):
                if m == 0:
                    emb_span = tf.nn.relu(encoded1 + encoded2)
                else:
                    emb_span = tf.nn.relu(encoded1[:, :-m, :] + encoded2[:, m:, :])
                    emb_span = tf.pad(emb_span, [[0, 0], [0, m], [0, 0]])
                spans.append(emb_span)
            return tf.stack(spans, axis=1)

        # [B, M, L, size]
        emb_spans_q = all_spans(q_outputs)
        emb_spans_s = all_spans(s_outputs)

        # [B, M * L, size]
        emb_spans_q_flat = tf.reshape(emb_spans_q, [batch_size, -1, size])
        emb_spans_s_flat = tf.reshape(emb_spans_s, [batch_size, -1, size])

        # [B, M * Lq, M * Ls]
        max_question_length = tf.reduce_max(question_length)
        max_support_length = tf.reduce_max(support_length)
        # bi-linear scoring
        scored_span_pairs = tf.einsum(
            "aik,ajk->aij", tf.layers.dense(emb_spans_q_flat, size), emb_spans_s_flat)

        scored_span_pairs = tf.reshape(scored_span_pairs, [batch_size, max_span, max_question_length,
                                                           max_span, max_support_length])
        # [B, M, Lq, 1, 1]
        q_mask = [misc.mask_for_lengths(question_length - m, max_question_length) for m in range(max_span)]
        q_mask = tf.stack(q_mask, axis=1)[:, :, :, tf.newaxis, tf.newaxis]

        s_mask = [misc.mask_for_lengths(support_length - m, max_support_length) for m in range(max_span)]
        s_mask = tf.stack(s_mask, axis=1)[:, tf.newaxis, tf.newaxis, :, :]

        # [B, M, Lq, M, Ls]
        scored_span_pairs += q_mask
        scored_span_pairs += s_mask

        scored_span_pairs_flat = tf.reshape(scored_span_pairs, [batch_size, -1])

        # [B, M * Lq * M * Ls + 1]
        pair_scores_with_stop = tf.concat([scored_span_pairs_flat, tf.zeros([batch_size, 1])], 1)

        with tf.device('/cpu:0'):
            offset1 = max_span * max_support_length
            offset2 = max_span * max_question_length * offset1
            pair_mask = tf.fill([batch_size, offset2], -1e6)
            pair_mask = tf.concat([pair_mask, tf.zeros([batch_size, 1])], axis=1)

            a_q_span = tf.gather(question_arg_span, assertion2question_arg_span)
            a_s_span = tf.gather(support_arg_span, assertion2support_arg_span)

            aq_idx = (a_q_span[:, 1] - a_q_span[:, 0] - 1) * max_question_length + a_q_span[:, 0]
            as_idx = (a_s_span[:, 1] - a_s_span[:, 0] - 1) * max_support_length + a_s_span[:, 0]

            assertion2pair = tf.stack([assertion2question, aq_idx * offset1 + as_idx], 1)
            assertion2pair_flat = assertion2question * offset2 + aq_idx * offset1 + as_idx
            uniq_assertion2pair_flat = tf.unique(assertion2pair_flat)[0]
            uniq_assertion2pair = tf.stack([tf.div(uniq_assertion2pair_flat, offset2),
                                            tf.mod(uniq_assertion2pair_flat, offset2)], 1)
            assertion_pair_mask = tf.SparseTensor(
                tf.to_int64(uniq_assertion2pair), tf.fill(tf.shape(uniq_assertion2pair_flat), False),
                tf.to_int64([batch_size, offset2]))
            assertion_pair_mask = tf.sparse_reorder(assertion_pair_mask)
            assertion_pair_mask = tf.sparse_tensor_to_dense(assertion_pair_mask, default_value=True)

        log_ps = []

        is_not_stop = tf.fill([batch_size], True)
        selected_assertions = None
        selected_pairs = None
        all_selected_assertions = []

        selection_mask = [is_not_stop]
        max_pairs = shared_resources.config.get("max_pairs", 4)

        batch_range = tf.range(0, batch_size, dtype=tf.int32)
        assertion2question_plus = assertion2question
        num_assertions = tf.shape(assertion2question)[0]

        # collect assertions and corresponding questions for each iteration
        collected_assertions_ids = []

        for i in range(max_pairs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            with tf.device('/cpu:0'):
                ############ Selection span pairs and corresponding assertions ############
                dist = tf.distributions.Categorical(logits=pair_scores_with_stop)
                sampled_pair_with_stop = tf.cond(
                    is_eval,
                    lambda: tf.argmax(pair_scores_with_stop, axis=1, output_type=tf.int32),
                    lambda: dist.sample())

                log_p = tf.where(
                    is_not_stop, dist.log_prob(sampled_pair_with_stop), tf.zeros([batch_size]))
                log_ps.append(log_p)

                is_not_stop = tf.logical_and(is_not_stop, tf.not_equal(sampled_pair_with_stop, offset2))
                selection_mask.append(is_not_stop)

                selected_pair_with_stop = tf.sparse_to_dense(
                    tf.to_int64(tf.stack([batch_range, sampled_pair_with_stop], 1)),
                    tf.shape(pair_scores_with_stop, out_type=tf.int64), tf.fill([batch_size], True),
                    default_value=False)
                # mask out sampled assertions to not sample again in next iteration
                pair_scores_with_stop = tf.where(selected_pair_with_stop, pair_mask, pair_scores_with_stop)

                # cut away stop probabilities at the end
                new_selected_pairs = selected_pair_with_stop[:, :-1]

                # sampled assertions, only allow if no stop in the past
                new_selected_assertions = tf.logical_and(
                    tf.gather_nd(selected_pair_with_stop, assertion2pair),
                    tf.gather(is_not_stop, assertion2question))

                ###### pairs without assertion from KB ######
                selected_pairs_no_assertion = tf.logical_and(new_selected_pairs, assertion_pair_mask)

            def generate_assertions():
                sampled_pairs_no_assertion = tf.where(selected_pairs_no_assertion)
                predicted_assertions2question = tf.to_int32(sampled_pairs_no_assertion[:, 0])
                question4gen = tf.gather(question, predicted_assertions2question)
                support4gen = tf.gather(support, predicted_assertions2question)
                pair_idx = tf.to_int32(sampled_pairs_no_assertion[:, 1])
                pair_idx, s_arg_start = tf.div(pair_idx, max_support_length), tf.mod(pair_idx, max_support_length)
                pair_idx, s_arg_length = tf.div(pair_idx, max_span), tf.mod(pair_idx, max_span)
                q_arg_length, q_arg_start = tf.div(pair_idx, max_question_length), tf.mod(pair_idx, max_question_length)

                if shared_resources.config.get('generator') is not None:
                    gen_assertions, gen_assertion_length, gen_log_p = _generate_assertion(
                        size, shared_resources.config.get('with_char_embeddings', False),
                        shared_resources.config.get('max_conv_width', 9), len(shared_resources.char_vocab),
                        shared_resources.full_predicate_vocab, shared_resources.predicate_vocab,
                        question4gen, q_arg_start, q_arg_length + 1,
                        support4gen, s_arg_start, s_arg_length + 1,
                        word_embeddings, word2chars, word_char_length, predicate2words, is_eval)
                    gen_log_p = tf.unsorted_segment_sum(gen_log_p, predicted_assertions2question, batch_size)
                else:
                    args1, args2 = _extract_args(question4gen, q_arg_start, q_arg_length + 1,
                                                 support4gen, s_arg_start, s_arg_length + 1)
                    gen_pred = tf.tile(tf.constant([[0, 1, 2]], tf.int32), [tf.shape(question4gen)[0], 1])
                    gen_pred_length = tf.tile(tf.constant([3], tf.int32), [tf.shape(question4gen)[0]])

                    gen_assertions, gen_assertion_length = _construct_assertions(
                        args1, args2, q_arg_length + 1, s_arg_length + 1, gen_pred, gen_pred_length, predicate2words)
                    gen_log_p = tf.zeros([batch_size])

                return gen_assertions, gen_assertion_length, gen_log_p, predicted_assertions2question

            gen_assertions, gen_assertion_length, log_p, predicted_assertions2question = tf.cond(
                tf.reduce_any(selected_pairs_no_assertion),
                generate_assertions,
                lambda: (tf.zeros([0, 0], tf.int32), tf.zeros([0], tf.int32), tf.zeros([batch_size]),
                         tf.zeros([0], tf.int32)))

            # log_ps[-1] += gen_log_p
            with tf.device('/cpu:0'):
                assertion2question_plus = tf.concat(
                    [assertion2question_plus, predicted_assertions2question], axis=0)
                assertion_length = tf.concat([assertion_length, gen_assertion_length], axis=0)
                max_assertion_length = tf.reduce_max(assertion_length)

                def pad(s):
                    s_length = tf.shape(s)[1]
                    return tf.cond(s_length < max_assertion_length,
                                   lambda: tf.concat(
                                       [s, tf.zeros([tf.shape(s)[0], max_assertion_length - s_length], dtype=tf.int32)],
                                       1),
                                   lambda: s)

                assertions = tf.concat([pad(assertions), pad(gen_assertions)], axis=0)

                new_selected_assertions = tf.concat([
                    new_selected_assertions,
                    tf.fill([tf.shape(assertion_length)[0] - tf.shape(new_selected_assertions)[0]], True)], axis=0)

                if selected_assertions is None:
                    selected_assertions = new_selected_assertions
                    selected_pairs = new_selected_pairs
                else:
                    selected_assertions = tf.concat([
                        selected_assertions, tf.fill([tf.shape(gen_assertions)[0]], True)], axis=0)
                    selected_assertions = tf.logical_or(selected_assertions, new_selected_assertions)
                    selected_pairs = tf.logical_or(selected_pairs, new_selected_pairs)

                ############ Compute predictions based on selections ############
                # prediction, but only for questions for which an assertion was sampled
                is_question_update = tf.reduce_any(new_selected_pairs, axis=1)
                # only rerun instances where new assertions were introduced
                selected_assertions4update = tf.logical_and(
                    tf.gather(is_question_update, assertion2question_plus), selected_assertions)
                sampled_assertions4update = tf.squeeze(tf.where(selected_assertions4update), axis=1)
                sampled_assertions4update.set_shape([None])
                collected_assertions_ids.append(sampled_assertions4update)

            all_selected_assertions.append(new_selected_assertions[:num_assertions])

        tf.get_variable_scope().reuse_variables()

        final_logits, all_logits = tf.cond(
            is_eval,
            lambda: self._compute_logits(
                shared_resources, word2lemma, new_word_embeddings, question, question_length,
                support, support_length, [tf.squeeze(tf.where(selected_assertions), axis=1)],
                assertions, assertion_length, assertion2question_plus, logits),
            lambda: self._compute_logits(
                shared_resources, word2lemma, new_word_embeddings, question, question_length,
                support, support_length, collected_assertions_ids,
                assertions, assertion_length, assertion2question_plus, logits)
        )

        if selected_pairs is None:
            selected_pairs = tf.fill(tf.shape(assertion2pair), False)

        selected_pairs_assertion = tf.logical_and(selected_pairs, tf.logical_not(assertion_pair_mask))
        num_kb_pairs = tf.reduce_sum(tf.to_float(selected_pairs_assertion), axis=1)
        num_pairs = tf.reduce_sum(tf.to_float(selected_pairs), axis=1)
        num_generated_pairs = num_pairs - num_kb_pairs
        tf.summary.scalar("num_kb_pairs", tf.reduce_mean(num_kb_pairs))
        tf.summary.scalar("num_generated_pairs", tf.reduce_mean(num_generated_pairs))
        tf.summary.scalar("num_pairs", tf.reduce_mean(num_pairs))
        tf.summary.scalar("fraction_possible_kb_pairs",
                          (tf.reduce_sum(num_kb_pairs) + 1e-6) /
                          tf.reduce_sum(tf.minimum(tf.reduce_sum(tf.to_float(tf.logical_not(assertion_pair_mask)), 1),
                                                   max_pairs)) + 1e-6)

        num_generated_assertions = tf.to_int32(tf.reduce_sum(num_generated_pairs))
        generated_assertions = assertions[-num_generated_assertions:, :]
        generated_assertion2question = assertion2question_plus[-num_generated_assertions:]

        return (final_logits, tf.argmax(final_logits, 1), all_logits, tf.stack(log_ps),
                tf.stack(selection_mask), tf.stack(all_selected_assertions), scored_span_pairs,
                generated_assertions, generated_assertion2question)

    def _compute_logits(self, shared_resources, word2lemma, word_embeddings,
                        questions, question_lengths, supports, support_lengths,
                        assertion_ids, assertions, assertion_lengths, assertion2question, first_logits):
        size = shared_resources.config['repr_dim']
        batch_size = tf.shape(question_lengths)[0]

        question_ids = []
        offset = None
        collected_ass, collected_ass_lengths, collected_assertion2question = [], [], []
        for i, a_id in enumerate(assertion_ids):
            question_id, sampled_assertion2question = tf.unique(tf.gather(assertion2question, a_id))
            question_ids.append(question_id)
            collected_ass.append(tf.gather(assertions, a_id))
            collected_ass_lengths.append(tf.gather(assertion_lengths, a_id))
            if i > 0:
                collected_assertion2question.append(sampled_assertion2question + offset)
                offset += tf.shape(question_id)[0]
            else:
                collected_assertion2question.append(sampled_assertion2question)
                offset = tf.shape(question_id)[0]

        question_ids_cat = tf.concat(question_ids, 0)

        gathered_embeddings = tf.gather(
            tf.reshape(word_embeddings, [batch_size, -1, size]), question_ids_cat)
        gathered_embeddings = tf.reshape(gathered_embeddings, [-1, size])

        assertion_refined_embeddings, _, offsets = embedding_refinement(
            size, gathered_embeddings,
            [tf.concat(collected_ass, 0)], [tf.concat(collected_assertion2question, 0)],
            [tf.concat(collected_ass_lengths, 0)], word2lemma, None,
            None, None, only_refine=True, sequence_indices=[2], batch_size=tf.shape(question_ids_cat)[0])

        question = tf.gather(questions, question_ids_cat)
        question_length = tf.gather(question_lengths, question_ids_cat)
        support = tf.gather(supports, question_ids_cat)
        support_length = tf.gather(support_lengths, question_ids_cat)

        emb_question = tf.nn.embedding_lookup(assertion_refined_embeddings, question + offsets)
        emb_support = tf.nn.embedding_lookup(assertion_refined_embeddings, support + offsets)

        logits = nli_model(size, shared_resources.config["answer_size"],
                           emb_question, question_length, emb_support, support_length)

        # logits = tf.unsorted_segment_sum(
        #    logits, tf.concat([qid + i * batch_size for i, qid in enumerate(question_ids)], 0),
        #    len(assertion_ids) * batch_size)

        # logits = tf.reshape(logits, [len(assertion_ids), batch_size, 3])
        #logits = [first_logits] + [logits[i] for i in range(len(assertion_ids))]

        split_logits = [first_logits]
        offset = 0
        for i, q_id in enumerate(question_ids):
            this_logits = tf.slice(logits, [offset, 0], [tf.shape(q_id)[0], -1])
            this_logits = tf.unsorted_segment_sum(this_logits, q_id, batch_size)
            split_logits.append(tf.where(tf.equal(this_logits, 0.0), split_logits[i], this_logits))
            offset += tf.shape(q_id)[0]

        return split_logits[-1], tf.stack(split_logits)

    @property
    def training_input_ports(self) -> List[TensorPort]:
        return self.input_ports + [
            Ports.Prediction.logits,
            AssertionGenerationPorts.span_pair_scores,
            AssertionMRPorts.logits_3D,
            AssertionMRPorts.policy_log_probs,
            AssertionMRPorts.assertion_selection_mask,
            Ports.Target.target_index]

    @property
    def training_output_ports(self) -> List[TensorPort]:
        return [Ports.loss]

    def create_training_output(self, shared_resources, question_length, support_length, word2chars, word_char_length,
                               question, support, is_eval, word_embeddings,
                               assertion_length, assertion2question, assertions, question_arg_span, support_arg_span,
                               assertion2question_arg_span, assertion2support_arg_span, unique_word2unique_lemma,
                               predicate2words,
                               logits, span_pair_scores, logits_3D, policy_log_probs, selection_mask, labels):
        # Supervise pair selection policy
        max_span = shared_resources.config.get('max_span', 5)
        pair_selection_loss = self.pair_selection_loss(assertion2question, assertion2question_arg_span,
                                                       assertion2support_arg_span, span_pair_scores, max_span, question,
                                                       question_arg_span, support, support_arg_span)

        # Train actual model
        labels_3D = tf.tile(tf.expand_dims(labels, 0), [tf.shape(logits_3D)[0], 1])
        weights = tf.to_float(selection_mask)
        losses = tf.losses.sparse_softmax_cross_entropy(logits=logits_3D, labels=labels_3D,
                                                        reduction=tf.losses.Reduction.NONE)
        loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
        neg_gain = tf.stop_gradient(losses[1:] - losses[:-1])

        baseline = tf.get_variable("baseline_gain", [], tf.float32, tf.zeros_initializer())
        baseline_loss = tf.losses.mean_squared_error(
            neg_gain, tf.tile(tf.reshape(baseline, [1, 1]), tf.shape(neg_gain)), weights=weights[:-1])

        policy_loss = tf.reduce_mean(tf.reduce_sum(
            policy_log_probs * tf.stop_gradient(neg_gain - baseline) * weights[:-1], axis=0))

        weight_sum = tf.maximum(1.0, tf.reduce_sum(weights[1:]))
        tf.summary.scalar("gain_per_concept_pair", -tf.reduce_sum(neg_gain) / weight_sum)
        tf.summary.scalar("gain_for_first_concept_pair",
                          -tf.reduce_sum(neg_gain[0]) / tf.maximum(1.0, tf.reduce_sum(weights[1])))
        tf.summary.scalar("cumulative_gain", -tf.reduce_mean(tf.reduce_sum(neg_gain, axis=0)))
        tf.summary.scalar("baseline_gain", -baseline)
        tf.summary.scalar("policy_prob",
                          tf.reduce_sum(tf.exp(policy_log_probs) * weights[:-1]) /
                          tf.maximum(1.0, tf.reduce_sum(weights[:-1])))
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("pair_selection_loss", pair_selection_loss)

        if shared_resources.config.get('only_policy'):
            return policy_loss + baseline_loss + pair_selection_loss,
        else:
            return loss + policy_loss + baseline_loss + pair_selection_loss,

    def pair_selection_loss(self, assertion2question, assertion2question_arg_span, assertion2support_arg_span,
                            span_pair_scores, max_span, question, a_q_span, support, a_s_span):
        span_pair_scores = tf.reshape(span_pair_scores, [tf.shape(question)[0], -1])
        max_question_length = tf.shape(question)[1]
        max_support_length = tf.shape(support)[1]
        offset1 = max_span * max_support_length
        offset2 = max_span * max_question_length * offset1
        a_q_span = tf.gather(a_q_span, assertion2question_arg_span)
        a_s_span = tf.gather(a_s_span, assertion2support_arg_span)
        aq_idx = (a_q_span[:, 1] - a_q_span[:, 0] - 1) * max_question_length + a_q_span[:, 0]
        as_idx = (a_s_span[:, 1] - a_s_span[:, 0] - 1) * max_support_length + a_s_span[:, 0]
        assertion2pair_flat = assertion2question * offset2 + aq_idx * offset1 + as_idx
        uniq_assertion2pair_flat = tf.unique(assertion2pair_flat)[0]
        uniq_assertion2pair = tf.stack([tf.div(uniq_assertion2pair_flat, offset2),
                                        tf.mod(uniq_assertion2pair_flat, offset2)], 1)
        assertion_pair_mask = tf.SparseTensor(
            tf.to_int64(uniq_assertion2pair), tf.fill(tf.shape(uniq_assertion2pair_flat), -1e6),
            tf.to_int64([tf.shape(question)[0], offset2]))
        assertion_pair_mask = tf.sparse_reorder(assertion_pair_mask)
        assertion_pair_mask = tf.sparse_tensor_to_dense(assertion_pair_mask, default_value=0.0)

        assertion_pair_scores = tf.gather_nd(
            span_pair_scores, tf.stack([assertion2question, aq_idx * offset1 + as_idx], 1))
        span_pair_scores = tf.gather(span_pair_scores, assertion2question)
        assertion_pair_mask = tf.gather(assertion_pair_mask, assertion2question)
        span_pair_scores += assertion_pair_mask
        span_pair_scores = tf.concat([tf.expand_dims(assertion_pair_scores, 1), span_pair_scores], 1)

        loss = tf.losses.sparse_softmax_cross_entropy(tf.zeros_like(assertion2question, tf.int32), span_pair_scores)

        return loss


def _extract_args(question, q_arg_start, q_arg_length, support, s_arg_start, s_arg_length):
    max_length = tf.maximum(tf.reduce_max(s_arg_length), tf.reduce_max(q_arg_length))

    def map_fn(x):
        sequence, start, length = x
        sliced = tf.slice(sequence, start, length)
        return tf.cond(tf.greater(max_length, length[0]),
                       lambda: tf.concat([sliced, tf.zeros(max_length - length, dtype=tf.int32)], 0),
                       lambda: sliced)

    args1 = tf.map_fn(map_fn, [question, tf.expand_dims(q_arg_start, 1), tf.expand_dims(q_arg_length, 1)],
                      dtype=tf.int32)
    args2 = tf.map_fn(map_fn, [support, tf.expand_dims(s_arg_start, 1), tf.expand_dims(s_arg_length, 1)],
                      dtype=tf.int32)

    return args1, args2


def _generate_assertion(size, with_char_embeddings, max_conv_width, num_chars,
                        full_predicate_vocab, predicate_vocab, question, q_arg_start, q_arg_length,
                        support, s_arg_start, s_arg_length,
                        word_embeddings, word2chars, char_lengths, predicate2words, is_eval):
    num_predicate_tokens = len(predicate_vocab)
    args1, args2 = _extract_args(question, q_arg_start, q_arg_length, support, s_arg_start, s_arg_length)

    emb_arg1 = tf.nn.embedding_lookup(word_embeddings, args1)
    emb_arg2 = tf.nn.embedding_lookup(word_embeddings, args2)

    chars1 = tf.gather(word2chars, args1)
    chars2 = tf.gather(word2chars, args2)
    char_lengths1 = tf.gather(char_lengths, args1)
    char_lengths2 = tf.gather(char_lengths, args2)

    # logits, predicted_predicates, pred_lengths, rev_ordering_logits = predicate_rnn_generator(
    #    size, with_char_embeddings, max_conv_width, num_chars, num_predicate_tokens, max_predicate_length,
    #    None, None, emb_arg1, q_arg_length, chars1, char_lengths1,
    #    emb_arg2, s_arg_length, chars2, char_lengths2, tf.constant(True))  # is_eval)

    logits, predicate_idx, predicted_predicates, pred_lengths, rev_ordering_logits = predicate_classification_generator(
        size, with_char_embeddings, full_predicate_vocab, predicate_vocab,
        chars1, chars2, char_lengths1, char_lengths2, emb_arg1, emb_arg2, q_arg_length, s_arg_length,
        is_eval, max_conv_width, num_chars)

    # weights = tf.sequence_mask(pred_lengths + 1, dtype=tf.float32)
    # log_symbol_probs = -tf.losses.sparse_softmax_cross_entropy(
    #    predicted_predicates, logits, weights, reduction=tf.losses.Reduction.NONE)

    rev_order_dist = tf.distributions.Bernoulli(logits=rev_ordering_logits)
    rev_order = tf.cond(is_eval, lambda: tf.to_int32(tf.round(rev_order_dist.probs)), rev_order_dist.sample)

    # log_p = rev_order_dist.log_prob(rev_order) + tf.reduce_sum(log_symbol_probs, axis=1)
    log_p = rev_order_dist.log_prob(rev_order) + -tf.losses.sparse_softmax_cross_entropy(predicate_idx, logits)

    arg1 = tf.where(tf.equal(rev_order, 0), args1, args2)
    arg_length1 = tf.where(tf.equal(rev_order, 0), q_arg_length, s_arg_length)
    arg2 = tf.where(tf.equal(rev_order, 1), args1, args2)
    arg_length2 = tf.where(tf.equal(rev_order, 1), q_arg_length, s_arg_length)

    predicted_predicates = tf.minimum(predicted_predicates, num_predicate_tokens - 1)
    assertions, assertion_length = _construct_assertions(
        arg1, arg2, arg_length1, arg_length2, predicted_predicates, pred_lengths, predicate2words)

    return assertions, assertion_length, log_p


def _construct_assertions(arg1, arg2, arg_length1, arg_length2, predicates, pred_lengths, predicate2words):
    predicates = tf.gather(predicate2words, predicates)
    assertions = _merge_padded_sequences(predicates, pred_lengths, arg2)
    assertions = _merge_padded_sequences(arg1, arg_length1, assertions)
    assertion_length = arg_length1 + arg_length2 + pred_lengths
    max_assertion_length = tf.reduce_max(assertion_length)
    return assertions[:, :max_assertion_length], assertion_length


def _merge_padded_sequences(seq1, len1, seq2, time_axis=1):
    r2 = tf.reverse(seq2, [time_axis])
    r1 = tf.reverse_sequence(seq1, len1, time_axis)
    r_merged = tf.concat([r2, r1], time_axis)
    return tf.reverse_sequence(r_merged, len1 + tf.shape(seq2)[time_axis], time_axis)
