from typing import List

import tensorflow as tf

from jack.core import TensorPort, Ports, SharedResources
from jack.core.tensorflow import TFModelModule
from jack.tf_util import segment
from jack.tf_util.rnn import birnn_with_projection
from projects.assertion_mr.multiple_choice_modules import NLIAssertionModel, nli_model
from projects.assertion_mr.shared import AssertionMRPorts
from projects.assertion_mr.tf_util import embedding_refinement, embed_span_pairs_uniq


class NLIAssertionSeekingModel(NLIAssertionModel, TFModelModule):
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
                AssertionMRPorts.word2lemma]

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits, Ports.Prediction.candidate_index,
                AssertionMRPorts.logits_3D,
                AssertionMRPorts.policy_log_probs,
                AssertionMRPorts.assertion_selection_mask,
                AssertionMRPorts.selected_assertions]

    @property
    def training_input_ports(self) -> List[TensorPort]:
        return [AssertionMRPorts.logits_3D,
                AssertionMRPorts.policy_log_probs,
                AssertionMRPorts.assertion_selection_mask,
                Ports.Target.target_index]

    def create_output(self, shared_resources, question_length, support_length,
                      word_chars, word_char_length,
                      question_words, support_words2uniq, is_eval,
                      word_embeddings, assertion_length, assertion2question, assertion_words,
                      question_arg_span, support_arg_span,
                      assertion2question_arg_span, assertion2support_arg_span, word2lemma):
        with tf.variable_scope("mc_assertion", initializer=tf.contrib.layers.xavier_initializer()):
            input_size = shared_resources.config["repr_dim_input"]
            size = shared_resources.config["repr_dim"]
            with_char_embeddings = shared_resources.config.geT("with_char_embeddings", False)
            batch_size = tf.shape(question_length)[0]
            word_embeddings.set_shape([None, input_size])

            reading_sequence = [support_words2uniq, question_words]
            reading_sequence_lengths = [support_length, question_length]
            reading_sequence_2_batch = [None, None]

            new_word_embeddings, reading_sequence_offset, _ = embedding_refinement(
                shared_resources, word_embeddings,
                reading_sequence, reading_sequence_2_batch, reading_sequence_lengths,
                word2lemma, word_chars, word_char_length, is_eval,
                with_char_embeddings=with_char_embeddings, num_chars=len(shared_resources.char_vocab))

            emb_question = tf.nn.embedding_lookup(new_word_embeddings, reading_sequence_offset[1])
            emb_support = tf.nn.embedding_lookup(new_word_embeddings, reading_sequence_offset[0])

            logits = nli_model(size, shared_resources.config["answer_size"],
                               emb_question, question_length, emb_support, support_length)

            ######################### Iterative Assertion Seeking ##########################

            ############ Embed concept spans of question and support ############
            question_arg_span = tf.gather(question_arg_span, assertion2question_arg_span)
            support_arg_span = tf.gather(support_arg_span, assertion2support_arg_span)
            # argument selection from policy net
            emb_question = tf.concat([
                tf.nn.embedding_lookup(word_embeddings, question_words), tf.stop_gradient(emb_question)], 2)
            # emb_question = tf.nn.embedding_lookup(word_embeddings, question_words2uniq)
            emb_support = tf.concat([
                tf.nn.embedding_lookup(word_embeddings, support_words2uniq), tf.stop_gradient(emb_support)], 2)
            # emb_support = tf.nn.embedding_lookup(word_embeddings, support_words2uniq)
            q_outputs = birnn_with_projection(
                size, tf.contrib.rnn.LSTMBlockFusedCell(size), emb_question, question_length,
                rnn_scope="q_rnn_selection", projection_scope="q_rnn_selection_projection")

            s_outputs = birnn_with_projection(
                size, tf.contrib.rnn.LSTMBlockFusedCell(size), emb_support, support_length,
                rnn_scope="s_rnn_selection", projection_scope="s_rnn_selection_projection")

            # [Args_q, size]
            emb_spans_q, emb_spans_s, assertion2pair, pairs2question = embed_span_pairs_uniq(
                assertion2question, question_arg_span, support_arg_span, q_outputs, s_outputs, size,
                scope="question_span_embedding")
            emb_concept_pairs = tf.layers.dense(tf.concat([emb_spans_q, emb_spans_s], 1), size, tf.nn.relu)

            stop_emb = tf.get_variable('stop_emb', [1, size], initializer=tf.zeros_initializer())
            stop_emb = tf.tile(stop_emb, [batch_size, 1])

            # emb_spans_assertion = tf.gather(emb_spans_q, assertion2span_q)
            emb_concept_pairs_with_stop = tf.concat([emb_concept_pairs, stop_emb], axis=0)
            pair_scores_with_stop = tf.squeeze(tf.layers.dense(emb_concept_pairs_with_stop, 1), 1)
            # span_scores_q_with_stop = tf.range(tf.to_float(tf.shape(span_scores_q_with_stop)[0]), 0.0, -1.0, tf.float32)

            pairs2question_with_stop = tf.concat([pairs2question, tf.range(0, batch_size, dtype=tf.int32)], 0)
            pair_mask = tf.ones_like(pairs2question, tf.float32) * -1e6
            pair_mask = tf.concat([pair_mask, tf.zeros([batch_size])], axis=0)

            log_ps = []
            all_logits = [logits]

            is_not_stop = tf.fill([batch_size], True)
            selected_assertions = None
            selected_pairs = None
            all_selected_assertions = []

            selection_mask = [is_not_stop]
            max_pairs = shared_resources.config.get("max_pairs", 4)
            for i in range(max_pairs):
                with tf.device('/cpu:0'):
                    ############ Selection of question span and corresponding assertions ############
                    pair_probs_with_stop = segment.segment_softmax(pair_scores_with_stop, pairs2question_with_stop)
                    selected_pair_with_stop = tf.cond(
                        is_eval,
                        lambda: segment.segment_is_max(pair_probs_with_stop, pairs2question_with_stop),
                        lambda: segment.segment_sample_select(pair_probs_with_stop, pairs2question_with_stop))

                    p_pair = tf.where(
                        is_not_stop,
                        tf.unsorted_segment_max(
                            tf.to_float(selected_pair_with_stop) * pair_probs_with_stop,
                            pairs2question_with_stop, batch_size),
                        tf.ones([batch_size]))
                    log_p = tf.log(p_pair + 1e-6)
                    log_ps.append(log_p)

                    is_not_stop = tf.logical_and(is_not_stop, tf.logical_not(selected_pair_with_stop[-batch_size:]))
                    selection_mask.append(is_not_stop)

                    # mask out sampled assertions to not sample again in next iteration
                    pair_scores_with_stop = tf.where(selected_pair_with_stop, pair_mask, pair_scores_with_stop)

                    # cut away stop probabilities at the end
                    new_selected_pairs = selected_pair_with_stop[:-batch_size]

                    # sampled assertions, only allow if no stop in the past
                    new_selected_assertions = tf.logical_and(
                        tf.gather(new_selected_pairs, assertion2pair), tf.gather(is_not_stop, assertion2question))

                if selected_assertions is None:
                    selected_assertions = new_selected_assertions
                    selected_pairs = new_selected_pairs
                else:
                    selected_assertions = tf.logical_or(selected_assertions, new_selected_assertions)
                    selected_pairs = tf.logical_or(selected_pairs, new_selected_pairs)

                def update_op():
                    ############ Compute predictions based on selections ############
                    # prediction, but only for questions for which an assertion was sampled
                    with tf.device('/cpu:0'):
                        new_sampled_assertions = tf.squeeze(tf.where(new_selected_assertions), axis=1)
                        sampled_questions2questions, _ = tf.unique(
                            tf.gather(assertion2question, new_sampled_assertions))
                        is_question_update = tf.sparse_to_dense(sampled_questions2questions, [batch_size], True, False)

                        # only rerun instances where new assertions were introduced
                        selected_assertions4update = tf.logical_and(
                            tf.gather(is_question_update, assertion2question), selected_assertions)
                        sampled_assertions4update = tf.squeeze(tf.where(selected_assertions4update), axis=1)
                        sampled_assertions4update.set_shape([None])
                        sampled_questions2questions, sampled_assertion2question = tf.unique(
                            tf.gather(assertion2question, sampled_assertions4update))

                    def gather_selected(t):
                        ret = tf.gather(t, sampled_assertions4update)
                        ret.set_shape([None] + [s.value for s in t.get_shape()[1:]])
                        return ret

                    sampled_assertion_length = gather_selected(assertion_length)
                    max_assertion_length = tf.reduce_max(sampled_assertion_length)
                    sampled_assertion_words2uniq = gather_selected(assertion_words)[:, :max_assertion_length]

                    sampled_embeddings = tf.gather(
                        tf.reshape(new_word_embeddings, [batch_size, -1, size]), sampled_questions2questions)
                    sampled_embeddings = tf.reshape(sampled_embeddings, [-1, size])
                    assertion_refined_embeddings, _, offsets = embedding_refinement(
                        size, sampled_embeddings,
                        [sampled_assertion_words2uniq], [sampled_assertion2question],
                        [sampled_assertion_length], word2lemma, word_chars,
                        word_char_length, is_eval, only_refine=True, sequence_indices=[2])

                    sampled_questions = tf.gather(question_words, sampled_questions2questions) + offsets
                    sampled_question_length = tf.gather(question_length, sampled_questions2questions)
                    sampled_max_question_length = tf.reduce_max(sampled_question_length)

                    sampled_supports = tf.gather(support_words2uniq, sampled_questions2questions) + offsets
                    sampled_support_length = tf.gather(support_length, sampled_questions2questions)
                    sampled_max_support_length = tf.reduce_max(sampled_support_length)

                    emb_question = tf.nn.embedding_lookup(assertion_refined_embeddings,
                                                          sampled_questions[:, :sampled_max_question_length])
                    emb_support = tf.nn.embedding_lookup(assertion_refined_embeddings,
                                                         sampled_supports[:, :sampled_max_support_length])

                    tf.get_variable_scope().reuse_variables()

                    logits = nli_model(size, shared_resources.config["answer_size"],
                                       emb_question, sampled_question_length, emb_support, sampled_support_length)

                    logits = tf.unsorted_segment_max(logits, sampled_questions2questions, batch_size)
                    # copy not updated logits through
                    logits = tf.where(is_question_update, logits, all_logits[-1])
                    return logits

                logits = tf.cond(tf.reduce_any(new_selected_assertions), update_op, lambda: all_logits[-1])
                tf.get_variable_scope().reuse_variables()
                all_logits.append(logits)
                all_selected_assertions.append(new_selected_assertions)

        if selected_pairs is None:
            selected_pairs = tf.fill(tf.shape(assertion2pair), False)

        num_pairs = tf.unsorted_segment_sum(tf.ones(tf.shape(pairs2question)), pairs2question, batch_size)
        has_pair = tf.minimum(num_pairs, 1.0)
        num_selected_pairs = tf.unsorted_segment_sum(tf.to_float(selected_pairs), pairs2question, batch_size)
        selected_pair_fraction = num_selected_pairs / tf.maximum(num_pairs, 1.0)

        used_all_pairs = tf.to_float(tf.equal(num_selected_pairs, tf.minimum(num_pairs, max_pairs)))
        fraction_is_stop = tf.reduce_sum(has_pair * (1.0 - used_all_pairs)) / tf.maximum(tf.reduce_sum(has_pair), 1)

        num_selected_pairs = tf.reduce_mean(num_selected_pairs)
        selected_pair_fraction = tf.reduce_sum(selected_pair_fraction) / tf.reduce_sum(has_pair)

        tf.summary.scalar("num_selected_pairs", num_selected_pairs)
        tf.summary.scalar("selected_pair_fraction", selected_pair_fraction)
        tf.summary.scalar("fraction_early_cut", fraction_is_stop)

        return (all_logits[-1], tf.argmax(all_logits[-1], 1), tf.stack(all_logits), tf.stack(log_ps),
                tf.stack(selection_mask), tf.stack(all_selected_assertions))

    def create_training_output(self, shared_resources: SharedResources,
                               logits_3D, policy_log_probs, selection_mask, labels):
        labels_3D = tf.tile(tf.expand_dims(labels, 0), [tf.shape(logits_3D)[0], 1])
        weights = tf.to_float(selection_mask)
        losses = tf.losses.sparse_softmax_cross_entropy(logits=logits_3D, labels=labels_3D,
                                                        reduction=tf.losses.Reduction.NONE)
        loss = tf.reduce_mean(losses[-1])
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
        if shared_resources.config.get('only_policy'):
            return policy_loss + baseline_loss,
        else:
            return loss + policy_loss + baseline_loss,

