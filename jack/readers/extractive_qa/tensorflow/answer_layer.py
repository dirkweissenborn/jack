import tensorflow as tf

from jack.tfutil import misc
from jack.tfutil import sequence_encoder
from jack.tfutil.segment import segment_top_k


def answer_layer(encoded_question, question_length, encoded_support, support_length,
                 support2question, answer2support, is_eval, correct_start=None, beam_size=1, max_span_size=10000,
                 encoder=None, module='bilinear', repr_dim=100, **kwargs):
    if module == 'bilinear':
        return bilinear_answer_layer(
            repr_dim, encoded_question, question_length, encoded_support, support_length,
            support2question, answer2support, is_eval, beam_size, max_span_size)
    elif module == 'mlp':
        return mlp_answer_layer(repr_dim, encoded_question, question_length, encoded_support, support_length,
                                support2question, answer2support, is_eval, beam_size, max_span_size)
    elif module == 'conditional':
        return conditional_answer_layer(
            repr_dim, encoded_question, question_length, encoded_support, support_length,
            correct_start, support2question, answer2support, is_eval, beam_size, max_span_size)
    elif module == 'conditional_bilinear':
        return conditional_answer_layer(
            repr_dim, encoded_question, question_length, encoded_support, support_length,
            correct_start, support2question, answer2support, is_eval, beam_size, max_span_size, bilinear=True)
    elif module == 'conditional_bilinear':
        return conditional_answer_layer(
            repr_dim, encoded_question, question_length, encoded_support, support_length,
            correct_start, support2question, answer2support, is_eval, beam_size, max_span_size, bilinear=True)
    elif module == 'bidaf':
        if 'repr_dim' not in encoder:
            encoder['repr_dim'] = repr_dim
        encoded_support_end = sequence_encoder.encoder(
            encoded_support, support_length, name='encoded_support_end', **encoder)
        encoded_support_end = tf.concat([encoded_support, encoded_support_end], 2)
        return bidaf_answer_layer(encoded_support, encoded_support_end, support_length,
                                  support2question, answer2support, is_eval, beam_size=1, max_span_size=10000)
    else:
        raise ValueError("Unknown answer layer type: %s" % module)


def compute_question_weights(encoded_question, question_length):
    attention_scores = tf.layers.dense(encoded_question, 1, name="question_attention")
    q_mask = misc.mask_for_lengths(question_length)
    attention_scores = attention_scores + tf.expand_dims(q_mask, 2)
    question_attention_weights = tf.nn.softmax(attention_scores, 1, name="question_attention_weights")
    return question_attention_weights


def compute_question_state(encoded_question, question_length):
    attention_scores = tf.layers.dense(encoded_question, 1, name="question_attention")
    q_mask = misc.mask_for_lengths(question_length)
    attention_scores = attention_scores + tf.expand_dims(q_mask, 2)
    question_attention_weights = tf.nn.softmax(attention_scores, 1, name="question_attention_weights")
    question_state = tf.reduce_sum(question_attention_weights * encoded_question, 1)
    return question_state


def bilinear_answer_layer(size, encoded_question, question_length, encoded_support, support_length,
                          support2question, answer2support, is_eval, beam_size=1,
                          max_span_size=10000):
    """Answer layer for multiple paragraph QA."""
    # computing single time attention over question
    size = encoded_support.get_shape()[-1].value
    question_state = compute_question_state(encoded_question, question_length)

    # compute logits
    hidden = tf.gather(tf.layers.dense(question_state, 2 * size, name="hidden"), support2question)
    hidden_start, hidden_end = tf.split(hidden, 2, 1)

    support_mask = misc.mask_for_lengths(support_length)

    start_scores = tf.einsum('ik,ijk->ij', hidden_start, encoded_support)
    start_scores = start_scores + support_mask

    end_scores = tf.einsum('ik,ijk->ij', hidden_end, encoded_support)
    end_scores = end_scores + support_mask

    return compute_spans(start_scores, end_scores, answer2support, is_eval, support2question,
                         beam_size, max_span_size)


def mlp_answer_layer(size, encoded_question, question_length, encoded_support, support_length,
                     support2question, answer2support, is_eval, beam_size=1, max_span_size=10000):
    """Answer layer for multiple paragraph QA."""
    # computing single time attention over question
    question_state = compute_question_state(encoded_question, question_length)

    # compute logits
    static_input = tf.concat([tf.gather(tf.expand_dims(question_state, 1), support2question) * encoded_support,
                              encoded_support], 2)

    hidden = tf.gather(tf.layers.dense(question_state, 2 * size, name="hidden_1"), support2question)
    hidden = tf.layers.dense(
        static_input, 2 * size, use_bias=False, name="hidden_2") + tf.expand_dims(hidden, 1)

    hidden_start, hidden_end = tf.split(tf.nn.relu(hidden), 2, 2)

    support_mask = misc.mask_for_lengths(support_length)

    start_scores = tf.layers.dense(hidden_start, 1, use_bias=False, name="start_scores")
    start_scores = tf.squeeze(start_scores, [2])
    start_scores = start_scores + support_mask

    end_scores = tf.layers.dense(hidden_end, 1, use_bias=False, name="end_scores")
    end_scores = tf.squeeze(end_scores, [2])
    end_scores = end_scores + support_mask

    return compute_spans(start_scores, end_scores, answer2support, is_eval, support2question,
                         beam_size, max_span_size)


def compute_spans(start_scores, end_scores, answer2support, is_eval, support2question,
                  beam_size=1, max_span_size=10000, correct_start=None):
    max_support_length = tf.shape(start_scores)[1]
    _, _, num_doc_per_question = tf.unique_with_counts(support2question)
    offsets = tf.cumsum(num_doc_per_question, exclusive=True)
    doc_idx_for_support = tf.range(tf.shape(support2question)[0]) - tf.gather(offsets, support2question)

    def train():
        gathered_end_scores = tf.gather(end_scores, answer2support)
        gathered_start_scores = tf.gather(start_scores, answer2support)

        if correct_start is not None:
            # assuming we know the correct start we only consider ends after that
            left_mask = misc.mask_for_lengths(tf.cast(correct_start, tf.int32), max_support_length, mask_right=False)
            gathered_end_scores = gathered_end_scores + left_mask

        predicted_start_pointer = tf.argmax(gathered_start_scores, axis=1, output_type=tf.int32)
        predicted_end_pointer = tf.argmax(gathered_end_scores, axis=1, output_type=tf.int32)

        return (start_scores, end_scores,
                tf.gather(doc_idx_for_support, answer2support), predicted_start_pointer, predicted_end_pointer)

    def eval():
        # we collect spans for top k starts and top k ends and select the top k from those top 2k
        doc_idx1, start_pointer1, end_pointer1, span_score1 = _get_top_k(
            start_scores, end_scores, beam_size, max_span_size, support2question)
        doc_idx2, end_pointer2, start_pointer2, span_score2 = _get_top_k(
            end_scores, start_scores, beam_size, -max_span_size, support2question)

        doc_idx = tf.concat([doc_idx1, doc_idx2], 1)
        start_pointer = tf.concat([start_pointer1, start_pointer2], 1)
        end_pointer = tf.concat([end_pointer1, end_pointer2], 1)
        span_score = tf.concat([span_score1, span_score2], 1)

        _, idx = tf.nn.top_k(span_score, beam_size)

        r = tf.range(tf.shape(span_score)[0], dtype=tf.int32)
        r = tf.reshape(tf.tile(tf.expand_dims(r, 1), [1, beam_size]), [-1, 1])

        idx = tf.concat([r, tf.reshape(idx, [-1, 1])], 1)
        doc_idx = tf.gather_nd(doc_idx, idx)
        start_pointer = tf.gather_nd(start_pointer, idx)
        end_pointer = tf.gather_nd(end_pointer, idx)

        return (start_scores, end_scores, tf.gather(doc_idx_for_support, doc_idx), start_pointer, end_pointer)

    return tf.cond(is_eval, eval, train)


def _get_top_k(scores1, scores2, k, max_span_size, support2question):
    max_support_length = tf.shape(scores1)[1]
    doc_idx, pointer1, topk_scores1 = segment_top_k(scores1, support2question, k)

    # [num_questions * beam_size]
    doc_idx_flat = tf.reshape(doc_idx, [-1])
    pointer_flat1 = tf.reshape(pointer1, [-1])

    # [num_questions * beam_size, support_length]
    scores_gathered2 = tf.gather(scores2, doc_idx_flat)
    if max_span_size < 0:
        pointer_flat1, max_span_size = pointer_flat1 + max_span_size + 1, -max_span_size
    left_mask = misc.mask_for_lengths(tf.cast(pointer_flat1, tf.int32),
                                      max_support_length, mask_right=False)
    right_mask = misc.mask_for_lengths(tf.cast(pointer_flat1 + max_span_size, tf.int32),
                                       max_support_length)
    scores_gathered2 = scores_gathered2 + left_mask + right_mask

    pointer2 = tf.argmax(scores_gathered2, axis=1, output_type=tf.int32)

    topk_score2 = tf.gather_nd(scores2, tf.stack([doc_idx_flat, pointer2], 1))

    return doc_idx, pointer1, tf.reshape(pointer2, [-1, k]), topk_scores1 + tf.reshape(topk_score2, [-1, k])


def conditional_answer_layer(size, encoded_question, question_length, encoded_support, support_length,
                             correct_start, support2question, answer2support, is_eval, beam_size=1, max_span_size=10000,
                             bilinear=False):
    question_state = compute_question_state(encoded_question, question_length)
    question_state = tf.gather(question_state, support2question)

    # Prediction
    # start
    if bilinear:
        hidden_start = tf.layers.dense(question_state, size, name="hidden_start")
        start_scores = tf.einsum('ik,ijk->ij', hidden_start, encoded_support)
    else:
        static_input = tf.concat([tf.expand_dims(question_state, 1) * encoded_support, encoded_support], 2)
        hidden_start = tf.layers.dense(question_state, size, name="hidden_start_1")
        hidden_start = tf.layers.dense(
            static_input, size, use_bias=False, name="hidden_start_2") + tf.expand_dims(hidden_start, 1)
        start_scores = tf.layers.dense(tf.nn.relu(hidden_start), 1, use_bias=False, name="start_scores")
        start_scores = tf.squeeze(start_scores, [2])

    support_mask = misc.mask_for_lengths(support_length)
    start_scores = start_scores + support_mask

    max_support_length = tf.shape(start_scores)[1]
    _, _, num_doc_per_question = tf.unique_with_counts(support2question)
    offsets = tf.cumsum(num_doc_per_question, exclusive=True)
    doc_idx_for_support = tf.range(tf.shape(support2question)[0]) - tf.gather(offsets, support2question)

    doc_idx, start_pointer = tf.cond(
        is_eval,
        lambda: segment_top_k(start_scores, support2question, beam_size)[:2],
        lambda: (tf.expand_dims(answer2support, 1), tf.expand_dims(correct_start, 1)))

    doc_idx_flat = tf.reshape(doc_idx, [-1])
    start_pointer = tf.reshape(start_pointer, [-1])

    start_state = tf.gather_nd(encoded_support, tf.stack([doc_idx_flat, start_pointer], 1))
    start_state.set_shape([None, size])

    encoded_support_gathered = tf.gather(encoded_support, doc_idx_flat)
    question_state = tf.gather(question_state, doc_idx_flat)
    if bilinear:
        hidden_end = tf.layers.dense(tf.concat([question_state, start_state], 1), size, name="hidden_end")
        end_scores = tf.einsum('ik,ijk->ij', hidden_end, encoded_support_gathered)
    else:
        end_input = tf.concat([tf.expand_dims(start_state, 1) * encoded_support_gathered,
                               tf.gather(static_input, doc_idx_flat)], 2)

        hidden_end = tf.layers.dense(tf.concat([question_state, start_state], 1), size,
                                     name="hidden_end_1")
        hidden_end = tf.layers.dense(
            end_input, size, use_bias=False, name="hidden_end_2") + tf.expand_dims(hidden_end, 1)

        end_scores = tf.layers.dense(tf.nn.relu(hidden_end), 1, use_bias=False, name="end_scores")
        end_scores = tf.squeeze(end_scores, [2])

    end_scores = end_scores + tf.gather(support_mask, doc_idx_flat)

    def train():
        predicted_end_pointer = tf.argmax(end_scores, axis=1, output_type=tf.int32)
        return start_scores, end_scores, doc_idx, start_pointer, predicted_end_pointer

    def eval():
        # [num_questions * beam_size, support_length]
        left_mask = misc.mask_for_lengths(tf.cast(start_pointer, tf.int32),
                                          max_support_length, mask_right=False)
        right_mask = misc.mask_for_lengths(tf.cast(start_pointer + max_span_size, tf.int32),
                                           max_support_length)
        masked_end_scores = end_scores + left_mask + right_mask
        predicted_ends = tf.argmax(masked_end_scores, axis=1, output_type=tf.int32)

        return (start_scores, masked_end_scores,
                tf.gather(doc_idx_for_support, doc_idx_flat), start_pointer, predicted_ends)

    return tf.cond(is_eval, eval, train)


def bidaf_answer_layer(encoded_support_start, encoded_support_end, support_length,
                       support2question, answer2support, is_eval, beam_size=1, max_span_size=10000):
    # BiLSTM(M) = M^2 = encoded_support_end
    start_scores = tf.squeeze(tf.layers.dense(encoded_support_start, 1, use_bias=False), 2)
    end_scores = tf.squeeze(tf.layers.dense(encoded_support_end, 1, use_bias=False), 2)
    # mask out-of-bounds slots by adding -1000
    support_mask = misc.mask_for_lengths(support_length)
    start_scores = start_scores + support_mask
    end_scores = end_scores + support_mask
    return compute_spans(start_scores, end_scores, answer2support, is_eval,
                         support2question, beam_size=beam_size, max_span_size=max_span_size)
