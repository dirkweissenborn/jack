import tensorflow as tf

from jack.tfutil import misc
from jack.tfutil import modular_encoder
from jack.tfutil.embedding import conv_char_embedding


def embedding_refinement(size, word_embeddings, sequence_module, reading_sequence, reading_sequence_2_batch,
                         reading_sequence_lengths, word2lemma, unique_word_chars=None,
                         unique_word_char_length=None, is_eval=False, sequence_indices=None, num_sequences=4,
                         only_refine=False, keep_prob=1.0, batch_size=None, with_char_embeddings=False, num_chars=0):
    if batch_size is None:
        batch_size = tf.reduce_max(tf.stack([tf.shape(s)[0] if s2b is None else tf.reduce_max(s2b) + 1
                                             for s, s2b in zip(reading_sequence, reading_sequence_2_batch)]))

    sequence_indices = sequence_indices if sequence_indices is not None else list(range(len(reading_sequence)))

    if not only_refine:
        word_embeddings = tf.layers.dense(word_embeddings, size, activation=tf.nn.relu, name="embeddings_projection")
        if with_char_embeddings:
            word_embeddings = word_with_char_embed(
                size, word_embeddings, unique_word_chars, unique_word_char_length, num_chars, is_eval, keep_prob)
        # tile word_embeddings by batch size (individual batches update embeddings individually)
        ctxt_word_embeddings = tf.tile(word_embeddings, tf.stack([batch_size, 1]))
        # HACK so that backprop works with indexed slices that come through here which are not handled by tile
        ctxt_word_embeddings *= 1.0
    else:
        ctxt_word_embeddings = word_embeddings

    num_words = tf.shape(word2lemma)[0]

    # divide uniq words for each question by offsets
    offsets = tf.expand_dims(tf.range(0, num_words * batch_size, num_words), 1)

    # each token is assigned a word idx + offset for distinguishing words between batch instances
    reading_sequence_offset = [
        s + offsets if s2b is None else s + tf.gather(offsets, s2b)
        for s, s2b in zip(reading_sequence, reading_sequence_2_batch)]

    word2lemma_off = tf.tile(tf.reshape(word2lemma, [1, -1]), [batch_size, 1]) + offsets
    word2lemma_off = tf.reshape(word2lemma_off, [-1])

    num_lemmas = tf.reduce_max(word2lemma_off) + 1

    fused_rnn = tf.contrib.rnn.LSTMBlockFusedCell(size)
    with tf.variable_scope("refinement") as vs:
        for i, seq, length in zip(sequence_indices, reading_sequence_offset, reading_sequence_lengths):
            if i > 0:
                vs.reuse_variables()
            batch_size = tf.shape(length)[0]

            def non_zero_batchsize_op():
                max_length = tf.shape(seq)[1]
                encoded = tf.nn.embedding_lookup(ctxt_word_embeddings, seq)
                one_hot = [0.0] * num_sequences
                one_hot[i] = 1.0
                mode_feature = tf.constant([[one_hot]], tf.float32)
                mode_feature = tf.tile(mode_feature, tf.stack([batch_size, max_length, 1]))
                encoded = tf.concat([encoded, mode_feature], 2)
                encoded = modular_encoder.modular_encoder(
                    sequence_module, {'text': encoded}, {'text': length}, {'text': None}, size, is_eval)[0]['text']

                mask = misc.mask_for_lengths(length, max_length, mask_right=False, value=1.0)
                encoded = encoded * tf.expand_dims(mask, 2)

                seq_lemmas = tf.gather(word2lemma_off, tf.reshape(seq, [-1]))
                new_lemma_embeddings = tf.unsorted_segment_max(
                    tf.reshape(encoded, [-1, size]), seq_lemmas, num_lemmas)
                new_lemma_embeddings = tf.nn.relu(new_lemma_embeddings)

                return tf.gather(new_lemma_embeddings, word2lemma_off)

            new_word_embeddings = tf.cond(batch_size > 0, non_zero_batchsize_op,
                                          lambda: tf.zeros_like(ctxt_word_embeddings))
            # update old word embeddings with new ones via gated addition
            gate = tf.layers.dense(tf.concat([ctxt_word_embeddings, new_word_embeddings], 1), size, tf.nn.sigmoid,
                                   bias_initializer=tf.constant_initializer(1.0), name="embeddings_gating")
            ctxt_word_embeddings = ctxt_word_embeddings * gate + (1.0 - gate) * new_word_embeddings

    return ctxt_word_embeddings, reading_sequence_offset, offsets


def word_with_char_embed(size, word_embeddings, unique_word_chars, unique_word_char_length, num_chars, is_eval,
                         keep_prob):
    # compute combined embeddings
    char_word_embeddings = conv_char_embedding(
        num_chars, size, unique_word_chars, unique_word_char_length)
    char_word_embeddings = tf.nn.relu(char_word_embeddings)
    gate = tf.layers.dense(tf.concat([word_embeddings, char_word_embeddings], 1), size, tf.nn.sigmoid,
                           bias_initializer=tf.constant_initializer(1.0), name="embeddings_gating")
    word_embeddings = word_embeddings * gate + (1.0 - gate) * char_word_embeddings
    if keep_prob < 1.0:
        word_embeddings = tf.cond(is_eval,
                                  lambda: word_embeddings,
                                  lambda: tf.nn.dropout(word_embeddings, keep_prob))

    return word_embeddings


def embed_spans(spans, sequence, size, scope="span_embedding"):
    """Spans have 3 entries [batch_id, start, end].

    Returns:
        unique embedded spans, indices of same length as spans pointing into unique embedded spans
    """
    with tf.variable_scope(scope):
        batch_id, starts, ends = tf.split(spans, 3, 1)

        embedded_starts = tf.gather_nd(sequence, tf.stack([batch_id, starts], 1))
        embedded_ends = tf.gather_nd(sequence, tf.stack([batch_id, ends - 1], 1))
        embedded_starts.set_shape([None, sequence.get_shape()[-1].value])
        embedded_ends.set_shape([None, sequence.get_shape()[-1].value])

        embedded_spans = tf.layers.dense(
            tf.concat([embedded_starts, embedded_ends], 1), size, activation=tf.nn.relu, name="span_projection")

        return embedded_spans


def embed_spans_uniq(spans, sequence, size, scope="span_embedding"):
    """Spans have 3 entries [batch_id, start, end].

    Returns:
        unique embedded spans, indices of same length as spans pointing into unique embedded spans
    """
    batch_id, starts, ends = tf.split(spans, 3, 1)
    end_len = tf.reduce_max(ends) + 1
    starts_len = tf.reduce_max(starts) + 1
    # single id as when treating start and end as 3d tensor indices
    span_idx = (batch_id * starts_len + starts) * end_len + ends
    # make them uniq and retain the idxs as back pointers
    uniq_spans, span_idx = tf.unique(tf.squeeze(span_idx, axis=1))

    uniq_batch_start, uniq_ends = tf.div(uniq_spans, end_len), tf.mod(uniq_spans, end_len)
    uniq_batch_ids, uniq_starts = tf.div(uniq_batch_start, starts_len), tf.mod(uniq_batch_start, starts_len)
    embedded_spans = embed_spans(tf.stack([uniq_batch_ids, uniq_starts, uniq_ends], axis=1), sequence, size, scope)

    return embedded_spans, span_idx, uniq_batch_ids, uniq_starts, uniq_ends


def span_pairs_uniq(batch_id, spans1, spans2, scope="span_embedding"):
    starts1, ends1 = tf.split(spans1, 2, 1)
    starts2, ends2 = tf.split(spans2, 2, 1)
    batch_id = tf.expand_dims(batch_id, 1)
    end_len1 = tf.reduce_max(ends1) + 1
    starts_len1 = tf.reduce_max(starts1) + 1
    end_len2 = tf.reduce_max(ends2) + 1
    starts_len2 = tf.reduce_max(starts2) + 1
    # single id as when treating start and end as 3d tensor indices
    span_pair_idx = (batch_id * starts_len1 + starts1) * end_len1 + ends1
    span_pair_idx = (span_pair_idx * starts_len2 + starts2) * end_len2 + ends2
    # make them uniq and retain the idxs as back pointers
    uniq_spans, span_pair_idx = tf.unique(tf.squeeze(span_pair_idx, axis=1))

    uniq_rest, uniq_ends2 = tf.div(uniq_spans, end_len2), tf.mod(uniq_spans, end_len2)
    uniq_rest, uniq_starts2 = tf.div(uniq_rest, starts_len2), tf.mod(uniq_rest, starts_len2)

    uniq_rest, uniq_ends1 = tf.div(uniq_rest, end_len1), tf.mod(uniq_rest, end_len1)
    uniq_batch_ids, uniq_starts1 = tf.div(uniq_rest, starts_len1), tf.mod(uniq_rest, starts_len1)

    return span_pair_idx, uniq_batch_ids

def embed_span_pairs_uniq(batch_id, spans1, spans2, sequence1, sequence2, size, scope="span_embedding"):
    """Spans have 3 entries [batch_id, start, end].

    Returns:
        unique embedded spans, indices of same length as spans pointing into unique embedded spans
    """
    with tf.variable_scope(scope):
        starts1, ends1 = tf.split(spans1, 2, 1)
        starts2, ends2 = tf.split(spans2, 2, 1)
        batch_id = tf.expand_dims(batch_id, 1)
        end_len1 = tf.reduce_max(ends1) + 1
        starts_len1 = tf.reduce_max(starts1) + 1
        end_len2 = tf.reduce_max(ends2) + 1
        starts_len2 = tf.reduce_max(starts2) + 1
        # single id as when treating start and end as 3d tensor indices
        span_pair_idx = (batch_id * starts_len1 + starts1) * end_len1 + ends1
        span_pair_idx = (span_pair_idx * starts_len2 + starts2) * end_len2 + ends2
        # make them uniq and retain the idxs as back pointers
        uniq_spans, span_pair_idx = tf.unique(tf.squeeze(span_pair_idx, axis=1))

        uniq_rest, uniq_ends2 = tf.div(uniq_spans, end_len2), tf.mod(uniq_spans, end_len2)
        uniq_rest, uniq_starts2 = tf.div(uniq_rest, starts_len2), tf.mod(uniq_rest, starts_len2)

        uniq_rest, uniq_ends1 = tf.div(uniq_rest, end_len1), tf.mod(uniq_rest, end_len1)
        uniq_batch_ids, uniq_starts1 = tf.div(uniq_rest, starts_len1), tf.mod(uniq_rest, starts_len1)

        embedded_starts1 = tf.gather_nd(sequence1, tf.stack([uniq_batch_ids, uniq_starts1], 1))
        embedded_ends1 = tf.gather_nd(sequence1, tf.stack([uniq_batch_ids, uniq_ends1 - 1], 1))
        embedded_starts1.set_shape([None, sequence1.get_shape()[-1].value])
        embedded_ends1.set_shape([None, sequence1.get_shape()[-1].value])

        embedded_starts2 = tf.gather_nd(sequence2, tf.stack([uniq_batch_ids, uniq_starts2], 1))
        embedded_ends2 = tf.gather_nd(sequence2, tf.stack([uniq_batch_ids, uniq_ends2 - 1], 1))
        embedded_starts2.set_shape([None, sequence2.get_shape()[-1].value])
        embedded_ends2.set_shape([None, sequence2.get_shape()[-1].value])

        embedded_span_pairs1 = tf.layers.dense(
            tf.concat([embedded_starts1, embedded_ends1], 1), size, activation=tf.nn.relu, name="span_projection1")
        embedded_span_pairs2 = tf.layers.dense(
            tf.concat([embedded_starts2, embedded_ends2], 1), size, activation=tf.nn.relu, name="span_projection2")

        return embedded_span_pairs1, embedded_span_pairs2, span_pair_idx, uniq_batch_ids
