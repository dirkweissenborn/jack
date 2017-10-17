# -*- coding: utf-8 -*-

import math

import tensorflow as tf

from jack.tf_util import misc


def conv_char_embedding(char_vocab, size, unique_word_chars, unique_word_lengths, sequences_as_uniqs,
                        conv_width=5, emb_initializer=tf.random_normal_initializer(0.0, 0.1), scope=None):
    # "fixed PADDING on character level"
    pad = tf.zeros(tf.stack([tf.shape(unique_word_lengths)[0], math.floor(conv_width / 2)]), tf.int32)
    unique_word_chars = tf.concat([pad, unique_word_chars, pad], 1)

    if not isinstance(sequences_as_uniqs, list):
        sequences_as_uniqs = [sequences_as_uniqs]

    with tf.variable_scope(scope or "char_embeddings") as vs:
        char_embedding_matrix = \
            tf.get_variable("char_embedding_matrix", shape=(len(char_vocab), size),
                            initializer=emb_initializer, trainable=True)

        max_word_length = tf.reduce_max(unique_word_lengths)
        embedded_chars = tf.nn.embedding_lookup(char_embedding_matrix, tf.cast(unique_word_chars, tf.int32))

        with tf.variable_scope("conv"):
            # create filter like this to get fan-in and fan-out right for initializers depending on those
            filter = tf.get_variable("filter", [conv_width*size, size])
            filter_reshaped = tf.reshape(filter, [conv_width, size, size])
            # [B, T, S + pad_right]
            conv_out = tf.nn.conv1d(embedded_chars, filter_reshaped, 1, "VALID")
            conv_mask = tf.expand_dims(misc.mask_for_lengths(unique_word_lengths, max_length=max_word_length), 2)
            conv_out = conv_out + conv_mask

        unique_embedded_words = tf.reduce_max(conv_out, [1])

        all_embedded = []
        for word_idxs in sequences_as_uniqs:
            embedded_words = tf.nn.embedding_lookup(unique_embedded_words, word_idxs)
            all_embedded.append(embedded_words)

    return all_embedded
