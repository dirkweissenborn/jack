import tensorflow as tf

from jack.core import TensorPort, Ports


class AssertionMRPorts:
    # When feeding embeddings directly
    question_length = Ports.Input.question_length
    support_length = Ports.Input.support_length

    # but also ids, for char-based embeddings
    question = Ports.Input.question
    support = Ports.Input.support

    word_char_length = TensorPort(tf.int32, [None], "word_char_length", "words length", "[U]")

    token_char_offsets = TensorPort(tf.int32, [None, None], "token_char_offsets",
                                    "Character offsets of tokens in support.", "[S, support_length]")

    keep_prob = Ports.keep_prob
    is_eval = Ports.is_eval

    word_embeddings = TensorPort(tf.float32, [None, None], "word_embeddings",
                                 "Embeddings only for words occuring in batch.", "[None, N]")

    assertion_lengths = TensorPort(tf.int32, [None], "assertion_lengths", "Length of assertion.", "[R]")

    assertions = TensorPort(tf.int32, [None, None], "assertions",
                            "Represents batch dependent assertion word ids.",
                            "[R, L]")
    assertion2question = TensorPort(tf.int32, [None], "assertion2question", "Question idx per assertion", "[R]")

    word2lemma = TensorPort(tf.int32, [None], "word2lemma", "Lemma idx per word", "[U]")

    word_chars = TensorPort(tf.int32, [None, None], "word_chars", "Represents words as sequence of chars",
                            "[U, max_num_chars]")

    question_arg_span = TensorPort(tf.int32, [None, 2], "question_arg_span",
                                   "span of an argument in the question", "[Q, 2]")

    support_arg_span = TensorPort(tf.int32, [None, 2], "support_arg_span",
                                  "span of an argument in the suppoort", "[S, 2]")

    assertion2question_arg_span = TensorPort(tf.int32, [None], "assertion2question_arg_span",
                                             "assertion to question span mapping", "[A]")
    assertion2support_arg_span = TensorPort(tf.int32, [None], "assertion2support_arg_span",
                                            "assertion to support span mapping", "[A]")

    logits_3D = TensorPort(tf.float32, [None, None, None], "logits_3D",
                           "3D logits.", "[None, None, num_classes]")

    policy_log_probs = TensorPort(tf.float32, [None, None], "policy_log_probs",
                                  "policy_log_probs.", "[time, batch]")

    selected_assertions = TensorPort(tf.int32, [None, None], "selected_assertions",
                                     "Indices of selected assertions over time steps.", "[None]")

    assertion_selection_mask = TensorPort(tf.bool, [None, None], "assertion_selection_mask",
                                          "Whether or not selection was stopped timestep.", "[time, batch]")
