import os
import pickle
import shelve

import tensorflow as tf
from nltk.corpus import stopwords

from jack.core import TensorPort, FlatPorts, Ports
from jack.readers.extractive_qa.shared import ParameterTensorPorts


class AssertionMRPorts:
    # When feeding embeddings directly
    question_length = FlatPorts.Input.question_length
    support_length = FlatPorts.Input.support_length

    # but also ids, for char-based embeddings
    question = Ports.Input.question
    support = Ports.Input.support

    word_char_length = TensorPort(tf.int32, [None], "char_length", "words length", "[U]")

    token_char_offsets = TensorPort(tf.int32, [None, None], "token_char_offsets",
                                    "Character offsets of tokens in support.", "[S, support_length]")

    keep_prob = ParameterTensorPorts.keep_prob
    is_eval = ParameterTensorPorts.is_eval

    word_embeddings = TensorPort(tf.float32, [None, None], "word_embeddings",
                                 "Embeddings only for words occuring in batch.", "[None, N]")

    assertion_lengths = TensorPort(tf.int32, [None], "assertion_lengths", "Length of assertion.", "[R]")

    assertions = TensorPort(tf.int32, [None, None], "assertion2unique",
                            "Represents batch dependent assertion word ids.",
                            "[R, L]")
    assertion2question = TensorPort(tf.int32, [None], "assertion2question", "Question idx per assertion", "[R]")

    word2lemma = TensorPort(tf.int32, [None], "uniqueword2uniquelemma", "Lemma idx per word", "[U]")

    word_chars = TensorPort(tf.int32, [None, None], "chars", "Represents words as sequence of chars",
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


class AssertionStore(object):
    def __init__(self, path, resources):
        self._resources = resources
        self._sws = set(stopwords.words('english'))
        self._assertion_db = shelve.open(os.path.join(path, 'assertions.shelve'), flag='r')
        with open(os.path.join(path, 'object2assertions.pkl'), "rb") as f:
            self._object2assertions = pickle.load(f)
        with open(os.path.join(path, 'subject2assertions.pkl'), "rb") as f:
            self._subject2assertions = pickle.load(f)

    def get_assertion_keys(self, source_tokens, target_tokens):
        """
        Returns:
            Iterable[str](assertions)
        """

        def key_iterator(tokens):
            for i in range(len(tokens)):
                for j in range(i + 1, min(i + 6, len(tokens) + 1)):
                    if tokens[j - 1] not in self._sws and tokens[j - 1].isalnum():
                        yield tokens[i:j], i, j

        source_obj_assertions = dict()
        source_subj_assertions = dict()
        keys = set()
        for ks, start, end in key_iterator(source_tokens):
            k = " ".join(ks)
            if k in keys:
                continue
            keys.add(k)
            for source in self._resources:
                k_assertions = self._object2assertions[source].get(k)
                if k_assertions is not None:
                    idf = 1.0 / len(k_assertions)
                    for a in k_assertions:
                        source_obj_assertions[a] = (
                            max(source_obj_assertions.get(a, (0.0, None))[0], idf), ks, start, end)
                k_assertions = self._subject2assertions[source].get(k)
                if k_assertions is not None:
                    idf = 1.0 / len(k_assertions)
                    for a in k_assertions:
                        source_subj_assertions[a] = (
                            max(source_subj_assertions.get(a, (0.0, None))[0], idf), ks, start, end)

        assertions = dict()
        assertion_args = dict()
        keys = set()
        for ks, start, end in key_iterator(target_tokens):
            k = " ".join(ks)
            if k in keys:
                continue
            keys.add(k)
            for source in self._resources:
                # subject from target, object from source
                k_assertions_subj = self._subject2assertions[source].get(k)
                if k_assertions_subj is not None:
                    idf2 = 1.0 / len(k_assertions_subj)
                    for a in k_assertions_subj:
                        idf, ks2, start2, end2 = source_obj_assertions.get(a, (None, None, None, None))
                        if idf is None or all(k in ks2 for k in ks) or all(k in ks for k in ks2):
                            continue
                        assertions[a] = max(assertions.get(a, 0.0), idf * idf2)
                        assertion_args[a] = [start2, end2], [start, end]
                # subject from source, object from target
                k_assertions = self._object2assertions[source].get(k)
                if k_assertions is not None:
                    idf2 = 1.0 / len(k_assertions)
                    for a in k_assertions:
                        idf, ks2, start2, end2 = source_subj_assertions.get(a, (None, None, None, None))
                        if idf is None or all(k in ks2 for k in ks) or all(k in ks for k in ks2):
                            continue
                        assertions[a] = max(assertions.get(a, 0.0), idf * idf2)
                        assertion_args[a] = [start2, end2], [start, end]
        return assertions, assertion_args

    def get_assertion(self, assertion_key):
        return self._assertion_db.get(assertion_key)
