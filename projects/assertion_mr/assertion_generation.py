import math
import random

import numpy as np
import tensorflow as tf

from jack.core import OnlineInputModule, TensorPort, Ports, OutputModule, Answer, TensorPortTensors
from jack.core.tensorflow import TFModelModule
from jack.tfutil.embedding import conv_char_embedding_multi_filter
from jack.util.hooks import EvalHook, ClassificationEvalHook
from jack.util.map import numpify


class PredicateGenPorts:
    predicate_idx = Ports.Target.target_index
    predicate_symbols = TensorPort(tf.int32, [None, None], "predicate",
                                   "Represents predicates using symbol vectors",
                                   "[batch_size, max_num_predicate_tokens]")
    predicate_length = TensorPort(tf.int32, [None], "predicate_length",
                                  "Represents length of predicates in batch",
                                  "[batch_size]")

    arg1 = TensorPort(tf.int32, [None, None], "embedded_arg1",
                      "one argument of assertion to be generated", "[batch_size]")
    arg_len1 = TensorPort(tf.int32, [None], "arg_len1", "number of argument tokens", "[batch_size]")

    arg2 = TensorPort(tf.int32, [None, None], "embedded_arg2",
                      "one argument of assertion to be generated", "[batch_size]")
    arg_len2 = TensorPort(tf.int32, [None], "arg_len2", "number of argument tokens", "[batch_size]")

    arg_chars1 = TensorPort(tf.int32, [None, None, None], "arg_chars1",
                            "seq of tokens represented by characters of argument 1",
                            "[batch_size, num_tokens, num_chars]")
    arg_chars_len1 = TensorPort(tf.int32, [None, None], "arg_chars_len1",
                                "length of tokens of argument 1", "[batch_size, num_tokens]")

    arg_chars2 = TensorPort(tf.int32, [None, None, None], "arg_chars2",
                            "seq of tokens represented by characters of argument 2",
                            "[batch_size, num_tokens, num_chars]")
    arg_chars_len2 = TensorPort(tf.int32, [None, None], "arg_chars_len2",
                                "length of tokens of argument 2", "[batch_size, num_tokens]")

    reversed_args_logits = TensorPort(tf.float32, [None], "reversed_args_prob",
                                    "order prob of arguments determining subject and object", "[batch_size]")

    reversed_args_target = TensorPort(tf.bool, [None], "reversed_args_target",
                                      "order of arguments determining subject and object", "[batch_size]")


def _np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - x.max(axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class PredicateGenEvalHook(EvalHook):
    def __init__(self, reader, dataset, batch_size: int,
                 iter_interval=None, epoch_interval=1, metrics=None, summary_writer=None,
                 write_metrics_to=None, info="", side_effect=None, **kwargs):
        ports = [Ports.Prediction.logits_3D, PredicateGenPorts.predicate_symbols,
                 PredicateGenPorts.predicate_length]
        super().__init__(reader, dataset, batch_size, ports, iter_interval, epoch_interval, metrics, summary_writer,
                         write_metrics_to, info, side_effect)

    @property
    def possible_metrics(self):
        return ["Neg_Perplexity", "Accuracy", "Exact"]

    @staticmethod
    def preferred_metric_and_best_score():
        return 'Exact', [0.0]

    def apply_metrics(self, tensors):
        logits = tensors.get(Ports.Prediction.logits_3D)
        symbols = tensors[PredicateGenPorts.predicate_symbols]
        lengths = tensors[PredicateGenPorts.predicate_length]
        log_probs = np.log(_np_softmax(logits) + 1e-6)
        summed_log_probs = []
        summed_accuracy = 0
        summed_exact = 0
        for p, l, s in zip(log_probs, lengths, symbols):
            summed_log_probs.append(sum(p1[s1] for _, p1, s1 in zip(range(l), p, s)) / l)
            summed_accuracy += sum(float(np.argmax(p1) == s1) for _, p1, s1 in zip(range(l), p, s)) / l
            summed_exact += float(all(np.argmax(p1) == s1 for _, p1, s1 in zip(range(l), p, s)))

        summed_log_probs = np.sum(np.stack(summed_log_probs))
        return {"Neg_Perplexity": summed_log_probs, "Accuracy": summed_accuracy, "Exact": summed_exact}

    def combine_metrics(self, accumulated_metrics):
        # 2x because we always generate 'not related to' assertions as well
        return {"Neg_Perplexity": -math.exp(-sum(accumulated_metrics["Neg_Perplexity"]) / (2 * self._total)),
                "Accuracy": sum(accumulated_metrics["Accuracy"]) / (2 * self._total),
                "Exact": sum(accumulated_metrics["Exact"]) / (2 * self._total)}


class PredicateClassificationEvalHook(ClassificationEvalHook):
    @property
    def possible_metrics(self):
        return ["Accuracy", "F1", "Precision", "Recall"]

    @staticmethod
    def preferred_metric_and_best_score():
        return 'Precision', [0.0]

    def apply_metrics(self, tensors):
        labels = tensors[Ports.Target.target_index]
        predictions = tensors[Ports.Prediction.candidate_index]

        idx = labels.shape[0]
        normalizer = predictions.shape[0] // idx

        equal = np.equal(labels, predictions[:idx])
        tp = np.sum(equal)
        exact = tp + np.sum(np.equal(predictions[idx:], 0))
        fp = np.sum(np.greater(predictions[idx:], 0))

        precision = tp / (fp + tp + 1e-6)
        recall = tp / idx
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return {"F1": f1 * idx, "Precision": precision * idx, "Recall": recall * idx, "Accuracy": exact / normalizer}


class PredicateGenerationInputModule(OnlineInputModule):
    def __init__(self, shared_resources):
        self.shared_resources = shared_resources

    def setup_from_data(self, data):
        max_concept_vocab = self.shared_resources.config.get('max_concept_vocab', 1000000)
        predicate_dict = {'not': 0, 'related': 1, 'to': 2}
        full_predicate_dict = {'not related to': 0}
        concept_dict = dict()
        max_length = 0
        for question, answers in data:
            if answers[0].text not in full_predicate_dict:
                full_predicate_dict[answers[0].text] = len(full_predicate_dict)
            split = answers[0].text.split()
            max_length = max(max_length, len(split))
            for w in split:
                if w not in predicate_dict:
                    predicate_dict[w] = len(predicate_dict)
            arg1, arg2 = question.question.split("|")
            arg1, arg2 = arg1.split(), arg2.split()
            for w in arg1:
                if w not in concept_dict:
                    concept_dict[w] = min(len(concept_dict), max_concept_vocab)
            for w in arg2:
                if w not in concept_dict:
                    concept_dict[w] = min(len(concept_dict), max_concept_vocab)
        self.shared_resources.predicate_vocab = predicate_dict
        self.shared_resources.full_predicate_vocab = full_predicate_dict
        self.shared_resources.max_predicate_length = max_length
        self.shared_resources.char_vocab = {chr(i): i for i in range(256)}
        self.shared_resources.concept_dict = concept_dict

    def setup(self):
        self.vocab = self.shared_resources.vocab
        self.config = self.shared_resources.config
        self.default_vec = np.zeros([self.vocab.emb_length])
        self.char_vocab = self.shared_resources.char_vocab

    @property
    def output_ports(self):
        return [PredicateGenPorts.predicate_symbols, PredicateGenPorts.predicate_length,
                PredicateGenPorts.predicate_idx,
                PredicateGenPorts.arg1, PredicateGenPorts.arg_len1,
                PredicateGenPorts.arg_chars1, PredicateGenPorts.arg_chars_len1,
                PredicateGenPorts.arg2, PredicateGenPorts.arg_len2,
                PredicateGenPorts.arg_chars2, PredicateGenPorts.arg_chars_len2,
                Ports.is_eval]

    @property
    def training_ports(self):
        return [PredicateGenPorts.reversed_args_target]

    def preprocess(self, questions, answers=None, is_eval=False):
        preprocessed = []
        rng = random.Random(123)
        for i, q in enumerate(questions):
            arg1, arg2 = q.question.split("|")
            reversed_args = not is_eval and rng.random() > 0.5
            if reversed_args:  # randomly switch arguments
                arg1, arg2 = arg2, arg1
            arg1, arg2 = arg1.split(), arg2.split()
            preprocessed.append({'arg1': arg1, 'arg2': arg2, 'reversed_args': reversed_args,
                                 'predicates': np.zeros([0], np.int32)})
            if answers is not None:
                preprocessed[-1]['predicates'] = [self.shared_resources.predicate_vocab[w]
                                                  for w in answers[i][0].text.split()]
                preprocessed[-1]['full_predicates'] = self.shared_resources.full_predicate_vocab[answers[i][0].text]
        return preprocessed

    def create_batch(self, annotations, is_eval: bool, with_answers: bool):
        full_predicates, predicates, pred_lengths = [], [], []
        arg1, arg_len1 = [], []
        arg_chars1, arg_chars_len1 = [], []
        arg2, arg_len2 = [], []
        arg_chars2, arg_chars_len2 = [], []
        for j, annot in enumerate(annotations):
            arg1.append([self.shared_resources.concept_dict.get(w, len(self.shared_resources.concept_dict))
                         for w in annot['arg1']])
            arg2.append([self.shared_resources.concept_dict.get(w, len(self.shared_resources.concept_dict))
                         for w in annot['arg2']])
            arg_len1.append(len(annot['arg1']))
            arg_len2.append(len(annot['arg2']))
            arg_chars1.append([[self.char_vocab.get(c, 0) for c in w] for w in annot['arg1']])
            arg_chars2.append([[self.char_vocab.get(c, 0) for c in w] for w in annot['arg2']])
            arg_chars_len1.append([len(w) for w in annot['arg1']])
            arg_chars_len2.append([len(w) for w in annot['arg2']])
            if with_answers:
                full_predicates.append(annot['full_predicates'])
                predicates.append(annot['predicates'])
                pred_lengths.append(len(annot['predicates'])
                                    if isinstance(annot['predicates'], list) else annot['predicates'].shape[0])

        batch = {
            PredicateGenPorts.predicate_symbols: predicates if with_answers else np.zeros([0, 0], np.float32),
            PredicateGenPorts.predicate_length: pred_lengths,
            PredicateGenPorts.predicate_idx: full_predicates,
            PredicateGenPorts.arg1: arg1,
            PredicateGenPorts.arg_len1: arg_len1,
            PredicateGenPorts.arg_chars1: arg_chars1,
            PredicateGenPorts.arg_chars_len1: arg_chars_len1,
            PredicateGenPorts.arg2: arg2,
            PredicateGenPorts.arg_len2: arg_len2,
            PredicateGenPorts.arg_chars2: arg_chars2,
            PredicateGenPorts.arg_chars_len2: arg_chars_len2,
            Ports.is_eval: is_eval
        }

        if with_answers:
            batch[PredicateGenPorts.reversed_args_target] = [annot["reversed_args"] for annot in annotations]

        return numpify(batch)

    def _get_emb(self, word):
        emb = self.vocab.emb(word)
        return emb if emb is not None else self.default_vec


class PredicateGenerationOutputModule(OutputModule):
    def __init__(self, shared_resources):
        self._shared_resources = shared_resources

    def setup(self):
        self._rev_vocab = {v: k for k, v in self._shared_resources.predicate_vocab.items()}

    @property
    def input_ports(self):
        return [Ports.Prediction.symbols, Ports.Prediction.seq_length, PredicateGenPorts.reversed_args_logits]

    def __call__(self, inputs, symbols, lengths, reversed_args_probs):
        answer = []
        for qa, s, l, p in zip(inputs, symbols, lengths, reversed_args_probs):
            arg1, arg2 = qa.question.split('|')
            answer_text = [arg2 if p > 0.5 else arg1]
            for symbol in s[:l]:
                answer_text.append(self._rev_vocab[symbol])
            answer_text.append(arg1 if p > 0.5 else arg2)
            answer.append(Answer(' '.join(answer_text)))
        return answer


class RNNPredicateGenerationModelModule(TFModelModule):
    @property
    def training_input_ports(self):
        return [Ports.Prediction.logits_3D, PredicateGenPorts.predicate_symbols,
                PredicateGenPorts.predicate_length, PredicateGenPorts.reversed_args_logits,
                PredicateGenPorts.reversed_args_target]

    @property
    def input_ports(self):
        return [PredicateGenPorts.predicate_symbols, PredicateGenPorts.predicate_length,
                PredicateGenPorts.arg1, PredicateGenPorts.arg_len1,
                PredicateGenPorts.arg_chars1, PredicateGenPorts.arg_chars_len1,
                PredicateGenPorts.arg2, PredicateGenPorts.arg_len2,
                PredicateGenPorts.arg_chars2, PredicateGenPorts.arg_chars_len2, Ports.is_eval]

    @property
    def output_ports(self):
        return [Ports.Prediction.logits_3D, Ports.Prediction.symbols,
                Ports.Prediction.seq_length, PredicateGenPorts.reversed_args_logits]

    @property
    def training_output_ports(self):
        return [Ports.loss]

    def create_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)

        arg1 = tensors.arg1
        arg_len1 = tensors.arg_len1
        arg_chars1 = tensors.arg_chars1
        arg_chars_len1 = tensors.arg_chars_len1
        arg2 = tensors.arg2
        arg_len2 = tensors.arg_len2
        arg_chars2 = tensors.arg_chars2
        arg_chars_len2 = tensors.arg_chars_len2
        is_eval = tensors.is_eval
        predicate_symbols = tensors.predicate_symbols
        predicate_length = tensors.predicate_length

        size = shared_resources.config['repr_dim']
        with_char_embeddings = shared_resources.config.get('with_char_embeddings', False)
        max_conv_width = shared_resources.config.get('max_conv_width', 9)
        num_chars = len(shared_resources.char_vocab)
        num_arg_words = max(shared_resources.concept_dict.values()) + 1
        num_predicate_tokens = len(shared_resources.predicate_vocab)
        return predicate_rnn_generator(
            size, num_arg_words, with_char_embeddings, max_conv_width, num_chars, num_predicate_tokens,
            shared_resources.max_predicate_length, predicate_symbols, predicate_length,
            arg1, arg_len1, arg_chars1, arg_chars_len1,
            arg2, arg_len2, arg_chars2, arg_chars_len2, is_eval)

    def create_training_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)
        rev_predicate_symbols = tf.reverse_sequence(input_tensors.predicate_symbols, input_tensors.predicate_length, 1,
                                                    0)
        eos = tf.constant(len(shared_resources.predicate_vocab) + 1, tf.int32, [1, 1])
        eos = tf.tile(eos, [tf.shape(tensors.predicate_length)[0], 1])
        rev_predicate_symbols = tf.concat([eos, rev_predicate_symbols], axis=1)
        predicate_length = tensors.predicate_length + 1
        predicate_symbols = tf.reverse_sequence(rev_predicate_symbols, predicate_length, 1, 0)
        weights = tf.sequence_mask(predicate_length, dtype=tf.float32)
        seq_loss = tf.losses.sparse_softmax_cross_entropy(labels=predicate_symbols, logits=tensors.logits_3D,
                                                          weights=weights)
        # seq_loss = multi_class_hinge_loss(labels=predicate_symbols, logits=logits_3D, weights=weights)
        arg_order_loss = tf.losses.log_loss(tensors.reversed_args, tf.sigmoid(tensors.arg_ordering_logits))
        return seq_loss + arg_order_loss,


class FixedPredicateGenerationModelModule(TFModelModule):

    @property
    def input_ports(self):
        return [PredicateGenPorts.predicate_idx,
                PredicateGenPorts.arg1, PredicateGenPorts.arg_len1,
                PredicateGenPorts.arg_chars1, PredicateGenPorts.arg_chars_len1,
                PredicateGenPorts.arg2, PredicateGenPorts.arg_len2,
                PredicateGenPorts.arg_chars2, PredicateGenPorts.arg_chars_len2, Ports.is_eval]

    @property
    def output_ports(self):
        return [Ports.Prediction.logits, Ports.Prediction.candidate_index, Ports.Prediction.symbols,
                Ports.Prediction.seq_length, PredicateGenPorts.reversed_args_logits]

    def create_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)

        predicate_idx = tensors.predicate_idx
        arg1 = tensors.arg1
        arg_len1 = tensors.arg_len1
        arg_chars1 = tensors.arg_chars1
        arg_chars_len1 = tensors.arg_chars_len1
        arg2 = tensors.arg2
        arg_len2 = tensors.arg_len2
        arg_chars2 = tensors.arg_chars2
        arg_chars_len2 = tensors.arg_chars_len2
        is_eval = tensors.is_eval

        size = shared_resources.config['repr_dim']
        with_char_embeddings = shared_resources.config.get('with_char_embeddings', False)
        max_conv_width = shared_resources.config.get('max_conv_width', 9)
        num_chars = len(shared_resources.char_vocab)
        num_arg_words = max(shared_resources.concept_dict.values()) + 1
        num_neg = shared_resources.config.get('num_negatives', 1)
        return predicate_classification_generator(
            size, num_arg_words, with_char_embeddings, shared_resources.full_predicate_vocab,
            shared_resources.predicate_vocab,
            arg_chars1, arg_chars2, arg_chars_len1, arg_chars_len2, arg1, arg2, arg_len1, arg_len2,
            is_eval, max_conv_width, num_chars, predicate_idx, num_neg)

    @property
    def training_input_ports(self):
        return [Ports.Prediction.logits, PredicateGenPorts.predicate_idx,
                PredicateGenPorts.reversed_args_logits, PredicateGenPorts.reversed_args_target]

    @property
    def training_output_ports(self):
        return [Ports.loss]

    def create_training_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)
        num_neg = shared_resources.config.get('num_negatives', 1)
        neg_labels = tf.zeros(tf.shape(tensors.predicate_idx) * num_neg, dtype=tf.int32)
        predicate_idx = tf.concat([tensors.predicate_idx, neg_labels], axis=0)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=predicate_idx, logits=tensors.logits)
        arg_order_loss = tf.losses.log_loss(tensors.reversed_args, tf.sigmoid(tensors.arg_ordering_logits))
        return loss + arg_order_loss,


def embed_with_chars(size, input_size, num_chars, max_conv_width, arg_chars, arg_chars_len):
    # compute combined embeddings
    arg_chars_flat = tf.reshape(arg_chars, [-1, tf.shape(arg_chars)[-1]])
    arg_chars_len_flat = tf.reshape(arg_chars_len, [-1])
    # 8 -> [16, 0, 32, 0, 64, 0, 128, 0, 256]
    conv_filter_sizes = [0 if i % 2 == 1 else 2 ** (i // 2 + 4) for i in range(max_conv_width)]
    char_embedded = conv_char_embedding_multi_filter(
        num_chars, conv_filter_sizes, 4 * size, arg_chars_flat, arg_chars_len_flat,
        projection_size=size)
    char_embedded = tf.tanh(char_embedded)
    char_embedded = tf.reshape(char_embedded, tf.concat([tf.shape(arg_chars_len), [size]], 0))
    return char_embedded


def predicate_rnn_generator(
        size, num_arg_words, with_char_embeddings, max_conv_width, num_chars, num_predicate_tokens,
        max_predicate_length, predicate_symbols, predicate_length,
        arg1, arg_len1, arg_chars1, arg_chars_len1,
        arg2, arg_len2, arg_chars2, arg_chars_len2, is_eval):
    with tf.variable_scope("predicate_generation"):
        embeddings = tf.get_variable('arg_embeddings', shape=[num_arg_words, size],
                                     initializer=tf.random_normal_initializer(0.0, 0.1))
        arg_embedded1 = tf.nn.embedding_lookup(embeddings, arg1)
        arg_embedded2 = tf.nn.embedding_lookup(embeddings, arg2)
        input_size = arg_embedded1.get_shape()[-1].value
        if with_char_embeddings:
            with tf.variable_scope("char_embeddings") as vs:
                char_embedded1 = embed_with_chars(
                    size, input_size, num_chars, max_conv_width, arg_chars1, arg_chars_len1)
                arg_embedded1 = tf.concat([arg_embedded1, char_embedded1], axis=-1)
                vs.reuse_variables()
                char_embedded2 = embed_with_chars(
                    size, input_size, num_chars, max_conv_width, arg_chars2, arg_chars_len2)
                arg_embedded2 = tf.concat([arg_embedded2, char_embedded2], axis=-1)

        predicted_predicates, pred_lengths, logits, arg_ordering_logits = predicate_rnn_decoder(
            size, num_predicate_tokens, arg_embedded1, arg_len1, arg_embedded2, arg_len2,
            max_predicate_length, is_eval, predicate_symbols, predicate_length)

        return logits, predicted_predicates, pred_lengths, arg_ordering_logits


def predicate_rnn_decoder(size, vocab_size, arg_embedded1, arg_len1, arg_embedded2, arg_len2, max_len, is_eval,
                          supervised_inputs=None, input_lengths=None):
    # first encode arguments
    with tf.variable_scope('arg_encoding') as vs:
        fused_rnn = tf.contrib.rnn.LSTMBlockFusedCell(size)
        embedded_arg1 = tf.transpose(arg_embedded1, [1, 0, 2])
        _, (_, encoded_args1) = fused_rnn(embedded_arg1, sequence_length=arg_len1, dtype=tf.float32)
        projected_args1 = tf.layers.dense(encoded_args1, size, tf.tanh, name='arg_projection')
        vs.reuse_variables()
        embedded_arg2 = tf.transpose(arg_embedded2, [1, 0, 2])
        _, (_, encoded_args2) = fused_rnn(embedded_arg2, sequence_length=arg_len2, dtype=tf.float32)
        projected_args2 = tf.layers.dense(encoded_args2, size, tf.tanh, name='arg_projection')

    hidden = projected_args1 * projected_args2

    with tf.variable_scope('predicate_decoding'):
        relation_init_state = tf.layers.dense(hidden, size, tf.tanh)
        e = tf.get_variable('embeddings', [vocab_size + 2, size],
                            initializer=tf.random_normal_initializer(0.0, 0.1))

        def eval_greedy():
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=e, start_tokens=tf.tile([vocab_size], [tf.shape(arg_len1)[0]]), end_token=vocab_size + 1)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=tf.contrib.rnn.GRUBlockCell(size), helper=helper,
                initial_state=relation_init_state, output_layer=tf.layers.Dense(vocab_size + 2))
            (logits, sample_ids), _, pred_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False, impute_finished=True, maximum_iterations=max_len + 1,
                scope='decode')
            return logits, sample_ids, pred_lengths - 1

        def train_sampling():
            tf.get_variable_scope().reuse_variables()
            helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                embedding=e, start_tokens=tf.tile([vocab_size], [tf.shape(arg_len1)[0]]),
                end_token=vocab_size + 1)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=tf.contrib.rnn.GRUBlockCell(size), helper=helper,
                initial_state=relation_init_state, output_layer=tf.layers.Dense(vocab_size + 2))
            (logits, sample_ids), _, pred_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False, impute_finished=True, maximum_iterations=max_len + 1,
                scope='decode')
            return logits, sample_ids, pred_lengths - 1

        def train_eval_supervised():
            tf.get_variable_scope().reuse_variables()
            sos_inputs = tf.constant(vocab_size, tf.int32, [1, 1])
            sos_inputs = tf.tile(sos_inputs, [tf.shape(arg_len1)[0], 1])
            supervised_inputs_sos = tf.concat([sos_inputs, supervised_inputs], axis=1)
            emb_supervised_inputs_sos = tf.nn.embedding_lookup(e, supervised_inputs_sos)
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=emb_supervised_inputs_sos,
                sequence_length=input_lengths + 1)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=tf.contrib.rnn.GRUBlockCell(size), helper=helper,
                initial_state=relation_init_state, output_layer=tf.layers.Dense(vocab_size + 2))
            (logits, sample_ids), _, pred_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False, impute_finished=True, maximum_iterations=max_len + 1,
                scope='decode')
            return tf.cond(is_eval,
                           lambda: (logits, sample_ids, input_lengths),
                           lambda: (logits, supervised_inputs, input_lengths))

        if supervised_inputs is not None:
            logits, predicates, pred_lengths = tf.cond(
                is_eval,
                lambda: tf.cond(tf.equal(tf.shape(supervised_inputs)[0], 0), eval_greedy, train_eval_supervised),
                train_eval_supervised)
        else:
            logits, predicates, pred_lengths = tf.cond(is_eval, eval_greedy, train_sampling)

    with tf.variable_scope('predicate_encoding'):
        e = tf.get_variable('embeddings', [vocab_size + 2, size],
                            initializer=tf.random_normal_initializer(0.0, 0.1))
        fused_rnn = tf.contrib.rnn.LSTMBlockFusedCell(size)
        embedded_predicates = tf.nn.embedding_lookup(e, tf.transpose(predicates))
        _, (_, encoded_predicates) = fused_rnn(embedded_predicates, sequence_length=pred_lengths, dtype=tf.float32)

    with tf.variable_scope('arg_ordering'):
        hidden_o = tf.layers.dense(
            tf.concat([projected_args1, projected_args2, encoded_predicates], axis=-1), size, tf.nn.relu)
        arg_ordering_logits = tf.squeeze(tf.layers.dense(hidden_o, 1), axis=-1)

    return predicates, pred_lengths, logits, arg_ordering_logits


def predicate_classification_generator(size, num_arg_words, with_char_embeddings,
                                       full_predicate_vocab, predicate_vocab,
                                       arg_chars1, arg_chars2, arg_chars_len1,
                                       arg_chars_len2, arg1, arg2, arg_len1, arg_len2, is_eval,
                                       max_conv_width, num_chars, predicate_idx=None, num_neg=0):
    num_predicates = len(full_predicate_vocab)
    embeddings = tf.get_variable('arg_embeddings', shape=[num_arg_words, size],
                                 initializer=tf.random_normal_initializer(0.0, 0.1))
    arg_embedded1 = tf.nn.embedding_lookup(embeddings, arg1)
    arg_embedded2 = tf.nn.embedding_lookup(embeddings, arg2)
    predicates, logits, arg_ordering_logits = predicate_fixed_decoder(
        size, with_char_embeddings, max_conv_width, num_chars, num_predicates, num_neg,
        arg_embedded1, arg_len1, arg_chars1, arg_chars_len1,
        arg_embedded2, arg_len2, arg_chars2, arg_chars_len2, is_eval, predicate_idx)
    max_predicate_length = max(len(p.split()) for p in full_predicate_vocab)
    predicates_seq_const = [[0] * max_predicate_length for _ in range(num_predicates)]
    predicates_length_const = [0 for _ in range(num_predicates)]
    for p, i in full_predicate_vocab.items():
        split = p.split()
        predicates_length_const[i] = len(split)
        for j, w in enumerate(split):
            predicates_seq_const[i][j] = predicate_vocab[w]
    predicates_seq_const = tf.constant(numpify([predicates_seq_const])[0])
    predicates_length_const = tf.constant(predicates_length_const)
    predicates_seq = tf.gather(predicates_seq_const, predicates)
    predicates_length = tf.gather(predicates_length_const, predicates)
    return logits, predicates, predicates_seq, predicates_length, arg_ordering_logits


def predicate_fixed_decoder(size, with_char_embeddings, max_conv_width, num_chars, num_predicates, num_neg,
                            arg_embedded1, arg_len1, arg_chars1, arg_chars_len1,
                            arg_embedded2, arg_len2, arg_chars2, arg_chars_len2, is_eval, predicate_idx=None):
    with tf.variable_scope("predicate_generation"):
        input_size = arg_embedded1.get_shape()[-1].value
        if with_char_embeddings:
            with tf.variable_scope("char_embeddings") as vs:
                char_embedded1 = embed_with_chars(
                    size, input_size, num_chars, max_conv_width, arg_chars1, arg_chars_len1)
                arg_embedded1 = tf.concat([arg_embedded1, char_embedded1], axis=-1)
                arg_embedded1.set_shape([None, None, input_size + size])
                vs.reuse_variables()
                char_embedded2 = embed_with_chars(
                    size, input_size, num_chars, max_conv_width, arg_chars2, arg_chars_len2)
                arg_embedded2 = tf.concat([arg_embedded2, char_embedded2], axis=-1)
                arg_embedded2.set_shape([None, None, input_size + size])

        with tf.variable_scope('arg_encoding') as vs:
            fused_rnn = tf.contrib.rnn.LSTMBlockFusedCell(size)
            embedded_arg1 = tf.transpose(arg_embedded1, [1, 0, 2])
            _, (_, encoded_args1) = fused_rnn(embedded_arg1, sequence_length=arg_len1, dtype=tf.float32)
            projected_args1 = tf.layers.dense(encoded_args1, size, tf.tanh, name='arg_projection')
            vs.reuse_variables()
            embedded_arg2 = tf.transpose(arg_embedded2, [1, 0, 2])
            _, (_, encoded_args2) = fused_rnn(embedded_arg2, sequence_length=arg_len2, dtype=tf.float32)
            projected_args2 = tf.layers.dense(encoded_args2, size, tf.tanh, name='arg_projection')

        def hidden_with_negatives():
            hidden = [projected_args1 * projected_args2]
            for n in range(1, num_neg + 1):
                hidden.append(projected_args1 * tf.concat([projected_args2[num_neg:], projected_args2[:num_neg]], 0))
            return tf.concat(hidden, 0)

        if predicate_idx is None:
            hidden = projected_args1 * projected_args2
        else:
            hidden = tf.cond(tf.logical_and(tf.shape(predicate_idx)[0] > 0, num_neg > 0), hidden_with_negatives,
                             lambda: projected_args1 * projected_args2)
        logits = tf.layers.dense(hidden, num_predicates)

        if predicate_idx is not None:
            predicates = tf.cond(is_eval, lambda: tf.argmax(logits, 1, output_type=tf.int32), lambda: predicate_idx)
        else:
            predicates = tf.argmax(logits, 1, output_type=tf.int32)

        with tf.variable_scope('arg_ordering'):
            e = tf.get_variable('embeddings', [num_predicates, size],
                                initializer=tf.random_normal_initializer(0.0, 0.1))
            encoded_predicates = tf.nn.embedding_lookup(e, predicates)
            hidden_o = tf.layers.dense(
                tf.concat([projected_args1, projected_args2, encoded_predicates], axis=-1), size, tf.nn.relu)
            arg_ordering_logits = tf.squeeze(tf.layers.dense(hidden_o, 1), axis=-1)

    return predicates, logits, arg_ordering_logits
