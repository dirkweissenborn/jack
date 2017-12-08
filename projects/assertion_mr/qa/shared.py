import random
from collections import defaultdict
from typing import *

import numpy as np
import tensorflow as tf

from jack.core import TensorPortTensors, TensorPort
from jack.readers.extractive_qa.shared import XQAInputModule, XQAPorts
from jack.readers.extractive_qa.tensorflow.abstract_model import AbstractXQAModelModule
from jack.readers.extractive_qa.tensorflow.answer_layer import answer_layer
from jack.readers.extractive_qa.util import prepare_data
from jack.tfutil.modular_encoder import modular_encoder
from jack.util import preprocessing
from jack.util.map import numpify
from projects.assertion_mr.shared import AssertionMRPorts, AssertionStore
from projects.assertion_mr.tfutil import embedding_refinement

XQAAssertionAnnotation = NamedTuple('XQAAssertionAnnotation', [
    ('question_tokens', List[str]),
    ('question_lemmas', List[str]),
    ('question_ids', List[int]),
    ('question_length', int),
    ('support_tokens', List[str]),
    ('support_lemmas', List[str]),
    ('support_ids', List[int]),
    ('support_length', int),
    ('word_in_question', List[float]),
    ('token_offsets', List[int]),
    ('answer_spans', Optional[List[Tuple[int, int]]]),
    ('selected_supports', List[int]),
])


class XQAAssertionInputModule(XQAInputModule):
    _output_ports = [AssertionMRPorts.question_length, AssertionMRPorts.support_length,
                     # char
                     AssertionMRPorts.word_chars, AssertionMRPorts.word_char_length,
                     AssertionMRPorts.question, AssertionMRPorts.support,
                     # optional, only during training
                     AssertionMRPorts.is_eval,
                     # for assertions
                     AssertionMRPorts.word_embeddings,
                     AssertionMRPorts.assertion_lengths,
                     AssertionMRPorts.assertion2question,
                     AssertionMRPorts.assertions,
                     AssertionMRPorts.question_arg_span,
                     AssertionMRPorts.assertion2question_arg_span,
                     AssertionMRPorts.support_arg_span,
                     AssertionMRPorts.assertion2support_arg_span,
                     AssertionMRPorts.word2lemma,
                     XQAPorts.support2question,
                     # optional, only during training
                     XQAPorts.answer2support_training, XQAPorts.correct_start,
                     # for output module
                     XQAPorts.token_offsets, XQAPorts.selected_support]

    def __init__(self, shared_resources):
        super(XQAAssertionInputModule, self).__init__(shared_resources)
        self.__nlp = preprocessing.spacy_nlp()
        self._rng = random.Random(123)

    def setup(self):
        self._assertion_store = AssertionStore(self.shared_resources.config["assertion_dir"],
                                               self.shared_resources.config["assertion_sources"])
        self._limit = self.shared_resources.config.get("assertion_limit", 10)
        self.vocab = self.shared_resources.vocab
        self.config = self.shared_resources.config
        self.batch_size = self.config.get("batch_size", 1)
        self.dropout = self.config.get("dropout", 0.0)
        self._rng = random.Random(self.config.get("seed", 123))
        self.emb_matrix = self.vocab.emb.lookup
        self.default_vec = np.zeros([self.vocab.emb_length])
        self.char_vocab = self.shared_resources.char_vocab

    @property
    def output_ports(self):
        return self._output_ports

    def preprocess_instance(self, question, answers=None):
        has_answers = answers is not None

        q_tokenized, q_ids, q_lemmas, q_length, s_tokenized, s_ids, s_lemmas, s_length, \
        word_in_question, token_offsets, answer_spans = prepare_data(
            question, answers, self.vocab, self.config.get("lowercase", False),
            with_answers=has_answers, max_support_length=self.config.get("max_support_length", None),
            spacy_nlp=True, with_lemmas=True)

        max_num_support = self.config.get("max_num_support")  # take all per default
        if max_num_support is not None and len(question.support) > max_num_support:
            # subsample by TF-IDF
            q_freqs = defaultdict(float)
            freqs = defaultdict(float)
            for w, i in zip(q_tokenized, q_ids):
                if w.isalnum():
                    q_freqs[i] += 1.0
                    freqs[i] += 1.0
            d_freqs = []
            for i, s in enumerate(s_ids):
                d_freqs.append(defaultdict(float))
                for j in s:
                    freqs[j] += 1.0
                    d_freqs[-1][j] += 1.0
            scores = []
            for i, d_freq in enumerate(d_freqs):
                score = sum(v / freqs[k] * d_freq.get(k, 0.0) / freqs[k] for k, v in q_freqs.items())
                scores.append((i, score))

            selected_supports = [s_idx for s_idx, _ in sorted(scores, key=lambda x: -x[1])[:max_num_support]]
            s_tokenized = [s_tokenized[s_idx] for s_idx in selected_supports]
            s_ids = [s_ids[s_idx] for s_idx in selected_supports]
            s_length = [s_length[s_idx] for s_idx in selected_supports]
            word_in_question = [word_in_question[s_idx] for s_idx in selected_supports]
            token_offsets = [token_offsets[s_idx] for s_idx in selected_supports]
            answer_spans = [answer_spans[s_idx] for s_idx in selected_supports]
        else:
            selected_supports = list(range(len(question.support)))

        return XQAAssertionAnnotation(
            question_tokens=q_tokenized,
            question_lemmas=q_lemmas,
            question_ids=q_ids,
            question_length=q_length,
            support_tokens=s_tokenized,
            support_lemmas=s_lemmas,
            support_ids=s_ids,
            support_length=s_length,
            word_in_question=word_in_question,
            token_offsets=token_offsets,
            answer_spans=answer_spans if has_answers else None,
            selected_supports=selected_supports,
        )

    def create_batch(self, annotations, is_eval: bool, with_answers: bool):
        q_tokenized = [a.question_tokens for a in annotations]
        question_lengths = [a.question_length for a in annotations]

        max_num_support = self.config.get("max_num_support")  # take all per default
        s_tokenized = []
        support_lengths = []
        wiq = []
        offsets = []
        support2question = []
        # aligns with support2question, used in output module to get correct index to original set of supports
        selected_support = []
        for j, a in enumerate(annotations):
            if max_num_support is not None and len(a.support_tokens) > max(1, max_num_support // 2) and not is_eval:
                # always take first (the best) and sample from rest during training, only consider half to speed
                # things up. Following https://arxiv.org/pdf/1710.10723.pdf we sample half during training
                selected = self._rng.sample(range(1, len(a.support_tokens)), max(1, max_num_support // 2) - 1)
                selected = set([0] + selected)
            else:
                selected = set(range(len(a.support_tokens)))
            for s in selected:
                s_tokenized.append(a.support_tokens[s])
                support_lengths.append(a.support_length[s])
                wiq.append(a.word_in_question[s])
                offsets.append(a.token_offsets[s])
                selected_support.append(a.selected_supports[s])
                support2question.append(j)

        word_chars, word_lengths, word_ids, vocab, rev_vocab = \
            preprocessing.unique_words_with_chars(q_tokenized + s_tokenized, self.char_vocab)

        question = word_ids[:len(q_tokenized)]
        support = word_ids[len(q_tokenized):]

        ass_lengths = []
        ass2question = []
        ass2unique = []
        lemma2idx = dict()
        question_arg_span = []
        support_arg_span = []
        assertion2question_arg_span = []
        assertion2support_arg_span = []
        question_arg_span_idx = dict()
        support_arg_span_idx = dict()

        word2lemma = [None] * len(rev_vocab)

        # we have to create batches here and cannot precompute them because of the batch-specific wiq feature
        s_offset = 0
        for i, annot in enumerate(annotations):
            # collect uniq lemmas:
            for k, l in enumerate(annot.question_lemmas):
                if l not in lemma2idx:
                    lemma2idx[l] = len(lemma2idx)
                word2lemma[question[i][k]] = lemma2idx[l]
            for k, ls in enumerate(annot.support_lemmas):
                for k2, l in enumerate(ls):
                    if l not in lemma2idx:
                        lemma2idx[l] = len(lemma2idx)
                    word2lemma[support[s_offset + k][k2]] = lemma2idx[l]

            assertions, assertion_args = self._assertion_store.get_assertion_keys(
                annot.question_lemmas, [l for ls in annot.support_lemmas for l in ls])
            sorted_assertions = sorted(assertions.items(), key=lambda x: -x[1])
            added_assertions = set()
            for key, _ in sorted_assertions:
                if len(added_assertions) == self._limit:
                    break
                a = self.__nlp(self._assertion_store.get_assertion(key))
                a_lemma = " ".join(t.lemma_ for t in a)
                if a_lemma in added_assertions:
                    continue
                else:
                    added_assertions.add(a_lemma)
                ass2question.append(i)
                ass_lengths.append(len(a))
                q_arg_span = assertion_args[key][0]
                q_arg_span = (i, q_arg_span[0], q_arg_span[1])
                s_arg_start, s_arg_end = assertion_args[key][1]
                doc_idx = 0
                for ls in annot.support_lemmas:
                    if s_arg_start < len(ls):
                        break
                    else:
                        doc_idx += 1
                        s_arg_start -= len(ls)
                        s_arg_end -= len(ls)
                s_arg_span = (s_offset + doc_idx, s_arg_start, s_arg_end)
                if q_arg_span not in question_arg_span_idx:
                    question_arg_span_idx[q_arg_span] = len(question_arg_span)
                    question_arg_span.append(assertion_args[key][0])
                if s_arg_span not in support_arg_span_idx:
                    support_arg_span_idx[s_arg_span] = len(support_arg_span)
                    support_arg_span.append(assertion_args[key][1])
                assertion2question_arg_span.append(question_arg_span_idx[q_arg_span])
                assertion2support_arg_span.append(support_arg_span_idx[s_arg_span])

                u_ass = []
                for t in a:
                    w = t.orth_
                    if w not in vocab:
                        vocab[w] = len(vocab)
                        word_lengths.append(min(len(w), 20))
                        word_chars.append([self.char_vocab.get(c, 0) for c in w[:20]])
                        rev_vocab.append(w)
                        if t.lemma_ not in lemma2idx:
                            lemma2idx[t.lemma_] = len(lemma2idx)
                        word2lemma.append(lemma2idx[t.lemma_])
                    u_ass.append(vocab[w])
                ass2unique.append(u_ass)

            s_offset += len(annot.support_lemmas)

        word_embeddings = np.zeros([len(rev_vocab), self.emb_matrix.shape[1]])
        for i, w in enumerate(rev_vocab):
            word_embeddings[i] = self._get_emb(self.vocab(w))

        if not ass2unique:
            ass2unique.append([])
            question_arg_span = support_arg_span = np.zeros([0, 2], dtype=np.int32)

        output = {
            AssertionMRPorts.word_chars: word_chars,
            AssertionMRPorts.word_char_length: word_lengths,
            AssertionMRPorts.question: question,
            AssertionMRPorts.support: support,
            AssertionMRPorts.support_length: support_lengths,
            AssertionMRPorts.question_length: question_lengths,
            AssertionMRPorts.is_eval: is_eval,
            AssertionMRPorts.word_embeddings: word_embeddings,
            AssertionMRPorts.assertion_lengths: ass_lengths,
            AssertionMRPorts.assertion2question: ass2question,
            AssertionMRPorts.assertions: ass2unique,
            AssertionMRPorts.word2lemma: word2lemma,
            AssertionMRPorts.question_arg_span: question_arg_span,
            AssertionMRPorts.support_arg_span: support_arg_span,
            AssertionMRPorts.assertion2question_arg_span: assertion2question_arg_span,
            AssertionMRPorts.assertion2support_arg_span: assertion2support_arg_span,
            XQAPorts.support2question: support2question,
            XQAPorts.token_offsets: offsets,
            XQAPorts.selected_support: selected_support,
            '__vocab': vocab,
            '__rev_vocab': rev_vocab,
            '__lemma_vocab': lemma2idx,
        }

        if with_answers:
            spans = [s for a in annotations for spans_per_support in a.answer_spans for s in spans_per_support]
            span2support = []
            support_idx = 0
            for a in annotations:
                for spans_per_support in a.answer_spans:
                    span2support.extend([support_idx] * len(spans_per_support))
                    support_idx += 1
            output.update({
                XQAPorts.answer_span: [span for span in spans],
                XQAPorts.correct_start: [] if is_eval else [span[0] for span in spans],
                XQAPorts.answer2support_training: span2support,
            })

        # we can only numpify in here, because bucketing is not possible prior
        batch = numpify(output, keys=self.output_ports + self.training_ports)
        return batch


class ModularAssertionQAModel(AbstractXQAModelModule):
    _input_ports = [AssertionMRPorts.question_length, AssertionMRPorts.support_length,
                    # char
                    AssertionMRPorts.word_chars, AssertionMRPorts.word_char_length,
                    AssertionMRPorts.question, AssertionMRPorts.support,
                    # optional, only during training
                    AssertionMRPorts.is_eval,
                    # for assertions
                    AssertionMRPorts.word_embeddings,
                    AssertionMRPorts.assertion_lengths,
                    AssertionMRPorts.assertion2question,
                    AssertionMRPorts.assertions,
                    AssertionMRPorts.word2lemma,
                    XQAPorts.support2question,
                    XQAPorts.correct_start,
                    XQAPorts.answer2support_training]

    @property
    def input_ports(self) -> Sequence[TensorPort]:
        return self._input_ports

    def create_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)

        question_length = tensors.question_length
        support_length = tensors.support_length
        support2question = tensors.support2question
        word_chars = tensors.word_chars
        word_char_length = tensors.word_char_length
        question = tensors.question
        support = tensors.support
        is_eval = tensors.is_eval
        word_embeddings = tensors.word_embeddings
        assertion_lengths = tensors.assertion_lengths
        assertion2question = tensors.assertion2question
        assertions = tensors.assertions
        word2lemma = tensors.word2lemma

        model = shared_resources.config['model']
        reading_encoder_config = shared_resources.config['reading_module']
        repr_dim = shared_resources.config['repr_dim']
        input_size = shared_resources.config["repr_dim_input"]
        size = shared_resources.config["repr_dim"]
        with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)

        word_embeddings.set_shape([None, input_size])

        reading_sequence = [support, question, assertions]
        reading_sequence_lengths = [support_length, question_length, assertion_lengths]
        reading_sequence_2_batch = [support2question, None, assertion2question]

        new_word_embeddings, reading_sequence_offset, _ = embedding_refinement(
            size, word_embeddings, reading_encoder_config,
            reading_sequence, reading_sequence_2_batch, reading_sequence_lengths,
            word2lemma, word_chars, word_char_length, is_eval,
            keep_prob=1.0 - shared_resources.config.get('dropout', 0.0),
            with_char_embeddings=with_char_embeddings, num_chars=len(shared_resources.char_vocab))

        emb_question = tf.nn.embedding_lookup(new_word_embeddings, reading_sequence_offset[1],
                                              name='embedded_question')
        emb_support = tf.nn.embedding_lookup(new_word_embeddings, reading_sequence_offset[0],
                                             name='embedded_support')

        inputs = {'question': emb_question, 'support': emb_support}
        inputs_length = {'question': question_length, 'support': support_length}
        inputs_mapping = {'question': None, 'support': support2question}

        encoder_config = model['encoder_layer']

        encoded, _, _ = modular_encoder(encoder_config, inputs, inputs_length, inputs_mapping, repr_dim, is_eval)

        with tf.variable_scope('answer_layer'):
            answer_layer_config = model['answer_layer']
            encoded_question = encoded[answer_layer_config.get('question', 'question')]
            encoded_support = encoded[answer_layer_config.get('support', 'support')]

            if 'repr_dim' not in answer_layer_config:
                answer_layer_config['repr_dim'] = repr_dim
            start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
                answer_layer(encoded_question, tensors.question_length, encoded_support,
                             tensors.support_length,
                             tensors.support2question, tensors.answer2support, tensors.is_eval,
                             tensors.correct_start, **answer_layer_config)

        span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)

        return TensorPort.to_mapping(self.output_ports, (start_scores, end_scores, span))
