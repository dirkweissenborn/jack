import random
from typing import *

import numpy as np

from jack.readers.extractive_qa.shared import XQAInputModule, XQAPorts, XQAAnnotation
from jack.readers.extractive_qa.util import prepare_data
from jack.util import preprocessing
from jack.util.map import numpify
from projects.assertion_mr.shared import AssertionMRPorts

XQAAssertionAnnotation = NamedTuple('XQAAnnotation', [
    ('question_tokens', List[str]),
    ('question_lemmas', List[str]),
    ('question_ids', List[int]),
    ('question_length', int),
    ('question_embeddings', np.ndarray),
    ('support_tokens', List[str]),
    ('support_lemmas', List[str]),
    ('support_ids', List[int]),
    ('support_length', int),
    ('support_embeddings', np.ndarray),
    ('word_in_question', List[float]),
    ('token_offsets', List[int]),
    ('answer_spans', Optional[List[Tuple[int, int]]]),
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
                     # features
                     XQAPorts.word_in_question,
                     # optional, only during training
                     XQAPorts.correct_start_training, XQAPorts.answer2question_training,
                     XQAPorts.keep_prob, XQAPorts.is_eval,
                     # for output module
                     XQAPorts.token_char_offsets]

    def __init__(self, shared_resources):
        super(XQAAssertionInputModule, self).__init__(shared_resources)
        self.__nlp = preprocessing.spacy_nlp()
        self._rng = random.Random(123)

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

        emb_support = np.zeros([s_length, self.emb_matrix.shape[1]])
        emb_question = np.zeros([q_length, self.emb_matrix.shape[1]])

        for k in range(len(s_ids)):
            emb_support[k] = self._get_emb(s_ids[k])
        for k in range(len(q_ids)):
            emb_question[k] = self._get_emb(q_ids[k])

        return XQAAnnotation(
            question_tokens=q_tokenized,
            question_lemmas=q_lemmas,
            question_ids=q_ids,
            question_length=q_length,
            question_embeddings=emb_question,
            support_tokens=s_tokenized,
            support_lemmas=s_lemmas,
            support_ids=s_ids,
            support_length=s_length,
            support_embeddings=emb_support,
            word_in_question=word_in_question,
            token_offsets=token_offsets,
            answer_spans=answer_spans if has_answers else None,
        )

    def create_batch(self, annotations, is_eval: bool, with_answers: bool):
        support_lengths = list()
        question_lengths = list()

        ass_lengths = []
        ass2question = []
        ass2unique = []
        lemma2idx = dict()
        answer_labels = []
        question_arg_span = []
        support_arg_span = []
        assertion2question_arg_span = []
        assertion2support_arg_span = []

        question_arg_span_idx = dict()
        support_arg_span_idx = dict()

        word_chars, word_lengths, question2unique, support2unique, vocab, rev_vocab = \
            preprocessing.unique_words_with_chars([a.question_tokens for a in annotations],
                                                  [a.support_tokens for a in annotations], self.char_vocab)

        word2lemma = [None] * len(rev_vocab)

        # we have to create batches here and cannot precompute them because of the batch-specific wiq feature
        for i, annot in enumerate(annotations):
            support_lengths.append(annot.support_length)
            question_lengths.append(annot.question_length)

            # collect uniq lemmas:
            for k, l in enumerate(annot.question_lemmas):
                if l not in lemma2idx:
                    lemma2idx[l] = len(lemma2idx)
                word2lemma[question2unique[i][k]] = lemma2idx[l]
            for k, l in enumerate(annot.support_lemmas):
                if l not in lemma2idx:
                    lemma2idx[l] = len(lemma2idx)
                word2lemma[support2unique[i][k]] = lemma2idx[l]

            assertions, assertion_args = self._assertion_store.get_assertion_keys(
                annot.question_lemmas, annot.support_lemmas)
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
                s_arg_span = assertion_args[key][1]
                s_arg_span = (i, s_arg_span[0], s_arg_span[1])
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

        word_embeddings = np.zeros([len(rev_vocab), self.emb_matrix.shape[1]])
        for i, w in enumerate(rev_vocab):
            word_embeddings[i] = self._get_emb(self.vocab(w))

        if not ass2unique:
            ass2unique.append([])
            question_arg_span = support_arg_span = np.zeros([0, 2], dtype=np.int32)

        output = {
            AssertionMRPorts.word_chars: word_chars,
            AssertionMRPorts.word_char_length: word_lengths,
            AssertionMRPorts.question: question2unique,
            AssertionMRPorts.support: support2unique,
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
            '__vocab': vocab,
            '__rev_vocab': rev_vocab,
            '__lemma_vocab': lemma2idx,
        }

        if with_answers:
            spans = [a.answer_spans for a in annotations]
            span2question = [i for i in range(len(annot)) for _ in spans[i]]
            output.update({
                XQAPorts.answer_span: [span for span_list in spans for span in span_list],
                XQAPorts.correct_start_training: [] if is_eval else [span[0] for span_list in spans for span in
                                                                     span_list],
                XQAPorts.answer2question_training: span2question,
            })

        return numpify(output, keys=self.output_ports + self.training_ports)
