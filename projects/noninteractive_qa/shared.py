"""
This file contains reusable modules for extractive QA models and ports
"""
import sys
from collections import defaultdict
from typing import NamedTuple

from jack.core import *
from jack.readers.extractive_qa.shared import XQAPorts, XQAInputModule, XQAOutputModule
from jack.readers.extractive_qa.util import prepare_data
from jack.util import preprocessing
from jack.util.map import numpify
from jack.util.preprocessing import sort_by_tfidf

logger = logging.getLogger(__name__)

question_lemmas = TensorPort(np.int32, [None, None], "question_lemmas", "Lemma idx per word", "[U]")
support_lemmas = TensorPort(np.int32, [None, None], "support_lemmas", "Lemma idx per word", "[U]")

XQAAnnotationWithLemma = NamedTuple('XQAAnnotationWithLemma', [
    ('question_tokens', List[str]),
    ('question_lemmas', List[str]),
    ('question_ids', List[int]),
    ('question_length', int),
    ('support_tokens', List[List[str]]),
    ('support_lemmas', List[List[str]]),
    ('support_ids', List[List[int]]),
    ('support_length', List[int]),
    ('word_in_question', List[List[float]]),
    ('token_offsets', List[List[int]]),
    ('answer_spans', Optional[List[Tuple[int, int]]]),
    ('selected_supports', List[int]),
])


class XQAInputModuleWithLemma(XQAInputModule, OnlineInputModule[XQAAnnotationWithLemma]):
    _output_ports = [XQAPorts.emb_question, XQAPorts.question_length,
                     XQAPorts.emb_support, XQAPorts.support_length,
                     XQAPorts.support2question,
                     # char
                     XQAPorts.word_chars, XQAPorts.word_char_length,
                     XQAPorts.question_words, XQAPorts.support_words,
                     question_lemmas, support_lemmas,
                     # features
                     XQAPorts.word_in_question,
                     # optional, only during training
                     XQAPorts.correct_start, XQAPorts.answer2support_training,
                     XQAPorts.is_eval,
                     # for output module
                     XQAPorts.token_offsets, XQAPorts.selected_support]

    @property
    def output_ports(self) -> List[TensorPort]:
        return self._output_ports

    def preprocess_instance(self, question: QASetting,
                            answers: Optional[List[Answer]] = None) -> XQAAnnotationWithLemma:
        has_answers = answers is not None

        q_tokenized, q_ids, q_lemma, q_length, s_tokenized, s_ids, s_lemma, s_length, \
        word_in_question, token_offsets, answer_spans = prepare_data(
            question, answers, self.vocab, self.config.get("lowercase", False),
            with_answers=has_answers, max_support_length=self.config.get("max_support_length", None),
            spacy_nlp=True, with_lemmas=True)

        max_num_support = self.config.get("max_num_support", len(question.support))  # take all per default

        # take max supports by TF-IDF (we subsample to max_num_support in create batch)
        # following https://arxiv.org/pdf/1710.10723.pdf
        if len(question.support) > 1:
            scores = sort_by_tfidf(' '.join(q_tokenized), [' '.join(s) for s in s_tokenized])
            selected_supports = [s_idx for s_idx, _ in scores[:max_num_support]]
            s_tokenized = [s_tokenized[s_idx] for s_idx in selected_supports]
            s_ids = [s_ids[s_idx] for s_idx in selected_supports]
            s_length = [s_length[s_idx] for s_idx in selected_supports]
            word_in_question = [word_in_question[s_idx] for s_idx in selected_supports]
            token_offsets = [token_offsets[s_idx] for s_idx in selected_supports]
            answer_spans = [answer_spans[s_idx] for s_idx in selected_supports]
            s_lemma = [s_lemma[s_idx] for s_idx in selected_supports]
        else:
            selected_supports = list(range(len(question.support)))

        return XQAAnnotationWithLemma(
            question_tokens=q_tokenized,
            question_ids=q_ids,
            question_length=q_length,
            support_tokens=s_tokenized,
            support_ids=s_ids,
            support_length=s_length,
            word_in_question=word_in_question,
            token_offsets=token_offsets,
            answer_spans=answer_spans if has_answers else None,
            selected_supports=selected_supports,
            question_lemmas=q_lemma,
            support_lemmas=s_lemma
        )

    def create_batch(self, annotations: List[XQAAnnotationWithLemma], is_eval: bool, with_answers: bool) \
            -> Mapping[TensorPort, np.ndarray]:
        batch = super(XQAInputModuleWithLemma, self).create_batch(annotations, is_eval, with_answers)
        s_lemmas = []
        q_lemmas = []
        lemma_vocab = defaultdict(lambda: len(lemma_vocab))
        lemma_vocab['PAD'] = 0
        for i, a in enumerate(annotations):
            q_lemmas.append([lemma_vocab[l] for l in a.question_lemmas])
        for selected, i in zip(batch[XQAPorts.selected_support], batch[XQAPorts.support2question]):
            a = annotations[i]
            s = a.selected_supports.index(selected)
            s_lemmas.append([lemma_vocab[l] for l in a.support_lemmas[s]])

        batch[support_lemmas] = s_lemmas
        batch[question_lemmas] = q_lemmas
        # we can only numpify in here, because bucketing is not possible prior
        batch = numpify(batch, keys=[question_lemmas, support_lemmas])
        return batch


class XQAOutputFilterModule(XQAOutputModule):
    def __call__(self, questions, span_prediction,
                 token_offsets, selected_support, support2question,
                 start_scores, end_scores) -> Sequence[Sequence[Answer]]:
        """Produces top-k answers for each question."""
        answers = super(XQAOutputFilterModule, self).__call__(
            questions, span_prediction,
            token_offsets, selected_support, support2question,
            start_scores, end_scores)

        for i, (ans, q) in enumerate(zip(answers, questions)):
            new_answers = []
            q_lower = q.question.lower()
            for a in ans:
                a_t = a.text.lower().split()
                good = True
                max_span = len(a_t) // 2 + 1
                for s in range(len(a_t) - max_span + 1):
                    if ' '.join(a_t[s:s + max_span]) in q_lower:
                        good = False
                        break
                if good:
                    new_answers.append(a)
            if new_answers:
                answers[i] = new_answers
        return answers
