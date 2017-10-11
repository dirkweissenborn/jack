"""
This file contains reusable modules for extractive QA models and ports
"""

from jack.core import *


class ParameterTensorPorts:
        # remove?
        keep_prob = TensorPortWithDefault(1.0, tf.float32, [], "keep_prob",
                                          "scalar representing keep probability when using dropout",
                                          "[]")

        is_eval = TensorPortWithDefault(True, tf.bool, [], "is_eval",
                                        "boolean that determines whether input is eval or training.",
                                        "[]")

class XQAPorts:
    # When feeding embeddings directly
    emb_question = FlatPorts.Misc.embedded_question
    question_length = FlatPorts.Input.question_length
    emb_support = FlatPorts.Misc.embedded_support
    support_length = FlatPorts.Input.support_length

    # but also ids, for char-based embeddings
    unique_word_chars = TensorPort(tf.int32, [None, None], "question_chars",
                                   "Represents questions using symbol vectors",
                                   "[U, max_num_chars]")
    unique_word_char_length = TensorPort(tf.int32, [None], "question_char_length",
                                         "Represents questions using symbol vectors",
                                         "[U]")
    question_words2unique = TensorPort(tf.int32, [None, None], "question_words2unique",
                                       "Represents support using symbol vectors",
                                       "[batch_size, max_num_question_tokens]")
    support_words2unique = TensorPort(tf.int32, [None, None], "support_words2unique",
                                      "Represents support using symbol vectors",
                                      "[batch_size, max_num_support_tokens, max]")

    keep_prob = ParameterTensorPorts.keep_prob
    is_eval = ParameterTensorPorts.is_eval

    # This feature is model specific and thus, not part of the conventional Ports
    word_in_question = TensorPort(tf.float32, [None, None], "word_in_question_feature",
                                  "Represents a 1/0 feature for all context tokens denoting"
                                  " whether it is part of the question or not",
                                  "[Q, support_length]")

    correct_start_training = TensorPortWithDefault(np.array([0], np.int32), tf.int32, [None], "correct_start_training",
                                                   "Represents the correct start of the span which is given to the"
                                                   "model during training for use to predicting end.",
                                                   "[A]")

    answer2question_training = TensorPortWithDefault([0], tf.int32, [None], "answer2question_training",
                                                     "Represents mapping to question idx per answer, which is used "
                                                     "together with correct_start_training during training.",
                                                     "[A]")

    # output ports
    start_scores = FlatPorts.Prediction.start_scores
    end_scores = FlatPorts.Prediction.end_scores
    span_prediction = FlatPorts.Prediction.answer_span
    token_offsets = TensorPort(tf.int32, [None, None, 2], "token_offsets",
                               "Document and character index of tokens in support.",
                               "[S, support_length, 2]")

    # ports used during training
    answer2question = FlatPorts.Input.answer2question
    answer_span = FlatPorts.Target.answer_span




def _np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_answer_and_span(question, support_length, span_prediction, token_offsets):
    start, end = span_prediction
    document_start, char_start = token_offsets[start]
    if end + 1 < support_length:
        document_end, char_end = token_offsets[end + 1]
    else:
        document_end, char_end = len(question.support) - 1, len(question.support[-1])

    if document_start != document_end:
        # Answer data structure does not support answers across multiple
        # supports.
        document_end, char_end = document_start, len(question.support[document_start])

    answer = question.support[document_start][char_start: char_end]

    answer = answer.rstrip()
    char_end = char_start + len(answer)

    return answer, document_start, (char_start, char_end)


class XQAOutputModule(OutputModule):
    def __init__(self, shared_vocab_confg: SharedResources):
        self.vocab = shared_vocab_confg.vocab
        self.setup()

    def __call__(self, questions, span_prediction, token_offsets, support_length, start_scores, end_scores) -> List[Answer]:
        answers = []
        for i, q in enumerate(questions):
            start, end = span_prediction[i]
            answer, doc_idx, span = get_answer_and_span(q, support_length[i],
                                                        span_prediction[i],
                                                        token_offsets[i])

            start_probs = _np_softmax(start_scores[i])
            end_probs = _np_softmax(end_scores[i])

            answers.append(Answer(answer, span=span, doc_idx=doc_idx,
                                  score=start_probs[start] * end_probs[end]))

        return answers

    @property
    def input_ports(self) -> List[TensorPort]:
        return [FlatPorts.Prediction.answer_span, XQAPorts.token_offsets,
                XQAPorts.support_length,
                FlatPorts.Prediction.start_scores, FlatPorts.Prediction.end_scores]


class XQANoScoreOutputModule(OutputModule):
    def __init__(self, shared_vocab_confg: SharedResources):
        self.vocab = shared_vocab_confg.vocab
        self.setup()

    def __call__(self, questions, span_prediction, token_offsets,
                 support_length) -> List[Answer]:
        answers = []
        for i, q in enumerate(questions):
            start, end = span_prediction[i]
            answer, doc_idx, span = get_answer_and_span(q, support_length[i],
                                                        span_prediction[i],
                                                        token_offsets[i])


            answers.append(Answer(answer, span=span, doc_idx=doc_idx, score=1.0))

        return answers

    @property
    def input_ports(self) -> List[TensorPort]:
        return [FlatPorts.Prediction.answer_span, XQAPorts.token_offsets,
                XQAPorts.support_length]
