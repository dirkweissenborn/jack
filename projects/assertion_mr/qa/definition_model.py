import numpy as np
import tensorflow as tf

from jack.core import TensorPortWithDefault, OnlineInputModule, TensorPortTensors, TensorPort
from jack.core.tensorflow import TFReader
from jack.readers.extractive_qa.shared import XQAPorts
from jack.readers.extractive_qa.tensorflow.abstract_model import AbstractXQAModelModule
from jack.readers.extractive_qa.tensorflow.answer_layer import answer_layer
from jack.tfutil.modular_encoder import modular_encoder
from jack.util.map import numpify
from jack.util.preprocessing import sort_by_tfidf
from projects.assertion_mr.qa.shared import XQAAssertionInputModule
from projects.assertion_mr.shared import AssertionMRPorts
from projects.assertion_mr.tfutil import word_with_char_embed, embedding_refinement


class DefinitionPorts:
    definition_lengths = TensorPortWithDefault(np.zeros([0], np.int32), tf.int32, [None],
                                               "definition_lengths", "Length of definition.", "[R]")
    definitions = TensorPortWithDefault(np.zeros([0, 0], np.int32),
                                        tf.int32, [None, None], "definitions",
                                        "Represents batch dependent definition word ids.", "[R, L]")
    definition2question = TensorPortWithDefault(np.zeros([0], np.int32), tf.int32, [None], "definition2question",
                                                "Question idx per definition", "[R]")


class XQAAssertionDefinitionInputModule(OnlineInputModule):
    def __init__(self, reader: TFReader):
        self.reader = reader
        self._underlying_input_module = reader.input_module
        assert isinstance(self._underlying_input_module, XQAAssertionInputModule)
        super(XQAAssertionDefinitionInputModule, self).__init__(reader.shared_resources)

    @property
    def output_ports(self):
        return self._underlying_input_module.output_ports + [
            DefinitionPorts.definitions, DefinitionPorts.definition_lengths, DefinitionPorts.definition2question]

    @property
    def training_ports(self):
        return self._underlying_input_module.training_ports

    def setup(self):
        self._underlying_input_module.setup()

    def preprocess(self, questions, answers=None, is_eval: bool = False):
        return self._underlying_input_module.preprocess(questions, answers, is_eval)

    def preprocess_instance(self, question, answers=None):
        return self._underlying_input_module.preprocess_instance(question, answers)

    def setup_from_data(self, data):
        self._underlying_input_module.setup_from_data(data)

    def create_batch(self, annotations, is_eval, with_answers):
        batch = self._underlying_input_module.create_batch(annotations, True, with_answers)
        lemma_vocab = batch['__lemma_vocab']
        vocab = batch['__vocab']
        rev_vocab = batch['__rev_vocab']
        word_chars = batch[AssertionMRPorts.word_chars].tolist()
        word_lengths = batch[AssertionMRPorts.word_char_length].tolist()
        word2lemma = batch[AssertionMRPorts.word2lemma].tolist()
        support = batch[AssertionMRPorts.support]

        rev_lemma_vocab = {v: k for k, v in lemma_vocab.items()}
        beam_size = self._underlying_input_module.config['beam_size']
        self.reader.model_module.set_beam_size(beam_size)
        out = self.reader.model_module(batch, self.reader.output_module.input_ports)

        spans = out[XQAPorts.span_prediction]

        definitions = []
        definition_lengths = []
        definition2question = []

        seen_answer_lemmas = None
        for i, s in enumerate(spans):
            j = i // beam_size
            if i % beam_size == 0:
                seen_answer_lemmas = set()
            doc_idx_map = [i for i, q_id in enumerate(batch[XQAPorts.support2question]) if q_id == j]
            doc_idx, start, end = s[0], s[1], s[2]
            answer_token_ids = support[doc_idx_map[doc_idx], start:end + 1]
            answer_lemma = ' '.join(rev_lemma_vocab[word2lemma[idd]] for idd in answer_token_ids)
            if answer_lemma in seen_answer_lemmas:
                continue
            seen_answer_lemmas.add(answer_lemma)
            ks = self._underlying_input_module._assertion_store.assertion_keys_for_subject(
                answer_lemma, resource='wikipedia_firstsent')
            defns = []
            for key in ks:
                defns.append(self._underlying_input_module._assertion_store.get_assertion(key))
            if defns:
                if len(defns) > 1:
                    indices_scores = sort_by_tfidf(' '.join(annotations[j].support_tokens[doc_idx]), defns)
                    # only select definition with best match to the support
                    defn = defns[indices_scores[0][0]]
                else:
                    defn = defns[0]
                defn = self._underlying_input_module._nlp(defn)
                definition_lengths.append(len(defn))
                definition2question.append(j)
                defn_ids = []
                for t in defn:
                    w = t.orth_
                    if w not in vocab:
                        vocab[w] = len(vocab)
                        word_lengths.append(min(len(w), 20))
                        word_chars.append([self._underlying_input_module.char_vocab.get(c, 0) for c in w[:20]])
                        rev_vocab.append(w)
                        if t.lemma_ not in lemma_vocab:
                            lemma_vocab[t.lemma_] = len(lemma_vocab)
                        word2lemma.append(lemma_vocab[t.lemma_])
                    defn_ids.append(vocab[w])
                definitions.append(defn_ids)

        batch[DefinitionPorts.definitions] = definitions
        batch[DefinitionPorts.definition_lengths] = definition_lengths
        batch[DefinitionPorts.definition2question] = definition2question
        batch[AssertionMRPorts.word_chars] = word_chars
        batch[AssertionMRPorts.word_char_length] = word_lengths
        batch[AssertionMRPorts.word2lemma] = word2lemma
        batch[AssertionMRPorts.is_eval] = is_eval

        word_embeddings = np.zeros([len(rev_vocab), self._underlying_input_module.emb_matrix.shape[1]])
        for i, w in enumerate(rev_vocab):
            word_embeddings[i] = self._underlying_input_module._get_emb(self._underlying_input_module.vocab(w))

        batch[AssertionMRPorts.word_embeddings] = word_embeddings

        return numpify(batch, keys=[
            DefinitionPorts.definitions, DefinitionPorts.definition_lengths, DefinitionPorts.definition2question,
            AssertionMRPorts.word_chars, AssertionMRPorts.word_char_length, AssertionMRPorts.word2lemma])


class ModularAssertionDefinitionQAModel(AbstractXQAModelModule):
    def __init__(self, wrapped_model):
        self._wrapped_model = wrapped_model
        super(ModularAssertionDefinitionQAModel, self).__init__(wrapped_model.shared_resources,
                                                                wrapped_model.tf_session)

    def setup(self, is_training=True, reuse=False):
        self._wrapped_model.setup(is_training, reuse)
        super().setup(is_training, reuse=True)

    @property
    def input_ports(self):
        return self._wrapped_model.input_ports + [
            DefinitionPorts.definitions, DefinitionPorts.definition_lengths, DefinitionPorts.definition2question]

    def set_beam_size(self, k):
        self._beam_size_assign(k)

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
        definition_lengths = tensors.definition_lengths
        definition2question = tensors.definition2question
        definitions = tensors.definitions
        word2lemma = tensors.word2lemma

        model = shared_resources.config['model']
        repr_dim = shared_resources.config['repr_dim']
        input_size = shared_resources.config["repr_dim_input"]
        size = shared_resources.config["repr_dim"]
        with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)

        word_embeddings.set_shape([None, input_size])

        if shared_resources.config.get('no_reading', False):
            word_embeddings = tf.layers.dense(word_embeddings, size, activation=tf.nn.relu,
                                              name="embeddings_projection")
            new_word_embeddings = word_with_char_embed(
                size, word_embeddings, tensors.word_chars, tensors.word_char_length,
                len(shared_resources.char_vocab), tensors.is_eval,
                keep_prob=1.0 - shared_resources.config.get('dropout', 0.0))
            reading_sequence_offset = [support, question, assertions]
        else:
            if shared_resources.config.get("assertion_limit", 0) > 0:
                reading_sequence = [support, question, assertions, definitions]
                reading_sequence_lengths = [support_length, question_length, assertion_lengths, definition_lengths]
                reading_sequence_2_batch = [support2question, None, assertion2question, definition2question]
            else:
                reading_sequence = [support, question, definitions]
                reading_sequence_lengths = [support_length, question_length, definition_lengths]
                reading_sequence_2_batch = [support2question, None, definition2question]

            reading_encoder_config = shared_resources.config['reading_module']
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

        inputs = {'question': emb_question, 'support': emb_support,
                  'word_in_question': tf.expand_dims(tensors.word_in_question, 2)}
        inputs_length = {'question': question_length, 'support': support_length,
                         'word_in_question': support_length}
        inputs_mapping = {'question': None, 'support': support2question}

        encoder_config = model['encoder_layer']

        encoded, _, _ = modular_encoder(encoder_config, inputs, inputs_length, inputs_mapping, repr_dim, is_eval)

        with tf.variable_scope('answer_layer'):
            answer_layer_config = model['answer_layer']
            encoded_question = encoded[answer_layer_config.get('question', 'question')]
            encoded_support = encoded[answer_layer_config.get('support', 'support')]

            if 'repr_dim' not in answer_layer_config:
                answer_layer_config['repr_dim'] = repr_dim
            if 'max_span_size' not in answer_layer_config:
                answer_layer_config['max_span_size'] = shared_resources.config.get('max_span_size', 16)

            beam_size = tf.get_variable(
                'beam_size', initializer=shared_resources.config.get('beam_size', 1), dtype=tf.int32, trainable=False)
            beam_size_p = tf.placeholder(tf.int32, [], 'beam_size_setter')
            beam_size_assign = beam_size.assign(beam_size_p)
            self._beam_size_assign = lambda k: self.tf_session.run(beam_size_assign, {beam_size_p: k})

            start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
                answer_layer(encoded_question, tensors.question_length, encoded_support,
                             tensors.support_length,
                             tensors.support2question, tensors.answer2support, tensors.is_eval,
                             tensors.correct_start, beam_size=beam_size, **answer_layer_config)

        span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)

        return TensorPort.to_mapping(self.output_ports, (start_scores, end_scores, span))
