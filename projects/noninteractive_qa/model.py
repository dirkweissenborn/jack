"""
This file contains FastQA specific modules and ports
"""

from jack.core import *
from jack.readers.extractive_qa.tensorflow.abstract_model import AbstractXQAModelModule
from jack.readers.extractive_qa.tensorflow.answer_layer import compute_question_state, compute_spans
from jack.tfutil import misc
from jack.tfutil.embedding import conv_char_embedding
from jack.tfutil.highway import highway_network
from jack.tfutil.modular_encoder import modular_encoder
from projects.noninteractive_qa.mem_net import bidirectional_associative_mem_net

support_state_start_port = TensorPort(tf.float32, [None, None, None], 'support_states_start')
support_state_end_port = TensorPort(tf.float32, [None, None, None], 'support_states_end')
question_state_start_port = TensorPort(tf.float32, [None, None], 'question_state_start')
question_state_end_port = TensorPort(tf.float32, [None, None], 'question_state_end')


class NonInteractiveModularQAModule(AbstractXQAModelModule):
    # @property
    # def training_input_ports(self) -> Sequence[TensorPort]:
    #    return super().training_input_ports

    # @property
    # def output_ports(self) -> Sequence[TensorPort]:
    #    return super().output_ports + [question_state_start_port, question_state_end_port]

    def create_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)

        input_size = shared_resources.config["repr_dim_input"]
        repr_dim = shared_resources.config["repr_dim"]
        with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)
        dropout = shared_resources.config.get("dropout", 0.0)

        # set shapes for inputs
        tensors.emb_question.set_shape([None, None, input_size])
        tensors.emb_support.set_shape([None, None, input_size])

        emb_question = tensors.emb_question
        emb_support = tensors.emb_support
        if with_char_embeddings:
            with tf.variable_scope("char_embeddings") as vs:
                # compute combined embeddings
                [char_emb_question, char_emb_support] = conv_char_embedding(
                    len(shared_resources.char_vocab), repr_dim, tensors.word_chars, tensors.word_char_length,
                    [tensors.question_words, tensors.support_words])

                emb_question = tf.concat([emb_question, char_emb_question], 2)
                emb_support = tf.concat([emb_support, char_emb_support], 2)
                input_size += repr_dim

                # set shapes for inputs
                emb_question.set_shape([None, None, input_size])
                emb_support.set_shape([None, None, input_size])

                emb_question = tf.layers.dense(emb_question, repr_dim, name="embeddings_projection")
                emb_question = highway_network(emb_question, 1)
                vs.reuse_variables()
                emb_support = tf.layers.dense(emb_support, repr_dim, name="embeddings_projection")
                emb_support = highway_network(emb_support, 1)

        with tf.variable_scope("encoder") as vs:
            encoded_question = modular_encoder(
                shared_resources.config['encoder'],
                {'text': emb_question}, {'text': tensors.question_length}, {},
                repr_dim, dropout, tensors.is_eval)[0]['text']
            vs.reuse_variables()
            encoded_support = modular_encoder(
                shared_resources.config['encoder'],
                {'text': emb_support}, {'text': tensors.support_length}, {},
                repr_dim, dropout, tensors.is_eval)[0]['text']

        start_scores, end_scores, span = _simple_answer_layer(
            encoded_question, encoded_support, repr_dim, shared_resources, tensors)

        return TensorPort.to_mapping(self.output_ports, (start_scores, end_scores, span))


class BidirectionalAssociativeMemoryQAModule(AbstractXQAModelModule):
    # @property
    # def training_input_ports(self) -> Sequence[TensorPort]:
    #    return super().training_input_ports

    # @property
    # def output_ports(self) -> Sequence[TensorPort]:
    #    return super().output_ports + [question_state_start_port, question_state_end_port]

    def create_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)

        input_size = shared_resources.config["repr_dim_input"]
        repr_dim = shared_resources.config["repr_dim"]
        with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)
        dropout = shared_resources.config.get("dropout", 0.0)

        # set shapes for inputs
        tensors.emb_question.set_shape([None, None, input_size])
        tensors.emb_support.set_shape([None, None, input_size])

        emb_question = tensors.emb_question
        emb_support = tensors.emb_support
        if with_char_embeddings:
            with tf.variable_scope("char_embeddings") as vs:
                # compute combined embeddings
                [char_emb_question, char_emb_support] = conv_char_embedding(
                    len(shared_resources.char_vocab), repr_dim, tensors.word_chars, tensors.word_char_length,
                    [tensors.question_words, tensors.support_words])

                emb_question = tf.concat([emb_question, char_emb_question], 2)
                emb_support = tf.concat([emb_support, char_emb_support], 2)
                input_size += repr_dim

                # set shapes for inputs
                emb_question.set_shape([None, None, input_size])
                emb_support.set_shape([None, None, input_size])

                emb_question = tf.layers.dense(emb_question, repr_dim, name="embeddings_projection")
                emb_question = highway_network(emb_question, 1)
                vs.reuse_variables()
                emb_support = tf.layers.dense(emb_support, repr_dim, name="embeddings_projection")
                emb_support = highway_network(emb_support, 1)

        with tf.variable_scope("encoder") as vs:
            encoded_question = modular_encoder(
                shared_resources.config['encoder'],
                {'text': emb_question}, {'text': tensors.question_length}, {},
                repr_dim, dropout, tensors.is_eval)[0]['text']
            vs.reuse_variables()
            encoded_support = modular_encoder(
                shared_resources.config['encoder'],
                {'text': emb_support}, {'text': tensors.support_length}, {},
                repr_dim, dropout, tensors.is_eval)[0]['text']

        with tf.variable_scope("memory"):
            memory_conf = shared_resources.config['memory']
            question_memory = bidirectional_associative_mem_net(
                emb_question, tensors.question_length,
                memory_conf['key_dim'],
                memory_conf['num_slots'],
                memory_conf['slot_dim'],
                memory_conf['controller'],
                tensors.is_eval)[0]
        with tf.variable_scope("memory", reuse=True):
            support_memory = bidirectional_associative_mem_net(
                emb_support, tensors.support_length,
                memory_conf['key_dim'],
                memory_conf['num_slots'],
                memory_conf['slot_dim'],
                memory_conf['controller'],
                tensors.is_eval)[0]

        encoded_question = tf.concat([encoded_question, question_memory], axis=2)
        encoded_support = tf.concat([encoded_support, support_memory], axis=2)

        start_scores, end_scores, span = _simple_answer_layer(
            encoded_question, encoded_support, repr_dim, shared_resources, tensors)

        return TensorPort.to_mapping(self.output_ports, (start_scores, end_scores, span))


def _simple_answer_layer(encoded_question, encoded_support, repr_dim, shared_resources, tensors):
    # computing single time attention over question
    question_state = compute_question_state(encoded_question, tensors.question_length)
    question_state = tf.gather(question_state, tensors.support2question)
    question_state = tf.layers.dense(question_state, 2 * repr_dim, tf.nn.relu, name="hidden")
    question_state_start, question_state_end = tf.split(question_state, 2, 1)
    tf.get_variable_scope().reuse_variables()
    hidden = tf.layers.dense(encoded_support, 2 * repr_dim, tf.nn.relu, name="hidden")
    hidden_start, hidden_end = tf.split(hidden, 2, 2)
    support_mask = misc.mask_for_lengths(tensors.support_length)
    start_scores = tf.einsum('ik,ijk->ij', question_state_start, hidden_start)
    start_scores = start_scores + support_mask
    end_scores = tf.einsum('ik,ijk->ij', question_state_end, hidden_end)
    end_scores = end_scores + support_mask
    start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
        compute_spans(start_scores, end_scores, tensors.answer2support, tensors.is_eval, tensors.support2question,
                      max_span_size=shared_resources.config.get('max_span_size', 16))
    span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)
    return start_scores, end_scores, span
