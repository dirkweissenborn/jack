import tensorflow as tf
import torch

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from jack.core import QASetting
from jack.readers import *


@TokenEmbedder.register("jack_embedder")
class JackEmbedder(TokenEmbedder):
    def __init__(self, reader_path: str, vocab: Vocabulary) -> None:
        super(JackEmbedder, self).__init__()
        self.jack_reader = reader_from_file(reader_path)
        self.vocab = vocab
        g = tf.get_default_graph()
        self.question_op = [op.outputs[0] for op in g.get_operations() if op.name.endswith('question_representation')][
            0]

        tensors = self.jack_reader.input_module([QASetting('.', [''])])
        encoded = self.jack_reader.session.run(self.question_op,
                                               feed_dict=self.jack_reader.model_module.convert_to_feed_dict(tensors))
        self.output_dim = encoded.shape[-1]

    def get_output_dim(self) -> int:
        """
        Returns the final output dimension that this ``TokenEmbedder`` uses to represent each
        token.  This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        return self.output_dim

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'TokenEmbedder':
        reader_path = params.pop('reader_path')
        return cls(reader_path, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        texts = []
        for i in range(tokens.shape[0]):
            texts.append([])
            for j in range(tokens.shape[1]):
                if tokens.data[i, j] == 0:
                    break
                texts[-1].append(self.vocab.get_index_to_token_vocabulary("tokens").get(tokens[i, j], 'UNK'))
            texts[-1] = QASetting(' '.join(texts[-1]), [''])

        tensors = self.jack_reader.input_module(texts)
        encoded = self.jack_reader.session.run(self.question_op,
                                               feed_dict=self.jack_reader.model_module.convert_to_feed_dict(tensors))

        return torch.autograd.Variable(torch.Tensor(encoded))
