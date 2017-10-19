import torch
from torch import nn
from torch.nn import functional as F

from jack.torch_util import embedding
from jack.torch_util import misc
from jack.torch_util import rnn
from jack.torch_util.rnn import BiLSTM


class ReadingModule(nn.Module):
    def __init__(self, shared_resources, num_sequences):
        super(ReadingModule, self).__init__()
        self._with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)
        input_size = shared_resources.config["repr_dim_input"]
        size = shared_resources.config["repr_dim"]
        self._size = size
        self._dropout = shared_resources.config["dropout"]
        self._num_sequences = num_sequences

        # modules
        self._embedding_projection = nn.Linear(input_size, size)
        if self._with_char_embeddings:
            self._conv_char_embedding = embedding.ConvCharEmbeddingModule(
                len(shared_resources.char_vocab), size)
            self._char_embedding_gate = nn.Linear(2 * size, size)
            self._char_embedding_gate.bias.data.fill_(1.0)

        self._bilstm = rnn.BiLSTM(size + num_sequences, size)

        self._encoded_projection = nn.Linear(2 * size, size)
        self._gate_projection = nn.Linear(2 * size, size)
        self._gate_projection.bias.data.fill_(1.0)

    def forward(self, word_embeddings, reading_sequence, reading_sequence_2_batch, reading_sequence_lengths,
                unique_word2unique_lemma, unique_word_chars=None, unique_word_char_length=None,
                is_eval=False, sequence_indices=None):
        # helpers
        prefix = torch.cuda if word_embeddings.is_cuda else torch
        batch_size = max(s.size(0) if s2b is None else s2b.max().data[0] + 1
                         for s, s2b in zip(reading_sequence, reading_sequence_2_batch))

        sequence_indices = sequence_indices if sequence_indices is not None else list(range(len(reading_sequence)))

        word_embeddings = F.relu(self._embedding_projection(word_embeddings))
        if self._with_char_embeddings:
            # compute combined embeddings
            char_word_embeddings = self._conv_char_embedding(unique_word_chars, unique_word_char_length)
            char_word_embeddings = F.relu(char_word_embeddings)

            gate = F.sigmoid(self._char_embedding_gate(torch.cat([word_embeddings, char_word_embeddings], 1)))
            word_embeddings = word_embeddings * gate + (1.0 - gate) * char_word_embeddings

        if not is_eval:
            word_embeddings = F.dropout(word_embeddings, self._dropout)

        num_words = unique_word_char_length.size(0)

        # divide uniq words for each question by offsets
        offsets = torch.autograd.Variable(
            torch.arange(0, num_words * batch_size, num_words, out=prefix.LongTensor())).unsqueeze(1)

        # each token is assigned a word idx + offset for distinguishing words between batch instances
        reading_sequence_offset = [
            s.long() + offsets if s2b is None else
            s.long() + torch.index_select(offsets, 0, s2b.long())
            for s, s2b in zip(reading_sequence, reading_sequence_2_batch)]

        unique2lemma_off = unique_word2unique_lemma.long().view(1, -1).expand(batch_size, num_words) + offsets
        unique2lemma_off = unique2lemma_off.view(-1)
        num_lemmas = unique2lemma_off.max().data[0] + 1

        # tile word_embeddings by batch size (individual batches update embeddings individually)
        ctxt_word_embeddings = word_embeddings.repeat(batch_size, 1)

        for i, seq, length in zip(sequence_indices, reading_sequence_offset, reading_sequence_lengths):
            batch_size = length.size(0)
            max_length = length.max().data[0]

            if batch_size > 0:
                encoded = F.embedding(seq, ctxt_word_embeddings)
                one_hot = torch.autograd.Variable(torch.zeros(1, 1, self._num_sequences, out=prefix.FloatTensor()))
                one_hot[0, 0, i] = 1.0
                mode_feature = one_hot.expand(batch_size, max_length, self._num_sequences)

                encoded = torch.cat([encoded, mode_feature], 2)
                encoded = self._bilstm(encoded, length)[0]
                encoded = self._encoded_projection(encoded)

                mask = misc.mask_for_lengths(length, max_length, mask_right=False, value=1.0)
                encoded = encoded * mask.unsqueeze(2)

                seq_lemmas = torch.index_select(unique2lemma_off, 0, seq.view(-1))
                new_lemma_embeddings = misc.segment_max(encoded.view(-1, self._size), seq_lemmas, num_lemmas)
                new_lemma_embeddings = F.relu(new_lemma_embeddings)
                new_word_embeddings = torch.index_select(new_lemma_embeddings, 0, unique2lemma_off)
            else:
                new_word_embeddings = torch.zeros_like(ctxt_word_embeddings)

            # update old word embeddings with new ones via gated addition
            gate = F.sigmoid(self._gate_projection(torch.cat([ctxt_word_embeddings, new_word_embeddings], 1)))
            ctxt_word_embeddings = ctxt_word_embeddings * gate + (1.0 - gate) * new_word_embeddings

        return ctxt_word_embeddings, reading_sequence_offset, word_embeddings


class MyCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input, target):
        target = target.type(torch.cuda.LongTensor) if target.is_cuda else target.type(torch.LongTensor)
        return super().forward(input, target)


class SingleSupportAssertionClassificationModule(nn.Module):
    def __init__(self, shared_resources):
        super(SingleSupportAssertionClassificationModule, self).__init__()
        self.shared_resources = shared_resources

        size = shared_resources.config["repr_dim"]
        num_classes = shared_resources.config["answer_size"]

        # modules
        self._reader_module = ReadingModule(shared_resources, 3)

        self._question_bilstm = BiLSTM(size, size)
        self._support_bilstm = BiLSTM(size, size, start_state_given=True)

        self._hidden = nn.Linear(2 * size, size)
        self._classification = nn.Linear(size, num_classes)

    def forward(self, question_length, support_length,
                unique_word_chars, unique_word_char_length,
                question_words2uniq, support_words2uniq, is_eval,
                word_embeddings, assertion_length, assertion2question, assertion_words2uniq,
                unique_word2unique_lemma):
        reading_sequence = [support_words2uniq, question_words2uniq, assertion_words2uniq]
        reading_sequence_lengths = [support_length, question_length, assertion_length]
        reading_sequence_2_batch = [None, None, assertion2question]

        new_word_embeddings, reading_sequence_offset, _ = self._reader_module(
            word_embeddings, reading_sequence, reading_sequence_2_batch, reading_sequence_lengths,
            unique_word2unique_lemma, unique_word_chars, unique_word_char_length, is_eval)

        emb_question = F.embedding(reading_sequence_offset[1], new_word_embeddings)
        emb_support = F.embedding(reading_sequence_offset[0], new_word_embeddings)

        _, q_state = self._question_bilstm(emb_question, question_length)
        outputs = self._support_bilstm(emb_support, support_length, q_state)[0]

        # [batch, T, 2 * dim] -> [batch, dim]
        hidden = self._hidden(outputs) * misc.mask_for_lengths(support_length, mask_right=False, value=1.0).unsqueeze(2)
        hidden = hidden.max(dim=1)[0]
        # [batch, dim] -> [batch, num_classes]
        logits = self._classification(F.relu(hidden))

        return logits, logits.max(1)[1]
