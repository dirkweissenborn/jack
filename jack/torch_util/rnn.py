import torch
from torch import nn


class BiLSTM(nn.Module):
    def __init__(self, input_size, size, start_state_given=False):
        super(BiLSTM, self).__init__()
        self._size = size
        self._bilstm = nn.LSTM(input_size, size, 1, bidirectional=True, batch_first=True)
        self._bilstm.bias_ih_l0.data[size:2 * size].fill_(1.0)
        self._bilstm.bias_ih_l0_reverse.data[size:2 * size].fill_(1.0)
        self._start_state_given = start_state_given
        if not start_state_given:
            self._lstm_start_hidden = nn.Parameter(torch.zeros(2, size))
            self._lstm_start_state = nn.Parameter(torch.zeros(2, size))

    def forward(self, inputs, lengths=None, start_state=None):
        if not self._start_state_given:
            batch_size = inputs.size(0)
            start_hidden = self._lstm_start_hidden.unsqueeze(1).expand(2, batch_size, self._size).contiguous()
            start_state = self._lstm_start_state.unsqueeze(1).expand(2, batch_size, self._size).contiguous()
            start_state = (start_hidden, start_state)

        if lengths is not None:
            new_lengths, indices = torch.sort(lengths, dim=0, descending=True)
            inputs = torch.index_select(inputs, 0, indices)
            if self._start_state_given:
                start_state = (torch.index_select(start_state[0], 1, indices),
                               torch.index_select(start_state[1], 1, indices))
            new_lengths = [l.data[0] for l in new_lengths]
            inputs = pack_padded_sequence(inputs, new_lengths, batch_first=True)

        output, (h_n, c_n) = self._bilstm(inputs, start_state)

        if lengths is not None:
            output = pad_packed_sequence(output, batch_first=True)[0]
            _, back_indices = torch.sort(indices, dim=0)
            output = torch.index_select(output, 0, back_indices)
            h_n = torch.index_select(h_n, 1, back_indices)
            c_n = torch.index_select(c_n, 1, back_indices)

        return output, (h_n, c_n)


def pack_padded_sequence(input, lengths, batch_first=False):
    """Packs a Variable containing padded sequences of variable length.

    Input can be of size ``TxBx*`` where T is the length of the longest sequence
    (equal to ``lengths[0]``), B is the batch size, and * is any number of
    dimensions (including 0). If ``batch_first`` is True ``BxTx*`` inputs are
    expected.

    The sequences should be sorted by length in a decreasing order, i.e.
    ``input[:,0]`` should be the longest sequence, and ``input[:,B-1]`` the
    shortest one.

    Note:
        This function accept any input that has at least two dimensions. You
        can apply it to pack the labels, and use the output of the RNN with
        them to compute the loss directly. A Variable can be retrieved from
        a :class:`PackedSequence` object by accessing its ``.data`` attribute.

    Arguments:
        input (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequences lengths of each batch element.
        batch_first (bool, optional): if True, the input is expected in BxTx*
            format.

    Returns:
        a :class:`PackedSequence` object
    """
    if lengths[-1] <= 0:
        raise ValueError("length of all samples has to be greater than 0, "
                         "but found an element in 'lengths' that is <=0")
    if batch_first:
        input = input.transpose(0, 1)

    steps = []
    batch_sizes = []
    lengths_iter = reversed(lengths)
    batch_size = input.size(1)
    if len(lengths) != batch_size:
        raise ValueError("lengths array has incorrect size")

    prev_l = 0
    for i, l in enumerate(lengths_iter):
        if l > prev_l:
            c_batch_size = batch_size - i
            steps.append(input[prev_l:l, :c_batch_size].contiguous().view(-1, input.size(2)))
            batch_sizes.extend([c_batch_size] * (l - prev_l))
            prev_l = l

    return nn.utils.rnn.PackedSequence(torch.cat(steps), batch_sizes)


def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0):
    """Pads a packed batch of variable length sequences.

    It is an inverse operation to :func:`pack_padded_sequence`.

    The returned Variable's data will be of size TxBx*, where T is the length
    of the longest sequence and B is the batch size. If ``batch_first`` is True,
    the data will be transposed into BxTx* format.

    Batch elements will be ordered decreasingly by their length.

    Arguments:
        sequence (PackedSequence): batch to pad
        batch_first (bool, optional): if True, the output will be in BxTx*
            format.
        padding_value (float, optional): values for padded elements

    Returns:
        Tuple of Variable containing the padded sequence, and a list of lengths
        of each sequence in the batch.
    """
    var_data, batch_sizes = sequence
    max_batch_size = batch_sizes[0]
    output = var_data.data.new(len(batch_sizes), max_batch_size, *var_data.size()[1:]).fill_(padding_value)
    output = torch.autograd.Variable(output)

    lengths = []
    data_offset = 0
    prev_batch_size = batch_sizes[0]
    prev_i = 0
    for i, batch_size in enumerate(batch_sizes):
        if batch_size != prev_batch_size:
            l = prev_batch_size * (i - prev_i)
            output[prev_i:i, :prev_batch_size] = var_data[data_offset:data_offset + l]
            data_offset += l
            prev_i = i
        dec = prev_batch_size - batch_size
        if dec > 0:
            lengths.extend((i,) * dec)
        prev_batch_size = batch_size

    l = prev_batch_size * (len(batch_sizes) - prev_i)
    output[prev_i:, :prev_batch_size] = var_data[data_offset:data_offset + l]

    lengths.extend((i + 1,) * batch_size)
    lengths.reverse()

    if batch_first:
        output = output.transpose(0, 1)
    return output, lengths
