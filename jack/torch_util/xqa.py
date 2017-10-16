# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional


class XQAMinCrossentropyLossModule(nn.Module):
    def forward(self, start_scores, end_scores, answer_span, answer_to_question):
        """very common XQA loss function."""
        answer_span = answer_span.type(torch.LongTensor)
        start, end = answer_span[:, 0], answer_span[:, 1]

        batch_size1 = start.data.shape[0]
        batch_size2 = start_scores.data.shape[0]
        is_aligned = batch_size1 == batch_size2

        start_scores = start_scores if is_aligned else torch.index_select(start_scores, 0, answer_to_question)
        end_scores = end_scores if is_aligned else torch.index_select(end_scores, 0, answer_to_question)
        loss = functional.nll_loss(
            functional.log_softmax(start_scores), start) + functional.nll_loss(functional.log_softmax(end_scores), end)
        return loss