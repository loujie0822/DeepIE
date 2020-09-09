#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# Author: Xiaoy LI
# Description:
# Bert Model for MRC-Based NER Task


import torch
import torch.nn as nn


from deepIE.chip_ent.ent_extract_mrc.layer.classifier import MultiNonLinearClassifier, SingleNonLinearClassifier
from deepIE.chip_ent.ent_extract_mrc.layer.bert_basic_model import BertModel, BertConfig


class BertQueryNER(nn.Module):
    def __init__(self, config, train_steps=1200000):
        super(BertQueryNER, self).__init__()
        bert_config = BertConfig.from_dict(config.bert_config.to_dict())
        self.bert = BertModel(bert_config)

        self.start_outputs = SingleNonLinearClassifier(config.hidden_size, 2, config.dropout)
        self.end_outputs = SingleNonLinearClassifier(config.hidden_size, 2, config.dropout)

        self.span_embedding = MultiNonLinearClassifier(config.hidden_size*2, 1, config.dropout)
        self.hidden_size = config.hidden_size
        self.bert = self.bert.from_pretrained(config.bert_model)
        self.train_steps = train_steps
        self.loss_wb = config.weight_start
        self.loss_we = config.weight_end
        self.loss_ws = config.weight_span

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.loss_type = config.loss_type
        if "dynamic_wce" in self.loss_type:
            start_sig = torch.empty(1)
            end_sig = torch.empty(1)
            span_sig = torch.empty(1)
            # test different init scale
            self._start_loss_sig = nn.init.normal_(start_sig,).to(self.device)
            self._end_loss_sig = nn.init.normal_(end_sig, ).to(self.device)
            self._span_loss_sig = nn.init.normal_(span_sig,).to(self.device)

    def update_loss_ratio(self, current_train_step=None, decay_step=5000,
                          lower_bound_weight=0.6, upper_bound_weight=1.5, decay_base=3.0, increase_base=1.5):
        if current_train_step is None:
            return
        if current_train_step > decay_step:
            loss_wb = self.loss_wb * (decay_base ** -(current_train_step/self.train_steps))
            loss_we = self.loss_we * (decay_base ** -(current_train_step/self.train_steps))
            self.loss_wb = loss_wb if loss_wb > lower_bound_weight else lower_bound_weight
            self.loss_we = loss_we if loss_we > lower_bound_weight else lower_bound_weight

            loss_ws = self.loss_ws * (increase_base ** (current_train_step/self.train_steps))
            self.loss_ws = loss_ws if loss_ws <= upper_bound_weight else upper_bound_weight
            if current_train_step % 1000 == 0:
                print(f"*** *** *** >>> update loss weight: {self.loss_wb}, {self.loss_we}, {self.loss_ws}")

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
        start_positions=None, end_positions=None, span_positions=None, span_label_mask=None, current_step=None):
        """
        Args:
            start_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]]
            end_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]]
            span_positions: (batch x max_len x max_len)
                span_positions[k][i][j] is one of [0, 1],
                span_positions[k][i][j] represents whether or not from start_pos{i} to end_pos{j} of the K-th sentence in the batch is an entity.
        """

        sequence_output, pooled_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        sequence_heatmap = sequence_output  # batch x seq_len x hidden
        batch_size, seq_len, hid_size = sequence_heatmap.size()

        start_logits = self.start_outputs(sequence_heatmap)  # batch x seq_len x 2
        end_logits = self.end_outputs(sequence_heatmap)  # batch x seq_len x 2

        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # the shape of start_end_concat[0] is : batch x 1 x seq_len x 2*hidden

        span_matrix = torch.cat([start_extend, end_extend], 3) # batch x seq_len x seq_len x 2*hidden

        span_logits = self.span_embedding(span_matrix)  # batch x seq_len x seq_len x 1
        span_logits = torch.squeeze(span_logits)  # batch x seq_len x seq_len

        if start_positions is not None and end_positions is not None:   # for train and is not negative
            # self.update_loss_ratio(current_train_step=current_step)
            valid_num = torch.sum(token_type_ids)
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            start_loss = loss_fct(start_logits.view(-1, 2), start_positions.view(-1))
            start_loss = torch.sum(start_loss * token_type_ids.view(-1).float())
            start_loss = start_loss / valid_num.float()

            end_loss = loss_fct(end_logits.view(-1, 2), end_positions.view(-1))
            end_loss = torch.sum(end_loss * token_type_ids.view(-1).float())
            end_loss = end_loss / valid_num.float()

            span_loss_fct = nn.BCEWithLogitsLoss(reduction="none")
            span_loss = span_loss_fct(span_logits.view(batch_size, -1), span_positions.view(batch_size, -1).float())
            valid_span_num = torch.sum(span_label_mask)
            span_loss = torch.sum(span_loss.view(-1) * span_label_mask.view(-1).float())
            span_loss = span_loss / valid_span_num.float()
            total_loss = self._compute_loss(start_loss, end_loss, span_loss, loss_type=self.loss_type)
            # total_loss = self.loss_wb * start_loss + self.loss_we * end_loss + self.loss_ws * span_loss
            return total_loss
        else:   # for eval or is position
            span_scores = torch.sigmoid(span_logits) # batch x seq_len x seq_len
            start_labels = torch.argmax(start_logits, dim=-1)
            end_labels = torch.argmax(end_logits, dim=-1)
            return start_labels, end_labels, span_scores

    def _compute_loss(self, start_loss, end_loss, span_loss, loss_type="ce"):
        if loss_type == "ce":
            total_loss = self.loss_wb * start_loss + self.loss_we * end_loss + self.loss_ws * span_loss
            return total_loss
        elif loss_type == "dynamic_wce":
            b_factor = torch.exp(- self._start_loss_sig)
            b_loss = b_factor * start_loss + self._start_loss_sig

            e_factor = torch.exp(- self._end_loss_sig)
            e_loss = e_factor * end_loss + self._end_loss_sig

            s_factor = torch.exp(- self._span_loss_sig)
            s_loss = s_factor * span_loss + self._span_loss_sig
            total_loss = b_loss + e_loss + s_loss
            return total_loss
        elif loss_type == "average_dynamic_wce":
            b_factor = torch.exp(- self._start_loss_sig)
            b_loss = b_factor * start_loss + self._start_loss_sig * 0.3

            e_factor = torch.exp(- self._end_loss_sig)
            e_loss = e_factor * end_loss + self._end_loss_sig * 0.3

            s_factor = torch.exp(- self._span_loss_sig)
            s_loss = s_factor * span_loss + self._span_loss_sig * 0.3
            total_loss = b_loss + e_loss + s_loss
            return total_loss
        else:
            raise ValueError("Loss Type doesnot exists. ")
