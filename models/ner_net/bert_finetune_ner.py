# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers.modeling_bert import BertModel

from layers.decoders.crf import CRF


class BertNER(nn.Module):
    def __init__(self, data):
        super(BertNER, self).__init__()

        self.gpu = data.HP_gpu
        self.use_bert = data.use_bert
        self.bertpath = data.bertpath

        char_feature_dim = 768
        print('total char_feature_dim is {}'.format(char_feature_dim))

        self.bert_encoder = BertModel.from_pretrained(self.bertpath)

        self.hidden2tag = nn.Linear(char_feature_dim, data.label_alphabet_size + 2)
        self.drop = nn.Dropout(p=data.HP_dropout)

        self.crf = CRF(data.label_alphabet_size, self.gpu)

        if self.gpu:
            self.bert_encoder = self.bert_encoder.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.crf = self.crf.cuda()

    def get_tags(self, batch_bert, bert_mask):
        seg_id = torch.zeros(bert_mask.size()).long().cuda() if self.gpu else torch.zeros(bert_mask.size()).long()
        outputs = self.bert_encoder(batch_bert, bert_mask, seg_id)
        outputs = outputs[0][:, 1:-1, :]
        tags = self.hidden2tag(outputs)

        return tags

    def neg_log_likelihood_loss(self, word_inputs, biword_inputs, word_seq_lengths, mask, batch_label, batch_bert,
                                bert_mask):
        tags = self.get_tags(batch_bert, bert_mask)

        total_loss = self.crf.neg_log_likelihood_loss(tags, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(tags, mask)

        return total_loss, tag_seq

    def forward(self, word_inputs, biword_inputs, word_seq_lengths, mask, batch_bert, bert_mask):
        tags = self.get_tags(batch_bert, bert_mask)

        scores, tag_seq = self.crf._viterbi_decode(tags, mask)

        return tag_seq
