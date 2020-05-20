# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers.modeling_bert import BertModel

from layers.decoders.crf import CRF
from layers.encoders.ner_layers import NERmodel


class GazLSTM(nn.Module):
    def __init__(self, data):
        super(GazLSTM, self).__init__()

        self.gpu = data.HP_gpu
        self.use_biword = data.use_bigram
        self.hidden_dim = data.HP_hidden_dim
        self.word_emb_dim = data.word_emb_dim
        self.biword_emb_dim = data.biword_emb_dim
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.num_layer = data.HP_num_layer
        self.model_type = data.model_type
        self.use_bert = data.use_bert
        self.device = data.device

        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.word_emb_dim, padding_idx=0)
        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))

        if self.use_biword:
            self.biword_embedding = nn.Embedding(data.biword_alphabet.size(), self.biword_emb_dim, padding_idx=0)
            if data.pretrain_biword_embedding is not None:
                self.biword_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_biword_embedding))

        char_feature_dim = self.word_emb_dim
        if self.use_biword:
            char_feature_dim += self.biword_emb_dim

        if self.use_bert:
            char_feature_dim = char_feature_dim + 768*2
        print('total char_feature_dim is {}'.format(char_feature_dim))

        ## lstm model
        if self.model_type == 'lstm':
            lstm_hidden = self.hidden_dim
            if self.bilstm_flag:
                self.hidden_dim *= 2
            self.NERmodel = NERmodel(model_type='lstm', input_dim=char_feature_dim, hidden_dim=lstm_hidden,
                                     num_layer=self.lstm_layer, biflag=self.bilstm_flag)
            self.hidden2tag = nn.Linear(self.hidden_dim, data.label_alphabet_size + 2)
        # ## cnn model
        # if self.model_type == 'cnn':
        #     self.NERmodel = NERmodel(model_type='cnn', input_dim=char_feature_dim, hidden_dim=self.hidden_dim,
        #                              num_layer=self.num_layer, dropout=data.HP_dropout, gpu=self.gpu)
        #
        # ## attention model
        if self.model_type == 'transformer':
            self.NERmodel = NERmodel(model_type='transformer', input_dim=char_feature_dim, hidden_dim=self.hidden_dim,
                                     num_layer=self.num_layer, dropout=data.HP_dropout)
            self.hidden2tag = nn.Linear(480, data.label_alphabet_size + 2)

        self.drop = nn.Dropout(p=data.HP_dropout)

        self.crf = CRF(data.label_alphabet_size, self.gpu, self.device)

        if self.use_bert:
            self.bert_encoder_1 = BertModel.from_pretrained('transformer_cpt/bert/')
            self.bert_encoder_2 = BertModel.from_pretrained('transformer_cpt/chinese_roberta_wwm_ext_pytorch/')
            for p in self.bert_encoder_1.parameters():
                p.requires_grad = False
            for p in self.bert_encoder_2.parameters():
                p.requires_grad = False
        if self.gpu:
            self.word_embedding = self.word_embedding.cuda(self.device)
            if self.use_biword:
                self.biword_embedding = self.biword_embedding.cuda(self.device)
            self.NERmodel = self.NERmodel.cuda(self.device)
            self.hidden2tag = self.hidden2tag.cuda(self.device)
            self.crf = self.crf.cuda(self.device)
            if self.use_bert:
                self.bert_encoder_1 = self.bert_encoder_1.cuda(self.device)
                self.bert_encoder_2 = self.bert_encoder_2.cuda(self.device)

    def get_tags(self, word_inputs, biword_inputs, mask, word_seq_lengths, batch_bert, bert_mask):

        batch_size = word_inputs.size()[0]
        seq_len = word_inputs.size()[1]

        word_embs = self.word_embedding(word_inputs)

        if self.use_biword:
            biword_embs = self.biword_embedding(biword_inputs)
            word_embs = torch.cat([word_embs, biword_embs], dim=-1)

        if self.model_type != 'transformer':
            word_inputs_d = self.drop(word_embs)  # (b,l,we)
        else:
            word_inputs_d = word_embs

        word_input_cat = torch.cat([word_inputs_d], dim=-1)  # (b,l,we+4*ge)

        if self.use_bert:
            seg_id = torch.zeros(bert_mask.size()).long().cuda(self.device) if self.gpu else torch.zeros(
                bert_mask.size()).long()
            outputs_1 = self.bert_encoder_1(batch_bert, bert_mask, seg_id)
            outputs_1 = outputs_1[0][:, 1:-1, :]

            outputs_2 = self.bert_encoder_2(batch_bert, bert_mask, seg_id)
            outputs_2 = outputs_2[0][:, 1:-1, :]

            word_input_cat = torch.cat([word_input_cat, outputs_1, outputs_2], dim=-1)

        feature_out_d = self.NERmodel(word_input_cat, word_inputs.ne(0))

        tags = self.hidden2tag(feature_out_d)

        return tags

    def neg_log_likelihood_loss(self, word_inputs, biword_inputs, word_seq_lengths, mask, batch_label, batch_bert,
                                bert_mask):

        tags = self.get_tags(word_inputs, biword_inputs, mask, word_seq_lengths, batch_bert, bert_mask)

        total_loss = self.crf.neg_log_likelihood_loss(tags, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(tags, mask)

        return total_loss, tag_seq

    def forward(self, word_inputs, biword_inputs, word_seq_lengths, mask, batch_bert, bert_mask):

        tags = self.get_tags(word_inputs, biword_inputs, mask, word_seq_lengths, batch_bert, bert_mask)

        scores, tag_seq = self.crf._viterbi_decode(tags, mask)

        return tag_seq
