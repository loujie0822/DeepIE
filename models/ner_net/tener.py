import torch
import torch.nn.functional as F
from fastNLP.modules import ConditionalRandomField, allowed_transitions
from torch import nn

from layers.encoders.transformers.transformer import TransformerEncoder


class TENER(nn.Module):
    def __init__(self, tag_vocab, embed, num_layers, d_model, n_head, feedforward_dim, dropout,
                 after_norm=True, attn_type='adatrans', bi_embed=None,bert_embed=None,
                 fc_dropout=0.3, pos_embed=None, scale=False, dropout_attn=None):
        """

        :param tag_vocab: fastNLP Vocabulary
        :param embed: fastNLP TokenEmbedding
        :param num_layers: number of self-attention layers
        :param d_model: input size
        :param n_head: number of head
        :param feedforward_dim: the dimension of ffn
        :param dropout: dropout in self-attention
        :param after_norm: normalization place
        :param attn_type: adatrans, naive
        :param rel_pos_embed: position embedding的类型，支持sin, fix, None. relative时可为None
        :param bi_embed: Used in Chinese scenerio
        :param fc_dropout: dropout rate before the fc layer
        """
        super().__init__()

        self.embed = embed
        embed_size = self.embed.embed_size
        self.bi_embed = None
        if bi_embed is not None:
            self.bi_embed = bi_embed
            embed_size += self.bi_embed.embed_size
        if bert_embed is not None:
            self.bert_embed = bert_embed
            embed_size += self.bert_embed.embed_size

        print('total embed_size is {}'.format(embed_size))

        self.in_fc = nn.Linear(embed_size, d_model)

        self.transformer = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.out_fc = nn.Linear(d_model, len(tag_vocab))

        trans = allowed_transitions(tag_vocab, include_start_end=True)
        self.crf = ConditionalRandomField(len(tag_vocab), include_start_end_trans=True, allowed_transitions=trans)

    def _forward(self, input_chars, target, bigrams=None):
        mask = input_chars.ne(0)
        chars = self.embed(input_chars)
        if self.bi_embed is not None:
            bigrams = self.bi_embed(bigrams)
            chars = torch.cat([chars, bigrams], dim=-1)
        if self.bert_embed is not None:
            bert_embeddings= self.bert_embed(input_chars)
            chars = torch.cat([chars, bert_embeddings], dim=-1)

        chars = self.in_fc(chars)
        chars = self.transformer(chars, mask)
        chars = self.fc_dropout(chars)
        chars = self.out_fc(chars)
        logits = F.log_softmax(chars, dim=-1)
        if target is None:
            paths, _ = self.crf.viterbi_decode(logits, mask)
            return {'pred': paths}
        else:
            loss = self.crf(logits, target, mask)
            return {'loss': loss}

    def forward(self, chars, target, bigrams=None):
        return self._forward(chars, target, bigrams)

    def predict(self, chars, bigrams=None):
        return self._forward(chars, target=None, bigrams=bigrams)
