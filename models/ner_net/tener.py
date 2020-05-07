import torch
import torch.nn.functional as F
from fastNLP.modules import ConditionalRandomField, allowed_transitions
from torch import nn

from layers.encoders.transformers.transformer import TransformerEncoder


class TENER(nn.Module):
    def __init__(self, model_conf, attn_type='adatrans', pos_embed=None, dropout_attn=None):
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

        print('current model is TENER')

        # origin paper param

        n_head = 6
        head_dims = 80
        num_layers = 2
        d_model = n_head * head_dims
        feedforward_dim = int(2 * d_model)
        dropout = 0.15
        fc_dropout = 0.4
        after_norm = 1
        scale = attn_type == 'transformer'
        tag_vocab = model_conf['entity_type']

        # embedding
        embed = model_conf['char_emb']
        bi_embed = model_conf['bichar_emb']

        self.embed = nn.Embedding(num_embeddings=embed.shape[0], embedding_dim=embed.shape[1],
                                         padding_idx=0, _weight=embed)
        embed_size = embed.size()[1]
        self.bi_embed = None
        if bi_embed is not None:
            self.bi_embed = nn.Embedding(num_embeddings=bi_embed.shape[0], embedding_dim=bi_embed.shape[1],
                                           padding_idx=0, _weight=bi_embed)
            embed_size += bi_embed.size()[1]

        self.in_fc = nn.Linear(embed_size, d_model)

        self.transformer = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.out_fc = nn.Linear(d_model, len(tag_vocab))
        trans = allowed_transitions({item: key for key, item in tag_vocab.items()}, include_start_end=True,
                                    encoding_type='bmeso')
        self.crf = ConditionalRandomField(len(tag_vocab), include_start_end_trans=True, allowed_transitions=trans)

    def forward(self, char_id, bichar_id, label_ids, is_eval=False):
        chars = char_id
        bigrams = bichar_id
        target = label_ids

        mask = chars.ne(0)
        chars = self.embed(chars)
        if self.bi_embed is not None:
            bigrams = self.bi_embed(bigrams)
            chars = torch.cat([chars, bigrams], dim=-1)

        chars = self.in_fc(chars)
        chars = self.transformer(chars, mask)
        chars = self.fc_dropout(chars)
        chars = self.out_fc(chars)
        logits = F.log_softmax(chars, dim=-1)
        if is_eval:
            paths, _ = self.crf.viterbi_decode(logits, mask)
            return paths
        else:
            loss = self.crf(logits, target, mask)
            return loss.mean()
