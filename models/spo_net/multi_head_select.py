# _*_ coding:utf-8 _*_
import copy
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from config.spo_config_v1 import BAIDU_ENTITY, BAIDU_RELATION
from layers.decoders.pytorch_crf import CRF
from layers.encoders.rnns.stacked_rnn import StackedBRNN

warnings.filterwarnings("ignore")


class SentenceEncoder(nn.Module):
    def __init__(self, args, input_size):
        super(SentenceEncoder, self).__init__()
        rnn_type = nn.LSTM if args.rnn_encoder == 'lstm' else nn.GRU
        self.encoder = StackedBRNN(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout_rate=args.dropout,
            dropout_output=True,
            concat_layers=False,
            rnn_type=rnn_type,
            padding=True
        )

    def forward(self, input, mask):
        return self.encoder(input, mask)


class ERENet(nn.Module):
    """
        ERENet : entity relation extraction
    """

    def __init__(self, args, word_emb):
        super(ERENet, self).__init__()
        print('mhs with w2v')

        if args.activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif args.activation.lower() == 'tanh':
            self.activation = nn.Tanh()

        self.word_emb = nn.Embedding.from_pretrained(torch.tensor(word_emb, dtype=torch.float32), freeze=True,
                                                     padding_idx=0)
        self.word_convert_char = nn.Linear(args.word_emb_size, args.char_emb_size, bias=False)
        self.char_emb = nn.Embedding(num_embeddings=args.char_vocab_size, embedding_dim=args.char_emb_size,
                                     padding_idx=0)
        self.rel_emb = nn.Embedding(num_embeddings=len(BAIDU_RELATION), embedding_dim=args.rel_emb_size)
        self.ent_emb = nn.Embedding(num_embeddings=len(BAIDU_ENTITY), embedding_dim=args.ent_emb_size)

        self.sentence_encoder = SentenceEncoder(args, args.char_emb_size)

        self.emission = nn.Linear(args.hidden_size * 2, len(BAIDU_ENTITY))

        self.crf = CRF(len(BAIDU_ENTITY), batch_first=True)

        self.selection_u = nn.Linear(2 * args.hidden_size + args.ent_emb_size,
                                     args.rel_emb_size)
        self.selection_v = nn.Linear(2 * args.hidden_size + args.ent_emb_size,
                                     args.rel_emb_size)
        self.selection_uv = nn.Linear(2 * args.rel_emb_size,
                                      args.rel_emb_size)

    def forward(self, char_ids=None, word_ids=None, label_ids=None, spo_ids=None, is_eval=False):

        # Entity Extraction
        mask = char_ids.eq(0)
        char_emb = self.char_emb(char_ids)
        word_emb = self.word_convert_char(self.word_emb(word_ids))
        emb = char_emb + word_emb
        sent_encoder = self.sentence_encoder(emb, mask)

        bio_mask = char_ids != 0
        emission = self.emission(sent_encoder)
        # TODO:check
        ent_pre = self.entity_decoder(bio_mask, emission, max_len=sent_encoder.size(1))

        # Relation Extraction
        if is_eval:
            ent_encoder = self.ent_emb(ent_pre)
        else:
            ent_encoder = self.ent_emb(label_ids)

        rel_encoder = torch.cat((sent_encoder, ent_encoder), dim=2)

        B, L, H = rel_encoder.size()
        u = self.activation(self.selection_u(rel_encoder)).unsqueeze(1).expand(B, L, L, -1)
        v = self.activation(self.selection_v(rel_encoder)).unsqueeze(2).expand(B, L, L, -1)
        uv = self.activation(self.selection_uv(torch.cat((u, v), dim=-1)))

        selection_logits = torch.einsum('bijh,rh->birj', uv, self.rel_emb.weight)

        if is_eval:
            return ent_pre, selection_logits
        else:
            crf_loss = -self.crf(emission, label_ids, mask=bio_mask, reduction='mean')
            selection_loss = self.masked_BCEloss(bio_mask, selection_logits, spo_ids)

            loss = crf_loss + selection_loss

            return loss

    def masked_BCEloss(self, mask, selection_logits, selection_gold):

        # batch x seq x rel x seq
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(-1, -1, len(BAIDU_RELATION), -1)

        selection_loss = F.binary_cross_entropy_with_logits(selection_logits,
                                                            selection_gold,
                                                            reduction='none')
        selection_loss = selection_loss.masked_select(selection_mask).sum()
        selection_loss /= mask.sum()
        return selection_loss

    def entity_decoder(self, bio_mask, emission, max_len):
        decoded_tag = self.crf.decode(emissions=emission, mask=bio_mask)

        temp_tag = copy.deepcopy(decoded_tag)
        for line in temp_tag:
            line.extend([0] * (max_len - len(line)))
        # TODO:check
        ent_pre = torch.tensor(temp_tag).to(emission.device)
        # print('entity predict embedding device is {}'.format(ent_pre.device))
        return ent_pre
