# _*_ coding:utf-8 _*_
import copy
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel

from deepIE.config.config import CMeIE_CONFIG, Ent_BIO
from layers.decoders.pytorch_crf import CRF

warnings.filterwarnings("ignore")


class MHSNet(nn.Module):
    """
        MHSNet : entity relation extraction
    """

    def __init__(self, args):
        super(MHSNet, self).__init__()

        if args.activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif args.activation.lower() == 'tanh':
            self.activation = nn.Tanh()

        self.rel_emb = nn.Embedding(num_embeddings=len(CMeIE_CONFIG), embedding_dim=args.rel_emb_size)
        self.ent_emb = nn.Embedding(num_embeddings=len(Ent_BIO), embedding_dim=args.ent_emb_size)
        self.bert = BertModel.from_pretrained(args.bert_model)
        self.emission = nn.Linear(self.bert.config.hidden_size, len(Ent_BIO))

        self.crf = CRF(len(Ent_BIO), batch_first=True)

        self.selection_u = nn.Linear(self.bert.config.hidden_size + args.ent_emb_size,
                                     args.rel_emb_size)
        self.selection_v = nn.Linear(self.bert.config.hidden_size + args.ent_emb_size,
                                     args.rel_emb_size)
        self.selection_uv = nn.Linear(2 * args.rel_emb_size,
                                      args.rel_emb_size)

    def forward(self, passage_ids=None, segment_ids=None, ent_ids=None, rel_ids=None,
                is_eval=False):

        bert_encoder = self.bert(passage_ids, token_type_ids=segment_ids, attention_mask=(passage_ids != 0).float())
        bert_encoder = bert_encoder[0][:, 1:-1, :]

        emission = self.emission(bert_encoder)

        passage_ids = passage_ids[:, 1:-1]

        bio_mask = passage_ids != 0

        if is_eval:
            ent_pre = self.entity_decoder(bio_mask, emission, max_len=bert_encoder.size(1))
            ent_encoder = self.ent_emb(ent_pre)
        else:
            ent_encoder = self.ent_emb(ent_ids)
        # ent_encoder = self.ent_emb(ent_ids)

        rel_encoder = torch.cat((bert_encoder, ent_encoder), dim=2)

        B, L, H = rel_encoder.size()
        u = self.activation(self.selection_u(rel_encoder)).unsqueeze(1).expand(B, L, L, -1)
        v = self.activation(self.selection_v(rel_encoder)).unsqueeze(2).expand(B, L, L, -1)
        uv = self.activation(self.selection_uv(torch.cat((u, v), dim=-1)))

        selection_logits = torch.einsum('bijh,rh->birj', uv, self.rel_emb.weight)

        if is_eval:
            return ent_pre, selection_logits
            # return ent_ids, selection_logits
        else:
            crf_loss = -self.crf(emission, ent_ids, mask=bio_mask, reduction='mean')
            selection_loss = self.masked_BCEloss(bio_mask, selection_logits, rel_ids)

            loss = crf_loss + 100*selection_loss

            return loss,crf_loss,selection_loss

    def masked_BCEloss(self, mask, selection_logits, selection_gold):

        # batch x seq x rel x seq
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(-1, -1, len(CMeIE_CONFIG), -1)

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
