# _*_ coding:utf-8 _*_
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel

from deepIE.config.config import CMeIE_CONFIG

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
        self.ent_emb = nn.Embedding(num_embeddings=2, embedding_dim=args.ent_emb_size)
        self.bert = BertModel.from_pretrained(args.bert_model)
        self.ent_dense = nn.Linear(self.bert.config.hidden_size, 2)

        self.selection_u = nn.Linear(self.bert.config.hidden_size + args.ent_emb_size,
                                     args.rel_emb_size)
        self.selection_v = nn.Linear(self.bert.config.hidden_size + args.ent_emb_size,
                                     args.rel_emb_size)
        self.selection_uv = nn.Linear(2 * args.rel_emb_size,
                                      args.rel_emb_size)
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, passage_ids=None, segment_ids=None, ent_ids=None, rel_ids=None,
                is_eval=False):

        bert_encoder = self.bert(passage_ids, token_type_ids=segment_ids, attention_mask=(passage_ids != 0).float())
        bert_encoder = bert_encoder[0]

        ent_pre = self.ent_dense(bert_encoder)

        mask = passage_ids != 0

        if is_eval:
            ent_label_ids = (nn.Sigmoid()(ent_pre) > .5)[:, :, 0].long().to(ent_pre.device)
        else:
            ent_label_ids = torch.tensor(ent_ids[:, :, 0], dtype=torch.long).to(ent_pre.device)
        ent_encoder = self.ent_emb(ent_label_ids)

        rel_encoder = torch.cat((bert_encoder, ent_encoder), dim=2)

        B, L, H = rel_encoder.size()
        u = self.activation(self.selection_u(rel_encoder)).unsqueeze(1).expand(B, L, L, -1)
        v = self.activation(self.selection_v(rel_encoder)).unsqueeze(2).expand(B, L, L, -1)
        uv = self.activation(self.selection_uv(torch.cat((u, v), dim=-1)))

        selection_logits = torch.einsum('bijh,rh->birj', uv, self.rel_emb.weight)

        if is_eval:
            return ent_pre, selection_logits
            # return ent_ids, rel_ids
        else:
            ent_loss = self.loss_fct(ent_pre, ent_ids)
            ent_loss = ent_loss.mean(2)
            ent_loss = torch.sum(ent_loss * mask.float()) / torch.sum(mask.float())

            selection_loss = self.masked_BCEloss(mask, selection_logits, rel_ids)

            loss = ent_loss + 100*selection_loss

            return loss, ent_loss, selection_loss

    def masked_BCEloss(self, mask, selection_logits, selection_gold):

        # batch x seq x rel x seq
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(-1, -1, len(CMeIE_CONFIG), -1)

        selection_loss = F.binary_cross_entropy_with_logits(selection_logits,
                                                            selection_gold,
                                                            reduction='none')
        selection_loss = selection_loss.masked_select(selection_mask).sum()
        selection_loss /= selection_mask.sum()
        return selection_loss
