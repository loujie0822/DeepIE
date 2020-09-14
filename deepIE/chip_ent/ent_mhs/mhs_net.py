import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from deepIE.config.config import CMeEnt_CONFIG


class MHSNet(nn.Module):
    """
        MHSNet : entity mhs
    """

    def __init__(self, args):
        super(MHSNet, self).__init__()

        if args.activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif args.activation.lower() == 'tanh':
            self.activation = nn.Tanh()

        self.rel_emb = nn.Embedding(num_embeddings=len(CMeEnt_CONFIG), embedding_dim=args.rel_emb_size)
        self.bert = BertModel.from_pretrained(args.bert_model)

        self.selection_u = nn.Linear(self.bert.config.hidden_size, args.rel_emb_size)
        self.selection_v = nn.Linear(self.bert.config.hidden_size, args.rel_emb_size)
        self.selection_uv = nn.Linear(2 * args.rel_emb_size, args.rel_emb_size)

    def forward(self, passage_id=None, segment_id=None, span_labels=None,
                is_eval=False):

        bert_encoder = self.bert(passage_id, token_type_ids=segment_id, attention_mask=(passage_id != 0).float())
        bert_encoder = bert_encoder[0]
        bio_mask = passage_id != 0

        B, L, H = bert_encoder.size()
        u = self.activation(self.selection_u(bert_encoder)).unsqueeze(1).expand(B, L, L, -1)
        v = self.activation(self.selection_v(bert_encoder)).unsqueeze(2).expand(B, L, L, -1)
        uv = self.activation(self.selection_uv(torch.cat((u, v), dim=-1)))

        selection_logits = torch.einsum('bijh,rh->birj', uv, self.rel_emb.weight)

        if is_eval:
            return selection_logits
        else:

            selection_loss = self.masked_BCEloss(bio_mask, selection_logits, span_labels)
            return selection_loss

    def masked_BCEloss(self, mask, selection_logits, selection_gold):

        # batch x seq x rel x seq
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(-1, -1, len(CMeEnt_CONFIG), -1)

        selection_loss = F.binary_cross_entropy_with_logits(selection_logits,
                                                            selection_gold,
                                                            reduction='none')
        selection_loss = selection_loss.masked_select(selection_mask).sum()
        selection_loss /= mask.sum()
        return selection_loss
