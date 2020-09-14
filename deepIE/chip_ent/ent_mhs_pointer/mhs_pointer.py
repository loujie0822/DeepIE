import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from deepIE.config.config import CMeEnt_CONFIG


class SingleNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(SingleNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_label)

    def forward(self, input_features):
        input_features = self.dropout(input_features)
        features_output = self.classifier(input_features)
        features_output = F.gelu(features_output)
        return features_output


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
        self.start_outputs = SingleNonLinearClassifier(self.bert.config.hidden_size, 2, dropout_rate=0.1)
        self.end_outputs = SingleNonLinearClassifier(self.bert.config.hidden_size, 2, dropout_rate=0.1)

        self.selection_u = nn.Linear(self.bert.config.hidden_size, args.rel_emb_size)
        self.selection_v = nn.Linear(self.bert.config.hidden_size, args.rel_emb_size)
        self.selection_uv = nn.Linear(2 * args.rel_emb_size, args.rel_emb_size)

    def forward(self, q_ids=None, passage_id=None, token_type_id=None, segment_id=None, point_labels=None,
                span_labels=None, is_eval=False):

        bert_encoder = self.bert(passage_id, token_type_ids=segment_id, attention_mask=(passage_id != 0).float())
        bert_encoder = bert_encoder[0]
        bio_mask = passage_id != 0
        mask = (passage_id != 0).float()

        start_logits = self.start_outputs(bert_encoder)  # batch x seq_len x 2
        end_logits = self.end_outputs(bert_encoder)  # batch x seq_len x 2

        B, L, H = bert_encoder.size()
        u = self.activation(self.selection_u(bert_encoder)).unsqueeze(1).expand(B, L, L, -1)
        v = self.activation(self.selection_v(bert_encoder)).unsqueeze(2).expand(B, L, L, -1)
        uv = self.activation(self.selection_uv(torch.cat((u, v), dim=-1)))

        span_logits = torch.einsum('bijh,rh->birj', uv, self.rel_emb.weight)

        if is_eval:
            span_scores = torch.sigmoid(span_logits)  # batch x seq_len x seq_len
            start_labels = torch.argmax(start_logits, dim=-1)
            end_labels = torch.argmax(end_logits, dim=-1)
            # start_positions = point_labels[:, :, 0]
            # end_positions = point_labels[:, :, 1]
            # return start_positions, end_positions, span_scores
            return start_labels, end_labels, span_scores
            # start_positions = point_labels[:, :, 0]
            # end_positions = point_labels[:, :, 1]
            # return start_positions, end_positions, span_labels

        else:
            start_positions = point_labels[:, :, 0]
            end_positions = point_labels[:, :, 1]
            valid_num = torch.sum(mask)
            loss_fct = nn.CrossEntropyLoss(reduction="none")

            start_loss = loss_fct(start_logits.view(-1, 2), start_positions.view(-1))
            start_loss = torch.sum(start_loss * mask.view(-1))
            start_loss = start_loss / valid_num.float()

            end_loss = loss_fct(end_logits.view(-1, 2), end_positions.view(-1))
            end_loss = torch.sum(end_loss * mask.view(-1))
            end_loss = end_loss / valid_num.float()
            span_loss = self.masked_BCEloss(bio_mask, span_logits, span_labels)
            total_loss = start_loss + end_loss + span_loss

            return total_loss, start_loss, end_loss, span_loss

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
