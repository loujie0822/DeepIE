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


class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        # self.classifier1 = nn.Linear(hidden_size, int(hidden_size / 2))
        # self.classifier2 = nn.Linear(int(hidden_size / 2), num_label)

        self.classifier1 = nn.Linear(hidden_size, num_label)
        # self.classifier2 = nn.Linear(int(hidden_size / 2), num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        input_features = self.dropout(input_features)
        features_output1 = self.classifier1(input_features)
        # features_output1 = nn.ReLU()(features_output1)
        # features_output2 = self.classifier2(features_output1)
        return features_output1


class MHSNet(nn.Module):
    """
        MHSNet : entity mhs
    """

    def __init__(self, args):
        super(MHSNet, self).__init__()

        self.rel_emb = nn.Embedding(num_embeddings=len(CMeEnt_CONFIG), embedding_dim=args.rel_emb_size)
        self.bert = BertModel.from_pretrained(args.bert_model)
        self.start_outputs = SingleNonLinearClassifier(self.bert.config.hidden_size, 2, dropout_rate=0.1)
        self.end_outputs = SingleNonLinearClassifier(self.bert.config.hidden_size, 2, dropout_rate=0.1)
        # self.span_embedding = MultiNonLinearClassifier(self.bert.config.hidden_size * 2, 1, dropout_rate=0.1)

    def forward(self, q_ids=None, passage_id=None, token_type_id=None, segment_id=None, point_labels=None,
                span_labels=None, is_eval=False):

        bert_encoder = self.bert(passage_id, token_type_ids=segment_id, attention_mask=(passage_id != 0).float())[0]
        mask = (passage_id != 0).float()

        # batch_size, seq_len, hid_size = bert_encoder.size()

        start_logits = self.start_outputs(bert_encoder)  # batch x seq_len x 2
        end_logits = self.end_outputs(bert_encoder)  # batch x seq_len x 2

        # start_extend = bert_encoder.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # end_extend = bert_encoder.unsqueeze(1).expand(-1, seq_len, -1, -1)
        #
        # span_matrix = torch.cat([start_extend, end_extend], 3)  # batch x seq_len x seq_len x 2*hidden
        #
        # span_logits = self.span_embedding(span_matrix)  # batch x seq_len x seq_len x 1
        # span_logits = torch.squeeze(span_logits, -1)  # batch x seq_len x seq_len

        if is_eval:
            # span_scores = torch.sigmoid(span_logits)  # batch x seq_len x seq_len
            start_labels = torch.argmax(start_logits, dim=-1)
            end_labels = torch.argmax(end_logits, dim=-1)
            return start_labels, end_labels, span_labels
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

            # span_loss_fct = nn.BCEWithLogitsLoss(reduction="none")
            # span_label_mask = (mask.unsqueeze(2) *
            #                    mask.unsqueeze(1)).unsqueeze(2)
            # span_loss = span_loss_fct(span_logits.view(batch_size, -1), span_labels.view(batch_size, -1).float())
            # valid_span_num = torch.sum(span_label_mask.float())
            # span_loss = torch.sum(span_loss.view(-1) * span_label_mask.view(-1).float())
            # span_loss = span_loss / valid_span_num.float()

            total_loss = start_loss + end_loss

            return total_loss, start_loss, end_loss, total_loss
