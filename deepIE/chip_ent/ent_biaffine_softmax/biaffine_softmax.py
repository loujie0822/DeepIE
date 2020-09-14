import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertPreTrainedModel

from layers.encoders.rnns.stacked_rnn import StackedBRNN


class SentenceEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SentenceEncoder, self).__init__()
        self.encoder = StackedBRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout_rate=0.5,
            dropout_output=True,
            concat_layers=False,
            rnn_type=nn.LSTM,
            padding=True
        )

    def forward(self, input, mask):
        return self.encoder(input, mask)


class BiaffineClassifier(nn.Module):
    def __init__(self, feature1_size, feature2_size, convert_feature_size, output_size):
        super(BiaffineClassifier, self).__init__()

        self.feature1_size = feature1_size
        self.feature2_size = feature2_size
        self.output_size = output_size
        self.input_feature1 = nn.Linear(feature1_size, convert_feature_size)
        self.input_feature2 = nn.Linear(feature2_size, convert_feature_size)
        self.affine_feature_size = convert_feature_size

        self.bilinear_map = nn.Linear(convert_feature_size, convert_feature_size * output_size)

    def forward(self, input):
        seq_len = input.size()[1]
        vector_set_1 = nn.ReLU()(self.input_feature1(input))
        vector_set_2 = nn.ReLU()(self.input_feature2(input))

        batch_size = input.size()[0]

        # [b, n, v1] -> [b*n, v1]
        vector_set_1 = vector_set_1.view(-1, self.affine_feature_size)

        # [b*n, v1] x [v1, r*v2] -> [b*n, r*v2]
        bilinear_mapping = self.bilinear_map(vector_set_1)

        # [b*n, r*v2] -> [b, n*r, v2]

        bilinear_mapping = bilinear_mapping.view(batch_size, seq_len * self.output_size, self.affine_feature_size)

        # [b, n*r, v2] x [b, n, v2]T -> [b, n*r, n]
        bilinear_mapping = torch.bmm(bilinear_mapping, vector_set_2.transpose(1, 2))

        # [b, n*r, n] -> [b, n, r, n]

        bilinear_mapping = bilinear_mapping.view(batch_size, seq_len, self.output_size, seq_len)
        return bilinear_mapping


class EntExtractNet(BertPreTrainedModel):
    """
    Attribute Extract Net with Multi-label Pointer Network(MPN) based Entity-aware and
    encoded by BERT
    """

    def __init__(self, config, classes_num):
        super(EntExtractNet, self).__init__(config, classes_num)
        print('ent biaffine')

        self.bert = BertModel(config)
        self.classes_num = classes_num+1
        # self.lstm_encoder = SentenceEncoder(input_size=config.hidden_size, hidden_size=config.hidden_size // 2)
        self.biaffine = BiaffineClassifier(feature1_size=config.hidden_size, feature2_size=config.hidden_size,
                                           convert_feature_size=150, output_size=self.classes_num)

        self.init_weights()

    def forward(self, q_ids=None, passage_id=None, token_type_id=None, segment_id=None,
                span_labels=None, is_eval=False):
        mask = (passage_id != 0).float()
        bert_encoder, _ = self.bert(passage_id, token_type_ids=segment_id,
                                    attention_mask=mask)  # batch x seq_len x hidden

        span_logits = self.biaffine(bert_encoder).transpose(3, 2)

        if not is_eval:
            span_mask = (mask.unsqueeze(2) * mask.unsqueeze(1))
            span_mask = span_mask.view(-1)
            valid_num = torch.sum(span_mask.float())
            span_loss_fct = nn.CrossEntropyLoss(reduction="none")
            span_loss = span_loss_fct(span_logits.reshape(-1, self.classes_num), span_labels.reshape(-1))
            span_loss = torch.sum(span_loss * span_mask.float())
            span_loss = span_loss / valid_num

            return span_loss
        else:
            return span_logits.argmax(-1)
            # return span_labels
