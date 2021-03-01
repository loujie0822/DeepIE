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


class EntExtractNet(BertPreTrainedModel):
    """
    Attribute Extract Net with Multi-label Pointer Network(MPN) based Entity-aware and
    encoded by BERT
    """

    def __init__(self, config, classes_num):
        super(EntExtractNet, self).__init__(config, classes_num)
        print('ent_po_net_lstm.py')

        self.bert = BertModel(config)
        # self.token_entity_emb = nn.Embedding(num_embeddings=2, embedding_dim=config.hidden_size,
        #                                      padding_idx=0)

        # # sentence_encoder using transformer
        # self.transformer_encoder_layer = TransformerEncoderLayer(config.hidden_size, args.nhead,
        #                                                          dim_feedforward=args.dim_feedforward)
        # self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, args.transformer_layers)

        self.lstm_encoder = SentenceEncoder(config.hidden_size, 150)



        self.classes_num = classes_num
        self.po_dense = nn.Linear(300, self.classes_num * 2)

        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

        self.init_weights()

    def forward(self, q_ids=None, passage_id=None, token_type_id=None, segment_id=None, pos_start=None, pos_end=None,
                object_labels=None, is_eval=False):
        mask = (passage_id != 0).float()
        bert_encoder = self.bert(passage_id, token_type_ids=segment_id, attention_mask=mask)[0]



        sent_encoder = self.lstm_encoder(bert_encoder.contiguous(), mask=passage_id.eq(0))
        po_preds = self.po_dense(sent_encoder).reshape(passage_id.size(0), -1, self.classes_num, 2)

        if not is_eval:
            po_loss = self.loss_fct(po_preds, object_labels)
            po_loss = torch.sum(po_loss.mean(3), 2)
            po_loss = torch.sum(po_loss * mask.float()) / torch.sum(mask.float())
            return po_loss
        else:
            po_tensor = nn.Sigmoid()(po_preds)
            # token_type_id = token_type_id.reshape(passage_id.size(0), -1, self.classes_num, 2)
            return po_tensor
