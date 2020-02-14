import torch
from torch import nn

from layers.encoders.transformers.bert.bert_model import BertModel
from layers.encoders.transformers.bert.bert_pretrain import BertPreTrainedModel


# class AttributeExtractNet(BertPreTrainedModel):
#     """
#     Attribute Extract Net with Multi-label Pointer Network(MPN) based Entity-aware and
#     encoded by BERT
#     """
#
#     def __init__(self, config, args, attribute_conf):
#         print('bert mpn baseline')
#         super(AttributeExtractNet, self).__init__(config, args, attribute_conf)
#
#         self.bert = BertModel(config)
#         self.token_entity_emb = nn.Embedding(num_embeddings=2, embedding_dim=config.hidden_size,
#                                              padding_idx=0)
#
#         # sentence_encoder using transformer
#         self.transformer_encoder_layer = TransformerEncoderLayer(config.hidden_size, args.nhead,
#                                                                  dim_feedforward=args.dim_feedforward)
#         self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, args.transformer_layers)
#
#         self.classes_num = len(attribute_conf)
#
#         # pointer net work
#         self.attr_start = nn.Linear(config.hidden_size, self.classes_num)
#         self.attr_end = nn.Linear(config.hidden_size, self.classes_num)
#
#         self.apply(self.init_bert_weights)
#
#     def forward(self, passage_id=None, token_type_id=None, segment_id=None, pos_start=None, pos_end=None, start_id=None,
#                 end_id=None, is_eval=False):
#         mask = passage_id.eq(0)
#         sent_mask = passage_id != 0
#
#         context_encoder, _ = self.bert(passage_id, segment_id, attention_mask=sent_mask,
#                                        output_all_encoded_layers=False)
#
#         token_entity_emb = self.token_entity_emb(token_type_id)
#
#         # sent encoder based entity-aware
#         sent_entity_encoder = context_encoder + token_entity_emb
#         transformer_encoder = self.transformer_encoder(sent_entity_encoder.transpose(1, 0),
#                                                        src_key_padding_mask=mask).transpose(0, 1)
#
#         attr_start = self.attr_start(transformer_encoder)
#         attr_end = self.attr_end(transformer_encoder)
#
#         loss_fct = nn.BCEWithLogitsLoss(reduction='none')
#
#         s1_loss = loss_fct(attr_start, start_id)
#         s1_loss = torch.sum(s1_loss, 2)
#         s1_loss = torch.sum(s1_loss * sent_mask.float()) / torch.sum(sent_mask.float()) / self.classes_num
#
#         s2_loss = loss_fct(attr_end, end_id)
#         s2_loss = torch.sum(s2_loss, 2)
#         s2_loss = torch.sum(s2_loss * sent_mask.float()) / torch.sum(sent_mask.float()) / self.classes_num
#
#         total_loss = s1_loss + s2_loss
#         po1 = nn.Sigmoid()(attr_start)
#         po2 = nn.Sigmoid()(attr_end)
#
#         return total_loss, po1, po2

class AttributeExtractNet(BertPreTrainedModel):
    """
    Attribute Extract Net with Multi-label Pointer Network(MPN) based Entity-aware and
    encoded by BERT
    """

    def __init__(self, config, args, attribute_conf):
        super(AttributeExtractNet, self).__init__(config, args, attribute_conf)
        print('bert debug - 2 ')

        self.bert = BertModel(config)
        self.token_entity_emb = nn.Embedding(num_embeddings=2, embedding_dim=config.hidden_size,
                                             padding_idx=0)

        # # sentence_encoder using transformer
        # self.transformer_encoder_layer = TransformerEncoderLayer(config.hidden_size, args.nhead,
        #                                                          dim_feedforward=args.dim_feedforward)
        # self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, args.transformer_layers)

        self.classes_num = len(attribute_conf)

        # pointer net work
        self.attr_start = nn.Linear(config.hidden_size, self.classes_num)
        self.attr_end = nn.Linear(config.hidden_size, self.classes_num)

        self.apply(self.init_bert_weights)

    def forward(self, passage_id=None, token_type_id=None, segment_id=None, pos_start=None, pos_end=None, start_id=None,
                end_id=None, is_eval=False):
        sent_mask = passage_id != 0

        context_encoder, _ = self.bert(passage_id, segment_id, attention_mask=sent_mask,
                                       output_all_encoded_layers=False)

        attr_start = self.attr_start(context_encoder)
        attr_end = self.attr_end(context_encoder)

        loss_fct = nn.BCEWithLogitsLoss(reduction='none')

        s1_loss = loss_fct(attr_start, start_id)
        s1_loss = torch.sum(s1_loss, 2)
        s1_loss = torch.sum(s1_loss * sent_mask.float()) / torch.sum(sent_mask.float()) / self.classes_num

        s2_loss = loss_fct(attr_end, end_id)
        s2_loss = torch.sum(s2_loss, 2)
        s2_loss = torch.sum(s2_loss * sent_mask.float()) / torch.sum(sent_mask.float()) / self.classes_num

        total_loss = s1_loss + s2_loss
        po1 = nn.Sigmoid()(attr_start)
        po2 = nn.Sigmoid()(attr_end)

        return total_loss, po1, po2
