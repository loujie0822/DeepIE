import warnings

import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

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


# class AttributeExtractNet(nn.Module):
#     """
#     Attribute Extract Net with Multi-label Pointer Network(MPN) based Entity-aware
#     实体感知方式:char_emb + token_entity_emb+pos_start_emb+pos_end_emb
#     """
#
#     def __init__(self, args, char_emb, attribute_conf):
#         super(AttributeExtractNet, self).__init__()
#         print('实体感知方式:char_emb + token_entity_emb+pos_start_emb+pos_end_emb')
#
#         if char_emb is not None:
#             self.char_emb = nn.Embedding.from_pretrained(torch.tensor(char_emb, dtype=torch.float32), freeze=False,
#                                                          padding_idx=0)
#         else:
#             self.char_emb = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.char_emb_size,
#                                          padding_idx=0)
#
#         self.pos_start = nn.Embedding(num_embeddings=args.pos_size, embedding_dim=args.pos_dim,
#                                       padding_idx=0)
#         self.pos_end = nn.Embedding(num_embeddings=args.pos_size, embedding_dim=args.pos_dim,
#                                     padding_idx=0)
#         # token whether belong to a entity, 1 represent a entity token, else 0;
#         self.token_entity_emb = nn.Embedding(num_embeddings=2, embedding_dim=args.entity_emb_size,
#                                              padding_idx=0)
#         # sentence_encoder using lstm
#         self.first_sentence_encoder = SentenceEncoder(args, args.char_emb_size)
#         self.second_sentence_encoder = SentenceEncoder(args, args.char_emb_size)
#
#         # sentence_encoder using transformer
#         self.transformer_encoder_layer = TransformerEncoderLayer(args.hidden_size * 2, args.nhead,
#                                                                  dim_feedforward=args.dim_feedforward)
#         self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, args.transformer_layers)
#
#         self.classes_num = len(attribute_conf)
#
#         # pointer net work
#         self.attr_start = nn.Linear(args.hidden_size * 2, self.classes_num)
#         self.attr_end = nn.Linear(args.hidden_size * 2, self.classes_num)
#
#     def forward(self, passage_id=None, token_type_id=None, segment_id=None, pos_start=None, pos_end=None, start_id=None,
#                 end_id=None, is_eval=False):
#         mask = passage_id.eq(0)
#         sent_mask = passage_id != 0
#
#         char_emb = self.char_emb(passage_id)
#         pos_start_emb = self.pos_start(pos_start)
#         pos_end_emb = self.pos_end(pos_end)
#         token_entity_emb = self.token_entity_emb(token_type_id)
#         # sent encoder based entity-aware
#         sent_entity_encoder = char_emb + token_entity_emb+pos_start_emb+pos_end_emb
#         sent_first_encoder = self.first_sentence_encoder(sent_entity_encoder, mask).transpose(1, 0)
#         transformer_encoder = self.transformer_encoder(sent_first_encoder, src_key_padding_mask=mask).transpose(0, 1)
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

# class AttributeExtractNet(nn.Module):
#     """
#     Attribute Extract Net with Multi-label Pointer Network(MPN) based Entity-aware
#     实体感知方式：sent-encoder+pos_start+pos_end
#     """
#
#     def __init__(self, args, char_emb, attribute_conf):
#         super(AttributeExtractNet, self).__init__()
#         print('实体感知方式: sent-encoder+pos_start')
#         if char_emb is not None:
#             self.char_emb = nn.Embedding.from_pretrained(torch.tensor(char_emb, dtype=torch.float32), freeze=False,
#                                                          padding_idx=0)
#         else:
#             self.char_emb = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.char_emb_size,
#                                          padding_idx=0)
#
#         self.pos_start = nn.Embedding(num_embeddings=args.pos_size, embedding_dim=args.pos_dim,
#                                      padding_idx=0)
#         self.pos_end = nn.Embedding(num_embeddings=args.pos_size, embedding_dim=args.pos_dim,
#                                      padding_idx=0)
#
#         # token whether belong to a entity, 1 represent a entity token, else 0;
#         self.token_entity_emb = nn.Embedding(num_embeddings=2, embedding_dim=args.entity_emb_size,
#                                              padding_idx=0)
#         # sentence_encoder using lstm
#         self.first_sentence_encoder = SentenceEncoder(args, args.char_emb_size)
#         self.second_sentence_encoder = SentenceEncoder(args, args.char_emb_size)
#
#         # sentence_encoder using transformer
#         self.transformer_encoder_layer = TransformerEncoderLayer(args.hidden_size * 2, args.nhead,
#                                                                  dim_feedforward=args.dim_feedforward)
#         self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, args.transformer_layers)
#
#         self.classes_num = len(attribute_conf)
#
#         # pointer net work
#         self.attr_start = nn.Linear(args.hidden_size * 2, self.classes_num)
#         self.attr_end = nn.Linear(args.hidden_size * 2, self.classes_num)
#
#     def forward(self, passage_id=None, token_type_id=None, segment_id=None, pos_start=None, pos_end=None, start_id=None,
#                 end_id=None, is_eval=False):
#         mask = passage_id.eq(0)
#         sent_mask = passage_id != 0
#
#         char_emb = self.char_emb(passage_id)
#         pos_start_emb = self.pos_start(pos_start)
#         pos_end_emb = self.pos_end(pos_end)
#         token_entity_emb = self.token_entity_emb(token_type_id)
#
#         sent_first_encoder = self.first_sentence_encoder(char_emb, mask)
#
#         # sent encoder based entity-aware
#         sent_entity_aware = sent_first_encoder+pos_start_emb
#
#         sent_second_encoder = self.second_sentence_encoder(sent_entity_aware, mask).transpose(1, 0)
#         transformer_encoder = self.transformer_encoder(sent_second_encoder, src_key_padding_mask=mask).transpose(0, 1)
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


# class AttributeExtractNet(nn.Module):
#     """
#     Attribute Extract Net with Multi-label Pointer Network(MPN) based Entity-aware
#     实体感知方式：[sent-encoder, token_ent_type, global_repre]
#     """
#
#     def __init__(self, args, char_emb, attribute_conf):
#         super(AttributeExtractNet, self).__init__()
#         print('实体感知方式：[sent-encoder, token_ent_type, global_repre]')
#         if char_emb is not None:
#             self.char_emb = nn.Embedding.from_pretrained(torch.tensor(char_emb, dtype=torch.float32), freeze=False,
#                                                          padding_idx=0)
#         else:
#             self.char_emb = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.char_emb_size,
#                                          padding_idx=0)
#
#         # token whether belong to a entity, 1 represent a entity token, else 0;
#         self.token_entity_emb = nn.Embedding(num_embeddings=2, embedding_dim=args.entity_emb_size,
#                                              padding_idx=0)
#         # sentence_encoder using lstm
#         self.first_sentence_encoder = SentenceEncoder(args, args.char_emb_size)
#         self.second_sentence_encoder = SentenceEncoder(args, args.char_emb_size * 3)
#
#         # sentence_encoder using transformer
#         self.transformer_encoder_layer = TransformerEncoderLayer(args.hidden_size * 2, args.nhead,
#                                                                  dim_feedforward=args.dim_feedforward)
#         self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, args.transformer_layers)
#
#         self.classes_num = len(attribute_conf)
#
#         # pointer net work
#         self.attr_start = nn.Linear(args.hidden_size * 2, self.classes_num)
#         self.attr_end = nn.Linear(args.hidden_size * 2, self.classes_num)
#
#     def forward(self, passage_id=None, token_type_id=None, segment_id=None, pos_start=None, pos_end=None, start_id=None,
#                 end_id=None, is_eval=False):
#         mask = passage_id.eq(0)
#         sent_mask = passage_id != 0
#
#         char_emb = self.char_emb(passage_id)
#         token_entity_emb = self.token_entity_emb(token_type_id)
#
#         sent_first_encoder = self.first_sentence_encoder(char_emb, mask)
#         global_encoder_, _ = torch.max(sent_first_encoder, 1)
#         global_encoder = global_encoder_.unsqueeze(1).expand_as(sent_first_encoder)
#         # sent encoder based entity-aware
#         sent_entity_aware = torch.cat([sent_first_encoder, token_entity_emb, global_encoder], -1)
#
#         sent_second_encoder = self.second_sentence_encoder(sent_entity_aware, mask).transpose(1, 0)
#         transformer_encoder = self.transformer_encoder(sent_second_encoder, src_key_padding_mask=mask).transpose(0, 1)
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

class AttributeExtractNet(nn.Module):
    """
    Attribute Extract Net with Multi-label Pointer Network(MPN) based Entity-aware
    实体感知方式：token_entity_embedding
    """

    def __init__(self, args, char_emb, attribute_conf):
        print('basline ')
        super(AttributeExtractNet, self).__init__()
        if char_emb is not None:
            self.char_emb = nn.Embedding.from_pretrained(torch.tensor(char_emb, dtype=torch.float32), freeze=False,
                                                         padding_idx=0)
        else:
            self.char_emb = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.char_emb_size,
                                         padding_idx=0)

        # token whether belong to a entity, 1 represent a entity token, else 0;
        self.token_entity_emb = nn.Embedding(num_embeddings=2, embedding_dim=args.entity_emb_size,
                                             padding_idx=0)
        # sentence_encoder using lstm
        self.first_sentence_encoder = SentenceEncoder(args, args.char_emb_size)
        self.second_sentence_encoder = SentenceEncoder(args, args.char_emb_size)

        # sentence_encoder using transformer
        self.transformer_encoder_layer = TransformerEncoderLayer(args.hidden_size * 2, args.nhead,
                                                                 dim_feedforward=args.dim_feedforward)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, args.transformer_layers)

        self.classes_num = len(attribute_conf)

        # pointer net work
        self.attr_start = nn.Linear(args.hidden_size * 2, self.classes_num)
        self.attr_end = nn.Linear(args.hidden_size * 2, self.classes_num)

    def forward(self, passage_id=None, token_type_id=None, segment_id=None, pos_start=None, pos_end=None, start_id=None,
                end_id=None, is_eval=False):
        mask = passage_id.eq(0)
        sent_mask = passage_id != 0

        char_emb = self.char_emb(passage_id)
        token_entity_emb = self.token_entity_emb(token_type_id)

        sent_first_encoder = self.first_sentence_encoder(char_emb, mask)
        # sent encoder based entity-aware
        sent_entity_encoder = sent_first_encoder + token_entity_emb
        sent_second_encoder = self.second_sentence_encoder(sent_entity_encoder, mask).transpose(1, 0)
        transformer_encoder = self.transformer_encoder(sent_second_encoder, src_key_padding_mask=mask).transpose(0, 1)

        attr_start = self.attr_start(transformer_encoder)
        attr_end = self.attr_end(transformer_encoder)

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

# class AttributeExtractNet(nn.Module):
#     """
#     Attribute Extract Net with Multi-label Pointer Network(MPN) based Entity-aware
#     实体感知方式:显示编码，将实体替换为<MASK>
#     """
#
#     def __init__(self, args, char_emb, attribute_conf):
#         super(AttributeExtractNet, self).__init__()
#         print('实体感知方式:显示编码，将实体替换为<MASK>')
#
#
#         if char_emb is not None:
#             self.char_emb = nn.Embedding.from_pretrained(torch.tensor(char_emb, dtype=torch.float32), freeze=False,
#                                                          padding_idx=0)
#         else:
#             self.char_emb = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.char_emb_size,
#                                          padding_idx=0)
#
#         # token whether belong to a entity, 1 represent a entity token, else 0;
#         self.token_entity_emb = nn.Embedding(num_embeddings=2, embedding_dim=args.entity_emb_size,
#                                              padding_idx=0)
#         # sentence_encoder using lstm
#         self.first_sentence_encoder = SentenceEncoder(args, args.char_emb_size)
#         self.second_sentence_encoder = SentenceEncoder(args, args.char_emb_size)
#
#         # sentence_encoder using transformer
#         self.transformer_encoder_layer = TransformerEncoderLayer(args.hidden_size * 2, args.nhead,
#                                                                  dim_feedforward=args.dim_feedforward)
#         self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, args.transformer_layers)
#
#         self.classes_num = len(attribute_conf)
#
#         # pointer net work
#         self.attr_start = nn.Linear(args.hidden_size * 2, self.classes_num)
#         self.attr_end = nn.Linear(args.hidden_size * 2, self.classes_num)
#
#     def forward(self, passage_id=None, token_type_id=None, segment_id=None, start_id=None, end_id=None, is_eval=False):
#         mask = passage_id.eq(0)
#         sent_mask = passage_id != 0
#
#         char_emb = self.char_emb(passage_id)
#         # token_entity_emb = self.token_entity_emb(token_type_id)
#
#         # char_emb = char_emb + token_entity_emb
#
#         sent_first_encoder = self.first_sentence_encoder(char_emb, mask).transpose(1, 0)
#         # sent encoder based entity-aware
#         # sent_entity_encoder = sent_first_encoder + token_entity_emb
#         # sent_second_encoder = self.second_sentence_encoder(sent_first_encoder, mask).transpose(1, 0)
#         transformer_encoder = self.transformer_encoder(sent_first_encoder, src_key_padding_mask=mask).transpose(0, 1)
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
