import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from layers.encoders.rnns.stacked_rnn import StackedBRNN
from layers.encoders.transformers.bert.layernorm import ConditionalLayerNorm
from utils.data_util import batch_gather

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
    ERENet : entity relation jointed extraction
    """

    def __init__(self, args, word_emb, spo_conf):
        print('using only char2v+w2v mixed  and word_emb is freeze ')
        super(ERENet, self).__init__()

        self.word_emb = nn.Embedding.from_pretrained(torch.tensor(word_emb, dtype=torch.float32), freeze=True,
                                                     padding_idx=0)
        self.char_emb = nn.Embedding(num_embeddings=args.char_vocab_size, embedding_dim=args.char_emb_size,
                                     padding_idx=0)

        self.word_convert_char = nn.Linear(args.word_emb_size, args.char_emb_size, bias=False)

        self.classes_num = len(spo_conf)

        self.first_sentence_encoder = SentenceEncoder(args, args.char_emb_size)
        # self.second_sentence_encoder = SentenceEncoder(args, args.hidden_size)
        # self.token_entity_emb = nn.Embedding(num_embeddings=2, embedding_dim=config.hidden_size,
        #                                      padding_idx=0)
        self.encoder_layer = TransformerEncoderLayer(args.hidden_size * 2, nhead=3)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=1)
        self.LayerNorm = ConditionalLayerNorm(args.hidden_size * 2, eps=1e-12)
        # pointer net work
        self.po_dense = nn.Linear(args.hidden_size * 2, self.classes_num * 2)
        self.subject_dense = nn.Linear(args.hidden_size * 2, 2)
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, q_ids=None, char_ids=None, word_ids=None, token_type_ids=None, subject_ids=None,
                subject_labels=None,
                object_labels=None, eval_file=None,
                is_eval=False):

        mask = char_ids != 0

        seq_mask = char_ids.eq(0)

        char_emb = self.char_emb(char_ids)
        word_emb = self.word_convert_char(self.word_emb(word_ids))
        # word_emb = self.word_emb(word_ids)
        emb = char_emb + word_emb
        # emb = char_emb
        # subject_encoder = sent_encoder + self.token_entity_emb(token_type_id)
        sent_encoder = self.first_sentence_encoder(emb, seq_mask)

        if not is_eval:
            # subject_encoder = self.token_entity_emb(token_type_ids)
            # context_encoder = bert_encoder + subject_encoder

            sub_start_encoder = batch_gather(sent_encoder, subject_ids[:, 0])
            sub_end_encoder = batch_gather(sent_encoder, subject_ids[:, 1])
            subject = torch.cat([sub_start_encoder, sub_end_encoder], 1)
            context_encoder = self.LayerNorm(sent_encoder, subject)
            context_encoder = self.transformer_encoder(context_encoder.transpose(1, 0),
                                                       src_key_padding_mask=seq_mask).transpose(0, 1)

            sub_preds = self.subject_dense(sent_encoder)
            po_preds = self.po_dense(context_encoder).reshape(char_ids.size(0), -1, self.classes_num, 2)

            subject_loss = self.loss_fct(sub_preds, subject_labels)
            subject_loss = subject_loss.mean(2)
            subject_loss = torch.sum(subject_loss * mask.float()) / torch.sum(mask.float())

            po_loss = self.loss_fct(po_preds, object_labels)
            po_loss = torch.sum(po_loss.mean(3), 2)
            po_loss = torch.sum(po_loss * mask.float()) / torch.sum(mask.float())

            loss = subject_loss + po_loss

            return loss

        else:

            subject_preds = nn.Sigmoid()(self.subject_dense(sent_encoder))
            answer_list = list()
            for qid, sub_pred in zip(q_ids.cpu().numpy(),
                                     subject_preds.cpu().numpy()):
                context = eval_file[qid].context
                start = np.where(sub_pred[:, 0] > 0.5)[0]
                end = np.where(sub_pred[:, 1] > 0.4)[0]
                subjects = []
                for i in start:
                    j = end[end >= i]
                    if i >= len(context):
                        continue
                    if len(j) > 0:
                        j = j[0]
                        if j >= len(context):
                            continue
                        subjects.append((i, j))

                answer_list.append(subjects)

            qid_ids, sent_encoders, pass_ids, subject_ids, token_type_ids = [], [], [], [], []
            for i, subjects in enumerate(answer_list):
                if subjects:
                    qid = q_ids[i].unsqueeze(0).expand(len(subjects))
                    pass_tensor = char_ids[i, :].unsqueeze(0).expand(len(subjects), char_ids.size(1))
                    new_sent_encoder = sent_encoder[i, :, :].unsqueeze(0).expand(len(subjects), sent_encoder.size(1),
                                                                                 sent_encoder.size(2))

                    token_type_id = torch.zeros((len(subjects), char_ids.size(1)), dtype=torch.long)
                    for index, (start, end) in enumerate(subjects):
                        token_type_id[index, start:end + 1] = 1

                    qid_ids.append(qid)
                    pass_ids.append(pass_tensor)
                    subject_ids.append(torch.tensor(subjects, dtype=torch.long))
                    sent_encoders.append(new_sent_encoder)
                    token_type_ids.append(token_type_id)

            if len(qid_ids) == 0:
                # print('len(qid_list)==0:')
                qid_tensor = torch.tensor([-1, -1], dtype=torch.long).to(sent_encoder.device)
                return qid_tensor, qid_tensor, qid_tensor

            # print('len(qid_list)!=========================0:')
            qids = torch.cat(qid_ids).to(sent_encoder.device)
            pass_ids = torch.cat(pass_ids).to(sent_encoder.device)
            sent_encoders = torch.cat(sent_encoders).to(sent_encoder.device)
            # token_type_ids = torch.cat(token_type_ids).to(bert_encoder.device)
            subject_ids = torch.cat(subject_ids).to(sent_encoder.device)

            flag = False
            split_heads = 1024

            sent_encoders_ = torch.split(sent_encoders, split_heads, dim=0)
            pass_ids_ = torch.split(pass_ids, split_heads, dim=0)
            # token_type_ids_ = torch.split(token_type_ids, split_heads, dim=0)
            subject_encoder_ = torch.split(subject_ids, split_heads, dim=0)
            # print('len(qid_list)!=========================1:')
            po_preds = list()
            for i in range(len(subject_encoder_)):
                sent_encoders = sent_encoders_[i]
                # token_type_ids = token_type_ids_[i]
                pass_ids = pass_ids_[i]
                subject_encoder = subject_encoder_[i]

                if sent_encoders.size(0) == 1:
                    flag = True
                    # print('flag = True**********')
                    sent_encoders = sent_encoders.expand(2, sent_encoders.size(1), sent_encoders.size(2))
                    subject_encoder = subject_encoder.expand(2, subject_encoder.size(1))
                    pass_ids = pass_ids.expand(2, pass_ids.size(1))
                # print('len(qid_list)!=========================2:')
                sub_start_encoder = batch_gather(sent_encoders, subject_encoder[:, 0])
                sub_end_encoder = batch_gather(sent_encoders, subject_encoder[:, 1])
                subject = torch.cat([sub_start_encoder, sub_end_encoder], 1)
                context_encoder = self.LayerNorm(sent_encoders, subject)
                context_encoder = self.transformer_encoder(context_encoder.transpose(1, 0),
                                                           src_key_padding_mask=pass_ids.eq(0)).transpose(0, 1)
                # print('len(qid_list)!=========================3')
                # context_encoder = self.LayerNorm(context_encoder)
                po_pred = self.po_dense(context_encoder).reshape(subject_encoder.size(0), -1, self.classes_num, 2)

                if flag:
                    po_pred = po_pred[1, :, :, :].unsqueeze(0)

                po_preds.append(po_pred)

            po_tensor = torch.cat(po_preds).to(qids.device)
            po_tensor = nn.Sigmoid()(po_tensor)
            return qids, subject_ids, po_tensor
