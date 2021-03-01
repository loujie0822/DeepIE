import copy
import warnings

import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from layers.encoders.rnns.stacked_rnn import StackedBRNN
from layers.encoders.transformers.bert.layernorm import ConditionalLayerNorm
from utils.data_util import batch_gather

warnings.filterwarnings("ignore")

from layers.decoders.pytorch_crf import CRF


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

    def __init__(self, args, word_emb, ent_conf, spo_conf):
        print('mhs using only char2v+w2v mixed  and word_emb is freeze ')
        super(ERENet, self).__init__()

        self.max_len = args.max_len

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
        # self.subject_dense = nn.Linear(args.hidden_size * 2, 2)

        self.ent_emission = nn.Linear(args.hidden_size * 2, len(ent_conf))
        self.ent_crf = CRF(len(ent_conf), batch_first=True)
        self.emission = nn.Linear(args.hidden_size * 2, len(spo_conf))
        self.crf = CRF(len(spo_conf), batch_first=True)
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
        ent_emission = self.ent_emission(sent_encoder)
        if not is_eval:
            # subject_encoder = self.token_entity_emb(token_type_ids)
            # context_encoder = bert_encoder + subject_encoder

            sub_start_encoder = batch_gather(sent_encoder, subject_ids[:, 0])
            sub_end_encoder = batch_gather(sent_encoder, subject_ids[:, 1])
            subject = torch.cat([sub_start_encoder, sub_end_encoder], 1)
            context_encoder = self.LayerNorm(sent_encoder, subject)
            context_encoder = self.transformer_encoder(context_encoder.transpose(1, 0),
                                                       src_key_padding_mask=seq_mask).transpose(0, 1)

            ent_loss = -self.ent_crf(ent_emission, subject_labels, mask=mask, reduction='mean')

            emission = self.emission(context_encoder)
            po_loss = -self.crf(emission, object_labels, mask=mask, reduction='mean')
            loss = ent_loss + po_loss

            return loss

        else:
            subject_preds = self.ent_crf.decode(emissions=ent_emission, mask=mask)
            answer_list = list()
            for qid, sub_pred in zip(q_ids.cpu().numpy(), subject_preds):
                seq_len = min(len(eval_file[qid].context), self.max_len)
                tag_list = list()
                j = 0
                while j < seq_len:
                    end = j
                    flag = True

                    if sub_pred[j] == 1:
                        start = j
                        for k in range(start + 1, seq_len):
                            if sub_pred[k] != sub_pred[start] + 1:
                                end = k - 1
                                flag = False
                                break
                        if flag:
                            end = seq_len - 1
                        tag_list.append((start, end))
                    j = end + 1

                answer_list.append(tag_list)

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
                subject_ids = torch.zeros(1, 2).long().to(sent_encoder.device)
                qid_tensor = torch.tensor([-1], dtype=torch.long).to(sent_encoder.device)
                po_tensor = torch.zeros(1, sent_encoder.size(1)).long().to(sent_encoder.device)
                return qid_tensor, subject_ids, po_tensor

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
            po_preds = list()
            for i in range(len(subject_encoder_)):
                sent_encoders = sent_encoders_[i]
                # token_type_ids = token_type_ids_[i]
                pass_ids = pass_ids_[i]
                subject_encoder = subject_encoder_[i]

                if sent_encoders.size(0) == 1:
                    flag = True
                    sent_encoders = sent_encoders.expand(2, sent_encoders.size(1), sent_encoders.size(2))
                    subject_encoder = subject_encoder.expand(2, subject_encoder.size(1))
                    pass_ids = pass_ids.expand(2, pass_ids.size(1))
                sub_start_encoder = batch_gather(sent_encoders, subject_encoder[:, 0])
                sub_end_encoder = batch_gather(sent_encoders, subject_encoder[:, 1])
                subject = torch.cat([sub_start_encoder, sub_end_encoder], 1)
                context_encoder = self.LayerNorm(sent_encoders, subject)
                context_encoder = self.transformer_encoder(context_encoder.transpose(1, 0),
                                                           src_key_padding_mask=pass_ids.eq(0)).transpose(0, 1)
                emission = self.emission(context_encoder)
                po_pred = self.crf.decode(emissions=emission, mask=(pass_ids != 0))
                max_len = pass_ids.size(1)
                temp_tag = copy.deepcopy(po_pred)
                for line in temp_tag:
                    line.extend([0] * (max_len - len(line)))
                # TODO:check
                po_pred = torch.tensor(temp_tag).to(emission.device)
                if flag:
                    po_pred = po_pred[1, :].unsqueeze(0)

                po_preds.append(po_pred)
            po_tensor = torch.cat(po_preds).to(qids.device)
            # print(subject_ids.device)
            # print(po_tensor.device)
            # print(qids.shape)
            # print(subject_ids.shape)
            # print(po_tensor.shape)
            return qids, subject_ids, po_tensor
