import warnings

import numpy as np
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


class EntityNET(nn.Module):
    """
        EntityNET : entity extraction using pointer network
    """

    def __init__(self, args, char_emb):
        super(EntityNET, self).__init__()

        if char_emb is not None:
            self.char_emb = nn.Embedding.from_pretrained(torch.tensor(char_emb, dtype=torch.float32), freeze=False,
                                                         padding_idx=0)
        else:
            self.char_emb = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.char_emb_size,
                                         padding_idx=0)

        self.sentence_encoder = SentenceEncoder(args, args.word_emb_size)
        self.s1 = nn.Linear(args.hidden_size * 2, 1)
        self.s2 = nn.Linear(args.hidden_size * 2, 1)

    def forward(self, q_ids=None, eval_file=None, passages=None, s1=None, s2=None, is_eval=False):
        mask = passages.eq(0)
        sequence_mask = passages != 0

        char_emb = self.char_emb(passages)

        sent_encoder = self.sentence_encoder(char_emb, mask)

        s1_ = self.s1(sent_encoder).squeeze()
        s2_ = self.s2(sent_encoder).squeeze()

        if not is_eval:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')

            sb1_loss = loss_fct(s1_, s1)
            s1_loss = torch.sum(sb1_loss * sequence_mask.float()) / torch.sum(sequence_mask.float())

            s2_loss = loss_fct(s2_, s2)
            s2_loss = torch.sum(s2_loss * sequence_mask.float()) / torch.sum(sequence_mask.float())

            ent_loss = s1_loss + s2_loss
            return sent_encoder, ent_loss
        else:
            answer_list = self.predict(eval_file, q_ids, s1_, s2_)
            return sent_encoder, answer_list

    def predict(self, eval_file, q_ids=None, s1=None, s2=None):
        sub_ans_list = list()
        for qid, p1, p2 in zip(q_ids.cpu().numpy(),
                               s1.cpu().numpy(),
                               s2.cpu().numpy()):
            start = None
            end = None
            threshold = 0.0
            positions = list()
            for idx in range(0, len(eval_file[qid].context)):
                if p1[idx] > threshold and start is None:
                    start = idx
                if p2[idx] > threshold and end is None:
                    end = idx
                if start is not None and end is not None and start <= end:
                    positions.append((start, end + 1))
                    start = None
                    end = None
            sub_ans_list.append(positions)

        return sub_ans_list


class RelNET(nn.Module):
    """
        ERENet : entity relation extraction
    """

    def __init__(self, args, spo_conf):
        super(RelNET, self).__init__()
        self.token_entity_emb = nn.Embedding(num_embeddings=2, embedding_dim=args.entity_emb_size,
                                             padding_idx=0)
        self.sentence_encoder = SentenceEncoder(args, args.word_emb_size)
        self.transformer_encoder_layer = TransformerEncoderLayer(args.hidden_size * 2, args.nhead)

        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, args.transformer_layers)

        self.classes_num = len(spo_conf)
        self.po1 = nn.Linear(args.hidden_size * 2, self.classes_num)
        self.po2 = nn.Linear(args.hidden_size * 2, self.classes_num)

    def forward(self, passages=None, sent_encoder=None, token_type_id=None, po1=None, po2=None, is_eval=False):
        mask = passages.eq(0)
        sequence_mask = passages != 0

        subject_encoder = sent_encoder + self.token_entity_emb(token_type_id)
        sent_sub_aware_encoder = self.sentence_encoder(subject_encoder, mask).transpose(1, 0)

        transformer_encoder = self.transformer_encoder(sent_sub_aware_encoder, src_key_padding_mask=mask).transpose(0,
                                                                                                                    1)

        po1_ = self.po1(transformer_encoder)
        po2_ = self.po2(transformer_encoder)

        if not is_eval:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')

            po1_loss = loss_fct(po1_, po1)
            po1_loss = torch.sum(po1_loss, 2)
            po1_loss = torch.sum(po1_loss * sequence_mask.float()) / torch.sum(sequence_mask.float())

            po2_loss = loss_fct(po2_, po2)
            po2_loss = torch.sum(po2_loss, 2)
            po2_loss = torch.sum(po2_loss * sequence_mask.float()) / torch.sum(sequence_mask.float())

            rel_loss = po1_loss + po2_loss

            return rel_loss

        else:
            po1 = nn.Sigmoid()(po1_)
            po2 = nn.Sigmoid()(po2_)
            return po1, po2


class ERENet(nn.Module):
    """
        ERENet : entity relation jointed extraction with Multi-label Pointer Network(MPN) based Entity-aware
    """

    def __init__(self, args, char_emb, spo_conf):
        super(ERENet, self).__init__()
        print('joint entity relation extraction')
        self.entity_extraction = EntityNET(args, char_emb)
        self.rel_extraction = RelNET(args, spo_conf)

    def forward(self, q_ids=None, eval_file=None, passages=None, token_type_ids=None, segment_ids=None, s1=None,
                s2=None, po1=None, po2=None, is_eval=False):

        if not is_eval:
            sent_encoder, ent_loss = self.entity_extraction(passages=passages, s1=s1, s2=s2, is_eval=is_eval)
            rel_loss = self.rel_extraction(passages=passages, sent_encoder=sent_encoder, token_type_id=token_type_ids,
                                           po1=po1, po2=po2, is_eval=False)
            total_loss = ent_loss + rel_loss

            return total_loss
        else:

            sent_encoder, answer_list = self.entity_extraction(q_ids=q_ids, eval_file=eval_file,
                                                               passages=passages, is_eval=is_eval)
            start_list, end_list = list(), list()
            qid_list, pass_list, posit_list, sent_list = list(), list(), list(), list()
            for i, ans_list in enumerate(answer_list):
                seq_len = passages.size(1)
                posit_ids = []
                for ans_tuple in ans_list:
                    posit_array = np.zeros(seq_len, dtype=np.int)
                    start, end = ans_tuple[0], ans_tuple[1]
                    start_list.append(start)
                    end_list.append(end)
                    posit_array[start:end] = 1
                    posit_ids.append(posit_array)

                if len(posit_ids) == 0:
                    continue
                qid_ = q_ids[i].unsqueeze(0).expand(len(posit_ids))
                sent_tensor = sent_encoder[i, :, :].unsqueeze(0).expand(len(posit_ids), sent_encoder.size(1),
                                                                        sent_encoder.size(2))
                pass_tensor = passages[i, :].unsqueeze(0).expand(len(posit_ids), passages.size(1))
                posit_tensor = torch.tensor(posit_ids, dtype=torch.long).to(sent_encoder.device)

                qid_list.append(qid_)
                pass_list.append(pass_tensor)
                posit_list.append(posit_tensor)
                sent_list.append(sent_tensor)

            if len(qid_list) == 0:
                # print('len(qid_list)==0:')
                qid_tensor = torch.tensor([-1, -1], dtype=torch.long).to(sent_encoder.device)
                return qid_tensor, qid_tensor, qid_tensor, qid_tensor, qid_tensor

            qid_tensor = torch.cat(qid_list).to(sent_encoder.device)
            sent_tensor = torch.cat(sent_list).to(sent_encoder.device)
            pass_tensor = torch.cat(pass_list).to(sent_encoder.device)
            posi_tensor = torch.cat(posit_list).to(sent_encoder.device)

            flag = False
            split_heads = 1024

            inputs = torch.split(pass_tensor, split_heads, dim=0)
            posits = torch.split(posi_tensor, split_heads, dim=0)
            sents = torch.split(sent_tensor, split_heads, dim=0)

            po1_list, po2_list = list(), list()
            for i in range(len(inputs)):
                passages = inputs[i]
                sent_encoder = sents[i]
                posit_ids = posits[i]

                if passages.size(0) == 1:
                    flag = True
                    # print('flag = True**********')
                    passages = passages.expand(2, passages.size(1))
                    sent_encoder = sent_encoder.expand(2, sent_encoder.size(1), sent_encoder.size(2))
                    posit_ids = posit_ids.expand(2, posit_ids.size(1))

                po1, po2 = self.rel_extraction(passages=passages, sent_encoder=sent_encoder, token_type_id=posit_ids,
                                               is_eval=is_eval)
                if flag:
                    po1 = po1[1, :, :].unsqueeze(0)
                    po2 = po2[1, :, :].unsqueeze(0)

                po1_list.append(po1)
                po2_list.append(po2)

            po1_tensor = torch.cat(po1_list).to(sent_encoder.device)
            po2_tensor = torch.cat(po2_list).to(sent_encoder.device)

            s_tensor = torch.tensor(start_list, dtype=torch.long).to(sent_encoder.device)
            e_tensor = torch.tensor(end_list, dtype=torch.long).to(sent_encoder.device)

            return qid_tensor, po1_tensor, po2_tensor, s_tensor, e_tensor
