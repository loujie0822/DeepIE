# _*_ coding:utf-8 _*_
import warnings

import numpy as np
import torch
import torch.nn as nn

from layers.encoders.transformers.bert.bert_model import BertModel

warnings.filterwarnings("ignore")
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer


class EntityNET(nn.Module):
    """
        ERENet : entity relation extraction
    """

    def __init__(self, args):
        super(EntityNET, self).__init__()

        self.sb1 = nn.Linear(args.bert_hidden_size, 1)
        self.sb2 = nn.Linear(args.bert_hidden_size, 1)

    def forward(self, sent_encoder, q_ids=None, eval_file=None, passages=None, s1=None, s2=None, is_eval=False):

        sequence_mask = passages != 0
        sb1 = self.sb1(sent_encoder).squeeze()
        sb2 = self.sb2(sent_encoder).squeeze()

        if not is_eval:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')

            sb1_loss = loss_fct(sb1, s1)
            s1_loss = torch.sum(sb1_loss * sequence_mask.float()) / torch.sum(sequence_mask.float())

            s2_loss = loss_fct(sb2, s2)
            s2_loss = torch.sum(s2_loss * sequence_mask.float()) / torch.sum(sequence_mask.float())

            ent_loss = s1_loss + s2_loss
            return ent_loss
        else:
            answer_list = self.predict(eval_file, q_ids, sb1, sb2)
            return answer_list

    def predict(self, eval_file, q_ids=None, sb1=None, sb2=None):
        answer_list = list()
        for qid, p1, p2 in zip(q_ids.cpu().numpy(),
                               sb1.cpu().numpy(),
                               sb2.cpu().numpy()):

            context = eval_file[qid].context
            start = None
            end = None
            threshold = 0.0
            positions = list()
            for idx in range(0, len(context) + 1):
                if idx == 0:
                    continue
                if p1[idx] > threshold and start is None:
                    start = idx
                if p2[idx] > threshold and end is None:
                    end = idx
                if start is not None and end is not None and start <= end:
                    positions.append((start, end + 1))
                    start = None
                    end = None
            answer_list.append(positions)

        return answer_list


class RelNET(nn.Module):
    """
        ERENet : entity relation extraction
    """

    def __init__(self, args, spo_conf):
        super(RelNET, self).__init__()
        self.token_entity_emb = nn.Embedding(num_embeddings=2, embedding_dim=args.bert_hidden_size,
                                             padding_idx=0)
        self.encoder_layer = TransformerEncoderLayer(args.bert_hidden_size, args.nhead)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, args.transformer_layers)

        self.classes_num = len(spo_conf)
        self.ob1 = nn.Linear(args.bert_hidden_size, self.classes_num)
        self.ob2 = nn.Linear(args.bert_hidden_size, self.classes_num)

    def forward(self, passages=None, sent_encoder=None, posit_ids=None, o1=None, o2=None, is_eval=False):
        mask = passages.eq(0)

        subject_encoder = sent_encoder + self.token_entity_emb(posit_ids)

        subject_encoder = torch.transpose(subject_encoder, 1, 0)
        transformer_encoder = self.transformer_encoder(subject_encoder, src_key_padding_mask=mask)
        transformer_encoder = torch.transpose(transformer_encoder, 0, 1)

        po1 = self.ob1(transformer_encoder)
        po2 = self.ob2(transformer_encoder)

        if not is_eval:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')

            sequence_mask = passages != 0

            s1_loss = loss_fct(po1, o1)
            s1_loss = torch.sum(s1_loss, 2)
            s1_loss = torch.sum(s1_loss * sequence_mask.float()) / torch.sum(sequence_mask.float()) / self.classes_num

            s2_loss = loss_fct(po2, o2)
            s2_loss = torch.sum(s2_loss, 2)
            s2_loss = torch.sum(s2_loss * sequence_mask.float()) / torch.sum(sequence_mask.float()) / self.classes_num

            rel_loss = s1_loss + s2_loss

            return rel_loss

        else:
            po1 = nn.Sigmoid()(po1)
            po2 = nn.Sigmoid()(po2)
            return po1, po2


class ERENet(nn.Module):
    """
        ERENet : entity relation extraction
    """

    def __init__(self, args, spo_conf):
        super(ERENet, self).__init__()
        print('joint entity relation extraction')
        self.bert_encoder = BertModel.from_pretrained(args.bert_model)
        self.entity_extraction = EntityNET(args)
        self.rel_extraction = RelNET(args, spo_conf)

    def forward(self, q_ids=None, eval_file=None, passages=None, token_type_ids=None, segment_ids=None, s1=None,
                s2=None, po1=None, po2=None, is_eval=False):

        sequence_mask = passages != 0
        sent_encoder, _ = self.bert_encoder(passages, token_type_ids=segment_ids, attention_mask=sequence_mask,
                                            output_all_encoded_layers=False)

        if not is_eval:
            # entity_extraction
            ent_loss = self.entity_extraction(sent_encoder, passages=passages, s1=s1, s2=s2,
                                              is_eval=is_eval)

            # rel_extraction
            rel_loss = self.rel_extraction(passages=passages, sent_encoder=sent_encoder, posit_ids=token_type_ids,
                                           o1=po1,
                                           o2=po2, is_eval=False)

            # add total loss
            total_loss = ent_loss + rel_loss

            return total_loss


        else:

            answer_list = self.entity_extraction(sent_encoder, q_ids=q_ids, eval_file=eval_file,
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
                    passages = passages.expand(2, passages.size(1))
                    sent_encoder = sent_encoder.expand(2, sent_encoder.size(1), sent_encoder.size(2))
                    posit_ids = posit_ids.expand(2, posit_ids.size(1))

                po1, po2 = self.rel_extraction(passages=passages, sent_encoder=sent_encoder, posit_ids=posit_ids,
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
