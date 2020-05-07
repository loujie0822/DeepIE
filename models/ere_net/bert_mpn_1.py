# _*_ coding:utf-8 _*_
import warnings

import numpy as np
import torch
import torch.nn as nn

from layers.encoders.transformers.bert.bert_model import BertModel
from layers.encoders.transformers.bert.bert_pretrain import BertPreTrainedModel

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
            for idx in range(0, len(context)):
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


class EREdNet(nn.Module):
    """
        ERENet : entity relation extraction
    """

    def __init__(self, args, spo_conf):
        super(EREdNet, self).__init__()
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


class ERENet(BertPreTrainedModel):
    """
    ERENet : entity relation jointed extraction
    """

    def __init__(self, config, classes_num):
        super(ERENet, self).__init__(config, classes_num)
        self.classes_num = classes_num
        self.bert = BertModel(config)
        self.token_entity_emb = nn.Embedding(num_embeddings=2, embedding_dim=config.hidden_size,
                                             padding_idx=0)
        self.encoder_layer = TransformerEncoderLayer(config.hidden_size, nhead=4)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=1)
        # self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # pointer net work
        self.po_dense = nn.Linear(config.hidden_size, self.classes_num * 2)
        self.subject_dense = nn.Linear(config.hidden_size, 2)
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

        self.apply(self.init_bert_weights)

    def forward(self, q_ids=None, passage_ids=None, segment_ids=None, token_type_ids=None, subject_ids=None,
                subject_labels=None,
                object_labels=None, eval_file=None,
                is_eval=False):
        mask = passage_ids != 0
        bert_encoder, _ = self.bert(passage_ids, segment_ids, attention_mask=mask,
                                    output_all_encoded_layers=False)
        if not is_eval:
            subject_encoder = self.token_entity_emb(token_type_ids)
            context_encoder = bert_encoder + subject_encoder

            # sub_start_encoder = batch_gather(passage_encoder, subject_ids[:, 0])
            # sub_end_encoder = batch_gather(passage_encoder, subject_ids[:, 1])
            # output = torch.cat([passage_encoder, sub_start_encoder, sub_end_encoder], 1)
            # context_encoder = self.LayerNorm(context_encoder)

            context_encoder = self.transformer_encoder(context_encoder.transpose(1, 0),
                                                       src_key_padding_mask=passage_ids.eq(0)).transpose(0, 1)

            sub_preds = self.subject_dense(bert_encoder)
            po_preds = self.po_dense(context_encoder).reshape(passage_ids.size(0), -1, self.classes_num, 2)

            subject_loss = self.loss_fct(sub_preds, subject_labels)
            subject_loss = subject_loss.mean(2)
            subject_loss = torch.sum(subject_loss * mask.float()) / torch.sum(mask.float())

            po_loss = self.loss_fct(po_preds, object_labels)
            po_loss = torch.sum(po_loss.mean(3), 2)
            po_loss = torch.sum(po_loss * mask.float()) / torch.sum(mask.float())

            loss = subject_loss + po_loss

            return loss

        else:

            subject_preds = nn.Sigmoid()(self.subject_dense(bert_encoder))
            answer_list = list()
            for qid, sub_pred in zip(q_ids.cpu().numpy(),
                                     subject_preds.cpu().numpy()):
                context = eval_file[qid].bert_tokens
                start = np.where(sub_pred[:, 0] > 0.6)[0]
                end = np.where(sub_pred[:, 1] > 0.5)[0]
                subjects = []
                for i in start:
                    j = end[end >= i]
                    if i == 0 or i > len(context) - 2:
                        continue

                    if len(j) > 0:
                        j = j[0]
                        if j > len(context) - 2:
                            continue
                        subjects.append((i, j))

                answer_list.append(subjects)

            qid_ids, bert_encoders, pass_ids, subject_ids, token_type_ids = [], [], [], [], []
            for i, subjects in enumerate(answer_list):
                if subjects:
                    qid = q_ids[i].unsqueeze(0).expand(len(subjects))
                    pass_tensor = passage_ids[i, :].unsqueeze(0).expand(len(subjects), passage_ids.size(1))
                    new_bert_encoder = bert_encoder[i, :, :].unsqueeze(0).expand(len(subjects), bert_encoder.size(1),
                                                                                 bert_encoder.size(2))

                    token_type_id = torch.zeros((len(subjects), passage_ids.size(1)), dtype=torch.long)
                    for index, (start, end) in enumerate(subjects):
                        token_type_id[index, start:end + 1] = 1

                    qid_ids.append(qid)
                    pass_ids.append(pass_tensor)
                    subject_ids.append(torch.tensor(subjects, dtype=torch.long))
                    bert_encoders.append(new_bert_encoder)
                    token_type_ids.append(token_type_id)

            if len(qid_ids) == 0:
                # print('len(qid_list)==0:')
                qid_tensor = torch.tensor([-1, -1], dtype=torch.long).to(bert_encoder.device)
                return qid_tensor, qid_tensor, qid_tensor
            qids = torch.cat(qid_ids).to(bert_encoder.device)
            pass_ids = torch.cat(pass_ids).to(bert_encoder.device)
            bert_encoders = torch.cat(bert_encoders).to(bert_encoder.device)
            token_type_ids = torch.cat(token_type_ids).to(bert_encoder.device)
            subject_ids = torch.cat(subject_ids).to(bert_encoder.device)

            flag = False
            split_heads = 1024

            bert_encoders = torch.split(bert_encoders, split_heads, dim=0)
            pass_ids = torch.split(pass_ids, split_heads, dim=0)
            token_type_ids = torch.split(token_type_ids, split_heads, dim=0)
            # subject_ids = torch.split(subject_ids, split_heads, dim=0)

            po_preds = list()
            for i in range(len(bert_encoders)):
                bert_encoders = bert_encoders[i]
                token_type_ids = token_type_ids[i]
                pass_ids = pass_ids[i]
                # subject_ids = subject_ids[i]

                if bert_encoders.size(0) == 1:
                    flag = True
                    # print('flag = True**********')
                    bert_encoders = bert_encoders.expand(2, bert_encoders.size(1), bert_encoders.size(2))
                    token_type_ids = token_type_ids.expand(2, token_type_ids.size(1))
                    pass_ids = pass_ids.expand(2, pass_ids.size(1))
                subject_encoder = self.token_entity_emb(token_type_ids)
                context_encoder = bert_encoders + subject_encoder

                context_encoder = self.transformer_encoder(context_encoder.transpose(1, 0),
                                                           src_key_padding_mask=pass_ids.eq(0)).transpose(0, 1)

                # context_encoder = self.LayerNorm(context_encoder)
                po_pred = self.po_dense(context_encoder).reshape(subject_encoder.size(0), -1, self.classes_num, 2)

                if flag:
                    po_pred = po_pred[1, :, :, :].unsqueeze(0)

                po_preds.append(po_pred)

            po_tensor = torch.cat(po_preds).to(qids.device)
            po_tensor = nn.Sigmoid()(po_tensor)
            return qids, subject_ids, po_tensor
