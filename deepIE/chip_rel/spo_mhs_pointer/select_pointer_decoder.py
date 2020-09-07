# _*_ coding:utf-8 _*_
import numpy as np
import torch

from deepIE.config.config import CMeIE_CONFIG, Ent_BIO

reversed_relation_vocab = {v: k for k, v in CMeIE_CONFIG.items()}
reversed_bio_vocab = {v: k for k, v in Ent_BIO.items()}


def find_tag_position(find_list, seq_len, text):
    tag_list = list()
    j = 0
    while j < seq_len:
        end = j
        flag = True

        if find_list[j] == 1:
            start = j
            for k in range(start + 1, seq_len):
                if find_list[k] != find_list[start] + 1:
                    end = k - 1
                    flag = False
                    break
            if flag:
                end = seq_len - 1
            tag_list.append(text[start:end + 1])
        j = end + 1
    return tag_list


def find_entity(pos, text, sequence_tags):
    entity = []
    if pos >= len(text):
        return ''
    if sequence_tags[pos] == 'B' and (pos == len(text) - 1 or sequence_tags[pos + 1] == 'O'):
        entity.append(text[pos])
    elif (sequence_tags[pos] == 'I' and pos == len(text) - 1) or (
            sequence_tags[pos] == 'I' and sequence_tags[pos + 1] == 'O'):
        temp_entity = []
        while sequence_tags[pos] == 'I':
            temp_entity.append(text[pos])
            pos -= 1
            if pos < 0:
                break
            if sequence_tags[pos] == 'B':
                temp_entity.append(text[pos])
                break
        entity = list(reversed(temp_entity))
    return ''.join(entity)


def selection_decode(q_ids, eval_file, ent_pre, rel_pre):
    ent_list = list()
    ent_start_list = list()
    for qid, sub_pred in zip(q_ids.cpu().numpy(), ent_pre.cpu().numpy()):
        context = eval_file[qid].bert_tokens
        raw_text = eval_file[qid].context
        start = np.where(sub_pred[:, 0] > 0.5)[0]
        end = np.where(sub_pred[:, 1] > 0.5)[0]
        ents = []
        ent_start = {}
        for i in start:
            j = end[end >= i]
            if i == 0 or i > len(context) - 2:
                continue

            if len(j) > 0:
                j = j[0]
                if j > len(context) - 2:
                    continue
                ent_name = raw_text[i - 1:j]
                ent_start[i] = ent_name
                ents.append(ent_name)

        # for i in end:
        #     j=start[start<=i]
        #     if i == 0 or i > len(context) - 2:
        #         continue
        #     if len(j) > 0:
        #         j = j[-1]
        #         if j > len(context) - 2:
        #             continue
        #         ent_name = raw_text[j - 1:i]
        #         ent_start[i] = ent_name
        #         ents.append(ent_name)
        ent_list.append(ents)
        ent_start_list.append(ent_start)

    batch_num = len(rel_pre)
    result = [[] for _ in range(batch_num)]
    idx = torch.nonzero(rel_pre.cpu())
    answer_dict = dict()
    for i in range(idx.size(0)):
        b, s, p, o = idx[i].tolist()
        predicate = reversed_relation_vocab[p]

        object = ent_start_list[b].get(o, '')
        if object == '':
            continue
        subject = ent_start_list[b].get(s, '')
        if subject == '':
            continue
        result[b].append((subject, predicate, object))

    for q_id, res, ent_ in zip(q_ids.cpu(), result, ent_list):
        answer_dict[q_id.item()] = (ent_, res)
    return answer_dict
