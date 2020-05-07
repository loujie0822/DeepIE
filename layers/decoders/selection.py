# _*_ coding:utf-8 _*_
import torch

from config.spo_config_v1 import BAIDU_ENTITY, BAIDU_RELATION

reversed_relation_vocab = {v: k for k, v in BAIDU_RELATION.items()}
reversed_bio_vocab = {v: k for k, v in BAIDU_ENTITY.items()}


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


# def find_entity(pos, text, sequence_tags):
#     entity = []
#     if pos >= len(text):
#         return ''
#
#     if sequence_tags[pos] in ('B', 'O'):
#
#         entity.append(text[pos])
#
#     else:
#         temp_entity = []
#         while sequence_tags[pos] == 'I':
#             temp_entity.append(text[pos])
#             pos -= 1
#             if pos < 0:
#                 break
#             if sequence_tags[pos] == 'B':
#                 temp_entity.append(text[pos])
#                 break
#         entity = list(reversed(temp_entity))
#     return ''.join(entity)

def selection_decode(q_ids, eval_file, text_list, sequence_tags, selection_tags):
    ent_list = []
    for qid, pred in zip(q_ids, sequence_tags.data.cpu().numpy()):
        raw_context = eval_file[qid.item()].raw_context
        ent_ = find_tag_position(pred, len(raw_context), raw_context)
        ent_list.append(ent_)

    text_list = list(map(list, text_list))

    batch_num = len(sequence_tags)
    result = [[] for _ in range(batch_num)]
    idx = torch.nonzero(selection_tags.cpu())

    sequence_tags = sequence_tags.tolist()

    answer_dict = dict()
    for i in range(idx.size(0)):
        # print(i,idx.size(0))
        b, s, p, o = idx[i].tolist()

        predicate = reversed_relation_vocab[p]
        # print(predicate)
        # if predicate == 'NA':
        #     continue
        tags = list(map(lambda x: reversed_bio_vocab[x], sequence_tags[b]))
        object = find_entity(o, text_list[b], tags)
        if object == '':
            continue
        subject = find_entity(s, text_list[b], tags)
        if subject == '':
            continue
        result[b].append((subject, predicate, object))

    for q_id, res, ent_ in zip(q_ids.cpu(), result, ent_list):
        answer_dict[q_id.item()] = (ent_, res)
    return answer_dict
