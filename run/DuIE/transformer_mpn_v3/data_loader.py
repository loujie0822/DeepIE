import json
import logging
import random
import re
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import extract_chinese_and_punct
from utils.data_util import search, sequence_padding

chineseandpunctuationextractor = extract_chinese_and_punct.ChineseAndPunctuationExtractor()


class Example(object):
    def __init__(self,
                 p_id=None,
                 context=None,
                 tok_to_orig_start_index=None,
                 tok_to_orig_end_index=None,
                 bert_tokens=None,
                 spoes=None,
                 sub_entity_list=None,
                 gold_answer=None, ):
        self.p_id = p_id
        self.context = context
        self.tok_to_orig_start_index = tok_to_orig_start_index
        self.tok_to_orig_end_index = tok_to_orig_end_index
        self.bert_tokens = bert_tokens
        self.spoes = spoes
        self.sub_entity_list = sub_entity_list
        self.gold_answer = gold_answer


class InputFeature(object):

    def __init__(self,
                 p_id=None,
                 passage_id=None,
                 token_type_id=None,
                 pos_start_id=None,
                 pos_end_id=None,
                 segment_id=None,
                 po_label=None,
                 s1=None,
                 s2=None):
        self.p_id = p_id
        self.passage_id = passage_id
        self.token_type_id = token_type_id
        self.pos_start_id = pos_start_id
        self.pos_end_id = pos_end_id
        self.segment_id = segment_id
        self.po_label = po_label
        self.s1 = s1
        self.s2 = s2


def covert_to_tokens(text, tokenizer, max_seq_length, return_orig_index=False):
    sub_text = []
    buff = ""
    flag_en = False
    flag_digit = False
    for char in text:
        if chineseandpunctuationextractor.is_chinese_or_punct(char):
            if buff != "":
                sub_text.append(buff)
                buff = ""
            sub_text.append(char)
            flag_en = False
            flag_digit = False
        else:
            if re.compile('\d').match(char):
                if buff != "" and flag_en:
                    sub_text.append(buff)
                    buff = ""
                    flag_en = False
                flag_digit = True
                buff += char
            else:
                if buff != "" and flag_digit:
                    sub_text.append(buff)
                    buff = ""
                    flag_digit = False
                flag_en = True
                buff += char
    if buff != "":
        sub_text.append(buff)

    tok_to_orig_start_index = []
    tok_to_orig_end_index = []
    tokens = []
    text_tmp = ''
    for (i, token) in enumerate(sub_text):
        sub_tokens = tokenizer.tokenize(token) if token != ' ' else []
        text_tmp += token
        for sub_token in sub_tokens:
            tok_to_orig_start_index.append(len(text_tmp) - len(token))
            tok_to_orig_end_index.append(len(text_tmp) - 1)
            tokens.append(sub_token)
            if len(tokens) >= max_seq_length - 2:
                break
        else:
            continue
        break
    if return_orig_index:
        return tokens, tok_to_orig_start_index, tok_to_orig_end_index
    else:
        return tokens


def search_spo_index(tokens, subject_sub_tokens, object_sub_tokens):
    subject_start_index, object_start_index = -1, -1
    forbidden_index = None
    if len(subject_sub_tokens) > len(object_sub_tokens):
        for index in range(
                len(tokens) - len(subject_sub_tokens) + 1):
            if tokens[index:index + len(
                    subject_sub_tokens)] == subject_sub_tokens:
                subject_start_index = index
                forbidden_index = index
                break

        for index in range(
                len(tokens) - len(object_sub_tokens) + 1):
            if tokens[index:index + len(
                    object_sub_tokens)] == object_sub_tokens:
                if forbidden_index is None:
                    object_start_index = index
                    break
                # check if labeled already
                elif index < forbidden_index or index >= forbidden_index + len(
                        subject_sub_tokens):
                    object_start_index = index

                    break

    else:
        for index in range(
                len(tokens) - len(object_sub_tokens) + 1):
            if tokens[index:index + len(
                    object_sub_tokens)] == object_sub_tokens:
                object_start_index = index
                forbidden_index = index
                break

        for index in range(
                len(tokens) - len(subject_sub_tokens) + 1):
            if tokens[index:index + len(
                    subject_sub_tokens)] == subject_sub_tokens:
                if forbidden_index is None:
                    subject_start_index = index
                    break
                elif index < forbidden_index or index >= forbidden_index + len(
                        object_sub_tokens):
                    subject_start_index = index
                    break

    return subject_start_index, object_start_index


class Reader(object):
    def __init__(self, spo_conf, tokenizer=None, max_seq_length=None):
        self.spo_conf = spo_conf
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def read_examples(self, filename, data_type):
        logging.info("Generating {} examples...".format(data_type))
        return self._read(filename, data_type)

    def _read(self, filename, data_type):
        complex_relation_label = [6, 8, 24, 30, 44]
        complex_relation_affi_label = [7, 9, 25, 26, 27, 31, 45]
        examples = []
        with open(filename, 'r') as fr:
            p_id = 0
            for line in tqdm(fr.readlines()):
                p_id += 1
                src_data = json.loads(line)
                text_raw = src_data['text']
                text_raw = text_raw.replace('®', '')
                text_raw = text_raw.replace('◆', '')

                tokens, tok_to_orig_start_index, tok_to_orig_end_index = covert_to_tokens(text_raw, self.tokenizer,
                                                                                          self.max_seq_length,
                                                                                          return_orig_index=True)
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                sub_po_dict, sub_ent_list, spo_list = dict(), list(), list()
                spoes = {}
                for spo in src_data['spo_list']:

                    spo_dict = dict()
                    for spo_object in spo['object'].keys():
                        if spo['predicate'] in self.spo_conf:
                            label = spo['predicate']
                        else:
                            label = spo['predicate'] + '_' + spo_object
                        spo_dict[self.spo_conf[label]] = spo['object'][spo_object]

                    for spo_object in spo['object'].keys():
                        # assign relation label
                        if spo['predicate'] in self.spo_conf:
                            # simple relation
                            predicate_label = self.spo_conf[spo['predicate']]

                            subject_sub_tokens = covert_to_tokens(spo['subject'], self.tokenizer, self.max_seq_length)
                            object_sub_tokens = covert_to_tokens(spo['object']['@value'], self.tokenizer,
                                                                 self.max_seq_length)
                            sub_ent_list.append(spo['subject'])
                        else:
                            # complex relation

                            predicate_label = self.spo_conf[spo['predicate'] + '_' + spo_object]

                            if predicate_label in complex_relation_affi_label:
                                subject_sub_tokens = covert_to_tokens(spo['object']['@value'], self.tokenizer,
                                                                      self.max_seq_length)
                                sub_ent_list.append(spo['object']['@value'])
                            else:
                                subject_sub_tokens = covert_to_tokens(spo['subject'], self.tokenizer,
                                                                      self.max_seq_length)
                                sub_ent_list.append(spo['subject'])
                            object_sub_tokens = covert_to_tokens(spo['object'][spo_object], self.tokenizer,
                                                                 self.max_seq_length)

                        subject_start, object_start = search_spo_index(tokens, subject_sub_tokens, object_sub_tokens)
                        if subject_start == -1:
                            subject_start = search(subject_sub_tokens, tokens)
                        if object_start == -1:
                            object_start = search(object_sub_tokens, tokens)

                        if subject_start != -1 and object_start != -1:
                            s = (subject_start, subject_start + len(subject_sub_tokens) - 1)
                            o = (object_start, object_start + len(object_sub_tokens) - 1, predicate_label)
                            if s not in spoes:
                                spoes[s] = []
                            spoes[s].append(o)
                examples.append(
                    Example(
                        p_id=p_id,
                        context=text_raw,
                        tok_to_orig_start_index=tok_to_orig_start_index,
                        tok_to_orig_end_index=tok_to_orig_end_index,
                        bert_tokens=tokens,
                        sub_entity_list=sub_ent_list,
                        gold_answer=src_data['spo_list'],
                        spoes=spoes
                    ))
                if data_type == 'train':
                    flag=False
                    for s,o in spoes.items():
                        for (o1,o2,p) in o:
                            if p in complex_relation_affi_label:
                                flag=True
                                continue
                        if not flag:
                            continue
                        tmp_spoes = {}
                        tmp_spoes[s] = spoes[s]

                        examples.append(
                            Example(
                                p_id=p_id,
                                context=text_raw,
                                tok_to_orig_start_index=tok_to_orig_start_index,
                                tok_to_orig_end_index=tok_to_orig_end_index,
                                bert_tokens=tokens,
                                sub_entity_list=sub_ent_list,
                                gold_answer=src_data['spo_list'],
                                spoes=tmp_spoes

                            ))
                # else:
                #     examples.append(
                #         Example(
                #             p_id=p_id,
                #             context=text_raw,
                #             tok_to_orig_start_index=tok_to_orig_start_index,
                #             tok_to_orig_end_index=tok_to_orig_end_index,
                #             bert_tokens=tokens,
                #             sub_entity_list=sub_ent_list,
                #             gold_answer=src_data['spo_list'],
                #             spoes=None
                #         ))

        # print('total gold num is {}'.format(gold_num))

        logging.info("{} total size is  {} ".format(data_type, len(examples)))

        return examples


class Feature(object):
    def __init__(self, max_len, spo_config, tokenizer):
        self.max_len = max_len
        self.spo_config = spo_config
        self.tokenizer = tokenizer

    def __call__(self, examples, data_type):
        return self.convert_examples_to_bert_features(examples, data_type)

    def convert_examples_to_bert_features(self, examples, data_type):
        logging.info("convert {}  examples to features .".format(data_type))

        examples2features = list()
        for index, example in enumerate(examples):
            examples2features.append((index, example))

        logging.info("Built instances is Completed")
        return SPODataset(examples2features, spo_config=self.spo_config, data_type=data_type,
                          tokenizer=self.tokenizer, max_len=self.max_len)


class SPODataset(Dataset):
    def __init__(self, data, spo_config, data_type, tokenizer=None, max_len=128):
        super(SPODataset, self).__init__()
        self.spo_config = spo_config
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.q_ids = [f[0] for f in data]
        self.features = [f[1] for f in data]
        self.is_train = True if data_type == 'train' else False

    def __len__(self):
        return len(self.q_ids)

    def __getitem__(self, index):
        return self.q_ids[index], self.features[index]

    def _create_collate_fn(self):
        def collate(examples):
            p_ids, examples = zip(*examples)
            p_ids = torch.tensor([p_id for p_id in p_ids], dtype=torch.long)
            batch_token_ids, batch_segment_ids = [], []
            batch_token_type_ids, batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], [], []
            for example in examples:
                spoes = example.spoes
                token_ids = self.tokenizer.encode(example.bert_tokens)[1:-1]
                segment_ids = len(token_ids) * [0]

                if self.is_train:
                    if spoes:
                        # subject标签
                        token_type_ids = np.zeros(len(token_ids), dtype=np.long)
                        subject_labels = np.zeros((len(token_ids), 2), dtype=np.float32)
                        for s in spoes:
                            subject_labels[s[0], 0] = 1
                            subject_labels[s[1], 1] = 1
                        # ⚠️不是随机选一个subject
                        subject_ids = random.choice(list(spoes.keys()))
                        # subject_ids = list(spoes.keys())[0]
                        # 对应的object标签
                        object_labels = np.zeros((len(token_ids), len(self.spo_config), 2), dtype=np.float32)
                        for o in spoes.get(subject_ids, []):
                            object_labels[o[0], o[2], 0] = 1
                            object_labels[o[1], o[2], 1] = 1
                        batch_token_ids.append(token_ids)
                        batch_token_type_ids.append(token_type_ids)

                        batch_segment_ids.append(segment_ids)
                        batch_subject_labels.append(subject_labels)
                        batch_subject_ids.append(subject_ids)
                        batch_object_labels.append(object_labels)
                else:
                    batch_token_ids.append(token_ids)
                    batch_segment_ids.append(segment_ids)

            batch_token_ids = sequence_padding(batch_token_ids, is_float=False)
            batch_segment_ids = sequence_padding(batch_segment_ids, is_float=False)
            if not self.is_train:
                return p_ids, batch_token_ids, batch_segment_ids
            else:
                batch_token_type_ids = sequence_padding(batch_token_type_ids, is_float=False)
                batch_subject_ids = torch.tensor(batch_subject_ids)
                batch_subject_labels = sequence_padding(batch_subject_labels, padding=np.zeros(2), is_float=True)
                batch_object_labels = sequence_padding(batch_object_labels, padding=np.zeros((len(self.spo_config), 2)),
                                                       is_float=True)
                return batch_token_ids, batch_segment_ids, batch_token_type_ids, batch_subject_ids, batch_subject_labels, batch_object_labels

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


def truncate_sequence(first_sequence,
                      max_length,
                      pop_index=-2):
    """截断总长度
    """
    while True:
        total_length = len(first_sequence)
        if total_length <= max_length:
            break
        else:
            first_sequence.pop(pop_index)

    return first_sequence
