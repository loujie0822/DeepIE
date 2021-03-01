"""
不再随机选择subject，而是将其全部flatten
"""
import json
import logging
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import extract_chinese_and_punct
from utils.data_util import sequence_padding

chineseandpunctuationextractor = extract_chinese_and_punct.ChineseAndPunctuationExtractor()


class Example(object):
    def __init__(self,
                 p_id=None,
                 text_id=None,
                 raw_text=None,
                 context=None,
                 choice_sub=None,
                 tok_to_orig_start_index=None,
                 tok_to_orig_end_index=None,
                 bert_tokens=None,
                 spoes=None,
                 po_lst=None,
                 sub_entity_list=None,
                 gold_answer=None, ):
        self.p_id = p_id
        self.text_id = text_id
        self.po_lst = po_lst
        self.context = context
        self.raw_text = raw_text
        self.choice_sub = choice_sub
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


class Reader(object):
    def __init__(self, spo_conf, tokenizer=None, max_seq_length=None):
        self.spo_conf = spo_conf
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def read_examples(self, filename, data_type):
        logging.info("Generating {} examples...".format(data_type))
        return self._read(filename, data_type)

    def _read(self, filename, data_type):

        examples = []
        with open(filename, 'r') as fr:
            src_data_lsts = json.load(fr)
            p_id = 0
            for data in src_data_lsts:
                p_id += 1
                text_raw = data['text']
                text_id = data['text_id']

                tokens = [text.lower() for text in text_raw]
                tokens = tokens[:self.max_seq_length - 2]
                tokens = ["[CLS]"] + tokens + ["[SEP]"]

                sub_ent = data['entity']
                sub_name, sub_start, sub_end = sub_ent

                s = (sub_start, sub_end)
                assert tokens[sub_start + 1:sub_end + 2] == list(sub_name),'{},{}'.format(sub_start,sub_end)

                if 'predicate' not in data:
                    examples.append(
                        Example(
                            p_id=p_id,
                            context=text_raw,
                            choice_sub=s,
                            bert_tokens=tokens,
                            gold_answer=None,
                        ))
                    continue
                gold_answer = data['gold_answer']
                predicate_lst = data['predicate']
                po_lst = []
                for po in predicate_lst:
                    predicate, object_name, object_start, object_end = po[0], po[1], po[2], po[3]
                    predicate_label = self.spo_conf[predicate]
                    assert tokens[object_start + 1:object_end + 2] == list(object_name),'{},{}'.format(object_start,object_end)
                    po_lst.append((object_start, object_end, predicate_label))

                examples.append(
                    Example(
                        p_id=p_id,
                        text_id=text_id,
                        context=text_raw,
                        choice_sub=s,
                        bert_tokens=tokens,
                        po_lst=po_lst,
                        gold_answer=gold_answer,
                    ))

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

                token_ids = self.tokenizer.encode(example.bert_tokens)[1:-1]
                token_type_ids = np.zeros(len(token_ids), dtype=np.long)
                segment_ids = len(token_ids) * [0]

                sub_start, sub_end = example.choice_sub

                for i, token in enumerate(example.bert_tokens):
                    if sub_start + 1 <= i <= sub_end + 1:
                        # token_type_id[i ] = 1
                        segment_ids[i] = 1

                batch_token_ids.append(token_ids)
                batch_token_type_ids.append(token_type_ids)
                batch_segment_ids.append(segment_ids)

                object_labels = np.zeros((len(token_ids), len(self.spo_config), 2), dtype=np.float32)
                for o in example.po_lst:
                    object_labels[o[0] + 1, o[2], 0] = 1
                    object_labels[o[1] + 1, o[2], 1] = 1
                batch_object_labels.append(object_labels)

            batch_token_ids = sequence_padding(batch_token_ids, is_float=False)
            batch_token_type_ids = sequence_padding(batch_token_type_ids, is_float=False)
            batch_segment_ids = sequence_padding(batch_segment_ids, is_float=False)

            batch_object_labels = sequence_padding(batch_object_labels, padding=np.zeros((len(self.spo_config), 2)),
                                                   is_float=True)
            if not self.is_train:
                return p_ids, batch_token_ids, batch_token_type_ids,batch_segment_ids
            else:

                return batch_token_ids, batch_token_type_ids, batch_segment_ids,batch_object_labels

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
