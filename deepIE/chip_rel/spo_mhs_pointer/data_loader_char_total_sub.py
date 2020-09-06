"""
不使用BERT自带的tokenizer，只是按照char进行序列标注。
不再随机选择subject，而是将其全部flatten
"""

import json
import logging
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from deepIE.chip_rel.utils.data_utils import search_spo_index
from utils import extract_chinese_and_punct
from utils.data_util import sequence_padding

chineseandpunctuationextractor = extract_chinese_and_punct.ChineseAndPunctuationExtractor()


class Example(object):
    def __init__(self,
                 p_id=None,
                 raw_text=None,
                 context=None,
                 choice_sub=None,
                 tok_to_orig_start_index=None,
                 tok_to_orig_end_index=None,
                 bert_tokens=None,
                 ent_labels=None,
                 rel_labels=None,
                 gold_rel=None,
                 gold_ent=None):
        self.p_id = p_id
        self.context = context
        self.raw_text = raw_text
        self.choice_sub = choice_sub
        self.tok_to_orig_start_index = tok_to_orig_start_index
        self.tok_to_orig_end_index = tok_to_orig_end_index
        self.bert_tokens = bert_tokens

        self.ent_labels = ent_labels
        self.rel_labels = rel_labels
        self.gold_rel = gold_rel
        self.gold_ent = gold_ent


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
        gold_num = 0
        with open(filename, 'r') as fr:
            p_id = 0
            for line in tqdm(fr.readlines()):
                p_id += 1
                data_line = json.loads(line.strip())
                text_raw = data_line['text']

                tokens = [text.lower() for text in text_raw]
                tokens = tokens[:self.max_seq_length - 2]
                tokens = ["[CLS]"] + tokens + ["[SEP]"]

                if 'spo_list' not in data_line:
                    examples.append(
                        Example(
                            p_id=p_id,
                            context=text_raw,
                            bert_tokens=tokens,
                        ))
                    continue

                gold_ent_lst, gold_spo_lst = [], []
                spo_list = data_line['spo_list']
                rel_labels = []
                ent_labels = []

                for spo in spo_list:

                    subject = spo['subject']
                    gold_ent_lst.append(subject)

                    predicate = spo['predicate']

                    object = spo['object']['@value']
                    gold_ent_lst.append(object)

                    gold_spo_lst.append((subject, predicate, object))

                    subject_sub_tokens = [text.lower() for text in subject]
                    object_sub_tokens = [text.lower() for text in object]
                    subject_start, object_start = search_spo_index(tokens, subject_sub_tokens, object_sub_tokens)

                    predicate_label = self.spo_conf[predicate]

                    if subject_start != -1 and object_start != -1:
                        s = (subject_start, subject_start + len(subject_sub_tokens) - 1)
                        o = (object_start, object_start + len(object_sub_tokens) - 1, predicate_label)

                        ent_labels.append((subject_start, subject_start + len(subject_sub_tokens) - 1))
                        ent_labels.append((object_start, object_start + len(object_sub_tokens) - 1))

                        rel_labels.append((s[0], o[0], o[2]))

                    if subject_start == -1 or object_start == -1:
                        print('error')
                        print(subject_sub_tokens, object_sub_tokens, text_raw)

                examples.append(
                    Example(
                        p_id=p_id,
                        context=text_raw,
                        bert_tokens=tokens,
                        gold_ent=gold_ent_lst,
                        gold_rel=gold_spo_lst,
                        ent_labels=ent_labels,
                        rel_labels=rel_labels
                    ))
                gold_num += len(gold_spo_lst)

        logging.info('total gold spo num in {} is {}'.format(data_type, gold_num))

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
            batch_ent_labels, batch_rel_labels = [], []
            for example in examples:
                ent_labels = example.ent_labels
                rel_labels = example.rel_labels
                bert_tokens = example.bert_tokens
                token_ids = self.tokenizer.encode(bert_tokens)[1:-1]
                segment_ids = len(token_ids) * [0]

                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)

                ent_label_ids = np.zeros((len(token_ids), 2), dtype=np.float32)
                for s in ent_labels:
                    ent_label_ids[s[0], 0] = 1
                    ent_label_ids[s[1], 1] = 1

                batch_ent_labels.append(ent_label_ids)
                batch_rel_labels.append(rel_labels)

            batch_token_ids = sequence_padding(batch_token_ids, is_float=False)
            batch_segment_ids = sequence_padding(batch_segment_ids, is_float=False)
            if not self.is_train:
                batch_ent_labels = sequence_padding(batch_ent_labels, padding=np.zeros(2), is_float=True)
                batch_rel_labels = select_padding(batch_token_ids, batch_rel_labels, is_float=True,
                                                  class_num=len(self.spo_config), use_bert=True)
                return p_ids, batch_token_ids, batch_segment_ids, batch_ent_labels, batch_rel_labels
            else:
                batch_ent_labels = sequence_padding(batch_ent_labels, padding=np.zeros(2), is_float=True)
                batch_rel_labels = select_padding(batch_token_ids, batch_rel_labels, is_float=True,
                                                  class_num=len(self.spo_config), use_bert=True)
                return batch_token_ids, batch_segment_ids, batch_ent_labels, batch_rel_labels

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


def select_padding(seqs, select, is_float=False, class_num=None, use_bert=False):
    lengths = [len(s) for s in seqs]
    batch_length = max(lengths)
    seq_tensor = torch.FloatTensor(len(seqs), batch_length, class_num, batch_length).fill_(float(0)) if is_float \
        else torch.LongTensor(len(seqs), batch_length, class_num, batch_length).fill_(0)

    for i, triplet_list in enumerate(select):
        for triplet in triplet_list:
            subject_pos = triplet[0]
            object_pos = triplet[1]
            predicate = triplet[2]

            seq_tensor[i, subject_pos, predicate, object_pos] = 1

    return seq_tensor
