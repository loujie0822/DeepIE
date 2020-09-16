"""
实体抽取，按照字切分
"""
import codecs
import copy
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import extract_chinese_and_punct
from utils.data_util import sequence_padding
from utils.logger import logger

chineseandpunctuationextractor = extract_chinese_and_punct.ChineseAndPunctuationExtractor()


class Example(object):
    def __init__(self,
                 p_id=None,  # 当前文本序号（经过拆分）
                 text_id=None,  # 原始文本序号
                 g_raw_text=None,  # 全局文本（未拆分）
                 context=None,  # 当前文本（经过拆分）
                 tok_to_orig_start_index=None,
                 tok_to_orig_end_index=None,
                 bert_tokens=None,
                 l_gold_ent=None,  # 局部答案（经过拆分）
                 g_gold_ent=None,  # 全局答案（未拆分）
                 is_split=None,
                 span_index=None,
                 po_list=None,
                 query_id=None,    # 转换为MRC的时候所引入的query
                 query=None,
                 is_impossible=None,
                 ):
        self.p_id = p_id
        self.text_id = text_id
        self.context = context
        self.g_raw_text = g_raw_text
        self.tok_to_orig_start_index = tok_to_orig_start_index
        self.tok_to_orig_end_index = tok_to_orig_end_index
        self.bert_tokens = bert_tokens
        self.l_gold_ent = l_gold_ent
        self.g_gold_ent = g_gold_ent
        self.is_split = is_split
        self.span_index = span_index
        self.po_list = po_list
        self.query_id = query_id
        self.query = query
        self.is_impossible = is_impossible


class InputFeature(object):

    def __init__(self,
                 p_id=None,
                 passage_id=None,
                 token_type_id=None,
                 pos_start_id=None,
                 pos_end_id=None,
                 segment_id=None,
                 po_label=None,
                 pos_span_id=None,  # 针对MRC模型
                 query_id=None,
                 is_impossible=None,
                 ):
        self.p_id = p_id
        self.passage_id = passage_id
        self.token_type_id = token_type_id
        self.pos_start_id = pos_start_id
        self.pos_end_id = pos_end_id
        self.segment_id = segment_id
        self.po_label = po_label
        self.pos_span_id = pos_span_id
        self.query_id = query_id,
        self.is_impossible = is_impossible


class Reader(object):
    def __init__(self, spo_conf, query_conf=None, tokenizer=None, max_seq_length=None):
        self.spo_conf = spo_conf
        self.query_conf = query_conf
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length - 2

    def mrc_examples(self, examples):
        query_type = 'default'  # 可以通过配置选择使用
        # mrc模型中增加query信息
        new_examples = []
        for exam in examples:
            for ent_type, query in self.query_conf[query_type].items():
                new_exam = copy.deepcopy(exam)
                new_exam.query = query
                new_exam.query_id = self.spo_conf[ent_type]
                # 更新bert_token
                query_tokens = [c.lower() for c in query]
                tokens = new_exam.bert_tokens + ["[SEP]"] + query_tokens
                new_exam.bert_tokens = tokens
                new_exam.is_impossible = False
                if not new_exam.l_gold_ent:    # 无实体元素
                    new_exam.is_impossible = True
                new_examples.append(new_exam)
        return new_examples

    def read_examples(self, filename, data_type):
        logger.info("Generating {} examples...".format(data_type))
        examples = self._read(filename, data_type)

        # 加入query信息，更新examples
        examples = self.mrc_examples(examples)
        logger.info("MRC-NER: {} total size is  {} ".format(data_type, len(examples)))
        logger.info("=" * 15)
        return examples

    def split_text(self, text):
        MAX_LEN = self.max_seq_length
        text_lst = []
        split_num = len(text) // MAX_LEN

        for i in range(split_num + 1):
            text_lst.append(text[i * MAX_LEN:(i + 1) * MAX_LEN])
        return text_lst

    def _read(self, filename, data_type):
        examples = []

        before_text_num = 0
        after_ent_num = 0
        before_ent_num = 0
        with codecs.open(filename, 'r') as fr:
            text_id = 0
            p_id = 0
            seq_len = []
            for line in fr.readlines():
                before_text_num += 1

                data_lst = line.strip().split('|||')
                raw_text = data_lst[0]
                seq_len.append(len(raw_text))

                ent_lst = []
                for data in data_lst[1:]:
                    if data == '': continue
                    start, end, ent_type = data.split()
                    ent_name = raw_text[int(start):int(end) + 1]
                    ent_lst.append((int(start), int(end), ent_name, ent_type))
                ent_lst = list(set(ent_lst))
                before_ent_num += len(ent_lst)

                text_lst = self.split_text(raw_text)

                for i, text in enumerate(text_lst):

                    tokens = [c.lower() for c in text]

                    l_gold_ent = []
                    po_list = []
                    for (start, end, ent_name_, ent_type) in ent_lst:
                        if (i * self.max_seq_length) <= start < ((i + 1) * self.max_seq_length) and (
                                i * self.max_seq_length) <= end < ((i + 1) * self.max_seq_length):
                            ent_name = text[start - i * self.max_seq_length:end + 1 - i * self.max_seq_length]
                            if ent_name == '':
                                print('error')
                            assert ent_name == ent_name_

                            po_list.append((start - i * self.max_seq_length, end - i * self.max_seq_length,
                                            self.spo_conf[ent_type]))
                            l_gold_ent.append(
                                (start - i * self.max_seq_length, end - i * self.max_seq_length, ent_name, ent_type))
                    after_ent_num += len(l_gold_ent)

                    examples.append(
                        Example(
                            p_id=p_id,
                            text_id=text_id,
                            g_raw_text=raw_text,
                            context=text,
                            g_gold_ent=ent_lst,
                            l_gold_ent=l_gold_ent,
                            is_split=True if len(text_lst) > 1 else False,
                            span_index=i if len(text_lst) > 1 else -1,
                            bert_tokens=tokens,
                            po_list=po_list,
                        ))

                    p_id += 1
                text_id += 1

        logger.info('total size before split in {} is {}'.format(data_type, before_text_num))
        logger.info('total size after split in {} is {}'.format(data_type, len(examples)))
        logger.info('after_ent_num in {} is {}'.format(data_type, after_ent_num))
        logger.info('before_ent_num in {} is {}'.format(data_type, before_ent_num))
        logger.info("{} total size is  {} ".format(data_type, len(examples)))
        logger.info("=" * 15)
        return examples


class Feature(object):
    def __init__(self, max_len, spo_config, tokenizer):
        self.max_len = max_len
        self.spo_config = spo_config
        self.tokenizer = tokenizer

    def __call__(self, examples, data_type):
        return self.convert_examples_to_bert_features(examples, data_type)

    def convert_examples_to_bert_features(self, examples, data_type):
        logger.info("convert {}  examples to features .".format(data_type))

        examples2features = list()
        for index, example in enumerate(examples):
            examples2features.append((index, example))

        logger.info("Built instances is Completed")
        return MRCDataset(examples2features, spo_config=self.spo_config, data_type=data_type,
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

                token_ids = self.tokenizer.encode(example.bert_tokens)  # 自动增加了首尾的CLS和SEP的编码
                token_type_ids = np.zeros(len(token_ids), dtype=np.long)
                segment_ids = len(token_ids) * [0]

                batch_token_ids.append(token_ids)
                batch_token_type_ids.append(token_type_ids)
                batch_segment_ids.append(segment_ids)

                object_labels = np.zeros((len(token_ids), len(self.spo_config), 2), dtype=np.float32)
                for o in example.po_list:
                    object_labels[o[0] + 1, o[2], 0] = 1
                    object_labels[o[1] + 1, o[2], 1] = 1
                batch_object_labels.append(object_labels)

            batch_token_ids = sequence_padding(batch_token_ids, is_float=False)
            batch_token_type_ids = sequence_padding(batch_token_type_ids, is_float=False)
            batch_segment_ids = sequence_padding(batch_segment_ids, is_float=False)

            batch_object_labels = sequence_padding(batch_object_labels, padding=np.zeros((len(self.spo_config), 2)),
                                                   is_float=True)
            if not self.is_train:
                return p_ids, batch_token_ids, batch_token_type_ids, batch_segment_ids
            else:

                return batch_token_ids, batch_token_type_ids, batch_segment_ids, batch_object_labels

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, pin_memory=False, drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


class MRCDataset(SPODataset):
    def _create_collate_fn(self):
        def collate(examples):
            p_ids, examples = zip(*examples)
            p_ids = torch.tensor([p_id for p_id in p_ids], dtype=torch.long)
            batch_token_ids, batch_segment_ids = [], []
            batch_token_type_ids, batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], [], []
            batch_is_impossible,  batch_pos_start, batch_pos_end, batch_pos_span = [], [], [], []
            batch_query_ids = []

            for example in examples:
                token_ids = self.tokenizer.encode(example.bert_tokens)
                token_type_ids = np.zeros(len(token_ids), dtype=np.long)
                segment_ids = (len(example.context)+2) * [0] + (len(example.query)+1) * [1]

                batch_token_ids.append(token_ids)
                batch_token_type_ids.append(token_type_ids)
                batch_segment_ids.append(segment_ids)

                # add new info ...
                batch_is_impossible.append(example.is_impossible)
                batch_query_ids.append(example.query_id)
                pos_start, pos_end = len(token_ids) * [0], len(token_ids) * [0]
                pos_span = np.zeros((len(token_ids), len(token_ids)), dtype=np.float32)
                for o in example.po_list:  # start&end 包含了首尾了吗，应该不包含
                    pos_start[o[0]+1] = 1
                    pos_end[o[1]+1] = 1
                    pos_span[o[0]+1][o[1]+1] = 1
                batch_pos_start.append(pos_start)
                batch_pos_end.append(pos_end)
                batch_pos_span.append(pos_span)

            batch_token_ids = sequence_padding(batch_token_ids, is_float=False)
            batch_token_type_ids = sequence_padding(batch_token_type_ids, is_float=False)
            batch_segment_ids = sequence_padding(batch_segment_ids, is_float=False)

            batch_pos_start = sequence_padding(batch_pos_start, is_float=False)
            batch_pos_end = sequence_padding(batch_pos_end, is_float=False)
            batch_pos_span, batch_pos_span_mask = sequence_padding_xy(batch_pos_span, is_float=True,
                                                           doc=len(example.context), query=len(example.query))

            if not self.is_train:
                return p_ids, batch_token_ids, batch_token_type_ids, batch_segment_ids, torch.IntTensor(batch_query_ids)
            else:
                return batch_token_ids, batch_token_type_ids, batch_segment_ids, batch_pos_start, batch_pos_end, \
                       batch_pos_span, batch_pos_span_mask, \
                       torch.BoolTensor(batch_is_impossible), torch.IntTensor(batch_query_ids)

        return partial(collate)


def sequence_padding_xy(inputs, length=None, padding=0, is_float=False, doc=0, query=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    outputs, outputs_mask = [], []
    for x in inputs:
        if len(x) < length:
            diff = length - len(x)
            x = np.concatenate((x, np.zeros((diff, x.shape[1]))), axis=0)
            x = np.concatenate((x, np.zeros((x.shape[0], diff))), axis=1)
        else:
            x[:length][:length]
        outputs.append(x)
        x_mask = np.zeros(x.shape, dtype=np.int)
        x_mask[:doc+1, query+1:] = 1
        x_mask[doc+1:, :query+1] = 1
        outputs_mask.append(x_mask)

    outputs = np.array(outputs)
    outputs_mask = np.array(outputs_mask)

    out_tensor = torch.FloatTensor(outputs) if is_float \
        else torch.LongTensor(outputs)
    return torch.tensor(out_tensor), torch.tensor(outputs_mask)

