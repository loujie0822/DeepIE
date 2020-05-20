import json
import logging
from collections import Counter
from functools import partial

import jieba
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from layers.encoders.transformers.bert.bert_tokenization import BertTokenizer
from utils.data_util import padding, mpn_padding, _handle_pos_limit

config = {

    'drug': {
        '药品-用药频率': 0,
        '药品-持续时间': 1,
        '药品-用药剂量': 2,
        '药品-用药方法': 3,
        '药品-不良反应': 4,
    },
    'disease': {
        '疾病-检查方法': 0,
        '疾病-临床表现': 1,
        '疾病-非药治疗': 2,
        '疾病-药品名称': 3,
        '疾病-部位': 4,
    },
    'oncology_drug': {
        '药物_剂量': 0,
        '药物_给药方式': 1,
    },
    'yingxiang_bingzao': {
        "病灶部位_异常描述因子": 0,
        "病灶部位_阴性描述因子": 1,
        "病灶部位-诊断因子": 2
    }
}


class Attribute(object):
    def __init__(self,
                 value,
                 value_pos_start,
                 value_pos_end,
                 attr_type,
                 attr_type_id
                 ):
        self.value = value
        self.value_pos_start = value_pos_start
        self.value_pos_end = value_pos_end
        self.attr_type = attr_type
        self.attr_type_id = attr_type_id


class Example(object):
    def __init__(self,
                 p_id=None,
                 context=None,
                 bert_tokens=None,
                 entity_name=None,
                 entity_position=None,
                 pos_start=None,
                 pos_end=None,
                 gold_attr_list=None,
                 gold_answer=None):
        self.p_id = p_id
        self.context = context
        self.bert_tokens = bert_tokens
        self.entity_name = entity_name
        self.entity_position = entity_position
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.gold_attr_list = gold_attr_list
        self.gold_answer = gold_answer

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        gold_repr = []
        for attr in self.gold_attr_list:
            gold_repr.append(attr.attr_type + '-' + attr.value)
        return 'entity {} with attribute:\n{}'.format(self.entity_name, '\t'.join(gold_repr))


class InputFeature(object):

    def __init__(self,
                 p_id=None,
                 passage_id=None,
                 token_type_id=None,
                 pos_start_id=None,
                 pos_end_id=None,
                 segment_id=None,
                 label=None):
        self.p_id = p_id
        self.passage_id = passage_id
        self.token_type_id = token_type_id
        self.pos_start_id = pos_start_id
        self.pos_end_id = pos_end_id
        self.segment_id = segment_id
        self.label = label


class Reader(object):
    def __init__(self, do_lowercase=False, seg_char=False, max_len=600, entity_type="药品名称"):

        self.do_lowercase = do_lowercase
        self.seg_char = seg_char
        self.max_len = max_len
        self.entity_config = config[entity_type]

        if self.seg_char:
            logging.info("seg_char...")
        else:
            logging.info("seg_word using jieba ...")

    def read_examples(self, filename, data_type):
        logging.info("Generating {} examples...".format(data_type))
        return self._read(filename, data_type)

    def _read(self, filename, data_type):

        with open(filename, 'r') as fh:
            source_data = json.load(fh)

        examples = []
        for p_id in tqdm(range(len(source_data))):
            data = source_data[p_id]
            para = data['text']
            context = para if self.seg_char else ''.join(jieba.lcut(para))
            if len(context) > self.max_len:
                context = context[:self.max_len]
            context = context.lower() if self.do_lowercase else context

            _data_dict = dict()
            _data_dict['id'] = p_id
            entity_name, entity_position = data['entity'][0], data['entity'][1]
            entity_name = entity_name.lower() if self.do_lowercase else entity_name
            start, end = entity_position
            assert entity_name == context[start:end]

            # pos_start&pos_end: 指句子中词语相对entity的position限制
            # 如：[-30, 30]，embed 时整体+31，变成[1, 61]
            # 则一共62个pos token，0 留给 pad

            pos_start = list(map(lambda i: i - start, list(range(len(context)))))
            pos_end = list(map(lambda i: i - end, list(range(len(context)))))
            pos_start = _handle_pos_limit(pos_start)
            pos_end = _handle_pos_limit(pos_end)

            attribute_list = data['attribute_list']

            gold_attr_list = []
            for attribute in attribute_list:
                attr_type = attribute['type']
                value = attribute['value']
                value, value_pos_start, value_pos_end = value[0], value[1], value[2]
                value = value.lower() if self.do_lowercase else value

                assert value == context[value_pos_start:value_pos_end]

                gold_attr_list.append(Attribute(
                    value=value,
                    value_pos_start=value_pos_start,
                    value_pos_end=value_pos_end,
                    attr_type=attr_type,
                    attr_type_id=self.entity_config[attr_type]
                ))
            gold_answer = data['spo_list']

            examples.append(
                Example(
                    p_id=p_id,
                    context=context,
                    entity_name=entity_name,
                    entity_position=entity_position,
                    pos_start=pos_start,
                    pos_end=pos_end,
                    gold_attr_list=gold_attr_list,
                    gold_answer=gold_answer
                )
            )
        logging.info("{} total size is  {} ".format(data_type, len(examples)))
        return examples


class Vocabulary(object):

    def __init__(self, special_tokens=["<OOV>", "<MASK>"]):

        self.char_vocab = None
        self.emb_mat = None
        self.char2idx = None
        self.char_counter = Counter()
        self.special_tokens = special_tokens

    def build_vocab_only_with_char(self, examples, min_char_count=-1):

        logging.info("Building vocabulary only with character...")

        self.char_vocab = ["<PAD>"]

        if self.special_tokens is not None and isinstance(self.special_tokens, list):
            self.char_vocab.extend(self.special_tokens)

        for example in tqdm(examples):
            for char in example.context:
                self.char_counter[char] += 1

        for w, v in self.char_counter.most_common():
            if v >= min_char_count:
                self.char_vocab.append(w)

        self.char2idx = {token: idx for idx, token in enumerate(self.char_vocab)}

        logging.info("total char counter size is {} ".format(len(self.char_counter)))
        logging.info("total char vocabulary size is {} ".format(len(self.char_vocab)))

    def _load_embedding(self, embedding_file, embedding_dict):

        with open(embedding_file) as f:
            for line in f:
                if len(line.rstrip().split(" ")) <= 2: continue
                token, vector = line.rstrip().split(" ", 1)
                embedding_dict[token] = np.fromstring(vector, dtype=np.float, sep=" ")
        return embedding_dict

    def make_embedding(self, vocab, embedding_file, emb_size):

        embedding_dict = dict()
        embedding_dict["<PAD>"] = np.array([0. for _ in range(emb_size)])

        self._load_embedding(embedding_file, embedding_dict)

        count = 0
        for token in tqdm(vocab):
            if token not in embedding_dict:
                count += 1
                embedding_dict[token] = np.array([np.random.normal(scale=0.1) for _ in range(emb_size)])
        logging.info(
            "{} / {} tokens have corresponding in embedding vector".format(len(vocab) - count, len(vocab)))

        emb_mat = [embedding_dict[token] for idx, token in enumerate(vocab)]

        return emb_mat


class Feature(object):
    def __init__(self, args, token2idx_dict):
        self.bert = args.use_bert
        self.token2idx_dict = token2idx_dict
        if self.bert:
            self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    def token2wid(self, token):
        if token in self.token2idx_dict:
            return self.token2idx_dict[token]
        return self.token2idx_dict["<OOV>"]

    def __call__(self, examples, entity_type, data_type):

        if self.bert:
            return self.convert_examples_to_bert_features(examples, entity_type, data_type)
        else:
            return self.convert_examples_to_features(examples, entity_type, data_type)

    def convert_examples_to_features(self, examples, entity_type, data_type):

        logging.info("Processing {} examples...".format(data_type))

        examples2features = list()
        for index, example in enumerate(examples):

            gold_attr_list = example.gold_attr_list
            ent_start, ent_end = example.entity_position[0], example.entity_position[1]

            passage_id = np.zeros(len(example.context), dtype=np.int)
            token_type_id = np.zeros(len(example.context), dtype=np.int)
            pos_start_id = np.zeros(len(example.context), dtype=np.int)
            pos_end_id = np.zeros(len(example.context), dtype=np.int)

            for i, token in enumerate(example.context):
                if ent_start <= i < ent_end:
                    # token = "<MASK>"
                    token_type_id[i] = 1
                passage_id[i] = self.token2wid(token)
                pos_start_id[i] = example.pos_start[i]
                pos_end_id[i] = example.pos_end[i]

            examples2features.append(
                InputFeature(
                    p_id=index,
                    passage_id=passage_id,
                    token_type_id=token_type_id,
                    pos_start_id=pos_start_id,
                    pos_end_id=pos_end_id,
                    segment_id=token_type_id,
                    label=gold_attr_list
                ))

        logging.info("Built instances is Completed")
        return AttributeMPNDataset(examples2features, attribute_num=len(config[entity_type]))

    def convert_examples_to_bert_features(self, examples, entity_type, data_type):

        logging.info("Processing {} examples...".format(data_type))

        examples2features = list()
        for index, example in enumerate(examples):

            gold_attr_list = example.gold_attr_list
            ent_start, ent_end = example.entity_position[0], example.entity_position[1]
            segment_id = np.zeros(len(example.context) + 2, dtype=np.int)
            token_type_id = np.zeros(len(example.context) + 2, dtype=np.int)
            pos_start_id = np.zeros(len(example.context) + 2, dtype=np.int)
            pos_end_id = np.zeros(len(example.context) + 2, dtype=np.int)

            tokens = ["[CLS]"]
            raw_tokens = ["[CLS]"]
            for i, token in enumerate(example.context):
                raw_tokens.append(token)
                if ent_start <= i < ent_end:
                    # token_type_id[i + 1] = 1
                    # segment_id[i + 1] = 1
                    token = '[unused1]'
                tokens.append(token)
                pos_start_id[i + 1] = example.pos_start[i]
                pos_end_id[i + 1] = example.pos_end[i]

            tokens.append("[SEP]")
            raw_tokens.append("[SEP]")
            passage_id = self.tokenizer.convert_tokens_to_ids(tokens)
            example.bert_tokens = raw_tokens
            examples2features.append(
                InputFeature(
                    p_id=index,
                    passage_id=passage_id,
                    token_type_id=token_type_id,
                    pos_start_id=pos_start_id,
                    pos_end_id=pos_end_id,
                    segment_id=segment_id,
                    label=gold_attr_list
                ))

        logging.info("Built instances is Completed")
        return AttributeMPNDataset(examples2features, attribute_num=len(config[entity_type]), use_bert=True)


class AttributeMPNDataset(Dataset):
    def __init__(self, features, attribute_num, use_bert=False):
        super(AttributeMPNDataset, self).__init__()
        self.use_bert = use_bert
        self.q_ids = [f.p_id for f in features]
        self.passages = [f.passage_id for f in features]
        self.token_type = [f.token_type_id for f in features]
        self.label = [f.label for f in features]
        self.attribute_num = attribute_num
        self.segment_ids = [f.segment_id for f in features]
        self.pos_start_ids = [f.pos_start_id for f in features]
        self.pos_end_ids = [f.pos_end_id for f in features]

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, index):
        return self.q_ids[index], self.passages[index], self.token_type[index], self.segment_ids[index], self.label[
            index], self.pos_start_ids[index], self.pos_end_ids[index]

    def _create_collate_fn(self, batch_first=False):
        def collate(examples):
            p_ids, passages, token_type, segment_ids, label, pos_start_ids, pos_end_ids = zip(*examples)
            p_ids = torch.tensor([p_id for p_id in p_ids], dtype=torch.long)
            passages_tensor, _ = padding(passages, is_float=False, batch_first=batch_first)
            pos_start_tensor, _ = padding(pos_start_ids, is_float=False, batch_first=batch_first)
            pos_end_tensor, _ = padding(pos_end_ids, is_float=False, batch_first=batch_first)
            token_type_tensor, _ = padding(token_type, is_float=False, batch_first=batch_first)
            segment_tensor, _ = padding(segment_ids, is_float=False, batch_first=batch_first)
            o1_tensor, o2_tensor = mpn_padding(passages, label, class_num=self.attribute_num, is_float=True,
                                               use_bert=self.use_bert)
            return p_ids, passages_tensor, token_type_tensor, segment_tensor, pos_start_tensor, pos_end_tensor, o1_tensor, o2_tensor

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
