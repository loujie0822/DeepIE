import codecs
import json
import logging
from collections import Counter
from functools import partial

import jieba
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config.spo_config_v1 import BAIDU_RELATION
from layers.encoders.transformers.bert.bert_tokenization import BertTokenizer
from utils.data_util import padding, _handle_pos_limit, find_position, spo_padding


class PredictObject(object):
    def __init__(self,
                 object_name,
                 object_start,
                 object_end,
                 predict_type,
                 predict_type_id
                 ):
        self.object_name = object_name
        self.object_start = object_start
        self.object_end = object_end
        self.predict_type = predict_type
        self.predict_type_id = predict_type_id


class Example(object):
    def __init__(self,
                 p_id=None,
                 context=None,
                 bert_tokens=None,
                 sub_pos=None,
                 sub_entity_list=None,
                 relative_pos_start=None,
                 relative_pos_end=None,
                 po_list=None,
                 gold_answer=None):
        self.p_id = p_id
        self.context = context
        self.bert_tokens = bert_tokens
        self.sub_pos = sub_pos
        self.sub_entity_list = sub_entity_list
        self.relative_pos_start = relative_pos_start
        self.relative_pos_end = relative_pos_end
        self.po_list = po_list
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
    def __init__(self, do_lowercase=False, seg_char=False, max_len=600):

        self.do_lowercase = do_lowercase
        self.seg_char = seg_char
        self.max_len = max_len
        self.relation_config = BAIDU_RELATION

        if self.seg_char:
            logging.info("seg_char...")
        else:
            logging.info("seg_word using jieba ...")

    def read_examples(self, filename, data_type):
        logging.info("Generating {} examples...".format(data_type))
        return self._read(filename, data_type)

    def _data_process(self, filename, data_type='train'):
        output_data = list()
        with codecs.open(filename, 'r') as f:
            gold_num = 0
            for line in tqdm(f):
                data_json = json.loads(line.strip())
                text = data_json['text'].lower()
                sub_po_dict, sub_ent_list, spo_list = dict(), list(), list()

                for spo in data_json['spo_list']:
                    # TODO .strip('《》').strip()
                    subject_name = spo['subject'].lower()
                    object_name = spo['object'].lower()
                    s_start, s_end = find_position(subject_name, text)
                    o_start, o_end = find_position(object_name, text)

                    if text[s_start:s_end] != subject_name:
                        # print(subject_name)
                        subject_name = spo['subject'].lower().replace('》', '').replace('《', '')
                        s_start, s_end = find_position(subject_name, text)
                    if s_start != -1 and o_start != -1:
                        sub_ent_list.append((subject_name, s_start, s_end))
                        spo_list.append((subject_name, spo['predicate'], object_name))

                        if subject_name not in sub_po_dict:
                            sub_po_dict[subject_name] = {}
                            sub_po_dict[subject_name]['sub_pos'] = [s_start, s_end]
                            sub_po_dict[subject_name]['po_list'] = [
                                {'predict': spo['predicate'], 'object': (object_name, o_start, o_end)}]
                        else:
                            sub_po_dict[subject_name]['po_list'].append(
                                {'predict': spo['predicate'], 'object': (object_name, o_start, o_end)})
                text_spo = dict()
                text_spo['context'] = text
                text_spo['sub_po_dict'] = sub_po_dict
                text_spo['spo_list'] = list(set(spo_list))
                text_spo['sub_ent_list'] = list(set(sub_ent_list))
                gold_num += len(set(spo_list))
                output_data.append(text_spo)

        if data_type == 'train':
            return self._convert_train_data(output_data)
        # print(f'total gold num is {gold_num}')
        return output_data

    @staticmethod
    def _convert_train_data(src_data):
        """
        将train_data转化为满足训练要求的形式，即：
        1条数据为：一个subject对应响应的(predict,object)-->sub_po_dict
        :param data:
        :return:
        """
        spo_data = []
        for data in src_data:
            for sub_ent, po_dict in data['sub_po_dict'].items():
                data['sub_name'] = sub_ent
                data['sub_pos'] = po_dict['sub_pos']
                data['po_list'] = po_dict['po_list']

                spo_data.append(data)
        return spo_data

    def _read(self, filename, data_type):

        data_set = self._data_process(filename, data_type)
        logging.info("{} data_set total size is  {} ".format(data_type, len(data_set)))
        examples = []
        for p_id in tqdm(range(len(data_set))):
            data = data_set[p_id]
            para = data['context']
            context = para if self.seg_char else ''.join(jieba.lcut(para))
            if len(context) > self.max_len:
                context = context[:self.max_len]

            if data_type == 'train':

                start, end = data['sub_pos']
                if start >= self.max_len or end >= self.max_len:
                    continue
                assert data['sub_name'] == context[start:end]

                # pos_start&pos_end: 指句子中词语相对subject_entity的position(相对距离)
                # 如：[-30, 30]，embed 时整体+31，变成[1, 61]
                # 则一共62个pos token，0 留给 pad
                pos_start = list(map(lambda i: i - start, list(range(len(context)))))
                pos_end = list(map(lambda i: i - end, list(range(len(context)))))
                relative_pos_start = _handle_pos_limit(pos_start)
                relative_pos_end = _handle_pos_limit(pos_end)

                po_list = []
                for predict_object in data['po_list']:
                    predict_type = predict_object['predict']
                    object_ = predict_object['object']
                    object_name, object_start, object_end = object_[0], object_[1], object_[2]

                    if object_start >= self.max_len or object_end >= self.max_len:
                        continue
                    assert object_name == context[object_start:object_end]

                    po_list.append(PredictObject(
                        object_name=object_name,
                        object_start=object_start,
                        object_end=object_end,
                        predict_type=predict_type,
                        predict_type_id=self.relation_config[predict_type]
                    ))

                examples.append(
                    Example(
                        p_id=p_id,
                        context=context,
                        sub_pos=data['sub_pos'],
                        sub_entity_list=data['sub_ent_list'],
                        relative_pos_start=relative_pos_start,
                        relative_pos_end=relative_pos_end,
                        po_list=po_list,
                        gold_answer=data['spo_list']
                    )
                )
            else:
                examples.append(
                    Example(
                        p_id=p_id,
                        context=context,
                        sub_pos=None,
                        sub_entity_list=data['sub_ent_list'],
                        relative_pos_start=None,
                        relative_pos_end=None,
                        po_list=None,
                        gold_answer=data['spo_list']
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

    def __call__(self, examples, data_type):

        if self.bert:
            return self.convert_examples_to_bert_features(examples, data_type)
        else:
            return self.convert_examples_to_features(examples, data_type)

    def convert_examples_to_features(self, examples, data_type):

        logging.info("convert {}  examples to features .".format(data_type))

        examples2features = list()
        for index, example in enumerate(examples):

            passage_id = np.zeros(len(example.context), dtype=np.int)
            segment_id = np.zeros(len(example.context), dtype=np.int)
            token_type_id = np.zeros(len(example.context), dtype=np.int)
            pos_start_id = np.zeros(len(example.context), dtype=np.int)
            pos_end_id = np.zeros(len(example.context), dtype=np.int)
            s1 = np.zeros(len(example.context), dtype=np.float)
            s2 = np.zeros(len(example.context), dtype=np.float)

            for (_, start, end) in example.sub_entity_list:
                if start >= len(example.context) or end >= len(example.context):
                    continue
                s1[start] = 1.0
                s2[end - 1] = 1.0

            for i, token in enumerate(example.context):
                passage_id[i] = self.token2wid(token)

            if data_type == 'train':
                sub_start, sub_end = example.sub_pos[0], example.sub_pos[1]
                for i, token in enumerate(example.context):
                    if sub_start <= i < sub_end:
                        # token = "<MASK>"
                        token_type_id[i] = 1
                    pos_start_id[i] = example.relative_pos_start[i]
                    pos_end_id[i] = example.relative_pos_end[i]

            examples2features.append(
                InputFeature(
                    p_id=index,
                    passage_id=passage_id,
                    token_type_id=token_type_id,
                    pos_start_id=pos_start_id,
                    pos_end_id=pos_end_id,
                    segment_id=segment_id,
                    po_label=example.po_list,
                    s1=s1,
                    s2=s2

                ))

        logging.info("Built instances is Completed")
        return SPODataset(examples2features, predict_num=len(BAIDU_RELATION), data_type=data_type)

    def convert_examples_to_bert_features(self, examples, data_type):

        logging.info("Processing {} examples...".format(data_type))

        examples2features = list()
        for index, example in enumerate(examples):

            segment_id = np.zeros(len(example.context) + 2, dtype=np.int)
            token_type_id = np.zeros(len(example.context) + 2, dtype=np.int)
            pos_start_id = np.zeros(len(example.context) + 2, dtype=np.int)
            pos_end_id = np.zeros(len(example.context) + 2, dtype=np.int)
            s1 = np.zeros(len(example.context) + 2, dtype=np.float)
            s2 = np.zeros(len(example.context) + 2, dtype=np.float)

            for (_, start, end) in example.sub_entity_list:
                if start >= len(example.context) or end >= len(example.context):
                    continue
                s1[start + 1] = 1.0
                s2[end] = 1.0

            tokens = ["[CLS]"]
            for i, token in enumerate(example.context):
                tokens.append(token)
            tokens.append("[SEP]")
            passage_id = self.tokenizer.convert_tokens_to_ids(tokens)
            example.bert_tokens = tokens

            if data_type == 'train':
                sub_start, sub_end = example.sub_pos[0], example.sub_pos[1]
                for i, token in enumerate(example.context):
                    if sub_start <= i < sub_end:
                        token_type_id[i + 1] = 1
                    pos_start_id[i + 1] = example.relative_pos_start[i]
                    pos_end_id[i + 1] = example.relative_pos_end[i]

            examples2features.append(
                InputFeature(
                    p_id=index,
                    passage_id=passage_id,
                    token_type_id=token_type_id,
                    pos_start_id=pos_start_id,
                    pos_end_id=pos_end_id,
                    segment_id=segment_id,
                    po_label=example.po_list,
                    s1=s1,
                    s2=s2
                ))

        logging.info("Built instances is Completed")
        return SPODataset(examples2features, predict_num=len(BAIDU_RELATION), use_bert=True,data_type=data_type)


class SPODataset(Dataset):
    def __init__(self, features, predict_num, data_type, use_bert=False):
        super(SPODataset, self).__init__()
        self.use_bert = use_bert
        self.is_train = True if data_type == 'train' else False
        self.q_ids = [f.p_id for f in features]
        self.passages = [f.passage_id for f in features]
        self.segment_ids = [f.segment_id for f in features]
        self.predict_num = predict_num

        if self.is_train:
            self.token_type = [f.token_type_id for f in features]
            self.pos_start_ids = [f.pos_start_id for f in features]
            self.pos_end_ids = [f.pos_end_id for f in features]
            self.s1 = [f.s1 for f in features]
            self.s2 = [f.s2 for f in features]
            self.po_label = [f.po_label for f in features]

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, index):
        if self.is_train:
            return self.q_ids[index], self.passages[index], self.segment_ids[index], self.token_type[index], \
                   self.pos_start_ids[index], self.pos_end_ids[index], self.s1[index], self.s2[index], self.po_label[
                       index]
        else:
            return self.q_ids[index], self.passages[index], self.segment_ids[index]

    def _create_collate_fn(self, batch_first=False):
        def collate(examples):
            if self.is_train:
                p_ids, passages, segment_ids, token_type, pos_start_ids, pos_end_ids, s1, s2, label = zip(*examples)

                p_ids = torch.tensor([p_id for p_id in p_ids], dtype=torch.long)
                passages_tensor, _ = padding(passages, is_float=False, batch_first=batch_first)
                segment_tensor, _ = padding(segment_ids, is_float=False, batch_first=batch_first)

                token_type_tensor, _ = padding(token_type, is_float=False, batch_first=batch_first)
                pos_start_tensor, _ = padding(pos_start_ids, is_float=False, batch_first=batch_first)
                pos_end_tensor, _ = padding(pos_end_ids, is_float=False, batch_first=batch_first)
                s1_tensor, _ = padding(s1, is_float=True, batch_first=batch_first)
                s2_tensor, _ = padding(s2, is_float=True, batch_first=batch_first)
                po1_tensor, po2_tensor = spo_padding(passages, label, class_num=self.predict_num, is_float=True,
                                                     use_bert=self.use_bert)
                return p_ids, passages_tensor, segment_tensor, token_type_tensor, s1_tensor, s2_tensor, po1_tensor, \
                       po2_tensor
            else:
                p_ids, passages, segment_ids = zip(*examples)
                p_ids = torch.tensor([p_id for p_id in p_ids], dtype=torch.long)
                passages_tensor, _ = padding(passages, is_float=False, batch_first=batch_first)
                segment_tensor, _ = padding(segment_ids, is_float=False, batch_first=batch_first)
                return p_ids, passages_tensor, segment_tensor

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
