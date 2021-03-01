import codecs
import json
import logging
from collections import Counter
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config.spo_config_v1 import BAIDU_RELATION
from utils.data_util import Tokenizer, search, sequence_padding


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
                 gold_answer=None,
                 token_ids=None):
        self.p_id = p_id
        self.context = context
        self.bert_tokens = bert_tokens
        self.sub_pos = sub_pos
        self.sub_entity_list = sub_entity_list
        self.relative_pos_start = relative_pos_start
        self.relative_pos_end = relative_pos_end
        self.po_list = po_list
        self.gold_answer = gold_answer
        self.token_ids = token_ids


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
    def __init__(self, do_lowercase=False, seg_char=False):

        self.do_lowercase = do_lowercase
        self.seg_char = seg_char
        self.relation_config = BAIDU_RELATION

        if self.seg_char:
            logging.info("seg_char...")
        else:
            logging.info("seg_word using jieba ...")

    def read_examples(self, filename, data_type):
        logging.info("Generating {} examples...".format(data_type))
        return self._read(filename, data_type)

    def _read(self, filename, data_type):

        examples = []
        with codecs.open(filename, 'r') as f:
            gold_num = 0
            p_id = 0
            for line in tqdm(f):
                p_id += 1
                data_json = json.loads(line.strip())
                text = data_json['text'].lower()
                sub_po_dict, sub_ent_list, spo_list = dict(), list(), list()

                for spo in data_json['spo_list']:
                    subject_name = spo['subject'].lower()
                    object_name = spo['object'].lower()
                    sub_ent_list.append(subject_name)
                    spo_list.append((subject_name, spo['predicate'], object_name))

                examples.append(
                    Example(
                        p_id=p_id,
                        context=text,
                        sub_entity_list=list(set(sub_ent_list)),
                        gold_answer=spo_list
                    )
                )
                gold_num += len(set(spo_list))
        print(f'total gold num is {gold_num}')

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
        self.max_len = args.max_len
        if self.bert:
            # self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
            self.tokenizer = Tokenizer(args.bert_model + '/vocab.txt', do_lower_case=True)

    def token2wid(self, token):
        if token in self.token2idx_dict:
            return self.token2idx_dict[token]
        return self.token2idx_dict["<OOV>"]

    def __call__(self, examples, data_type):

        if self.bert:
            return self.convert_examples_to_bert_features(examples, data_type)
        # else:
        #     return self.convert_examples_to_features(examples, data_type)

    def convert_examples_to_bert_features(self, examples, data_type):

        logging.info("convert {}  examples to features .".format(data_type))

        examples2features = list()
        for index, example in enumerate(examples):
            examples2features.append((index, example))

        logging.info("Built instances is Completed")
        return SPODataset(examples2features, predict_num=len(BAIDU_RELATION), use_bert=True, data_type=data_type,
                          tokenizer=self.tokenizer, max_len=self.max_len)


class SPODataset(Dataset):
    def __init__(self, data, predict_num, data_type, use_bert=False, tokenizer=None, max_len=128):
        super(SPODataset, self).__init__()
        self.use_bert = use_bert
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.q_ids = [f[0] for f in data]
        self.features = [f[1] for f in data]
        self.is_train = True if data_type == 'train' else False

    def __len__(self):
        return len(self.q_ids)

    def __getitem__(self, index):
        return self.q_ids[index], self.features[index]

    def _create_collate_fn(self, batch_first=False):
        def collate(examples):
            p_ids, examples = zip(*examples)
            p_ids = torch.tensor([p_id for p_id in p_ids], dtype=torch.long)
            batch_token_ids, batch_segment_ids = [], []
            batch_token_type_ids, batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], [], []
            for example in examples:
                # todo maxlen
                token_ids, segment_ids = self.tokenizer.encode(example.context, max_length=self.max_len)
                example.bert_tokens = self.tokenizer.tokenize(example.context)
                example.token_ids = token_ids
                # pp = self.tokenizer.tokenize(example.context)
                # ppp = self.tokenizer.decode(token_ids)
                if self.is_train:
                    spoes = {}
                    for s, p, o in example.gold_answer:
                        s = self.tokenizer.encode(s)[0][1:-1]
                        p = BAIDU_RELATION[p]
                        o = self.tokenizer.encode(o)[0][1:-1]
                        s_idx = search(s, token_ids)
                        o_idx = search(o, token_ids)
                        if s_idx != -1 and o_idx != -1:
                            s = (s_idx, s_idx + len(s) - 1)
                            o = (o_idx, o_idx + len(o) - 1, p)
                            if s not in spoes:
                                spoes[s] = []
                            spoes[s].append(o)

                    if spoes:
                        # subject标签
                        token_type_ids = np.zeros(len(token_ids), dtype=np.long)
                        subject_labels = np.zeros((len(token_ids), 2), dtype=np.float32)
                        for s in spoes:
                            subject_labels[s[0], 0] = 1
                            subject_labels[s[1], 1] = 1
                        # 随机选一个subject
                        start, end = np.array(list(spoes.keys())).T
                        start = np.random.choice(start)
                        end = np.random.choice(end[end >= start])
                        token_type_ids[start:end + 1] = 1
                        subject_ids = (start, end)
                        # 对应的object标签
                        object_labels = np.zeros((len(token_ids), len(BAIDU_RELATION), 2), dtype=np.float32)
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
                batch_object_labels = sequence_padding(batch_object_labels, padding=np.zeros((len(BAIDU_RELATION), 2)),
                                                       is_float=True)
                return batch_token_ids, batch_segment_ids, batch_token_type_ids, batch_subject_ids, batch_subject_labels, batch_object_labels

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
