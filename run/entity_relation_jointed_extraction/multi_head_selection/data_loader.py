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

from config.spo_config_v1 import BAIDU_RELATION, BAIDU_ENTITY
from utils.data_util import Tokenizer, search, sequence_padding, select_padding


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
                 raw_context=None,
                 bert_tokens=None,
                 text_word=None,
                 entity_list=None,
                 gold_answer=None):
        self.p_id = p_id
        self.context = context
        self.raw_context = raw_context
        self.text_word = text_word
        self.bert_tokens = bert_tokens
        self.entity_list = entity_list
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
                text = data_json['text'].lower().replace(' ', '')
                text_word = jieba.lcut(text)
                sub_po_dict, ent_list, spo_list = dict(), list(), list()

                for spo in data_json['spo_list']:
                    subject_name = spo['subject'].lower().replace(' ', '')
                    object_name = spo['object'].lower().replace(' ', '')
                    ent_list.append(subject_name)
                    ent_list.append(object_name)
                    spo_list.append((subject_name, spo['predicate'], object_name))

                examples.append(
                    Example(
                        p_id=p_id,
                        context=text,
                        text_word=text_word,
                        entity_list=ent_list,
                        gold_answer=spo_list
                    )
                )
                gold_num += len(set(spo_list))
        logging.info('total gold num is {}'.format(gold_num))

        logging.info("{} total size is  {} ".format(data_type, len(examples)))

        return examples


class Vocabulary(object):

    def __init__(self, special_tokens=["<OOV>", "<MASK>"]):

        self.char_vocab = None
        self.emb_mat = None
        self.char2idx = dict()
        self.word2idx = dict()
        self.char_counter = Counter()
        self.word_counter = Counter()
        self.special_tokens = special_tokens

    def build_vocab_only_with_char(self, examples, min_char_count=-1, min_word_count=-1):

        logging.info("Building vocabulary only with character...")

        self.char_vocab = ["<PAD>"]
        self.word_vocab = ["<PAD>"]

        if self.special_tokens is not None and isinstance(self.special_tokens, list):
            self.char_vocab.extend(self.special_tokens)
            self.word_vocab.extend(self.special_tokens)

        for example in tqdm(examples):
            for char in example.context:
                self.char_counter[char] += 1
            for word in example.text_word:
                self.word_counter[word] += 1

        for c, v in self.char_counter.most_common():
            if v >= min_char_count:
                self.char_vocab.append(c)

        for w, v in self.word_counter.most_common():
            if v >= min_word_count:
                self.word_vocab.append(w)

        self.char2idx = {token: idx for idx, token in enumerate(self.char_vocab)}

        logging.info("total char counter size is {} ".format(len(self.char_counter)))
        logging.info("total char vocabulary size is {} ".format(len(self.char_vocab)))

        logging.info("total word vocabulary size without embedding is {} ".format(len(self.word_vocab)))

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
        logging.info("total embedding size is {} ".format(len(embedding_dict)))

        # emb_mat = [embedding_dict[token] for token in vocab if token in embedding_dict]
        #
        # index = 0
        # for token in embedding_dict.keys():
        #     if token in vocab:
        #         self.word2idx.update({token: index})
        #         index += 1

        count = 0
        emb_mat = []
        index = 0
        for token in tqdm(vocab):
            if token in embedding_dict.keys():
                self.word2idx.update({token: index})
                emb_mat.append(embedding_dict[token])
                index += 1
            else:
                count += 1
        logging.info(
            "{} / {} tokens have corresponding in embedding vector".format(len(vocab) - count, len(vocab)))
        logging.info("total word vocabulary size is {} ".format(len(self.word2idx)))
        return emb_mat


class Feature(object):
    def __init__(self, args, char2idx, word2idx):
        self.bert = args.use_bert
        self.char2idx = char2idx
        self.word2idx = word2idx
        self.max_len = args.max_len
        if self.bert:
            self.tokenizer = Tokenizer(args.bert_model + '/vocab.txt', do_lower_case=True)

    def __call__(self, examples, data_type):

        if self.bert:
            return self.convert_examples_to_bert_features(examples, data_type)
        else:
            return self.convert_examples_to_features(examples, data_type)

    def convert_examples_to_bert_features(self, examples, data_type):

        logging.info("convert {}  examples to features .".format(data_type))

        examples2features = list()
        for index, example in enumerate(examples):
            examples2features.append((index, example))

        logging.info("Built instances is Completed")
        return SPOBERTDataset(examples2features, use_bert=True, data_type=data_type,
                              tokenizer=self.tokenizer, max_len=self.max_len)

    def convert_examples_to_features(self, examples, data_type):

        logging.info("convert {}  examples to features .".format(data_type))

        examples2features = list()
        for index, example in enumerate(examples):
            examples2features.append((index, example))

        logging.info("Built instances is Completed")
        return SPODataset(examples2features, use_bert=True, data_type=data_type,
                          word2idx=self.word2idx, char2idx=self.char2idx, max_len=self.max_len)


class SPODataset(Dataset):
    def __init__(self, data, data_type, use_bert=False, word2idx=None, char2idx=None, max_len=128):
        super(SPODataset, self).__init__()
        self.use_bert = use_bert
        self.word2idx = word2idx
        self.char2idx = char2idx
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
            batch_char_ids, batch_word_ids = [], []
            batch_ent_labels, batch_rel_labels = [], []
            for example in examples:
                # todo maxlen
                char_ids = [self.char2idx.get(char, 1) for char in example.context]
                word_ids = [self.word2idx.get(word, 0) for word in example.text_word for _ in word]
                if len(char_ids) != len(word_ids):
                    print(example.context)
                    print(char_ids)
                    print(len(char_ids))
                    print(example.text_word)
                    print(word_ids)
                    print(len(word_ids))
                assert len(char_ids) == len(word_ids)
                char_ids = char_ids[:self.max_len]
                word_ids = word_ids[:self.max_len]
                example.raw_context = example.context[:self.max_len]

                if self.is_train:
                    rel_labels = []
                    bio = ['O'] * len(char_ids)
                    for s, p, o in example.gold_answer:
                        s = [self.char2idx.get(s_, 1) for s_ in s]
                        p = BAIDU_RELATION[p]
                        o = [self.char2idx.get(o_, 1) for o_ in o]
                        s_idx = search(s, char_ids)
                        o_idx = search(o, char_ids)
                        if s_idx != -1 and o_idx != -1:
                            bio[s_idx] = 'B'
                            bio[s_idx + 1: s_idx + len(s)] = 'I'*(len(s)-1)
                            bio[o_idx] = 'B'
                            bio[o_idx + 1: o_idx + len(o)] = 'I'*(len(o)-1)
                            s = (s_idx, s_idx + len(s) - 1)
                            o = (o_idx, o_idx + len(o) - 1, p)
                            rel_labels.append((s[1], o[1], o[2]))

                    if rel_labels:
                        ent_labels = np.zeros((len(char_ids)), dtype=np.long)
                        for index, label_ in enumerate(bio):
                            ent_labels[index] = BAIDU_ENTITY[label_]
                        batch_char_ids.append(char_ids)
                        batch_word_ids.append(word_ids)
                        batch_ent_labels.append(ent_labels)
                        batch_rel_labels.append(rel_labels)
                else:
                    batch_char_ids.append(char_ids)
                    batch_word_ids.append(word_ids)

            batch_char_ids = sequence_padding(batch_char_ids, is_float=False)
            batch_word_ids = sequence_padding(batch_word_ids, is_float=False)
            if not self.is_train:
                return p_ids, batch_char_ids, batch_word_ids
            else:
                batch_ent_labels = sequence_padding(batch_ent_labels, is_float=False)
                batch_rel_labels = select_padding(batch_char_ids, batch_rel_labels, is_float=True,class_num=len(BAIDU_RELATION))
                return batch_char_ids, batch_word_ids, batch_ent_labels, batch_rel_labels

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


class SPOBERTDataset(Dataset):
    def __init__(self, data, data_type, use_bert=False, tokenizer=None, max_len=128):
        super(SPOBERTDataset, self).__init__()
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
