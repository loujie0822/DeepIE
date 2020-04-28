import logging
import os
import warnings
from collections import Counter
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from layers.encoders.transformers.bert.bert_tokenization import BertTokenizer
from utils.data_util import padding
from utils.file_util import _read_conll

config = {

    'drug': {
        'PAD': 0,
        'O': 1,
        'B-药物': 2,
        'I-药物': 3,
    },
    'CT': {
        'PAD': 0,
        'O': 1,
        'B-病灶部位': 2,
        'I-病灶部位': 3,
        'B-淋巴结部位': 4,
        'I-淋巴结部位': 5,
        'B-远处转移部位': 6,
        'I-远处转移部位': 7
    }
}


class Example(object):
    def __init__(self,
                 p_id=None,
                 char=None,
                 bichar=None,
                 gold_answer=None):
        self.p_id = p_id
        self.char = char
        self.bichar = bichar
        self.gold_answer = gold_answer


class InputFeature(object):

    def __init__(self,
                 p_id=None,
                 passage_id=None,
                 label_id=None):
        self.p_id = p_id
        self.passage_id = passage_id
        self.label_id = label_id


class Reader(object):
    def __init__(self, do_lowercase=False, bi_char=True):

        self.do_lowercase = do_lowercase
        self.bi_char = bi_char

    def read_examples(self, filename, data_type):
        logging.info("Generating {} examples...".format(data_type))
        return self._read(filename, data_type)

    def _read(self, filename, data_type):

        examples = []

        for idx, data in _read_conll(filename):
            chars = data[0]
            gold_answer = data[1]
            bichars = None
            if self.bi_char:
                bichars = [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])]
            examples.append(
                Example(
                    char=chars,
                    bichar=bichars,
                    gold_answer=gold_answer
                )
            )
        logging.info("{} total size is  {} ".format(data_type, len(examples)))
        return examples


class Vocabulary(object):

    def __init__(self, char_type='char', special_tokens=[], min_char_count=-1, lower=True):

        self.vocab = []
        self.emb_mat = None
        self.word2idx = None
        self.counter = Counter()
        self.special_tokens = special_tokens
        self.char_type = char_type
        self.min_char_count = min_char_count
        self.lower = True
        self.padding = "<pad>"
        self.unknown = "<unk>"

    def _add_word(self, char):
        if self.lower:
            self.counter[char.lower()] += 1
        else:
            self.counter[char] += 1

    def build_vocab(self, examples):

        logging.info("Building vocabulary only with character...")

        self.vocab = [self.padding, self.unknown]

        if self.special_tokens is not None and isinstance(self.special_tokens, list):
            self.vocab.extend(self.special_tokens)

        for example in tqdm(examples):
            if self.char_type == 'char':
                for char in example.char:
                    self._add_word(char)
            elif self.char_type == 'bichar':
                for char in example.bichar:
                    self._add_word(char)

        for w, v in self.counter.most_common():
            if v >= self.min_char_count:
                self.vocab.append(w)

        self.word2idx = {token: idx for idx, token in enumerate(self.vocab)}

        logging.info("total {} counter size is {} ".format(self.char_type, len(self.counter)))
        logging.info("total {} vocabulary size is {} ".format(self.char_type, len(self.vocab)))

    # def _load_embedding(self, embedding_file, embedding_dict):
    #
    #     with open(embedding_file) as f:
    #         for line in f:
    #             if len(line.rstrip().split(" ")) <= 2: continue
    #             token, vector = line.rstrip().split(" ", 1)
    #             embedding_dict[token] = np.fromstring(vector, dtype=np.float, sep=" ")
    #     return embedding_dict
    #
    # def make_embedding(self, vocab, embedding_file, emb_size):
    #
    #     embedding_dict = dict()
    #     embedding_dict["<PAD>"] = np.array([0. for _ in range(emb_size)])
    #
    #     self._load_embedding(embedding_file, embedding_dict)
    #
    #     count = 0
    #     for token in tqdm(vocab):
    #         if token not in embedding_dict:
    #             count += 1
    #             embedding_dict[token] = np.array([np.random.normal(scale=0.1) for _ in range(emb_size)])
    #     logging.info(
    #         "{} / {} tokens have corresponding in embedding vector".format(len(vocab) - count, len(vocab)))
    #
    #     emb_mat = [embedding_dict[token] for idx, token in enumerate(vocab)]
    #
    #     return emb_mat


class StaticEmbedding(object):
    def __init__(self, vocab: Vocabulary, model_path: str, **kwargs):
        """
        通过词表构建embedding

        如果是满足下列情况，embedding的大小与vocab词表大小一致
        1）char 或者 bichar
        2）引入分词结果，word embedding 可以finetune的时候

        如果是下列情况，embedding的大小<=vocab:
        1) 引入分词结果，word embedding freeze

        """
        self.only_norm_found_vector = kwargs.get('only_norm_found_vector', False)
        self.emb_vectors = self._load_with_vocab(model_path, vocab)

    def _randomly_init_embed(self, num_embedding, embedding_dim):
        """

        :param int num_embedding: embedding的entry的数量
        :param int embedding_dim: embedding的维度大小
        :param callable init_embed: 初始化方法
        :return: torch.FloatTensor
        """
        embed = torch.zeros(num_embedding, embedding_dim)

        nn.init.uniform_(embed, -np.sqrt(3 / embedding_dim), np.sqrt(3 / embedding_dim))

        return embed

    def _load_with_vocab(self, model_path, vocab, error='ignore', padding='<pad>', unknown='<unk>', dtype=np.float32):

        assert isinstance(vocab, Vocabulary), "Only Vocabulary is supported."
        if not os.path.exists(model_path):
            raise FileNotFoundError("`{}` does not exist.".format(model_path))
        with open(model_path, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            parts = line.split()
            start_idx = 0
            if len(parts) == 2:
                dim = int(parts[1])
                start_idx += 1
            else:
                dim = len(parts) - 1
                f.seek(0)
            matrix = {}
            if vocab.padding:
                matrix[vocab.word2idx[padding]] = torch.zeros(dim)
            if vocab.unknown:
                matrix[vocab.word2idx[unknown]] = torch.zeros(dim)
            found_count = 0
            found_unknown = False
            for idx, line in enumerate(f, start_idx):
                try:
                    parts = line.strip().split()
                    word = ''.join(parts[:-dim])
                    nums = parts[-dim:]
                    # 对齐unk与pad
                    if word == padding and vocab.padding is not None:
                        word = vocab.padding
                    elif word == unknown and vocab.unknown is not None:
                        word = vocab.unknown
                        found_unknown = True
                    if word in vocab.word2idx:
                        index = vocab.word2idx[word]
                        matrix[index] = torch.from_numpy(np.fromstring(' '.join(nums), sep=' ', dtype=dtype, count=dim))
                        if self.only_norm_found_vector:
                            matrix[index] = matrix[index] / np.linalg.norm(matrix[index])
                        found_count += 1
                except Exception as e:
                    if error == 'ignore':
                        warnings.warn("Error occurred at the {} line.".format(idx))
                    else:
                        logging.error("Error occurred at the {} line.".format(idx))
                        raise e
            logging.info(
                "Found {} out of {} words in the pre-training embedding.".format(found_count, len(vocab.vocab)))
            for word, index in vocab.word2idx.items():
                if index not in matrix:
                    if found_unknown:  # 如果有unkonwn，用unknown初始化
                        matrix[index] = matrix[vocab.word2idx[unknown]]
                    else:
                        matrix[index] = None
            vectors = self._randomly_init_embed(len(matrix), dim)
            for index_in_vocab, vec in matrix.items():
                if vec is not None:
                    vectors[index_in_vocab] = vec
            return vectors


class Feature(object):
    def __init__(self, args, token2idx_dict, entity_type):
        self.bert = args.use_bert
        self.token2idx_dict = token2idx_dict
        self.entity_config = config[entity_type]
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

            passage_id = np.zeros(len(example.context), dtype=np.int)
            label_id = np.zeros(len(example.context), dtype=np.int)
            for i, token in enumerate(example.context):
                passage_id[i] = self.token2wid(token)
            for i, label in enumerate(example.ent_label):
                label_id[i] = self.entity_config[label]

            examples2features.append(
                InputFeature(
                    p_id=index,
                    passage_id=passage_id,
                    label_id=label_id
                ))

        logging.info("Built instances is Completed")
        return CRFDataset(examples2features)

    def convert_examples_to_bert_features(self, examples, entity_type, data_type):
        pass


class CRFDataset(Dataset):
    def __init__(self, features):
        super(CRFDataset, self).__init__()
        self.q_ids = [f.p_id for f in features]
        self.passages = [f.passage_id for f in features]
        self.label_id = [f.label_id for f in features]

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, index):
        return self.q_ids[index], self.passages[index], self.label_id[index]

    def _create_collate_fn(self, batch_first=False):
        def collate(examples):
            p_ids, passages, label_id = zip(*examples)
            p_ids = torch.tensor([p_id for p_id in p_ids], dtype=torch.long)
            passages_tensor, _ = padding(passages, is_float=False, batch_first=batch_first)
            label_tensor, _ = padding(label_id, is_float=False, batch_first=batch_first)

            return p_ids, passages_tensor, label_tensor

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
