import codecs
import json
import logging
from collections import Counter
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from layers.encoders.transformers.bert.bert_tokenization import BertTokenizer
from utils.data_util import padding

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
                 ent_label=None,
                 gold_answer=None):
        self.p_id = p_id
        self.context = context
        self.ent_label = ent_label
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
    def __init__(self, do_lowercase=False, seg_char=False, max_len=600):

        self.do_lowercase = do_lowercase
        self.seg_char = seg_char
        self.max_len = max_len

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
            p_id = 0
            for line in tqdm(f):
                p_id += 1
                data_json = json.loads(line.strip())
                text = data_json['text']
                ent_label = ['O'] * len(text)
                gold_answer = []
                for (ent_name, ent_start, ent_end) in data_json['ent_list']:
                    gold_answer.append(ent_name)
                    ent_label[ent_start] = 'B-药物'
                    for index in range(ent_start + 1, ent_end):
                        ent_label[index] = 'I-药物'

                examples.append(
                    Example(
                        p_id=p_id,
                        context=text,
                        ent_label=ent_label,
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
