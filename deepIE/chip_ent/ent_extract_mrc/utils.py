#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from transformers import BertTokenizer

import json
from tqdm import tqdm
import copy
from collections import OrderedDict


class Config(object):

    @classmethod
    def from_dict(cls, json_object):
        config_instance = Config()
        for key, value in json_object.items():
            try:
                tmp_value = Config.from_json_dict(value)
                config_instance.__dict__[key] = tmp_value
            except:
                config_instance.__dict__[key] = value
        return config_instance

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r") as f:
            text = f.read()
        return Config.from_dict(json.loads(text))

    @classmethod
    def from_json_dict(cls, json_str):
        return Config.from_dict(json_str)

    @classmethod
    def from_json_str(cls, json_str):
        return Config.from_dict(json.loads(json_str))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output = {k: v.to_dict() if isinstance(v, Config) else v for k, v in output.items()}
        return output

    def print_config(self, logger=None):
        model_config = OrderedDict(sorted(self.to_dict().items()))
        logger.info("$$" * 40)
        logger.info("Arg and Model Configs ...")
        for key, value in model_config.items():
            logger.info(f"{key} -> {value}")
        logger.info("$$" * 40)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def update_args(self, args_namespace):
        args_dict = args_namespace.__dict__
        print("Please notice that merge the args_dict and json_config ... ...")
        for args_key, args_value in args_dict.items():
            if args_key not in self.__dict__.keys():
                self.__dict__[args_key] = args_value
            else:
                print("update the config from args input ... ...")
                self.__dict__[args_key] = args_values


def whitespace_tokenize(text):
    """
    Desc:
        runs basic whitespace cleaning and splitting on a piece of text
    """
    text = text.strip()
    text = text.strip('\ufeff')
    if not text:
        return []
    #tokens = text.split()  # by lh
    tokens = list(text)
    return tokens


class BertTokenizer4Tagger(BertTokenizer):
    """
    Desc:
        slove the problem of tagging span can not fit after run word_piece tokenizing
    """
    def __init__(self, vocab_file, do_lower_case=False, max_len=None,
        never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):

        super(BertTokenizer4Tagger, self).__init__(vocab_file, do_lower_case=do_lower_case,
            max_len=max_len, never_split=never_split)

    def tokenize(self, text, label_lst=None):
        """
        Desc:
            text:
            label_lst: ["B", "M", "E", "S", "O"]
        """
        split_tokens = []
        split_labels = []

        if label_lst is None:
            for token in self.basic_tokenizer.tokenize(text, never_split=tuple()):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
            return split_tokens

        for token, label in zip(self.basic_tokenizer.tokenize(text, never_split=tuple()), label_lst):
            # cureent token should be 1 single word
            sub_tokens = self.wordpiece_tokenizer.tokenize(token)
            if len(sub_tokens) > 1:
                for tmp_idx, tmp_sub_token in enumerate(sub_tokens):
                    if tmp_idx == 0:
                        split_tokens.append(tmp_sub_token)
                        split_labels.append(label)
                    else:
                        split_tokens.append(tmp_sub_token)
                        split_labels.append("X")
            else:
                split_tokens.append(sub_token)
                split_labels.append(label)

        return split_tokens, split_labels


class InputExample(object):
    def __init__(self,
        qas_id,
        query_item,
        context_item,
        doc_tokens = None,
        orig_answer_text=None,
        start_position=None,
        end_position=None,
        span_position=None,
        is_impossible=None,
        ner_cate=None):

        """
        Desc:
            is_impossible: bool, [True, False]
        """

        self.qas_id = qas_id
        self.query_item = query_item
        self.context_item = context_item
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.span_position = span_position
        self.is_impossible = is_impossible
        self.ner_cate = ner_cate



class InputFeatures(object):
    """
    Desc:
        a single set of features of data
    Args:
        start_pos: start position is a list of symbol
        end_pos: end position is a list of symbol
    """
    def __init__(self,
        unique_id,
        tokens,
        input_ids,
        input_mask,
        segment_ids,
        ner_cate,
        start_position=None,
        end_position=None,
        span_position=None,
        span_label_mask=None,
        is_impossible=None):

        self.unique_id = unique_id
        self.tokens = tokens
        self.input_mask = input_mask
        self.input_ids = input_ids
        self.ner_cate = ner_cate
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.span_position = span_position
        self.is_impossible = is_impossible
        self.span_label_mask = span_label_mask


def examples_to_features(examples, tokenizer, label_lst, max_seq_length, is_training=True,
                                allow_impossible=True, pad_sign=True, entity_scheme="bes"):
    # 转换为模型所需特征
    label_map = {tmp: idx for idx, tmp in enumerate(label_lst)}
    features = []

    for (example_idx, example) in tqdm(enumerate(examples), total=len(examples), desc='process :'):

        query_tokens = tokenizer.tokenize(example.query_item)
        whitespace_doc = whitespace_tokenize(example.context_item)
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        if len(example.start_position) == 0 and len(example.end_position) == 0:     # 负例 不存在ner
            doc_start_pos = []
            doc_end_pos = []
            all_doc_tokens = []

            for token_item in whitespace_doc:
                tmp_subword_lst = tokenizer.tokenize(token_item)
                all_doc_tokens.extend(tmp_subword_lst)
            doc_start_pos = [0] * len(all_doc_tokens)
            doc_end_pos = [0] * len(all_doc_tokens)
            doc_span_pos = np.zeros((max_seq_length, max_seq_length), dtype=int)

        else:
            doc_start_pos = []
            doc_end_pos = []
            doc_span_pos = np.zeros((max_seq_length, max_seq_length), dtype=int)

            all_doc_tokens = []
            offset_idx_dict = {}

            fake_start_pos = [0] * len(whitespace_doc)
            fake_end_pos = [0] * len(whitespace_doc)

            # 处理中文会出问题的, 所以处理为各个字符char...  by lh
            for start_item in example.start_position:
                fake_start_pos[start_item] = 1
            for end_item in example.end_position:
                fake_end_pos[end_item] = 1

            # improve answer span
            for idx, (token, start_label, end_label) in enumerate(zip(whitespace_doc, fake_start_pos, fake_end_pos)):
                tmp_subword_lst = tokenizer.tokenize(token)

                if len(tmp_subword_lst) > 1:
                    offset_idx_dict[idx] = len(all_doc_tokens)   # offset_idx_dict是为了token后对应到span矩阵上，并没有用于还原回去

                    doc_start_pos.append(start_label)
                    doc_start_pos.extend([0]*(len(tmp_subword_lst) - 1))

                    doc_end_pos.append(end_label)
                    doc_end_pos.extend([0]*(len(tmp_subword_lst) - 1))

                    all_doc_tokens.extend(tmp_subword_lst)
                elif len(tmp_subword_lst) == 1:
                    offset_idx_dict[idx] = len(all_doc_tokens)
                    doc_start_pos.append(start_label)
                    doc_end_pos.append(end_label)
                    all_doc_tokens.extend(tmp_subword_lst)
                else:
                    raise ValueError("Please check the result of tokenizer !!! !!! ")

            if entity_scheme == "bes":
                for span_item in example.span_position:
                    s_idx, e_idx = span_item.split(";")
                    if offset_idx_dict[int(s_idx)] <= max_tokens_for_doc and offset_idx_dict[int(e_idx)] <= max_tokens_for_doc :
                        doc_span_pos[len(query_tokens)+2+offset_idx_dict[int(s_idx)]][len(query_tokens)+2+offset_idx_dict[int(e_idx)]] = 1
                        doc_span_pos[len(query_tokens) + 2 + offset_idx_dict[int(e_idx)]][len(query_tokens) + 2 + offset_idx_dict[int(s_idx)]] = 1
                    else:
                        continue

        assert len(all_doc_tokens) == len(doc_start_pos)
        assert len(all_doc_tokens) == len(doc_end_pos)
        assert len(doc_start_pos) == len(doc_end_pos)

        if len(all_doc_tokens) >= max_tokens_for_doc:
            all_doc_tokens = all_doc_tokens[: max_tokens_for_doc]
            doc_start_pos = doc_start_pos[: max_tokens_for_doc]
            doc_end_pos = doc_end_pos[: max_tokens_for_doc]
        if len(example.start_position) == 0 and len(example.end_position) == 0:
            doc_span_pos = np.zeros((max_seq_length, max_seq_length), dtype=int)


        # input_mask:
        #   the mask has 1 for real tokens and 0 for padding tokens.
        #   only real tokens are attended to.
        # segment_ids:
        #   segment token indices to indicate first and second portions of the inputs.
        input_tokens = []
        segment_ids = []
        input_mask = []
        start_pos = []
        end_pos = []

        input_tokens.append("[CLS]")
        segment_ids.append(0)
        start_pos.append(0)
        end_pos.append(0)

        # for query
        for query_item in query_tokens:  # query在tokenizer之后不需要还原回去
            input_tokens.append(query_item)
            segment_ids.append(0)
            start_pos.append(0)
            end_pos.append(0)

        # for passage
        input_tokens.append("[SEP]")
        segment_ids.append(0)
        input_mask.append(1)
        start_pos.append(0)
        end_pos.append(0)

        input_tokens.extend(all_doc_tokens)
        segment_ids.extend([1] * len(all_doc_tokens))
        start_pos.extend(doc_start_pos)
        end_pos.extend(doc_end_pos)

        input_tokens.append("[SEP]")
        segment_ids.append(1)
        start_pos.append(0)
        end_pos.append(0)
        input_mask = [1] * len(input_tokens)
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

        if entity_scheme == "bes":
            span_label_mask = np.zeros((max_seq_length, max_seq_length), dtype=int)
            span_label_mask[len(query_tokens):len(input_ids), len(query_tokens):len(input_ids)] = 1
        else:
            span_label_mask = None

        # zero-padding up to the sequence length
        if len(input_ids) < max_seq_length and pad_sign:
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            start_pos += padding
            end_pos += padding

        input_ids = np.array(input_ids, dtype=np.int32)
        input_mask = np.array(input_mask, dtype=np.int32)
        segment_ids = np.array(segment_ids, dtype=np.int32)
        start_pos = np.array(start_pos, dtype=np.int32)
        end_pos = np.array(end_pos, dtype=np.int32)
        doc_span_pos = np.array(doc_span_pos, dtype=np.int32)
        span_label_mask = np.array(span_label_mask, dtype=np.int32)

        features.append(
            InputFeatures(
                unique_id=example.qas_id,
                tokens=input_tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_pos,
                end_position=end_pos,
                span_position=doc_span_pos,
                span_label_mask=span_label_mask,
                is_impossible=example.is_impossible,
                ner_cate=label_map[example.ner_cate]
                ))

    return features


def read_mrc_ner_examples(input_file, is_training=True, with_negative=True):
    """
    Desc:
        read MRC-NER data
    """

    with open(input_file, "r") as f:
        input_data = json.load(f)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        qas_id = entry["qas_id"]
        query_item = entry["query"]
        context_item = entry["context"]
        start_position = entry["start_position"]
        end_position = entry["end_position"]
        is_impossible = entry["impossible"]
        ner_cate = entry["entity_label"]
        span_position = entry["span_position"]

        example = InputExample(qas_id=qas_id,
            query_item=query_item,
            context_item=context_item,
            start_position=start_position,
            end_position=end_position,
            span_position=span_position,
            is_impossible=is_impossible,
            ner_cate=ner_cate)
        examples.append(example)
    return examples


def _generate_bmes_label_sequence(sequence_length, start_pos_lst, end_pos_lst):
    """
    Assume that O -> 0, B -> 1, M -> 2, E -> 3, S -> 4.
    """

    target_symbol_sequence = ["O"] * sequence_length
    target_label_sequence = [0] * sequence_length

    for tmp_start, tmp_end in zip(start_pos_lst, end_pos_lst):
        if tmp_start == tmp_end:
            target_label_sequence[tmp_start] = 4
            target_symbol_sequence[tmp_start] = "S"

        target_label_sequence[tmp_start] = 1
        target_symbol_sequence[tmp_start] = "B"

        target_label_sequence[tmp_end] = 3
        target_symbol_sequence[tmp_end] = "E"

        for tmp_middle in range(tmp_start+1, tmp_end):
            target_label_sequence[tmp_middle] = 2
            target_symbol_sequence[tmp_middle] = "M"

    return target_symbol_sequence, target_label_sequence

