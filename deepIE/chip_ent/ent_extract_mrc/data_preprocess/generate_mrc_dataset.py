#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# description:
# data_generate.py 
# transform data from sequence labeling to mrc formulation 
# ----------------------------------------------------------
# input data structure 
# --------------------------------------------------------- 
# this module is to generate mrc-style ner task. 
# for nested ner, the input file in json


import os
import sys
import json
from tqdm import tqdm


def load_query_map():
    chip_ent_query = 'deepIE/chip_ent/ent_extract_mrc/data_preprocess/chip2020/query/zh_chip_ent.json'

    with open(chip_ent_query, "r") as f:
        query_map = json.load(f)

    return query_map


def generate_query_ner_dataset(source_file_path, dump_file_path, query_sign="default", total=5000):
    query_map = load_query_map()
    label_query_map = query_map[query_sign]
    entity_labels = query_map["labels"]

    target_data = transform_examples_to_qa_features(label_query_map, entity_labels, source_file_path, total=total)

    with open(dump_file_path, "w") as f:
        json.dump(target_data, f, sort_keys=True, ensure_ascii=False, indent=2)


def reformat(data):
    # 将原始描述格式转换为json描述
    anns = {'context': '', 'label': {}}
    terms = data.strip().split('|||')
    anns['context'] = terms[0]
    if len(terms) > 1:
        for elem in terms[1:]:
            if len(elem) == 0:
                continue

            s, e, tag = elem.split(' ' * 4)
            se = '{};{}'.format(s, e)
            if tag in anns['label']:
                anns['label'][tag].append(se)
            else:
                anns['label'][tag] = [se]
    return anns


def transform_examples_to_qa_features(label_query_map, entity_labels, source_filepath, total=5000):
    mrc_ner_dataset = []
    tmp_qas_id = 1
    source_fd = open(source_filepath)

    for line in tqdm(source_fd, total=total, desc='preprocess file: '):
        data_item = reformat(line)
        tmp_query_id = 1
        tmp_context = data_item["context"]
        for label_idx, tmp_label in enumerate(entity_labels):
            tmp_query = label_query_map[tmp_label]

            tmp_start_pos = []
            tmp_end_pos = []
            tmp_entity_pos = []

            start_end_label = data_item["label"][tmp_label] if tmp_label in data_item["label"].keys() else -1

            if start_end_label == -1:
                tmp_impossible = True
            else:
                for start_end_item in start_end_label:
                    start_idx, end_idx = [int(ix) for ix in start_end_item.split(";")]
                    tmp_start_pos.append(start_idx)
                    tmp_end_pos.append(end_idx)
                    tmp_entity_pos.append(start_end_item)
                tmp_impossible = False

            # 对每一个tag类型分别对应到一条样本
            mrc_ner_dataset.append({
                "qas_id": "{}.{}".format(str(tmp_qas_id), str(tmp_query_id)),
                "query": tmp_query,
                "context": tmp_context,
                "entity_label": tmp_label,
                "start_position": tmp_start_pos,
                "end_position": tmp_end_pos,
                "span_position": tmp_entity_pos,
                "impossible": tmp_impossible
                })
            tmp_query_id += 1
        tmp_qas_id += 1

    return mrc_ner_dataset







