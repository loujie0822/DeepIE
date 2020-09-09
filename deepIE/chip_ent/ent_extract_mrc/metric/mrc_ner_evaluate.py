#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Author: xiaoy li
# description:
#


import math
from metric import flat_span_f1
from metric import nest_span_f1


def compute_acc(pred_label, gold_label):
    dict_match = list(filter(lambda x: x[0] == x[1], zip(pred_label, gold_label)))
    acc = len(dict_match) / float(len(gold_label))
    return acc


def update_label_lst(label_lst):
    """
    Desc:
        label_lst is a list of entity category such as: ["NS", "NT", "NM"]
        after update, ["B-NS", "E-NS", "S-NS"]
    """
    update_label_lst = []
    for label_item in label_lst:
        if label_item != "O":
            update_label_lst.append("B-{}".format(label_item))
            update_label_lst.append("M-{}".format(label_item))
            update_label_lst.append("E-{}".format(label_item))
            update_label_lst.append("S-{}".format(label_item))
        else:
            update_label_lst.append(label_item)
    return update_label_lst


def flat_transform_bmes_label(start_labels, end_labels, span_labels, ner_cate, threshold=1):
    bmes_labels = len(start_labels) * ["O"]
    start_labels = [idx for idx, tmp in enumerate(start_labels) if tmp != 0]
    end_labels = [idx for idx, tmp in enumerate(end_labels) if tmp != 0]

    for start_item in start_labels:
        bmes_labels[start_item] = "B-{}".format(ner_cate)
    for end_item in end_labels:
        bmes_labels[end_item] = "E-{}".format(ner_cate)

    for tmp_start in start_labels:
        tmp_end = [tmp for tmp in end_labels if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end)
        score = span_labels[tmp_start][tmp_end]
        if score >= threshold:
            if tmp_start != tmp_end:
                for i in range(tmp_start + 1, tmp_end):
                    bmes_labels[i] = "M-{}".format(ner_cate)
            else:
                bmes_labels[tmp_end] = "S-{}".format(ner_cate)
    return bmes_labels


def nested_transform_span_triple(start_labels, end_labels, span_labels, ner_cate, print_info=False, threshold=0.5):
    span_triple_lst = []
    # element in span_triple_lst is (ner_cate, start_index, end_index)

    start_labels = [idx for idx, tmp in enumerate(start_labels) if tmp != 0]
    end_labels = [idx for idx, tmp in enumerate(end_labels) if tmp != 0]

    for tmp_start in start_labels:
        tmp_end = [tmp for tmp in end_labels if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        for candidate_end in tmp_end:
            if span_labels[tmp_start][candidate_end] >= threshold:
                tmp_tag = nest_span_f1.Tag(ner_cate, tmp_start, candidate_end)
                span_triple_lst.append(tmp_tag)

    return span_triple_lst


def flat_ner_performance(pred_start, pred_end, pred_span, \
                         gold_start, gold_end, gold_span, ner_cate, label_lst, threshold=0.5, dims=2):
    # transform label index to label tag: "3" -> "PER"
    cate_idx2label = {idx: value for idx, value in enumerate(label_lst)}
    # label_list: ["PER", "ORG", "O", "LOC"]
    up_label_lst = update_label_lst(label_lst)
    # up_label_lst: ["B-PER", "M-PER", "E-PER", "S-PER", ]
    label2idx = {label: i for i, label in enumerate(up_label_lst)}
    # label2idx: {"B-PER": 1, "M-PER": 2, "O": 3, }

    if dims == 1:
        # ner_cate: one of [0, 1, 3]
        ner_cate = cate_idx2label[ner_cate]
        # ner_cate: one of ["PER", "LOC", "ORG"]

        # transform (begin, end, span) labels to conventional BMESO tagging scheme
        # pred_bmes_label, gold_bmes_label: a list of ["B-PER", "M-PER", ]
        pred_bmes_label = flat_transform_bmes_label(pred_start, pred_end, pred_span, ner_cate, threshold=threshold)
        gold_bmes_label = flat_transform_bmes_label(gold_start, gold_end, gold_span, ner_cate, threshold=threshold)

        # map a list of string ["B-PER", "M-PER", ] to a list of label index [1, 2, 3]
        pred_bmes_idx = [label2idx[tmp] for tmp in pred_bmes_label]
        gold_bmes_idx = [label2idx[tmp] for tmp in gold_bmes_label]

        return pred_bmes_idx, gold_bmes_idx, pred_bmes_label, gold_bmes_label

    elif dims == 2:
        pred_bmes_idx_lst = []
        gold_bmes_idx_lst = []
        pred_bmes_label_lst = []
        gold_bmes_label_lst = []
        acc_lst = []

        for pred_start_item, pred_end_item, pred_span_item, gold_start_item, gold_end_item, gold_span_item, ner_cate_item in zip(
                pred_start, pred_end, pred_span, gold_start, gold_end, gold_span, ner_cate):
            item_pred_bmes_idx, item_gold_bmes_idx, item_pred_bmes_label, item_gold_bmes_label = flat_ner_performance(
                pred_start_item, pred_end_item, pred_span_item,
                gold_start_item, gold_end_item, gold_span_item,
                ner_cate_item, label_lst, threshold=threshold, dims=1)

            pred_bmes_idx_lst.append(item_pred_bmes_idx)
            gold_bmes_idx_lst.append(item_gold_bmes_idx)

            pred_bmes_label_lst.append(item_pred_bmes_label)
            gold_bmes_label_lst.append(item_gold_bmes_label)

            # compute_acc
            tmp_acc = compute_acc(pred_bmes_idx_lst, gold_bmes_idx_lst)
            acc_lst.append(tmp_acc)

        # input pred_bmes_idx_lst: [[1, 2, 3, ], [1, 3, 3], [0, 0, 0 ] ]
        # input gold_bmes_idx_lst: [[1, 2, 3, ], [1, 3, 3], [0, 0, 0 ] ]
        # up_label_lst: ["B-PER", "M-PER", "E-PER"]
        result_score = flat_span_f1.mask_span_f1(pred_bmes_idx_lst, gold_bmes_idx_lst, label_list=up_label_lst)
        span_precision, span_recall, span_f1 = result_score["span-precision"], result_score["span-recall"], \
                                               result_score["span-f1"]
        average_acc = sum(acc_lst) / (len(acc_lst) * 1.0)

        return average_acc, span_precision, span_recall, span_f1


def nested_ner_performance(pred_start, pred_end, pred_span, gold_start, gold_end, gold_span, ner_cate, label_lst,
                           threshold=0.5, dims=2):
    # transform label index to label tag: "3" -> "per"
    cate_idx2label = {idx: value for idx, value in enumerate(label_lst)}
    # label_list: ["PER", "ORG", "O", "LOC"]

    if dims == 1:
        # ner_cate: one of [1, 2, 3]
        ner_cate = cate_idx2label[ner_cate]
        # ner_cate: one of ["PER", "LOC", "ORG"]

        # transform (begin, end, span) labels to span list
        pred_span_triple = nested_transform_span_triple(pred_start, pred_end, pred_span, ner_cate, threshold=threshold)
        gold_span_triple = nested_transform_span_triple(gold_start, gold_end, gold_span, ner_cate, threshold=threshold)

        return pred_span_triple, gold_span_triple
    elif dims == 2:
        pred_span_triple_lst = []
        gold_span_triple_lst = []

        acc_lst = []

        for pred_start_item, pred_end_item, pred_span_item, gold_start_item, gold_end_item, gold_span_item, ner_cate_item in zip(
                pred_start, pred_end, pred_span, gold_start, gold_end, gold_span, ner_cate):
            pred_span_triple, gold_span_triple = nested_ner_performance(pred_start_item, pred_end_item, \
                                                                        pred_span_item, gold_start_item, gold_end_item,
                                                                        gold_span_item, ner_cate_item, label_lst,
                                                                        dims=1)

            pred_span_triple_lst.append(pred_span_triple)
            gold_span_triple_lst.append(gold_span_triple)

            tmp_acc_s = compute_acc(pred_start_item, gold_start_item)
            tmp_acc_e = compute_acc(pred_end_item, gold_end_item)
            acc_lst.append((tmp_acc_s + tmp_acc_e) / 2.0)

        span_precision, span_recall, span_f1 = nest_span_f1.nested_calculate_f1(pred_span_triple_lst,
                                                                                gold_span_triple_lst, dims=2)
        average_acc = sum(acc_lst) / (len(acc_lst) * 1.0)

        return average_acc, span_precision, span_recall, span_f1

    else:
        raise ValueError("Please notice that dims can only be 1 or 2 !")




