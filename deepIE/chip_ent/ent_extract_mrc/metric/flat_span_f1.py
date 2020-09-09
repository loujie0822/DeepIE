# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: span_f1
"""

import json
from typing import List, Set, Tuple

def mask_span_f1(batch_preds, batch_labels, batch_masks=None, label_list: List[str] = None,
                 output_path = None):
    """
    compute  span-based F1
    Args:
        batch_preds: predication . [batch, length]
        batch_labels: ground truth. [batch, length]
        label_list: label_list[idx] = label_idx. one label for every position 
        batch_masks: [batch, length]

    Returns:
        span-based f1

    Examples:
        >>> label_list = ["B-W", "M-W", "E-W", "S-W", "O"]
        >>> batch_golden = [[0, 1, 2, 3, 4], [0, 2, 4]]
        >>> batch_preds = [[0, 1, 2, 3, 4], [4, 4, 4]]
        >>> metric_dic = mask_span_f1(batch_preds=batch_preds, batch_labels=batch_golden, label_list=label_list)
    """
    fake_term = "一"
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    if batch_masks is None:
        batch_masks = [None] * len(batch_preds)

    outputs = []

    for preds, labels, masks in zip(batch_preds, batch_labels, batch_masks):
        if masks is not None:
            preds = trunc_by_mask(preds, masks)
            labels = trunc_by_mask(labels, masks)

        preds = [label_list[idx] if idx < len(label_list) else "O" for idx in preds]
        labels = [label_list[idx] for idx in labels]

        pred_tags: List[Tag] = bmes_decode(char_label_list=[(fake_term, pred) for pred in preds])[1]
        golden_tags: List[Tag] = bmes_decode(char_label_list=[(fake_term, label) for label in labels])[1]

        pred_set: Set[Tuple] = set((tag.begin, tag.end, tag.tag) for tag in pred_tags)
        golden_set: Set[Tuple] = set((tag.begin, tag.end, tag.tag) for tag in golden_tags)
        pred_tags = sorted([list(s) for s in pred_set], key=lambda x: x[0])
        golden_tags = sorted([list(s) for s in golden_set], key=lambda x: x[0])
        outputs.append(
            {
                "preds": " ".join(preds),
                "golden": " ".join(labels),
                "pred_tags:": "|".join(" ".join(str(s) for s in tag) for tag in pred_tags),
                "gold_tags:": "|".join(" ".join(str(s) for s in tag) for tag in golden_tags)
            }
        )

        for pred in pred_set:
            if pred in golden_set:
                true_positives += 1
            else:
                false_positives += 1

        for pred in golden_set:
            if pred not in pred_set:
                false_negatives += 1

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    if output_path:
        json.dump(outputs, open(output_path, "w"), indent=4, sort_keys=True, ensure_ascii=False)
        print(f"Wrote visualization to {output_path}")

    return {
        "span-precision": precision,
        "span-recall": recall,
        "span-f1": f1
    }


def trunc_by_mask(lst: List, masks: List) -> List:
    """mask according to truncate lst"""
    out = []
    for item, mask in zip(lst, masks):
        if mask:
            out.append(item)
    return out



class Tag(object):
    def __init__(self, term, tag, begin, end):
        self.term = term
        self.tag = tag
        self.begin = begin
        self.end = end

    def to_tuple(self):
        return tuple([self.term, self.begin, self.end])

    def __str__(self):
        return str({key: value for key, value in self.__dict__.items()})

    def __repr__(self):
        return str({key: value for key, value in self.__dict__.items()})


def bmes_decode(char_label_list: List[Tuple[str, str]]) -> Tuple[str, List[Tag]]:
    """
    Desc:
        char_label_list: the input of char_label_list respond to one sentence.
            example: [['上', 'B-GPE'], ['海', 'E-GPE'], ['浦', 'B-GPE'], ['东', 'E-GPE'], ['开', 'O'], ['发', 'O'], ['与', 'O'], ['法', 'O'], ['制', 'O'], ['建', 'O'], ['设', 'O'], ['同', 'O'], ['步', 'O']]
    """
    idx = 0
    length = len(char_label_list)
    tags = []
    while idx < length:
        term, label = char_label_list[idx]
        current_label = label[0]

        # correct labels
        if idx + 1 == length and current_label == "B":
            current_label = "S"

        # merge chars
        if current_label == "O":
            idx += 1
            continue
        if current_label == "S":
            tags.append(Tag(term, label[2:], idx, idx + 1))
            idx += 1
            continue
        if current_label == "B":
            end = idx + 1
            while end + 1 < length and char_label_list[end][1][0] == "M":
                end += 1
            if char_label_list[end][1][0] == "E":  # end with E
                entity = "".join(char_label_list[i][0] for i in range(idx, end + 1))
                tags.append(Tag(entity, label[2:], idx, end + 1))
                idx = end + 1
            else:  # end with M/B
                entity = "".join(char_label_list[i][0] for i in range(idx, end))
                tags.append(Tag(entity, label[2:], idx, end))
                idx = end
            continue
        else:
            idx += 1
            continue 

    sentence = "".join(term for term, _ in char_label_list)
    return sentence, tags



    

