#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: xiaoyli  
# description:
# 



class Tag(object):
    def __init__(self, tag, begin, end):
        self.tag = tag
        self.begin = begin
        self.end = end

    def to_tuple(self):
        return tuple([self.tag, self.begin, self.end])

    def __str__(self):
        return str({key: value for key, value in self.__dict__.items()})

    def __repr__(self):
        return str({key: value for key, value in self.__dict__.items()})


def nested_calculate_f1(pred_span_tag_lst, gold_span_tag_lst, dims=2):
    if dims == 2:
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred_span_tags, gold_span_tags in zip(pred_span_tag_lst, gold_span_tag_lst):
            pred_set = set((tag.begin, tag.end, tag.tag) for tag in pred_span_tags)
            gold_set = set((tag.begin, tag.end, tag.tag) for tag in gold_span_tags)

            for pred in pred_set:
                if pred in gold_set:
                    true_positives += 1
                else:
                    false_positives += 1

            for pred in gold_set:
                if pred not in pred_set:
                    false_negatives += 1


        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return precision, recall, f1 
    
    else:
        raise ValueError("Can not be other number except 2 !")


