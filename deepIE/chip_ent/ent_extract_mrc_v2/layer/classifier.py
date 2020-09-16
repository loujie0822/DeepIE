#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Description:
# 


import torch.nn as nn 
import torch.nn.functional as F


class SingleLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label):
        super(SingleLinearClassifier, self).__init__()
        self.num_label = num_label 
        self.classifier = nn.Linear(hidden_size, num_label)

    def forward(self, input_features):
        features_output = self.classifier(input_features)

        return features_output


class SingleNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(SingleNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_label)

    def forward(self, input_features):
        input_features = self.dropout(input_features)
        features_output = self.classifier(input_features)
        features_output = F.gelu(features_output)
        return features_output


class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label 
        self.classifier1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.classifier2 = nn.Linear(int(hidden_size/2), num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        input_features = self.dropout(input_features)
        features_output1 = self.classifier1(input_features)
        features_output1 = nn.ReLU()(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2 







