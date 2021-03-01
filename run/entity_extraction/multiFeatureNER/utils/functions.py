# -*- coding: utf-8 -*-

import numpy as np

# from transformers.tokenization_bert import BertTokenizer
from transformers import BertTokenizer
from transformers.tokenization_xlnet import XLNetTokenizer

NULLKEY = "-null-"


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(input_file, word_alphabet, biword_alphabet, label_alphabet, number_normalized,
                  max_sent_length, bertpath):
    tokenizer = BertTokenizer.from_pretrained(bertpath, do_lower_case=True)
    xlnet_tokenizer = XLNetTokenizer.from_pretrained('transformer_cpt/chinese_xlnet_base_pytorch/',
                                                     add_special_tokens=False)
    in_lines = open(input_file, 'r', encoding="utf-8").readlines()
    instence_texts = []
    instence_Ids = []
    word_Ids = []
    biword_Ids = []
    label_Ids = []

    words = []
    biwords = []
    labels = []

    for idx in range(len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split('\t')
            if len(pairs) == 1:
                word = ' '
                # print('word ==  ')
            else:
                word = pairs[0]
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            if idx < len(in_lines) - 1 and len(in_lines[idx + 1]) > 2:
                biword = word + in_lines[idx + 1].strip().split()[0]
                # todo
                biword = normalize_word(biword)
            else:
                biword = word + NULLKEY
            biwords.append(biword)
            words.append(word.lower())
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word.lower()))
            biword_index = biword_alphabet.get_index(biword.lower())
            biword_Ids.append(biword_index)
            label_Ids.append(label_alphabet.get_index(label))


        else:
            # todo 这里直接截断了，做医疗相关不能这么干
            if len(words) <= 0:
                raise ValueError('len(words) <= 0')
            texts = ['[CLS]'] + words[:max_sent_length] + ['[SEP]']

            bert_text_ids = tokenizer.convert_tokens_to_ids(texts)
            xlnet_text_ids = xlnet_tokenizer.convert_tokens_to_ids(words[:max_sent_length])
            instence_texts.append([words, biwords, labels])

            word_Ids = word_Ids[:max_sent_length]
            biword_Ids = biword_Ids[:max_sent_length]
            label_Ids = label_Ids[:max_sent_length]

            assert len(texts) - 2 == len(word_Ids)
            instence_Ids.append([word_Ids, biword_Ids, label_Ids, bert_text_ids, xlnet_text_ids])

            words = []
            biwords = []
            labels = []

            word_Ids = []
            biword_Ids = []
            label_Ids = []

    return instence_texts, instence_Ids


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)

    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.zeros([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    # pretrain_emb[0,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
    for word, index in word_alphabet.instance2index.items():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / word_alphabet.size()))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim
