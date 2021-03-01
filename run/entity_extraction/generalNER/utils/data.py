# -*- coding: utf-8 -*-
import sys

from tqdm import tqdm

from run.entity_extraction.generalNER.utils.alphabet import Alphabet
from run.entity_extraction.generalNER.utils.functions import *

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"


class Data:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 400
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_word_emb = True
        self.norm_biword_emb = True
        self.norm_gaz_emb = False
        self.min_freq = 1
        self.word_alphabet = Alphabet('word')
        self.biword_alphabet = Alphabet('biword', min_freq=self.min_freq)
        self.label_alphabet = Alphabet('label', True)
        self.device = 0
        self.transfer=False

        self.biword_count = {}

        self.tagScheme = "NoSeg"

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []

        self.train_split_index = []
        self.dev_split_index = []

        self.use_bigram = True
        self.word_emb_dim = 50
        self.biword_emb_dim = 50

        self.pretrain_word_embedding = None
        self.pretrain_biword_embedding = None
        self.label_size = 0
        self.word_alphabet_size = 0
        self.biword_alphabet_size = 0
        self.label_alphabet_size = 0

        self.bertpath = 'transformer_cpt/bert/'
        self.bert_finetune = False

        ### hyperparameters
        self.HP_iteration = 100
        self.HP_batch_size = 10
        self.HP_hidden_dim = 128
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.HP_gpu = True
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = 5.0
        self.HP_momentum = 0
        self.HP_num_layer = 4
        self.HP_warm_up = False

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Tag          scheme: %s" % (self.tagScheme))
        print("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s" % (self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s" % (self.number_normalized))
        print("     Use          bigram: %s" % (self.use_bigram))
        print("     Word  alphabet size: %s" % (self.word_alphabet_size))
        print("     Biword alphabet size: %s" % (self.biword_alphabet_size))
        print("     Label alphabet size: %s" % (self.label_alphabet_size))
        print("     Word embedding size: %s" % (self.word_emb_dim))
        print("     Biword embedding size: %s" % (self.biword_emb_dim))
        print("     Norm     word   emb: %s" % (self.norm_word_emb))
        print("     Norm     biword emb: %s" % (self.norm_biword_emb))
        print("     Norm     gaz    emb: %s" % (self.norm_gaz_emb))
        print("     bert file is : %s" % (self.bertpath))
        print("     Train instance number: %s" % (len(self.train_texts)))
        print("     Dev   instance number: %s" % (len(self.dev_texts)))
        print("     Test  instance number: %s" % (len(self.test_texts)))
        print("     Raw   instance number: %s" % (len(self.raw_texts)))
        print("     Hyperpara  iteration: %s" % (self.HP_iteration))
        print("     Hyperpara  batch size: %s" % (self.HP_batch_size))
        print("     Hyperpara          lr: %s" % (self.HP_lr))
        print("     Hyperpara    lr_decay: %s" % (self.HP_lr_decay))
        print("     Hyperpara     HP_clip: %s" % (self.HP_clip))
        print("     Hyperpara    momentum: %s" % (self.HP_momentum))
        print("     Hyperpara  hidden_dim: %s" % (self.HP_hidden_dim))
        print("     Hyperpara     dropout: %s" % (self.HP_dropout))
        print("     Hyperpara  lstm_layer: %s" % (self.HP_lstm_layer))
        print("     Hyperpara      bilstm: %s" % (self.HP_bilstm))
        print("     Hyperpara         GPU: %s" % (self.HP_gpu))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def refresh_label_alphabet(self, input_file):
        old_size = self.label_alphabet_size
        self.label_alphabet.clear(True)
        in_lines = open(input_file, 'r', encoding="utf-8").readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                label = pairs[-1]
                self.label_alphabet.add(label)
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
        self.fix_alphabet()
        print("Refresh label alphabet finished: old:%s -> new:%s" % (old_size, self.label_alphabet_size))

    def build_alphabet(self, input_file,only_label=False,use_label=True):
        in_lines = open(input_file, 'r', encoding="utf-8").readlines()
        seqlen = 0
        for idx in tqdm(range(len(in_lines))):
            line = in_lines[idx]
            if len(line) > 2:
                pairs = line.strip().split()
                if len(pairs) == 1:
                    # print(pairs[0])
                    word = ' '
                else:
                    word = pairs[0]
                if self.number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                if use_label:
                    self.label_alphabet.add(label)
                if only_label:
                    continue
                self.word_alphabet.add(word.lower())
                if idx < len(in_lines) - 1 and len(in_lines[idx + 1]) > 2:
                    biword = word + in_lines[idx + 1].strip().split()[0]
                    biword = normalize_word(biword)
                else:
                    biword = word + NULLKEY
                self.biword_alphabet.add(biword.lower())
                self.biword_count[biword] = self.biword_count.get(biword, 0) + 1
                seqlen += 1
            else:
                seqlen = 0

        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"

    def fix_alphabet(self):

        self.word_alphabet.close()
        self.biword_alphabet.close()
        self.label_alphabet.close()
        self.word_alphabet_size = self.word_alphabet.size()
        self.biword_alphabet_size = self.biword_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()

    def build_word_pretrain_emb(self, emb_path):
        print("build word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet,
                                                                                   self.word_emb_dim,
                                                                                   self.norm_word_emb)

    def build_biword_pretrain_emb(self, emb_path):
        print("build biword pretrain emb...")
        self.pretrain_biword_embedding, self.biword_emb_dim = build_pretrain_embedding(emb_path, self.biword_alphabet,
                                                                                       self.biword_emb_dim,
                                                                                       self.norm_biword_emb)

    def write_decoded_results(self, output_file, predict_results, name):
        fout = open(output_file, 'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert (sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                ## content_list[idx] is a list with [word, char, label]
                fout.write(content_list[idx][0][idy].encode('utf-8') + " " + predict_results[idx][idy] + '\n')

            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s" % (name, output_file))

    def generate_instance(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                             self.label_alphabet, self.number_normalized,
                                                             self.MAX_SENTENCE_LENGTH, self.bertpath)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                         self.label_alphabet, self.number_normalized,
                                                         self.MAX_SENTENCE_LENGTH, self.bertpath)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                           self.label_alphabet, self.number_normalized,
                                                           self.MAX_SENTENCE_LENGTH, self.bertpath)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                         self.label_alphabet, self.number_normalized,
                                                         self.MAX_SENTENCE_LENGTH, self.bertpath)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % (name))
