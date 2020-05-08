# -*- coding: utf-8 -*-
import warnings

from layers.encoders.transformers.bert.bert_optimization import BertAdam

warnings.filterwarnings("ignore")

import argparse
import copy
import gc
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim

from models.ner_net.augment_ner import GazLSTM as SeqModel
from run.entity_extraction.augmentedNER.data import Data
from run.entity_extraction.augmentedNER.metric import get_ner_fmeasure


def data_initialization(data, train_file, dev_file, test_file):
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    # data.build_alphabet(test_file)
    print(data.word_alphabet_size)
    print(data.biword_alphabet_size)
    data.fix_alphabet()
    return data


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()

    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(int(pred_tag[idx][idy])) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]

        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)

    return pred_label, gold_label


def print_batchword(data, batch_word, n):
    with open("labels/batchwords.txt", "a") as fp:
        for i in range(len(batch_word)):
            words = []
            for id in batch_word[i]:
                words.append(data.word_alphabet.get_instance(id))
            fp.write(str(words))


def save_data_setting(data, save_file):
    new_data = copy.deepcopy(data)
    ## remove input instances
    new_data.train_texts = []
    new_data.dev_texts = []
    new_data.test_texts = []
    new_data.raw_texts = []

    new_data.train_Ids = []
    new_data.dev_Ids = []
    new_data.test_Ids = []
    new_data.raw_Ids = []
    ## save data settings
    with open(save_file, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting saved to file: ", save_file)


def load_data_setting(save_file):
    with open(save_file, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", save_file)
    data.show_data_summary()
    return data


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def set_seed(seed_num=1023):
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)


def evaluate(data, model, name):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
    right_token = 0
    whole_token = 0
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1

    for batch_id in range(total_batch):
        with torch.no_grad():
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = instances[start:end]

            if not instance:
                continue
            batch_word, batch_biword, batch_wordlen, batch_label, mask, batch_bert, bert_mask = batchify_with_label(
                instance, data.HP_gpu)
            tag_seq = model(batch_word, batch_biword, batch_wordlen, mask, batch_bert, bert_mask)

            pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet)
            pred_results += pred_label
            # todo 检查gold_label
            # gold_label_ = data.train_texts[start:end][0][2]
            gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances) / decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    return speed, acc, p, r, f, pred_results


def get_text_input(self, caption):
    caption_tokens = self.tokenizer.tokenize(caption)
    caption_tokens = ['[CLS]'] + caption_tokens + ['[SEP]']
    caption_ids = self.tokenizer.convert_tokens_to_ids(caption_tokens)
    if len(caption_ids) >= self.max_seq_len:
        caption_ids = caption_ids[:self.max_seq_len]
    else:
        caption_ids = caption_ids + [0] * (self.max_seq_len - len(caption_ids))
    caption = torch.tensor(caption_ids)
    return caption


def batchify_with_label(input_batch_list, gpu):
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    biwords = [sent[1] for sent in input_batch_list]
    labels = [sent[2] for sent in input_batch_list]
    ### bert tokens
    bert_ids = [sent[3] for sent in input_batch_list]

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    biword_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len))).byte()
    ### bert seq tensor
    bert_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len + 2))).long()
    bert_mask = autograd.Variable(torch.zeros((batch_size, max_seq_len + 2))).long()

    for b, (seq, biseq, label, seqlen, bert_id) in enumerate(
            zip(words, biwords, labels, word_seq_lengths, bert_ids)):
        word_seq_tensor[b, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[b, :seqlen] = torch.LongTensor(biseq)
        label_seq_tensor[b, :seqlen] = torch.LongTensor(label)
        mask[b, :seqlen] = torch.Tensor([1] * int(seqlen))
        bert_mask[b, :seqlen + 2] = torch.LongTensor([1] * int(seqlen + 2))
        ##bert
        bert_seq_tensor[b, :seqlen + 2] = torch.LongTensor(bert_id)

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        biword_seq_tensor = biword_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()
        bert_seq_tensor = bert_seq_tensor.cuda()
        bert_mask = bert_mask.cuda()

    return word_seq_tensor, biword_seq_tensor, word_seq_lengths, label_seq_tensor, mask, bert_seq_tensor, bert_mask


def train(data, save_model_dir, seg=True, debug=False):
    print("Training with {} model.".format(data.model_type))

    # data.show_data_summary()

    model = SeqModel(data)
    print("finish building model.")

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adamax(parameters, lr=data.HP_lr)

    if data.warm_up:
        print('using warm_up...')
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        num_train_optimization_steps = int(len(data.train_Ids) / data.HP_batch_size) * data.HP_iteration
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=data.HP_lr,
                             warmup=0.01,
                             t_total=num_train_optimization_steps)

    best_dev = -1
    best_dev_p = -1
    best_dev_r = -1

    best_test = -1
    best_test_p = -1
    best_test_r = -1

    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print(("Epoch: %s/%s" % (idx, data.HP_iteration)))
        if not data.warm_up:
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_loss = 0
        batch_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        ## set model in train model
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1

        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            words = data.train_texts[start:end]
            if not instance:
                continue

            batch_word, batch_biword, batch_wordlen, batch_label, mask, batch_bert, bert_mask = batchify_with_label(
                instance, data.HP_gpu)

            instance_count += 1
            loss, tag_seq = model.neg_log_likelihood_loss(batch_word, batch_biword, batch_wordlen, mask,
                                                          batch_label, batch_bert, bert_mask)

            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.data
            total_loss += loss.data
            batch_loss += loss

            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print(("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
                    end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token)))
                sys.stdout.flush()
                sample_loss = 0
            if end % data.HP_batch_size == 0:
                batch_loss.backward()
                optimizer.step()
                model.zero_grad()
                batch_loss = 0
            if debug:
                break

        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print(("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
            end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token)))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print(("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
            idx, epoch_cost, train_num / epoch_cost, total_loss)))

        speed, acc, p, r, f, pred_labels = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if seg:
            current_score = f
            print(("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                dev_cost, speed, acc, p, r, f)))
        else:
            current_score = acc
            print(("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc)))

        if current_score > best_dev:
            if seg:
                print("Exceed previous best f score:", best_dev)

            else:
                print("Exceed previous best acc score:", best_dev)

            model_name = save_model_dir
            torch.save(model.state_dict(), model_name)
            # best_dev = current_score
            best_dev_p = p
            best_dev_r = r

        # ## decode test
        speed, acc, p, r, f, pred_labels = evaluate(data, model, "test")
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if seg:
            current_test_score = f
            print(("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                test_cost, speed, acc, p, r, f)))
        else:
            current_test_score = acc
            print(("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed, acc)))

        if current_score > best_dev:
            best_dev = current_score
            best_test = current_test_score
            best_test_p = p
            best_test_r = r

        print("Best dev score: p:{}, r:{}, f:{}".format(best_dev_p, best_dev_r, best_dev))
        print("Test score: p:{}, r:{}, f:{}".format(best_test_p, best_test_r, best_test))
        gc.collect()

    with open(data.result_file, "a") as f:
        f.write(save_model_dir + '\n')
        f.write("Best dev score: p:{}, r:{}, f:{}\n".format(best_dev_p, best_dev_r, best_dev))
        f.write("Test score: p:{}, r:{}, f:{}\n\n".format(best_test_p, best_test_r, best_test))
        f.close()


def load_model_decode(model_dir, data, name, gpu, seg=True):
    data.HP_gpu = gpu
    print("Load Model from file: ", model_dir)
    model = SeqModel(data)

    model.load_state_dict(torch.load(model_dir))

    print(("Decode %s data ..." % (name)))
    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(data, model, name)
    end_time = time.time()
    time_cost = end_time - start_time
    if seg:
        print(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
            name, time_cost, speed, acc, p, r, f)))
    else:
        print(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" % (name, time_cost, speed, acc)))

    return pred_results


def print_results(pred, modelname=""):
    toprint = []
    for sen in pred:
        sen = " ".join(sen) + '\n'
        toprint.append(sen)
    with open(modelname + '_labels.txt', 'w') as f:
        f.writelines(toprint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', help='Embedding for words', default='None')
    parser.add_argument('--status', choices=['train', 'test'], help='update algorithm', default='train')
    parser.add_argument('--modelpath', default="save_model/")
    parser.add_argument('--modelname', default="model")
    parser.add_argument('--savedset', help='Dir of saved data setting', default="data/save.dset")
    parser.add_argument('--train', default="ResumeNER/train.char.bmes")
    parser.add_argument('--dev', default="ResumeNER/dev.char.bmes")
    parser.add_argument('--test', default="ResumeNER/test.char.bmes")
    parser.add_argument('--seg', default="True")
    parser.add_argument('--extendalphabet', default="True")
    parser.add_argument('--raw')
    parser.add_argument('--output')
    parser.add_argument('--seed', default=1023, type=int)
    parser.add_argument('--labelcomment', default="")
    parser.add_argument('--resultfile', default="result/result.txt")
    parser.add_argument('--num_iter', default=100, type=int)
    parser.add_argument('--num_layer', default=4, type=int)
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--warm_up', dest='warm_up', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--model_type', default='transformer')
    parser.add_argument('--drop', type=float, default=0.5)

    parser.add_argument('--use_biword', dest='use_biword', action='store_true', default=False)
    # parser.set_defaults(use_biword=False)
    parser.add_argument('--use_char', dest='use_char', action='store_true', default=False)
    # parser.set_defaults(use_biword=False)
    parser.add_argument('--use_count', action='store_true', default=True)
    parser.add_argument('--use_bert', action='store_true', default=False)

    args = parser.parse_args()

    seed_num = args.seed
    set_seed(seed_num)

    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    raw_file = args.raw
    # model_dir = args.loadmodel
    output_file = args.output
    if args.seg.lower() == "true":
        seg = True
    else:
        seg = False
    status = args.status.lower()

    save_model_dir = args.modelpath + args.modelname
    save_data_name = args.savedset
    gpu = torch.cuda.is_available()

    char_emb = "cpt/gigaword/uni.ite50.vec"
    bichar_emb = "cpt/gigaword/bi.ite50.vec"
    gaz_file = "cpt/gigaword/ctb.50d.vec"

    sys.stdout.flush()

    if status == 'train':
        if os.path.exists(save_data_name):
            print('Loading processed data')
            with open(save_data_name, 'rb') as fp:
                data = pickle.load(fp)
            data.HP_num_layer = args.num_layer
            data.HP_batch_size = args.batch_size
            data.HP_iteration = args.num_iter
            data.label_comment = args.labelcomment
            data.result_file = args.resultfile
            data.HP_lr = args.lr
            data.use_bigram = args.use_biword
            data.HP_use_char = args.use_char
            data.HP_hidden_dim = args.hidden_dim
            data.HP_dropout = args.drop
            data.HP_use_count = args.use_count
            data.model_type = args.model_type
            data.use_bert = args.use_bert
            data.warm_up = args.warm_up
        else:
            data = Data()
            data.HP_gpu = gpu
            data.HP_use_char = args.use_char
            data.HP_batch_size = args.batch_size
            data.HP_num_layer = args.num_layer
            data.HP_iteration = args.num_iter
            data.use_bigram = args.use_biword
            data.HP_dropout = args.drop
            data.label_comment = args.labelcomment
            data.result_file = args.resultfile
            data.HP_lr = args.lr
            data.HP_hidden_dim = args.hidden_dim
            data.HP_use_count = args.use_count
            data.model_type = args.model_type
            data.use_bert = args.use_bert
            data.warm_up = args.warm_up

            data_initialization(data, train_file, dev_file, test_file)
            data.generate_instance(train_file, 'train')
            data.generate_instance(dev_file, 'dev')
            data.build_word_pretrain_emb(char_emb)
            data.build_biword_pretrain_emb(bichar_emb)

            print('Dumping data')
            with open(save_data_name, 'wb') as f:
                pickle.dump(data, f)
            set_seed(seed_num)

        data.show_data_summary()
        print('data.use_biword=', data.use_bigram)
        print('data.HP_batch_size=', data.HP_batch_size)
        train(data, save_model_dir, seg, debug=False)
    elif status == 'test':
        print('Loading processed data')
        with open(save_data_name, 'rb') as fp:
            data = pickle.load(fp)
        data.HP_num_layer = args.num_layer
        data.HP_iteration = args.num_iter
        data.label_comment = args.labelcomment
        data.result_file = args.resultfile
        data.HP_lr = args.lr
        data.use_bigram = args.use_biword
        data.HP_use_char = args.use_char
        data.model_type = args.model_type
        data.HP_hidden_dim = args.hidden_dim
        data.HP_use_count = args.use_count
        load_model_decode(save_model_dir, data, 'test', gpu, seg)
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")
