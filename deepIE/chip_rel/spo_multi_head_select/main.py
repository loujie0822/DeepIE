# _*_ coding:utf-8 _*_
"""
mhs_bert:

"""
import argparse
import logging
import os
import random
from warnings import simplefilter

import numpy as np
import torch
from transformers import BertTokenizer

from deepIE.chip_rel.spo_multi_head_select.data_loader_char_total_sub import Reader, Feature
from deepIE.chip_rel.spo_multi_head_select.train import Trainer
from deepIE.config.config import CMeIE_CONFIG
from utils.file_util import save, load

simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_args():
    parser = argparse.ArgumentParser()

    # file parameters
    parser.add_argument("--input", default=None, type=str, required=True)
    parser.add_argument("--res_path", default=None, type=str, required=False)
    parser.add_argument("--output"
                        , default=None, type=str, required=False,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # choice parameters
    parser.add_argument('--spo_version', type=str, default="v1")

    # train parameters
    parser.add_argument('--train_mode', type=str, default="train")
    parser.add_argument("--train_batch_size", default=4, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--epoch_num", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--patience_stop', type=int, default=10, help='Patience for learning early stop')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument("--debug",
                        action='store_true', )
    parser.add_argument("--use_bert",
                        action='store_true', )
    parser.add_argument("--diff_lr",
                        action='store_true', )

    # bert parameters
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--bert_model", default=None, type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    # parser.add_argument("--tokenizer_path", default='bert-base-chinese', type=str)

    # model parameters
    parser.add_argument("--max_len", default=1000, type=int)
    parser.add_argument('--entity_emb_size', type=int, default=300)
    parser.add_argument('--pos_limit', type=int, default=30)
    parser.add_argument('--pos_dim', type=int, default=300)
    parser.add_argument('--pos_size', type=int, default=62)

    parser.add_argument('--hidden_size', type=int, default=150)
    parser.add_argument('--bert_hidden_size', type=int, default=768)
    parser.add_argument('--dropout', type=int, default=0.5)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=False)


    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--rel_emb_size', type=int, default=100)
    parser.add_argument('--ent_emb_size', type=int, default=100)
    args = parser.parse_args()
    args.cache_data = args.input + '/{}_mhs_cache_data_{}/'.format(str(args.bert_model).split('/')[1],
                                                                    str(args.max_len))
    return args


def bulid_dataset(args, spo_config, reader, tokenizer, debug=False):
    train_src = args.input + "/train_data.json"
    dev_src = args.input + "/val_data.json"
    test_src = args.input + "/test1.json"

    train_examples_file = args.cache_data + "/train-examples.pkl"
    dev_examples_file = args.cache_data + "/dev-examples.pkl"
    test_examples_file = args.cache_data + "/test-examples.pkl"

    if not os.path.exists(train_examples_file):
        train_examples = reader.read_examples(train_src, data_type='train')
        dev_examples = reader.read_examples(dev_src, data_type='dev')
        test_examples = reader.read_examples(test_src, data_type='test')

        save(train_examples_file, train_examples, message="train examples")
        save(dev_examples_file, dev_examples, message="dev examples")
        save(test_examples_file, test_examples, message="test examples")
    else:
        logging.info('loading train cache_data {}'.format(train_examples_file))
        logging.info('loading dev cache_data {}'.format(dev_examples_file))
        logging.info('loading test cache_data {}'.format(test_examples_file))
        train_examples, dev_examples, test_examples = load(train_examples_file), load(dev_examples_file), load(
            test_examples_file)

        logging.info('train examples size is {}'.format(len(train_examples)))
        logging.info('dev examples size is {}'.format(len(dev_examples)))
        logging.info('test examples size is {}'.format(len(test_examples)))

    convert_examples_features = Feature(max_len=args.max_len, spo_config=spo_config, tokenizer=tokenizer)

    train_examples = train_examples[:2] if debug else train_examples
    # train_examples = train_examples[:1000]
    dev_examples = dev_examples[:2] if debug else dev_examples
    test_examples = test_examples[:2] if debug else test_examples

    train_data_set = convert_examples_features(train_examples, data_type='train')
    dev_data_set = convert_examples_features(dev_examples, data_type='dev')
    test_data_set = convert_examples_features(test_examples, data_type='test')

    train_data_loader = train_data_set.get_dataloader(args.train_batch_size, shuffle=True, pin_memory=args.pin_memory)
    dev_data_loader = dev_data_set.get_dataloader(args.train_batch_size)
    test_data_loader = test_data_set.get_dataloader(args.train_batch_size)

    data_loaders = train_data_loader, dev_data_loader, test_data_loader
    eval_examples = train_examples, dev_examples, test_examples

    return eval_examples, data_loaders, tokenizer


def main():
    args = get_args()
    if not os.path.exists(args.output):
        print('mkdir {}'.format(args.output))
        os.makedirs(args.output)
    if not os.path.exists(args.cache_data):
        print('mkdir {}'.format(args.cache_data))
        os.makedirs(args.cache_data)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info("** ** * bulid dataset ** ** * ")

    spo_conf = CMeIE_CONFIG
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    reader = Reader(spo_conf, tokenizer, max_seq_length=args.max_len)
    eval_examples, data_loaders, tokenizer = bulid_dataset(args, spo_conf, reader, tokenizer, debug=args.debug)
    trainer = Trainer(args, data_loaders, eval_examples, spo_conf=spo_conf, tokenizer=tokenizer)

    if args.train_mode == "train":
        trainer.train(args)
    elif args.train_mode == "eval":
        # trainer.resume(args)
        # trainer.eval_data_set("train")
        trainer.eval_data_set("dev")
    elif args.train_mode == "predict":
        trainer.predict_data_set("test")
    elif args.train_mode == "resume":
        # trainer.resume(args)
        trainer.show("dev")  # bad case analysis


if __name__ == '__main__':
    main()
