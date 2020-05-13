# _*_ coding:utf-8 _*_
import argparse
import logging
import os
import pickle

from run.entity_extraction.baseNER.data_loader import Reader, Vocabulary, Feature, StaticEmbedding
from run.entity_extraction.baseNER.train import Trainer
from utils.file_util import load

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_args():
    parser = argparse.ArgumentParser()

    # file parameters
    parser.add_argument("--input", default=None, type=str, required=True)
    parser.add_argument("--output", default=None, type=str, required=False,
                        help="The output directory where the model checkpoints and predictions will be written.")
    # "cpt/baidu_w2v/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5"
    # 'cpt/baidu_w2v/w2v.txt'
    parser.add_argument('--embedding_file', type=str,
                        default='cpt/baidu_w2v/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5')

    # choice parameters
    parser.add_argument('--entity_type', type=str, default='drug')
    parser.add_argument('--use_static_emb', type=bool, default=True)
    parser.add_argument('--use_dynamic_emb', type=bool, default=False)
    parser.add_argument('--bi_char', type=bool, default=True)
    parser.add_argument('--soft_word', type=bool, default=True)
    parser.add_argument('--warm_up', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--encoder', type=str, default='lstm', choices=['lstm'])

    # train parameters
    parser.add_argument('--train_mode', type=str, default="train")
    parser.add_argument("--train_batch_size", default=4, type=int, help="Total batch size for training.")
    parser.add_argument("--dev_batch_size", default=4, type=int, help="Total batch size for dev.")

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--epoch_num", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--patience_stop', type=int, default=10, help='Patience for learning early stop')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

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

    # model parameters
    parser.add_argument("--max_len", default=1000, type=int)
    parser.add_argument('--word_emb_dim', type=int, default=300)
    parser.add_argument('--char_emb_dim', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=150)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=int, default=0.5)
    parser.add_argument('--rnn_encoder', type=str, default='lstm', help="must choose in blow: lstm or gru")
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=False)
    args = parser.parse_args()
    if args.use_static_emb:
        args.cache_data = args.input + '/static_char_cache_data/'
        if args.bi_char:
            args.cache_data = args.input + '/static_bichar_cache_data/'
    elif args.use_dynamic_emb:
        args.cache_data = args.input + '/dynamic_emb_cache_data/'
    else:
        args.cache_data = args.input + '/random_emb_cache_data/'
    return args


def bulid_dataset(args, debug=False):
    train_src = args.input + "/train.txt"
    dev_src = args.input + "/dev.txt"
    cache_data_file = args.cache_data + "/cache_data.pkl"
    cache_data = {}
    if not os.path.exists(cache_data_file):

        reader = Reader(bi_char=args.bi_char)
        train_examples = reader.read_examples(train_src, data_type='train')
        dev_examples = reader.read_examples(dev_src, data_type='dev')

        char_vocab = Vocabulary(min_char_count=1)
        char_vocab.build_vocab(train_examples+dev_examples)
        char_emb, bichar_emb, bichar_vocab = None, None, None
        if args.use_static_emb:
            char_emb = StaticEmbedding(char_vocab, model_path='cpt/gigaword/uni.ite50.vec',
                                       only_norm_found_vector=True).emb_vectors
            if args.bi_char:
                bichar_vocab = Vocabulary(char_type='bichar', min_char_count=1)
                bichar_vocab.build_vocab(train_examples+dev_examples)
                bichar_emb = StaticEmbedding(bichar_vocab, model_path='cpt/gigaword/bi.ite50.vec',
                                             only_norm_found_vector=True).emb_vectors

        cache_data['train_data'] = train_examples
        cache_data['dev_data'] = dev_examples
        cache_data['char_emb'] = char_emb
        cache_data['bichar_emb'] = bichar_emb
        cache_data['char_vocab'] = char_vocab.word2idx
        cache_data['bichar_vocab'] = bichar_vocab.word2idx if bichar_vocab is not None else []
        cache_data['entity_type'] = reader.ent_type

        pickle.dump(cache_data, open(cache_data_file, 'wb'))
    else:
        logging.info('loadding  file {}'.format(cache_data_file))
        cache_data = load(cache_data_file)
    logging.info('train examples size is {}'.format(len(cache_data['train_data'])))
    logging.info('dev examples size is {}'.format(len(cache_data['dev_data'])))
    logging.info("total char vocabulary size is {} ".format(len(cache_data['char_vocab'])))
    logging.info("total bichar vocabulary size is {} ".format(len(cache_data['bichar_vocab'])))
    logging.info("entity type dict is {} ".format(cache_data['entity_type']))

    convert_examples_features = Feature(args, char_vocab=cache_data['char_vocab'],
                                        bichar_vocab=cache_data['bichar_vocab'], entity_type=cache_data['entity_type'])

    train_examples = cache_data['train_data'][:20] if debug else cache_data['train_data']
    dev_examples = cache_data['dev_data'][:20] if debug else cache_data['dev_data']

    train_data_set = convert_examples_features(train_examples, entity_type=args.entity_type, data_type='train')
    dev_data_set = convert_examples_features(dev_examples, entity_type=args.entity_type, data_type='dev')
    train_data_loader = train_data_set.get_dataloader(args.train_batch_size, shuffle=True, pin_memory=args.pin_memory)
    dev_data_loader = dev_data_set.get_dataloader(args.dev_batch_size)

    data_loaders = train_data_loader, dev_data_loader
    eval_examples = train_examples, dev_examples
    model_conf = {'char_vocab': cache_data['char_vocab'], 'bichar_vocab': cache_data['bichar_vocab'],
                  'char_emb': cache_data['char_emb'], 'bichar_emb': cache_data['bichar_emb'],
                  'entity_type': cache_data['entity_type']}

    return eval_examples, data_loaders, model_conf


def main():
    args = get_args()
    if not os.path.exists(args.output):
        print('mkdir {}'.format(args.output))
        os.makedirs(args.output)
    if not os.path.exists(args.cache_data):
        print('mkdir {}'.format(args.cache_data))
        os.makedirs(args.cache_data)

    logger.info("** ** * bulid dataset ** ** * ")

    eval_examples, data_loaders, model_conf = bulid_dataset(args, debug=False)

    trainer = Trainer(args, data_loaders, eval_examples, model_conf)

    if args.train_mode == "train":
        trainer.train(args)
    elif args.train_mode == "eval":
        trainer.resume(args)
        trainer.eval_data_set("train")
        trainer.eval_data_set("dev")
    elif args.train_mode == "resume":
        trainer.show("dev")  # bad case analysis


if __name__ == '__main__':
    main()
