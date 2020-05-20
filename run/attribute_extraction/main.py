# _*_ coding:utf-8 _*_
import argparse
import logging
import os

from run.attribute_extraction.data_loader import Reader, Vocabulary, config, Feature
from run.attribute_extraction.train import Trainer
from utils.file_util import save, load

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_args():
    parser = argparse.ArgumentParser()

    # file parameters
    parser.add_argument("--input", default=None, type=str, required=True)
    parser.add_argument("--output"
                        , default=None, type=str, required=False,
                        help="The output directory where the model checkpoints and predictions will be written.")
    # "cpt/baidu_w2v/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5"
    # 'cpt/baidu_w2v/w2v.txt'
    parser.add_argument('--embedding_file', type=str,
                        default='cpt/baidu_w2v/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5')

    # choice parameters
    parser.add_argument('--entity_type', type=str, default='disease')
    parser.add_argument('--use_word2vec', type=bool, default=False)
    parser.add_argument('--use_bert', type=bool, default=False)
    parser.add_argument('--seg_char', type=bool, default=True)

    # train parameters
    parser.add_argument('--train_mode', type=str, default="train")
    parser.add_argument("--train_batch_size", default=4, type=int, help="Total batch size for training.")
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
    parser.add_argument('--word_emb_size', type=int, default=300)
    parser.add_argument('--char_emb_size', type=int, default=300)
    parser.add_argument('--entity_emb_size', type=int, default=300)
    parser.add_argument('--pos_limit', type=int, default=30)
    parser.add_argument('--pos_dim', type=int, default=300)
    parser.add_argument('--pos_size', type=int, default=62)

    parser.add_argument('--hidden_size', type=int, default=150)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=int, default=0.5)
    parser.add_argument('--rnn_encoder', type=str, default='lstm', help="must choose in blow: lstm or gru")
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--transformer_layers', type=int, default=1)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    args = parser.parse_args()
    if args.use_word2vec:
        args.cache_data = args.input + '/char2v_cache_data/'
    elif args.use_bert:
        args.cache_data = args.input + '/char_bert_cache_data/'
    else:
        args.cache_data = args.input + '/char_cache_data/'
    return args


def bulid_dataset(args, reader, vocab, debug=False):
    char2idx, char_emb = None, None
    train_src = args.input + "/train.json"
    dev_src = args.input + "/dev.json"

    train_examples_file = args.cache_data + "/train-examples.pkl"
    dev_examples_file = args.cache_data + "/dev-examples.pkl"

    char_emb_file = args.cache_data + "/char_emb.pkl"
    char_dictionary = args.cache_data + "/char_dict.pkl"

    if not os.path.exists(train_examples_file):

        train_examples = reader.read_examples(train_src, data_type='train')
        dev_examples = reader.read_examples(dev_src, data_type='dev')

        if not args.use_bert:
            # todo : min_word_count=3 ?
            vocab.build_vocab_only_with_char(train_examples, min_char_count=1)
            if args.use_word2vec and args.embedding_file:
                char_emb = vocab.make_embedding(vocab=vocab.char_vocab,
                                                embedding_file=args.embedding_file,
                                                emb_size=args.word_emb_size)
                save(char_emb_file, char_emb, message="char embedding")
            save(char_dictionary, vocab.char2idx, message="char dictionary")
            char2idx = vocab.char2idx
        save(train_examples_file, train_examples, message="train examples")
        save(dev_examples_file, dev_examples, message="dev examples")
    else:
        if not args.use_bert:
            if args.use_word2vec and args.embedding_file:
                char_emb = load(char_emb_file)
            char2idx = load(char_dictionary)
            logging.info("total char vocabulary size is {} ".format(len(char2idx)))
        train_examples, dev_examples = load(train_examples_file), load(dev_examples_file)

        logging.info('train examples size is {}'.format(len(train_examples)))
        logging.info('dev examples size is {}'.format(len(dev_examples)))

    if not args.use_bert:
        args.vocab_size = len(char2idx)
    convert_examples_features = Feature(args, token2idx_dict=char2idx)

    train_examples = train_examples[:10] if debug else train_examples
    dev_examples = dev_examples[:10] if debug else dev_examples

    train_data_set = convert_examples_features(train_examples, entity_type=args.entity_type,
                                               data_type='train')
    dev_data_set = convert_examples_features(dev_examples, entity_type=args.entity_type,
                                             data_type='dev')
    train_data_loader = train_data_set.get_dataloader(args.train_batch_size, shuffle=True, pin_memory=args.pin_memory)
    dev_data_loader = dev_data_set.get_dataloader(args.train_batch_size)

    data_loaders = train_data_loader, dev_data_loader
    eval_examples = train_examples, dev_examples

    return eval_examples, data_loaders, char_emb


# TODO
'''
1、增加自动构建 词向量的逻辑
2、对比不同实体感知构建方式
3、增加elmo方式
4、增加对抗训练

'''


def main():
    args = get_args()
    if not os.path.exists(args.output):
        print('mkdir {}'.format(args.output))
        os.makedirs(args.output)
    if not os.path.exists(args.cache_data):
        print('mkdir {}'.format(args.cache_data))
        os.makedirs(args.cache_data)

    logger.info("** ** * bulid dataset ** ** * ")
    reader = Reader(seg_char=args.seg_char, max_len=args.max_len, entity_type=args.entity_type)
    vocab = Vocabulary()

    eval_examples, data_loaders, char_emb = bulid_dataset(args, reader, vocab, debug=False)

    trainer = Trainer(args, data_loaders, eval_examples, char_emb, attribute_conf=config[args.entity_type])

    if args.train_mode == "train":
        trainer.train(args)
    elif args.train_mode == "eval":
        trainer.resume(args)
        trainer.eval_data_set("train")
        trainer.eval_data_set("dev")
    elif args.train_mode == "resume":
        # trainer.resume(args)
        trainer.show("dev")  # bad case analysis


if __name__ == '__main__':
    main()
