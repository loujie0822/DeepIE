# _*_ coding:utf-8 _*_
import argparse
import json
import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from data.DrugSPOData.yaowu_unify import check_drug_unify
from run.entity_relation_jointed_extraction.drug_jointed_extract.data_loader import Reader, Vocabulary, Feature, \
    DRUG_RELATION, Example
from run.entity_relation_jointed_extraction.drug_jointed_extract.train import Trainer
from utils.data_util import sequence_padding
from utils.file_util import save, load, write_json, read_json

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
                        default='cpt/baidu_w2v/w2v.txt')

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
    parser.add_argument('--bert_hidden_size', type=int, default=768)
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
    word2idx, char2idx, word_emb = None, None, None
    train_src = args.input + "/train_data.json"
    dev_src = args.input + "/dev_data.json"

    train_examples_file = args.cache_data + "/train-examples.pkl"
    dev_examples_file = args.cache_data + "/dev-examples.pkl"

    word_emb_file = args.cache_data + "/word_emb.pkl"
    char_dictionary = args.cache_data + "/char_dict.pkl"
    word_dictionary = args.cache_data + "/word_dict.pkl"

    if not os.path.exists(train_examples_file):

        train_examples = reader.read_examples(train_src, data_type='train')
        dev_examples = reader.read_examples(dev_src, data_type='dev')

        if not args.use_bert:
            # todo : min_word_count=3 ?
            vocab.build_vocab_only_with_char(train_examples, min_char_count=1, min_word_count=5)
            if args.use_word2vec and args.embedding_file:
                word_emb = vocab.make_embedding(vocab=vocab.word_vocab,
                                                embedding_file=args.embedding_file,
                                                emb_size=args.word_emb_size)
                save(word_emb_file, word_emb, message="word_emb embedding")
            save(char_dictionary, vocab.char2idx, message="char dictionary")
            save(word_dictionary, vocab.word2idx, message="char dictionary")
            char2idx = vocab.char2idx
            word2idx = vocab.word2idx
        save(train_examples_file, train_examples, message="train examples")
        save(dev_examples_file, dev_examples, message="dev examples")
    else:
        if not args.use_bert:
            if args.use_word2vec and args.embedding_file:
                word_emb = load(word_emb_file)
            char2idx = load(char_dictionary)
            word2idx = load(word_dictionary)
            logging.info("total char vocabulary size is {} ".format(len(char2idx)))
        train_examples, dev_examples = load(train_examples_file), load(dev_examples_file)

        logging.info('train examples size is {}'.format(len(train_examples)))
        logging.info('dev examples size is {}'.format(len(dev_examples)))

    if not args.use_bert:
        args.char_vocab_size = len(char2idx)
    convert_examples_features = Feature(args, char2idx=char2idx, word2idx=word2idx)

    train_examples = train_examples[:1] if debug else train_examples
    dev_examples = dev_examples[:3] if debug else dev_examples

    train_data_set = convert_examples_features(train_examples, data_type='train')
    dev_data_set = convert_examples_features(dev_examples, data_type='dev')
    train_data_loader = train_data_set.get_dataloader(args.train_batch_size, shuffle=True, pin_memory=args.pin_memory)
    dev_data_loader = dev_data_set.get_dataloader(args.train_batch_size)

    data_loaders = train_data_loader, dev_data_loader
    eval_examples = train_examples, dev_examples

    return eval_examples, data_loaders, word_emb


def main(args):
    if not os.path.exists(args.output):
        print('mkdir {}'.format(args.output))
        os.makedirs(args.output)
    if not os.path.exists(args.cache_data):
        print('mkdir {}'.format(args.cache_data))
        os.makedirs(args.cache_data)

    logger.info("** ** * bulid dataset ** ** * ")
    reader = Reader(seg_char=args.seg_char)
    vocab = Vocabulary()

    eval_examples, data_loaders, word_emb = bulid_dataset(args, reader, vocab, debug=True)

    trainer = Trainer(args, data_loaders, eval_examples, word_emb, spo_conf=DRUG_RELATION)

    if args.train_mode == "train":
        trainer.train(args)
    elif args.train_mode == "eval":
        # trainer.resume(args)
        # trainer.eval_data_set("train")
        trainer.eval_data_set("dev")
    elif args.train_mode == "resume":
        # trainer.resume(args)
        trainer.show("dev")  # bad case analysis


def convert_spo_contour(qids, subject_preds, po_preds, eval_file, answer_dict, id2rel):
    for qid, subject, po_pred in zip(qids.data.cpu().numpy(), subject_preds.data.cpu().numpy(),
                                     po_preds.data.cpu().numpy()):
        if qid == -1:
            continue
        tokens = eval_file[qid.item()].context
        start = np.where(po_pred[:, :, 0] > 0.5)
        end = np.where(po_pred[:, :, 1] > 0.5)

        spoes = []
        for _start, predicate1 in zip(*start):
            if _start >= len(tokens):
                continue
            for _end, predicate2 in zip(*end):
                if _start <= _end < len(tokens) and predicate1 == predicate2:
                    spoes.append((subject, predicate1, (_start, _end)))
                    break
        po_predict = []
        for s, p, o in spoes:
            po_predict.append((tokens[s[0]:s[1] + 1], s[0],
                               id2rel[p],
                               tokens[o[0]:o[1] + 1], o[0])
                              )
        answer_dict[qid][0].append(
            (subject[0], subject[1] + 1))
        answer_dict[qid][1].extend(po_predict)


def text2id(args, char2idx, raw_text: list):
    p_ids = []
    batch_char_ids = []
    examples = []
    for index, text_ in enumerate(raw_text):
        p_ids.append(index)
        char_ids = [char2idx.get(char, 1) for char in text_]
        batch_char_ids.append(char_ids)
        examples.append(
            Example(
                p_id=index,
                context=text_,
            ))
    p_ids = torch.tensor([p_id for p_id in p_ids], dtype=torch.long)
    text_ids = sequence_padding(batch_char_ids, is_float=False)
    return examples, (p_ids, text_ids)


def load_model(args):
    device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
    import run.entity_relation_jointed_extraction.drug_jointed_extract.mpn_drug as mpn
    model = mpn.ERENet(args, word_emb=None, spo_conf=DRUG_RELATION)
    model.to(device)
    checkpoint = torch.load(args.output + "/pytorch_model.bin", map_location='cpu')
    new_cpt = {k: v for k, v in checkpoint.items() if not k.startswith('word_emb.weight')}
    model.load_state_dict(new_cpt)
    return model


def model_predict(args, model, char2idx, raw_text: list):
    id2rel = {item: key for key, item in DRUG_RELATION.items()}
    device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")

    examples, (p_ids, char_ids) = text2id(args, char2idx, raw_text)

    model.eval()
    with torch.no_grad():
        qids, subject_pred, po_pred = model(q_ids=p_ids.to(device),
                                            char_ids=char_ids.to(device), eval_file=examples, is_eval=True)
    answer_dict = {i: [[], []] for i in range(len(examples))}
    convert_spo_contour(qids, subject_pred, po_pred, examples, answer_dict, id2rel)

    res_list = []
    for index, ans in answer_dict.items():

        text = raw_text[index]
        if text.find('方案') != -1:
            q_index = text.find('方案')
        else:
            q_index = -1
        # 三元组处理规则
        # 规则1：关于中心实体「药物」
        # 同一治疗方案中，如果相同「药物名称」由多个，则只取1个，原则上是向后取。注意：要保留同一位置药物对应的多个三元组
        # 如果显示出现'方案'二字，
        # 如果'方案'后有药物，则不要方案前的药物（方案前的药物存在和方案后的药物不一致、但代表相同药物的情况），如：行“DDP+PTX+恩度”方案(顺铂 30mg ivgtt d1-3,紫杉醇 180mg ivgtt d1,恩度
        # TODO 如果'方案'后没有药物，则方案前的药物还是要取回来的
        # 规则2：关于属性处理
        # 重点处理这种情况，可能会存在一个药物对应多个剂量，但多个剂量之间还插入其它药物 ⚠️不能处理下述情况，如（A+B）分别50mg 或者 ⚠️如（A+B+C）口服

        # 规则1
        # tmp = {}
        # for (s, s_index, p, o, o_index) in ans[1]:
        #     # if s_index < q_index:
        #     #     continue
        #     if s not in tmp:
        #         tmp[s] = {s_index: [(s, p, o, o_index)]}
        #     else:
        #         if s_index in tmp[s]:
        #             tmp[s][s_index].append((s, p, o, o_index))
        #         else:
        #             tmp.pop(s)
        #             tmp[s] = {s_index: [(s, p, o, o_index)]}
        # # 规则2
        # tmp_jiliang = {}
        # for s, v in tmp.items():
        #     for s_index, spo_list in v.items():
        #         for (s, p, o, o_index) in spo_list:
        #             if o_index not in tmp_jiliang:
        #                 tmp_jiliang[o_index] = (s_index, s, p, o)
        #             elif tmp_jiliang[o_index][0] < s_index:
        #                 tmp_jiliang.pop(o_index)
        #                 tmp_jiliang[o_index] = (s_index, s, p, o)

        # 规则1 和 规则2 合并
        
        ent_dict = dict()
        for s_start,s_end in ans[0]:
            ent_dict[s_start]=text[s_start:s_end]
        tmp_drug =[]
        tmp_jiliang = {}
        for (s, s_index, p, o, o_index) in ans[1]:
            tmp_drug.append(s)
            if o_index not in tmp_jiliang:
                tmp_jiliang[o_index] = (s_index, s, p, o)
            elif tmp_jiliang[o_index][0] < s_index:
                if tmp_jiliang[o_index][0] in ent_dict:
                    ent_dict.pop(tmp_jiliang[o_index][0])
                tmp_jiliang.pop(o_index)
                tmp_jiliang[o_index] = (s_index, s, p, o)

        rel_ = []
        for o_index, (s_index, s, p, o) in tmp_jiliang.items():
            rel_.append((s, p, o))


        # 药物处理规则
        # 如果三元组为空，但entity不为空，则采用entity_list构建药物
        ent_candidate = []
        # if rel_:
        #     for (s, p, o) in rel_:
        #         ent_candidate.append(s)
        # else:
        #     for (b, e) in ans[0]:
        #         ent_candidate.append(text[b:e])

        for index,ent in ent_dict.items():
            ent_candidate.append(ent)



        # 药物归一、商品名、化学名查询
        ent_candidate =list(set(ent_candidate))
        ent = []
        for drug_name in ent_candidate:
            ent_dict = check_drug_unify(drug_name)
            ent.append(ent_dict)

        # 方案
        # 由实体ent构建方案

        # res_list.append({'text': raw_text[index], 'rel': rel_})
        res_list.append({'方案': '+'.join(list(set(ent_candidate))), '药物名称': ent, '属性': list(set(rel_))})
    return res_list


if __name__ == '__main__':
    args = get_args()
    file_input = False
    if args.train_mode == 'predict':

        char2idx = load(args.cache_data + "/char_dict.pkl")
        args.char_vocab_size = len(char2idx)
        model = load_model(args)
        if file_input:
            file_path = 'data/DrugSPOData/cgywzl.json'
            data_json = read_json(file_path)
            for data_ in tqdm(data_json):
                for data_ans in data_['ans']:
                    if len(data_ans[0]) <= 5:
                        res = [{
                            "方案": "",
                            "药物名称": [],
                            "属性": []
                        }]
                    else:
                        res = model_predict(args, model, char2idx, raw_text=[data_ans[0]])
                    data_ans[0] = {'ans': data_ans[0], 'ans_weizhi': data_ans[1]}
                    data_ans[1] = res[0]
            write_json(data_json, 'data/DrugSPOData/cgywzl_res.json')
        else:
            input_text = ['2017.02.23、2017.03.17行“DDP+PTX+恩度”方案(顺铂 30mg ivgtt d1-3,紫杉醇 180mg ivgtt d1,恩度 210mg civ7天;q3w )化疗2周期,过程顺利']
            res = model_predict(args, model, char2idx, raw_text=input_text)
            # print(res)
            print(input_text)
            print(json.dumps(res, ensure_ascii=False, indent=4))
    else:
        main(args)
# new case
# 2017.01.10予PEM+CBP方案(培美曲塞二钠 0.7g ivgtt d1;q3w,卡铂 300mg ivgtt d1;q3w )化疗1周期,过程顺利,共6周期
# 2019-08-30、2019-09-24、2019-10-15行培美曲塞+顺铂化疗+帕博利珠单抗单抗(200mg)免疫治疗3周期”。过程顺利,无明显不适



# 2018.11.22予“顺铂  40mg  d1-3;依托泊苷  0.16g d1、d2;0.1g d3;21天为一周期”方案化疗,过程顺利
# 2017.5.17予培美曲塞+顺铂化疗1周期(顺铂40mg  d1-3 + 培美曲塞850mg d1;21天/周期),过程顺利
# 培美曲塞 800mg ivgtt d2;q3w)化疗4周期,过程顺利/
# 阿托伐他汀钙片(立普妥) 20mg 口服 1次/日”降脂治疗
# 收住我科,于2019-1-22、2019-02-15行二线姑息化疗:培美曲塞+洛铂+安维汀方案(培美曲塞 0.8 d1+洛铂 45mg d1+安维汀 400mg d2,q3w)化疗2周期,过程顺利
# 2018-02-09、2018-03-02、2018-03-22、2018-04-17、2018-05-11行培美曲塞+顺铂+安维汀方案(培美曲塞 750mg d1+顺铂 30mg d1,40mg d2-3+安维汀 400mg d2)靶向化疗5周期,过程顺利,未见明显不良反应,化疗后咳嗽、咳痰明显减轻,左颈部肿胀消退,疼痛缓解
# 2017.02.23、2017.03.17行“DDP+PTX+恩度”方案(顺铂 30mg ivgtt d1-3,紫杉醇 180mg ivgtt d1,恩度 210mg civ7天;q3w )化疗2周期,过程顺利
# 注射用培美曲塞二钠(力比泰) 0.7g ivgtt d1+卡铂注射液 350mg ivgtt  d1

