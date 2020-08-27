# _*_ coding:utf-8 _*_
import codecs
import logging
import sys
import time
from warnings import simplefilter

import numpy as np
import torch
from tqdm import tqdm

from deepIE.chip_ent.ent_extract_pointer import ent_po_net as ent_net
from deepIE.chip_ent.ent_extract_pointer import ent_po_net_lstm as ent_net_lstm
from layers.encoders.transformers.bert.bert_optimization import BertAdam

simplefilter(action='ignore', category=FutureWarning)
logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, args, data_loaders, examples, spo_conf, tokenizer):

        self.args = args
        self.tokenizer = tokenizer
        self.max_len = args.max_len - 2
        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.load_ent_dict()

        self.id2rel = {item: key for key, item in spo_conf.items()}
        self.rel2id = spo_conf

        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
        if args.encoder_type == 'lstm':
            self.model = ent_net_lstm.EntExtractNet.from_pretrained(args.bert_model, classes_num=len(spo_conf))
        else:
            self.model = ent_net.EntExtractNet.from_pretrained(args.bert_model, classes_num=len(spo_conf))

        self.model.to(self.device)
        if args.train_mode != "train":
            self.resume(args)

        # if self.n_gpu > 1:
        #     logging.info('total gpu num is {}'.format(self.n_gpu))
        #     self.model = nn.DataParallel(self.model.cuda(), device_ids=[0, 1])

        train_dataloader, dev_dataloader, test_dataloader = data_loaders
        train_eval, dev_eval, test_eval = examples
        self.eval_file_choice = {
            "train": train_eval,
            "dev": dev_eval,
            "test": test_eval
        }
        self.data_loader_choice = {
            "train": train_dataloader,
            "dev": dev_dataloader,
            "test": test_dataloader
        }
        # todo 稍后要改成新的优化器，并加入梯度截断
        self.optimizer = self.set_optimizer(args, self.model,
                                            train_steps=(int(
                                                len(train_eval) / args.train_batch_size) + 1) * args.epoch_num)

    def set_optimizer(self, args, model, train_steps=None):
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        # TODO:设置不同学习率
        if args.diff_lr:
            logging.info('设置不同学习率')
            for n, p in param_optimizer:
                if not n.startswith('bert') and not any(nd in n for nd in no_decay):
                    print(n)
            print('+' * 10)
            for n, p in param_optimizer:
                if not n.startswith('bert') and any(nd in n for nd in no_decay):
                    print(n)
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if
                            not any(nd in n for nd in no_decay) and n.startswith('bert')],
                 'weight_decay': 0.01, 'lr': args.learning_rate},
                {'params': [p for n, p in param_optimizer if
                            not any(nd in n for nd in no_decay) and not n.startswith('bert')],
                 'weight_decay': 0.01, 'lr': args.learning_rate * 10},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and n.startswith('bert')],
                 'weight_decay': 0.0, 'lr': args.learning_rate},
                {'params': [p for n, p in param_optimizer if
                            any(nd in n for nd in no_decay) and not n.startswith('bert')],
                 'weight_decay': 0.0, 'lr': args.learning_rate * 10}
            ]
        else:
            logging.info('原始设置学习率设置')

            # TODO:原始设置
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=train_steps)
        return optimizer

    def train(self, args):

        best_f1 = 0.0
        patience_stop = 0
        self.model.train()
        step_gap = 20
        for epoch in range(int(args.epoch_num)):

            global_loss = 0.0

            for step, batch in tqdm(enumerate(self.data_loader_choice[u"train"]), mininterval=5,
                                    desc=u'training at epoch : %d ' % epoch, leave=False, file=sys.stdout):

                loss = self.forward(batch)

                if step % step_gap == 0:
                    global_loss += loss
                    current_loss = global_loss / step_gap
                    print(
                        u"step {} / {} of epoch {}, train/loss: {}".format(step, len(self.data_loader_choice["train"]),
                                                                           epoch, current_loss))
                    global_loss = 0.0

            res_dev = self.eval_data_set("dev")
            if res_dev['f1'] >= best_f1:
                best_f1 = res_dev['f1']
                logging.info("** ** * Saving fine-tuned model ** ** * ")
                model_to_save = self.model.module if hasattr(self.model,
                                                             'module') else self.model  # Only save the model it-self
                output_model_file = args.output + "/pytorch_model.bin"
                torch.save(model_to_save.state_dict(), str(output_model_file))
                patience_stop = 0
            else:
                patience_stop += 1
            if patience_stop >= args.patience_stop:
                return

    def resume(self, args):
        resume_model_file = args.output + "/pytorch_model.bin"
        logging.info("=> loading checkpoint '{}'".format(resume_model_file))
        checkpoint = torch.load(resume_model_file, map_location='cpu')
        self.model.load_state_dict(checkpoint)

    def forward(self, batch, chosen=u'train', eval=False, answer_dict=None):

        batch = tuple(t.to(self.device) for t in batch)
        if not eval:
            input_ids, token_type_ids, segment_ids, object_labels = batch
            loss = self.model(passage_id=input_ids, token_type_id=token_type_ids, segment_id=segment_ids,
                              object_labels=object_labels)
            if self.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            loss.backward()
            loss = loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss
        else:
            p_ids, input_ids, token_type_ids, segment_ids = batch
            eval_file = self.eval_file_choice[chosen]
            po_pred = self.model(passage_id=input_ids,
                                 token_type_id=token_type_ids,
                                 segment_id=segment_ids,
                                 is_eval=eval)
            ans_dict = self.convert_spo_contour(p_ids, po_pred, eval_file,
                                                answer_dict)
            return ans_dict

    def eval_data_set(self, chosen="dev"):

        self.model.eval()

        data_loader = self.data_loader_choice[chosen]
        eval_file = self.eval_file_choice[chosen]
        answer_dict = {i: [[], [], []] for i in range(len(eval_file))}

        last_time = time.time()
        with torch.no_grad():
            for _, batch in tqdm(enumerate(data_loader), mininterval=5, leave=False, file=sys.stdout):
                self.forward(batch, chosen, eval=True, answer_dict=answer_dict)
        used_time = time.time() - last_time
        logging.info('chosen {} took : {} sec'.format(chosen, used_time))

        # self.convert2result(eval_file, answer_dict)

        res = self.evaluate(eval_file, answer_dict, chosen)
        self.model.train()
        return res

    def predict_data_set(self, chosen="dev"):

        self.model.eval()

        data_loader = self.data_loader_choice[chosen]
        eval_file = self.eval_file_choice[chosen]
        answer_dict = {i: [[], [], []] for i in range(len(eval_file))}

        last_time = time.time()
        with torch.no_grad():
            for _, batch in tqdm(enumerate(data_loader), mininterval=5, leave=False, file=sys.stdout):
                self.forward(batch, chosen, eval=True, answer_dict=answer_dict)
        used_time = time.time() - last_time
        logging.info('chosen {} took : {} sec'.format(chosen, used_time))

        # self.convert2result(eval_file, answer_dict)

        with codecs.open(self.args.res_path, 'w', 'utf-8') as f:
            for key in answer_dict.keys():

                raw_text = answer_dict[key][2]
                if raw_text == []:
                    continue
                pred = answer_dict[key][1]
                # pred = self.clean_result_with_dct(raw_text, pred)
                pred_text = []
                for (s, e, ent_name, ent_type) in pred:
                    pred_text.append('    '.join([str(s), str(e), ent_type]))
                if len(pred_text) == 0:
                    f.write(raw_text + '\n')
                else:
                    f.write(raw_text + '|||' + '|||'.join(pred_text) + '|||' + '\n')

    def clean_result(self, text, po_lst):
        """
        清洗结果
        :return:
        """

        po_lst = list(set(po_lst))
        po_lst.sort(key=lambda x: x[0])
        po_lst.sort(key=lambda x: x[1] - x[0], reverse=True)

        area_mask = [0] * len(text)
        area_type = [False] * len(text)
        new_po_list = []
        for (s, e, ent_name, ent_type) in po_lst:
            if (area_mask[s] == 1 or area_mask[e] == 1) and (not area_type[s] or not area_type[e]):
                continue
            else:
                area_mask[s:e + 1] = [1] * (e - s + 1)
                if ent_type == 'sym':
                    area_type[s:e + 1] = [True] * (e - s + 1)
                else:
                    area_type[s:e + 1] = [False] * (e - s + 1)

                new_po_list.append((s, e, ent_name, ent_type))
        new_po_list.sort(key=lambda x: x[0])
        return new_po_list

    def clean_result_with_dct(self, text, po_lst):
        """
        清洗结果 利用词典来纠正实体类型
        :return:
        """
        logging.info('清洗结果 利用词典来纠正实体类型')
        new_po_list = []
        for (s, e, ent_name, ent_type) in po_lst:
            ent_type_ = self.ent_dct.get(ent_name, None)
            if ent_type_ is not None:
                ent_type = ent_type_
            new_po_list.append((s, e, ent_name, ent_type))
        return new_po_list

    def load_ent_dict(self):
        ent_dct = {}
        logging.info('loading ent dict in {}'.format('deepIE/chip_ent/data/' + 'ent_dict.txt'))
        with open('deepIE/chip_ent/data/' + 'ent_dict.txt', 'r') as fr:
            for line in fr.readlines():
                ent_name, ent_type = line.strip().split()
                ent_dct[ent_name] = ent_type
        self.ent_dct = ent_dct

    def evaluate(self, eval_file, answer_dict, chosen):

        spo_em, spo_pred_num, spo_gold_num = 0.0, 0.0, 0.0

        for key in answer_dict.keys():
            raw_text = answer_dict[key][2]
            triple_gold = answer_dict[key][0]
            triple_pred = answer_dict[key][1]
            # triple_pred = self.clean_result_with_dct(raw_text, triple_pred)

            # if set(triple_pred) != set(triple_gold):
            #     print()
            #     print(raw_text)
            #     triple_pred.sort(key=lambda x: x[0])
            #     triple_gold.sort(key=lambda x: x[0])
            #     print(triple_pred)
            #     print(triple_gold)

            spo_em += len(set(triple_pred) & set(triple_gold))
            spo_pred_num += len(set(triple_pred))
            spo_gold_num += len(set(triple_gold))

        p = spo_em / spo_pred_num if spo_pred_num != 0 else 0
        r = spo_em / spo_gold_num if spo_gold_num != 0 else 0
        f = 2 * p * r / (p + r) if p + r != 0 else 0

        print('============================================')
        print("{}/em: {},\tpre&gold: {}\t{} ".format(chosen, spo_em, spo_pred_num, spo_gold_num))
        print("{}/f1: {}, \tPrecision: {},\tRecall: {} ".format(chosen, f * 100, p * 100,
                                                                r * 100))
        return {'f1': f, "recall": r, "precision": p}

    def convert2result(self, eval_file, answer_dict):
        for qid in answer_dict.keys():
            spoes = answer_dict[qid][2]

            context = eval_file[qid].context
            tok_to_orig_start_index = eval_file[qid].tok_to_orig_start_index
            tok_to_orig_end_index = eval_file[qid].tok_to_orig_end_index

            po_predict = []
            for s, po in spoes.items():
                po.sort(key=lambda x: x[2])
                sub_ent = context[tok_to_orig_start_index[s[0] - 1]:tok_to_orig_end_index[s[1] - 1] + 1]
                for (o1, o2, p) in po:
                    obj_ent = context[tok_to_orig_start_index[o1 - 1]:tok_to_orig_end_index[o2 - 1] + 1]
                    predicate = self.id2rel[p]

                    # TODO:到时候选择
                    # if sub_ent.replace(' ','') in context:
                    #     sub_ent = sub_ent.replace(' ', '')
                    # if obj_ent.replace(' ','') in context:
                    #     obj_ent = obj_ent.replace(' ', '')
                    po_predict.append((sub_ent, predicate, obj_ent))
            answer_dict[qid][1].extend(po_predict)

    def convert_spo_contour(self, qids, po_preds, eval_file, answer_dict):

        for qid, po_pred in zip(qids.data.cpu().numpy(), po_preds.data.cpu().numpy()):
            example = eval_file[qid.item()]

            text_id = example.text_id
            tokens = example.bert_tokens

            context = example.context
            start = np.where(po_pred[:, :, 0] > 0.5)
            end = np.where(po_pred[:, :, 1] > 0.5)

            po_index_lst = []

            for _start, predicate1 in zip(*start):
                if _start > len(tokens) - 2 or _start == 0:
                    continue
                for _end, predicate2 in zip(*end):
                    if _start <= _end <= len(tokens) - 2 and predicate1 == predicate2:
                        po_index_lst.append((_start, _end, predicate1))
                        break
            po_lst = []
            for po in po_index_lst:
                start, end, p = po
                ent_name = context[start - 1:end]
                predicate = self.id2rel[p]
                po_lst.append((start - 1, end - 1, ent_name, predicate))

            if text_id not in answer_dict:
                raise ValueError('text_id error in answer_dict ')
            else:
                if example.is_split:
                    split_index = example.span_index
                    new_ent_lst = []
                    for (start, end, ent_name, ent_type) in po_lst:
                        start += split_index * self.max_len
                        end += split_index * self.max_len
                        new_ent_lst.append((start, end, ent_name, ent_type))
                    po_lst = new_ent_lst

                answer_dict[text_id][1].extend(po_lst)
                if len(answer_dict[text_id][0]) > 1:
                    continue
                answer_dict[text_id][0] = example.g_gold_ent
                answer_dict[text_id][2] = example.g_raw_text
