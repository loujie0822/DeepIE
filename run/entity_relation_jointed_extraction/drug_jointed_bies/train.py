# _*_ coding:utf-8 _*_
import logging
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import models.ere_net.bert_mpn_2 as bert_mpn
import run.entity_relation_jointed_extraction.drug_jointed_extract.mpn_drug as mpn
from utils.optimizer_util import set_optimizer

logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, args, data_loaders, examples, word_emb, spo_conf):

        self.args = args
        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.id2rel = {item: key for key, item in spo_conf.items()}
        self.rel2id = spo_conf

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

        if args.use_bert:
            self.model = bert_mpn.ERENet.from_pretrained(args.bert_model, classes_num=len(spo_conf))
        else:
            self.model = mpn.ERENet(args, word_emb, spo_conf)
        self.model.to(self.device)
        # self.resume(args)
        logging.info('total gpu num is {}'.format(self.n_gpu))

        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model.cuda(), device_ids=[0, 1])

        train_dataloader, dev_dataloader = data_loaders
        train_eval, dev_eval = examples
        self.eval_file_choice = {
            "train": train_eval,
            "dev": dev_eval,
        }
        self.data_loader_choice = {
            "train": train_dataloader,
            "dev": dev_dataloader,
        }
        # args.use_bert = False
        self.optimizer = set_optimizer(args, self.model,
                                       train_steps=(int(len(train_eval) / args.train_batch_size) + 1) * args.epoch_num)

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
            char_ids, word_ids, token_type_ids, subject_ids, subject_labels, object_labels = batch
            loss = self.model(char_ids=char_ids, word_ids=word_ids, token_type_ids=token_type_ids,
                              subject_ids=subject_ids, subject_labels=subject_labels, object_labels=object_labels)
            if self.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            loss.backward()
            loss = loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss
        else:
            p_ids, char_ids, word_ids = batch
            eval_file = self.eval_file_choice[chosen]
            qids, subject_pred, po_pred = self.model(q_ids=p_ids,
                                                     char_ids=char_ids,
                                                     word_ids=word_ids,
                                                     eval_file=eval_file, is_eval=eval)
            ans_dict = self.convert_spo_contour(qids, subject_pred, po_pred, eval_file,
                                                answer_dict, use_bert=self.args.use_bert)
            return ans_dict

    def eval_data_set(self, chosen="dev"):

        self.model.eval()

        data_loader = self.data_loader_choice[chosen]
        eval_file = self.eval_file_choice[chosen]
        answer_dict = {i: [[], []] for i in range(len(eval_file))}

        last_time = time.time()
        with torch.no_grad():
            for _, batch in tqdm(enumerate(data_loader), mininterval=5, leave=False, file=sys.stdout):
                self.forward(batch, chosen, eval=True, answer_dict=answer_dict)
        used_time = time.time() - last_time
        logging.info('chosen {} took : {} sec'.format(chosen, used_time))
        res = self.evaluate(eval_file, answer_dict, chosen)
        self.model.train()
        return res

    def show(self, chosen="dev"):

        self.model.eval()
        answer_dict = {}

        data_loader = self.data_loader_choice[chosen]
        eval_file = self.eval_file_choice[chosen]
        with torch.no_grad():
            for _, batch in tqdm(enumerate(data_loader), mininterval=5, leave=False, file=sys.stdout):
                loss, answer_dict_ = self.forward(batch, chosen, eval=True)
                answer_dict.update(answer_dict_)
        self.badcase_analysis(eval_file, answer_dict, chosen)

    @staticmethod
    def evaluate(eval_file, answer_dict, chosen):

        entity_em = 0
        entity_pred_num = 0
        entity_gold_num = 0

        triple_em = 0
        triple_pred_num = 0
        triple_gold_num = 0
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for key, value in answer_dict.items():
            triple_gold = eval_file[key].gold_answer
            entity_gold = eval_file[key].ent_list

            entity_pred, triple_pred = value
            entity_em += len(set(entity_pred) & set(entity_gold))
            entity_pred_num += len(set(entity_pred))
            entity_gold_num += len(set(entity_gold))

            # triple_em += len(set(triple_pred) & set(triple_gold))
            # triple_pred_num += len(set(triple_pred))
            # triple_gold_num += len(set(triple_gold))

            R = set([spo for spo in triple_pred])
            T = set([spo for spo in triple_gold])

            # R = set([spo for spo in triple_pred])
            # T = set([spo for spo in triple_gold])
            # print('-'*10)
            # if R != T:
            #     print(eval_file[key].context)
            #     print(R)
            #     print(T)
            X += len(R & T)
            Y += len(R)
            Z += len(T)

        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

        entity_precision = 100.0 * entity_em / entity_pred_num if entity_pred_num > 0 else 0.
        entity_recall = 100.0 * entity_em / entity_gold_num if entity_gold_num > 0 else 0.
        entity_f1 = 2 * entity_recall * entity_precision / (entity_recall + entity_precision) if (
                                                                                                         entity_recall + entity_precision) != 0 else 0.0

        print('============================================')
        print("{}/entity_em: {},\tentity_pred_num&entity_gold_num: {}\t{} ".format(chosen, entity_em, entity_pred_num,
                                                                                   entity_gold_num))
        print(
            "{}/entity_f1: {}, \tentity_precision: {},\tentity_recall: {} ".format(chosen, entity_f1, entity_precision,
                                                                                   entity_recall))
        print('============================================')
        print("{}/em: {},\tpre&gold: {}\t{} ".format(chosen, X, Y, Z))
        print("{}/f1: {}, \tPrecision: {},\tRecall: {} ".format(chosen, f1 * 100, precision * 100,
                                                                recall * 100))
        return {'f1': f1, "recall": recall, "precision": precision}

    def convert_spo_contour(self, qids, subject_preds, po_preds, eval_file, answer_dict, use_bert=False):

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
                                   self.id2rel[p],
                                   tokens[o[0]:o[1] + 1], o[0], o[1] + 1)
                                  )
            if qid not in answer_dict:
                print('erro in answer_dict ')
            else:
                answer_dict[qid][0].append(
                    (subject[0], subject[1] + 1))
                answer_dict[qid][1].extend(po_predict)
