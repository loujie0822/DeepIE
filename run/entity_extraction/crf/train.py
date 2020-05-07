# _*_ coding:utf-8 _*_
import logging
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import models.ner_net.lstm_crf as ner
from utils.optimizer_util import set_optimizer

logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, args, data_loaders, examples, char_emb, ent_conf):
        if args.use_bert:
            self.model = None
        else:
            self.model = ner.NERNet(args, char_emb,ent_conf)

        self.args = args
        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.id2ent = {item: key for key, item in ent_conf.items()}
        self.ent2id = ent_conf

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
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

                loss, answer_dict_ = self.forward(batch)

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

    def forward(self, batch, chosen=u'train', grad=True, eval=False, detail=False):

        batch = tuple(t.to(self.device) for t in batch)

        p_ids, passage_id, label_id = batch
        loss, pred, gold = self.model(passages=passage_id, label_ids=label_id, is_eval=eval)

        if self.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.

        if grad:
            loss.backward()
            loss = loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()

        if eval:
            eval_file = self.eval_file_choice[chosen]
            answer_dict_ = convert_stl_contour(self.id2ent, eval_file, p_ids, pred, gold, detail=detail)
        else:
            answer_dict_ = None
        return loss, answer_dict_

    def eval_data_set(self, chosen="dev"):

        self.model.eval()
        answer_dict = {}

        data_loader = self.data_loader_choice[chosen]
        eval_file = self.eval_file_choice[chosen]
        last_time = time.time()
        with torch.no_grad():
            for _, batch in tqdm(enumerate(data_loader), mininterval=5, leave=False, file=sys.stdout):
                loss, answer_dict_ = self.forward(batch, chosen, grad=False, eval=True)
                answer_dict.update(answer_dict_)
        used_time = time.time() - last_time
        logging.info('chosen {} took : {} sec'.format(chosen, used_time))
        res = self.evaluate(eval_file, answer_dict, chosen)
        # self.detail_evaluate(eval_file, answer_dict, chosen)
        self.model.train()
        return res

    def show(self, chosen="dev"):

        self.model.eval()
        answer_dict = {}

        data_loader = self.data_loader_choice[chosen]
        eval_file = self.eval_file_choice[chosen]
        with torch.no_grad():
            for _, batch in tqdm(enumerate(data_loader), mininterval=5, leave=False, file=sys.stdout):
                loss, answer_dict_ = self.forward(batch, chosen, grad=False, eval=True, detail=True)
                answer_dict.update(answer_dict_)

    @staticmethod
    def evaluate(eval_file, answer_dict, chosen):

        em,pre_num,gold_num = 0,0,0
        for key, value in answer_dict.items():
            ground_truths = eval_file[int(key)].gold_answer
            value, pred, gold, _, _ = value
            em += len(set(pred) & set(gold))
            pre_num += len(set(pred))
            gold_num += len(set(gold))
        precision = 100.0 * em / pre_num if pre_num > 0 else 0.
        recall = 100.0 * em / gold_num if gold_num > 0 else 0.
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) != 0 else 0.0
        print('============================================')
        print("{}/em: {},\tpre&gold: {}\t{} ".format(chosen, em, pre_num, gold_num))
        print("{}/f1: {}, \tPrecision: {},\tRecall: {} ".format(chosen, f1, precision,
                                                                recall))
        return {'f1': f1, "recall": recall, "precision": precision, 'em': em, 'pre': pre_num, 'gold': gold_num}


def convert_stl_contour(id2rel, eval_file, q_ids, predict, gold, detail):
    answer_dict = dict()

    for qid, pred, gold in zip(q_ids, predict, gold.data.cpu().numpy()):
        context = eval_file[qid.item()].context
        seq_len = len(context)

        pred_entities = find_tag_position(pred, seq_len, id2rel)
        gold_entities = find_tag_position(gold, seq_len, id2rel)

        pred_answer_ = generat_ans(id2rel, pred, pred_entities, seq_len)

        if detail:
            pred_detail = generat_detail_dict(id2rel, pred, pred_entities)
            gold_detail = generat_detail_dict(id2rel, gold, gold_entities)
            answer_dict[str(qid.item())] = [pred_answer_, pred_entities, gold_entities, pred_detail, gold_detail]

        else:
            answer_dict[str(qid.item())] = [pred_answer_, pred_entities, gold_entities, None, None]

    return answer_dict


def generat_detail_dict(tag_ids, tag_list, tag_entity_pos):
    dict_detail = dict()
    for i, tag in enumerate(tag_entity_pos):
        start, end = tag[0], tag[1]
        detail_name = tag_ids[tag_list[start]].split('-')[1]
        if detail_name not in dict_detail:
            dict_detail[detail_name] = [(start, end)]
        else:
            dict_detail[detail_name].append((start, end))
    return dict_detail


def generat_ans(tag_ids, tag_list, tag_entity_pos, seq_len):
    ans_list = ['O'] * seq_len
    for i, tag in enumerate(tag_entity_pos):
        start, end = tag[0], tag[1]
        ans_list[start] = tag_ids[tag_list[start]]
        for k in range(start + 1, end):
            ans_list[k] = tag_ids[tag_list[k]]
    return ans_list


def find_raw_context(ans_list, context):
    new_ans = list()
    for i, ans in enumerate(ans_list):
        if ans != 'O':
            new_ans.append(context[i])
        else:
            new_ans.append('*')
    return ''.join(new_ans)


def find_tag_position(find_list, seq_len, id2rel):
    tag_list = list()

    j = 0
    while j < seq_len:
        end = j
        flag = True

        if find_list[j] % 2 == 0 and find_list[j] != 0:
            start = j
            tag = id2rel[find_list[start]].split('-')[1]
            for k in range(start + 1, seq_len):
                if find_list[k] != find_list[start] + 1:
                    end = k - 1
                    flag = False
                    break
            if flag:
                end = seq_len - 1
            tag_list.append((start, end + 1, tag))
        j = end + 1
    return tag_list
