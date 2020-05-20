# _*_ coding:utf-8 _*_
import logging
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import models.attribute_net.bert_mpn as bert_mpn
import run.attribute_extraction.drug_mpn.drug_mpn as mpn
from utils.optimizer_util import set_optimizer

logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, args, data_loaders, examples, char_emb, attribute_conf):
        if args.use_bert:
            self.model = bert_mpn.AttributeExtractNet.from_pretrained(args.bert_model, args, attribute_conf)
        else:
            self.model = mpn.AttributeExtractNet(args, char_emb, attribute_conf)

        self.args = args

        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.id2rel = {item: key for key, item in attribute_conf.items()}
        self.rel2id = attribute_conf

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

        p_ids, passage_id, token_type_id, segment_id, pos_start, pos_end, start_id, end_id = batch
        loss, po1, po2 = self.model(passage_id=passage_id, token_type_id=token_type_id, segment_id=segment_id,
                                    pos_start=pos_start, pos_end=pos_end, start_id=start_id, end_id=end_id,
                                    is_eval=eval)

        if self.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.

        if grad:
            loss.backward()
            loss = loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()

        if eval:
            eval_file = self.eval_file_choice[chosen]
            answer_dict_ = convert_pointer_net_contour(eval_file, p_ids, po1, po2, self.id2rel,
                                                       use_bert=self.args.use_bert)
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
        self.detail_evaluate(eval_file, answer_dict, chosen)
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
        self.badcase_analysis(eval_file, answer_dict, chosen)

    @staticmethod
    def evaluate(eval_file, answer_dict, chosen):

        em = 0
        pre = 0
        gold = 0
        for key, value in answer_dict.items():
            ground_truths = eval_file[int(key)].gold_answer
            value, l1, l2 = value
            prediction = value if len(value) else []
            assert type(prediction) == type(ground_truths)
            pre += len(set(prediction))
            gold += len(set(ground_truths))
            intersection = set(prediction) & set(ground_truths)

            em += len(intersection)

        precision = 100.0 * em / pre if pre > 0 else 0.
        recall = 100.0 * em / gold if gold > 0 else 0.
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) != 0 else 0.0
        print('============================================')
        print("{}/em: {},\tpre&gold: {}\t{} ".format(chosen, em, pre, gold))
        print("{}/f1: {}, \tPrecision: {},\tRecall: {} ".format(chosen, f1, precision,
                                                                recall))
        return {'f1': f1, "recall": recall, "precision": precision, 'em': em, 'pre': pre, 'gold': gold}

    def detail_evaluate(self, eval_file, answer_dict, chosen):
        def generate_detail_dict(spo_list):
            dict_detail = dict()
            for i, tag in enumerate(spo_list):
                detail_name = tag.split('@')[0]
                if detail_name not in dict_detail:
                    dict_detail[detail_name] = [tag]
                else:
                    dict_detail[detail_name].append(tag)
            return dict_detail

        total_detail = {}
        for key, value in answer_dict.items():
            ground_truths = eval_file[int(key)].gold_answer
            value, l1, l2 = value
            prediction = value if len(value) else []

            gold_detail = generate_detail_dict(ground_truths)
            pred_detail = generate_detail_dict(prediction)
            for key in self.rel2id.keys():

                pred = pred_detail.get(key, [])
                gold = gold_detail.get(key, [])
                pred_num = len(set(pred))
                gold_num = len(set(gold))
                em = len(set(pred) & set(gold))

                if key not in total_detail:
                    total_detail[key] = dict()
                    total_detail[key]['em'] = em
                    total_detail[key]['pred_num'] = pred_num
                    total_detail[key]['gold_num'] = gold_num
                else:
                    total_detail[key]['em'] += em
                    total_detail[key]['pred_num'] += pred_num
                    total_detail[key]['gold_num'] += gold_num
        for key, res_dict_ in total_detail.items():
            res_dict_['p'] = 100.0 * res_dict_['em'] / res_dict_['pred_num'] if res_dict_['pred_num'] != 0 else 0.0
            res_dict_['r'] = 100.0 * res_dict_['em'] / res_dict_['gold_num'] if res_dict_['gold_num'] != 0 else 0.0
            res_dict_['f'] = 2 * res_dict_['p'] * res_dict_['r'] / (res_dict_['p'] + res_dict_['r']) if res_dict_['p'] + \
                                                                                                        res_dict_[
                                                                                                            'r'] != 0 else 0.0

        for gold_key, res_dict_ in total_detail.items():
            print('===============================================================')
            print("{}/em: {},\tpred_num&gold_num: {}\t{} ".format(gold_key, res_dict_['em'], res_dict_['pred_num'],
                                                                  res_dict_['gold_num']))
            print(
                "{}/f1: {},\tprecison&recall: {}\t{}".format(gold_key, res_dict_['f'], res_dict_['p'], res_dict_['r']))

    @staticmethod
    def badcase_analysis(eval_file, answer_dict, chosen):
        em = 0
        pre = 0
        gold = 0
        content = ''
        for key, value in answer_dict.items():
            entity_name = eval_file[int(key)].entity_name
            context = eval_file[int(key)].context
            ground_truths = eval_file[int(key)].gold_answer
            value, l1, l2 = value
            prediction = list(value) if len(value) else ['']
            assert type(prediction) == type(ground_truths)

            intersection = set(prediction) & set(ground_truths)

            if prediction == ground_truths == ['']:
                continue
            if set(prediction) != set(ground_truths):
                ground_truths = list(sorted(set(ground_truths)))
                prediction = list(sorted(set(prediction)))
                print('raw context is:\t' + context)
                print('subject_name is:\t' + entity_name)
                print('pred_text is:\t' + '\t'.join(prediction))
                print('gold_text is:\t' + '\t'.join(ground_truths))
                content += 'raw context is:\t' + context + '\n'
                content += 'subject_name is:\t' + entity_name + '\n'
                content += 'pred_text is:\t' + '\t'.join(prediction) + '\n'
                content += 'gold_text is:\t' + '\t'.join(ground_truths) + '\n'
                content += '==============================='
            em += len(intersection)
            pre += len(set(prediction))
            gold += len(set(ground_truths))
        with open('badcase_{}.txt'.format(chosen), 'w') as f:
            f.write(content)


def convert_pointer_net_contour(eval_file, q_ids, po1, po2, id2rel, use_bert=False):
    answer_dict = dict()
    for qid, o1, o2 in zip(q_ids, po1.data.cpu().numpy(), po2.data.cpu().numpy()):

        context = eval_file[qid.item()].context if not use_bert else eval_file[qid.item()].bert_tokens
        # gold_attr_list = eval_file[qid.item()].gold_attr_list
        # gold_answer = [attr.attr_type + '@' + attr.value for attr in gold_attr_list]
        entity_name = eval_file[qid.item()].entity_name

        entity_position = eval_file[qid.item()].entity_position

        drug_name = entity_name + '_' + str(entity_position[0])
        answers = list()
        start, end = np.where(o1 > 0.5), np.where(o2 > 0.5)
        for _start, _attr_type_id_start in zip(*start):
            if _start > len(context) or (_start == 0 and use_bert):
                continue
            for _end, _attr_type_id_end in zip(*end):
                if _start <= _end < len(context) and _attr_type_id_start == _attr_type_id_end:
                    _attr_value = ''.join(context[_start: _end + 1]) if use_bert else context[_start: _end + 1]
                    _attr_type = id2rel[_attr_type_id_start]
                    _attr = _attr_type + '@' + _attr_value + str(_start) + str(_end + 1)
                    # _attr = (drug_name, _attr_type, _attr_value, _start, _end + 1)
                    answers.append(_attr)
                    break

        answer_dict[str(qid.item())] = [answers, o1, o2]

    return answer_dict
