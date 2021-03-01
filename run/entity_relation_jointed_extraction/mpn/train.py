# _*_ coding:utf-8 _*_
import logging
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import models.ere_net.bert_mpn_old as bert_mpn
import models.ere_net.mpn as mpn
from utils.optimizer_util import set_optimizer

logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, args, data_loaders, examples, char_emb, spo_conf):
        if args.use_bert:
            self.model = bert_mpn.ERENet(args, spo_conf)
        else:
            self.model = mpn.ERENet(args, char_emb, spo_conf)

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

            p_ids, input_ids, segment_ids, token_type_ids, s1, s2, po1, po2 = batch
            loss = self.model(passages=input_ids, token_type_ids=token_type_ids, segment_ids=segment_ids, s1=s1, s2=s2,
                              po1=po1, po2=po2,
                              is_eval=eval)
            if self.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            loss.backward()
            loss = loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss
        else:
            p_ids, input_ids, segment_ids = batch
            eval_file = self.eval_file_choice[chosen]
            qid_tensor, po1_tensor, po2_tensor, s_tensor, e_tensor = self.model(q_ids=p_ids, eval_file=eval_file,
                                                                                passages=input_ids, is_eval=eval)
            ans_dict = self.convert_spo_contour(qid_tensor, po1_tensor, po2_tensor, s_tensor, e_tensor, eval_file,
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
        for key, value in answer_dict.items():
            triple_gold = eval_file[key].gold_answer
            entity_gold = eval_file[key].sub_entity_list

            entity_pred, triple_pred = value

            entity_em += len(set(entity_pred) & set(entity_gold))
            entity_pred_num += len(set(entity_pred))
            entity_gold_num += len(set(entity_gold))

            triple_em += len(set(triple_pred) & set(triple_gold))
            triple_pred_num += len(set(triple_pred))
            triple_gold_num += len(set(triple_gold))

        entity_precision = 100.0 * entity_em / entity_pred_num if entity_pred_num > 0 else 0.
        entity_recall = 100.0 * entity_em / entity_gold_num if entity_gold_num > 0 else 0.
        entity_f1 = 2 * entity_recall * entity_precision / (entity_recall + entity_precision) if (
                                                                                                         entity_recall + entity_precision) != 0 else 0.0

        precision = 100.0 * triple_em / triple_pred_num if triple_pred_num > 0 else 0.
        recall = 100.0 * triple_em / triple_gold_num if triple_gold_num > 0 else 0.
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) != 0 else 0.0
        print('============================================')
        print("{}/entity_em: {},\tentity_pred_num&entity_gold_num: {}\t{} ".format(chosen, entity_em, entity_pred_num,
                                                                                   entity_gold_num))
        print(
            "{}/entity_f1: {}, \tentity_precision: {},\tentity_recall: {} ".format(chosen, entity_f1, entity_precision,
                                                                                   entity_recall))
        print('============================================')
        print("{}/em: {},\tpre&gold: {}\t{} ".format(chosen, triple_em, triple_pred_num, triple_gold_num))
        print("{}/f1: {}, \tPrecision: {},\tRecall: {} ".format(chosen, f1, precision,
                                                                recall))
        return {'f1': f1, "recall": recall, "precision": precision}

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

    def convert_spo_contour(self, qid_tensor, po1, po2, s_tensor, e_tensor, eval_file, answer_dict, use_bert=False):
        for qid, s, e, o1, o2 in zip(qid_tensor.data.cpu().numpy(), s_tensor.data.cpu().numpy(),
                                     e_tensor.data.cpu().numpy(), po1.data.cpu().numpy(), po2.data.cpu().numpy()):
            if qid == -1:
                continue
            context = eval_file[qid.item()].context if not use_bert else eval_file[qid.item()].bert_tokens
            gold_answer = eval_file[qid].gold_answer

            _subject = ''.join(context[s:e]) if use_bert else context[s:e]
            answers = list()
            start, end = np.where(o1 > 0.5), np.where(o2 > 0.5)
            for _start, _predict_id_start in zip(*start):
                if _start > len(context) or (_start == 0 and use_bert):
                    continue
                for _end, _predict_id_end in zip(*end):
                    if _start <= _end < len(context) and _predict_id_start == _predict_id_end:
                        _obeject = ''.join(context[_start: _end + 1]) if use_bert else context[_start: _end + 1]
                        _predicate = self.id2rel[_predict_id_start]
                        answers.append((_subject, _predicate, _obeject))
                        break

            if qid not in answer_dict:
                print('erro in answer_dict ')
            else:
                answer_dict[qid][0].append((_subject, s, e))
                answer_dict[qid][1].extend(answers)
