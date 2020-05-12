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
from utils.metrics import SpanFPreRecMetric
from utils.optimizer_util import set_optimizer

logger = logging.getLogger(__name__)


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


class Trainer(object):

    def __init__(self, args, data_loaders, examples, model_conf):


        self.model = ner.NERNet(args, model_conf)

        self.args = args
        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.ent2id = model_conf['entity_type']
        self.id2ent = {item: key for key, item in self.ent2id.items()}
        self.metric = SpanFPreRecMetric(tag_type=self.id2ent)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
        self.model.to(self.device)
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
        step_gap = int(int(len(self.eval_file_choice['train']) / args.train_batch_size) / 20)
        for epoch in range(int(args.epoch_num)):

            # self.optimizer = lr_decay(self.optimizer, epoch, 0.05, args.learning_rate)

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
            if res_dev['f'] >= best_f1:
                best_f1 = res_dev['f']
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

    def forward(self, batch, chosen=u'train', eval=False, detail=False):

        batch = tuple(t.to(self.device) for t in batch)

        p_ids, char_id, bichar_id, label_id = batch
        if not eval:
            loss = self.model(char_id=char_id, bichar_id=bichar_id, label_id=label_id, is_eval=eval)
            if self.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.\

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            loss = loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss
        else:
            pred = self.model(char_id=char_id, bichar_id=bichar_id, label_id=label_id, is_eval=eval)
            eval_file = self.eval_file_choice[chosen]
            answer_dict = self.metric(p_ids, pred, eval_file)
            return answer_dict

    def eval_data_set(self, chosen="dev"):

        self.model.eval()
        answer_dict = {}

        data_loader = self.data_loader_choice[chosen]
        eval_file = self.eval_file_choice[chosen]
        last_time = time.time()
        with torch.no_grad():
            for _, batch in tqdm(enumerate(data_loader), mininterval=5, leave=False, file=sys.stdout):
                answer_dict_ = self.forward(batch, chosen, eval=True)
                answer_dict.update(answer_dict_)
        used_time = time.time() - last_time
        logging.info('chosen {} took : {} sec'.format(chosen, used_time))
        # res = self.evaluate(eval_file, answer_dict, chosen)
        # self.detail_evaluate(eval_file, answer_dict, chosen)
        self.model.train()
        return self.metric.get_metric()

    def show(self, chosen="dev"):

        self.model.eval()
        answer_dict = {}

        data_loader = self.data_loader_choice[chosen]
        eval_file = self.eval_file_choice[chosen]
        with torch.no_grad():
            for _, batch in tqdm(enumerate(data_loader), mininterval=5, leave=False, file=sys.stdout):
                loss, answer_dict_ = self.forward(batch, chosen, eval=True, detail=True)
                answer_dict.update(answer_dict_)

    @staticmethod
    def evaluate(eval_file, answer_dict, chosen):

        em, pre_num, gold_num = 0, 0, 0
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
