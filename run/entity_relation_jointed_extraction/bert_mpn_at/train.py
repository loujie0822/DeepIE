# _*_ coding:utf-8 _*_
import logging
import sys
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import models.ere_net.bert_mpn as bert_mpn
from utils.data_util import Tokenizer
from utils.optimizer_util import set_optimizer
from utils.train_util import FGM

logger = logging.getLogger(__name__)

tokenizer = Tokenizer('cpt/bert-base-chinese/vocab.txt', do_lower_case=True)


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """

    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0].replace(' ',''))),
            spo[1],
            tuple(tokenizer.tokenize(spo[2].replace(' ',''))),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


class Trainer(object):

    def __init__(self, args, data_loaders, examples, char_emb, spo_conf):
        print('using ad')
        self.args = args
        self.tokenizer = Tokenizer(args.bert_model + '/vocab.txt', do_lower_case=True)
        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.id2rel = {item: key for key, item in spo_conf.items()}
        self.rel2id = spo_conf

        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
        if args.use_bert:
            self.model = bert_mpn.ERENet.from_pretrained(args.bert_model, classes_num=len(spo_conf))
        else:
            self.model = mpn.ERENet(args, char_emb, spo_conf)

        self.model.to(self.device)
        self.resume(args)
        logging.info('total gpu num is {}'.format(self.n_gpu))
        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model.cuda(), device_ids=[0, 1])

        self.adversarial_train = FGM(self.model)

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

                if step % 500 == 0 and epoch >= 6:
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
            input_ids, segment_ids, token_type_ids, subject_ids, subject_labels, object_labels = batch
            loss = self.model(passage_ids=input_ids, segment_ids=segment_ids, token_type_ids=token_type_ids,
                              subject_ids=subject_ids, subject_labels=subject_labels, object_labels=object_labels)
            if self.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            loss.backward()
            self.adversarial_train.attack()
            loss_adv =self.model(passage_ids=input_ids, segment_ids=segment_ids, token_type_ids=token_type_ids,
                              subject_ids=subject_ids, subject_labels=subject_labels, object_labels=object_labels)
            if self.n_gpu > 1:
                loss_adv = loss_adv.mean()
            loss_adv.backward()
            self.adversarial_train.restore()

            loss = loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss
        else:
            p_ids, input_ids, segment_ids = batch
            eval_file = self.eval_file_choice[chosen]
            qids, subject_pred, po_pred = self.model(q_ids=p_ids,
                                                     passage_ids=input_ids,
                                                     segment_ids=segment_ids,
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
    def evaluate_(eval_file, answer_dict, chosen):

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

            if set(triple_pred) != set(triple_gold):
                print(set(triple_pred))
                print(set(triple_gold))
                print('-' * 10)
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
            entity_gold = eval_file[key].sub_entity_list

            entity_pred, triple_pred = value

            entity_em += len(set(entity_pred) & set(entity_gold))
            entity_pred_num += len(set(entity_pred))
            entity_gold_num += len(set(entity_gold))

            # triple_em += len(set(triple_pred) & set(triple_gold))
            # triple_pred_num += len(set(triple_pred))
            # triple_gold_num += len(set(triple_gold))

            R = set([SPO(spo) for spo in triple_pred])
            T = set([SPO(spo) for spo in triple_gold])
            # if R != T:
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
            tokens = eval_file[qid.item()].bert_tokens
            token_ids = eval_file[qid.item()].token_ids
            start = np.where(po_pred[:, :, 0] > 0.6)
            end = np.where(po_pred[:, :, 1] > 0.5)

            spoes = []
            for _start, predicate1 in zip(*start):
                if _start > len(tokens) - 2 or _start == 0:
                    continue
                for _end, predicate2 in zip(*end):
                    if _start <= _end <= len(tokens) - 2 and predicate1 == predicate2:
                        spoes.append((subject, predicate1, (_start, _end)))
                        break
            po_predict = []
            for s, p, o in spoes:
                po_predict.append((self.tokenizer.decode(token_ids[s[0]:s[1] + 1], tokens[s[0]:s[1] + 1]),
                                   self.id2rel[p],
                                   self.tokenizer.decode(token_ids[o[0]:o[1] + 1], tokens[o[0]:o[1] + 1]))
                                  )

            if qid not in answer_dict:
                print('erro in answer_dict ')
            else:
                answer_dict[qid][0].append(
                    self.tokenizer.decode(token_ids[subject[0]:subject[1] + 1], tokens[subject[0]:subject[1] + 1]))
                answer_dict[qid][1].extend(po_predict)
