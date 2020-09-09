#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from deepIE.chip_ent.ent_extract_mrc.metric.mrc_ner_evaluate import nested_ner_performance

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


def train(model, optimizer, scheduler,
          train_dataloader, dev_dataloader, test_dataloader,
          config, device, n_gpu, label_list, logger):

    if config.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=config.output_dir)

    global_step = 1
    dev_best_acc = 0
    dev_best_precision = 0
    dev_best_recall = 0
    dev_best_f1 = 0
    dev_best_loss = 10000000000000
    best_dev_model_path = ""

    test_acc_when_dev_best = 0
    test_pre_when_dev_best = 0
    test_rec_when_dev_best = 0
    test_f1_when_dev_best = 0
    test_loss_when_dev_best = 1000000000000000

    tr_loss, logging_loss = 0, 0

    model.train()
    for idx in range(int(config.num_train_epochs)):
        nb_tr_examples, nb_tr_steps = 0, 0
        logger.info("#######" * 10)
        logger.info(f"EPOCH: {idx}")
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, start_pos, end_pos, span_pos, span_label_mask, ner_cate = batch

            loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                         start_positions=start_pos, end_positions=end_pos, span_positions=span_pos,
                         span_label_mask=span_label_mask, current_step=step)

            if config.n_gpu > 1:
                loss = loss.mean()

            if config.gradient_accumulation_steps > 1:
                if config.debug:
                    print("DEBUG MODE: GRAD ACCUMULATION 1 STEP . ")
                loss = loss / config.gradient_accumulation_steps

            loss.backward()
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.debug:
                    print("DEBUG MODE: BACK PROPAGATION 1 STEP .")
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if nb_tr_steps % config.checkpoint == 0:
                logger.info("-*-" * 15)
                logger.info("current training loss is : ")
                logger.info(f"{loss.item()}")
                tb_writer.add_scalar("train_loss", (tr_loss - logging_loss) / config.checkpoint, global_step)
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                logging_loss = tr_loss

                if config.only_train:
                    model_to_save = model.module if hasattr(model, "module") else model
                    output_model_file = os.path.join(config.output_dir,
                                                     "bert_finetune_model_{}_{}.bin".format(str(idx), str(nb_tr_steps)))
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("SAVED model path is :")
                    logger.info(output_model_file)
                else:
                    model, tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1 = eval_checkpoint(model,
                                                                                                              dev_dataloader,
                                                                                                              config,
                                                                                                              device,
                                                                                                              n_gpu,
                                                                                                              label_list,
                                                                                                              eval_sign="dev")
                    logger.info("......" * 10)
                    logger.info("DEV: loss, acc, precision, recall, f1")
                    logger.info(f"{tmp_dev_loss}, {tmp_dev_acc}, {tmp_dev_prec}, {tmp_dev_rec}, {tmp_dev_f1}")
                    tb_writer.add_scalar("dev_loss", tmp_dev_loss, global_step)
                    tb_writer.add_scalar("dev_f1", tmp_dev_f1, global_step)
                    tb_writer.add_scalar("dev_acc", tmp_dev_acc, global_step)

                    if tmp_dev_f1 > dev_best_f1:
                        dev_best_acc = tmp_dev_acc
                        dev_best_loss = tmp_dev_loss
                        dev_best_precision = tmp_dev_prec
                        dev_best_recall = tmp_dev_rec
                        dev_best_f1 = tmp_dev_f1

                        # export model
                        if config.export_model:
                            model_to_save = model.module if hasattr(model, "module") else model
                            output_model_file = os.path.join(config.output_dir,
                                                             "bert_finetune_model_{}_{}.bin".format(str(idx),
                                                                                                    str(nb_tr_steps)))
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("SAVED model path is :")
                            logger.info(output_model_file)
                            best_dev_model_path = output_model_file

                        model = model.cuda().to(device)

                        if not config.only_eval_dev:
                            model, tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1 = eval_checkpoint(
                                model, test_dataloader, config, device, n_gpu, label_list, eval_sign="test")
                            logger.info("......" * 10)
                            logger.info("TEST: loss, acc, precision, recall, f1")
                            logger.info(
                                f"{tmp_test_loss}, {tmp_test_acc}, {tmp_test_prec}, {tmp_test_rec}, {tmp_test_f1}")

                            test_acc_when_dev_best = tmp_test_acc
                            test_pre_when_dev_best = tmp_test_prec
                            test_rec_when_dev_best = tmp_test_rec
                            test_f1_when_dev_best = tmp_test_f1
                            test_loss_when_dev_best = tmp_test_loss
                            model = model.cuda().to(device)

                logger.info("-*-" * 15)

    if config.local_rank in [-1, 0]:
        tb_writer.close()

    logger.info("=&=" * 15)
    logger.info("Best DEV : overall best loss, acc, precision, recall, f1 ")
    logger.info(f"{dev_best_loss}, {dev_best_acc}, {dev_best_precision}, {dev_best_recall}, {dev_best_f1}")
    if not config.only_eval_dev:
        logger.info("scores on TEST when Best DEV:loss, acc, precision, recall, f1 ")
        logger.info(
            f"{test_loss_when_dev_best}, {test_acc_when_dev_best}, {test_pre_when_dev_best}, {test_rec_when_dev_best}, {test_f1_when_dev_best}")
    else:
        logger.info("Please Evaluate the saved CKPT on TEST SET using the Best dev model. ")
        logger.info(f"Best Dev Model is saved : {best_dev_model_path}")
        logger.info("Please run [evaluate_mrc_ner.py] to get the test performance. ")
    logger.info("=&=" * 15)


def eval_checkpoint(model_object, eval_dataloader, config, device, n_gpu, label_list, eval_sign="dev"):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader

    eval_loss = 0
    start_pred_lst = []
    end_pred_lst = []
    span_pred_lst = []
    start_scores_lst = []
    end_scores_lst = []
    mask_lst = []
    start_gold_lst = []
    span_gold_lst = []
    end_gold_lst = []
    eval_steps = 0
    ner_cate_lst = []

    for eval_idx, eval_batch in enumerate(eval_dataloader):
        input_ids, input_mask, segment_ids, start_pos, end_pos, span_pos, span_label_mask, ner_cate = eval_batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        start_pos = start_pos.to(device)
        end_pos = end_pos.to(device)
        span_pos = span_pos.to(device)
        span_label_mask = span_label_mask.to(device)

        with torch.no_grad():
            model_object.eval()
            tmp_eval_loss = model_object(input_ids, segment_ids, input_mask, start_pos, end_pos, span_pos,
                                         span_label_mask)
            start_labels, end_labels, span_scores = model_object(input_ids, segment_ids, input_mask)

        start_pos = start_pos.to("cpu").numpy().tolist()
        end_pos = end_pos.to("cpu").numpy().tolist()
        span_pos = span_pos.to("cpu").numpy().tolist()

        start_label = start_labels.detach().cpu().numpy().tolist()
        end_label = end_labels.detach().cpu().numpy().tolist()
        span_scores = span_scores.detach().cpu().numpy().tolist()
        span_label = span_scores
        input_mask = input_mask.to("cpu").detach().numpy().tolist()

        ner_cate_lst += ner_cate.numpy().tolist()
        eval_loss += tmp_eval_loss.mean().item()
        mask_lst += input_mask
        eval_steps += 1

        start_pred_lst += start_label
        end_pred_lst += end_label
        span_pred_lst += span_label

        start_gold_lst += start_pos
        end_gold_lst += end_pos
        span_gold_lst += span_pos

        eval_accuracy, eval_precision, eval_recall, eval_f1 = nested_ner_performance(start_pred_lst, end_pred_lst,
                                                                                     span_pred_lst, start_gold_lst,
                                                                                     end_gold_lst, span_gold_lst,
                                                                                     ner_cate_lst, label_list,
                                                                                     threshold=config.entity_threshold,
                                                                                     dims=2)

    average_loss = round(eval_loss / eval_steps, 4)
    eval_f1 = round(eval_f1, 4)
    eval_precision = round(eval_precision, 4)
    eval_recall = round(eval_recall, 4)
    eval_accuracy = round(eval_accuracy, 4)
    model_object.train()

    return model_object, average_loss, eval_accuracy, eval_precision, eval_recall, eval_f1

