#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# run_machine_comprehension.py
# Please Notice that the data should contain
# multi answers
# need pay MORE attention when loading data


import os
import argparse
import numpy as np
import random
import torch


from deepIE.chip_ent.ent_extract_mrc.utils import BertTokenizer4Tagger
from deepIE.chip_ent.ent_extract_mrc.get_logger import logger_to_file
from deepIE.chip_ent.ent_extract_mrc.utils import Config
from deepIE.chip_ent.ent_extract_mrc.model import BertQueryNER
from deepIE.chip_ent.ent_extract_mrc.optim.optimizers import build_fp32_optimizer
from deepIE.chip_ent.ent_extract_mrc.optim.lr_scheduler import build_lr_scheduler
from deepIE.chip_ent.ent_extract_mrc.data_loader import CHIP2020Processor, MRCNERDataLoader
from deepIE.chip_ent.ent_extract_mrc.train import train


def args_parser():
    # start parser
    parser = argparse.ArgumentParser()

    # requires parameters
    parser.add_argument("--config_path", default="./config/zh_bert.json", type=str)
    parser.add_argument("--data_dir", default="./data_preprocess/chip2020/chip_2020_1_train_debug/", type=str)
    parser.add_argument("--bert_model", default="./pretrain_models/bert-base-chinese/", type=str)
    parser.add_argument("--checkpoint", default=100, type=int)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--logfile_name", type=str, default="log.txt")

    parser.add_argument("--do_lower_case", default=False, action='store_true', help="lower case of input tokens.")
    parser.add_argument("--entity_sign", type=str, default="flat")
    parser.add_argument("--num_data_processor", default=1, type=int, help="number of data processor.")
    parser.add_argument("--data_cache", default=True, action='store_false')
    parser.add_argument("--n_gpu", type=int, default=1)

    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--dev_batch_size", default=8, type=int)
    parser.add_argument("--test_batch_size", default=8, type=int)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--optimizer_type", default="adamw", type=str, )
    parser.add_argument("--lr_scheduler_type", default="polynomial_warmup", type=str)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--lr_min", default=5e-6, type=float, help="minimal learning rate for lr scheduler.")
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--warmup_steps", default=1000, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", type=float, default=1)
    parser.add_argument("--data_parallel", default="dp", type=str,
                        help="dp -> data parallel for multi GPU cards; ddp -> distributed data parallel for multiple GPU card.")

    parser.add_argument("--weight_start", type=float, default=1.0)
    parser.add_argument("--weight_end", type=float, default=1.0)
    parser.add_argument("--weight_span", type=float, default=1.0)
    parser.add_argument("--entity_threshold", type=float, default=0.5)
    parser.add_argument("--loss_type", default="ce", type=str, help="ce, wce, dynamic_wce, dice, focal, dsc. ")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--entity_scheme", type=str, default="bes", help="bes -> begin+end+span; BMES -> ")

    parser.add_argument("--seed", type=int, default=3006)
    parser.add_argument("--only_eval_dev", default=False, action="store_true", help="only evaluate on dev set. ")
    parser.add_argument("--debug", default=False, action="store_true",
                        help="replace traindataloader with traindataloader. ")
    parser.add_argument("--only_train", default=False, action="store_true",
                        help="only train and save checkpoints. evaluation should be conducted after training. ")
    parser.add_argument("--export_model", default=True, action='store_false')

    args = parser.parse_args()

    args.train_batch_size = int(args.train_batch_size // args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    return args


def load_data(config, logger):
    logger.info("-*-" * 10)

    data_processor = CHIP2020Processor()
    label_list = data_processor.get_labels()

    tokenizer = BertTokenizer4Tagger.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)

    dataset_loaders = MRCNERDataLoader(config, data_processor, label_list,
                                       tokenizer, mode="train", allow_impossible=True)  # entity_scheme=config.entity_scheme)
    data_sign = 'train'
    if config.debug:
        logger.info("%=" * 20)
        logger.info("=" * 10 + " DEBUG MODE " + "=" * 10)
        data_sign = 'dev'
    train_dataloader = dataset_loaders.get_dataloader(data_sign=data_sign,
                                                      num_data_processor=config.num_data_processor,
                                                      logger=logger)
    dev_dataloader = dataset_loaders.get_dataloader(data_sign="dev",
                                                    num_data_processor=config.num_data_processor,
                                                    logger=logger)
    test_dataloader = dataset_loaders.get_dataloader(data_sign="test",
                                                     num_data_processor=config.num_data_processor,
                                                     logger=logger)

    train_instances = dataset_loaders.get_train_instance()
    num_train_steps = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs
    per_gpu_train_batch_size = config.train_batch_size // config.n_gpu

    logger.info("****** Running Training ******")
    logger.info(f"Number of Training Data: {train_instances}")
    logger.info(f"Train Epoch {config.num_train_epochs}; Total Train Steps: {num_train_steps}; Warmup Train Steps: {config.warmup_steps}")
    logger.info(f"Per GPU Train Batch Size: {per_gpu_train_batch_size}")

    return train_dataloader, dev_dataloader, test_dataloader, num_train_steps, label_list


def load_model(config, num_train_steps, label_list, logger):
    model = BertQueryNER(config, train_steps=num_train_steps)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        if config.n_gpu > 1:
            model = torch.nn.DataParallel(model)
    else:
        device = torch.device('cpu')
        model.to(device)

    # prepare optimzier
    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": config.weight_decay},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = build_fp32_optimizer(config, optimizer_grouped_parameters, )
    scheduler = build_lr_scheduler(config, optimizer, num_train_steps)

    # Distributed training (should be after apex fp16 initialization)
    if config.local_rank != -1 and config.data_parallel == "ddp":
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=True)

    return model, optimizer, scheduler, device, config.n_gpu


def merge_config(args_config, logger=None):
    # 将bert config与args config融合
    model_config_path = args_config.config_path
    model_config = Config.from_json_file(model_config_path)
    model_config.update_args(args_config)
    model_config.print_config(logger=logger)
    return model_config


def main():
    args_config = args_parser()
    path_to_logfile = os.path.join(args_config.output_dir, args_config.logfile_name)
    logger = logger_to_file(path_to_logfile)
    config = merge_config(args_config, logger=logger)

    print('load data...')
    train_loader, dev_loader, test_loader, num_train_steps, label_list = load_data(config, logger)

    print('load model ...')
    model, optimizer, scheduler, device, n_gpu = load_model(config, num_train_steps, label_list, logger)

    print('train start ...')
    train(model, optimizer, scheduler, train_loader, dev_loader, test_loader, config, device, n_gpu, label_list, logger)


if __name__ == "__main__":
    main()


