#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# 

import os
import math
import gzip
import io
import torch
from glob import glob
from multiprocessing import Pool
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from deepIE.chip_ent.ent_extract_mrc.utils import examples_to_features
from deepIE.chip_ent.ent_extract_mrc.utils import read_mrc_ner_examples


class QueryNERProcessor(object):
    # processor for the query-based ner dataset
    def get_train_examples(self, data_dir):
        data = read_mrc_ner_examples(os.path.join(data_dir, "mrc-val_data.json"))
        return data

    def get_dev_examples(self, data_dir):
        return read_mrc_ner_examples(os.path.join(data_dir, "mrc-val_data.json"))

    def get_test_examples(self, data_dir):
        return read_mrc_ner_examples(os.path.join(data_dir, "mrc-test1.json"))


class CHIP2020Processor(QueryNERProcessor):
    def get_labels(self, ):
        return ["dis", "sym", "pro", "equ", "dru", "ite", "bod", "dep", "mic"]


class MRCNERDataLoader(object):
    def __init__(self, config, data_processor, label_list, tokenizer, mode="train", allow_impossible=True, entity_scheme="bes"):

        self.data_dir = config.data_dir
        self.max_seq_length= config.max_seq_length
        self.entity_scheme = entity_scheme
        self.distributed_data_sampler = config.n_gpu > 1 and config.data_parallel == "ddp"

        if mode == "train":
            self.train_batch_size = config.train_batch_size
            self.dev_batch_size = config.dev_batch_size
            self.test_batch_size = config.test_batch_size
            self.num_train_epochs = config.num_train_epochs 
        elif mode == "test":
            self.test_batch_size = config.test_batch_size
        elif mode == "transform_binary_files":
            print("=*="*15)
            print("Transform pre-processed MRC-NER datasets into binary files. ")
            print("max_sequence_length is : ", config.max_seq_length)
            print("data_dir is : ", config.data_dir)
            print("=*="*15)
        else:
            raise ValueError("[mode] for MRCNERDataLoader does not exist.")

        self.data_processor = data_processor 
        self.label_list = label_list 
        self.allow_impossible = allow_impossible
        self.tokenizer = tokenizer
        self.max_seq_len = config.max_seq_length 
        self.data_cache = config.data_cache

        self.num_train_instances = 0 
        self.num_dev_instances = 0 
        self.num_test_instances = 0

    def convert_examples_to_features(self, data_sign="train", num_data_processor=1, logger=None):

        logger.info("=*="*10)
        logger.info(f"loading {data_sign} data ... ...")

        if data_sign == "train":
            examples = self.data_processor.get_train_examples(self.data_dir)
            self.num_train_instances = len(examples)
        elif data_sign == "dev":
            examples = self.data_processor.get_dev_examples(self.data_dir)
            self.num_dev_instances = len(examples)
        elif data_sign == "test":
            examples = self.data_processor.get_test_examples(self.data_dir)
            self.num_test_instances = len(examples)
        else:
            raise ValueError("please notice that the data_sign can only be train/dev/test !!")

        if num_data_processor == 1:
            cache_path = os.path.join(self.data_dir, "mrc-ner.{}.cache.{}".format(data_sign, str(self.max_seq_len)))
            if os.path.exists(cache_path):
                logger.info(f"%%%% %%%% Load Saved Cache files in {cache_path} %%% %%% ")
                cache_path = gzip.open(cache_path, mode='rb')
                cache_path = io.BufferedReader(cache_path)
                features = torch.load(cache_path)
            else:
                features = examples_to_features(examples, self.tokenizer, self.label_list, self.max_seq_length,
                                                    allow_impossible=self.allow_impossible, entity_scheme=self.entity_scheme)
                cache_path = gzip.open(cache_path, mode='wb')
                torch.save(features, cache_path)
            return features

        # for mutil processor
        def export_features_to_cache_file(idx, sliced_features, num_data_processor):
            cache_path = os.path.join(self.data_dir, "mrc-ner.{}.cache.{}.{}-{}".format(data_sign, str(self.max_seq_len), str(num_data_processor), str(idx)))
            cache_path = gzip.open(cache_path, mode='wb')
            torch.save(sliced_features, cache_path)
            logger.info(f">>> >>> >>> export sliced features to : {cache_path}")

        features_lst = []
        total_examples = len(examples)
        size_of_one_process = math.ceil(total_examples / num_data_processor)
        path_to_preprocessed_cache = os.path.join(self.data_dir, "mrc-ner.{}.cache.{}.{}-*".format(data_sign, str(self.max_seq_len), str(num_data_processor)))
        collection_of_preprocessed_cache = glob(path_to_preprocessed_cache)

        if len(collection_of_preprocessed_cache) == num_data_processor:
            logger.info(f"%%%% %%%% Load Saved Cache files in {self.data_dir} %%% %%% ")
        elif len(collection_of_preprocessed_cache) != 0:
            for item_of_preprocessed_cache in collection_of_preprocessed_cache:
                os.remove(item_of_preprocessed_cache)
            for idx in range(num_data_processor):
                start = size_of_one_process * idx
                end = (idx+1) * size_of_one_process if (idx+1)* size_of_one_process < total_examples else total_examples
                sliced_examples = examples[start:end]
                sliced_features = convert_examples_to_features(sliced_examples, self.tokenizer, self.label_list,
                                                               self.max_seq_length, allow_impossible=self.allow_impossible, entity_scheme=self.entity_scheme)
                export_features_to_cache_file(idx, sliced_features, num_data_processor)
            del examples
        else:
            for idx in range(num_data_processor):
                start = size_of_one_process * idx
                end = (idx+1) * size_of_one_process if (idx+1)* size_of_one_process < total_examples else total_examples
                sliced_examples = examples[start:end]
                sliced_features = convert_examples_to_features(sliced_examples, self.tokenizer, self.label_list,
                                                               self.max_seq_length, allow_impossible=self.allow_impossible, entity_scheme=self.entity_scheme)
                export_features_to_cache_file(idx, sliced_features, num_data_processor)
            del examples

        multi_process_for_data = Pool(num_data_processor)
        for idx in range(num_data_processor):
            features_lst.append(multi_process_for_data.apply_async(MRCNERDataLoader.read_features_from_cache_file, args=(idx, self.data_dir, data_sign, self.max_seq_len, num_data_processor, logger)))

        multi_process_for_data.close()
        multi_process_for_data.join()
        features = []
        for feature_slice in features_lst:
            features.extend(feature_slice.get())

        logger.info("check number of examples before and after data processing : ")
        logger.info(f"{len(features)}, {total_examples}")
        assert len(features) == total_examples

        return features

    def get_dataloader(self, data_sign="train", num_data_processor=1, logger=None):
        
        features = self.convert_examples_to_features(data_sign=data_sign, num_data_processor=num_data_processor, logger=logger)
        logger.info(f"{len(features)} {data_sign} data loaded")
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        start_pos = torch.tensor([f.start_position for f in features], dtype=torch.long)   
        end_pos = torch.tensor([f.end_position for f in features], dtype=torch.long)
        span_pos = torch.tensor([f.span_position for f in features], dtype=torch.long)
        ner_cate = torch.tensor([f.ner_cate for f in features], dtype=torch.long)
        span_label_mask = torch.tensor([f.span_label_mask for f in features], dtype=torch.long)
        dataset = TensorDataset(input_ids, input_mask, segment_ids, start_pos, end_pos, span_pos, span_label_mask, ner_cate)
        
        if data_sign == "train":
            if self.distributed_data_sampler:
                # Please Note that DistributedSampler samples randomly
                datasampler = DistributedSampler(dataset)
            else:
                datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size)
        elif data_sign == "dev":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.dev_batch_size)
        elif data_sign == "test":
            datasampler = SequentialSampler(dataset) 
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size)

        return dataloader 

    @staticmethod
    def read_features_from_cache_file(idx, data_dir, data_sign, max_seq_len, num_data_processor, logger):
        cache_path = os.path.join(data_dir,
                                  "mrc-ner.{}.cache.{}.{}-{}".format(data_sign, str(max_seq_len), str(num_data_processor), str(idx)))
        cache_path = gzip.open(cache_path, mode='rb')
        cache_path = io.BufferedReader(cache_path)
        sliced_features = torch.load(cache_path)
        logger.info(f"load sliced features from : {cache_path} <<< <<< <<<")
        return sliced_features

    def get_train_instance(self, ):
        return self.num_train_instances








