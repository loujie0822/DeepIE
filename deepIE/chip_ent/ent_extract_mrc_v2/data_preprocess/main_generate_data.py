#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


import os 
import sys 


from deepIE.chip_ent.ent_extract_mrc.data_preprocess.generate_mrc_dataset import generate_query_ner_dataset


def test_nested_ner():
    #source_file_path = os.path.join(ROOT_PATH, "data_preprocess/chip2020/chip_2020_1_train_reformat/val_data.json")
    #target_file_path = os.path.join(ROOT_PATH, "data_preprocess/chip2020/chip_2020_1_train_reformat/mrc-val_data.json")
    source_file_path = 'deepIE/chip_ent/ent_extract_mrc/data_preprocess/chip2020/chip_2020_1_train_debug/'
    target_file_path = 'deepIE/chip_ent/ent_extract_mrc/data_preprocess/chip2020/chip_2020_1_train_debug/'
    generate_query_ner_dataset(source_file_path+'val_data.txt', target_file_path+'mrc-val_data.json',
                               query_sign='default', total=5000)
    generate_query_ner_dataset(source_file_path+'train_data.txt', target_file_path+'mrc-train_data.json',
                               query_sign='default', total=15000)
    generate_query_ner_dataset(source_file_path+'test1.txt', target_file_path+'mrc-test1.json',
                               query_sign='default', total=3000)


if __name__ == "__main__":
    # using:  python deepIE/chip_ent/ent_extract_mrc/data_preprocess/main_generate_data.py
    test_nested_ner()
