# -*- coding: utf-8 -*-

import os
import sys 

from data_utils import bigartm_vw2matrix_market, extract_vw_subsample, train_test_split_mm_file

data_path = os.path.join('data', 'vw.wiki-en.txt')
sample_path = os.path.join('data', 'vw.wiki-en-sample.txt')
mm_data = os.path.join('data', 'wiki-sample.mm')
mm_vocab = os.path.join('data', 'wiki-sample.mm-vocab.txt')
mm_data_train = os.path.join('data', 'wiki-sample_train.mm')
mm_data_test = os.path.join('data', 'wiki-sample_test.mm')

extract_vw_subsample(data_path, sample_path, num_docs=-1)
bigartm_vw2matrix_market(sample_path, mm_data, mm_vocab)
train_test_split_mm_file(mm_data, mm_data_train, mm_data_test, test_size=100000)