# -*- coding: utf-8 -*-
import logging, codecs
from gensim.corpora import WikiCorpus
from collections import defaultdict, Counter
import artm.messages_pb2, artm.library 
import os
import glob

#===============================================================================
#         #for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
#             #for tokens, title, pageid in pool.imap(process_article, group): # chunksize=10):
#                 ... // continue with processing tokens
# =>
#         for text in texts:
#                 tokens, title, pageid = process_article(text) # chunksize=10):
#                 ... // continue with processing tokens
#===============================================================================


def load_title_list(csv_path='ru2en.csv'):
    title_list = list()
    with open(csv_path, 'r') as csv_file:
        for line in csv_file:
            (id_ru, title_ru, id_en, title_en) = line.split('|') 
            title_list.append((unicode(title_ru, 'utf-8').strip(' \t\n\r'), unicode(title_en, 'utf-8').strip(' \t\n\r')))
    print 'Found ' + str(len(title_list)) + ' en-ru pairs in ' + csv_path + ', first 10 pairs are as follows : '
    for (en, ru) in title_list[0:10]:
        print en + " <-> " + ru
    print '...\n'
    return title_list

def process_item(orig_batch, orig_item, targ_batch, targ_item, lang_map, lang):
    orig_field = orig_item.field[0]
    targ_field = targ_item.field[0]
    for token_num in xrange(len(orig_field.token_id)):
        token_id = orig_field.token_id[token_num]
        token_count = orig_field.token_count[token_num]
        token = orig_batch.token[token_id]
        if not lang_map.has_key(token):
            lang_map[token] = len(targ_batch.token)
            targ_batch.token.append(token)
            targ_batch.class_id.append(lang)
        targ_field.token_id.append(lang_map[token])
        targ_field.token_count.append(token_count)

def merge_batches(title_list=None, ru_path=None, en_path=None, batch_size=1000, target_path="merged_batches"):
    if not ru_path or not en_path:
        raise 'ru_path and en_path are not provided'

    batch_list = list()
    ru_title_to_batch_id = {}
    en_title_to_batch_id = {}

    all_batch_names = glob.glob(ru_path + "*.batch") + glob.glob(en_path + "*.batch")

    print 'Loading batches... '
    for batch_file in all_batch_names:
        batch = artm.library.Library().LoadBatch(batch_file)
        batch_id = len(batch_list)
        for (item_id, item) in enumerate(batch.item):
            if item.field[0].name == "@english":
                en_title_to_batch_id[item.title] = (batch_id, item_id)
            else:
                ru_title_to_batch_id[item.title] = (batch_id, item_id)
        batch_list.append(batch)
        print str(len(batch_list)) + " of " + str(len(all_batch_names)) + " batches done."

    total_items_processed = 0

    batch = artm.messages_pb2.Batch()
    ru_map = {}
    en_map = {}
    for (title_ru, title_en) in title_list:
        ru_found = (title_ru in ru_title_to_batch_id)
        en_found = (title_en in en_title_to_batch_id)

        if ru_found and en_found:
            (batch_id_ru, item_id_ru) = ru_title_to_batch_id[title_ru]
            (batch_id_en, item_id_en) = en_title_to_batch_id[title_en]

            total_items_processed += 1
            print total_items_processed, title_ru, title_en
            item = batch.item.add()
            item.title = (title_ru + "|" + title_en)
            item.id = total_items_processed
            field = item.field.add()

            ru_batch = batch_list[batch_id_ru]
            process_item(ru_batch, ru_batch.item[item_id_ru], batch, item, ru_map, '@russian')

            en_batch = batch_list[batch_id_en]
            process_item(en_batch, en_batch.item[item_id_en], batch, item, en_map, '@english')

        if len(batch.item) == batch_size:
            artm.library.Library().SaveBatch(batch, target_path)
            batch = artm.messages_pb2.Batch()
            ru_map = {}
            en_map = {}
            print 'Batch done.'

    if len(batch.item) > 0:
        artm.library.Library().SaveBatch(batch, target_path)
        print 'Last batch done.'

    print 'Processing done.'

title_list = load_title_list(csv_path='D:\\datasets\\multilang_wiki\\ru2en.csv')
ru_path = 'D:\\datasets\\multilang_wiki\\ru_batches\\'
en_path = 'D:\\datasets\\multilang_wiki\\en_batches\\'

'''
# Script to prepare data for testing of this script
import artm.messages_pb2, artm.library, sys, time, random, glob, math, codecs
from random import shuffle
en1 = artm.library.Library().LoadBatch(r'D:\datasets\multilang_wiki\test_merge\en_fe673cac-2e2b-4ab8-b92b-874b9efc162e.batch')
en2 = artm.library.Library().LoadBatch(r'D:\datasets\multilang_wiki\test_merge\en_fec6563c-097c-4f75-86f1-75c7f85d1fbb.batch')
ru1 = artm.library.Library().LoadBatch(r'D:\datasets\multilang_wiki\test_merge\ru_0ac1373f-8151-4bb1-a92b-ccc8758f1d78.batch')
ru2 = artm.library.Library().LoadBatch(r'D:\datasets\multilang_wiki\test_merge\ru_0adc98bb-8857-459f-ad07-44f84520e959.batch')

en_title = []
ru_title = []
for item in en1.item:
    en_title.append(item.title)
for item in en2.item:
    en_title.append(item.title)

for item in ru1.item:
    ru_title.append(item.title)
for item in ru2.item:
    ru_title.append(item.title)

shuffle(en_title)
shuffle(ru_title)

f = codecs.open("ru2en_test.csv", "w", "utf-8")
for i in range(0, min(len(en_title), len(ru_title))):
    f.write(str(-1) + "|" + ru_title[i] + "|" + str(-1) + "|" + en_title[i] + "\n")
f.close()
'''

#title_list = load_title_list(csv_path='D:\\datasets\\multilang_wiki\\test_merge\\ru2en_test.csv')
#ru_path = 'D:\\datasets\\multilang_wiki\\test_merge\\ru\\'
#en_path = 'D:\\datasets\\multilang_wiki\\test_merge\\en\\'

merge_batches(title_list = title_list, ru_path = ru_path, en_path = en_path, batch_size = 300, target_path = "merged_batches")
