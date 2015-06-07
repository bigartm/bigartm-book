# -*- coding: utf-8 -*-

from collections import defaultdict
import artm.messages_pb2, artm.library 
import os
import glob
import pymystem3
import multiprocessing 

def stem_russian_batches(batch_filename):
    if not hasattr(stem_russian_batches, "stemmer"):
       stem_russian_batches.stemmer = pymystem3.Mystem()

    batch_stem_path='./stem/'

    print batch_filename + " loading..."
    batch = artm.library.Library().LoadBatch(batch_filename)
    print batch_filename + " loading done."
    batch_stem = artm.messages_pb2.Batch() 
    # stem tokens
    token_list = list()
    for token in batch.token:
        token_list.append(token)
    text = ' '.join(token_list)
    text_stem = stem_russian_batches.stemmer.lemmatize(text)
    token_stem_list = ''.join(text_stem).strip().split(' ')

    token_id_to_token_stem_id = dict()
    token_stem_to_token_stem_id = dict()
    for (token_id, token_stem) in enumerate(token_stem_list):
        #print token_id, token_stem
        if not token_stem_to_token_stem_id.has_key(token_stem):
            token_stem_to_token_stem_id[token_stem] = len(batch_stem.token)
            batch_stem.token.append(token_stem)
        token_id_to_token_stem_id[token_id] = token_stem_to_token_stem_id[token_stem]
    print batch_filename + " " + str(len(batch.token)) + " -> " + str(len(batch_stem.token))
    # convert items
    for item in batch.item:
        # print item.title
        # add item
        item_stem = batch_stem.item.add()
        item_stem.id = item.id 
        item_stem.title = item.title   
        # add fields
        for field in item.field:
            field_stem_dict = defaultdict(int)
            for token_num in xrange(len(field.token_id)):
                token_id = field.token_id[token_num]
                token_stem_id = token_id_to_token_stem_id[token_id]
                token_count = field.token_count[token_num]
                field_stem_dict[token_stem_id] += token_count 

            field_stem = item_stem.field.add()
            field_stem.name = field.name
            for token_stem_id in field_stem_dict:
                field_stem.token_id.append(token_stem_id)
                field_stem.token_count.append(field_stem_dict[token_stem_id])
    # save batch
    print batch_filename + " saving result..."
    artm.library.Library().SaveBatch(batch_stem, batch_stem_path)
    print batch_filename + " saving done."
    return 0

if __name__ == '__main__':
    batch_path = 'ru_batches_2/'

    print 'Testing pymystem3... '
    stemmer = pymystem3.Mystem()
    text = "Красивая мама красиво мыла раму"
    lemmas = stemmer.lemmatize(text)
    print(''.join(lemmas))
    print 'Testing pymystem3 done.',

    pool = multiprocessing.Pool()
    pool.map(stem_russian_batches, glob.glob(batch_path + "*.batch"))
    print 'Add batches done.'
