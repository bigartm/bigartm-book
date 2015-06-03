# EUR-lex Bag-of-Words -> BigARTM-batches parser.
__author__ = 'Murat Apishev, great-mel@yandex.ru'

import os
import sys
import glob
import string
import pickle
import time
import uuid

import artm.messages_pb2 as mes

def adapt_char (text): 
  return ''.join([i if ord(i) < 128 else chr(ord(i) % 128) for i in text])

data_folder = os.path.dirname(os.path.abspath(__file__)) + '/eurlex_data'

vocab_file = data_folder + '/vocab.eurlex'
labels_file = data_folder + '/labels.eurlex'
data_documents_train_file = data_folder + '/docword_train.eurlex'
data_documents_test_file = data_folder + '/docword_test.eurlex'

test_labels_file = data_folder + '/test_labels.eurlex_artm'  # Format:
                                                             # [[item1_label], [item2_label], ...]
                                                             # where 'item_label' is a vector with length
                                                             # equal to naumber of all labels. It has zero
                                                             # in all positions except ones that contains
                                                             # labels of current item. These positions countain
                                                             # 1 / '# of these labels'
dictionary_file = data_folder + '/dictionary_with_p_c.eurlex_artm'

batch_size = 1000  # number of documents in each train batch
labels_class = '@labels_class'
tokens_class = '@default_class'

# Parser's work consits of several steps:
# 1.1) read vocabulary into list from 'vocab_file';
# 1.2) read all labels into list from 'labels_file';
# 1.3) prepare and save 'dictionary_file'
# 2.1) read 'data_documents_train_file' and create nessasary number of full batches;
# 2.2) read 'data_documents_test_file' and create one test batch without labels and put it's true
#      labels into 'test_labels_file';

#####
# 1)
global_start_time = time.clock()

tokens = []
tokens_counts = {}
tokens_norm = 0.0
with open(vocab_file, 'r') as f:
  for line in f:
    # remove '\n' and add new token
    line_list = list(line)
    del line_list[-1]
    token_and_count = ''.join(line_list).split(' ')
    token = adapt_char(token_and_count[0])
    count = int(token_and_count[1])
    tokens.append(token)
    tokens_counts[token] = count
    tokens_norm += count

labels = []
labels_counts = {}
labels_norm = 0.0
with open(labels_file, 'r') as f:
  for line in f:
    # remove '\n' and add new label
    line_list = list(line)
    del line_list[-1]
    label_and_count = ''.join(line_list).split(' ')
    label = adapt_char(label_and_count[0])
    count = int(label_and_count[1])
    labels.append(label)
    labels_counts[label] = count
    labels_norm += count

print 'Create dictionary file...'
dict_cfg = mes.DictionaryConfig()
dict_cfg.name = 'dictionary_with_p_c'

for token in tokens:
  entry = dict_cfg.entry.add()
  entry.key_token = token
  entry.class_id  = tokens_class
  entry.value     = tokens_counts[token] / tokens_norm
for label in labels:
  entry = dict_cfg.entry.add()
  entry.key_token = label
  entry.class_id  = labels_class
  entry.value     = labels_counts[label] / labels_norm

dict_str = dict_cfg.SerializeToString()
with open(dictionary_file, 'wb') as f: f.write(dict_str)
print 'Done'

#####
# 2)
print 'Create batches...'
test_labels = []  # will be put into 'test_labels_file'
for file in [data_documents_test_file, data_documents_train_file]:
  with open(file, 'r') as f:
    # declarations
    batch_token = []
    batch = mes.Batch()
    item = mes.Item()
    field = mes.Field()
    item_index = -1

    state = 0  # 'state' variable denotes one of following states:
               # 0 - current line is the document id (no need line);
               # 1 - current line is labels;
               # 2 - current line is the list of indices of labels (no need line);
               # 3 - current line is the pair 'token_id token_count'.
    for line in f:
      if (state == 0): 
        state = 1  # read 'doc_id' string
      elif (state == 1):  # read 'labels' string
        item_index += 1
        item = batch.item.add()
        item.id = item_index
        field = item.field.add()

        # read string with list of labels
        line_list = list(line)
        for i in [0, 1]: del line_list[-1]  # remove ' \n'
        
        test_item_label_indices = []  # this variable is using if file == test

        for label in ''.join(line_list).split(' '):
          adapt_label = adapt_char(label)
          if (file == data_documents_train_file):
            if (not adapt_label in batch_token):
              batch_token.append(adapt_label)
              batch.token.append(adapt_label)
              batch.class_id.append(labels_class)

            label_index = batch_token.index(adapt_label)
            field.token_id.append(label_index)
            field.token_count.append(1)
          else:
            test_item_label_indices.append(labels.index(adapt_label))

        if (file == data_documents_test_file):
          test_labels.append([0] * len(labels))
          for label_index in test_item_label_indices:
            test_labels[item_index][label_index] = 1.0

        state = 2
      elif (state == 2): state = 3  # read labels ids string
      elif (state == 3):
        if (line == '\n'):
          if (item_index == batch_size and file == data_documents_train_file):
            # save batch
            name = str(uuid.uuid4())
            print 'New train batch was prepared: ' + name
            batch_str = batch.SerializeToString()
            with open(data_folder + '/' + name + '.batch', 'wb') as b_f: b_f.write(batch_str)

            batch_token = []
            batch = mes.Batch()
            item_index = -1

          state = 0
        else:
          # process next triple 'token token_id token_count'
          triple = line.split(' ')
          token = triple[0]
          if (not token in batch_token):
            batch_token.append(adapt_char(token))
            batch.token.append(adapt_char(token))
            batch.class_id.append(tokens_class)
          token_index = batch_token.index(adapt_char(token))
          field.token_id.append(token_index)
          field.token_count.append(int(triple[2]))

  # save last batch and test_labels
  name = str(uuid.uuid4())
  batch_str = batch.SerializeToString()
  with open(data_folder + '/' + name + '.batch', 'wb') as f: f.write(batch_str)
  if (file == data_documents_train_file):
    print 'New train batch was prepared: ' + name
  else:
    print 'Test batch was prepared: ' + name
    with open(test_labels_file, 'wb') as f: pickle.dump(test_labels, f)

print 'Done. All elapsed time: ' + str(time.clock() - global_start_time) + ' sec.\n'