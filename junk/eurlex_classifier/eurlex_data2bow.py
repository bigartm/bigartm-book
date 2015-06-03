# EUR-lex collection -> Bag-of-Words parser.
__author__ = 'Murat Apishev, great-mel@yandex.ru'

import os
import sys
import glob
import string
import time

data_folder = os.path.dirname(os.path.abspath(__file__)) + '/eurlex_data'

data_documents_all_file = data_folder + '/eurlex_tokenstring.arff'
data_documents_train_file = data_folder + '/eurlex_tokenstring_CV1-10_train.arff'
data_documents_test_file = data_folder + '/eurlex_tokenstring_CV1-10_test.arff'
data_labels_file = data_folder + '/id2class_eurlex_eurovoc.qrels'

docword_train_file = data_folder + '/docword_train.eurlex' # The docword files have following format:
docword_test_file = data_folder + '/docword_test.eurlex'   # 'document_id1'
                                                           # 'label1 label2 ... labelN '
                                                           # 'label_id1 label_id2 ... label_idN '
                                                           # 'token1 token_id1 count1'
                                                           # 'token2 token_id2 count2'
                                                           # ...
                                                           # 'tokenK token_idK countK'
                                                           #
                                                           # 'doucument_id2'
                                                           # ...

vocab_file = data_folder + '/vocab.eurlex'     # Both of these files contains such strings as next:
labels_file = data_folder + '/labels.eurlex'   # 'token1 count1'
                                               # 'token2 count2'
                                               # ...

no_occurrences_to_be_seldom = 20  # how many times token should occur to be useful

# Parser's work consits of several steps:
# 1.1) look through 'data_documents_all_file' and get all unique tokens from there into map.
#      Then map is used to remove too seldom tokens, and info from it moves to list;
# 1.2) sort this list and save it's content into 'vocab_file';
# 2.1) look through 'data_labels_file' and get all unique labels from there to list;
# 2.2) save info about labels in documents into special map;
# 2.3) sort list of unique labels and save it's content into 'labels_file';
# 3)   look through both files 'data_documents_train_file' and 'data_documents_test_file' and fill the
#      'docword_train_file' and 'docword_test_file', using info from 'vocab_file' and 'labels_file';

#####
# 1)
tokens = []
tokens_map = {}
print 'Create vocab file...'
start_time = time.clock()
global_start_time = time.clock()
with open(data_documents_all_file, 'r') as f:
  for i in range(0, 6):  f.readline()  # skip first 6 non-useful lines

  for line in f:
    # remove non-useful parts of the line
    line_list = list(line)
    for i in range(-2, line.index(',') + 2): line_list[i] = ''
    line = ''.join(line_list)

    for token in line.split(' '): # put info into map
      if (tokens_map.has_key(token)): tokens_map[token] += 1
      else:                           tokens_map[token] = 1

# remove too seldom tokens from dictionary for train set
for key in tokens_map.keys():
  if (tokens_map[key] < no_occurrences_to_be_seldom): del tokens_map[key]
  
tokens = list(tokens_map.keys())

# fill 'vocab_file' with collection vocabulary
with open(vocab_file, 'w') as f:
  tokens = sorted(tokens)
  for token in tokens: f.write(token + ' ' + str(tokens_map[token]) + '\n')

elapsed_time = time.clock() - start_time
print 'Done. Final dictionary size is ' + str(len(tokens)) + ' tokens. Elapsed time: ' + str(elapsed_time) + ' sec\n'

#####
# 2)

#########################################################################################
# remove labels, that appeares only in test documents (step 1)
doc_ids_train = []
with open(data_documents_train_file, 'r') as f:
  for i in range(0, 6): f.readline()  # skip first 6 non-useful lines
  for line in f: doc_ids_train.append(''.join(list(line)[0 : line.index(',')]))
#########################################################################################

labels = []
labels_map = {}
labels_in_documents = {}
labels_appeared_in_train = []
print 'Create labels file...'
with open(data_labels_file, 'r') as f:
  for line in f:
    # remove non-useful parts of the line
    line_list = list(line)
    for i in range(-3, 0): line_list[i] = ''
    
    line = ''.join(line_list).split(' ')
    doc_id = line[1]
    doc_label = line[0] + '_@LABEL'

#########################################################################################
# remove labels, that appeares only in test documents (step 2)
    if (doc_id in doc_ids_train): labels_appeared_in_train.append(doc_label)
#########################################################################################

    if (labels_in_documents.has_key(doc_id)): labels_in_documents[doc_id].append(doc_label)
    else:                                     labels_in_documents[doc_id] = [doc_label]

    if (labels_map.has_key(doc_label)): labels_map[doc_label] += 1
    else:                               labels_map[doc_label] = 1

# remove labels, that occure once, from dictionary for train set ...
#for key in labels_map.keys():
#  if (labels_map[key] == 1): del labels_map[key]

#########################################################################################
# remove labels, that appeares only in test documents (step 3)
labels_appeared_in_train = set(labels_appeared_in_train)
for key in labels_map.keys():
  if not (key in labels_appeared_in_train): del labels_map[key]
#########################################################################################

labels = list(labels_map.keys())
set_labels = set(labels)
# ... and from future documents labels
for doc_id, doc_labels in labels_in_documents.iteritems():
  labels_in_documents[doc_id] = list(set.intersection(*[set(doc_labels), set_labels]))

# fill 'labels_file' with collection vocabulary
with open(labels_file, 'w') as f:
  labels = sorted(labels)
  for label in labels: f.write(label + ' ' + str(labels_map[label]) + '\n')

elapsed_time = time.clock() - start_time
print 'Done. Final labels count is ' + str(len(labels)) + ' ones. Elapsed time: ' + str(elapsed_time) + ' sec.\n'

#####
# 3)
print 'Create docword files...'
for file in [data_documents_test_file, data_documents_train_file]:
  with open(file, 'r') as f:
    if (file == data_documents_train_file): docword_file = open(docword_train_file, 'w')
    else:                                   docword_file = open(docword_test_file, 'w')

    for i in range(0, 6): f.readline()  # skip first 6 non-useful lines

    doc_counter_to_print = 0
    for line in f:
      line_list = list(line)
      doc_id = ''.join(line_list[0 : line.index(',')])
      # remove non-useful parts of the line
      for i in range(-2, line.index(',') + 2): line_list[i] = ''

      # token -> id and save both
      doc_tokens_id = []
      doc_tokens = []
      for token in ''.join(line_list).split(' '):
        try:
          id = str(tokens.index(token))
          doc_tokens_id.append(id)
          doc_tokens.append(token)
        except: continue

      # get counters for each token in document
      doc_tokens_map = {}
      id_index = 0
      for token_id in doc_tokens_id:
        if (doc_tokens_map.has_key(token_id)): doc_tokens_map[token_id][0] += 1
        else:
          doc_tokens_map[token_id] = []
          doc_tokens_map[token_id].append(1)
          doc_tokens_map[token_id].append(doc_tokens[id_index])
        id_index += 1
    
      # put info into docword file
      if (labels_in_documents.has_key(doc_id) and len(doc_tokens_map.keys()) > 0):
        # write line of lables
        docword_file.write(doc_id + '\n')
        for label in labels_in_documents[doc_id]:
          docword_file.write(label + ' ')
        docword_file.write('\n')

        # write line of lables ids
        for label in labels_in_documents[doc_id]:
          docword_file.write(str(labels.index(label)) + ' ')
        docword_file.write('\n')
        

        for (token_id, pair) in doc_tokens_map.items():
          docword_file.write(pair[1] + ' ' + token_id + ' ' + str(pair[0]) + '\n')
        docword_file.write('\n')

        doc_counter_to_print += 1
        print 'Document #' + str(doc_counter_to_print) + ' had been processed\n\n'

  docword_file.close()

elapsed_time = time.clock() - start_time
print 'Done. Elapsed time: ' + str(elapsed_time) + ' sec.\n'
print 'All elapsed time: ' + str(time.clock() - global_start_time) + ' sec.\n'