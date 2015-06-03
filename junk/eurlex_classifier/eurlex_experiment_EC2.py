from __future__ import division

import os
import sys
import glob
import random
from random import randint
import string
import uuid
import time
import pickle
import numpy as np
from numpy import array

home_folder = '/home/ubuntu/'
sys.path.append(home_folder + 'bigartm/src/python')
sys.path.append(home_folder + 'bigartm/src/python/artm')

import artm.messages_pb2 as mes
import artm.library as lib

import sklearn
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

def perfect_classification(true_labels, probs):
  temp_true_labels = list(true_labels)
  temp_probs = list(probs)
  for i in range(0, sum(true_labels)):
    idx = temp_probs.index(max(temp_probs))
    if (temp_true_labels[idx] == 0): return False
    del temp_true_labels[idx]
    del temp_probs[idx]
  return True

def count_precision(true_labels, probs):
  retval = 0
  index = -1
  for label in true_labels:
    denominator = 0
    numerator = 0
    index += 1
    if (label):
      for prob_idx in range(0, len(probs)):
        if (probs[prob_idx] > probs[index]):
          denominator += 1
          if (true_labels[prob_idx] == 1):
            numerator += 1
    if (denominator > 0): retval += numerator / denominator
  retval /= sum(true_labels)
  return retval

def count_rank_loss(true_labels, probs):
  retval = 0
  index = -1
  for label in true_labels:
    denominator = 0
    numerator = 0
    index += 1
    if (label):
      for prob_idx in range(0, len(probs)):
        if (true_labels[prob_idx] != 1):
          denominator += 1
          if (probs[prob_idx] > probs[index]):
            numerator += 1
    if (denominator > 0): retval += numerator / denominator
  retval /= sum(true_labels)
  return retval

data_folder = os.path.dirname(os.path.abspath(__file__)) + '/eurlex_data/CV2'

labels_class = '@labels_class'
tokens_class = '@default_class'

processors_count        = 1

#### useful parameters ####
topics_count            = 10000
outer_iterations_count  = 4

inner_iterations_count  = [25,  25,  25,  22 ]
labels_class_weight     = [0.3, 0.4, 0.6, 0.8]
tokens_class_weight     = [5,   10,  15,  20 ]

smooth_theta_tau        = [0.001,  0.001,  0.001,  0.001 ]
smooth_phi_ct_tau       = [0.0075, 0.0075, 0.0075, 0.0075]
label_phi_ct_tau        = [2e+6,   2e+6,   2e+6,   2e+6  ]

##########################
sparse_phi_wt_tau =       [0, 0, 0, 0]  # currently useless regularizer
decorrelator_phi_wt_tau = [0, 0, 0, 0]  # currently useless regularizer
decorrelator_phi_ct_tau = [0, 0, 0, 0]  # currently useless regularizer

count_last_auc = False

################################
### batch size == 1000

#topics_count            = 10000
#outer_iterations_count  = 4

#inner_iterations_count  = [25,  25,  25,  22 ]
#labels_class_weight     = [0.3, 0.4, 0.6, 0.8]
#tokens_class_weight     = [5,   10,  15,  20 ]

#smooth_theta_tau        = [0.001,  0.001,  0.001,  0.001 ]
#smooth_phi_ct_tau       = [0.0075, 0.0075, 0.0075, 0.0075]
#label_phi_ct_tau        = [2e+6,   2e+6,   2e+6,   2e+6  ]

basic_average_auc       = [0.919, 0.975, 0.981, 0.982]
basic_average_one_error = [73.9,  40.4,  32.7,  29.4 ]
basic_average_is_error  = [100.0, 98.8,  96.8,  95.9 ]
basic_average_precision = [0.067, 0.246, 0.317, 0.339]
basic_average_auc_pr    = [0.137, 0.396, 0.483, 0.511]
basic_average_rank_loss = [8.13,  2.55,  1.86,  1.84 ]

##########

################################

test_labels_file = data_folder + '/test_labels.eurlex_artm'
test_documents_file = data_folder + '/267a6985-25ae-4276-adc3-a8f40cd8c32e.batch_test'

# Create master component and infer topic model
dictionary_config = lib.Library('').LoadDictionary(data_folder + '/dictionary_with_p_c.eurlex_artm');
master_config = mes.MasterComponentConfig()
master_config.processors_count = processors_count
master_config.disk_path = data_folder
with lib.MasterComponent(config = master_config) as master:
  # Create dictionary with tokens frequencies
  dictionary                 = master.CreateDictionary(dictionary_config)

  # Configure basic scores
  perplexity_score           = master.CreatePerplexityScore()
  sparsity_phi_wt_score      = master.CreateSparsityPhiScore()
  phi_sp_cfg = mes.SparsityPhiScoreConfig()
  phi_sp_cfg.class_id        = labels_class
  sparsity_phi_ct_score      = master.CreateSparsityPhiScore(config = phi_sp_cfg)
  top_tokens_score           = master.CreateTopTokensScore()

  # Configure basic regularizers
  decorrelator_reg_wt_config = mes.DecorrelatorPhiConfig()
  decorrelator_reg_wt_config.class_id.append(tokens_class)
  decorrelator_reg_wt        = master.CreateDecorrelatorPhiRegularizer(config = decorrelator_reg_wt_config)

  sp_phi_reg_wt_config       = mes.SmoothSparsePhiConfig()
  sp_phi_reg_wt_config.class_id.append(tokens_class)
  sp_phi_reg_wt              = master.CreateSmoothSparsePhiRegularizer(config = sp_phi_reg_wt_config)

  sp_theta_reg = master.CreateSmoothSparseThetaRegularizer()

  decorrelator_reg_ct_config = mes.DecorrelatorPhiConfig()
  decorrelator_reg_ct_config.class_id.append(labels_class)
  decorrelator_reg_ct        = master.CreateDecorrelatorPhiRegularizer(config = decorrelator_reg_ct_config)

  label_reg_ct_config = mes.LabelRegularizationPhiConfig()
  label_reg_ct_config.class_id.append(labels_class)
  label_reg_ct_config.dictionary_name = dictionary_config.name
  label_reg_ct        = master.CreateLabelRegularizationPhiRegularizer(config = label_reg_ct_config)

  sp_phi_reg_ct_config       = mes.SmoothSparsePhiConfig()
  sp_phi_reg_ct_config.class_id.append(labels_class)
  sp_phi_reg_ct              = master.CreateSmoothSparsePhiRegularizer(config = sp_phi_reg_ct_config)

  # Configure the model
  model_config = mes.ModelConfig()
  model_config.topics_count = topics_count
  model_config.inner_iterations_count = 1
  model_config.class_id.append(tokens_class)
  model_config.class_weight.append(1)
  model_config.class_id.append(labels_class)
  model_config.class_weight.append(1)
  # model_config.reuse_theta = True

  model = master.CreateModel(model_config)
  model.EnableScore(perplexity_score)
  model.EnableScore(sparsity_phi_wt_score)
  model.EnableScore(sparsity_phi_ct_score)
  model.EnableScore(top_tokens_score)

  model.EnableRegularizer(decorrelator_reg_wt, 0)
  model.EnableRegularizer(sp_phi_reg_wt, 0)
  model.EnableRegularizer(sp_theta_reg, 0)
  model.EnableRegularizer(decorrelator_reg_ct, 0)
  model.EnableRegularizer(label_reg_ct, 0)
  model.EnableRegularizer(sp_phi_reg_ct, 0)


  model.Initialize(dictionary)       # Setup initial approximation for Phi matrix.

  #rand_doc_count = 5    # number of ducuments id's to sample for estimation
  #rand_batch_count = 2
  #smooth_coef = 0.001   # smooth value for zero values
  #rand = random.Random()
    
  #topic_model = mes.TopicModel()
  #topic_model.name = model.name()
  #topic_model.topics_count = topics_count
  #token_weights = []

  #batches_list = []
  #for file in os.listdir(data_folder):
  #  if file.endswith(".batch"): batches_list.append(data_folder + file)

  #batch = []
  #batch_token = []
  #for i in range (0, rand_batch_count):
  #  batch.append(mes.Batch())
  #  batch_index = randint(0, len(batches_list) - 1)
  #  with open(batches_list[batch_index], 'rb') as file:
  #    batch[i].ParseFromString(file.read())
  #  batch_token.append(list(batch[i].token))

  #for topic in range(0, topics_count):
  #  rand_doc_index = []
  #  batch_index = randint(0, rand_batch_count - 1)
  #  print 'New topic'
  #  t = time.clock()

  #  if (topic == 0):
  #    for entry in dictionary_config.entry:
  #      topic_model.token.append(entry.key_token)
  #      topic_model.class_id.append(entry.class_id)
  #      token_weights.append(topic_model.token_weights.add())

  #  for i in range(0, rand_doc_count):
  #    rand_doc_index.append(randint(0, len(batch[batch_index].item) - 1))

  #  for token_index in range(0, len(dictionary_config.entry)):
  #    value = 0.0
  #    batch_has_token = True
  #    # here we assume that there're no labels and tokens with similar keywords
  #    try:    token_index_in_batch = batch_token[batch_index].index(dictionary_config.entry[token_index])
  #    except: batch_has_token = False
  #    denominator = 0
  #    if (batch_has_token):
  #      for i in range(0, rand_doc_count):
  #        try:
  #          idx = list(batch[batch_index].item[rand_doc_index[i]].field[0].token).index(token_index_in_batch)
  #          value += batch[batch_index].item[rand_doc_index[i]].field[0].token_count[idx]
  #          denominator += 1
  #        except: continue
  #    if (value == 0.0): value = smooth_coef
  #    else:              value /= denominator
  #    token_weights[token_index].value.append(value)
  #  print time.clock() - t
     
  ##overwrite the model and add this batch
  #model.Overwrite(topic_model, commit = False)
  #master.WaitIdle()
  #model.Synchronize(apply_weight = 1.0, decay_weight = 1.0)

  print "Load test batch and it's labels..."
  batch = mes.Batch()
  Theta_test = mes.ThetaMatrix()

  with open(test_labels_file, 'rb') as f:
    true_p_cd = pickle.load(f)

  with open(test_documents_file, 'rb') as test_file:
    batch.ParseFromString(test_file.read())
  batch.id = str(uuid.uuid4())

  time_start = time.clock()
  print 'Start processing'
  for iter in range(0, outer_iterations_count):
    config_copy = mes.ModelConfig()
    config_copy.CopyFrom(model.config())
    config_copy.regularizer_tau[0]     = decorrelator_phi_wt_tau[iter]
    config_copy.regularizer_tau[1]     = sparse_phi_wt_tau[iter]
    config_copy.regularizer_tau[2]     = smooth_theta_tau[iter]
    config_copy.regularizer_tau[3]     = decorrelator_phi_ct_tau[iter]
    config_copy.regularizer_tau[4]     = label_phi_ct_tau[iter]
    config_copy.regularizer_tau[5]     = smooth_phi_ct_tau[iter]
    config_copy.class_weight[0]        = tokens_class_weight[iter]
    config_copy.class_weight[1]        = labels_class_weight[iter]
    config_copy.inner_iterations_count = inner_iterations_count[iter]
    model.Reconfigure(config_copy)

    master.InvokeIteration(1)       # Invoke one scan of the entire collection...
    master.WaitIdle()               # and wait until it completes.
    model.Synchronize()             # Synchronize topic model.

    if (not count_last_auc or (iter == outer_iterations_count - 1)):
      get_theta_args = mes.GetThetaMatrixArgs()
      get_theta_args.model_name = model.name()
      get_theta_args.batch.CopyFrom(batch)
      print 'Get Theta...'
      Theta_test = master.GetThetaMatrix(args = get_theta_args)

      get_phi_args = mes.GetTopicModelArgs()
      get_phi_args.model_name = model.name()
      get_phi_args.class_id.append(labels_class)
      print 'Get Phi...'
      topic_model = master.GetTopicModel(args = get_phi_args)

      Phi = []
      for w in range(0, len(topic_model.token_weights)):
        Phi.append([])
        for t in range(0, topics_count):
          Phi[w].append(topic_model.token_weights[w].value[t])
      Phi = np.array(Phi)

      items_auc = []
      items_auc_pr = []
      one_error, is_error, precision, rank_loss = 0, 0, 0, 0

      item_index = -1
      for p_td in Theta_test.item_weights:
        p_td_array = np.array(p_td.value)
        item_index += 1
        p_cd = []
        for p_wt in Phi: 
          p_cd.append(np.dot(p_td_array, p_wt))

        true_cd = [int(bool(p)) for p in true_p_cd[item_index]]

        items_auc.append(sklearn.metrics.roc_auc_score(true_cd, p_cd))

        prec, rec, temp = precision_recall_curve(true_cd, p_cd)
        items_auc_pr.append(auc(rec, prec))

        if (true_cd[p_cd.index(max(p_cd))] == 0): one_error += 1
        if (not perfect_classification(true_cd, p_cd)): is_error += 1
        precision += count_precision(true_cd, p_cd)
        rank_loss += count_rank_loss(true_cd, p_cd)

      average_auc = sum(items_auc) / len(items_auc)
      average_auc_pr = sum(items_auc_pr) / len(items_auc)
      average_one_error = (one_error / len(items_auc)) * 100
      average_is_error = (is_error / len(items_auc)) * 100
      average_precision = precision / len(items_auc)
      average_rank_loss = rank_loss / len(items_auc) * 100

      print '-----------------------------------------------------------------------'
      print "#" + str(iter),
      print ": Perplexity: %.0f" % perplexity_score.GetValue(model).value,
      print "| Phi: %.1f" % sparsity_phi_wt_score.GetValue(model).value,
      print "| Psi: %.1f\n" % sparsity_phi_ct_score.GetValue(model).value

      print "     AUC = %.3f" % average_auc,
      print "(%.3f) " % (average_auc - basic_average_auc[iter]),
      print "| 1Err = %.1f" % average_one_error,
      print "(%.1f) " % (basic_average_one_error[iter] - average_one_error)
      print "      IsErr = %.1f" % average_is_error,
      print "(%.1f) " % (basic_average_is_error[iter] - average_is_error),
      print "| Prec = %.3f" % average_precision,
      print "(%.3f) " % (average_precision - basic_average_precision[iter])
      print "     Rnk_L = %.2f" % average_rank_loss,
      print "(%.2f)" % (basic_average_rank_loss[iter] - average_rank_loss),
      print "|  AUC_PR = %.3f" % average_auc_pr,
      print "(%.3f) " % (average_auc_pr - basic_average_auc_pr[iter])
    else:
      print "#" + str(iter),
      print ": Perp = %.0f" % perplexity_score.GetValue(model).value,
      print "| Phi = %.2f" % sparsity_phi_wt_score.GetValue(model).value,
      print "| Psi = %.2f" % sparsity_phi_ct_score.GetValue(model).value

  print "Model is ready, elapsed time = " + str(time.clock() - time_start) + ' sec.\n'
  #lib.Visualizers.PrintTopTokensScore(top_tokens_score.GetValue(model))
#######################################################################################################################