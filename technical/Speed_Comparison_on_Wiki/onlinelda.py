# -*- coding: utf-8 -*-

import os
import shutil
import json
import tempfile
import math
import subprocess

import gensim
import artm
import numpy as np


from performance import ResourceTracker, track_cmd_resource


DATA_PATH = 'data'
TARGET_PATH = 'target'


def infer_stochastic_matrix(alphas, a=0, tol=1e-10):
    """
    Infer stochastic matrix from Dirichlet distribution.

    Input: matrix with rows corresponding to parameters of
    the asymmetric Dirichlet distributions, parameter a.

    a=0 => expected distributions
    a=1 => most probable distributions
    a=1/2 => normalized median-marginal distributions

    Returns: inferred stochastic matrix.
    """
    assert isinstance(alphas, np.ndarray)
    A = alphas - a
    A[A < tol] = 0
    A /= A.sum(axis=1, keepdims=True) + 1e-15
    return A


def convert_mm_to_vw(mm_filename, vw_filename):
    with open(vw_filename, 'w') as fout, open(mm_filename) as fin:
        fin.readline()
        D, W, N = map(int, fin.readline().rstrip().split(' '))

        cur_d = None
        cur_feats = ''

        def emit_example():
            if cur_d is not None:
                fout.write('0 \'{} | '.format(cur_d) + cur_feats + '\n')

        for line in fin:
            d, w, cnt = map(int, line.rstrip().split(' '))
            if cur_d != d:
                emit_example()
                cur_feats = ''
                cur_d = d

            if cnt > 1:
                cur_feats += ' {}:{}'.format(w, cnt)
            else:
                cur_feats += ' {}'.format(w)
        emit_example()


def compute_perplexity_artm(corpus, Phi, Theta, word_shift=0):
    sum_n = 0.0
    sum_loglike = 0.0
    for doc_id, doc in enumerate(corpus):
        for term_id, count in doc:
            sum_n += count
            sum_loglike += count * np.log( np.dot(Theta[doc_id, :], Phi[:, term_id + word_shift]))
    print sum_loglike, sum_n
    perplexity = np.exp(- sum_loglike / sum_n)
    return perplexity


def start_report():
    try:
        uname = subprocess.check_output(['uname', '-a'])
    except OSError:
        uname = None
    try:
        lscpu = subprocess.check_output(['lscpu'])
    except OSError:
        lscpu = None

    report = {
        'machine': {
            'uname': uname,
            'lscpu': lscpu,
        },
    }
    return report


### Interface for OnlineLDA implementation

def run_impl(impl, name, train, test, wordids=None,
             num_processors=1, num_topics=100, batch_size=10000, passes=1,
             kappa=0.5, tau0=64, alpha=0.1, beta=0.1, num_document_passes=1):
    """
    Run OnlineLDA algorithm.

    :param name: name of experiment
    :param train: train corpus in MM format (should be in DATA_PATH folder, without extension)
    :param test: dict {id: <test corpus>} where id is 'test' and 'valid' (should be in DATA_PATH folder, without extension)
    :param wordids: id-to-word mapping file (gensim Dictionary)
    :param num_processors: number of processors to use (more than one means parallelization)
    :param num_topics: number of topics
    :param batch_size: number of documents in the batch
    :param passes: number of passes through the train corpus
    :param kappa: power of learning rate
    :param tau0: initial learning rate
    :param alpha: document distribution smoothing coefficient (parameter dirichlet prior)
    :param beta: topic distribution smoothing coefficient (parameter dirichlet prior)
    :param num_document_passes: maximal number of inner iterations on E-step

    Note: create TARGET_PATH folder for results
    """
    func = globals()['run_{}'.format(impl)]
    func(name, train, test, wordids,
         num_processors=num_processors, num_topics=num_topics, batch_size=batch_size, passes=passes,
         kappa=kappa, tau0=tau0, alpha=alpha, beta=beta, num_document_passes=num_document_passes)


### Gensim
#
# Website: http://radimrehurek.com/gensim/
# Tutorial: http://radimrehurek.com/gensim/tut2.html
#

def run_gensim(name, train, test, wordids,
               num_processors=1, num_topics=100, batch_size=10000, passes=1,
               kappa=0.5, tau0=1.0, alpha=0.1, beta=0.1, update_every=1, num_document_passes=1):

    if tau0 != 1.0:
        print 'Warning: Gensim does not support tau0 != 1.0, so custom tau0 value will be ignored'

    id2word = gensim.corpora.Dictionary.load_from_text(os.path.join(DATA_PATH, wordids))
    train_corpus = gensim.corpora.MmCorpus(os.path.join(DATA_PATH, '{}.mm'.format(train)))

    # see https://github.com/piskvorky/gensim/issues/288
    #from gensim.corpora.sharded_corpus import ShardedCorpus
    #train_corpus = ShardedCorpus(os.path.join(DATA_PATH, '{}.sc'.format(train)), train_corpus_mm, shardsize=1000, overwrite=False)

    gamma_threshold = 0.001

    with ResourceTracker() as tracker:

        model = gensim.models.LdaMulticore(
            corpus=train_corpus,
            id2word=id2word,
            num_topics=num_topics,
            chunksize=batch_size,
            passes=passes,
            batch=False,
            alpha=alpha,
            eta=beta,
            decay=kappa,
            eval_every=0,
            iterations=num_document_passes,
            gamma_threshold=gamma_threshold,
            workers=num_processors
        )

    model.save(os.path.join(TARGET_PATH, '{}.gensim_model'.format(name)))

    report = start_report()
    report['train_resources'] = tracker.report()

    Lambda = model.state.get_lambda()
    Phi = infer_stochastic_matrix(Lambda, 0)
    matrices = {
        'Lambda': Lambda,
        'Phi_mean': Phi,
        'Phi_map': infer_stochastic_matrix(Lambda, 1),
    }

    for identifier, corpus_name in test.iteritems():
        test_corpus = gensim.corpora.MmCorpus(os.path.join(DATA_PATH, '{}.mm'.format(corpus_name)))

        with ResourceTracker() as tracker:
            Gamma, _ = model.inference(test_corpus)

        Theta = infer_stochastic_matrix(Gamma, 0)
        matrices['{}_Gamma'.format(identifier)] = Gamma
        matrices['{}_Theta_mean'.format(identifier)] = Theta
        matrices['{}_Theta_map'.format(identifier)] = infer_stochastic_matrix(Gamma, 1)

        report[identifier] = {
            'inference_resources': tracker.report(),
            'perplexity_gensim': np.exp(-model.log_perplexity(test_corpus)),
            'perplexity_artm': compute_perplexity_artm(test_corpus, Phi, Theta),
        }

    with open(os.path.join(TARGET_PATH, '{}.report.json'.format(name)), 'w') as report_file:
        json.dump(report, report_file, indent=2)
    np.savez_compressed(os.path.join(TARGET_PATH, '{}.matrices.npz'.format(name)), **matrices)


### Vowpal Wabbit LDA
#
# Website: https://github.com/JohnLangford/vowpal_wabbit/wiki
# Tutorial: https://github.com/JohnLangford/vowpal_wabbit/wiki/Latent-Dirichlet-Allocation
#

def run_vw(name, train, test, wordids,
           num_processors=1, num_topics=100, batch_size=10000, passes=1,
           kappa=0.5, tau0=64, alpha=0.1, beta=0.1, update_every=1, num_document_passes=1, limit_docs=None, seed=123):

    def read_vw_matrix(filename, topics=False, n_term=None):
        with open(filename) as f:
            if topics:
                for i in xrange(11):
                    f.readline()
            result_matrix = []
            for line in f:
                parts = line.strip().replace('  ', ' ').split(' ')
                if topics:
                    index = int(parts[0])
                    matrix_line = map(float, parts[1:])
                    if index <= n_term or not n_term:
                        result_matrix.append(matrix_line)
                else:
                    index = int(parts[-1])
                    matrix_line = map(float, parts[:-1])
                    result_matrix.append(matrix_line)
        return np.array(result_matrix, dtype=float)

    def read_vw_gammas(predictions_path):
        """
        Read matrix of inferred document distributions (gammas) from vw predictions file.
        :return: np.ndarray, size = num_docs x num_topics
        """
        gammas = read_vw_matrix(predictions_path, topics=False)
        return gammas

    def read_vw_lambdas(topics_path, n_term=None):
        """
        Read matrix of inferred topic distributions (lambdas) from vw readable model file.
        :param n_term: number of words
        :return: np.ndarray, size = num_topics x num_terms
        """
        lambdas = read_vw_matrix(topics_path, topics=True, n_term=n_term).T
        return lambdas

    if num_processors != 1:
        raise ValueError('Vowpal Wabbit LDA does not support parallelization')

    if update_every != 1:
        raise ValueError('Vowpal Wabbit LDA does not support update_every != 1')

    id2word = gensim.corpora.Dictionary.load_from_text(os.path.join(DATA_PATH, wordids))
    train_corpus = gensim.corpora.MmCorpus(os.path.join(DATA_PATH, '{}.mm'.format(train)))

    for n in [train] + test.values():
        if not os.path.exists(os.path.join(DATA_PATH, '{}.vw'.format(n))):
            print 'Converting {}.mm -> {}.vw'.format(n, n)
            convert_mm_to_vw(os.path.join(DATA_PATH, '{}.mm'.format(n)), os.path.join(DATA_PATH, '{}.vw'.format(n)))

    tempdir = tempfile.mkdtemp()
    print 'Temp dir:', tempdir

    cmd = [
        '/usr/local/bin/vw',
        '-d', os.path.join(DATA_PATH, '{}.vw'.format(train)),
        '-b', '{}'.format(int(np.ceil(np.log2(len(id2word))))),
        '--noconstant',
        '--cache_file', os.path.join(tempdir, 'cache_file'),
        '--random_seed', str(seed),
        '--lda', str(num_topics),
        '--lda_alpha', str(alpha),
        '--lda_rho', str(beta),
        '--lda_D', str(train_corpus.num_docs),
        '--minibatch', str(batch_size),
        '--power_t', str(kappa),
        '--initial_t', str(tau0),
        '--passes', str(passes),
        '--readable_model', os.path.join(tempdir, 'readable_model'),
        '-p', os.path.join(tempdir, 'predictions'),
        '-f', os.path.join(TARGET_PATH, '{}.vw_model'.format(name)),
    ]

    if limit_docs:
        cmd += ['--examples', str(limit_docs)]

    exitcode, tracker = track_cmd_resource(cmd)
    if exitcode != 0:
        raise RuntimeError('VW exited with non-zero code {}'.format(exitcode))

    report = start_report()
    report['train_resources'] = tracker.report()

    Lambda = read_vw_lambdas(os.path.join(tempdir, 'readable_model'), n_term=len(id2word))
    Phi = infer_stochastic_matrix(Lambda, 0)
    matrices = {
        'Lambda': Lambda,
        'Phi_mean': Phi,
        'Phi_map': infer_stochastic_matrix(Lambda, 1),
    }

    for identifier, corpus_name in test.iteritems():
        test_corpus = gensim.corpora.MmCorpus(os.path.join(DATA_PATH, '{}.mm'.format(corpus_name)))

        predictions_path = os.path.join(tempdir, 'predictions_{}'.format(identifier))
        cmd = [
            'vw',
            os.path.join(DATA_PATH, '{}.vw'.format(corpus_name)),
            '--minibatch', str(test_corpus.num_docs),
            '--initial_regressor', os.path.join(TARGET_PATH, '{}.vw_model'.format(name)),
            '-p', predictions_path,
        ]

        exitcode, tracker = track_cmd_resource(cmd)

        Gamma = read_vw_gammas(predictions_path)
        Theta = infer_stochastic_matrix(Gamma, 0)
        matrices['{}_Gamma'.format(identifier)] = Gamma
        matrices['{}_Theta_mean'.format(identifier)] = Theta
        matrices['{}_Theta_map'.format(identifier)] = infer_stochastic_matrix(Gamma, 1)

        report[identifier] = {
            'inference_resources': tracker.report(),
            'perplexity_artm': compute_perplexity_artm(test_corpus, Phi, Theta, word_shift=0),#-1),
        }

    with open(os.path.join(TARGET_PATH, '{}.report.json'.format(name)), 'w') as report_file:
        json.dump(report, report_file, indent=2)
    np.savez_compressed(os.path.join(TARGET_PATH, '{}.matrices.npz'.format(name)), **matrices)

    shutil.rmtree(tempdir)


### BigARTM
#
# Website: http://bigartm.org/
#

def run_bigartm(name, train, test, wordids,
                num_processors=1, num_topics=100, batch_size=10000, passes=1,
                kappa=0.5, tau0=64, alpha=0.1, beta=0.1, update_every=1,
                num_document_passes=1, test_batch_size=10000, async=False):

    def prepare_batch_files(name, batch_size, batches_path, id2word=None):
        if not os.path.exists(batches_path):
            vw_path = os.path.join(DATA_PATH, '{}.vw'.format(name))
            if not os.path.exists(vw_path):
                print 'Converting {}.mm -> {}.vw'.format(name, name)
                convert_mm_to_vw(os.path.join(DATA_PATH, '{}.mm'.format(name)), vw_path)

            bv = artm.BatchVectorizer(data_path=vw_path, 
                                      data_format='vowpal_wabbit',
                                      target_folder=batches_path,
                                      batch_size=batch_size,
                                      gather_dictionary=False)
            if not id2word is None:
                with open(os.path.join(batches_path, 'vocab.txt'), 'w') as fout:
                    for i in xrange(len(id2word.keys())):
                        fout.write('{}\n'.format(i + 1))
            return bv
        else:
            return artm.BatchVectorizer(data_path=batches_path, data_format='batches', batch_size=batch_size)            

    report = start_report()
    train_batches_path = os.path.join(DATA_PATH, 'bigartm_batches_train_{}'.format(batch_size))

    id2word = gensim.corpora.Dictionary.load_from_text(os.path.join(DATA_PATH, wordids))
    train_vectorizer = prepare_batch_files(train, batch_size, train_batches_path, id2word)

    dictionary = artm.Dictionary()
    dictionary.gather(train_batches_path, vocab_file_path=os.path.join(train_batches_path, 'vocab.txt'))

    model = artm.ARTM(num_processors=num_processors,
                      cache_theta=False,
                      dictionary=dictionary,
                      num_topics=num_topics,
                      num_document_passes=num_document_passes)

    model.regularizers.add(artm.SmoothSparsePhiRegularizer(tau=beta))
    model.regularizers.add(artm.SmoothSparseThetaRegularizer(tau=alpha))
    model.scores.add(artm.PerplexityScore(name='ps', dictionary=dictionary))

    with ResourceTracker() as tracker:
        for _ in xrange(passes):
            model.fit_online(train_vectorizer, async=async, tau0=tau0, kappa=kappa, update_every=update_every)
            #model.fit_offline(train_vectorizer)

    Phi = model.get_phi()

    report['train_resources'] = tracker.report()
    with open(os.path.join(TARGET_PATH, '{}.report.json'.format(name)), 'w') as report_file:
        json.dump(report, report_file, indent=2)

    for test_key, test_name in test.iteritems():
        print 'Testing on hold-out set "{}"'.format(test_key)

        report[test_key] = {}
        test_batches_path = os.path.join(DATA_PATH, 'bigartm_batches_{}_{}'.format(test_key, test_batch_size))
        test_vectorizer = prepare_batch_files(test_name, test_batch_size, test_batches_path)
        test_corpus = gensim.corpora.MmCorpus(os.path.join(DATA_PATH, '{}.mm'.format(test_name)))

        with ResourceTracker() as tracker:
            Theta = model.transform(test_vectorizer)

            report[test_key]['inference_resources'] = tracker.report()
            report[test_key]['perplexity_artm'] = compute_perplexity_artm(test_corpus, Phi.as_matrix().T, Theta.as_matrix().T)

            if not async:
                report[test_key]['perplexity_bigartm'] = model.score_tracker['ps'].last_value
            #    report[test_key]['iter_perplexity_bigartm'] = model.score_tracker['ps'].value

    with open(os.path.join(TARGET_PATH, '{}.report.json'.format(name)), 'w') as report_file:
        json.dump(report, report_file, indent=2)
