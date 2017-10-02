# -*- coding: utf-8 -*-

import os
import sys
import time

import onlinelda


if __name__ == '__main__':

    num_topics = 50

    alpha = 1.0 / num_topics
    beta = 1.0 / num_topics 
    kappa = 0.5
    tau0 = 64
    batch_size = 10000
    test_batch_size = 100000  # all test documents should be in one batch
    num_document_passes = 5


    num_processors = 1
    update_every = num_processors

########################################################################################################################

    print '\nStart vw-lda...\n'
    time_start = time.time()
    name = 'wiki_vw_threads({})_batch({})_update({})_topics({})_doc_passes({})_tau0({})'.format(
        num_processors,
        batch_size,
        update_every,
        num_topics,
        num_document_passes,
        tau0
    )

    if not os.path.exists(os.path.join('target', '{}.report.json'.format(name))):
        onlinelda.run_vw(
            name=name,
            train='wiki-sample_train',
            test={'test': 'wiki-sample_test'},
            wordids='wiki-sample.mm-vocab.txt',
            num_processors=num_processors,
            num_topics=num_topics, alpha=alpha, beta=beta,
            batch_size=batch_size, update_every=update_every,
            num_document_passes=num_document_passes,
            kappa=kappa, tau0=tau0
        )

    print '\nFinished, elapsed {} sec'.format(time.time() - time_start)

########################################################################################################################
