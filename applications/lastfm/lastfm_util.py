
# coding: utf-8

import os
import sys
import csv
import shutil
import uuid
import glob
import time

HOME = '/home/vovapolu/Projects/'
BIGARTM_PATH = HOME + 'bigartm/'
BIGARTM_BUILD_PATH = BIGARTM_PATH + 'build/'
sys.path.append(os.path.join(BIGARTM_PATH, 'src/python'))
os.environ['ARTM_SHARED_LIBRARY'] = os.path.join(BIGARTM_BUILD_PATH, 'src/artm/libartm.so')

import artm.artm_model
from artm.artm_model import *
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

plays_file = 'usersha1-artmbid-artname-plays.tsv'
users = {}

def create_batches(users_to_handle, users_in_batch):
    batch_path = 'batches' #Папка с батчами
    if os.path.exists(batch_path):
        shutil.rmtree(batch_path)

    artist_id_to_name = {} #Мапа, переводящая artist_id в имя
    artists_idxs = {} #Мапа, переводящая artist_id в номер в батче
    artists = [] #Имена артистов в батче

    last_user_id = ''
    handled_users = 0

    batch = None

    with open(plays_file, 'rb') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t', quoting=csv.QUOTE_NONE)

        field = None

        for row in tsvin:

            user_id, artist_id, artist_name, plays = row

            if user_id != last_user_id:
                if handled_users > users_to_handle or handled_users % users_in_batch == 0:
                    if batch is not None:
                        for artist in artists:
                            batch.token.append(artist.decode('utf8'))
                        artm.library.Library().SaveBatch(batch, batch_path)
                        artists = []
                        artists_idxs = {}
                    batch = artm.messages_pb2.Batch()
                    batch.id = str(uuid.uuid4())

                if handled_users > users_to_handle:
                    break

                item = batch.item.add()
                item.id = handled_users
                field = item.field.add()

                if last_user_id != "":
                    plays_sum = 0
                    for artist in users[last_user_id]:
                        plays_sum += users[last_user_id][artist]
                    for artist in users[last_user_id]:
                        users[last_user_id][artist] /= float(plays_sum)
                users[user_id] = {}
                
                last_user_id = user_id
                handled_users += 1

            if artist_id not in artist_id_to_name:
                artist_id_to_name[artist_id] = artist_name
            if artist_id not in artists_idxs:
                artists_idxs[artist_id] = len(artists)
                artists.append(artist_name)
                
            if (artist_name in users[user_id]):
                users[user_id][artist_name] += int(plays)
            else:
                users[user_id][artist_name] = int(plays)

            field.token_id.append(artists_idxs[artist_id])
            field.token_count.append(int(plays))
            
    return batch_path

def create_topic_names(topic_count, background_topic_count):

    background_topics = []
    objective_topics = []
    all_topics = []

    for i in xrange(topic_count):
        topic_name = ("background topic " + str(i)) if i < background_topic_count \
            else "objective topic " + str(i - background_topic_count)
        all_topics.append(topic_name)
        if i < background_topic_count:
            background_topics.append(topic_name)
        else:
            objective_topics.append(topic_name)

    return all_topics, objective_topics, background_topics

def create_model(all_topics, objective_topics, background_topics, batch_path):
    model = ArtmModel(topic_names=all_topics)
    model.num_processors = 4

    # Configure scores

    model.scores.add(SparsityPhiScore(name='ObjectiveSparsityPhiScore', topic_names=objective_topics))
    model.scores.add(SparsityThetaScore(name='ObjectiveSparsityThetaScore', topic_names=objective_topics))
    model.scores.add(SparsityPhiScore(name='BackgroundSparsityPhiScore', topic_names=background_topics))
    model.scores.add(SparsityThetaScore(name='BackgroundSparsityThetaScore', topic_names=background_topics))
    model.scores.add(SparsityThetaScore(name='SparsityThetaScore'))
    model.scores.add(PerplexityScore(name='PerplexityScore'))
    model.scores.add(TopTokensScore(name='TopTokensScore', num_tokens=5))
    model.scores.add(TopTokensScore(name='TopTokensScoreExtended', num_tokens=20))

    # Configure regularizers
    model.regularizers.add(SmoothSparsePhiRegularizer(name='ObjectiveSparsePhi',
                                                      topic_names=objective_topics, tau=-0.1))
    model.regularizers.add(SmoothSparseThetaRegularizer(name='ObjectiveSparseTheta',
                                                        topic_names=objective_topics, tau=-2.0))
    model.regularizers.add(SmoothSparsePhiRegularizer(name='BackgroundSparsePhi',
                                                      topic_names=background_topics, tau=0.1))
    model.regularizers.add(SmoothSparseThetaRegularizer(name='BackgroundSparseTheta',
                                                        topic_names=background_topics, tau=2.0))
    model.regularizers.add(DecorrelatorPhiRegularizer(name='DecorrelatorPhi',
                                                      topic_names=objective_topics, tau=100000.0))

    model.initialize(data_path=batch_path)
    return model, objective_topics, background_topics


def run_model(model, batch_path):
    start = time.clock()
    print "Start fitting..."
    model.fit_offline(data_path=batch_path, num_collection_passes=15)
    print "Fitting tooks %.1f s" % ((finish - start) / 4)


def plot_graphics(model):
    plt.plot(range(model.num_phi_updates), model.scores_info['PerplexityScore'].value, 'r--', linewidth=2)
    plt.xlabel('Iterations count')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.show()

    plt.plot(range(model.num_phi_updates), model.scores_info['ObjectiveSparsityPhiScore'].value, 'b--',
             range(model.num_phi_updates), model.scores_info['ObjectiveSparsityThetaScore'].value, 'r--', linewidth=2)
    plt.xlabel('Iterations count')
    plt.ylabel('Objective Phi sparsity, Theta sparsity')
    plt.grid(True)
    plt.show()

    #plt.plot(range(model.num_phi_updates), model.scores_info['BackgroundSparsityPhiScore'].value, 'b--',
    #         range(model.num_phi_updates), model.scores_info['BackgroundSparsityThetaScore'].value, 'r--', linewidth=2)
    #plt.xlabel('Iterations count')
    #plt.ylabel('Background Phi sparsity, Theta sparsity')
    #plt.grid(True)
    #plt.show()


def print_genres(model, objective_topics, background_topics):
    print "Genres"
    for topic_name in objective_topics:
        print topic_name + ': ',
        print model.scores_info['TopTokensScore'].last_topic_info[topic_name].tokens
    print "Background genres"
    for topic_name in background_topics:
        print topic_name + ': ',
        print model.scores_info['TopTokensScore'].last_topic_info[topic_name].tokens


def find_similar_musicians(model, musician, top_musicians_count, objective_topics):
    top_matches = {}

    for topic in objective_topics:
        topic_musicians = model.scores_info['TopTokensScoreExtended'].last_topic_info[topic].tokens
        weights = model.scores_info['TopTokensScoreExtended'].last_topic_info[topic].weights
        if musician in topic_musicians:
            main_musician_ind = topic_musicians.index(musician)
            for i in xrange(len(topic_musicians)):
                if topic_musicians[i] != musician:
                    if topic_musicians[i] in top_matches:
                        top_matches[topic_musicians[i]] += weights[i] * weights[main_musician_ind]
                    else:
                        top_matches[topic_musicians[i]] = weights[i] * weights[main_musician_ind]

    return sorted([(match[1], match[0]) for match in top_matches.items()], reverse=True)[:top_musicians_count]
