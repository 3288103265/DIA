
import argparse
import glob
import os
import struct

import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', help='Choose feature aggregating method.',
                    choices=['bow', 'vlad'], type=str, required=True)
args = parser.parse_args()

# def get_features(path):
#     # read file and return list of sift vectors.
#     with open(path, 'rb') as f:
#         header = f.read(4)
#         body = f.read()

#     n_feats = struct.unpack('@i', header)
#     feats = struct.iter_unpack('@128B4f', body)

#     feat_list = [item[:-4] for item in feats]
#     assert len(feat_list) == n_feats[0]

#     return feat_list


class Img(object):

    # Img defined by sift descriptors

    def __init__(self, path):
        with open(path, 'rb') as f:
            header = f.read(4)
            body = f.read()

        n_feats = struct.unpack('@i', header)
        feats = struct.iter_unpack('@128B4f', body)

        feat_list = [item[:-4] for item in feats]
        assert len(feat_list) == n_feats[0]

        self.shape = np.array(feat_list).shape
        self.feats = np.array(feat_list)


def get_codebook(feats, k_cluster):
    # features and return list of cluster center.
    # return kmeans is better.:)
    from sklearn.cluster import KMeans, MiniBatchKMeans

    if k_cluster < 5000:
        kmeans = KMeans(n_clusters=k_cluster,
                        random_state=0, n_jobs=-1).fit(feats)
    else:
        kmeans = MiniBatchKMeans(k_cluster, max_iter=200, random_state=0,
                                 init_size=3*k_cluster).fit(feats)
    # TODO: how to visualise process to see if cluster cover.

    return kmeans


def get_bow(sample, codebook, norm='l1'):
    # sanples is a list of features of one img
    # codebook, kmeans model, containing cluster center vecter.
    # return bags of words vector
    label = codebook.predict(sample)
    k_cluster = codebook.cluster_centers_.shape[0]
    bow = np.zeros(k_cluster)

    from collections import Counter
    counts = Counter(label)

    for key, value in counts.items():
        bow[key] = value

    if norm == 'l1':
        l1_norm = len(label)
        bow = bow/l1_norm
    else:
        l2_norm = np.linalg.norm(bow)
        bow = bow/l2_norm

    return bow


def get_vlad(sample, codebook, norm='l2'):
    # return vector of locally aggregated descriptor
    k_cluster = codebook.cluster_centers_.shape[0]

    label = codebook.predict(sample)
    df = pd.DataFrame(sample)
    df['label'] = label
    label_counts = df['label'].value_counts()
    df = df.groupby('label').sum()

    vlad = []

    for i in range(k_cluster):
        if i not in df.index:
            padding = np.zeros(sample.shape[1])
            vlad.append(padding)
        else:
            v = df.loc[i] - label_counts.loc[i]*codebook.cluster_centers_[i]
            vlad.append(v.values)

    vlad = np.hstack(vlad)
    assert len(vlad) == k_cluster*sample.shape[1]
    if norm == 'l2':
        l2_norm = np.linalg.norm(vlad)
        vlad = vlad/l2_norm
    else:
        l1_norm = np.linalg.norm(vlad, ord=1)
        vlad = vlad/l1_norm

    return vlad


def similarity_matrix(database):
    from numpy.linalg import norm
    # return socre matrix S. S[i,j] means score of i and j
    score = np.zeros((len(database), len(database)))
    for i in range(len(database)):
        # cosine similarity
        # score[i] = [database[i].dot(each) for each in database]
        # # 
        # # intersection of histgrom
        # score[i] = [np.minimum(database[i], each).sum() for each in database]
        # # 
        # # l1 distance
        # score[i] = [-norm(database[i]-each, ord=1) for each in database]

        # l2 distance
        score[i] = [-norm(database[i]-each) for each in database]
        
        # # new distance: intersection and l2
        # score[i] = [np.minimum(database[i], each).sum(
        # )-norm(database[i]-each) for each in database]

    # import pickle
    # with open('score_matrix.pkl', 'wb') as f:
    #     pickle.dump(similarity_matrix, f)

    return score


# TODO:should input source image/ dsift for real application
# def retrive(sample_index, similarity_matrix):
#     # import sample index and score_matrix
#     # res = [i,res1, res2, res3]
#     return sample_index,np.argpartition(similarity_matrix[sample_index],-4, axis=1)[:,-4:]


def evalute(similarity_matrix):

    top4 = np.argpartition(similarity_matrix, -4, axis=1)[:, -4:]

    precision = [score(i, row) for i, row in enumerate(top4)]
    mean_precision = np.array(precision).mean()
    return mean_precision


def score(index, res):
    # index and res[]
    # calculate precision
    return sum(res//4 == index//4)


if __name__ == "__main__":

    # name should be sorted.
    feats_path = sorted(glob.glob('jpg.dsift/*.dsift'))
    print('Loading...')
    all_samples = [Img(path).feats for path in tqdm(feats_path)]
    all_feats = np.vstack(all_samples)

    print('Number of features: {}'.format(len(all_feats)))

    if args.method == 'bow':
        bow_or_vlad = get_bow
        k_cluster = [1000, 2000, 5000, 10000, 20000]
        norms = ['l1', 'l2']
    else:
        bow_or_vlad = get_vlad
        k_cluster = [8, 16, 32, 64]
        norms = ['l1','l2']

    res = []
    np.random.seed(1000)

    for normi in norms:
        for k in k_cluster:
            print('(Norm, N_cluster) = ({}, {})'.format(normi, k))
            sample_feats = all_feats[np.random.randint(
                all_feats.shape[0], size=10*k)]
            print('Clustering...')

            codebook = get_codebook(sample_feats, k)
            print('Generating {}...'.format(args.method))
            database = np.array([bow_or_vlad(sample, codebook, norm=normi)
                                 for sample in tqdm(all_samples, ascii=True)])
            print('Evaluting...')

            s_matrix = similarity_matrix(database)
            mp = evalute(s_matrix)
            res.append(mp)
            print('Average precision: {}'.format(mp))
    res_df = pd.DataFrame(np.array(res).reshape(
        len(norms), len(k_cluster)), index=norms, columns=k_cluster)
    print(res_df)
    res_df.to_csv('{}_result.csv'.format(args.method), index=0)
