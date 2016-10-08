from __future__ import print_function
import json
import numpy as np
import theano.tensor as T
import theano
from collections import OrderedDict
import os
import time

from .ModelUtils.EmbeddingLayer import EmbeddingLayer
from .ModelUtils.GRU import GRU
from .ModelUtils.Sampler.Sampler import Sampler
from .ModelUtils.Estimator.AlexEstimator import AlexEstimator
from .ModelUtils.Estimator.BernoulliEstimator import BernoulliEstimator
from .ModelUtils.Estimator.ImportanceEstimator import ImportanceEstimator
from .ModelUtils.Estimator.BlackOutEstimator import BlackOutEstimator
from .ModelUtils.Estimator.NegativeEstimator import NegativeEstimator


def make_shuffle_function(data):
    rand_ind = T.ivector()
    updates = OrderedDict()
    updates[data] = data[rand_ind]
    return theano.function([rand_ind], updates=updates)


def make_train_function(sampler, data, embedding_layer, gru, estimator,
                        l_rate_gru, l_rate_embed, lamb=0.0):
    # Initialise the input nodes
    # 1
    batch_start_index = T.iscalar()
    # 1
    batch_end_index = T.iscalar()
    # K
    sample_ids = T.ivector(name="samples")
    sample_qs = sampler.freq_embedding[sample_ids]
    # N x t
    sentence_ids = T.imatrix()
    # sentence_ids = data[batch_start_index:batch_end_index]
    # Helper vars
    N = sentence_ids.shape[0]
    t = data.shape[1]
    K = sample_ids.shape[0]
    D = embedding_layer.dim

    # (N*T+K) x D
    all_ind = T.concatenate((sample_ids, T.flatten(sentence_ids)), axis=0)
    all_embed = embedding_layer(all_ind)

    # K x D
    samples = all_embed[:K]
    # N x t x D
    sent_embed = T.reshape(all_embed[K:], (N, t, D))

    # N x (t - 1) x D
    h = gru(sent_embed[:, :-1])
    # N x (t - 1) x D
    targets = sent_embed[:, 1:]
    # N x (t - 1)
    target_ids = sentence_ids[:, 1:]
    # N x (t - 1) - True whenever we have a target word
    target_mask = T.neq(sentence_ids[:, 1:], 0)

    # Flatten the targets
    h = h.reshape((N * (t - 1), D))
    targets = targets.reshape((N * (t - 1), D))
    target_ids = T.flatten(target_ids)
    target_mask = T.flatten(target_mask)

    # Mask them
    h = h[target_mask.nonzero()]
    targets = targets[target_mask.nonzero()]
    target_ids = target_ids[target_mask.nonzero()]
    target_qs = sampler.freq_embedding[target_ids]

    loss = estimator.loss(h, targets, target_ids, target_qs,
                          samples, sample_ids, sample_qs)
    if lamb > 0.0:
        print("Adding L2 Regularization: %.3f" % lamb)
        loss += lamb * gru.l2_regular()

    grads = theano.grad(loss, [all_embed] + gru.get_params())
    updates = OrderedDict()
    for p, g in zip(gru.get_params(), grads[1:]):
        updates[p] = p - l_rate_gru * g
    updates[embedding_layer.embedding_] = T.inc_subtensor(all_embed, - l_rate_embed * grads[0])

    # return theano.function([batch_start_index, batch_end_index, sample_ids], loss, updates=updates)
    return theano.function([sentence_ids, sample_ids], loss, updates=updates)


def make_ll_function(sampler, data, embedding_layer, gru, estimator):
    # Initialise the input nodes
    # 1
    batch_start_index = T.iscalar()
    # 1
    batch_end_index = T.iscalar()
    # N x t
    sentence_ids = T.imatrix()
    # sentence_ids = data[batch_start_index:batch_end_index]
    # Helper vars
    N = sentence_ids.shape[0]
    t = data.shape[1]
    D = embedding_layer.dim

    # (N*T+K) x D
    all_ind = T.flatten(sentence_ids)
    all_embed = embedding_layer(all_ind)

    # N x t x D
    sent_embed = T.reshape(all_embed, (N, t, D))

    # N x (t - 1) x D
    h = gru(sent_embed[:, :-1])
    # N x (t - 1)
    target_ids = sentence_ids[:, 1:]
    # N x (t - 1) - True whenever we have a target word
    target_mask = T.neq(sentence_ids[:, 1:], 0)

    # Flatten the targets
    h = h.reshape((N * (t - 1), D))
    target_ids = T.flatten(target_ids)
    target_mask = T.flatten(target_mask)

    # Mask them
    h = h[target_mask.nonzero()]
    target_ids = target_ids[target_mask.nonzero()]
    target_qs = sampler.freq_embedding[target_ids]

    exact_ll = estimator.log_likelihood(embedding_layer.embedding_, sampler.freq_embedding,
                                        h, target_ids, target_qs)
    # return theano.function([batch_start_index, batch_end_index], exact_ll)
    return theano.function([sentence_ids], exact_ll)


def training(estimator_name, folder, sample_size=250, batch_size=100,
             epochs=10, max_len=70,
             embed_dim=100, gru_dim=100, lamb=0.0,
             l_rate_gru=0.02, l_rate_embed=0.02,
             distortion=1.0, record=100):
    """
    The RNN model to predict next word

    @Param params: A dictionary of training hyper parameters
    """
    print("Running the %s estimator test" % estimator_name)

    if os.environ.get("LARGE_CLASS_FOLDER") is not None:
        data_folder = os.environ["LARGE_CLASS_FOLDER"]
    else:
        data_folder = "ProcessedData"

    # Set the estimator
    if estimator_name == "BER":
        estimator = BernoulliEstimator()
    elif estimator_name == "IMP":
        estimator = ImportanceEstimator()
    elif estimator_name == "NEG":
        estimator = NegativeEstimator()
    elif estimator_name == "BLA":
        estimator = BlackOutEstimator()
    elif estimator_name == "ALEX":
        estimator = AlexEstimator()
    else:
        raise Exception("{} type estimator is not supported".format(estimator_name))

    # Load data
    batch = []
    with open(os.path.join(data_folder, "sentences_100000.txt"), 'r') as data:
        for d in data:
            d = json.loads(d)
            if len(d) > max_len:
                v = np.asarray(d[:max_len], dtype="int32")
                batch.append(v)
            else:
                v = np.zeros((max_len,), dtype="int32")
                v[:len(d)] = np.asarray(d, dtype="int32")
                batch.append(v)
        data.close()
    data = np.stack(batch)
    # data = theano.shared(batch, name="sentences")
    print("Shape of data:", data.shape, " size in memory: %.2fMB" %
          (float(np.prod(data.shape) * 4.0) / (10.0 ** 6)))

    # Initialise sampler
    with open(os.path.join(data_folder, "frequency_100000.txt"), 'r') as freq:
        p_dist = json.loads(freq.read())
        num_classes = len(p_dist)
        sampler = Sampler(num_classes, sample_size,
                          proposed_dist=p_dist,
                          distortion=distortion,
                          extra=estimator.extra)
        freq.close()

    # Initialise the word embedding layer
    embedding_layer = EmbeddingLayer(num_classes, embed_dim,
                                     "%s_%d_word_embedding" % (estimator_name, int(100*distortion)))

    # Initialize and compute GRU outputs
    gru = GRU(embed_dim, gru_dim, embed_dim,
              "%s_%d_gru" % (estimator_name, int(100 * distortion)))

    # Make functions
    # shuffle_func = make_shuffle_function(data)
    train_func = make_train_function(sampler, data, embedding_layer, gru, estimator,
                                     l_rate_gru, l_rate_embed, lamb)
    ll_func = make_ll_function(sampler, data, embedding_layer, gru, estimator)

    # Make index for shuffling
    N = data.shape[0]
    shuffle_index = np.arange(N, dtype="int32")
    D1 = 10000
    iter = 0
    loss = np.zeros((D1, ), dtype=theano.config.floatX)
    iter_ll = 0
    exact_ll = np.zeros((D1, ), dtype=theano.config.floatX)
    start_time = time.time()
    many_samples = None
    for e in range(epochs):
        for i in range(0, N, batch_size):
            if iter % record == 0:
                if iter_ll == exact_ll.shape[0]:
                    exact_ll = np.concatenate((exact_ll, np.zeros((D1,), dtype=theano.config.floatX)), axis=0)
                j = i + batch_size * 10 if (i + 10 * batch_size) < N else N
                exact_ll[iter_ll] = ll_func(data[j - 10 * batch_size: j])
                # exact_ll[iter_ll] = ll_func(j - 10 * batch_size, j)
                avg_loss = np.mean(loss[iter-record: iter]) if iter >= record else 0
                print("Iteration %d: LL: %.3e, Avg Loss: %.3e, Time: %.2f" %
                      (iter, exact_ll[iter_ll], avg_loss, time.time() - start_time))
                iter_ll += 1
            if iter == loss.shape[0]:
                loss = np.concatenate((loss, np.zeros((D1, ), dtype=theano.config.floatX)), axis=0)
            if iter % 100 == 0:
                many_samples = sampler.draw_sample((100, sampler.num_samples_))
            j = i + batch_size if (i + batch_size) < N else N
            loss[iter] = train_func(data[i: j], many_samples[iter % 100])
            # loss[iter] = train_func(i, j, many_samples[iter % 100])
            iter += 1
        # Shuffle data
        np.random.shuffle(shuffle_index)
        data = data[shuffle_index]
        print(data.shape)
        # shuffle_func(shuffle_index)
    loss = loss[:iter]
    exact_ll = exact_ll[:iter_ll]

    file_prefix = "%s_%d_%d_%d_%d_" % (estimator_name, gru_dim, int(100*distortion),
                                       int(1000 * l_rate_gru), int(1000 * l_rate_embed))
    file_prefix = ""
    print("Total time: %.2f" % (time.time() - start_time))
    file_name = os.path.join(folder, file_prefix + "loss.csv")
    print("Saving loss to", file_name, "with shape", loss.shape)
    np.savetxt(file_name, loss, delimiter=",")
    file_name = os.path.join(folder, file_prefix + "ll.csv")
    print("Saving LL to", file_name, "with shape", exact_ll.shape)
    np.savetxt(file_name, exact_ll, delimiter=",")
    file_name = os.path.join(folder, file_prefix + "params.npz")
    params = [embedding_layer.embedding_] + gru.get_params()
    pd = OrderedDict()
    for p in params:
        pd[p.name] = p.get_value()
    print("Saving parameters to", file_name, list(pd.keys()))
    np.savez(file_name, **pd)

