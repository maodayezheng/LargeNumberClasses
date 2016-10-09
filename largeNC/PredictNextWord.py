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


def sgd_update_embed(embed_matrix, outs, index, cost, lr_rate):
    updates = OrderedDict()
    grad = theano.grad(cost, outs)
    updates[embed_matrix] = T.inc_subtensor(outs, - lr_rate * grad)
    return updates


def sgd(params, cost, lr_rate, updates):
    for p, g in zip(params, T.grad(cost, params)):
        updates[p] = p - lr_rate * g
    return updates


def adam_update_embed(embed_matrix, outs, index, cost, lr_rate,
                      beta1=0.9, beta2=0.999, epsilon=1e-6):
    updates = OrderedDict()
    V = embed_matrix.shape[0].eval()
    t = theano.shared(np.ones((V, ), dtype=theano.config.floatX))
    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)
    grad = T.grad(cost, outs)
    # mask = T.neq(index, 0)
    # grad = grad[mask.nonzero()]
    # index = index[mask.nonzero()]

    value = embed_matrix.get_value(borrow=True)
    m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                           broadcastable=embed_matrix.broadcastable)
    v_prev = theano.shared(np.ones(value.shape, dtype=value.dtype),
                           broadcastable=embed_matrix.broadcastable)
    # from theano.printing import Print
    # g = Print("Grad check")(T.stack((T.min(T.abs_(grad)), T.max(T.abs_(grad)))))
    # grad = T.switch(T.lt(0, 1), grad, g[0])
    m_t = beta1 * m_prev[index] + (one - beta1) * grad
    v_t = beta2 * v_prev[index] + (one - beta2) * grad * grad
    # v = Print("V check")(T.stack((T.min(T.abs_(v_t)), T.max(T.abs_(v_t)))))
    # v_t = T.switch(T.lt(0, 1), v_t, v[0])
    a_t = lr_rate * T.sqrt(one - beta2 ** t[index]) / (one - beta1 ** t[index])
    step = a_t.dimshuffle(0, 'x') * m_t / (T.sqrt(v_t) + epsilon)
    # step = grad
    # m_s = Print("m_t shape")(m_t.shape)
    # v_s = Print("v_t shape")(v_t.shape)
    # a_s = Print("a_t shape")(a_t.shape)
    # s_s = Print("step shape")(step.shape)
    # step = T.switch(T.lt(0, 1), step, m_s[0] + v_s[0] + a_s[0] + s_s[0])

    updates[t] = T.inc_subtensor(t[index], one)
    updates[m_prev] = T.set_subtensor(m_prev[index], m_t)
    updates[v_prev] = T.set_subtensor(v_prev[index], v_t)
    updates[embed_matrix] = T.inc_subtensor(embed_matrix[index], - step)
    return updates


def adam(params, cost, lr_rate, updates,
         beta1=0.9, beta2=0.999, epsilon=1e-8):
    t = theano.shared(np.asarray(1.0, theano.config.floatX))
    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    a_t = lr_rate * T.sqrt(one - beta2 ** t) / (one - beta1 ** t)

    for param, g_t in zip(params, T.grad(cost, params)):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1 * m_prev + (one - beta1) * g_t
        v_t = beta2 * v_prev + (one - beta2) * g_t ** 2
        step = a_t * m_t / (T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t] = t + one
    return updates


def make_train_function(sampler, data, embedding_layer, gru, estimator,
                        l_rate_gru, l_rate_embed, optim_method="sgd",
                        lamb=0.0, in_memory=True):
    # Initialise the input nodes
    if in_memory:
        # 1
        batch_start_index = T.iscalar()
        # 1
        batch_end_index = T.iscalar()
        # N x t
        sentence_ids = data[batch_start_index:batch_end_index]
    else:
        # N x t
        sentence_ids = T.imatrix()
    # K
    sample_ids = T.ivector(name="samples")
    sample_qs = sampler.freq_embedding[sample_ids]

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
    tm = target_mask
    ti = target_ids

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

    if optim_method == "sgd":
        updates = sgd_update_embed(embedding_layer.embedding_, all_embed, all_ind, loss, l_rate_embed)
        updates = sgd(gru.get_params(), loss, l_rate_gru, updates)
    elif optim_method == "adam":
        updates = adam_update_embed(embedding_layer.embedding_, all_embed, all_ind, loss, l_rate_embed)
        updates = adam(gru.get_params(), loss, l_rate_gru, updates)
    elif optim_method == "sga":
        updates = sgd_update_embed(embedding_layer.embedding_, all_embed, all_ind, loss, l_rate_embed)
        updates = adam(gru.get_params(), loss, l_rate_gru, updates)
    else:
        raise ValueError("Unrecognized optim_method", optim_method)

    if in_memory:
        return theano.function([batch_start_index, batch_end_index, sample_ids],
                               loss, updates=updates)
    else:
        return theano.function([sentence_ids, sample_ids], loss, updates=updates)


def make_ll_function(sampler, data, embedding_layer, gru, estimator, in_memory=True):
    # Initialise the input nodes
    if in_memory:
        # 1
        batch_start_index = T.iscalar()
        # 1
        batch_end_index = T.iscalar()
        # N x t
        sentence_ids = data[batch_start_index:batch_end_index]
    else:
        # N x t
        sentence_ids = T.imatrix()
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
    if in_memory:
        return theano.function([batch_start_index, batch_end_index], exact_ll)
    else:
        return theano.function([sentence_ids], exact_ll)


def training(estimator_name, folder, sample_size=250, batch_size=100,
             epochs=10, max_len=70,
             embed_dim=100, gru_dim=100, lamb=0.0,
             l_rate_gru=0.05, l_rate_embed=0.05, l_decay=0.9,
             optim_method="sga",
             distortion=1.0, record=100, in_memory=True):
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
    with open(os.path.join(data_folder, "sentences.txt"), 'r') as data:
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
    if in_memory:
        data = theano.shared(np.stack(batch), name="sentences")
        data_shape = data.shape.eval()
    else:
        data = np.stack(batch)
        data_shape = data.shape
    u, c = np.unique(T.flatten(data).eval(), return_counts=True)
    c[0] = 0
    print("Shape of original data:", data_shape, " size in memory: %.2fMB" %
          (float(np.prod(data_shape) * 4.0) / (10.0 ** 6)))
    print("Check u:", u[0], u[-1])
    c = c / np.sum(c)
    with open(os.path.join(data_folder, "frequency.txt"), 'r') as freq:
        p_dist = json.loads(freq.read())

    p = np.asarray(p_dist)
    print("Compare c to p:", np.min(c[1:]), "-", np.min(p[1:]))
    print("Compare c to p:", np.max(c[1:]), "-", np.max(p[1:]))

    if "full" in data_folder.lower():
        print("Taking the last 5000 sentences away")
        if in_memory:
            data.set_value(data[:-100000].eval())
            data_shape = data.shape.eval()
        else:
            data = data[:-100000]
            data_shape = data.shape
        print("Shape of training data:", data_shape, " size in memory: %.2fMB" %
              (float(np.prod(data_shape) * 4.0) / (10.0 ** 6)))
        print("Vocabulary size: %d, min count: %d, max_count: %d" %
              (u.shape[0], np.min(c[1:]), np.max(c[1:])))

    num_classes = u.shape[0]
    sampler = Sampler(num_classes, sample_size,
                      proposed_dist=c,
                      distortion=distortion,
                      extra=estimator.extra)

    # Initialise the word embedding layer
    embedding_layer = EmbeddingLayer(num_classes, embed_dim,
                                     "%s_%d_word_embedding" % (estimator_name, int(100*distortion)))

    # Initialize and compute GRU outputs
    gru = GRU(embed_dim, gru_dim, embed_dim,
              "%s_%d_gru" % (estimator_name, int(100 * distortion)))

    # Make functions
    if in_memory:
        shuffle_func = make_shuffle_function(data)
    # Learning rates
    l_rate_gru = theano.shared(np.asarray(l_rate_gru, dtype=theano.config.floatX))
    l_rate_embed = theano.shared(np.asarray(l_rate_embed, dtype=theano.config.floatX))

    train_func = make_train_function(sampler, data, embedding_layer, gru, estimator,
                                     l_rate_gru, l_rate_embed, optim_method, lamb, in_memory)
    ll_func = make_ll_function(sampler, data, embedding_layer, gru, estimator, in_memory)

    # Make index for shuffling
    N = data_shape[0]
    shuffle_index = np.arange(N, dtype="int32")
    D1 = 10000
    iter = 0
    loss = np.zeros((D1, ), dtype=theano.config.floatX)
    iter_ll = 0
    exact_ll = np.zeros((D1, ), dtype=theano.config.floatX)
    print("Start training")
    start_time = time.time()
    many_samples = None
    for e in range(epochs):
        for i in range(0, N, batch_size):
            if iter % record == 0:
                if iter_ll == exact_ll.shape[0]:
                    exact_ll = np.concatenate((exact_ll, np.zeros((D1,), dtype=theano.config.floatX)), axis=0)
                j = i + batch_size * 10 if (i + 10 * batch_size) < N else N
                if in_memory:
                    exact_ll[iter_ll] = ll_func(j - 10 * batch_size, j)
                else:
                    exact_ll[iter_ll] = ll_func(data[j - 10 * batch_size: j])
                avg_loss = np.mean(loss[iter-record: iter]) if iter >= record else 0
                print("Iteration %d: LL: %.3e, Avg Loss: %.3e, Time: %.2f" %
                      (iter, exact_ll[iter_ll], avg_loss, time.time() - start_time))
                iter_ll += 1
            if iter == loss.shape[0]:
                loss = np.concatenate((loss, np.zeros((D1, ), dtype=theano.config.floatX)), axis=0)
            if iter % 100 == 0:
                many_samples = sampler.draw_sample((100, sampler.num_samples_))
            j = i + batch_size if (i + batch_size) < N else N
            if in_memory:
                loss[iter] = train_func(i, j, many_samples[iter % 100])
            else:
                loss[iter] = train_func(data[i: j], many_samples[iter % 100])
            print(loss[iter])
            iter += 1
        # Update learning rates
        l_rate_gru.set_value((l_rate_gru * l_decay).eval())
        l_rate_embed.set_value((l_rate_embed * l_decay).eval())
        # Shuffle data
        np.random.shuffle(shuffle_index)
        if in_memory:
            shuffle_func(shuffle_index)
        else:
            data = data[shuffle_index]
    loss = loss[:iter]
    exact_ll = exact_ll[:iter_ll]

    # file_prefix = "%s_%d_%d_%d_%d_" % (estimator_name, gru_dim, int(100*distortion),
    #                                    int(1000 * l_rate_gru), int(1000 * l_rate_embed))
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

