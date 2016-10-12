from __future__ import print_function
import json
import numpy as np
import theano.tensor as T
import theano
from theano.printing import Print
from collections import OrderedDict
import os
import time

from .ModelUtils.EmbeddingLayer import EmbeddingLayer
from .ModelUtils.GRU import GRU
from .ModelUtils.Sampler.Sampler import Sampler
from .ModelUtils.Estimator.AlexEstimator import AlexEstimator
from .ModelUtils.Estimator.BernoulliEstimator import BernoulliEstimator
from .ModelUtils.Estimator.ImportanceEstimator import ImportanceEstimator
from .ModelUtils.Estimator.ImportanceTwoEstimator import ImportanceTwoEstimator
from .ModelUtils.Estimator.BlackOutEstimator import BlackOutEstimator
from .ModelUtils.Estimator.NegativeEstimator import NegativeEstimator
from .ModelUtils.Estimator.RankingHinge import RankingHingeEstimator
from .ModelUtils.Estimator.RankingOurs import RankingOursEstimator


def make_shuffle_function(data):
    rand_ind = T.ivector()
    updates = OrderedDict()
    updates[data] = data[rand_ind]
    return theano.function([rand_ind], updates=updates)


def sgd_update_embed(embed_matrix, outs, index, cost, lr_rate, check=False):
    updates = OrderedDict()
    grad = theano.grad(cost, outs)
    if check:
        g = theano.shared(np.zeros((2, ), dtype=theano.config.floatX))
        updates[g] = Print("Grad embed:")(T.stack((T.min(T.abs_(grad)), T.max(T.abs_(grad)))))
    updates[embed_matrix] = T.inc_subtensor(outs, - lr_rate * grad)
    return updates


def sgd(params, cost, lr_rate, updates, check=False):
    for p, grad in zip(params, T.grad(cost, params)):
        if check:
            g = theano.shared(np.zeros((2,), dtype=theano.config.floatX))
            updates[g] = Print("Grad " + p.name + ":")(T.stack((T.min(T.abs_(grad)), T.max(T.abs_(grad)))))
        updates[p] = p - lr_rate * grad
    return updates


def adam_update_embed(embed_matrix, outs, index, cost, lr_rate,
                      beta1=0.9, beta2=0.999, epsilon=1e-6, check=False):
    updates = OrderedDict()
    V = embed_matrix.shape[0].eval()
    t = theano.shared(np.ones((V, ), dtype=theano.config.floatX))
    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)
    grad = T.grad(cost, outs)
    if check:
        g = theano.shared(np.zeros((2,), dtype=theano.config.floatX))
        updates[g] = Print("Grad embed:")(T.stack((T.min(T.abs_(grad)), T.max(T.abs_(grad)))))
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
         beta1=0.9, beta2=0.999, epsilon=1e-8, check=False):
    t = theano.shared(np.asarray(1.0, theano.config.floatX))
    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    a_t = lr_rate * T.sqrt(one - beta2 ** t) / (one - beta1 ** t)

    for p, grad in zip(params, T.grad(cost, params)):
        if check:
            g = theano.shared(np.zeros((2,), dtype=theano.config.floatX))
            updates[g] = Print("Grad " + p.name + ":")(T.stack((T.min(T.abs_(grad)), T.max(T.abs_(grad)))))

        value = p.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=p.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=p.broadcastable)

        m_t = beta1 * m_prev + (one - beta1) * grad
        v_t = beta2 * v_prev + (one - beta2) * grad ** 2
        step = a_t * m_t / (T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[p] = p - step

    updates[t] = t + one
    return updates


def make_train_function(embedding_layer, freq_param, gru, estimator,
                        l_rate_gru, l_rate_embed, optim_method="sgd",
                        lamb=0.0, optimize_q=False, check=False):
    freq_embed = T.nnet.softmax(freq_param)[0, :]
    # Initialise the input nodes
    # U
    unique_ids = T.ivector("unique_words")
    # U
    unique_qs = freq_embed[unique_ids]
    # N x t
    sentence_sub_ids = T.imatrix("sentences")
    # U x D
    unique_embed = embedding_layer(unique_ids)
    # N x t x D
    sent_embed = unique_embed[sentence_sub_ids]
    # N x t
    N = sentence_sub_ids.shape[0]
    t = sentence_sub_ids.shape[1]
    U = unique_ids.shape[0]
    D = embedding_layer.dim

    # N x (t - 1) x D
    h = gru(sent_embed[:, :-1]).reshape((N * (t - 1), D))
    # N x (t - 1) x D
    targets = sent_embed[:, 1:].reshape((N * (t - 1), D))
    # N x (t - 1)
    target_sub_ids = T.flatten(sentence_sub_ids[:, 1:])
    # N x (t - 1) - True whenever we have a target word
    target_mask = T.neq(sentence_sub_ids[:, 1:], 0).flatten()

    # Mask them
    h = h[target_mask.nonzero()]
    targets = targets[target_mask.nonzero()]
    target_sub_ids = target_sub_ids[target_mask.nonzero()]

    loss, logvar = estimator.loss(h, targets, target_sub_ids, unique_embed, unique_ids, unique_qs)
    if lamb > 0.0:
        print("Adding L2 Regularization: %.3f" % lamb)
        loss += lamb * gru.l2_regular()

    if optim_method == "sgd":
        updates = sgd_update_embed(embedding_layer.embedding_, unique_embed, unique_ids,
                                   loss, l_rate_embed, check=check)
        updates = sgd(gru.get_params(), loss, l_rate_gru, updates, check=check)
    elif optim_method == "adam":
        updates = adam_update_embed(embedding_layer.embedding_, unique_embed, unique_ids,
                                   loss, l_rate_embed, check=check)
        updates = adam(gru.get_params(), loss, l_rate_gru, updates, check=check)
    elif optim_method == "sga":
        updates = sgd_update_embed(embedding_layer.embedding_, unique_embed, unique_ids,
                                   loss, l_rate_embed, check=check)
        updates = adam(gru.get_params(), loss, l_rate_gru, updates, check=check)
    else:
        raise ValueError("Unrecognized optim_method", optim_method)

    if optimize_q:
        updates[freq_param] = freq_param - l_rate_embed * T.grad(logvar, freq_param)

    return theano.function([unique_ids, sentence_sub_ids], [loss, logvar, unique_ids.shape[0]], updates=updates)


def make_ll_function(embedding_layer, freq_param, gru, estimator):
    freq_embed = T.nnet.softmax(freq_param)[0, :]
    # Initialise the input nodes
    # N x t
    sentence_ids = T.imatrix()
    # Helper vars
    N = sentence_ids.shape[0]
    t = sentence_ids.shape[1]
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
    target_qs = freq_embed[target_ids]

    exact_ll = estimator.log_likelihood(embedding_layer.embedding_, freq_embed,
                                        h, target_ids, target_qs)
    return theano.function([sentence_ids], exact_ll)


def training(estimator_name, folder, batch_size=100,
             epochs=30, max_len=70, embed_dim=100, gru_dim=128, lamb=0.0,
             l_rate_gru=0.02, l_rate_embed=0.02, l_decay=0.9, optimize_q=False,
             optim_method="sgd", check=False, record=100):
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
    elif estimator_name == "RH":
        estimator = RankingHingeEstimator()
    elif estimator_name == "RO":
        estimator = RankingOursEstimator()
    elif estimator_name == "IST":
        estimator = ImportanceTwoEstimator()
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
    data = np.stack(batch).astype("int32")
    data_shape = data.shape
    u, c = np.unique(data, return_counts=True)
    c[0] = np.min(c[1:])
    print("Shape of original data:", data_shape, " size in memory: %.2fMB" %
          (float(np.prod(data_shape) * 4.0) / (10.0 ** 6)))
    print("Check u:", u[0], u[-1])
    freq_param = theano.shared(np.log(c).astype(theano.config.floatX), name="freq_embed")
    if "full" in data_folder.lower():
        print("Taking the last 100,000 sentences away")
        test_data = data[-100000:]
        test_shape = test_data.shape
        data = data[:-100000]
        data_shape = data.shape
        print("Shape of training data:", data_shape, " size in memory: %.2fMB" %
              (float(np.prod(data_shape) * 4.0) / (10.0 ** 6)))
        print("Vocabulary size: %d, min count: %d, max_count: %d" %
              (u.shape[0], np.min(c[1:]), np.max(c[1:])))
    else:
        test_data = data
        test_shape = data_shape

    num_classes = u.shape[0]
    # Initialise the word embedding layer
    embedding_layer = EmbeddingLayer(num_classes, embed_dim,
                                     "%s_%d_word_embedding" % (estimator_name, int(100)))

    # Initialize and compute GRU outputs
    gru = GRU(embed_dim, gru_dim, embed_dim,
              "%s_%d_gru" % (estimator_name, int(100 * 1)))

    # Learning rates
    l_rate_gru = theano.shared(np.asarray(l_rate_gru, dtype=theano.config.floatX))
    l_rate_embed = theano.shared(np.asarray(l_rate_embed, dtype=theano.config.floatX))
    train_func = make_train_function(embedding_layer, freq_param, gru, estimator,
                                     l_rate_gru, l_rate_embed, optim_method,
                                     lamb, optimize_q, check=check)
    ll_func = make_ll_function(embedding_layer, freq_param, gru, estimator)

    # Make index for shuffling
    N = data_shape[0]
    NT = test_shape[0]
    shuffle_index = np.arange(N, dtype="int32")
    D1 = 10000
    iter = 0
    loss = np.zeros((D1, 3), dtype=theano.config.floatX)
    iter_ll = 0
    exact_ll = np.zeros((D1, ), dtype=theano.config.floatX)
    exact_ll_full = np.zeros((epochs+1, ), dtype=theano.config.floatX)
    test_ll_full = np.zeros((epochs+1, ), dtype=theano.config.floatX)
    print("Start training")
    start_time = time.time()
    be = 1000
    for e in range(epochs):
        # if not check:
        #     # Calculate exact LL
        #     for i in range(0, (N // be) * be, be):
        #         exact_ll_full[e] += ll_func(data[i, i + be])
        #     exact_ll_full[e] /= N // be
        #     print("Exact full LL for %d epoch: %.3e, Time: %.2f" % (e, exact_ll_full[e], time.time() - start_time))
        #     # Calculate exact Test LL
        #     for i in range(0, (NT // be) * be, be):
        #         test_ll_full[e] += ll_func(test_data[i, i + be])
        #     test_ll_full[e] /= NT // be
        #     print("Exact test LL for %d epoch: %.3e, Time: %.2f" % (e, test_ll_full[e], time.time() - start_time))
        for i in range(0, N, batch_size):
            if iter % record == 0:
                if iter_ll == exact_ll.shape[0]:
                    exact_ll = np.concatenate((exact_ll, np.zeros((D1,), dtype=theano.config.floatX)), axis=0)
                j = i + be if (i + be) < N else N
                exact_ll[iter_ll] = ll_func(data[j - be: j])
                avg_loss = np.mean(loss[iter - record: iter, 0]) if iter >= record else 0
                avg_var = np.mean(loss[iter - record: iter, 1]) if iter >= record else 0
                print("Iteration %d: LL: %.3e, Avg Loss: %.3e, Avg Var: %.3e, Time: %.2f" %
                      (iter, exact_ll[iter_ll], avg_loss, avg_var, time.time() - start_time))
                iter_ll += 1
            if iter == loss.shape[0]:
                loss = np.concatenate((loss, np.zeros((D1, 3), dtype=theano.config.floatX)), axis=0)
            j = i + batch_size if (i + batch_size) < N else N
            batch = data[i: j]
            unique_id, inverse_index = np.unique(batch, return_inverse=True)
            loss[iter] = train_func(unique_id.astype("int32"), inverse_index.reshape(batch.shape).astype("int32"))
            print(loss[iter])
            iter += 1
        print("Epoch finished")
        # Update learning rates
        l_rate_gru.set_value((l_rate_gru * l_decay).eval())
        l_rate_embed.set_value((l_rate_embed * l_decay).eval())
        # Shuffle data
        np.random.shuffle(shuffle_index)
        data = data[shuffle_index]
    if not check:
        # Calculate exact LL
        for i in range(0, (N // be) * be, be):
            exact_ll_full[epochs] += ll_func(data[i, i + be])
        exact_ll_full[epochs] /= N // be
        print("Exact full LL for %d epoch: %.3e, Time: %.2f" %
              (epochs, exact_ll_full[epochs], time.time() - start_time))
        # Calculate exact Test LL
        for i in range(0, (NT // be) * be, be):
            test_ll_full[epochs] += ll_func(test_data[i, i + be])
        test_ll_full[epochs] /= NT // be
        print("Exact test LL for %d epoch: %.3e, Time: %.2f" %
              (epochs, test_ll_full[epochs], time.time() - start_time))
    # Cut of what is not needed
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
    file_name = os.path.join(folder, file_prefix + "ll_full.csv")
    print("Saving full LL to", file_name, "with shape", exact_ll_full.shape)
    np.savetxt(file_name, exact_ll_full, delimiter=",")
    file_name = os.path.join(folder, file_prefix + "ll_test.csv")
    print("Saving test LL to", file_name, "with shape", test_ll_full.shape)
    np.savetxt(file_name, test_ll_full, delimiter=",")
    file_name = os.path.join(folder, file_prefix + "params.npz")
    params = [embedding_layer.embedding_] + gru.get_params()
    pd = OrderedDict()
    for p in params:
        pd[p.name] = p.get_value()
    print("Saving parameters to", file_name, list(pd.keys()))
    np.savez(file_name, **pd)

