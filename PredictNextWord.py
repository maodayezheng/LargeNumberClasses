from __future__ import print_function
import json
import random
import numpy as np
import tensorflow as tf
from ModelUtils.EmbeddingLayer import EmbeddingLayer
from ModelUtils.GRU import GRU
from ModelUtils.Sampler.UnigramSampler import UnigramSampler
from ModelUtils.Estimator.AlexEstimator import AlexEstimator
from ModelUtils.Estimator.BernoulliEstimator import BernoulliEstimator
from ModelUtils.Estimator.ImportanceEstimator import ImportanceEstimator
from ModelUtils.Estimator.BlackOutEstimator import BlackOutEstimator
from ModelUtils.Estimator.NegativeEstimator import NegativeEstimator


def training(params):
    """
    The RNN model to predict next word

    @Param params: A dictionary of training hyper parameters
    """

    # Get training hyper parameters
    sampler_type = params["sampler_type"]
    estimator_type = params["estimator_type"]
    sample_size = params["sample_size"]
    batch_size = params["batch_size"]
    sentence_len = params["sentence_len"]
    epoch = params["epoch"]
    save_step = params["save_step"]
    input_dim = params["input_dim"]
    hidden_dim = params["hidden_dim"]
    output_dim = params["output_dim"]
    lamb = params["lamb"]
    l_rate = params["l_rate"]
    distortion = params['distortion']
    print("Runing the "+estimator_type + "Estimator Test")
    num_classes = 0
    # Initialise the input nodes
    inputs = []
    for i in range(batch_size):
        s = tf.placeholder(tf.int64, shape=None, name="sentence_{}".format(i))
        inputs.append(s)
    decay = tf.placeholder(tf.int64,shape=None, name="decay_step")
    # Initialise sampler and loss estimator
    sampler = None
    with open("ProcessedData/frequency.txt", 'r') as freq:
            p_dist = json.loads(freq.read())
            num_classes = len(p_dist)
            sampler = UnigramSampler(num_classes, sample_size, proposed_dist=p_dist, distortion=distortion)
            freq.close()
    estimator = None
    if estimator_type is "BER":
        estimator = BernoulliEstimator(sampler)
    elif estimator_type is "IMP":
        estimator = ImportanceEstimator(sampler)
    elif estimator_type is "NEG":
        estimator = NegativeEstimator(sampler)
    elif estimator_type is "BLA":
        estimator = BlackOutEstimator(sampler)
    elif estimator_type is "ALEX":
        estimator = AlexEstimator(sampler)
    else:
        raise Exception("{} type estimator is not support".format(estimator_type))

    # Initialise the word embedding layer
    word_embedding = EmbeddingLayer(num_classes, input_dim, sampler_type+estimator_type+str(distortion)+"word")

    # Reshape the input sentences
    sentences = tf.squeeze(tf.pack(inputs))
    sentences = tf.split(1, sentence_len, sentences)
    l = len(sentences)

    # Initialise Recurrent network
    cell = GRU(input_dim, hidden_dim, output_dim, sampler_type+estimator_type+str(distortion))
    init_state = tf.zeros([batch_size, hidden_dim], dtype=tf.float32, name="init_state")
    state = init_state
    embedding = word_embedding.get_embedding()
    target_states = []
    target_words = []
    for i in range(l):
        word = word_embedding(sentences[i])
        state, output = cell(word, state)
        if 0 < i:
            target_words.append(word)
        if i < l-1:
            target_states.append(state)

    # Masking the parameters
    target_states = tf.concat(0, target_states)
    target_words = tf.concat(0, target_words)

    # Draw samples
    targets = tf.concat(0, sentences[1:])
    masks = tf.reshape(tf.squeeze(tf.not_equal(targets, 0)), [batch_size*(l-1)])
    targets = tf.boolean_mask(targets, masks)
    target_states = tf.boolean_mask(target_states, masks)
    target_words = tf.boolean_mask(target_words, masks)
    ss, tc, sc = estimator.draw_samples(targets, 1)
    estimator.set_sample_weights(sc)
    estimator.set_sample(word_embedding(ss))
    # Estimate loss
    loss = tf.check_numerics(estimator.loss(target_words, target_states, q=tc), message="The loss is ")
    exact_log_like = estimator.log_likelihood(targets, target_states, embedding)

    # Training Loss
    objective = loss + cell.l2_regular() * lamb
    l_rate = tf.train.exponential_decay(l_rate, decay, 1, 0.9)
    update = tf.train.GradientDescentOptimizer(l_rate).minimize(objective)

    # Initialise Variables
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init = tf.initialize_all_variables()
    session.run(init)

    # Get the training batch
    print("Start Training")
    batch = []
    with open('ProcessedData/sentences.txt', 'r') as data:
        for d in data:
            d = json.loads(d)
            if len(d) > sentence_len:
                continue
            batch.append(d)
        data.close()

    print("Finished getting batch")
    input_dict = {}
    iteration = 0
    exact_log_like_save = []
    average_loss_save = []

    batch = batch[0:(len(batch)-5000)]
    data_len = len(batch)
    start_pos = 0
    end_pos = start_pos + batch_size
    epoch_count = 0
    mini_batch = batch[start_pos:end_pos]
    aver_loss = 0.0
    check_point = 1
    print("Start epoch {}".format(epoch_count + 1))
    while epoch_count < epoch:
        iteration += 1
        for i in range(batch_size):
            d = mini_batch[i]
            d = [0] * (sentence_len - len(d)) + d
            input_dict[inputs[i].name] = d
        input_dict[decay.name] = epoch_count
        if iteration % save_step is 0:
            _, exact, aprox_l = session.run([update, exact_log_like, loss], feed_dict=input_dict)
            exact_log_like_save.append(exact)
            aver_loss += aprox_l
            aver_loss /= 10.0
            average_loss_save.append(aver_loss)
            check_point += 1
            print(estimator_type+" " + sampler_type + " " + str(distortion) +
                  " At iteration {}, the average estimate loss is {}, the exact log like is {}"
                  .format(iteration, aver_loss, exact))
            aver_loss = 0.0
        else:
            _, aprox_l = session.run([update, loss], feed_dict=input_dict)
            if save_step * check_point - iteration < 10:
                 aver_loss += aprox_l

        start_pos = end_pos + 1
        end_pos = start_pos + batch_size


        # Reset batch
        if end_pos > data_len:
            epoch_count += 1
            print("Start epoch {}".format(epoch_count + 1))
            mini_batch = batch[start_pos:]
            end_pos = end_pos % data_len
            mini_batch += batch[0:end_pos]
        else:
            mini_batch = batch[start_pos:end_pos]

    word_embedding.save_param(session, "ModelParams/")
    cell.save_param(session, "ModelParams/")
    np.savetxt("ModelParams/" + sampler_type + "_" + estimator_type + "_" + str(distortion) + "_exact_like.txt",
               exact_log_like_save)
    np.savetxt("ModelParams/" + sampler_type + "_" + estimator_type + "_" + str(distortion) + "_loss.txt",
               average_loss_save)