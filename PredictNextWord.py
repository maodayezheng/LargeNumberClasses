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
    epoch_step = params["epoch_step"]
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

    # Initialise sampler and loss estimator
    sampler = None
    with open("ProcessedData/frequency_100000.txt", 'r') as freq:
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
    x_r = word_embedding(targets)
    exact_log_like = estimator.imp_log_like(target_words, target_states, embedding, x_r)

    # Training Loss
    l2 = lamb * (cell.l2_regular()+word_embedding.l2_regular())
    objective = l2 + loss
    update = tf.train.GradientDescentOptimizer(l_rate).minimize(objective)

    # Initialise Variables
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init = tf.initialize_all_variables()
    session.run(init)

    # Get the training batch
    print("Start Training")
    batch = []
    with open('ProcessedData/sentences_100000.txt', 'r') as data:
        for d in data:
            d = json.loads(d)
            if len(d) > sentence_len:
                continue
            batch.append(d)
        data.close()
    print("Finished getting batch")
    input_dict = {}
    iteration = 0
    loss_check = iteration + epoch_step
    average_loss = 0
    exact_log_like_save = []
    average_loss_save =[]
    while iteration < 40000:
        iteration += 1

        # Randomly pick a data point from batch
        for i in range(batch_size):
            d = random.choice(batch)
            d = [0] * (sentence_len - len(d)) + d
            input_dict[inputs[i].name] = d

        if iteration % epoch_step is 0:
            loss_check += epoch_step
            _, exact, l = session.run([update, exact_log_like, loss], feed_dict=input_dict)
            average_loss = (average_loss + l) / 10
            exact_log_like_save.append(exact)
            average_loss_save.append(average_loss)
            print(estimator_type+" " + sampler_type + " " + str(distortion) +
                  " At iteration {}, the average estimate loss is {}, the exact log like is {}"
                  .format(iteration, average_loss, exact))
            average_loss = 0
        else:
            _ = session.run([update], feed_dict=input_dict)
        if loss_check - iteration < 9:
            average_loss += l

    word_embedding.save_param(session, "ModelParams/")
    cell.save_param(session, "ModelParams/")
    np.savetxt("ModelParams/" + sampler_type + "_" + estimator_type + "_" + str(distortion) + "_exact_like.txt",
               exact_log_like_save)
    np.savetxt("ModelParams/" + sampler_type + "_" + estimator_type + "_" + str(distortion) + "_loss.txt",
               average_loss_save)