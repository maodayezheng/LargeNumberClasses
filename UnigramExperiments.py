from __future__ import print_function
import json
import random
import tensorflow as tf
from ModelUtils.EmbeddingLayer import EmbeddingLayer
from ModelUtils.GRU import GRU
from ModelUtils.Sampler.UnigramSampler import UnigramSampler
from ModelUtils.Estimator.AlexEstimator import AlexEstimator
from ModelUtils.Estimator.BernoulliEstimator import BernoulliEstimator
from ModelUtils.Estimator.ImportanceEstimator import ImportanceEstimator
from ModelUtils.Estimator.BlackOutEstimator import BlackOutEstimator
from ModelUtils.Estimator.NegativeEstimator import NegativeEstimator


def main():
    print("Dealing with Large number unigram test")
    estimator_types =["BLA", "BER", "IMP", "ALEX", "NEG"]
    params = {"sampler_type": "unigram", "sample_size": 250,
              "batch_size": 10,
              "num_classes": 28095, "sentence_len": 70, "epoch_step": 100, "input_dim": 100, "hidden_dim": 100,
              "output_dim": 100,
              "lamb": 0.001, "l_rate": 0.02}

    for e in estimator_types:
         params["estimator_type"] = e
         predict_next_word(params)


def predict_next_word(params):
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
    num_classes = params["num_classes"]
    print("Runing the "+estimator_type+ "Estimator Test")
    # Initialise the input nodes
    inputs = []
    for i in range(batch_size):
        s = tf.placeholder(tf.int64, shape=None, name="sentence_{}".format(i))
        inputs.append(s)

    # Initialise sampler and loss estimator
    sampler = None
    with open("ProcessedData/frequency_100000.txt", 'r') as freq:
            p_dist = json.loads(freq.read())
            sampler = UnigramSampler(num_classes, sample_size, proposed_dist=p_dist)
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
    word_embedding = EmbeddingLayer(num_classes, input_dim, sampler_type+estimator_type+"word")

    # Reshape the input sentences
    sentences = tf.squeeze(tf.pack(inputs))
    sentences = tf.split(1, sentence_len, sentences)
    mask = tf.zeros([batch_size, 1], dtype=tf.int64)
    l = len(sentences)

    # Initialise Recurrent network
    cell = GRU(input_dim, hidden_dim, output_dim, sampler_type+estimator_type)
    init_state = tf.zeros([batch_size, hidden_dim], dtype=tf.float32, name="init_state")
    state = init_state
    embedding = word_embedding.get_embedding()
    states = []
    words = []
    masks = []
    for i in range(l):
        mask_t = tf.not_equal(sentences[i], mask)
        word = word_embedding(sentences[i])
        if i > 0:
            words.append(word)
        state, output = cell(word, state)
        states.append(state)
        masks.append(mask_t)
    words.append(init_state)

    # Masking the parameters
    states = tf.concat(0, states)
    words = tf.concat(0, words)
    masks = tf.concat(0, masks)
    masks = tf.reshape(masks, [batch_size*sentence_len])

    # Draw samples
    ss, tc, sc = estimator.draw_samples(sentences, 1, masks)
    estimator.set_sample_weights(sc)
    estimator.set_sample(word_embedding(ss))
    states = tf.boolean_mask(states, masks)
    words = tf.boolean_mask(words, masks)
    tc = tf.boolean_mask(tc, masks)
    # Estimate loss
    loss = tf.check_numerics(estimator.loss(words, states, q=tc), message="The loss is ")
    exact_log_like = estimator.log_likelihood(words, states, embedding)

    # Training Loss
    l2 = lamb * (cell.l2_regular())
    objective = l2 + loss
    update = tf.train.GradientDescentOptimizer(l_rate).minimize(objective)

    # Initialise Variables
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
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
    average_loss_save = []
    exact_log_like_save = []
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
            average_loss_save.append(average_loss)
            exact_log_like_save.append(exact)
            print("At iteration {}, the average estimate loss is {}, the exact log like is {}".format(iteration,
                                                                                                      average_loss,
                                                                                                      exact))
            average_loss = 0
        else:
            _, l = session.run([update, loss], feed_dict=input_dict)
        if loss_check - iteration < 9:
            average_loss += l

    word_embedding.save_param(session, "ModelParams/")
    cell.save_param(session, "ModelParams/")

    with open("ModelParams/"+sampler_type+"_"+estimator_type+"_aver_loss.txt", "r") as save_estimate_loss:
            save_estimate_loss.write(json.dumps(average_loss_save))

    with open("ModelParams/" + sampler_type + "_" + estimator_type + "_exact_loss.txt", "r") as save_estimate_loss:
        save_estimate_loss.write(json.dumps(exact_log_like_save))


main()
