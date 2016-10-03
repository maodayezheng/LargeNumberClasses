from __future__ import print_function
import json
import tensorflow as tf
from ModelUtils.EmbeddingLayer import EmbeddingLayer
from ModelUtils.GRU import GRU
from ModelUtils.Sampler.UniformSampler import UniformSampler
from ModelUtils.Sampler.UnigramSampler import UnigramSampler
from ModelUtils.Estimator.AlexEstimator import AlexEstimator
from ModelUtils.Estimator.BernoulliEstimator import BernoulliEstimator
from ModelUtils.Estimator.ImportanceEstimator import ImportanceEstimator
from ModelUtils.Estimator.BlackOutEstimator import BlackOutEstimator
from ModelUtils.Estimator.NegativeEstimator import NegativeEstimator


def main():
    print("Dealing with Large number")
    params = {"sampler_type": "uniform", "estimator_type": "IMP", "sample_size": 250,
              "batch_size": 25,
              "num_classes": 40004, "sentence_len": 70, "epoch_step": 100, "input_dim": 100, "hidden_dim": 100,
              "output_dim": 100,
              "lamb": 0.0001, "l_rate": 0.05}
    predict_next_word(params)


def predict_next_word(params):
    """
    The RNN model to predict next word

    @Param params: A dictionary of training hyper parameters
    """

    """
    Get training hyper parameters
    """
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

    """
    Initialise the input nodes
    """
    inputs = []
    for i in range(batch_size):
        s = tf.placeholder(tf.int64, shape=None, name="sentence_{}".format(i))
        inputs.append(s)

    """
    Initialise sampler and loss estimator
    """
    sampler = None
    if sampler_type is "uniform":
        sampler = UniformSampler(num_classes, sample_size)
    elif sampler_type is "unigram":
        with open("../ProcessedData/frequency.txt", 'r') as freq:
            p_dist = json.loads(freq.read())
            sampler = UnigramSampler(num_classes, sample_size, proposed_dist=p_dist)
            freq.close()
    else:
        raise Exception("{} type sampler is not support".format(sampler_type))

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

    """
    Initialise the word embedding layer
    """
    word_embedding = EmbeddingLayer(num_classes, input_dim, "word")

    """
    Reshape the input sentences
    """
    sentences = tf.squeeze(tf.pack(inputs))
    ss, tc, sc = estimator.draw_samples(sentences, 1)
    estimator.set_sample_weights(sc)
    estimator.set_sample(word_embedding(ss))
    sentences = tf.split(1, sentence_len, sentences)
    mask = tf.zeros([batch_size, 1], dtype=tf.int64)
    l = len(sentences)
    """
    Initialise Recurrent network
    """
    cell = GRU(input_dim, hidden_dim, output_dim, "next-word")
    init_state = tf.zeros([batch_size, hidden_dim], dtype=tf.float32, name="init_state")
    state = init_state
    embedding = word_embedding.get_embedding()
    approx_log_like = 0.0
    states = []
    words = []
    masks = []
    for i in range(l):
        mask_t = tf.cast(tf.not_equal(sentences[i], mask), tf.float32)
        word = word_embedding(sentences[i])
        if i > 0:
            words.append(word)
        state, output = cell(word, state)
        state = state*mask_t
        states.append(state)
        masks.append(mask_t)
    words.append(init_state)
    states = tf.concat(0, states)
    words = tf.concat(0, words)
    masks = tf.concat(0, masks)
    loss = estimator.loss(words, states, masks, q=tc)
    exact_log_like = exact_log_likelihood(words, states, embedding)
    #approx_log_like = tf.reduce_mean(approx_log_like)

    """
    Training Loss
    """
    l2 = lamb * (cell.l2_regular())
    objective = l2 + loss
    update = tf.train.GradientDescentOptimizer(l_rate).minimize(objective)

    """
    Initialise Variables
    """
    session = tf.Session()
    init = tf.initialize_all_variables()
    session.run(init)

    """
    Get the training batch
    """
    print("Start Training")
    batch = []
    with open('ProcessedData/sentences.txt', 'r') as data:
        for d in data:
            d = json.loads(d)
            if len(d) > sentence_len:
                continue
            batch.append(d)
        data.close()
    print("Get 100000 test sample")
    start_pos = 0
    iteration = 0
    input_dict = {}
    while True:
        iteration += 1
        end_pos = start_pos + batch_size
        # Stop criteria
        if end_pos > len(batch):
            break
        mini_bacth = batch[start_pos:end_pos]
        start_pos = end_pos + 1

        """
        Feed the require inputs
        """
        for j in range(batch_size):
            d = mini_bacth[j]
            if len(d) < sentence_len:
                d = [0]*(sentence_len-len(d)) + d
            input_dict[inputs[j].name] = d

        if iteration % epoch_step is 0:
            _, exact = session.run([update, exact_log_like], feed_dict=input_dict)
        else:
            _, l = session.run([update, loss], feed_dict=input_dict)
            if iteration % 50 is 0:
                print("The loss at iteration {} is {}".format(iteration, l))


def exact_log_likelihood(x, h, embedding):
    target_scores = tf.reduce_sum(x * h, 1)
    Z = tf.reduce_sum(tf.matmul(h, embedding, transpose_b=True), 1)
    return tf.reduce_sum(tf.log(target_scores)) - tf.reduce_sum(tf.log(Z))

main()
