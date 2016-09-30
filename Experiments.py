from __future__ import print_function
import tensorflow as tf
import json
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
    params = {"sampler_type": "uniform", "estimator_type": "IMP", "sample_size": [250, 500, 1000],
              "batch_size": [100, 50, 25],
              "num_classes": 50000, "sentence_len": 70, "epoch_step": 100, "input_dim": 100, "hidden_dim": 100,
              "output_dim": 100,
              "lamb": 0.0001, "l_rate": 0.004}
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
    paddings = []
    sentences = []
    for i in range(batch_size):
        s = tf.placeholder(tf.int32, shape=None, name="sentence_{}".format(i))
        inputs.append(s)
        padding = tf.placeholder(tf.int32, shape=None, name="padding_{}".format(i))
        s_pad = tf.pad(s, padding)
        sentences.append(s_pad)
        paddings.append(padding)
    """
    Initialise sampler and loss estimator
    """
    sampler = None
    if sampler_type is "uniform":
        sampler = UniformSampler(num_classes, sample_size)
    elif sampler_type is "unigram":
        sampler = UnigramSampler(num_classes, sample_size)
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
    word_embedding = EmbeddingLayer("word", num_classes, input_dim)

    """
    Reshape the input sentences
    """
    sentences = tf.squeeze(tf.pack(sentences))
    ss, tc, sc = estimator.draw_samples(sentences, sentence_len)
    estimator.set_sample_weights(sc)
    estimator.set_sample(ss)
    sentences = tf.split(1, sentence_len, sentences)
    mask = tf.zeros([batch_size, 1], dtype=tf.int32)
    l = len(inputs)
    loss = 0

    """
    Initialise Recurrent network
    """
    exact_log_like = 0.0
    cell = GRU(input_dim, hidden_dim, output_dim, "next-word")
    state = tf.zeros([batch_size, hidden_dim], dtype=tf.float32, name="init_state")
    embedding = word_embedding.get_embedding()
    approx_log_like = 0.0
    for i in range(l):
        mask_t = tf.cast(tf.not_equal(sentences[i], mask), tf.float32)
        words = word_embedding(sentences[i])
        state, output = cell(words, state)
        state = state*mask_t
        if i < l:
            targets = word_embedding(sentences[i + 1])
            loss += estimator.loss(targets, state, q=tc)*mask_t
            exact_log_like += exact_log_likelihood(targets, state, embedding)
            approx_log_like += estimator.likelihood(targets, state)

    exact_log_like = tf.reduce_mean(exact_log_like)
    approx_log_like = tf.reduce_mean(approx_log_like)
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
    batch = []
    start_pos = 0
    iteration = 0
    input_dict = {}
    while True:
        iteration += 1
        end_pos = start_pos + batch_size - 1
        # Stop criteria
        if end_pos > len(batch):
            break
        mini_bacth = batch[start_pos:end_pos]
        start_pos += batch_size

        """
        Feed the require inputs
        """
        for j in range(batch_size):
            input_dict[inputs[j].name] = mini_bacth[j]
            input_dict[paddings[j].name] = [[sentence_len-len(mini_bacth[j]), 0]]

        if iteration % epoch_step is 0:
            _, exact, approx = session.run([update, exact_log_like, approx_log_like], feed_dict=dict)
        else:
            _ = session.run([update], feed_dict=dict)


def exact_log_likelihood(x, h, embedding):
    print("exact log likelihood")
    target_scores = tf.reduce_sum(x * h, 1)
    Z = tf.reduce_sum(embedding*h, 1)
    return tf.log(target_scores/Z)

if __name__ == "main":
    main()
