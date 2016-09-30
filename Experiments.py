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
from DataUtils import DataUtils
from DataUtils import BatchUtils


def main():
    print "Dealing with Large number"
    params = {"sampler_type": "uniform", "estimator_type": "IMP", "sample_size": [250, 500, 1000], "batch_size": [100, 50, 25],
              "window_size": 70, "epoch_step": 100, "input_dim": 100, "hidden_dim": 100, "output_dim": 100,
              "lamb": 0.0001, "l_rate": 0.004, "embedding_path": None}
    predict_next_word(params)
    tf.nn.uniform_candidate_sampler()


def predict_next_word(params):
    """
    Get the training hyper parameters
    """

    """
    Get training hyper parameters
    """
    sampler_type = params["sampler_type"]
    estimator_type = params["estimator_type"]
    sample_size = params["sample_size"]
    batch_size = params["batch_size"]
    window_size = params["window_size"]
    epoch_step = params["epoch_step"]
    input_dim = params["input_dim"]
    hidden_dim = params["hidden_dim"]
    output_dim = params["output_dim"]
    lamb = params["lamb"]
    l_rate = params["l_rate"]
    embedding_path = params["embedding_path"]

    """
    Loads the data stats and pre define approximation samples
    """
    frequency = None
    with open("ModelParams/Frequency.txt", "r") as token_freq:
        frequency = json.loads(token_freq.read())

    """
    Initialise the RNN computational graph
    """
    inputs = []

    for i in range(batch_size):
        w = tf.placeholder(tf.int32, shape=None, name="window_{}".format(i))
        inputs.append(w)

    word_embedding = EmbeddingLayer("word", 40000, 300)
    cell = GRU(input_dim, hidden_dim, output_dim, "next-word")

    sampler = None
    if sampler_type is "uniform":
        sampler = UniformSampler()
    elif sampler_type is "unigram":
        sampler = UnigramSampler()
    else:
      raise Exception("{} type sampler is not support".format(sampler_type))

    estimator = None
    if estimator_type is "BER":
       estimator = BernoulliEstimator()
    elif estimator_type is "IMP":
       estimator = ImportanceEstimator()
    elif estimator_type is "NEG":
        estimator = NegativeEstimator()
    elif estimator_type is "BLA":
        estimator = BlackOutEstimator()
    elif estimator_type is "ALEX":
        estimator = AlexEstimator()
    else:
        raise Exception("{} type estimator is not support".format(estimator_type))

    init_state = tf.placeholder(tf.float32, shape=None, name="init_state")

    """
    Reshape the input to feed in RNN
    """

    windows = tf.squeeze(tf.pack(inputs))
    windows = tf.split(1, window_size, windows)
    l = len(windows)
    loss = 0
    state = init_state
    for i in range(l):
        words = word_embedding(windows[i])
        state, output = cell(words, state)
        if i < l:
            '''
            sample_set = sampler.get_sample_set(windows[i+1])
            sample_vec = word_embedding(sample_set)
            targets = word_embedding(windows[i+1])
            loss += sampler.loss(targets, state, sample_vec)
            '''

    likelihood_exact = None

    l2 = lamb * (cell.l2_regular())
    objective = l2 + loss
    update = tf.train.GradientDescentOptimizer(l_rate).minimize(objective)

    """
    Training
    """
    session = tf.Session()
    init = tf.initialize_all_variables()
    session.run(init)
    s0 = None
    s = s0
    for i in range(40000):
        print "training"
        batch = []

        """
        Feed the require inputs
        """
        dict = {init_state.name: s}
        for j in range(len(batch)):
            dict[inputs[j].name] = batch[j]

        if i % epoch_step is 0:
            dict[init_state.name] = s0
            _, s, approx, exact = session.run([update, state, likelihood_approximate,
                                                  likelihood_exact], feed_dict=dict)
        else:
            _, s = session.run([update, state], feed_dict=dict)


if __name__ == "main":
    main()
