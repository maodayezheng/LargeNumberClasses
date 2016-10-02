import tensorflow as tf
from DataUtils.DataUtils import DataUtils
from ModelUtils.Sampler.UniformSampler import UniformSampler
from ModelUtils.Sampler.UnigramSampler import UnigramSampler
from ModelUtils.Estimator.AlexEstimator import AlexEstimator
from ModelUtils.Estimator.BlackOutEstimator import BlackOutEstimator
from ModelUtils.Estimator.BernoulliEstimator import BernoulliEstimator
from ModelUtils.Estimator.ImportanceEstimator import ImportanceEstimator
from ModelUtils.Estimator.NegativeEstimator import NegativeEstimator
from ModelUtils.EmbeddingLayer import EmbeddingLayer
from ModelUtils.GRU import GRU


def DataUtilsUnitTest():
    source_path = "../OriginalData/europarl-v7.en"
    save_path = "ProceessedData/"
    DataUtils.create_vocabulary(source_path)


def AlexEstimatorUnitTest(sess):
    print("Start test AlexEstimator")
    sampler = UniformSampler(40, 4)
    embedding = EmbeddingLayer(40, 10, "test")
    estimator = AlexEstimator(sampler)
    target = tf.constant([[1]], dtype=tf.int64)
    s, tc, sc = estimator.draw_samples(target, 1)
    samples = embedding(s)
    estimator.set_sample(samples)
    estimator.set_sample_weights(sc)
    x = tf.random_uniform([1, 10])
    h = tf.random_uniform([1, 10])
    loss = estimator.loss(x, h, q=0.025)
    init = tf.initialize_all_variables()
    sess.run(init)
    loss = sess.run([loss])
    print(loss)

def BernoulliEstimatorUnitTest(sess):
    print("Start test BernoulliEstimator")
    sampler = UniformSampler(40, 4)
    embedding = EmbeddingLayer(40, 10, "test")
    estimator = BernoulliEstimator(sampler)
    target = tf.constant([[1]], dtype=tf.int64)
    s, tc, sc = estimator.draw_samples(target, 1)
    samples = embedding(s)
    estimator.set_sample(samples)
    estimator.set_sample_weights(sc)
    x = tf.random_uniform([1, 10])
    h = tf.random_uniform([1, 10])
    loss = estimator.loss(x, h, q=0.025)
    init = tf.initialize_all_variables()
    sess.run(init)
    loss = sess.run([loss])
    print (loss)



def ImportanceEstimatorUnitTest(sess):
    print("Start test ImportanceEstimator")
    sampler = UniformSampler(40, 4)
    embedding = EmbeddingLayer(40, 10, "test")
    estimator = ImportanceEstimator(sampler)
    target = tf.constant([[1]], dtype=tf.int64)
    s, tc, sc = estimator.draw_samples(target, 1)
    samples = embedding(s)
    estimator.set_sample(samples)
    estimator.set_sample_weights(sc)
    x = tf.random_uniform([1, 10])
    h = tf.random_uniform([1, 10])
    loss = estimator.loss(x, h, q=0.025)
    init = tf.initialize_all_variables()
    sess.run(init)
    loss = sess.run([loss])
    print (loss)

def NegativeEstimatorUnitTest(sess):
    print("Start test NegativeEstimator")
    sampler = UniformSampler(40, 4)
    embedding = EmbeddingLayer(40, 10, "test")
    estimator = NegativeEstimator(sampler)
    target = tf.constant([[1]], dtype=tf.int64)
    s, tc, sc = estimator.draw_samples(target, 1)
    samples = embedding(s)
    estimator.set_sample(samples)
    estimator.set_sample_weights(sc)
    x = tf.random_uniform([1, 10])
    h = tf.random_uniform([1, 10])
    loss = estimator.loss(x, h)
    init = tf.initialize_all_variables()
    sess.run(init)
    loss = sess.run([loss])
    print (loss)

def BlackOutEstimatorUnitTest(sess):
    print ("Start test NegativeEstimator")
    sampler = UniformSampler(40, 4)
    embedding = EmbeddingLayer(40, 10, "test")
    estimator = NegativeEstimator(sampler)
    target = tf.constant([[1]], dtype=tf.int64)
    s, tc, sc = estimator.draw_samples(target, 1)
    samples = embedding(s)
    estimator.set_sample(samples)
    estimator.set_sample_weights(sc)
    x = tf.random_uniform([1, 10])
    h = tf.random_uniform([1, 10])
    loss = estimator.loss(x, h)
    init = tf.initialize_all_variables()
    sess.run(init)
    loss = sess.run([loss])

def UniformSamplerUnitTest(sess):
    print ("Start test UniformSampler")
    k = 10
    sampler = UniformSampler(40, k)
    target = tf.constant([1, 5, 6], dtype=tf.int64)
    sample, true_count, sample_count = sampler.draw_sample(target, 1)
    s, tc, sc = sess.run([sample, true_count, sample_count])
    print(sc)
    assert (len(s) is k), "Uniform Sampler test fail"

def UnigramSamplerUnitTest(sess):
    print("Start test UnigramSampler")
    dist = [0.1, 0.2, 0.3, 0.4]
    sampler = UnigramSampler(4, 2, proposed_dist=dist)
    target = tf.constant([[1]], dtype=tf.int64)
    sample, true_count, sample_count = sampler.draw_sample(target, 1)
    s, tc, sc = sess.run([sample, true_count, sample_count])
    assert (len(s) is 2), "Uniform Sampler test fail"


def GRUUnitTest(sess):
    print("Start test GRU")
    input_dim = 10
    hidden_dim = 30
    output_dim = 30
    cell = GRU(input_dim, hidden_dim, output_dim, "test")
    i = tf.random_uniform([1, 10])
    s = tf.random_uniform([1, 30])
    o, h = cell(i, s)
    init = tf.initialize_all_variables()
    sess.run(init)
    o, h = sess.run([o, h])
    assert (len(o[0]) is output_dim), "output dimension not consist"
    assert (len(h[0]) is hidden_dim), "hidden dimension not consist"

def EmbeddingLayerUnitTest(sess):
    print("Start test EmbeddingLayer")
    embedding = EmbeddingLayer(10, 1, "test")
    target = tf.constant([0, 1], dtype=tf.int64)
    v = embedding(target)
    init = tf.initialize_all_variables()
    sess.run(init)
    v = sess.run(v)
    assert(len(v) is 2), "The size of look up vectors is not consist"

def SmallRNNTest(sess):
    embedding = EmbeddingLayer(40, 10, "test")
    cell = GRU(10, 10, 10, "test")
    target = tf.constant([0, 3, 4], dtype=tf.int64)
    s = tf.random_uniform([3, 10])
    o, h = cell(embedding(target), s)
    init = tf.initialize_all_variables()
    sess.run(init)
    v = sess.run(o)
    print v

print("Start unit tests")
#sess = tf.Session()
#UniformSamplerUnitTest(sess)
#UnigramSamplerUnitTest(sess)
#GRUUnitTest(sess)
#EmbeddingLayerUnitTest(sess)
#AlexEstimatorUnitTest(sess)
#BernoulliEstimatorUnitTest(sess)
#ImportanceEstimatorUnitTest(sess)
#NegativeEstimatorUnitTest(sess)
#SmallRNNTest(sess)
#BatchUtilsUnitTest()
DataUtilsUnitTest()
