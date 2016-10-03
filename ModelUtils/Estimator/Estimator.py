import tensorflow as tf
from ModelUtils.Sampler import Sampler


class Estimator(object):
    def __init__(self, sampler):
        """
        The constructor of estimator

        @Param sampler: An instance of Sampler
        @Param embedding_layer: The embedding_layer for looking up word vector
        """
        self.sampler_ = sampler
        self.weights_ = None
        self.samples_ = None
        self.target_exp_ = None
        self.Z_ = None

    def loss(self, x, h, mask):
        """
        Abstract method requires to be implement by sub classes

        @Param x: The target words or batch
        @Param h: This is usually the output of neural network
        @Param mask: The mask that is used to filter losss of some element
        """
        raise Exception("Can not call abstract method loss in Estimator")

    def likelihood(self, x, h):
        """
        Abstract method requires to be implement by sub classes

        @Param x: The target words or batch
        @Param h: This is usually the output of neural network
        """
        raise Exception("Can not call abstract method likelihood in Estimator")

    def draw_samples(self, target, num_targets):
        """
        draw sample set and sample weights for approximation

        @Param target: the target words or target batch
        @Param num_targets: the length of target words or target batch

        @Return samples: The index of samples
        @Return target_prob: The probability of target occurrence
        @Return sample_prob: The probability of sample occurrence
        """
        samples, target_prob, sample_prob = self.sampler_.draw_sample(target, num_targets)
        #target_prob = tf.expand_dims(target_prob, 0)
        #sample_prob = tf.expand_dims(sample_prob, 0)
        return samples, target_prob, sample_prob

    def set_sample(self, samples):
        self.samples_ = samples

    def get_samples(self):
        return self.samples_

    def set_sample_weights(self, wieghts):
        self.weights_ = wieghts

    def get_sample_weights(self):
        return self.weights_

    def get_sample_size(self):
        return self.sampler_.num_samples_
