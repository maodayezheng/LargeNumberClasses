import tensorflow as tf
from ModelUtils.Sampler import Sampler


class Estimator(object):
    def __init__(self, sampler, extra=0):
        """
        The constructor of estimator

        @Param sampler: An instance of Sampler
        @Param embedding_layer: The embedding_layer for looking up word vector
        """
        self.sampler_ = sampler
        self.sampler_.num_samples_ += extra
        self.extra = extra
        self.weights_ = None
        self.samples_ = None
        self.target_exp_ = None
        self.Z_ = None
        self.bm = None

    def loss(self, x, h):
        """
        Abstract method requires to be implement by sub classes

        @Param x: The target words or batch
        @Param h: This is usually the output of neural network
        """
        raise Exception("Can not call abstract method loss in Estimator")

    def log_likelihood(self, x, h, embedding):
        """
            Abstract method requires to be implement by sub classes

            @Param x: The target words or batch
            @Param h: This is usually the output of neural network
            @Param embedding: The embedding vectors of all words

            @Return log_like: The exact log likelihood average over words
        """
        target_scores = tf.reduce_sum(x * h, 1)
        Z = tf.reduce_sum(tf.matmul(h, embedding, transpose_b=True), 1)
        log_like = tf.reduce_mean((target_scores-tf.log(Z)))
        return log_like

    def draw_samples(self, target, num_targets, mask):
        """
        draw sample set and sample weights for approximation

        @Param target: the target words or target batch
        @Param num_targets: the length of target words or target batch

        @Return samples: The index of samples
        @Return target_prob: The probability of target probability
        @Return sample_prob: The probability of sample probability
        """
        samples, target_prob, sample_prob = self.sampler_.draw_sample(target, num_targets)
        # samples = tf.Print(samples, [tf.shape(samples)], "Samples shape:")
        # targets = tf.Print(target, [tf.shape(target)], "Targets shape:")
        # mask = tf.Print(mask, [tf.shape(mask)], "Mask shape:")
        target = tf.boolean_mask(tf.reshape(target, [-1]), mask)
        # targets = tf.Print(target, [tf.shape(target)], "Targets2 shape:")
        N = tf.shape(target)[0]
        K = tf.shape(samples)[0] - self.extra

        # Indicator with 0 if they coincide
        ind = tf.cast(tf.not_equal(tf.reshape(target, (-1, 1)), tf.reshape(samples, (1, -1))), tf.int32)
        # ind = tf.Print(ind, [tf.shape(ind)], "Ind shape:")
        # The first K samples which are not equal to the target
        _, i = tf.nn.top_k(ind, sorted=True, k=K)
        # i = tf.Print(i, [tf.shape(ind)], "I shape:")
        i = tf.reshape(i, [-1])
        r = tf.range(0, tf.shape(target)[0])
        r = tf.reshape(tf.transpose(tf.reshape(tf.tile(r, [K]), (K, N))), [-1])
        coords = tf.transpose(tf.pack([r, i]))
        # coords = tf.Print(coords, [tf.shape(coords)], "Coords shape:")
        self.bm = tf.cast(tf.sparse_to_dense(coords, tf.shape(ind), 1), tf.bool)
        # self.bm = tf.Print(self.bm, [tf.shape(self.bm)], "Bm shape:")
        return samples, target_prob, sample_prob

    def get_unique(self, sample_scores):
        """
        Given a K'=K + self.extra samples, and their scores returns the NxK matrix of scores
        of K samples which do not coincide with the targets.
        :param sample_scores(NxK')
        :return:
        """
        if self.extra == 0:
            return sample_scores
        # N =
        # sample_scores =
        # sample_scores = tf.Print(sample_scores, [tf.shape(sample_scores)], "SS shape:")
        # a = tf.boolean_mask(sample_scores, self.bm)
        # a = tf.Print(sample_scores, [tf.shape(a)], "A shape:")
        return tf.reshape(tf.boolean_mask(sample_scores, self.bm), (tf.shape(sample_scores)[0],
                                                                    self.sampler_.num_samples_ - self.extra))

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
