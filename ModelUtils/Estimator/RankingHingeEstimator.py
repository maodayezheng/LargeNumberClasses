from __future__ import print_function
from .Estimator import Estimator

import tensorflow as tf


class RankingOursEstimator(Estimator):
    def __init__(self):
        super(RankingOursEstimator, self).__init__(10)

    def loss(self, x, h, q = None, eps=1e-9):
        """
         Calculate the estimate loss of negative sampling approximation
        :param h: NxD
        :param x: NxD
        :param eps: conditioning number
        :return:
        """
        weights = self.get_sample_weights()
        if weights is None:
            raise ValueError("sample weights must be set")
        # KExD
        samples = self.get_samples()
        if samples is None:
            raise ValueError("samples must be set")


        # N
        target_scores = tf.reduce_sum(x * h, 1)
        # N x KE
        samples_scores = tf.matmul(h, samples, transpose_b=True)
        samples_scores -= 1
        # N x K
        samples_scores = self.get_unique(samples_scores)
        # N
        element_loss = - tf.nn.softplus(samples_scores - tf.expand_dims(target_scores))
        loss = tf.reduce_mean(element_loss)
        return -loss