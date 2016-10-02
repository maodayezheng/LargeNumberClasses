from __future__ import print_function
import tensorflow as tf
from .Estimator import Estimator


class NegativeEstimator(Estimator):
    def loss(self, x, h, mask, q=None):
        """
        Calculate the estimate loss of negative sampling approximation

        @Param x(NxD): the target word or batch, NxD
        @Param h(NxD): This is usually the output of neural network NxD
        """
        # N x D
        samples = self.get_samples()
        # N
        target_scores = tf.reduce_sum(x * h, 1)
        # N x K
        samples_scores = tf.matmul(h, samples, transpose_b=True)
        loss = tf.reduce_mean(tf.log(tf.nn.sigmoid(target_scores)) * mask) - \
               tf.reduce_mean(tf.reduce_sum(tf.log(tf.nn.sigmoid(samples_scores)), 1) * mask)

        return -loss

    def likelihood(self, x, h):
        """
            Calculate the estimate likelihood of negative sampling approximation

            @Param x: the target word or batch
            @Param h: This is usually the output of neural network
        """
        print("likelihood")
