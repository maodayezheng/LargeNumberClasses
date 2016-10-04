from __future__ import print_function
import tensorflow as tf
from .Estimator import Estimator


class NegativeEstimator(Estimator):
    def loss(self, x, h, q=None):
        """
        Calculate the estimate loss of negative sampling approximation

        @Param x(NxD): the target word or batch, NxD
        @Param h(NxD): This is usually the output of neural network NxD
        """
        # N x D
        samples = self.get_samples()
        # N
        self.target_score_ = tf.reduce_sum(x * h, 1)
        # N x K
        samples_scores = tf.matmul(h, -samples, transpose_b=True)
        # N
        element_loss = tf.log(tf.nn.sigmoid(self.target_score_)) - \
            tf.reduce_sum(tf.log(tf.nn.sigmoid(samples_scores)), 1)
        loss = tf.reduce_mean(element_loss)
        return loss

