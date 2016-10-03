from __future__ import print_function
import tensorflow as tf
from .Estimator import Estimator


class BlackOutEstimator(Estimator):
    def __init__(self, *args, **kwargs):
        super(BlackOutEstimator, self).__init__(extra=10, *args, **kwargs)

    def loss(self, x, h, mask, q=None):
        """
            Calculate the estimate loss of blackout approximation

            @Param x(NxD): The target word or batch
            @Param h(NxD): This is usually the output of neural network
            @Param q(N): The Weight of target
        """
        # KE
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
        # N x K
        samples_scores = self.get_unique(x, samples, samples_scores)
        # N
        Z = tf.exp(target_scores) / q + tf.reduce_sum(tf.exp(samples_scores) / weights, 1)
        # N x K
        neg_scores = tf.log(tf.reshape(Z, (-1, 1)) - tf.exp(samples_scores) / weights)
        # The loss of each element in target
        # N
        element_loss = target_scores - tf.log(q) + tf.reduce_sum(neg_scores, 1) -\
                       (tf.cast(tf.shape(samples_scores)[1], dtype=tf.float32) + 1.0) * tf.log(Z)
        loss = tf.reduce_mean(element_loss * mask)
        return -loss

    def likelihood(self, x, h, q=None):
        """
            Calculate the estimate likelihood of blackout approximation

            @Param x: The target word or batch
            @Param h: This is usually the output of neural network
            @Param q: The Weight of target
        """
        print("likelihood")