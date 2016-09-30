from __future__ import print_function
import tensorflow as tf
from .Estimator import Estimator


class ImportanceEstimator(Estimator):
    def loss(self, x, h, q=None):
        """
            Calculate the estimate loss of Importance sampling approximation

            @Param x(NxD): The target word or batch
            @Param h(NxD): This is usually the output of neural network
            @Param q(N): The Weight of target
        """
        # K
        weights = self.get_sample_weights()
        if weights is None:
            raise ValueError("sample weights must be set")
        # KxD
        samples = self.get_samples()
        if samples is None:
            raise ValueError("samples must be set")
        # N
        target_scores = tf.reduce_sum(x * h, 1)
        # N x K
        samples_scores = tf.matmul(h, samples, transpose_b=True)
        # N
        Z = tf.reduce_sum(tf.exp(samples_scores) / weights, 1)
        loss = tf.reduce_mean(target_scores - tf.log(q) - tf.log(Z))
        return loss
        # return target_scores, q, Z
        # return x, h, weights, samples

    def likelihood(self, x, h, q=None):
        """
            Calculate the estimate likelihood of importance sampling approximation

            @Param x: The target word or batch
            @Param h: This is usually the output of neural network
            @Param q: The Weight of target
        """
        print("likelihood")
