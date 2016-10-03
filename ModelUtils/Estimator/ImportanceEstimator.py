from __future__ import print_function
import tensorflow as tf
from .Estimator import Estimator


class ImportanceEstimator(Estimator):
    def loss(self, x, h, mask, q=None):
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
        self.target_exp_ = tf.exp(target_scores)
        # N x K
        samples_scores = tf.matmul(h, samples, transpose_b=True)
        # N
        self.Z_ = tf.reduce_sum(tf.exp(samples_scores) / weights, 1)
        # The loss of each element in target
        # N
        element_loss = target_scores - tf.log(q) -tf.log(self.Z_)
        loss = tf.reduce_mean(element_loss * mask)
        return -loss
        # return target_scores, q, Z
        # return x, h, weights, samples

    def likelihood(self, x, h, q=None):
        """
            Calculate the estimate likelihood of importance sampling approximation

            @Param x: The target word or batch
            @Param h: This is usually the output of neural network
            @Param q: The Weight of target
        """
        if self.target_exp_ is None:
            self.target_exp_ = tf.exp(tf.reduce_sum(x * h, 1))
        if self.Z_ is None:
            samples = self.get_samples()
            weights = self.get_sample_weights()
            self.Z_ = tf.reduce_sum(tf.exp(tf.matmul(h, samples, transpose_b=True)) / weights, 1)

        log_likelihood = tf.log(self.target_exp_) - tf.log(self.Z_)
        return tf.reduce_mean(log_likelihood)

