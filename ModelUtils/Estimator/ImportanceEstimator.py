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
        self.target_exp_ = tf.exp(target_scores)
        # N x K
        samples_scores = tf.check_numerics(tf.matmul(h, samples, transpose_b=True), message="The sample score is")
        # N
        self.Z_ = tf.check_numerics(tf.reduce_sum(tf.exp(samples_scores) / weights, 1), message="The Z is ")

        # The loss of each element in target
        # N
        log_Z = tf.check_numerics(tf.log(self.Z_), message="The log Z")
        element_loss = target_scores - tf.log(q) - log_Z
        loss = tf.reduce_mean(element_loss)
        return -loss
