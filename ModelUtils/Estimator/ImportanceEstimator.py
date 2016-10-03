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
        samples_scores = tf.matmul(h, samples, transpose_b=True)
        m = tf.stop_gradient(tf.maximum(target_scores, tf.reduce_max(samples_scores, 1)))
        target_scores -= m
        samples_scores -= m.reshape((-1, 1))
        # N
        exp_weight = tf.exp(samples_scores) / weights
        self.Z_ = tf.reduce_sum(tf.check_numerics(exp_weight, "each Z "), 1)

        # The loss of each element in target
        # N
        element_loss = target_scores - tf.log(q) - tf.log(self.Z_)
        loss = tf.reduce_mean(element_loss)
        return -loss
