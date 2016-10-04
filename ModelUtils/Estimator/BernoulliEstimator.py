from __future__ import print_function
import tensorflow as tf
from .Estimator import Estimator


class BernoulliEstimator(Estimator):
    def __init__(self, *args, **kwargs):
        super(BernoulliEstimator, self).__init__(extra=10, *args, **kwargs)

    def loss(self, x, h, q=None, eps=1e-9):
        """
            Calculate the estimate loss of bernoulli sampling approximation

            @Param x(NxD): The target word or batch
            @Param h(NxD): This is usually the output of neural network
            @Param q(N): The Weight of target

            @Return loss: The estimate loss
        """
        # K
        weights = self.get_sample_weights()
        if weights is None:
            raise ValueError("sample weights must be set")
        # KExD
        samples = self.get_samples()
        if samples is None:
            raise ValueError("samples must be set")
        # N
        self.target_score_ = tf.reduce_sum(x * h, 1)
        # N x KE
        samples_scores = tf.matmul(h, samples, transpose_b=True)
        # N x KE - Effectively making exp(ss) = exp(s) / weights
        samples_scores -= tf.log(weights)
        # N x K
        samples_scores = self.get_unique(samples_scores)
        # N - Conditioning
        max_t = tf.reduce_max(tf.concat(1, (tf.reshape(self.target_score_, (-1, 1)), samples_scores)), 1)
        m = tf.stop_gradient(max_t)
        target_scores = self.target_score_ - m
        samples_scores -= tf.reshape(m, (-1, 1))
        # N
        self.Z_ = tf.exp(target_scores) + tf.reduce_mean(tf.exp(samples_scores), 1)
        # The loss of each element in target
        # N
        element_loss = target_scores - tf.log(self.Z_ + eps)
        loss = tf.reduce_mean(element_loss)
        return -loss
