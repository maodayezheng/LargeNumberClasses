from __future__ import print_function
import tensorflow as tf
from .Estimator import Estimator


class AlexEstimator(Estimator):
    def __init__(self, *args, **kwargs):
        super(AlexEstimator, self).__init__(extra=10, *args, **kwargs)

    def loss(self, x, h, q=None):
        """
            Calculate the estimate loss of Alex approach approximation

            @Param x(NxD): The target word or batch
            @Param h(NxD): This is usually the output of neural network
            @Param q(N): The Weight of target

            @Return loss: The estimate loss
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
        self.target_exp_ = tf.exp(target_scores) * q
        self.Z_ = self.target_exp_ + tf.reduce_sum(tf.exp(samples_scores), 1)

        # The loss of each element in target
        # N
        element_loss = target_scores + tf.log(q) - tf.log(self.Z_)
        loss = tf.reduce_mean(element_loss)
        return -loss

