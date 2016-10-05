from __future__ import print_function
import tensorflow as tf
from .Estimator import Estimator


class ImportanceEstimator(Estimator):
    def loss(self, x, h, q=None, eps=1e-9):
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
        self.target_score_ = tf.reduce_sum(x * h, 1)
        # N - This makes exp(ts) = exp(t)/q
        target_scores = self.target_score_ # - tf.reshape(tf.log(q), [-1])
        # N x K
        samples_scores = tf.matmul(h, samples, transpose_b=True)
        # N - This makes exp(ss) = exp(s)/ weights
        samples_scores -= tf.reshape(tf.log(weights), (1, -1))
        # N - Conditioning
        target_scores, samples_scores = self.clip_likelihood(target_scores, samples_scores)
        # N
        self.Z_ = tf.reduce_sum(tf.exp(samples_scores), 1)
        # N - The loss of each element in target
        element_loss = target_scores - tf.log(self.Z_ + eps)
        loss = tf.reduce_mean(element_loss)
        return -loss
