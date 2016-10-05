from __future__ import print_function
import tensorflow as tf
from .Estimator import Estimator


class AlexEstimator(Estimator):
    def __init__(self, *args, **kwargs):
        super(AlexEstimator, self).__init__(extra=10, *args, **kwargs)

    def loss(self, x, h, q=None, eps=1e-9):
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
        self.target_score_ = tf.reduce_sum(x * h, 1)
        # N x KE
        samples_scores = tf.matmul(h, samples, transpose_b=True)
        # N x K
        samples_scores = self.get_unique(samples_scores)
        # N - Effectively making exp(ts) = exp(t) * q
        self.target_score_ = self.target_score_ + tf.log(tf.reshape(q, [-1]))
        # N - Conditioning
        target_scores, samples_scores = self.clip_likelihood(self.target_score_, samples_scores)
        # N
        self.Z_ = tf.exp(target_scores) + tf.reduce_mean(tf.exp(samples_scores), 1)
        # N - The loss of each element in target
        element_loss = target_scores - tf.log(self.Z_ + eps)
        loss = tf.reduce_mean(element_loss)
        return -loss

    def log_likelihood(self, x, h, embedding):
        """
            Abstract method requires to be implement by sub classes

            @Param x: The target words or batch
            @Param h: This is usually the output of neural network
            @Param embedding: The embedding vectors of all words

            @Return log_like: The exact log likelihood average over words
        """
        samples_scores = tf.matmul(h, embedding, transpose_b=True)
        sample_q = self.sampler_.freq_embedding
        samples_scores += tf.log(sample_q)
        target_score = self.target_score_
        Z = tf.reduce_sum(tf.exp(samples_scores), 1)
        log_like = tf.reduce_mean((target_score-tf.log(Z)))
        return log_like

