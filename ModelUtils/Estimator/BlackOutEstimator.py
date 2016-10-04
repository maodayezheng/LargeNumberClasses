from __future__ import print_function
import tensorflow as tf
from .Estimator import Estimator


class BlackOutEstimator(Estimator):
    def __init__(self, *args, **kwargs):
        super(BlackOutEstimator, self).__init__(extra=10, *args, **kwargs)

    def loss(self, x, h, q=None, eps=1e-9):
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
        self.target_score_ = tf.reduce_sum(x * h, 1)
        target_scores = self.target_score_ - tf.log(tf.reshape(q, [-1]))
        # N x KE
        samples_scores = tf.matmul(h, samples, transpose_b=True)
        samples_scores -= tf.log(weights)
        # N x K
        samples_scores = self.get_unique(samples_scores)
        # Conditioning
        max_t = tf.reduce_max(tf.concat(1, (tf.reshape(target_scores, (-1, 1)), samples_scores)), 1)
        m = tf.stop_gradient(max_t)
        target_scores -= m
        samples_scores -= tf.reshape(m, (-1, 1))
        # N
        self.Z_ = tf.exp(target_scores) + tf.reduce_sum(tf.exp(samples_scores), 1)
        # N x K
        neg_scores = tf.log(tf.reshape(self.Z_, (-1, 1)) - tf.exp(samples_scores) + eps)
        # The loss of each element in target
        # N
        element_loss = target_scores - tf.log(q) + tf.reduce_sum(neg_scores, 1) -\
            (tf.cast(tf.shape(samples_scores)[1], dtype=tf.float32) + 1.0) * tf.log(self.Z_ + eps)
        loss = tf.reduce_mean(element_loss)
        return -loss


