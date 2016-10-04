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
        target_scores = tf.reduce_sum(x * h, 1)
        self.target_exp_ = tf.exp(target_scores)
        # N x K
        samples_scores = tf.matmul(h, samples, transpose_b=True)
        target_scores -= tf.reshape(tf.log(q), [-1])
        samples_scores -= tf.reshape(tf.log(weights), (1, -1))
        # Conditioning
        max_t = tf.reduce_max(tf.concat(1, (tf.reshape(target_scores, (-1, 1)), samples_scores)), 1)
        m = tf.stop_gradient(max_t)
        target_scores -= m
        samples_scores -= tf.reshape(m, (-1, 1))
        # N
        self.Z_ = tf.reduce_sum(tf.exp(samples_scores), 1)

        # The loss of each element in target
        # N
        element_loss = target_scores - tf.log(self.Z_ + eps)
        loss = tf.reduce_mean(element_loss)
        return -loss
