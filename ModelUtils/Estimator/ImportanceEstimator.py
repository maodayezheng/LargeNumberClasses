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
        #samples_scores = tf.Print(samples_scores, [tf.shape(samples_scores)], message="The shape of sample score is : ")
        log_weights = tf.check_numerics(tf.log(weights), "each weights")
        log_q = tf.check_numerics(tf.log(q), "each q")
        #target_scores = tf.Print(target_scores, [tf.shape(log_q)], message="The shape of q is :")
        #target_scores = tf.Print(target_scores, [tf.shape(target_scores)], message="The shape of ts is :")
        target_scores -= tf.reshape(log_q, [-1])
        #target_scores = tf.Print(target_scores, [tf.shape(target_scores)], message="The shape of ts2 is :")
        samples_scores -= tf.reshape(log_weights, (1, -1))
        max_t = tf.reduce_max(tf.concat(1, (tf.reshape(target_scores, (-1, 1)), samples_scores)), 1)
        # max_t = tf.maximum(target_scores, samples_scores)
        #max_t = tf.Print(max_t, [tf.shape(max_t)], message="The shape of max_t is :")
        m = tf.stop_gradient(max_t)
        target_scores -= m
        samples_scores -= tf.reshape(m, (-1, 1))
        # N
        exp_weight = tf.exp(samples_scores)
        self.Z_ = tf.reduce_sum(tf.check_numerics(exp_weight, "each Z"), 1)

        # The loss of each element in target
        # N
        #Z = tf.Print(self.Z_, [tf.reduce_max(self.Z_), tf.reduce_min(self.Z_)], message="The max and min Z")
        element_loss = tf.check_numerics(target_scores, message="Target score ") - tf.check_numerics(tf.log(tf.check_numerics(self.Z_ + eps, message="The Z")), message="log Z")
        element_loss = tf.check_numerics(element_loss, "each element_loss")
        loss = tf.reduce_mean(element_loss)
        return -loss
