import tensorflow as tf
from Estimator import Estimator


class ImportanceEstimator(Estimator):
    def loss(self, x, h, q=None):
        """
            Calculate the estimate loss of Importance sampling approximation

            @Param x: The target word or batch
            @Param h: This is usually the output of neural network
            @Param q: The Weight of target
        """
        weights = self.get_sample_weights()
        if weights is None:
            raise ValueError("sample weights must be set")
        samples = self.get_samples()
        if samples is None:
            raise ValueError("samples must be set")
        domain = tf.matmul(tf.transpose(x), h)
        normalizor = tf.matmul(tf.exp(tf.matmul(samples, h)), weights)
        loss = tf.reduce_mean(domain - tf.log(q) - tf.log(normalizor))
        return loss

    def likelihood(self, x, h, q=None):
        """
            Calculate the estimate likelihood of importance sampling approximation

            @Param x: The target word or batch
            @Param h: This is usually the output of neural network
            @Param q: The Weight of target
        """
        print "likelihood"
