import tensorflow as tf
from Estimator import Estimator


class BernoulliEstimator(Estimator):
    def loss(self, x, h, q=None):
        """
            Calculate the estimate loss of bernoulli sampling approximation

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
        if q is None:
            raise ValueError("target word weight must be provided")

        domain = tf.matmul(x, h)
        normalizor = tf.matmul(tf.exp(tf.mul(samples, h)), weights) + tf.exp(domain) * q
        return tf.reduce_mean(domain - tf.log(normalizor))

    def likelihood(self, x, h, q=None):
        """
            Calculate the estimate likelihood of bernoulli sampling approximation

            @Param x: The target word or batch
            @Param h: This is usually the output of neural network
            @Param q: The Weight of target
        """
        print "likelihood"