import tensorflow as tf
from Estimator import Estimator


class BlackOutEstimator(Estimator):
    def loss(self, x, h, q=None):
        """
            Calculate the estimate loss of blackout approximation

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
        sample_partition= tf.mul(tf.exp(tf.mul(samples, h)), weights)
        normalizor = tf.exp(domain)*q + tf.reduce_sum(sample_partition, 1)
        loss = domain - tf.log(q) + tf.reduce_sum(tf.log(normalizor-sample_partition))
        loss -= self.get_sample_size()*tf.log(normalizor)
        return loss

    def likelihood(self, x, h, q=None):
        """
            Calculate the estimate likelihood of blackout approximation

            @Param x: The target word or batch
            @Param h: This is usually the output of neural network
            @Param q: The Weight of target
        """
        print "likelihood"