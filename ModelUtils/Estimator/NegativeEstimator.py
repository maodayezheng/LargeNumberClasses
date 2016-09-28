import tensorflow as tf

from Estimator import Estimator


class NegativeEstimator(Estimator):
    def loss(self, x, h):
        """
        Calculate the estimate loss of negative sampling approximation

        @Param x: the target word or batch
        @Param h: This is usually the output of neural network
        """
        return tf.reduce_mean(tf.log(tf.nn.sigmoid(tf.matmul(x, h))) - tf.reduce_sum(tf.log(tf.nn.sigmoid(
                                                                             tf.matmul(self.samples_, h))), 0))

    def likelihood(self, x, h):
        """
            Calculate the estimate likelihood of negative sampling approximation

            @Param x: the target word or batch
            @Param h: This is usually the output of neural network
        """
