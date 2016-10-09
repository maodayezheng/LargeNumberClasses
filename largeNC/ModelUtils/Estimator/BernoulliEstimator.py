from __future__ import print_function
from .Estimator import Estimator

import theano
import theano.tensor as T


class BernoulliEstimator(Estimator):
    def __init__(self):
        super(BernoulliEstimator, self).__init__(10)

    def loss(self, h, targets, target_ids, target_qs,
             samples, sample_ids, sample_qs, eps=1e-8):
        """
        Calculate the estimate loss of blackout approximation
        :param h: NxD
        :param targets: NxD
        :param target_ids: N
        :param target_qs: N
        :param samples: KxD
        :param sample_ids: K
        :param sample_qs: K
        :param eps: conditioning number
        :return:
        """
        # N
        target_scores = T.sum(h * targets, 1)
        # N x KE
        samples_scores = T.dot(h, samples.T)
        samples_scores -= T.log(sample_qs).dimshuffle('x', 0)
        # N x K
        samples_scores = self.get_unique(target_ids, sample_ids, samples_scores)
        # Essentially dividing by K
        samples_scores -= T.log(T.cast(samples_scores.shape[1], theano.config.floatX))
        # N x (K + 1)
        merged = T.concatenate((target_scores.dimshuffle(0, 'x'), samples_scores), axis=1)
        # Take a standard softmax
        softmax = T.nnet.softmax(merged)
        # Need only first column
        element_loss = T.log(softmax[:, 0] + eps)
        loss = T.mean(element_loss)
        return -loss
