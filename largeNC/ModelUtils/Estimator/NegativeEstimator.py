from __future__ import print_function
from .Estimator import Estimator

import theano.tensor as T


class NegativeEstimator(Estimator):
    def loss(self, h, targets, target_ids, target_qs,
             samples, sample_ids, sample_qs, eps=1e-9):
        """
         Calculate the estimate loss of negative sampling approximation
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
        # N x K
        samples_scores = T.dot(h, samples.T)
        # N
        pos_loss = T.log(T.nnet.sigmoid(target_scores))
        neg_loss = T.sum(T.log(T.nnet.sigmoid(- samples_scores)), 1)
        element_loss = pos_loss + neg_loss
        loss = T.mean(element_loss)
        return -loss
