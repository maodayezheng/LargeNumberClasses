from __future__ import print_function
from .Estimator import Estimator

import theano.tensor as T
from theano.gradient import zero_grad


class ImportanceEstimator(Estimator):
    def loss(self, h, targets, target_ids, target_qs,
             samples, sample_ids, sample_qs, eps=1e-8):
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
        samples_scores -= T.log(sample_qs).dimshuffle('x', 0)
        # N x (K + 1)
        max_t = T.max(samples_scores, axis=1)
        max_t = zero_grad(max_t)
        samples_scores = samples_scores - max_t.dimshuffle(0, 'x')
        target_scores = target_scores - max_t
        Z = T.sum(T.exp(samples_scores), axis=1)

        element_loss = target_scores - T.log(Z)
        loss = T.mean(element_loss)
        return -loss

        # Condition scores
        # target_scores, samples_scores = Estimator.clip_likelihood(target_scores, samples_scores)
        # # N
        # Z = T.sum(T.exp(samples_scores), 1)
        # element_loss = target_scores - T.log(Z + eps)
        # loss = T.mean(element_loss)
        # return -loss
