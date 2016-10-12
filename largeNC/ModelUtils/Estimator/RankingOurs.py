from __future__ import print_function
from .Estimator import Estimator

from theano.printing import Print
import theano.tensor as T


def theano_print(var, msg):
    pr = Print(msg)(T.stack(T.min(var), T.max(var)))
    return T.switch(T.lt(0, 1), var, pr[0])


class RankingOursEstimator(Estimator):
    def __init__(self):
        super(RankingOursEstimator, self).__init__(10)

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
        # N x KE
        samples_scores = T.dot(h, samples.T)
        # Removing log Q\i(k) = log Q(k) - log(1.0 - Q(i))
        a = - T.log(sample_qs).dimshuffle('x', 0) + \
            T.log(T.constant(1) - target_qs).dimshuffle(0, 'x')
        # a = theano_print(a, "a")
        samples_scores = samples_scores + a
        # N x K
        samples_scores = self.get_unique(target_ids, sample_ids, samples_scores)
        # N
        element_loss = - T.nnet.softplus(samples_scores - target_scores.dimshuffle(0, 'x'))
        loss = T.mean(element_loss)
        return -loss