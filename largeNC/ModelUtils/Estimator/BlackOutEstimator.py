from __future__ import print_function
from .Estimator import Estimator
from .debug import debug

import theano
import theano.tensor as T
from theano.printing import Print
import numpy as np


def theano_print(var, msg):
    pr = Print(msg)(T.stack(T.min(var), T.max(var)))
    return T.switch(T.lt(0, 1), var, pr[0])


class BlackOutEstimator(Estimator):
    def __init__(self):
        super(BlackOutEstimator, self).__init__(10)

    def loss(self, h, targets, target_ids, target_qs,
             samples, sample_ids, sample_qs, eps=1e-9):
        """
        Calculate the estimate loss of blackout approximation
        :param h: NxD
        :param targets: NxD
        :param target_ids: N
        :param target_qs: N
        :param samples: KxD
        :param sample_ids: K
        :param sample_qs: K
        :return:
        """
        # si = theano.shared(np.random.randint(0, 5, (12, )).astype("int32"))
        # ss = theano.shared(np.random.randn(5, 12).astype(theano.config.floatX))
        # sa = self.get_unique(T.arange(5), si, ss)
        # print(si.eval())
        # print(ss.eval())
        # print(sa.eval())
        # N
        target_scores = T.sum(h * targets, 1)
        target_scores -= T.log(target_qs)
        # N x KE
        samples_scores = T.dot(h, samples.T)
        samples_scores -= T.log(sample_qs).dimshuffle('x', 0)
        # N x K
        samples_scores = self.get_unique(target_ids, sample_ids, samples_scores)
        # Condition scores
        # target_scores, samples_scores = Estimator.clip_likelihood(target_scores, samples_scores)
        # N x (K + 1)
        merged = T.concatenate((target_scores.dimshuffle(0, 'x'), samples_scores), axis=1)
        # Take a standard softmax
        softmax = T.nnet.softmax(merged)
        # First column is the target scores
        ts = softmax[:, 0]
        # All other columns are the negative scores
        ns = softmax[:, 0:]
        element_loss = T.log(ts) + T.sum(T.log(1 - ns), axis=1)
        loss = T.mean(element_loss)
        return -loss
        #
        # exp_ss = T.exp(samples_scores)
        # Z = T.exp(target_scores) + T.sum(exp_ss, 1)
        # Z = theano_print(Z, "Z=")
        # exp_ss = theano_print(exp_ss, "exp_ss=")
        # target_scores = theano_print(target_scores, "ts=")
        # # Z = debug(Z, 'Z check', 1, lambda x: x < 0.5,
        # #                         check_not_all=False,
        # #                         check_not_any=True,
        # #                         raise_on_failed_check=True)
        # # Z = debug(Z, 'Z check2', 1, lambda x: x > 0.5,
        # #           check_not_all=False,
        # #           check_not_any=True,
        # #           raise_on_failed_check=True)
        # # N x K
        # neg_scores = T.log(Z.dimshuffle(0, 'x') - exp_ss + eps)
        # K1 = T.cast(samples_scores.shape[1] + 1, theano.config.floatX)
        # element_loss = target_scores + T.sum(neg_scores) - K1 * T.log(Z + eps)





