from __future__ import print_function
from .Estimator import Estimator

import theano
import theano.tensor as T
from theano.printing import Print


def theano_print_min_max(var, msg):
    pr = Print(msg)(T.stack(T.min(var), T.max(var)))
    return T.switch(T.lt(0, 1), var, pr[0])


def theano_print_shape(var, msg):
    pr = Print(msg)(var.shape)
    return T.switch(T.lt(0, 1), var, T.cast(pr[0], var.dtype))


class BernoulliEstimator(Estimator):
    def __init__(self):
        super(BernoulliEstimator, self).__init__(10)

    def loss(self, h, targets, target_sub_ids, unique_embed, unique_ids, unique_qs, eps=1e-8):
        """
        Calculate the estimate loss of blackout approximation
        :param h: NxD
        :param targets: NxD
        :param target_sub_ids: N
        :param unique_embed: UxD
        :param unique_ids: U
        :param unique_qs: U
        :param eps: conditioning number
        :return:
        """
        # N
        # target_scores = T.sum(h * targets, 1)
        # N x U
        all_scores = T.dot(h, unique_embed.T) - T.log(unique_qs).dimshuffle('x', 0)
        # all_scores = theano_print_shape(all_scores, "shapes")
        # all_scores = theano_print_min_max(all_scores, "min/max")
        # N
        m_t = theano.gradient.zero_grad(T.max(all_scores, axis=1))
        # m_t = theano_print_shape(m_t, "m_t shape")
        # m_t = theano_print_min_max(m_t, "m_t min/max")
        all_scores = all_scores - m_t.dimshuffle(0, 'x')
        soft = T.nnet.softmax(all_scores)[T.arange(target_sub_ids.shape[0]), target_sub_ids]
        element_loss = T.log(soft)
        # N
        # Z = T.sum(T.exp(all_scores), axis=1)
        # Need only first column
        # element_loss = target_scores - T.log(Z)
        loss = T.mean(element_loss)
        # N
        element_logvar = T.log(T.sum(T.exp(all_scores * 2), axis=1))
        return -loss, T.mean(element_logvar)
