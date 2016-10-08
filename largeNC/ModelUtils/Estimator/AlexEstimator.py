from __future__ import print_function
from .Estimator import Estimator

import theano
import theano.tensor as T


class AlexEstimator(Estimator):
    def __init__(self):
        super(AlexEstimator, self).__init__(10)

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
        # N
        target_scores = T.sum(h * targets, 1)
        target_scores += T.log(target_qs)
        # N x KE
        samples_scores = T.dot(h, samples.T)
        # N x K
        samples_scores = self.get_unique(target_ids, sample_ids, samples_scores)
        # Essentially dividing by K
        checker = T.printing.Print("Check the shape[1] : ")
        checking = theano.function(samples_scores.shape[1], [checker])
        samples_scores -= T.log(T.cast(samples_scores.shape[1], theano.config.floatX))
        # N x (K + 1)
        merged = T.concatenate((target_scores.dimshuffle(0, 'x'), samples_scores), axis=1)
        # Take a standard softmax
        softmax = T.nnet.softmax(merged)
        # Need only first column
        element_loss = T.log(softmax[:, 0])
        loss = T.mean(element_loss)
        return -loss

    def log_likelihood(self, embedding_matrix, freq_embedding, h, target_ids, target_qs=None):
        # N x V
        all_scores = T.dot(h, T.transpose(embedding_matrix))
        all_scores += T.log(freq_embedding.dimshuffle('x', 0))
        # N x V
        softmax = T.nnet.softmax(all_scores)
        # N
        target_softmax = softmax[T.arange(target_ids.shape[0]), target_ids]
        # N
        return T.mean(T.log(target_softmax))

