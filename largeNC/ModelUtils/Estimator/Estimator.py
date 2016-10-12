import theano.tensor as T
from theano.gradient import zero_grad
import numpy as np
import theano


class Estimator(object):
    def __init__(self, extra=0):
        self.extra = extra

    def loss(self, h, targets, target_sub_ids, unique_embed, unique_ids, unique_qs, eps=1e-9):
        """
        Abstract method requires to be implement by sub classes
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
        raise Exception("Can not call abstract method loss in Estimator")

    def log_likelihood(self, embedding_matrix, freq_embedding, h, target_ids, target_qs=None):
        """
        Abstract method requires to be implement by sub classes
        :param embedding_matrix: VxD
        :param freq_embedding: V
        :param h: NxD
        :param target_ids: N
        :param target_qs: N
        :return:
        """
        # N x V
        all_scores = T.dot(h, T.transpose(embedding_matrix))
        # N x V
        softmax = T.nnet.softmax(all_scores)
        # N
        target_softmax = softmax[T.arange(target_ids.shape[0]), target_ids]
        # N
        return T.mean(T.log(target_softmax))

    @staticmethod
    def clip_likelihood(target_scores, samples_scores):
        """
        Clip the likelihood to ensure it does not go to Inf or Nan
        :param target_scores: N
        :param samples_scores: N x K
        :return:
        """
        merged = T.concatenate((target_scores.dimshuffle(0, 'x'), samples_scores), axis=1)
        max_t = T.max(merged, axis=1)
        max_t = zero_grad(max_t)
        t_scores = target_scores - max_t
        s_scores = samples_scores - max_t.dimshuffle(0, 'x')
        return t_scores, s_scores

    def get_unique(self, target_ids, sample_ids, sample_scores):
        """
        Given K + self.extra sample pick from the samples scores K which are not equal to target
        :param target_ids: N
        :param sample_ids: K
        :param sample_scores: N x K
        :return:
        """
        if self.extra == 0:
            return sample_scores
        N = target_ids.shape[0]
        K = sample_ids.shape[0] - self.extra
        # Indicator with 0 if they DO NOT COINCIDE and 1 if they do since the sorting picks small to big,
        # so first it will pick whoever is False in the condition
        cond = T.eq(target_ids.reshape((N, 1)), sample_ids.reshape((1, -1)))
        i = T.argsort(cond, 1)[:, :K].flatten()
        r = T.arange(N)
        r = T.tile(r, [K]).reshape((K, N)).T.flatten()
        return sample_scores[r, i].reshape((N, K))
        #
        # def clip_likelihood(self, target_scores, samples_scores):
        #     """
        #     Clip the likelihood to ensure it does not go to Inf or Nan
        #
        #     @Param target_scores: The score of target
        #     @Param samples_scores: The score of normalizer
        #
        #     @Return t_score: clipped target_score
        #     @Return s_score: clipped samples_scores
        #     """
        #     max_t = tf.reduce_max(tf.concat(1, (tf.reshape(target_scores, (-1, 1)), samples_scores)), 1)
        #     m = tf.stop_gradient(max_t)
        #     t_scores = target_scores - m
        #     s_scores = samples_scores - tf.reshape(m, (-1, 1))
        #     return t_scores, s_scores
        #
        # def set_sample(self, samples):
        #     self.samples_ = samples
        #
        # def get_samples(self):
        #     return self.samples_
        #
        # def set_sample_weights(self, wieghts):
        #     self.weights_ = wieghts
        #
        # def get_sample_weights(self):
        #     return self.weights_
        #
        # def get_sample_size(self):
        #     return self.sampler_.num_samples_
