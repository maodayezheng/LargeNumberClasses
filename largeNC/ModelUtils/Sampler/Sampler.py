import theano
import numpy as np
from scipy.stats import rv_discrete


class Sampler(object):
    def __init__(self, num_classes, num_samples,
                 distortion=1.0,
                 proposed_dist=None,
                 extra=0):
        """
        The constructor of Sampler

        @Param num_classes: The total number of different classes
        @Param num_samples: Number of samples in sample set
        @Param proposed_dist: The proposed distribution for draw sample
        @Param unique: To determine whether there is repeated samples in sample set
        """
        self.num_classes_ = num_classes
        self.num_samples_ = num_samples
        self.proposed_dist_ = proposed_dist
        self.distortion_ = distortion
        self.extra = extra
        distorted_freq = np.power(proposed_dist, distortion)
        distorted_freq = distorted_freq / np.sum(distorted_freq)
        self.freq_embedding = theano.shared(distorted_freq.astype(theano.config.floatX), "freq_embeddings")
        v = np.stack((np.arange(num_classes), distorted_freq))
        self.sampler = rv_discrete(name="sampler", values=v)

    def draw_sample(self):
        """
        Draw samples from the underlying distribution
        :return:
        """
        return self.sampler.rvs(size=(self.num_samples_ + self.extra, )).astype("int32")
