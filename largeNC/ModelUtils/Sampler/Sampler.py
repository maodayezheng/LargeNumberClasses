import theano
import numpy as np
# from scipy.stats import rv_discrete

from functools import partial
from multiprocessing import Pool


def sampler(_, num_classes, replace, p, size):
    return np.random.choice(np.arange(num_classes),
                            replace=replace,
                            p=p,
                            size=size)

class Sampler(object):
    def __init__(self, num_classes, num_samples,
                 distortion=1.0,
                 proposed_dist=None,
                 replace=True,
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
        self.distorted_freq = np.power(proposed_dist, distortion)
        self.distorted_freq = self.distorted_freq / np.sum(self.distorted_freq)
        self.freq_embedding = theano.shared(self.distorted_freq.astype(theano.config.floatX), "freq_embeddings")
        self.replace = replace
        # v = np.stack((np.arange(num_classes), self.distorted_freq))
        # self.sampler = rv_discrete(name="sampler", values=v)

    def draw_sample(self, shape=None):
        """
        Draw samples from the underlying distribution
        :return:
        """
        shape = (self.num_samples_, ) if shape is None else shape
        if self.replace:
            return np.random.choice(np.arange(self.num_classes_),
                                    p=self.distorted_freq,
                                    size=shape).astype("int32")
        else:
            if len(shape) == 1:
                return np.random.choice(np.arange(self.num_classes_),
                                        replace=False,
                                        p=self.distorted_freq,
                                        size=shape).astype("int32")
            else:
                pool = Pool(8)
                func = partial(sampler, num_classes=self.num_classes_, replace=False,
                               p=self.distorted_freq, size=shape[1:])
                result = np.stack(pool.map(func, range(shape[0]))).astype("int32")
                pool.close()
                return result
