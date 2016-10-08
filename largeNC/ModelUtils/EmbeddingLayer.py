import numpy as np
import theano
import theano.tensor as T


class EmbeddingLayer(object):
    def __init__(self, num_class, dim, name):
        """
        The constructor of Embedding layer

        @Param name: A string name
        @Param num_class: The number of different classes in embedding
        @Param dim: The dimension of embedding vector
        """
        self.name_ = name
        self.dim = dim
        init = np.random.rand(num_class, dim) / 5.0 - 0.1
        self.embedding_ = theano.shared(init.astype(theano.config.floatX), name=self.name_)

    def __call__(self, word_ids):
        """
        :param word_ids: the indices of the words to look up
        :return: the embeddings of those words
        """
        return self.embedding_[word_ids]

    def l2_regular(self, word_ids):
        return T.sum(self.embedding_[word_ids] ** 2)

    def set_params(self, values):
        """
        :param values: Should be a dict mapping name: np.ndarray
        """
        set_keys = []
        for k, v in values:
            for p in self.get_params():
                if p.name == k:
                    if np.any(p.get_value().shape != v.shape):
                        raise ValueError("Parameter %s with shape " % k, p.get_value().shape,
                                         "does not match provided value shape", v.shape)
                    p.set_value(v)
                    set_keys.append(k)
        for k in values.keys():
            if k not in set_keys:
                print("Value for parameter with name %s was not used for Embedding" % k)

    def get_params(self):
        return [self.embedding_]

