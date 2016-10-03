import tensorflow as tf


class EmbeddingLayer(object):
    def __init__(self, num_class, dim, name):
        """
        The constructor of Embedding layer

        @Param name: A string name
        @Param num_class: The number of different classes in embedding
        @Param dim: The dimension of embedding vector
        """
        self.name_ = name + "-embedding"
        self.embedding_ = tf.Variable(tf.random_uniform([num_class, dim], minval=-0.1, maxval=0.1, dtype=tf.float32))
        # The saver is used to store parameters
        self.saver = tf.train.Saver({'embedding': self.embedding_})

    def __call__(self, targets):
        """
        Look up target vector

        @Param targets: The integer type index

        @Return o: The correspond vectors of targets
        """
        o = tf.nn.embedding_lookup(self.embedding_, targets)
        o = tf.squeeze(o)
        return o

    def save_param(self, sess, path):
        """
        Save variables registered in saver

        @Param sess: An active tensorflow session
        @Param path: The target path to save variables
        """
        save_path = self.saver.save(sess, path + self.name_)
        print("The GRU parameters saved to %s " % save_path)

    def restore_param(self, sess, path, name):
        """
        restore saved parameters

        @Param sess: An active tensorflow session
        @Param path: The default path to restore parameter
        @Param name: The file to restore
        """
        if name is None:
            path += self.name_
        else:
            path += name
        self.saver.restore(sess, path)

    def get_embedding(self):
        """
        Get embedding
        """
        return self.embedding_

