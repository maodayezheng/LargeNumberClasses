import tensorflow as tf


class GRU(object):
    def __init__(self, input_dim, hidden_dim, output_dim, name, gate_bias=0.0, output_bias=0.0, update_bias=0.0):
        """
        The constructor of GRU

        @Param input_dim: dimension of input
        @Param hidden_dim: dimension of hidden unit
        @Param output_dim: dimension of output
        @Param name: A string name
        @Param gate_bias: The bias of gate unit
        @Param output_bias: The bias of output unit
        @Param update_bias: The bias of update unit
        """
        self.input_dim_ = input_dim
        self.hidden_dim_ = hidden_dim
        self.output_dim_ = output_dim
        self.name_ = name + "-GRU"
        self.gate_bias_ = tf.Variable(tf.constant(gate_bias))
        self.output_bias_ = tf.Variable(tf.constant(output_bias))
        self.update_bias_ = tf.Variable(tf.constant(update_bias))
        self.w_i_u = tf.Variable(tf.random_uniform(shape=[input_dim, hidden_dim], minval=-0.005, maxval=0.005))
        self.w_i_r = tf.Variable(tf.random_uniform(shape=[input_dim, hidden_dim], minval=-0.005, maxval=0.005))
        self.w_i_c = tf.Variable(tf.random_uniform(shape=[input_dim, hidden_dim], minval=-0.005, maxval=0.005))
        self.w_s_u = tf.Variable(tf.random_uniform(shape=[hidden_dim, output_dim], minval=-0.005, maxval=0.005))
        self.w_s_r = tf.Variable(tf.random_uniform(shape=[hidden_dim, output_dim], minval=-0.005, maxval=0.005))
        self.w_s_c = tf.Variable(tf.random_uniform(shape=[hidden_dim, output_dim], minval=-0.005, maxval=0.005))
        # The saver is used to store parameters
        self.saver = tf.train.Saver({'w_i_u': self.w_i_u, 'w_s_u': self.w_s_u,
                                     'w_s_r': self.w_s_r, 'w_i_r': self.w_i_r,
                                     'w_i_c': self.w_i_c, 'w_s_c': self.w_s_c,
                                     'gate_bias': self.gate_bias_, 'output_bias': self.output_bias_,
                                     'update_bias': self.update_bias_})

    def __call__(self, inputs, state):
        """
        Create the computational graph of GRU

        @Param inputs: The input
        @Param state: The hidden state from previous step

        @Return h, h: The output and hidden state of GRU
        """
        u = tf.sigmoid(tf.matmul(inputs, self.w_i_u) + tf.matmul(state, self.w_s_u) + self.update_bias_)
        r = tf.sigmoid(tf.matmul(inputs, self.w_i_r) + tf.matmul(state, self.w_s_r) + self.gate_bias_)
        a = tf.matmul(inputs, self.w_i_c)
        b = tf.matmul(tf.mul(r, state), self.w_s_c)
        c = tf.tanh(a + b + self.output_bias_)
        h = tf.mul((1 - u), state) + tf.mul(u, c)
        return h, h

    def l2_regular(self):
        """
        The l2 regularization term of GRU

        @Return l2: The l2 regularization term of GRU
        """
        l2 = 0.0
        l2 += tf.nn.l2_loss(self.w_i_u)
        l2 += tf.nn.l2_loss(self.w_s_u)
        l2 += tf.nn.l2_loss(self.w_i_r)
        l2 += tf.nn.l2_loss(self.w_s_r)
        l2 += tf.nn.l2_loss(self.w_i_c)
        l2 += tf.nn.l2_loss(self.w_s_c)
        l2 += tf.nn.l2_loss(self.gate_bias_)
        l2 += tf.nn.l2_loss(self.update_bias_)
        l2 += tf.nn.l2_loss(self.output_bias_)
        return l2

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