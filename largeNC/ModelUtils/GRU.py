import numpy as np
import theano
import theano.tensor as T


class GRU(object):
    def __init__(self, input_dim, hidden_dim, output_dim, name):
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
        self.name_ = name

        self.ur_bias = theano.shared(np.zeros((2*hidden_dim, ), dtype=theano.config.floatX),
                                     name=name + "_ur_bias")
        self.c_bias = theano.shared(np.zeros((hidden_dim, ), dtype=theano.config.floatX),
                                    name=name + "gru_ur_bias")
        self.w_in = theano.shared(init_w((input_dim, 3 * hidden_dim)),
                                  name=name + "gru_w_in")
        self.w_h = theano.shared(init_w((input_dim, 3 * hidden_dim)),
                                 name=name + "gru_w_h")
        self.h_init = theano.shared(init_w((hidden_dim, )),
                                    name=name + "gru_h_init")

    def __call__(self, inputs):
        """
        Compute the GRU outputs for the provided inputs
        :param inputs: NxTxD Tensor of input sentences embeddings
        :return: NxTxD Tensor of outputs of the GRU
        """
        # Precompute inputs to GRU
        n = inputs.shape[0]
        t = inputs.shape[1]
        d = inputs.shape[2]
        precompute_input = T.dot(inputs.reshape((n * t, d)), self.w_in)
        precompute_input = precompute_input.reshape((n, t, 3 * self.hidden_dim_))
        # Change to TxNxD
        precompute_input = precompute_input.dimshuffle((1, 0, 2))
        h_init = T.outer(T.ones((n, )), self.h_init)
        outputs = theano.scan(fn=self.step,
                              sequences=[precompute_input],
                              outputs_info=h_init,
                              non_sequences=self.get_params(),
                              strict=True)[0]
        # Go back to NxTxD
        return outputs.dimshuffle((1, 0, 2))

    def step(self, in_contrib, h_prev, *args):
        precompute_h = T.dot(h_prev, self.w_h)
        ru = T.nnet.sigmoid(in_contrib[:, :2*self.hidden_dim_] +
                            precompute_h[:, :2*self.hidden_dim_] +
                            self.ur_bias.dimshuffle('x', 0))
        r = ru[:, :self.hidden_dim_]
        u = ru[:, self.hidden_dim_:]
        c = T.nnet.sigmoid(in_contrib[:, 2*self.hidden_dim_:] +
                           r * precompute_h[:, 2*self.hidden_dim_:] +
                           self.c_bias.dimshuffle('x', 0))
        return (1 - u) * h_prev + u * c

    def l2_regular(self):
        return sum(T.sum(p*p) for p in self.get_params())

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
                print("Value for parameter with name %s was not used for GRU" % k)

    def get_params(self):
        """
        :return: list of the parameters of the GRU
        """
        return [self.ur_bias, self.c_bias, self.w_in, self.w_h, self.h_init]


def init_w(shape):
    v = np.random.rand(*shape) / 100.0 - 0.005
    return v.astype(theano.config.floatX)
