import tensorflow as tf
from Sampler import Sampler


class UniformSampler(Sampler):
    def draw_sample(self, targets, num_targets):
        """
        Draw samples from uniform distribution

        @Param targets: The target words or batch
        @Param num_targets: The length of target words or batch
        """
        ss, _, _ = tf.nn.uniform_candidate_sampler(targets, num_targets, self.num_samples_,
                                               self.unique_, self.num_classes_)
        tc = tf.ones_like(targets) / self.num_classes_
        sc = tf.ones_like(ss) / self.num_classes_
        return ss, tc, sc

