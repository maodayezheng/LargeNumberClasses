import tensorflow as tf
from ModelUtils.Sampler import Sampler


class UniformSampler(Sampler):
    def draw_sample(self, targets, num_targets):
        """
        Draw samples from uniform distribution

        @Param targets: The target words or batch
        @Param num_targets: The length of target words or batch
        """
        return tf.nn.uniform_candidate_sampler(targets, num_targets, self.num_samples_,
                                               self.unique_, self.num_classes_)
