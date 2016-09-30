import tensorflow as tf
from .Sampler import Sampler


class UnigramSampler(Sampler):
    def draw_sample(self, targets, num_targets, distort=1.0):
        """
            Draw samples from unigram distribution

            @Param targets(N): The target words or batch
            @Param num_targets: The length of target words or batch
        """
        ss, _, _ = tf.nn.fixed_unigram_candidate_sampler(tf.reshape(targets, (-1, 1)),
                                                         num_targets, self.num_samples_, self.unique_,
                                                         self.num_classes_, unigrams=self.proposed_dist_, distortion=distort)
        tc = tf.pow(tf.nn.embedding_lookup(self.proposed_dist_, targets), distort)
        sc = tf.pow(tf.nn.embedding_lookup(self.proposed_dist_, ss), distort)
        return ss, tc, sc
