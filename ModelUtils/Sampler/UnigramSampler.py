import tensorflow as tf
from .Sampler import Sampler


class UnigramSampler(Sampler):
    def draw_sample(self, targets, num_targets):
        """
            Draw samples from unigram distribution

            @Param targets(N): The target words or batch
            @Param num_targets: The length of target words or batch
        """
        ss, _, _ = tf.nn.fixed_unigram_candidate_sampler(targets,
                                                         num_targets, self.num_samples_-1, self.unique_,
                                                         self.num_classes_, unigrams=self.proposed_dist_[1:],
                                                         distortion=self.distortion_)

        tc = tf.nn.embedding_lookup(self.freq_embedding, targets)
        tc = tf.squeeze(tc)
        sc = tf.squeeze(tf.nn.embedding_lookup(self.freq_embedding, ss+1))
        return ss, tc, sc
