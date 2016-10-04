import tensorflow as tf
from .Sampler import Sampler


class UnigramSampler(Sampler):
    def draw_sample(self, targets, num_targets):
        """
            Draw samples from unigram distribution

            @Param targets(N): The target words or batch
            @Param num_targets: The length of target words or batch
        """
        targets = tf.reshape(targets, (-1, 1))
        # Sample one class less because of the 0th padding token and shift the indices +1
        ss, _, _ = tf.nn.fixed_unigram_candidate_sampler(targets,
                                                         num_targets, self.num_samples_, self.unique_,
                                                         self.num_classes_ - 1, unigrams=self.proposed_dist_[1:],
                                                         distortion=self.distortion_) + 1

        tc = tf.nn.embedding_lookup(self.freq_embedding, targets)
        tc = tf.squeeze(tf.pow(tc, self.distortion_))
        sc = tf.squeeze(tf.pow(tf.nn.embedding_lookup(self.freq_embedding, ss), self.distortion_))
        return ss, tc, sc
