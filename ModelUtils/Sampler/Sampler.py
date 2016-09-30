class Sampler(object):
    def __init__(self, num_classes, num_samples, proposed_dist= None, unique=True):
        """
        The constructor of Sampler

        @Param num_classes: The total number of different classes
        @Param num_samples: Number of samples in sample set
        @Param proposed_dist: The proposed distribution for draw sample
        @Param unique: To determine whether there is repeated samples in sample set
        """
        self.num_classes_ = num_classes
        self.num_samples_ = num_samples
        self.unique_ = unique
        self.proposed_dist_ = proposed_dist

    def draw_sample(self, targets, num_targets):
        """
        Abstract method requires to implement by sub class

        @Param targets: The target words or batch
        @Param num_target: The number of targets
        """
        raise Exception("Can not call abstract method draw_sample in Sampler")
