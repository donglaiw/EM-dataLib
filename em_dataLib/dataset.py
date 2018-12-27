from collections import OrderedDict
import copy
import numpy as np

class Dataset(object):
    """
    Dataset interface.
    """
    def next_sample(self):
        raise NotImplementedError

    def random_sample(self):
        raise NotImplementedError

class VolumeDataset(Dataset):
    """
    Dataset for volumetric data.
    Attributes:
        _params: Dataset-specific parameters.
        _data: Dictionary mapping key to TensorData, each of which contains
                4D volumetric data. (e.g. EM image stacks, segmentation, etc.)
        _spec: Sample specification. Dictionary mapping key to dimension,
                which can be either a list or tuple with at least 3 elements.
        _range: Range of valid coordinates for accessing data given the sample
                spec. It depends both on the data and sample spec.
        _sequence:
        _locs: Valid locations.
    """

    def __init__(self, opt):
        # Initialize attributes.
        self.data = optToDataLoader(opt)
        self.aug = optToAugmentor(opt)
        self.sampler = optToSampler(self.data, self.aug, opt)

    def next_sample(self):
        return self.sampler.next_sample()

    def random_sample(self):
        return self.sampler.random_sample()

