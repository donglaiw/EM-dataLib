
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

    def __init__(self, opt_sample=None, opt_data=None, opt_aug=None):
        # Initialize attributes.
        self.data = DataLoader(**opt_data)
        self.aug = Augmentor(**opt_aug)
        self.sampler = Sampler(self.data, self.aug, **opt_sampler)

    def next_sample(self):
        return self.sampler.next_sample()

    def random_sample(self):
        return self.sampler.random_sample()

########################################################################
## VolumeDataset demo.
########################################################################
if __name__ == "__main__":

    import argparse
    import emio
    import h5py
    import os
    import time
    import transform

    dsc = 'VolumeDataset demo.'
    parser = argparse.ArgumentParser(description=dsc)

    parser.add_argument('z', type=int, help='sample z dim.')
    parser.add_argument('y', type=int, help='sample y dim.')
    parser.add_argument('x', type=int, help='sample x dim.')
    parser.add_argument('img', help='image file (h5 or tif) path.')
    parser.add_argument('lbl', help='label file (h5 or tif) path.')

    args = parser.parse_args()

    # Load data.
    img = emio.imread(args.img)
    lbl = emio.imread(args.lbl)

    # Preprocess.
    img = transform.divideby(img, val=255.0)

    # Create dataset and add data.
    vdset = VolumeDataset()
    vdset.add_raw_data(key='input', data=img)
    vdset.add_raw_data(key='label', data=lbl)

    # Random sample.
    size = (args.z, args.y, args.x)
    spec = dict(input=size, label=size)
    vdset.set_spec(spec)
    sample = vdset.random_sample()

    # Dump a single random sample.
    print 'Save as file...'
    fname = 'sample.h5'
    if os.path.exists(fname):
        os.remove(fname)
    f = h5py.File(fname)
    for key, data in sample.iteritems():
        f.create_dataset('/' + key, data=data)
    f.close()
