import numpy as np

from em_segLib.transform import flip, crop
from .augmentor import DataAugment

class Flip(DataAugment):
    """
    Random flip.
    """

    def __init__(self, spec, param=[0]):
        """Initialize parameters.
        Args:
            mode=param[0]: -1='random', 0~15 for different inference choice 
        """
        self.mode = param[0]
        if self.mode != -1:
            self.aug_rot = [int(x) for x in '{0:04b}'.format(self.mode)]
        
        self.aug_rot_st = np.ones((3,3), dtype=int)
        spec['rot_pad'] = np.zeros(3, dtype=int)
        if self.mode > 0: # need to pad 1 to recompute the flipped affinity
            spec['rot_pad'] = np.array(self.aug_rot[:3])
            for i in range(3):
                self.aug_rot_st[i,i] = 1-self.aug_rot[i]

    def update_spec(self, spec):
        # append augment -> update spec 
        spec['rot_pad'] = np.zeros(3, dtype=int)
        if self.mode > 0: # need to pad 1 to recompute the flipped affinity
            spec['rot_pad'] = np.array(self.aug_rot[:3])

    def prepare(self, spec):
        # before data sample
        if self.mode == -1: # random
            # rules
            self.aug_rot = ''.join([str(int(x>0.5)) for x in np.random.random(4)])
            # for sampler to sample volume
            spec['rot_pad'] = np.array(self.aug_rot[:3])
            # for augmentor to flip/crop volume
            self.aug_rot_st[:] = 1
            for i in range(3):
                self.aug_rot_st[i,i] = 1- self.aug_rot[i]

    def __call__(self, sample):
        # after data sample
        for k, v in sample.iteritems():
            sample[k] = flip(v, self.aug_rot, k, max(self.aug_rot[:3])==1, self.aug_rot_st)
        return sample
