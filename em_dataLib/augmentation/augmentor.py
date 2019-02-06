from collections import OrderedDict
import numpy as np

from ..options import strToArr

def buildAugmentor(opt):
    aug_opt = strToArr(opt.aug_opt, '@', int)

    aug_param_flip = strToArr(opt.aug_param_flip, '@', float)
    aug_param_elastic = strToArr(opt.aug_param_elastic, '@', float)
    aug_param_color = strToArr(opt.aug_param_color, '@', float)

    return Augmentor(aug_opt, aug_param_flip, aug_param_elastic, aug_param_color)

class DataAugment(object):
    """
    DataAugment interface.
    """
    def prepare(self, spec, **kwargs):
        """Prepare data augmentation.
        Some data augmentation require larger sample size than the original,
        depending on the augmentation parameters that are randomly chosen.
        For such cases, here we determine random augmentation parameters and
        return an updated sample spec accordingly.
        """
        raise NotImplementedError

    def __call__(self, sample, **kwargs):
        """Apply data augmentation."""
        raise NotImplementedError

    def factory(aug_type, **kwargs):
        """Factory method for data augmentation classes."""
        if aug_type is 'box':       return BoxOcclusion(**kwargs)
        if aug_type is 'blur':      return Blur(**kwargs)
        if aug_type is 'flip':      return Flip(**kwargs)
        if aug_type is 'warp':      return Warp(**kwargs)
        if aug_type is 'misalign':  return Misalign(**kwargs)
        if aug_type is 'missing':   return MissingSection(**kwargs)
        if aug_type is 'greyscale': return Greyscale(**kwargs)
        if aug_type is 'longaff': return LongAff(**kwargs)
        assert False, "Unknown data augmentation type: [%s]" % aug_type
    factory = staticmethod(factory)


class Augmentor(object):
    """
    Data augmentor.
    """
    def __init__(self, aug_opt=None, aug_param_flip=None, aug_param_elastic=None, aug_param_color=None):
        self._augments = list()
        self._spec = {}
        # 1. non-rigid deformation
        if aug_opt[0]>0: # elastic
            self.append('warp', aug_param_warp)
        # 2. rigid deformation
        if aug_opt[1]!=0: # elastic
            self.append('flip', aug_opt[1])
        if aug_opt[2]!=0: # flip/swap
            self.append('flip', aug_opt[1])
        # 3. appearance
        if aug_opt[3]!=0: # greyscale
            self.append('greyscale', aug_param_grey)

    def append(self, aug, **kwargs):
        # append either augmenation function or string
        if isinstance(aug, DataAugment):
            aug.update_spec(self._spec)
        elif type(aug) is str:
            aug = DataAugment.factory(aug, self._spec, **kwargs)
        else:
            assert False, "Bad data augmentation " + aug
        self._augments.append(aug)

    def prepare(self):
        """Prepare random parameters and modify sample spec accordingly."""
        for aug in self._augments:
            aug.prepare(self._spec)
        # get sample size
        return dict(spec)

    def getSize(self):
        if 'sz' in self._spec:
            return self._spec['sz']
        else:
            return None

    def __call__(self, sample):
        # Apply a list of data augmentation.
        for aug in self._augments:
            sample = aug(sample)
        return sample
