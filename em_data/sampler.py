from collections import OrderedDict
import numpy as np

from augmentor import DataAugment, Augmentor
from em_segLib.seg_util import countVolume
# based on: https://github.com/torms3/DataProvider/blob/refactoring/python/dataprovider/data_provider.py

def getVolumeSampler(opt):
    img_name={}
    img_name_dst={}
    seg_name={}
    seg_name_dst={}
    modes = ['train', 'val'] if opt.mode=='train' else ['test']
    vol_input = optArr(opt.vol_input)
    vol_output = optArr(opt.vol_output)

    for nn in modes:
        img_name[nn] = getattr(opt, nn+'_img').split('@')
        img_name_dst[nn] = getattr(opt, nn+'_img_name').split('@')
        seg_name[nn] = getattr(opt, nn+'_seg').split('@')
        seg_name_dst[nn] = getattr(opt, nn+'_img_name').split('@')
    return VolumeSampler()

# different sample stragegy
class VolumeSamplerTrain(object):
    def __init__(self, mode, data_loader=None, dataset_p=None, drange=None,
                 pre=None, aug=None, post=None):
        super(VolumeSamplerTrain, self).__init__()
        self.set_sampling_weights(dataset_p)

    def set_sampling_weights(self, p=None):
        """Set probability of each dataset being chosen at each sampling."""
        if p is None:
            p = self.data_loader.img_size
        # Normalize.
        p = np.asarray(p, dtype='float32')
        p = p/np.sum(p)
        # Set sampling weights.
        self.p = p

    def random_dataset_id(self):
        """Pick one dataset randomly."""
        assert len(self.datasets) > 0
        if self.drange is None:
            drange = range(len(self.datasets))
            p = self.p
        else:
            p = [self.p[d] for d in drange]
            # Normalize again.
            p = np.array(p)
            p /= p.sum()
        idx = np.random.choice(len(drange), size=1, p=p)
        return idx[0]

    def getPos(self):
        # random sample which dataset
        pos = [0,0,0,0]
        pos[0] = self.random_dataset_id()
        # sample one augmentation configuration
        spec = self.augment.prepare()

        pos[1:] = []
        return pos

    def sample(self, index):
        # index: dummy variable
        pos, spec = self.getPos()
        # no warping aug
        data = self.data_loader.getPatch(pos[0], pos[1:], spec, 'img-seg')
        return self._process(data)

class VolumeSamplerTest(object):
    def __init__(self, data_loader=None, pre=None, aug=None, post=None, 
                 sample_border=np.array([0,0,0]), sample_stride=np.array([1,1,1])):
        super(VolumeSamplerTest, self).__init__(data_loader, pre, aug, post)
        # border: extra pad for the test volume
        self.sample_border = sample_border
        self.sample_stride = sample_stride
        self.sample_size = [countVolume(x+2*sample_border, self.data_loader.vol_input, sample_stride) \
                                for x in self.data_loader.img_shape]
        self.sample_num = np.array([np.prod(x) for x in self.sample_size], dtype=int)
        # index range
        self.sample_num_total = np.sum(self.sample_num)
        # index -> dataset_id
        self.sample_num_cumsum = np.cumsum([0]+self.sample_num)
        # index -> zyx
        self.sample_size_vol = [np.array([np.prod(x[1:3]),x[2]], dtype=int) for x in self.sample_size]

    def index2zyx(self, index): # for test
        # int division = int(floor(.))
        pos = [0,0,0,0]
        did = np.argmax(index<self.sample_num_cumsum)-1 
        pos[0] = did
        index2 = index - self.sample_num_c[did]
        pos[1:] = self.pos2zyx(index2, self.sample_size_vol[did])
        return pos
    
    def getPos(self, index):
        # which dataset
        pos = self.index2zyx(index)
        # take care of the boundary case
        for i in range(1,4):
            if pos[i] != self.sample_size[pos[0]][i-1]-1:
                pos[i] = pos[i] * self.sample_stride[i-1] - self.sample_border
            else:
                pos[i] = self.img_size[pos[0]][i-1] + self.sample_border[i-1] - self.vol_img_size[i-1]
        return pos

    def sample(self, index):
        pos = self.getPos(index)
        # no warping aug
        data = self.data_loader.getPatch(pos[0], pos[1:], self.data_loader.vol_input, 'img')
        return self._process(data)

# implement different scheduler
class VolumeSampler(object):
    """
    Sampler for volumetric data.
    Attributes:
        data: data loader.
        preprocessor: Sample transformer, before augmentation.
        augmentor: Sample augmentor.
        postprocessor: Sample transformer, after augmentation.
    """

    def __init__(self, mode, data_loader=None, dataset_p=None, drange=None,
                 pre=None, aug=None, post=None):
        # Datasets.
        self.data_loader = data_loader
        # Preprocessing.
        self.set_preprocessor(pre)
        # Data augmentation.
        self.set_augmentor(aug)
        # Postprocessing.
        self.set_postprocessor(post)
        # Sampling weights.

    def pos2zyx(self, index, sz):
        # sz: [y*x, x]
        pos= [0,0,0]
        pos[0] = index / sz[0]
        pz_r = index % sz[0]
        pos[1] = pz_r / sz[1]
        pos[2] = pz_r % sz[1]
        return pos

    def sample(self, index):
        raise NotImplementedError("Need to implement sample() !")

    def _process(self, mode, **kwargs):
        # Pick sample randomly.
        spec = dataset.get_spec()
        sample = cropPad(self.data_loader, mode+'_sample')(spec=spec)
        # Preprocessing.
        sample = self.preprocess(sample, **params)
        # Apply data augmentation.
        sample = self.augment(sample, **params)
        # Postprocessing.
        sample = self.postprocess(sample, **params)
        # Ensure that sample is ordered by key.
        return OrderedDict(sorted(sample.items(), key=lambda x: x[0]))

    ####################################################################
    ## Setters.
    ####################################################################

    def add_dataset(self, dataset):
        assert isinstance(dataset, Dataset)
        self.datasets.append(dataset)

    def add_preprocess(self, tf):
        assert isinstance(tf, Transform)
        self.preprocess.append(tf)

    def add_augment(self, aug):
        assert isinstance(aug, DataAugment)
        self.augment.append(aug)

    def add_postprocess(self, tf):
        assert isinstance(tf, Transform)
        self.postprocess.append(tf)

    def set_preprocessor(self, tf):
        if isinstance(tf, Transformer):
            self.preprocess = tf
        else:
            self.preprocess = Transformer()

    def set_augmentor(self, aug):
        if isinstance(aug, Augmentor):
            self.augment = aug
        else:
            self.augment = Augmentor()

    def set_postprocessor(self, tf):
        if isinstance(tf, Transformer):
            self.postprocess = tf
        else:
            self.postprocess = Transformer()
