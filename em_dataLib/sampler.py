import numpy as np

from em_segLib.seg_util import countVolume

from augmentor import DataAugment, Augmentor
from options import strToArr
# based on: https://github.com/torms3/DataProvider/blob/refactoring/python/dataprovider/data_provider.py

def getVolumeSampler(mode, data_loader, aug, opt):
    stride = strToArr(opt.sample_stride)
    if mode=='train':
        return VolumeSamplerTrain(data_loader, aug, opt.size_input)
    elif mode=='test':
        return VolumeSamplerTest(data_loader, aug, opt.size_input, opt.size_output, 
                                 stride)

# base class
class VolumeSampler(object):
    """
    Sampler for volumetric data.
    Attributes:
        data: data loader.
        preprocessor: Sample transformer, before augmentation.
        augmentor: Sample augmentor.
        postprocessor: Sample transformer, after augmentation.
    """

    def __init__(self, data_loader=None, aug=None, 
                 size_input=[1,1,1], size_output=[1,1,1], sample_stride=[1,1,1]):
        # Datasets.
        self.data_loader = data_loader

        # Data augmentation.
        self.augmentor = aug

        # set data specs
        self.sample_stride = np.array(sample_stride)
        self.size_input = np.array(size_input)
        self.initParam()

    def initParam(self):
        self.size_input = size_input
        self.size_output = size_output
        self.img_size = [x[1:] for x in self.data_loader.img_shape] # volume size
        # compute number of samples for each dataset
        self.sample_size = [ countVolume(x, self.size_input, self.sample_stride) \
                            for x in self.img_size]
        self.sample_num = np.array([np.prod(x) for x in self.sample_size], dtype=int)
        self.sample_num_a = np.sum(self.sample_num)
        self.sample_num_c = np.cumsum([0]+self.sample_num)


    def pos2zyx(self, index, sz):
        # sz: [y*x, x]
        pos= [0,0,0]
        pos[0] = index / sz[0]
        pz_r = index % sz[0]
        pos[1] = pz_r / sz[1]
        pos[2] = pz_r % sz[1]
        return pos

    def sample(self, index, mode='img'):
        raise NotImplementedError("Need to implement sample() !")

    def _process(self, did, pos, sz, mode):
        """
            did: which dataset
            pos: starting position
            sz: cropped volume size
            mode: ['img','seg'] for train, ['img'] for test
        """
        # get data
        sample = self.data_loader.getPatch(did, pos, sz, mode)
        # Apply data augmentation.
        if self.augmentor is not None:
            sample = self.augmentor(sample)
        return sample

# training sampler
class VolumeSamplerTrain(VolumeSampler):
    def __init__(self, data_loader=None, aug=None,
                 size_input=[1,1,1], size_output=[1,1,1], sample_stride=[1,1,1],
                 dataset_p=None):
        super(VolumeSamplerTrain, self).__init__(data_loader, aug, size_input, size_output, sample_stride)
        self.set_sampling_weights(dataset_p)

    def set_sampling_weights(self, p=None):
        """Set probability of each dataset being chosen at each sampling."""
        if p is None:
            p = self.sample_num
        else:
            assert len(p) == self.data_loader.num
        # Set sampling weights.
        self.p = p

    def random_dataset_id(self, drange=None):
        """Pick one dataset randomly."""
        if drange is None:
            p = self.p
            drange = range(len(p))
        else:
            p = np.array([self.p[d] for d in drange])
        idx = np.random.choice(drange, size=1, p=p)
        return idx[0]

    def sample(self, index, modes):
        # train: random sample
        # sample one augmentation configuration
        if self.augmentor is not None:
            self.augmentor.prepare()

        # random sample which position
        # get dataset
        did = self.random_dataset_id()
        # get sample size
        if self.augmentor.getSize() is None:
            # directly use the pre-computed sample_num
            for mode in modes:
                if mode == 'img':
                    sz[mode] = self.size_input
                else:
                    sz[mode] = self.size_output
            index = 
        pos = [0,0,0]

        return self._process(did, pos, sz, modes)

class VolumeSamplerTest(VolumeSampler):
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


