from em_data.sampler import *
from em_data.augmentation import *
from em_data.data_loader import *

from test_util import foo
from test_data_loader import test_snemi


def test_sampler():
    data = test_snemi()
    aug = None 

    opt = foo()
    opt.size_input = data.size_input
    opt.size_output = data.size_output
    opt.sample_stride = '1,1,1'
    sampler = getVolumeSampler('train',data, aug, opt)


if __name__ == '__main__':
    test_sampler()
