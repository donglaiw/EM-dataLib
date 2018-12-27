import h5py
import numpy as np

from em_data.augmentation import *
from test_util import check_diff

def test_flip(data):
    spec={}
    for i in [0,1,2]:
        aug = Flip(spec, [i])
        aug.prepare(spec)
        data2 = aug(data.copy()) 
        if i==0:
            for k in data.keys():
                check_diff(data2[k], data[k])
        elif i==1:# swap xy
            for k in data.keys():
                check_diff(data2[k].transpose((0,1,3,2)), data[k])
        elif i==2:# flip x
            for k in data.keys():
                check_diff(data2[k], data[k][:,:,:,1:][:,:,:,::-1])

def test_grey(data):
    spec={}
    aug = Greyscale(spec,param=[0,0.5,0.1,0.3,2])
    aug.prepare(spec)
    data2 = aug(data.copy()) 
    check_diff(data2[k], data[k][:,:,:,1:][:,:,:,::-1])


def test_warp(img, seg):
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    # tensorboard --logdir test
    D0='/n/coxfs01/donglai/data/cremi/mala_v2/data_align_crop/'
    img = np.array(h5py.File(D0+'sample_B_im_crop.hdf')['main'])
    seg = np.array(h5py.File(D0+'sample_B_seg_crop.hdf')['stack'])
    data = {'img': img[30:33,:300,:300][None,:], 
            'seg': seg[30:33,:300,:300][None,:]}

    test_grey(data)
    #test_flip(data)

