from em_data.data_loader import *

class foo(object):
    pass

def test_io():
    D0='/n/coxfs01/donglai/data/cremi/mala_v2/data_align_crop/'
    opt = foo()
    opt.mode = 'train'
    opt.vol_input = '18,160,160'
    opt.vol_output = '18,160,160'
    opt.train_img = D0+'sample_A_im_crop.hdf@'+D0+'sample_B_im_crop.hdf'
    opt.train_seg = D0+'sample_A_seg_crop.hdf@'+D0+'sample_B_seg_crop.hdf'
    opt.train_img_name = 'main@main'
    opt.train_seg_name = 'stack@stack'
    opt.val_img = ''
    opt.val_seg = ''
    opt.val_img_name = ''
    opt.val_seg_name = ''
    ll = getDataLoader(opt)

if __name__ == '__main__':
    test_io()
