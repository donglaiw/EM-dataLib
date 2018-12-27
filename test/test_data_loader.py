from em_data.data_loader import *

from test_util import foo

def test_snemi():
    D0='/n/coxfs01/donglai/data/SNEMI3D/'
    opt = foo()
    opt.mode = 'train'
    opt.size_input = '18,160,160'
    opt.size_output = '18,160,160'
    opt.train_img = D0+'train-input_df_150.h5'
    opt.train_seg = D0+'train-labels.h5'
    opt.train_img_name = 'main'
    opt.train_seg_name = 'main'
    opt.val_img = ''
    opt.val_seg = ''
    opt.val_img_name = ''
    opt.val_seg_name = ''
    ll = getDataLoader(opt)
    return ll

if __name__ == '__main__':
    test_snemi()
