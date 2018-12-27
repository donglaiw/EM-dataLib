import h5py
import os
import time

from em_data.augmentation import *
from test_util import check_diff

def test_dataset():
    DD = '/n/coxfs01/donglai/data/SNEMI3D/'
    opt = foo()
    opt.add_argument('img', help='image file (h5 or tif) path.')
    opt.add_argument('lbl', help='label file (h5 or tif) path.')

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

if __name__ == '__main__':
    # tensorboard --logdir test
    D0='/n/coxfs01/donglai/data/cremi/mala_v2/data_align_crop/'
