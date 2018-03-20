import h5py
import numpy as np

import tensorflow as tf

from em_data.augmentation import *

def test_flip(data, writer):
    aug = Flip()
    data2 = aug(data.copy(),rule=[0,1,0,0]) 
    sz = data['img'][0].shape
    out = tf.convert_to_tensor(np.vstack([data['img'][0].reshape(sz[0],sz[1],sz[2],1),
                    data['seg'][0].reshape(sz[0],sz[1],sz[2],1),
                    data2['img'][0].reshape(sz[0],sz[1],sz[2],1),
                    data2['seg'][0].reshape(sz[0],sz[1],sz[2],1)
                   ]),dtype=tf.float32)
    for i in range(4):
        summary_op = tf.summary.image("image_"+str(i), out[i*3:(i+1)*3-1])
        summary = sess.run(summary_op)
        writer.add_summary(summary,1)
    writer.flush()

def test_warp(img, seg):
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    # tensorboard --logdir test
    D0='/n/coxfs01/donglai/data/cremi/mala_v2/data_align_crop/'
    img = np.array(h5py.File(D0+'sample_B_im_crop.hdf')['main'])
    seg = np.array(h5py.File(D0+'sample_B_seg_crop.hdf')['stack'])
    data = {'img': img[30:33,:300,:300][None,:], 
            'seg': seg[30:33,:300,:300][None,:]}

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess: 
        sess.run(init_op)
        writer = tf.summary.FileWriter('./', sess.graph)
        test_flip(data, writer)
        #test_warp(img, seg)
        writer.close()
        sess.close()
