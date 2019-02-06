import argparse
import numpy as np

def optIO(parser, mode='train'):
    parser.add_argument('-ds', '--data-shape', type=str, default='18,160,160',
                        help="Model's input size (shape) formatted 'z, y, x' with no channel number")
    parser.add_argument('-do', '--output-dir', default='result/train/',
                        help='Output directory used to save the prediction results as .h5 file(s). The directory is ' +
                             'automatically created if already not created')
    if mode == 'train':
        parser.add_argument('-di', '--input-vol', default='train-input.h5@train2-input.h5',
                            help='Path to the training volume .h5 file(s), separated by @')
        parser.add_argument('-dl', '--label-vol', default='train-labels.h5@train2-labels.h5',
                            help='Path to the training segmentation .h5 file(s), separated by @')
        # one set of volumes: divide into train-val by ratio
        parser.add_argument('-dtr', '--train-ratio', type=str, default='0.7',
                            help='The ratio of the data used for training. The rest will be used for validation')
        parser.add_argument('-dim', '--invalid-mask', type=str, default='0',
                            help='Mask out unlabeled regions')
        parser.add_argument('-dc', '--data-chunk', type=str, default='',
                            help='slices of the input data')
        # two sets of volumes
        parser.add_argument('-div', '--input-vol-val', default='',
                            help='Path to the validatation volume .h5 file(s)')
        parser.add_argument('-dlv', '--label-vol-val', default='',
                            help='Path to the validation segmentation .h5 file(s)')

    elif mode=='test':
        parser.add_argument('-dt', '--test-volume', default='test-input.h5',
                            help='Path to the test volume .h5 file(s)')
        parser.add_argument('-dxp', '--x-pad', type=int, default=48,
                            help="Number of voxels for mirror padding the x axis of the test volume. Required for " +
                                 "eliminating the gray grid on the edges of the prediction resulting from the Gaussian" +
                                 "blending. The volume will be padded by this amount on both ends")
        parser.add_argument('-dyp', '--y-pad', type=int, default=48,
                            help="Number of voxels for mirror padding the y axis of the test volume. Required for " +
                                 "eliminating the gray grid on the edges of the prediction resulting from the Gaussian" +
                                 "blending. The volume will be padded by this amount on both ends")
        parser.add_argument('-dzp', '--z-pad', type=int, default=8,
                            help="Number of voxels for mirror padding the z axis of the test volume. Required for " +
                                 "eliminating the gray grid on the edges of the prediction resulting from the Gaussian" +
                                 "blending. The volume will be padded by this amount on both ends")


def optDataAug(parser):
    # reduce the number of input arguments by stacking into one string
    parser.add_argument('-dao','--aug-opt', type=str,  default='1@-1@0@5',
                        help='data aug type')
    parser.add_argument('-daf','--aug-param-flip', type=str,  default='0.95,1.05@-0.15,0.15@0.5,2@0,1',
                        help='data color aug parameter')

    parser.add_argument('-dae','--aug-param-elastic', type=str,  default='15@3@1.1@0.1',
                        help='data augmentation elastic deformation parameter')
    parser.add_argument('-dac','--aug-param-color', type=str,  default='0.95,1.05@-0.15,0.15@0.5,2@0,1',
                        help='data color aug parameter')

