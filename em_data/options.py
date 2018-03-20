import argparse
import numpy as np

def optArr(in_str, delim=','):
    # input string to int array
    return np.array([int(x) for x in in_str.split(delim)])

def optDataLoad(parser, mode='train'):
    if mode == 'train':
        parser.add_argument('-dti', '--train-img', default='',
                            help='input train image')
        parser.add_argument('-dts', '--train-seg', default='',
                            help='input train segmentation')
        parser.add_argument('-dvi', '--val-img', default='',
                            help='input validataion image')
        parser.add_argument('-dvs', '--val-seg', default='',
                            help='input validation segmentation')
        # if h5
        parser.add_argument('-dtid','--train-img-name',  default='main',
                            help='dataset name in train image')
        parser.add_argument('-dtsd','--train-seg-name',  default='main',
                            help='dataset name in train segmentation')
        parser.add_argument('-dvid','--val-img-name',  default='main',
                            help='dataset name in validation image')
        parser.add_argument('-dvsd','--val-seg-name',  default='main',
                            help='dataset name in validation segmentation')

    elif mode=='test':
        parser.add_argument('-dei', '--test-img', default='',
                            help='input test image')
        parser.add_argument('-des', '--test-seg', default='',
                            help='input test segmentation')
        # if h5
        parser.add_argument('-deid','--test-img-name',  default='main',
                            help='dataset name in test image')
        parser.add_argument('-desd','--test-seg-name',  default='main',
                            help='dataset name in test segmentation')

    parser.add_argument('-bi','--vol-input', type=str,  default='31,204,204',
                        help='input volume size')
    parser.add_argument('-bo','--vol-output', type=str,  default='3,116,116',
                        help='output volume size')
    parser.add_argument('-o','--output', default='result/train/',
                        help='output path')

def optDataAug(parser):
    # reduce the number of input arguments by stacking into one string
    parser.add_argument('-ao','--aug-opt', type=str,  default='1@-1@0@5',
                        help='data aug type')
    parser.add_argument('-apw','--aug-param-warp', type=str,  default='15@3@1.1@0.1',
                        help='data warp aug parameter')
    parser.add_argument('-apc','--aug-param-color', type=str,  default='0.95,1.05@-0.15,0.15@0.5,2@0,1',
                        help='data color aug parameter')

