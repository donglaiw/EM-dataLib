import h5py
import numpy as np
import glob
import imageio

from em_segLib.transform import cropCentralN, crop
from options import optArr

def getDataLoader(opt):
    img_name={}
    img_name_dst={}
    seg_name={}
    seg_name_dst={}
    modes = ['train', 'val'] if opt.mode=='train' else ['test']
    vol_input = optArr(opt.vol_input)
    vol_output = optArr(opt.vol_output)

    for nn in modes:
        if len(getattr(opt, nn+'_img'))>0:
            img_name[nn] = getattr(opt, nn+'_img').split('@')
            img_name_dst[nn] = getattr(opt, nn+'_img_name').split('@')
            seg_name[nn] = getattr(opt, nn+'_seg').split('@')
            seg_name_dst[nn] = getattr(opt, nn+'_seg_name').split('@')
    return DataLoader(img_name, img_name_dst, seg_name, seg_name_dst, vol_input, vol_output)


class DataLoader(object):
    # in-memory dataLoader
    def __init__(self, img_name, img_name_dst, seg_name, seg_name_dst, vol_input, vol_output):
        # get input/output size
        self.vol_input = vol_input
        self.vol_output = vol_output

        # load image and segmentation
        self._img = []
        self._seg = []
        for nn in img_name.keys():
            self.addData(img_name[nn], img_name_dst[nn], 'img')
            self.addData(seg_name[nn], seg_name_dst[nn], 'seg')
        if len(self._seg)>0:
            if len(self._img)!=len(self._seg):
                raise ValueError('dataset mismatch: #img=%d, #seg=%d'%())
            # crop img and seg: based on the model io
            offset = (self.vol_input-self.vol_output) // 2
            self._img, self._seg = cropCentralN(self._img, self._seg, offset)
        
        # get image size
        self.img_shape = [np.array(x.shape) for x in self._img]
        self.img_size = [np.prod(x) for x in self.img_shape] # for sampling weight
        self.seg_shape = [np.array(x.shape) for x in self._seg]

    # construction
    def addData(self, data_name, data_dataset_name, modality='seg'):
        for i in range(len(data_name)):
            # data: DxHxW -> add 
            if data_name[i][-3:] == '.h5' or data_name[i][-3:] == 'hdf':
                file_name = data_dataset_name[i].split('/')
                # print data_name[i], file_name[0]
                file_data = h5py.File(data_name[i], 'r')[file_name[0]]
                for j in range(1,len(file_name)):
                    file_data = file_data[file_name[j]]
                data = np.array(file_data)
            elif data_name[i][-4:] == '.pkl':
                data = np.array(pickle.load(data_name[i], 'rb'))
            else: # folder of images
                imN = sorted(glob.glob(data_name[i]))
                im0 =  imageio.imread(imN[0])
                if modality == 'img': # gray scale
                    data = np.zeros((1, len(imN), im0.shape[1], im0.shape[0]),dtype=np.uint8)
                    for j in range(len(imN)):
                        data[0,j] = imageio.imread(imN[j], 'L')
                elif modality == 'seg': # gray scale
                    data = np.zeros((1, len(imN), im0.shape[1], im0.shape[0]),dtype=np.uint32)
                    for j in range(len(imN)):
                        tmp = imageio.imread(imN[j])
                        if tmp.ndim==3:
                            tmp = 255*255*tmp[0]
                        data[0,j] = imageio.imread(imN[j], 'RGB')

            # make sure of the input shape/value
            if data.ndim==3:
                data = data[None,:]
            if modality=='img':
                data = data.astype(np.float32)
                if data.max()>2: # normalize uint8 to 0-1
                    data = data/(2.**8)
            
            getattr(self,'_'+modality).append(data)

    # get data
    def getPatch(self, did, pos, sz, mode):
        # pos: top-left position
        if mode=='img':
            return {'img' : crop(self._img[did], sz, pos).copy()}
        elif mode=='img-seg':
            return {'img' : crop(self._img[did], sz[0], pos).copy(), 
                    'seg' : crop(self._seg[did], sz[1], pos).copy()}

