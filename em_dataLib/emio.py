import json
import numpy as np
import h5py
import tifffile
from scipy.misc import imread, imsave
from scipy.ndimage.interpolation import zoom

def imread(fname):
    # todo: add folder of images
    if '.hdf' in fname or '.h5' in fname:
        with h5py.File(fname, 'r') as f:
            kk = [k for k in f.keys()][0]
            vol = np.asarray(f[kk])
    elif '.tif' in fname:
        vol = tifffile.imread(fname)
    elif fname[fname.rfind('.')+1:] in ['png','jpg']:
        vol = imread(fname)
    else:
        print("unknown input type: we only read folder/, .tif and .h5/.hdf5")
        raise
    return vol

def imsave( vol, fname ):
    # todo: add folder of images
    if '.hdf5' in fname or '.h5' in fname:
        with h5py.File( fname ) as f:
            f.create_dataset('/main', data=vol, compression="gzip")
    elif '.tif' in fname:
        tifffile.imsave(fname, vol)
    elif fname[fname.rfind('.')+1:] in ['png','jpg']:
        imsave(fname, vol)
    else:
        print("unknown outinput type: we only read folder/, .tif and .h5/.hdf5")
        raise

# 1. I/O h5 file
def writeh5(fname, data, dname='main'):
    fid=h5py.File(fname,'w')
    if isinstance(dname, (list,)):
        for i,dd in enumerate(dname):
            ds = fid.create_dataset(dd, data[i].shape, compression="gzip", dtype=data[i].dtype)
            ds[:] = data[i]
    else:
        ds = fid.create_dataset(dname, data.shape, compression="gzip", dtype=data.dtype)
        ds[:] = data
    fid.close()

def readh5(fname, dname='main'):
    """
    read dataset in hdf5 file
    """
    return np.array(h5py.File(fname, 'r')[dname])

def resizeh5(path_in, path_out, dataset, ratio=(0.5,0.5), interp=2, offset=[0,0,0]):
    # for half-res
    im = h5py.File( path_in, 'r')[ dataset ][:]
    shape = im.shape
    if len(shape)==3:
        im_out = np.zeros((shape[0]-2*offset[0], int(np.ceil(shape[1]*ratio[0])), int(np.ceil(shape[2]*ratio[1]))), dtype=im.dtype)
        for i in xrange(shape[0]-2*offset[0]):
            im_out[i,...] = zoom( im[i+offset[0],...], zoom=ratio,  order=interp)
        if offset[1]!=0:
            im_out=im_out[:,offset[1]:-offset[1],offset[2]:-offset[2]]
    elif len(shape)==4:
        im_out = np.zeros((shape[0]-2*offset[0], shape[1], int(shape[2]*ratio[0]), int(shape[3]*ratio[1])), dtype=im.dtype)
        for i in xrange(shape[0]-2*offset[0]):
            for j in xrange(shape[1]):
                im_out[i,j,...] = zoom( im[i+offset[0],j,...], ratio, order=interp)
        if offset[1]!=0:
            im_out=im_out[:,offset[1]:-offset[1],offset[2]:-offset[2],offset[3]:-offset[3]]
    if path_out is None:
        return im_out
    writeh5(path_out, dataset, im_out)

# 2. I/O txt file
def readtxt(filename):
    a= open(filename)
    content = a.readlines()
    a.close()
    return content

def writetxt(filename, content):
    a= open(filename,'w')
    if isinstance(content, (list,)):
        for ll in content:
            a.write(ll)
    else:
        a.write(content)
    a.close()



def readvol(fn, x0, x1, y0, y1, z0, z1, tile_sz, dt=np.uint8,st=1, tile_ratio=1, tile_resize_mode='bilinear'):
    # no padding at the boundary
    # st: starting index 0 or 1
    result = np.zeros((z1-z0, y1-y0, x1-x0), dt)
    c0 = x0 // tile_sz # floor
    c1 = (x1 + tile_sz-1) // tile_sz # ceil
    r0 = y0 // tile_sz
    r1 = (y1 + tile_sz-1) // tile_sz
    for z in range(z0, z1):
        pattern = bfly_db["sections"][z]
        for row in range(r0, r1):
            for column in range(c0, c1):
                path = pattern.format(row=row+st, column=column+st)
                if not os.path.exists(path): 
                    #return None
                    patch = 128*np.ones((tile_sz,tile_sz),dtype=np.uint8)
                else:
                    if path[-3:]=='tif':
                        import tifffile
                        patch = tifffile.imread(path)
                    else:
                        import scipy
                        patch = scipy.misc.imread(path, 'L')
                if tile_ratio != 1:
                    patch = scipy.misc.imresize(patch, tile_ratio, tile_resize_mode)

                xp0 = column * tile_sz
                xp1 = (column+1) * tile_sz
                yp0 = row * tile_sz
                yp1 = (row + 1) * tile_sz
                if patch is not None:
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0]
    return result

def writevol(sz, numT, imN, zPad=0, im_id=None, outName=None):                                     
    # one tile for each section
    dim={'depth':sz[0]+sum(zPad), 'height':sz[1], 'width':sz[2],
         'dtype':'uint8', 'n_columns':numT[1], 'n_rows':numT[0]}
    # 1-index
    if im_id is None:
        im_id = range(zPad[0]+1,1,-1)+range(1,sz[0]+1)+range(sz[0]-1,sz[0]-zPad[1]-1,-1)
    sec=[imN(x) for x in im_id]
    out={'sections':sec, 'dimensions':dim}
    if outName is None:
        return out
    else:
        with open(outName,'w') as fid:
            json.dump(out, fid)
