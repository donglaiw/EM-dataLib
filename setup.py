from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_inc
from Cython.Distutils import build_ext
from setuptools import find_packages 
import numpy as np

def getExt_model():
    return []


def getExt_data():
    return [Extension('em_dataLib.augmentation.warping',
                 sources=['em_dataLib/augmentation/warping.pyx',
                         'em_dataLib/augmentation/cpp/warping/c-warping.c'],
                 extra_compile_args=['-std=c99', '-fno-strict-aliasing', '-O3', '-Wall', '-Wextra'])]

def setup_cython():

    ext_modules = getExt_data()

    setup(name='em_dataLib',
       version='1.0',
       cmdclass = {'build_ext': build_ext}, 
       install_requires=['cython','scipy','numpy','imageio'],
       include_dirs=[np.get_include(), get_python_inc()], 
       packages=find_packages(),
       ext_modules = ext_modules)

if __name__=='__main__':
	setup_cython()
