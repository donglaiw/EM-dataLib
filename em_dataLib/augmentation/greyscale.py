import numpy as np
from em_dataLib.augmentor import DataAugment

class Greyscale(DataAugment):
    """
    Greyscale value augmentation.
    Randomly adjust contrast/brightness, and apply random gamma correction.
    """

    def __init__(self, param=[0,0.5,0.1,0.3,2], modality='img'):
        """Initialize parameters.
        Args:
            mode=param[0]: 0='2D', 1='3D', 0-1='mix'
            skip_ratio=param[1]: Probability of skipping augmentation.
            contrast=param[2]
            brightness=param[3]
            gamma=param[4]
        """
        self.set_mode(param[0])
        self.set_skip_ratio(param[1])
        self.set_contrast(param[2])
        self.set_brightness(param[3])
        self.set_gamma(param[4])
        self.key = modality

    def prepare(self):
        # sample to skip or not
        self.skip = np.random.rand() < self.skip_ratio

    def __call__(self, sample):
        if not self.skip:
            do_3d = np.random.rand() <= self.mode
            if do_3d: # same aug for all slices
                self.augment3D(sample)
            else: # different aug for each slice
                self.augment2D(sample)
        return sample

    def augment3D(self, sample):
        sample[self.key] *= 1 + (np.random.rand() - 0.5)*self.contrast
        sample[self.key] += (np.random.rand() - 0.5)*self.brightness
        sample[self.key] = np.clip(sample[self.key], 0, 1)
        if self.gamma > 0: # if 0, no randomness
            sample[self.key] **= 2.0**(np.random.rand()*self.gamma - 1)
        return sample
 
    def augment2D(self, sample, randNum):
        numZ = sample[self.key].shape[1]
        randNum = np.random.rand((numZ,3))
        for z in xrange(numZ):
            img = sample[self.key][:,z,:,:]
            img *= 1 + (randNum[z,0] - 0.5)*self.contrast
            img += (randNum[z,1] - 0.5)*self.brightness
            img = np.clip(img, 0, 1)
            if self.gamma > 0: # if 0, no randomness
                img **= 2.0**(randNum[z,2]*self.gamma - 1)
            sample[self.key][:,z,:,:] = img
        return sample

    ####################################################################
    ## Setters.
    ####################################################################
    def set_mode(self, mode):
        """Set 2D/3D/mix greyscale value augmentation mode."""
        assert mode>=0.0 and ratio <= 1.0
        self.mode = mode

    def set_skip_ratio(self, ratio):
        """Set the probability of skipping augmentation."""
        assert ratio >= 0.0 and ratio <= 1.0
        self.skip_ratio = ratio

    def set_contrast(self, contrast):
        """Set contrast"""
        self.contrast = contrast

    def set_brightness(self, brightness):
        """Set brightness"""
        self.brightness = brightness

    def set_gamma(self, gamma):
        """Set gamma"""
        self.gamma = gamma
