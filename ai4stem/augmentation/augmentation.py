from copy import deepcopy
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise
import cv2
import numpy as np
from scipy import ndimage

class Augmentation():
    
    def __init__(self, noise_type=None, 
                 blurrings=[2, 4], var=[0.005, 0.01],
                 sampling=0.12, fov=12, 
                 rot_range=np.linspace(0, 90, 10, dtype=int)):
        """

        Parameters
        ----------
        noise_type : string, optional
            Type of noise applied to image, either 'blurring', 'gaussian' or 'poisson'. The default is None.
        blurrings : 1D float list/array, optional
            Width of the gaussian filter used for blurring.
            The default is None.
        var : 1D float list/array, optional
            Standard deviations used for applying Gaussian noise to the input image.
            Only used if noise_type='gaussian'
            The default is None.

        Returns
        -------
        None.

        """
        
        self.noise_type = noise_type
        self.blurrings = blurrings
        self.var = var
        self.sampling = sampling # units Angstrom
        self.fov = fov # units Angstrom
        self.rot_range = rot_range
    
    def rotate(self, img):
        pixel_size = [self.sampling, self.sampling]
        
        # Original image size
        Nx, Ny = img.shape
        
        # If image size is odd, make even in the corresponding direction by removing a one pixel boundary
        if (Nx % 2) == 0:
            Nx = Nx
        else:
            img = img[:Nx-1,:]
            
        if (Ny % 2) == 0:
            Ny = Ny
        else:
            img = img[:,:Ny-1]
            
        Nx, Ny = img.shape
        
        
        size_crop = int(self.fov / (np.round(pixel_size[0]*100)/100))
        img_rotated = np.zeros((size_crop, size_crop,
                                self.rot_range.size), dtype=float)
        Nx_rot, Ny_rot, Nz_rot = np.shape(img_rotated)

        for a0 in range(0, self.rot_range.size, 1):

            #Rotate images
            img_rotated_temp = ndimage.rotate(img, self.rot_range[a0], reshape=False)


            center_x = int(np.round(img_rotated_temp.shape[0] / 2.))
            center_y = int(np.round(img_rotated_temp.shape[1] / 2.))
            half_cropsize = int(size_crop / 2)
            if size_crop % 2 == 0:
                img_rotated[:, :, a0] = img_rotated_temp[center_x - half_cropsize: center_x + half_cropsize,
                                                         center_y - half_cropsize: center_y + half_cropsize]
            else:
                img_rotated[:, :, a0] = img_rotated_temp[center_x - half_cropsize: center_x + half_cropsize + 1,
                                         center_y - half_cropsize: center_y + half_cropsize + 1]
        return img_rotated
        
    def apply_noise(self, img):
        
        distorted_images = []
        if self.noise_type == 'gaussian':
            for v in self.var:
                distorted_image = random_noise(img, mode='gaussian', var=v)
                distorted_images.append(distorted_image)
        elif self.noise_type == 'poisson':
            distorted_image = random_noise(img, mode='poisson')
            distorted_images.append(distorted_image)
        elif self.noise_type == 'blurring':
            for width in self.blurrings:
                distorted_image = gaussian_filter(img, sigma=width)
                distorted_images.append(distorted_image)
        else:
            raise NotImplementedError("Specified noise type not supported.")
            
        return np.asarray(distorted_images)
    
    def calculate(self, img):
        
        tmp_img = deepcopy(img)
        
        # Normalization
        tmp_img = cv2.normalize(tmp_img, None,
                                alpha=0, beta=1,
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # apply noise
        distorted_images = self.apply_noise(tmp_img)
        
        return distorted_images
        