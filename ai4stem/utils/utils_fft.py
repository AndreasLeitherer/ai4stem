import cv2
import numpy as np
from scipy.signal import get_window

def calc_fft(img, padding=(0, 0), power=2,
             sigma=None, r_cut=None,
             thresholding=True, apply_window=True, output_size=None,
             output_shape=(64, 64)):
    """
    Given HAADF image, calculate HAADF-FFT descriptor.
    
    Parameters
    ----------
    img: ndarray
        HAADF input image
    padding: tuple
        zero padding employed to bring image size to power of 2
    power: int
        Number by which FFT amplitude is exponentiated
        in order to supress small fluctuations and
        emphasize peaks
    sigma: int
        Width of gaussian window employed to cut out central
        part of the FFT. In the standard setting (sigma=None),
        no cutting employed.
    r_cut: int
        Size of rectangular window
        that is used to cut the center of the FFT.
        In the standard setting (sigma=None),
        no cutting employed.
    thresholding: bool
        [incompletely implemented] If True, apply thresholding
        procedure to mitigate influence of central peak
    output_size: tuple
        Output size of fft, if None, fft size will be given
        by img.shape[0] and img.shape[1], if output size
        larger than image size, crop image, if smaller, apply 
        zero padding
        
    Returns
    ----------
    fft_haadf_desc: ndarray
        2D array containing FFT-HAADF descriptor
        
    .. codeauthors:: Andreas Leitherer <andreas.leitherer@gmail.com>,
                     Byungchul Yeo <byeo@pknu.ac.kr>
    
    """

    # First step: normalize image
    img = cv2.normalize(img, None,
                       alpha=0, beta=1,
                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    if apply_window:
        # windowing
        bw2d = np.outer(get_window('hann',img.shape[0]), 
                        np.ones(img.shape[1]))
        bw2d_1 = np.transpose(np.outer(get_window('hann',img.shape[1]), 
                                       np.ones(img.shape[0])))
        w = np.sqrt(bw2d * bw2d_1)
        img_windowed = img * w
    else:
        img_windowed = img
    
    # Calculate FFT
    f = np.fft.fft2(img_windowed, s=output_size)
    
    # Calculate power spectrum (or higher order exponential)
    fshift = np.fft.fftshift(np.power(np.abs(f), power))
    
    # Normalization
    fshift = cv2.normalize(fshift, None,
                           alpha=0, beta=1,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    
    # Remove central part of image, several options:
    # Spherical cut:
    if not r_cut == None:

        xc = (fshift.shape[0] - 1.0) / 2.0
        yc = (fshift.shape[1] - 1.0) / 2.0
        # spherical mask
        a, b = xc, yc
        x, y = np.ogrid[-a:fshift.shape[0] - a, -b:fshift.shape[1] - b]

        mask_out = x * x + y * y <= r_cut * r_cut

        for i in range(fshift.shape[0]):
            for j in range(fshift.shape[1]):
                if mask_out[i, j]:
                    fshift[i, j] = 0.0
   
    # cut using gaussian window: 
    if not sigma == None:
        bw2d = np.outer(get_window(('gaussian', sigma), fshift.shape[0]), 
                    np.ones(fshift.shape[1]))
        bw2d_1 = np.transpose(np.outer(get_window(('gaussian', sigma), fshift.shape[0]), 
                                       np.ones(fshift.shape[0])))
        w = np.sqrt(bw2d * bw2d_1)
        fshift = fshift * (1-w)

    if thresholding:
        fshift = cv2.normalize(fshift, None, 
                               alpha=0, beta=1, 
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        fshift = fshift/.1
        fshift[fshift>1] = 1
        fshift = cv2.normalize(fshift, None, 
                               alpha=0, beta=1, 
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
    
    # Cut out 64x64 window around center of FFT
    output = fshift
    #output2 = np.zeros((64,64))
    #for i in range(0,64):
    #    for j in range(0,64):
    #        output2[i,j] = output[int(float(output.shape[0])/float(2.0))-32+i,int(float(output.shape[1])/float(2.0))-32+j]

    output2 = np.zeros(output_shape)
    for i in range(0, output_shape[0]):
        for j in range(0, output_shape[1]):
            output2[i,j] = output[int(float(output.shape[0])/2.) - int(output_shape[0]/2.) + i,
                                  int(float(output.shape[1])/2.0) - int(output_shape[1]/2.) + j]

    
    fft_haadf_desc = cv2.normalize(output2, None, 
                                   alpha=0, beta=1, 
                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    return fft_haadf_desc
