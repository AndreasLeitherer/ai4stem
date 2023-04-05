import numpy as np
import cv2
from ai4stem.descriptors.fft_haadf import FFT_HAADF
from ai4stem.utils.utils_nn import predict_with_uncertainty
from ai4stem.utils.utils_data import load_pretrained_model
from ai4stem.utils.utils_fft import calc_fft
def localwindow(image_in, stride_size, pixel_max=100):
    x_max = image_in.shape[0]
    y_max = image_in.shape[1]

    images = []
    spm_pos = []

    i = 0
    ni = 0
    while i < x_max-pixel_max:
        j = 0
        nj = 0
        ni = ni + 1

        while j < y_max-pixel_max:
            nj = nj + 1
            image = np.zeros((pixel_max,pixel_max))
            for x in range(0,pixel_max):
                for y in range(0,pixel_max):
                    image[x,y] = image_in[x+i,y+j]
            j += stride_size[1]
            images.append(image)
            spm_pos.append([i, j])
        i += stride_size[0]
    return images, np.asarray(spm_pos), ni, nj






def predict(image,
            model=None, n_iter=100,
            stride_size=[12, 12], window_size=100,
            descriptor_params={'sigma': None, 'thresholding': True}):
    """
    Given input funciton, apply neural-network classifier to classify the image
    in ai4stem-style (i.e., predict symmetry, lattice orientation, as well as quantify
    uncertainty that allows to spot interface regions).

    Parameters
    ----------
    image : numpy array
        2D Input image.
    model : tensorflow model object, optional
        Neural-network classifier. The default is None.
    n_iter : int, optional
        Number of Monte Carlo samples. The default is 100.
    stride_size : list, optional
        1D Stride list, in units of pixels. The default is [12, 12].
    window_size : int, optional
        Window size in units of pixels. The default is 100.
    descriptor_params : dict, optional
        Dictionary with options for FFT-HAADF descritpor. 
        The default is {'sigma': None, 'thresholding': False}.

    Returns
    -------
    sliced_images : ndarray
        2D array, size determined by input-image, stride, and window size.
    fft_descriptors : ndarray
        2D array, size determined by input-image, stride, and window size.
    prediction : ndarray
        2D array, size determined by input-image, stride, and window size.
    uncertainty : ndarray
        2D array, size determined by input-image, stride, and window size.

    """
    
    # Extract windows
    sliced_images, spm_pos, ni, nj = localwindow(image, 
                                                 stride_size=stride_size, 
                                                 pixel_max=window_size)
    
    # Calculate FFT
    fft_descriptors = []
    for sliced_image in sliced_images:
        # Under construction: employ descriptor object.
        # fft_obj = FFT_HAADF(**descriptor_params)
        # fft_desc = fft_obj.calculate(sliced_image)
        fft_desc = calc_fft(sliced_image, sigma=descriptor_params['sigma'], 
                            thresholding=descriptor_params['thresholding'])
        fft_descriptors.append(fft_desc)
    
    # Load model
    if model == None:
        model = load_pretrained_model()
    repeated_images = np.array([np.stack([_]) for _ in fft_descriptors])
    repeated_images = np.moveaxis(repeated_images, 1, -1)


    prediction, uncertainty = predict_with_uncertainty(repeated_images, 
                                                   model=model, 
                                                   model_type='classification', 
                                                   n_iter=n_iter)
    sliced_images = np.reshape(sliced_images, (ni, nj, window_size, window_size))
    fft_descriptors = np.reshape(fft_descriptors, (ni, nj, 
                                                   fft_descriptors[0].shape[0],
                                                   fft_descriptors[0].shape[1]))
    prediction = np.reshape(prediction, (ni, nj, prediction.shape[-1]))
    mutual_information = np.reshape(uncertainty['mutual_information'], (ni, nj))
    return sliced_images, fft_descriptors, prediction, mutual_information
                                 
