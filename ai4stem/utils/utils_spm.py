import numpy as np
import cv2
from ai4stem.descriptors.fft_haadf import FFT_HAADF
from ai4stem.utils.utils_nn import predict_with_uncertainty
from ai4stem.utils.utils_data import load_pretrained_model
def localwindow(image_in, stride_size, pixel_max=100,
                normalize_before_fft=False, normalize_after_window=False):
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
            if normalize_before_fft:
                image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
            """
            filename = "local_images_" + str(ni) + "_" + str(nj)
            plt.figure()
            plt.imshow(image,cmap='gray')
            #plt.colorbar()
            #plt.draw()
            plt.axis('off')
            plt.savefig(filename + '.png',bbox_inches='tight',pad_inches=0)
            plt.close()
            np.save(filename + '.npy', image)

            makeWindowingFFT.windowFFT(image,filename +'.png',normalize_after_window)
            """
            j += stride_size[1]
            images.append(image)
            spm_pos.append([i, j])
        i += stride_size[0]
    return images, np.asarray(spm_pos), ni, nj



def stem_spm(image,
             model=None, n_iter=100,
             stride_size=[12, 12], window_size=100,
             descriptor_params={'sigma': None, 'thresholding': False}):
    
    # Extract windows
    sliced_images, spm_pos, ni, nj = localwindow(image, 
                                                 stride_size=stride_size, 
                                                 pixel_max=window_size)
    
    # Calculate FFT
    fft_descriptors = []
    for sliced_image in sliced_images:
        fft_obj = FFT_HAADF(**descriptor_params)
        fft_desc = fft_obj.calculate(sliced_image)
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
    return sliced_images, fft_descriptors, prediction, uncertainty
                                 