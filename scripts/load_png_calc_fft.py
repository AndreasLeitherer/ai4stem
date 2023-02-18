import numpy as np
from PIL import Image
import cv2
import os
from scipy.signal import get_window
import matplotlib
import matplotlib.pyplot as plt

def calc_fft(filename, padding=(40, 40), output_size=(128, 128)):
    
    #fileHAADF = filename
    fileHAADF = filename
    #filename = filename.rstrip()

    #print(fileHAADF)


    img = cv2.imread(fileHAADF,0)

    bw2d = np.outer(get_window('hanning',img.shape[0]), np.ones(img.shape[1]))
    bw2d_1 = np.transpose(np.outer(get_window('hanning',img.shape[1]), np.ones(img.shape[0])))
    w = np.sqrt(bw2d * bw2d_1)

    img = img * w

    f = np.fft.fft2(img)#, s=(img.shape[0]*2, img.shape[1]*2))
    max_f = np.max(np.abs(f))
    f = f/max_f

    fshift = np.fft.fftshift(np.abs(f)) # shift zero freq. comp to center of spectrum
    
    intfft = np.sort(fshift.ravel())[::-1]
    thresh = intfft[1]

    #print(intfft.shape)
    #print(thresh.shape)

    output = fshift / thresh
    output[np.where(output[:]<0)] = 0
    output[np.where(output[:]>thresh)] = 1


    

    #output = 10*np.log(np.abs(fshift))
#    print(output)
    #print(output.shape)


    output2=np.zeros((64,64))
    for i in range(0,64):
        for j in range(0,64):
            output2[i,j] = output[int(float(output.shape[0])/float(2.0))-32+i,int(float(output.shape[1])/float(2.0))-32+j]


    matplotlib.image.imsave('./fft_pngs/FFT_' + filename.split('/')[-1], output2)
    #im = Image.fromarray(output2)
    #im.save('./fft_pngs/FFT_' + filename.split('/')[-1])
    #plt.imshow(output2)
    #plt.savefig('./fft_pngs/FFT_' + filename.split('/')[-1])
    #plt.close()
    return output2
    """
    name_Windowed = 'Windowed/Windowed_' + filename.split('/')[-1][9:]
    make_plot.savefig_2D(img,name_Windowed)

    name_FFT = 'FFT/FFTimage_' + filename.split('/')[-1][:-4]
    make_plot.savefig_2D(output2,name_FFT)
    np.save(name_FFT[:-4], output2)
    """

if __name__ == '__main__':
    
    class_list = ["bcc100", "bcc110", "bcc111","fcc100",
                  "fcc110", "fcc111", "fcc211", "hcp0001", "hcp10m10",
                   "hcp11m20", "hcp2m1m10"]
    numerical_to_text_label = dict(zip(range(len(class_list)), class_list))
    text_to_numerical_label = dict(zip(class_list, range(len(class_list))))
    
    save_path = '../data'
    haadf_path = '/home/leitherer/Real_space_AI_STEM/AI4STEM/data/HAADF'
    haadf_images = [_ for _ in os.listdir(haadf_path) if _.endswith('.png')]
    fft_haadf_images = []
    fft_haadf_images_targets = []
    
    for haadf_image in haadf_images:
        fft_haadf_image = calc_fft(os.path.join(haadf_path, haadf_image))
        fft_haadf_image_3channels = np.stack((fft_haadf_image,
                                              fft_haadf_image,
                                              fft_haadf_image))
        fft_haadf_images.append(fft_haadf_image_3channels)
        
        target = ''.join(haadf_image.split('_')[2:4])
        fft_haadf_images_targets.append(target)
        
    np.save('../data/fft_haadf_training_x.npy', np.asarray(fft_haadf_images))
    
    numerical_labels = np.array([text_to_numerical_label[_] for _ in fft_haadf_images_targets])
    np.save('../data/fft_haadf_training_y.npy', numerical_labels)