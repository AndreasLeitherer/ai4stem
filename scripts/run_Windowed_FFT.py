import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from scipy.signal import get_window

import make_plot

#img = Image.open("test3.png")

f_list = open("list_HAADF.txt",'r')
while True:
    line = f_list.readline()

    if not line:
        break

    cols = line.split()


    filename =  cols[0]
    #fileHAADF = filename
    fileHAADF = filename
    #filename = filename.rstrip()

    print(fileHAADF)


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

    print(intfft.shape)
    print(thresh.shape)

    output = fshift / thresh
    output[np.where(output[:]<0)] = 0
    output[np.where(output[:]>thresh)] = 1


    

    #output = 10*np.log(np.abs(fshift))
#    print(output)
    print(output.shape)


    output2=np.zeros((64,64))
    for i in range(0,64):
        for j in range(0,64):
            output2[i,j] = output[int(float(output.shape[0])/float(2.0))-32+i,int(float(output.shape[1])/float(2.0))-32+j]

    name_Windowed = 'Windowed/Windowed_' + filename.split('/')[-1][9:]
    make_plot.savefig_2D(img,name_Windowed)

    name_FFT = 'FFT/FFTimage_' + filename.split('/')[-1][:-4]
    make_plot.savefig_2D(output2,name_FFT)
    np.save(name_FFT[:-4], output2)

