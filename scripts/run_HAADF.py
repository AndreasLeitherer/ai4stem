import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import math

from skimage.feature import peak_local_max

import matplotlib.cm as cm
from matplotlib.pyplot import figure, show, axes, sci
from matplotlib import cm, colors
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap

import separateBackGMM, hierCluster, learnDescriptor

import make_plot

#if __name__ == "__main__": 

f_list = open("list.txt",'r')

while True:
    line = f_list.readline()
    
    if not line:
        break
   
    cols = line.split()
 

    filename = cols[0] #"HAADF.png" #Target STEM-HAADF image
    print(filename)


    image = Image.open(filename)
    size = (64,64)
    image.thumbnail(size)
    image = np.array(image,dtype=float)
    image = np.mean(image,axis=2)
    
    image = separateBackGMM.seperate_background(image,64)

    atom_columns = peak_local_max(image, min_distance=1)
    atom_columns = hierCluster.hier_clustering(atom_columns)
	
    g_dist_array = np.zeros((64,64))

    for i in range(len(atom_columns)):
        x, y = np.meshgrid(np.linspace(-1,1,192), np.linspace(-1,1,192))
        d = np.sqrt(x*x+y*y)
        sigma, mu = 0.02, 0.0 #0.0156, 0.0
        g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

        xx = int(atom_columns[i][0])
        yy = int(atom_columns[i][1])	

        pure = np.zeros((64,64))

        for x in range(64):
            for y in range(64): 
                pure[x][y] = g[32+x+xx][32+y+yy]

        g_dist_array = g_dist_array*float(i+1)
        g_dist_array = np.add(g_dist_array,abs(pure))/float(i+2)


    image = g_dist_array
 
    print(image.shape)
    file_name_write = "HAADF/" + filename[9:]
    height, width = image.shape
    figsize = (1, height/width) if height>=width else (width/height, 1)
    
    plt.figure(figsize=figsize)
    plt.imshow(image,cmap='gray')
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
    plt.savefig(file_name_write,bbox_inches='tight',pad_inches=0, dpi=128)


f_list.close()
