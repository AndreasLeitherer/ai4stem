#!/bin/python
import sys
import os.path

import numpy as np
import math

from matplotlib import pyplot as plt

def savefig(new_image, filename):
        height, width, channel = new_image.shape
        figsize = (1, height/width) if height>=width else (width/height, 1)

        plt.figure(figsize=figsize)
        plt.imshow(new_image,cmap='gray')
        #plt.colorbar()
        #plt.draw()
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        #plt.savefig('HAADFimage.png')#,bbox_inches='tight',pad_inches=0)#, dpi=17.5)
        #plt.savefig("output.png",bbox_inches='tight',pad_inches=0)#, dpi=17.5)
        plt.savefig(filename,bbox_inches='tight',pad_inches=0, dpi=128)
        plt.close()


def savefig_2D(new_image, filename):
        height, width = new_image.shape
        figsize = (1, height/width) if height>=width else (width/height, 1)

        plt.figure(figsize=figsize)
        plt.imshow(new_image,cmap='gray')
        #plt.colorbar()
        #plt.draw()
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        #plt.savefig('HAADFimage.png')#,bbox_inches='tight',pad_inches=0)#, dpi=17.5)
        #plt.savefig("output.png",bbox_inches='tight',pad_inches=0)#, dpi=17.5)
        plt.savefig(filename,bbox_inches='tight',pad_inches=0, dpi=64)
        plt.close()


