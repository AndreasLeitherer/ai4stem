"""
Example script for conducting unsupervised analysis based 
on neural-network representations of a pre-trained model.
"""

import os
# tensorflow info/warnings switched off
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ai4stem.utils.utils_data import load_example_image
from ai4stem.utils.utils_unsupervised import unsupervised_analysis

import matplotlib
matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    
    image = load_example_image()
    image_name = 'Fe_bcc_100'
    pixel_to_angstrom = 0.1245
    window_size = int(12. / pixel_to_angstrom)
    stride_size = [36, 36]
    method = 'umap'
    
    embedding = unsupervised_analysis(image, window_size, stride_size, method=method)
    
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5)
    plt.axis('off')
    plt.savefig('scatterplot_{}_for_image_{}.png'.format(method, image_name))
    plt.close()
