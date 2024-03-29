"""
Example scriptfor conducting supervised analysis  (classification and uncertainty estimation)
using AI-STEM.

"""

import os
# tensorflow info/warnings switched off
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np

from ai4stem.utils.utils_data import load_pretrained_model, load_example_image, load_class_dicts
from ai4stem.utils.utils_prediction import predict

import logging

if __name__ == '__main__':
    
    # set up log file
    logging.basicConfig(level=logging.INFO)
    
    # Specify example image
    # Here: Fe bcc [100] image loaded automatically
    input_image = load_example_image()
    image_name = 'Fe_bcc'

    # Specify path where to save the results:
    results_folder = '.'

    # Specify window size in untis of angstrom
    # (converted to pixels automatically, given pixel/angstrom relation)
    window_size_angstrom = 12.
    pixel_to_angstrom = 0.1245
    # Adapt window size
    window_size = window_size_angstrom * (1. / pixel_to_angstrom)
    window_size = int(round(window_size))
    logging.info('For image called {}, window {} [Angstrom] corresponds to {} pixels'.format(image_name, 
                                                                                             window_size_angstrom, 
                                                                                             window_size))

    # Strides [pixels] used in ai4stem algorithm
    # for fragmenting the input image 
    stride_size = [36, 36]

    # FFT HAADF descriptor settings
    sigma = None # optional parameter
    thresholding = True # very important for mitigating low-frequency contributions
    descriptor_params = {'sigma': sigma, 'thresholding': thresholding}

    # specify nn model, here load pretrained model
    model = load_pretrained_model()
    model_name = 'pretrained_model'

    # MC dropout samples
    n_iter = 100

    sliced_images, fft_descriptors, prediction, uncertainty = predict(input_image,
                                                                      model,
                                                                      n_iter,
                                                                      stride_size,
                                                                      window_size,
                                                                      descriptor_params)
    
    # Visualize results
    numerical_to_text_labels, text_to_numerical_labels = load_class_dicts()
    matplotlib.rcParams.update({'font.size': 10})
    assignments = prediction.argmax(axis=-1)

    fig, axs = plt.subplots(1, 3, figsize=(10, 10))

    im1 = axs[0].imshow(input_image, cmap='gray')
    axs[0].set_title('Input image')
    fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.05)

    im2 = axs[1].imshow(assignments, cmap='tab10')
    axs[1].set_title('Assigned label')
    # add nice legend for assignments
    all_colors = plt.cm.tab10.colors
    unique_assignments = np.unique(assignments.flatten())
    my_colors = [all_colors[idx] for idx in unique_assignments]
    patches = [mpatches.Patch(facecolor=c, edgecolor=c) for c in my_colors]
    axs[1].legend(patches, sorted(text_to_numerical_labels.keys()), handlelength=0.8, loc='lower right')

    im3 = axs[2].imshow(uncertainty, cmap='hot', vmin=0.0)
    axs[2].set_title('Bayesian uncertainty \n (mutual information)')
    fig.colorbar(im3, ax=axs[2],  orientation='vertical', fraction=0.05)
    
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')

    fig.tight_layout()
    
    plt.savefig(os.path.join(results_folder, 'results_summary.svg'))
    plt.close()

