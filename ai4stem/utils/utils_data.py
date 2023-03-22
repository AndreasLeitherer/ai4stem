import pkgutil
import os, sys
import numpy as np
from tensorflow.keras.models import load_model
import json

def get_data_filename(resource, package='ai4stem'):
    """Rewrite of pkgutil.get_data() that return the file path.
    Taken from: https://stackoverflow.com/questions/5003755/how-to-use-pkgutils-get-data-with-csv-reader-in-python
    """
    loader = pkgutil.get_loader(package)
    if loader is None or not hasattr(loader, 'get_data'):
        return None
    mod = sys.modules.get(package) or loader.load_module(package)
    if mod is None or not hasattr(mod, '__file__'):
        return None

    # Modify the resource name to be compatible with the loader.get_data
    # signature - an os.path format "filename" starting with the dirname of
    # the package's __file__
    parts = resource.split('/')
    parts.insert(0, os.path.dirname(mod.__file__))
    resource_name = os.path.normpath(os.path.join(*parts))

    return resource_name


def load_class_dicts():
    path_num_to_text = get_data_filename('data/class_definitions/numerical_to_text_labels.json')
    path_text_to_num = get_data_filename('data/class_definitions/text_to_numerical_labels.json')

    with open(path_num_to_text, 'r') as f:
        numerical_to_text_labels = json.load(f)
    with open(path_text_to_num, 'r') as f:
        text_to_numerical_labels = json.load(f)
    return numerical_to_text_labels, text_to_numerical_labels

def load_example_image():
    path_to_image = get_data_filename('data/experimental_images/Fe_bcc_100.npy')
    example_image = np.load(path_to_image)
    return example_image

def load_expimages():
    
    # Load data
    path_to_exp_images = get_data_filename('data/haadf_experimental_lattice_parameters.npy', package='ai4stem')
    haadf_images = np.load(path_to_exp_images)
    
    path_to_labels = get_data_filename('data/haadf_experimental_lattice_parameters_labels.npy', package='ai4stem')
    haadf_labels = np.load(path_to_labels, allow_pickle=True)
    
    return haadf_images, haadf_labels

def load_synthetic_image():
    path_to_synthetic_image = get_data_filename('data/synthetic_images/synthetic_polycrystal_w_amorphous.npy')
    synthetic_image = np.load(path_to_synthetic_image)
    return synthetic_image


def load_pretrained_model():
    path_to_pretrained_model = get_data_filename('data/pretrained_models/optimal_model.h5', package='ai4stem')
    model = load_model(path_to_pretrained_model)
    return model
