import pkgutil
import os, sys
import numpy as np
from tensorflow.keras.models import load_model
import json

def get_data_filename(resource, package='ai4stem'):
    """
    "Rewrite of pkgutil.get_data() that return the file path.
    Taken from: https://stackoverflow.com/questions/5003755/how-to-use-pkgutils-get-data-with-csv-reader-in-python

    Parameters
    ----------
    resource : string
        Path to data to be load, relative to the location
        of the python package specified in the package argument.
    package : string, optional
        Python package to which the data to be loaded refers to.
        The default is 'ai4stem'.

    Returns
    -------
    resource_name : string
        Absolute path to data.
    
    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

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
    """
    Load json dicts that specify the relation between output neurons
    of pretrained neural-network model and the corresponding class labels.
    
    Returns
    -------
    numerical_to_text_labels : dict
        Dictionary containing integer labels as keys
        and class labels (text) as values.
    text_to_numerical_labels: dict
        Dictionary containing class labels (text) as keys
        and integer labels as values.
    
    .. codeauthor:: Andreas Leitherer <andreas.leitherer@gmail.com>
    
    """
    path_num_to_text = get_data_filename('data/class_definitions/numerical_to_text_labels.json')
    path_text_to_num = get_data_filename('data/class_definitions/text_to_numerical_labels.json')

    with open(path_num_to_text, 'r') as f:
        numerical_to_text_labels = json.load(f)
    with open(path_text_to_num, 'r') as f:
        text_to_numerical_labels = json.load(f)
    return numerical_to_text_labels, text_to_numerical_labels

def load_example_image():
    """
    Load Fe bcc [100] example image.

    Returns
    -------
    example_image : ndarray
        Fe bcc [100] example image.
    
    .. codeauthor:: Andreas Leitherer <andreas.leitherer@gmail.com>

    """
    path_to_image = get_data_filename('data/experimental_images/Fe_bcc_100.npy')
    example_image = np.load(path_to_image)
    return example_image


def load_pretrained_model():
    """
    Load pretrained neural-network model employed 
    in Leitherer et al. arXiv:2303.12702 (2023).

    Returns
    -------
    model : keras.engine.functional.Functional
        Tensorflow/Keras model object.
        
    .. codeauthor:: Andreas Leitherer <andreas.leitherer@gmail.com>

    """
    path_to_pretrained_model = get_data_filename('data/pretrained_models/optimal_model.h5', package='ai4stem')
    model = load_model(path_to_pretrained_model)
    return model


def load_reference_lattices():
    """
    Load reference lattices (employed for training model
    in Leitherer et al. arXiv:2303.12702 (2023)) that 
    can be used to calculate the local lattice rotation
    (cf. Fig. 4 in Leitherer et al. arXiv:2303.12702 (2023))

    Returns
    -------
    reference_dict : TYPE
        DESCRIPTION.
        
    .. codeauthor:: Andreas Leitherer <andreas.leitherer@gmail.com>

    """

    reference_data_path = get_data_filename('data/reference_images')
    reference_images_paths = os.listdir(reference_data_path)

    numerical_to_text_labels, text_to_numerical_labels = load_class_dicts()
    class_labels = list(text_to_numerical_labels.keys())

    reference_dict = {}
    for class_label in class_labels:
        for ref_image in reference_images_paths:
            if class_label in ref_image:
                reference_dict[class_label] = np.load(os.path.join(reference_data_path,
                                                                  ref_image))
    return reference_dict
