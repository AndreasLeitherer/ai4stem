import umap
from ai4stem.utils.utils_data import load_pretrained_model
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from ai4stem.utils.utils_prediction import predict
from ai4stem.utils.utils_nn import get_truncated_model, get_nn_representations, reshape_data_to_input_size
import logging

class UnsupervisedLearning():
    """
    UNDER CONSTRUCTION.
    Class for generalizing unsupervised learning (here dimension reduction / manifold learning).
    In the current implementation, only umap is contained.
    """
    
    def __init__(self, method='umap', params={'n_neighbors': 200,
                                              'min_dist': 0.9,
                                              'metric': 'euclidean',
                                              'n_components': 2}):
        
        self.method = method
        self.params = params
    
    def fit(self, data):
        if self.method == 'umap':
            mapper = umap.UMAP(**self.params).fit(data)
            return mapper
    def fit_transform(self, data):
        if self.method == 'umap':
            embedding = umap.UMAP(**self.params).fit_transform(data)
            return embedding
    def transform(self, data, mapper):
        if self.method == 'umap':
            embedding = mapper.transform(data)
            return embedding

def embeddable_image(data):
    """
    Convert input data (an image) such that it can be embedded into 
    an interactive plot (in particular, bokeh hover plot in jupter notebook).
    Using BytesIO, the input data can be kept as bytes in an in-memory buffer
    (see https://www.digitalocean.com/community/tutorials/python-io-bytesio-stringio
     for more details)

    Parameters
    ----------
    data : ndarray
        2D input data (an image).

    Returns
    -------
    string
        Value string of BytesIO object that  
        can be passed to interactive bokeh tools in jupyter notebook 
        (see unuspervised learning notebook for an example).

    """
    data = cv2.normalize(data, None,
                       alpha=0, beta=1,
                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    #img_data = 255 - 15 * data#.astype(np.uint8)
    #image = Image.fromarray(img_data, mode='L')
    # image = image.convert("L")
    data = (255 * data).astype(np.uint8)
    image = Image.fromarray(data)
    image = image.convert("L")
    buffer = BytesIO()
    image.save(buffer, format='jpeg')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()
        
        
def unsupervised_analysis(image, window_size=100, stride_size=[36, 36], 
                          model=None, layer_name=None, n_iter=100, 
                          method='umap', params=None):


    # Check if model and network layer are specified correctly
    if model == None and layer_name == None:
        model = load_pretrained_model()
        layer_name = 'Dense_1'
    elif model == None and not (layer_name == None):
        model = load_pretrained_model()
        if not (layer_name in [_.name for _ in model.layers]):
            raise ValueError("Model was None, so used pretrained model - but specified layer name is not compatible.")
    elif not (model == None) and layer_name == None:
        raise ValueError("Layer name was not specified but the model was. Please specify both or none.")
        
    
    # Set unsupervised analysis parameters
    if params == None:
        params = {'n_neighbors': 200,
                  'min_dist': 0.9,
                  'metric': 'euclidean',
                  'n_components': 2}
    
    # get local fragments and FFT-HAADF descriptors
    sliced_images, fft_descriptors = predict(image, model, n_iter, 
                                             stride_size, window_size,
                                             only_fragments_and_descriptor=True)
    
    # reshape data
    data = reshape_data_to_input_size(fft_descriptors, model)
    
    # get neural-network representations
    logging.info('Calculate neural-network representations.')
    nn_representations = get_nn_representations(model, data, 
                                                layer_name, n_iter)
    
    # conduct unsupervised analysis
    logging.info('Apply unsupervised method {}'.format(method))
    unsupervised_method = UnsupervisedLearning(method, params)
    embedding = unsupervised_method.fit_transform(nn_representations)
    
    logging.info('Analysis finished.')
    
    return embedding
    
    