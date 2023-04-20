import umap
from ai4stem.utils.utils_data import load_pretrained_model
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from ai4stem.utils.utils_prediction import predict

class UnsupervisedLearning():
    """
    UNDER CONSTRUCTION.
    Class for generalizing unsupervised learning (here dimension reduction / manifold learning).
    In the current implementation, only umap is contained.
    """
    
    def __init__(self, method='umap', params={'n_neighbors': 200,
                                              'min_dist': 0.9,
                                              'metric': 'euclidean',
                                              'n_components': 2},
                 mapper=None):
        
        self.method = method
        self.params = params
        self.mapper = None
    
    def fit(self, data):
        if self.method == 'umap':
            mapper = umap.fit(data, **self.params)
            return mapper
    def fit_transform(self, data):
        if self.method == 'umap':
            embedding = umap.fit_transform(data, **self.params)
            return embedding
    def transform(self, data):
        if self.method == 'umap':
            embedding = self.mapper.transform(data)
            return embedding

def embeddable_image(data):
    """
    Convert input data into image that can be embedded into 
    an interactive plot (in particular, bokeh hover plot in jupter notebook).

    Parameters
    ----------
    data : ndarray
        2D input data (an image).

    Returns
    -------
    string
        Value string of BytesIO object that  
        can be passed to interactive bokeh tools in jupyter notebook 
        (see unuspervised learning notebook).

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
                          model=None, n_iter=100, method='umap', params=None):
    
    if model == None:
        model = load_pretrained_model()
    if params == None:
        params = {'n_neighbors': 200,
                  'min_dist': 0.9,
                  'metric': 'euclidean',
                  'n_components': 2}
        
    
    
    sliced_images, fft_descriptors, prediction, mutual_information = predict(image, model, n_iter, 
                                                                             stride_size, window_size)
    
    
    