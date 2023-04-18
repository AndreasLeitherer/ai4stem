import umap
from ai4stem.utils.utils_data import load_pretrained_model

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
        
        
def unsupervised_analysis(image, window_size=100, model=None, n_iter=100, method='umap', params=None):
    
    if model == None:
        model = load_pretrained_model()
    if params == None:
        params = {'n_neighbors': 200,
                  'min_dist': 0.9,
                  'metric': 'euclidean',
                  'n_components': 2}
        
    
        
    