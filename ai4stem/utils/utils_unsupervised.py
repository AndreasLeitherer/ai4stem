import umap

class UnsupervisedLearning():
    
    
    def __init__(self, method='umap', params={'nn': 200}):
        
        self.method = method
        self.params = params
        pass
    
    def fit(self, data):
        if self.method == 'umap':
            mapper = umap.fit(data, **self.params)
            return mapper
    def fit_transform(self, data):
        if self.method == 'umap':
            embedding = umap.fit_transform(data, **self.params)
            return embedding