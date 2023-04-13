import numpy as np

class SVD:
    def __init__(self, n_components=None):
        self.n_components = n_components
        
    def fit(self, X):
        U, sigma, VT = np.linalg.svd(X, full_matrices=False)
        self.U = U[:, :self.n_components]
        self.sigma = sigma[:self.n_components]
        self.VT = VT[:self.n_components, :]
        
    def fit_transform(self, X):
        self.fit(X)
        X_transformed = self.U @ np.diag(self.sigma)
        return X_transformed
    
    def transform(self, X):
        X_transformed = self.U @ np.diag(self.sigma)
        return X_transformed