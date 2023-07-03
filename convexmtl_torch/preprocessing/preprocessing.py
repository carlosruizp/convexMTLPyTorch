import numpy as np
from sklearn.preprocessing import StandardScaler

from icecream import ic

    
class MTLStandardScaler(StandardScaler):
    def fit(self, X, y=None, task_col=-1):
        X_data = np.delete(X, [task_col], axis=1)
        # ic(X_data.shape)

        super().fit(X_data)
                
        return self
        
    def transform(self, X, task_col=-1):
        # ic(X.shape)
        X_data = np.delete(X, [task_col], axis=1)
        # ic(X_data.shape)
        X_tr = super().transform(X_data)
        X_concat = np.concatenate((X_tr, X[:, task_col][:, None]), axis=1)
        
        return X_concat

    def inverse_transform(self, X, task_col=-1):
        X_data = np.delete(X, [task_col], axis=1)
        X_tr = super().inverse_transform(X_data)
        X_concat = np.concatenate((X_tr, X[:, task_col][:, None]), axis=1)
        
        return X_concat