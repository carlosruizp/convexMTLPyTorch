from icecream import ic
from mtlskl.data.dataLoader import load_datasets


class DataLoader:
    """
    A class for loading popular datasets from scikit-learn library.
    
    Parameters
    ----------
    dataset_name : str, default: 'iris'
        The name of the dataset to load. Can be 'iris', 'digits'
    
    Attributes
    ----------
    data : array-like
        The data of the dataset
    target : array-like
        The target values of the dataset
    feature_names : array-like
        The feature names of the dataset
    target_names : array-like
        The target names of the dataset
    DESCR : str
        The description of the dataset
        
    """
    def __init__(self, dataset_name='iris'):
        self.dataset_name = dataset_name
        self.data, self.target, self.inner_cv, self.outer_cv, self.task_info = self._load_dataset()
        
    def _load_dataset(self, task_type='predefined', nested_cv=False,
                      seed=42, max_size=-1):
        try:
            X, y, inner_cv, outer_cv, task_info = load_datasets(self.dataset_name, task_type=task_type, 
                                                                nested_cv=nested_cv, 
                                                                seed=seed, max_size=max_size)
        except ValueError as e:
            raise e

        return X, y, inner_cv, outer_cv, task_info

# Example 
# data_loader = DataLoader(dataset_name='boston')
# X, y = data_loader.data, data_loader.target
