from icecream import ic
import convexmtl_torch.data.datasets as datasets
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
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        
        
    def load_dataset(self, dataset_name, **kwargs):
        try: 
            load_function = getattr(datasets, 'load_dataset_{}'.format(dataset_name))
            df, target, outer_cv, inner_cv, task_info = load_function(data_dir=self.data_dir, **kwargs)
        except AttributeError as e:
            print(e)
            # print(f"Invalid dataset_name: {dataset_name}.")
            raise(AttributeError(f"Invalid dataset_name: {dataset_name}."))
        return df, target, outer_cv, inner_cv, task_info
