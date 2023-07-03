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
            if 'mini' in dataset_name:
                dataset_name_ = '_'.join(dataset_name.split('_')[:-1])
                ic(dataset_name_)
                kwargs = {**{'max_size': 500}, **kwargs}
            else:
                dataset_name_ = dataset_name
            if 'synthetic' in kwargs and kwargs['synthetic'] is True:
                kwargs.pop('synthetic')
                df, target, outer_cv, inner_cv, task_info = datasets.load_synthetic_dataset(dataset_name=dataset_name, **kwargs)
            else:
                load_function = getattr(datasets, 'load_dataset_{}'.format(dataset_name_))
                df, target, outer_cv, inner_cv, task_info = load_function(data_dir=self.data_dir, **kwargs)
        except AttributeError as e:
            print(e)
            # print(f"Invalid dataset_name: {dataset_name}.")
            raise(AttributeError(f"Invalid dataset_name: {dataset_name}."))
        return df, target, outer_cv, inner_cv, task_info
