import unittest
import torch
from torch.nn import MSELoss
from torch.utils.data import TensorDataset
from convexmtl_torch.data.DataLoader import DataLoader
from convexmtl_torch.model.GraphLaplacianMTLPytorchModel import GraphLaplacianMTLPytorchRegressor
from convexmtl_torch.model.utils import NeuralNetwork, NeuralNetworkFeatLearn
from convexmtl_torch.model.utils import ConvNet
from pytorch_lightning import Trainer

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.exceptions import NotFittedError

from icecream import ic


class TestGraphLaplacianMTLPytorchRegressor(unittest.TestCase):
    def setUp(self):
        
        self.module = NeuralNetworkFeatLearn
        self.specific_lambda = True
        self.epochs = 50
        self.batch_size = 32
        self.max_train = None
        
        self.model = GraphLaplacianMTLPytorchRegressor(common_module=self.module, 
                                                       adj_trainable=True,
                                                       train_mode='lightning')
        ic(self.model)

    def test_fit_tabular(self):
        dataloader = DataLoader('my_data')
        X, y, _,  _,  task_info = dataloader.load_dataset(dataset_name='school_mini')
        X_train, y_train = X[:self.max_train], y[:self.max_train]
        self.model.fit(X_train, y_train)
        try:
            self.model.predict(X_train)
        except NotFittedError as e:
            print(repr(e))

    # def test_score_train_tabular(self):
    #     dataloader = DataLoader('my_data')
    #     X, y, _,  _,  task_info = dataloader.load_dataset(dataset_name='school')
    #     X_train, y_train = X[:self.max_train], y[:self.max_train]
    #     self.model.fit(X_train, y_train)
    #     try:
    #         pred = self.model.predict(X_train)
    #     except NotFittedError as e:
    #         print(repr(e))
    #     score_train = mean_squared_error(y_train, pred)
    #     ic(score_train)

    # def test_fit_image(self):
    #     dataloader = DataLoader('my_data')
        
    #     X, y, outer_cv,  _,  task_info = dataloader.load_dataset(dataset_name='variations_mnist_mini')
    #     for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, X[:, task_info].astype(str))):
    #         X_train = X[train_idx]
    #         y_train = y[train_idx]
    #         break
    #     X_train, y_train = X_train[:self.max_train], y_train[:self.max_train]
    #     self.model.fit(X_train, y_train, new_shape=(1, 28, 28))
    #     try:
    #         self.model.predict(X_train)
    #     except NotFittedError as e:
    #         print(repr(e))



if __name__ == '__main__':
    unittest.main()