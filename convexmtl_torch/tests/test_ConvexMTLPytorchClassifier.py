import unittest
import torch
from torch.nn import MSELoss
from torch.utils.data import TensorDataset
from convexmtl_torch.data.DataLoader import DataLoader
from convexmtl_torch.model.ConvexMTLPytorchModel import ConvexMTLPytorchClassifier
from convexmtl_torch.model.utils import NeuralNetwork
from convexmtl_torch.model.utils import ConvNet
from pytorch_lightning import Trainer

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.exceptions import NotFittedError

from icecream import ic


class TestConvexMTLPytorchClassifier(unittest.TestCase):
    def setUp(self):
        
        self.module = ConvNet # NeuralNetwork
        self.specific_modules = {0: NeuralNetwork, 1: NeuralNetwork}
        self.specific_lambda = True
        self.epochs = 50
        self.batch_size = 32
        self.max_train = 500
        

        self.model = ConvexMTLPytorchClassifier(common_module=self.module)
        ic(self.model)

    # def test_fit_tabular(self):
    #     dataloader = DataLoader('my_data')
    #     X, y, _,  _,  task_info = dataloader.load_dataset(dataset_name='school')
    #     X_train, y_train = X[:self.max_train], y[:self.max_train]
    #     self.model.fit(X_train, y_train)
    #     try:
    #         self.model.predict(X_train)
    #     except NotFittedError as e:
    #         print(repr(e))

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

    def test_fit_image(self):
        dataloader = DataLoader('my_data')
        X, y, _,  _,  task_info = dataloader.load_dataset(dataset_name='variations_mnist')
        X_train, y_train = X[:self.max_train], y[:self.max_train]
        self.model.fit(X_train, y_train, new_shape=(1, 28, 28))
        try:
            self.model.predict(X_train)
        except NotFittedError as e:
            print(repr(e))



if __name__ == '__main__':
    unittest.main()