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

from timeit import default_timer as timer

class TestConvexMTLPytorchClassifier(unittest.TestCase):
    def setUp(self):
        
        self.specific_modules = {0: NeuralNetwork, 1: NeuralNetwork}
        self.specific_lambda = True
        self.epochs = 50
        self.batch_size = 32
        self.max_train = 500
        

    # def test_fit_tabular(self):
    #     model = ConvexMTLPytorchClassifier(common_module=NeuralNetwork, train_mode='lightning')

    #     dataloader = DataLoader('my_data')
    #     X, y, _,  _,  task_info = dataloader.load_dataset(dataset_name='landmine_mini')
    #     X_train, y_train = X[:self.max_train], y[:self.max_train]
    #     model.fit(X_train, y_train)
    #     try:
    #         model.predict(X_train)
    #     except NotFittedError as e:
    #         print(repr(e))

    # def test_fit_image(self):
    #     model = ConvexMTLPytorchClassifier(common_module=ConvNet)

    #     dataloader = DataLoader('my_data')
    #     X, y, _,  _,  task_info = dataloader.load_dataset(dataset_name='variations_mnist_mini')
    #     X_train, y_train = X[:self.max_train], y[:self.max_train]
    #     model.fit(X_train, y_train, new_shape=(1, 28, 28))
    #     try:
    #         model.predict(X_train)
    #     except NotFittedError as e:
    #         print(repr(e))

    # def test_fit_time_variations_mnist(self):
    #     dataloader = DataLoader('my_data')
    #     X, y, _,  _,  task_info = dataloader.load_dataset(dataset_name='variations_mnist')
    #     X_train, y_train = X[:1000], y[:1000]

    #     model = ConvexMTLPytorchClassifier(common_module=ConvNet, train_mode='numpy', epochs=10)        
    #     start = timer()
    #     model.fit(X_train, y_train, new_shape=(1, 28, 28))
    #     stop = timer()
    #     time_numpy = stop - start
        

    #     model = ConvexMTLPytorchClassifier(common_module=ConvNet, train_mode='lightning', epochs=10)        
    #     start = timer()
    #     model.fit(X_train, y_train, new_shape=(1, 28, 28))
    #     stop = timer()
    #     time_lightning = stop - start

    #     ic(time_numpy)
    #     ic(time_lightning)

    def test_fit_time_landmine(self):
        dataloader = DataLoader('my_data')
        X, y, _,  _,  task_info = dataloader.load_dataset(dataset_name='landmine')
        X_train, y_train = X[:1000], y[:1000]

        model = ConvexMTLPytorchClassifier(train_mode='numpy', epochs=100)        
        start = timer()
        model.fit(X_train, y_train)
        stop = timer()
        time_numpy = stop - start
        

        model = ConvexMTLPytorchClassifier(train_mode='lightning', epochs=100)        
        start = timer()
        model.fit(X_train, y_train)
        stop = timer()
        time_lightning = stop - start

        ic(time_numpy)
        ic(time_lightning)
        



if __name__ == '__main__':
    unittest.main()