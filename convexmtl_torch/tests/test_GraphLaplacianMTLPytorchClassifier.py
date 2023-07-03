import unittest
import torch
from torch.nn import MSELoss
from torch.utils.data import TensorDataset
from convexmtl_torch.data.DataLoader import DataLoader
from convexmtl_torch.model.GraphLaplacianMTLPytorchModel import GraphLaplacianMTLPytorchClassifier
from convexmtl_torch.model.utils import NeuralNetworkFeatLearn
from convexmtl_torch.model.utils import ConvNetFeatLearn
from pytorch_lightning import Trainer

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.exceptions import NotFittedError

from icecream import ic


class TestGraphLaplacianMTLPytorchClassifier(unittest.TestCase):
    def setUp(self):
        
        self.specific_lambda = True
        self.epochs = 10
        self.batch_size = 32
        self.max_train = 500
        

    def test_fit_tabular(self):
        model = GraphLaplacianMTLPytorchClassifier(common_module=NeuralNetworkFeatLearn, train_mode='lightning', epochs=self.epochs)

        dataloader = DataLoader('my_data')
        X, y, _,  _,  task_info = dataloader.load_dataset(dataset_name='landmine_mini')
        X_train, y_train = X[:self.max_train], y[:self.max_train]
        model.fit(X_train, y_train)
        try:
            model.predict(X_train)
        except NotFittedError as e:
            print(repr(e))

    # def test_fit_image(self):
    #     model = GraphLaplacianMTLPytorchClassifier(common_module=ConvNetFeatLearn, train_mode='numpy', epochs=self.epochs)

    #     dataloader = DataLoader('my_data')
    #     X, y, _,  _,  task_info = dataloader.load_dataset(dataset_name='variations_mnist_mini')
    #     X_train, y_train = X[:self.max_train], y[:self.max_train]
    #     model.fit(X_train, y_train, new_shape=(1, 28, 28))
    #     try:
    #         model.predict(X_train)
    #     except NotFittedError as e:
    #         print(repr(e))



if __name__ == '__main__':
    unittest.main()