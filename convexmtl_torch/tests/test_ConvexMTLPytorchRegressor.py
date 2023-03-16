import unittest
import torch
from torch.nn import MSELoss
from torch.utils.data import TensorDataset, DataLoader
from convexmtl_torch.data.DataLoader import DataLoader as DL

from convexmtl_torch.model.ConvexMTLPytorchModel import ConvexMTLPytorchRegressor
from convexmtl_torch.model.utils import NeuralNetwork
from convexmtl_torch.model.utils import ConvNet
from pytorch_lightning import Trainer

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.exceptions import NotFittedError

from icecream import ic

import copy


class TestConvexMTLPytorchRegressor(unittest.TestCase):
    def setUp(self):
        
        self.module = NeuralNetwork
        self.specific_modules = {0: NeuralNetwork, 1: NeuralNetwork}
        self.specific_lambda = True
        self.epochs = 3
        self.batch_size = 32
        self.max_train = None
        
        self.model = ConvexMTLPytorchRegressor(common_module=self.module, 
                                               lambda_trainable=True,
                                               specific_lambda=True,
                                               epochs=self.epochs,
                                               batch_size=self.batch_size,
                                               train_mode='lightning')
        ic(self.model)

    def test_fit_tabular(self):
        dataloader = DL('my_data')
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


    # def test_fit_equiv(self):
    #     dataloader = DL('my_data')

    #     model = ConvexMTLPytorchRegressor(common_module=NeuralNetwork, 
    #                                            lambda_trainable=True,
    #                                            specific_lambda=True,
    #                                            epochs=1,
    #                                            batch_size=self.batch_size,
    #                                            train_mode='lightning')

        
    #     X, y, outer_cv,  _,  task_info = dataloader.load_dataset(dataset_name='school_mini')
    #     for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, X[:, task_info].astype(str))):
    #         X_train = X[train_idx]
    #         y_train = y[train_idx]
    #         break
    #     X_train, y_train = X_train[:self.max_train], y_train[:self.max_train]
    #     model.fit(X_train, y_train)
    #     self.model.fit(X_train, y_train)
    #     try:
    #         pred = self.model.predict(X_train)
    #     except NotFittedError as e:
    #         print(repr(e))
    #     ic(mean_squared_error(y_train, pred))

        
    #     X_train_tensor = torch.tensor(X_train[:, :-1]).float()
    #     t_train_tensor = torch.tensor(X_train[:, -1]).float()
    #     y_train_tensor = torch.tensor(y_train).float()

    #     train_ds = TensorDataset(X_train_tensor, t_train_tensor, y_train_tensor)
    #     train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        
    #     trainer = Trainer(max_epochs=self.epochs)
    #     ic(self.model)

    #     model_torch = model.model
        
    #     trainer.fit(model_torch, train_dl)

    #     pred_torch = model_torch(X_train_tensor, t_train_tensor).detach().numpy()
    #     ic(mean_squared_error(y_train, pred_torch))
        
    #     ic(mean_squared_error(pred, pred_torch))
        



if __name__ == '__main__':
    unittest.main()