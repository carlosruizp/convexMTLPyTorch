import torch
from zmq import EVENT_CLOSE_FAILED
torch.manual_seed(42)
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

import numpy as np
import numpy_indexed as npi
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import get_scorer, mean_squared_error
from sklearn.utils import gen_batches, shuffle
from tqdm import trange

from pytorch_lightning import Trainer

from icecream import ic

from convexmtl_torch.model.utils import *

import matplotlib.pyplot as plt


MAX_N_SKLEARN = 120000

class SimplePytorchModel(BaseEstimator):
    '''
    Simple Pytorch Model
    '''
    _opt_keys = {'weight_decay'}
    _loss_keys = {}

    @kwargs_decorator(
        {"lambda_lr": 0.01,
        "lambda_trainable" : True,
        "weight_decay": 0,
        "train_mode": "numpy",
        "random_state": 42})
    def __init__(self,
                 loss_fun: str,
                 common_module = None,
                 specific_modules: dict = None,
                 epochs: int=100,
                 learning_rate=0.1,
                 batch_size=128,
                 verbose: int=1,
                 lamb: float = 0.5,
                 **kwargs):
        super(SimplePytorchModel, self).__init__()
        self.common_module = common_module
        self.specific_modules = specific_modules
        self.loss_fun = loss_fun
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.lamb = lamb
        self.opt_kwargs = {}
        self.loss_kwargs = {}
        for k, v in kwargs.items():
            if k in SimplePytorchModel._opt_keys:
                self.opt_kwargs[k] = v
            elif k in SimplePytorchModel._loss_keys:
                self.loss_kwargs[k] = v
        self.lambda_lr = kwargs["lambda_lr"]
        self.lambda_trainable = kwargs["lambda_trainable"]
        self.train_mode = kwargs["train_mode"]
        self._random_state = kwargs["random_state"]

    
    def predict_(self, X, task_info=None):
        n, m = X.shape
        if task_info is None:
            task_info = self.task_info
        if task_info is not None:
            task_col = task_info
            self.unique, self.groups_idx = npi.group_by(X[:, task_col],
                                                        np.arange(n))
            X_data = np.delete(X, task_col, axis=1).astype(float)
            X_task = X[:, task_col]
            self.map_dic = dict(zip(self.unique, range(len(self.unique))))
            X_task_map = np.array([self.map_dic[x] for x in X_task])
        else:
            X_data = X

        X_tensor_data = torch.tensor(X_data)
        return self.model(X_tensor_data)

    def predict(self, X, task_info=None):
        """Predict using the multi-layer perceptron classifier
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        Returns
        -------
        y : array-like, shape (n_samples,) or (n_samples, n_classes)
            The predicted classes.
        """

        pred_tensor = self.predict_(X, task_info=task_info)
        y_pred = pred_tensor.detach().numpy()

        return y_pred

    def create_model(self, X_data, y, name=None):
        n, m = X_data.shape

        input_dim = m
        
        self.model = self._create_model(input_dim, name)
        return self.model

    def _create_model(self, input_dim, name=None, **model_kwargs):
        model = NeuralNetwork(n_features=input_dim)
        return model
    
    def get_model(self, X, y):
        if not hasattr(self, 'model'):
             self.model = self.create_model_(X, y, **self.model_kwargs)
        return self.model

    def get_opt_(self, **opt_kwargs):
        param_list = torch.nn.ParameterList(
                      [p for n, p in self.model.named_parameters() if p.requires_grad and n != "lamb"])
        
        opt = optim.AdamW(param_list, **opt_kwargs)#, lr=self.learning_rate)
        
        return opt

    def get_opt(self):
        assert hasattr(self, 'model')
        if not hasattr(self, 'opt'):
             self.opt = self.get_opt_(**self.opt_kwargs)
        return self.opt

    def _get_loss_fun(self, **loss_kwargs):
        switcher = {
            'mse': nn.MSELoss,
            'l1': nn.L1Loss,
            'cross_entropy': nn.CrossEntropyLoss
        }
        return switcher.get(self.loss_fun, "Invalid Loss Function.")(**loss_kwargs)
    
    
    # @profile(output_file='train_lightning_simple.prof', sort_by='cumulative', lines_to_print=20, strip_dirs=True)
    def _train_lightning(self, train_dl):
        trainer = Trainer(max_epochs=self.epochs)
        trainer.fit(self.model, train_dl)

    # @profile(output_file='train_dataloader_simple.prof', sort_by='cumulative', lines_to_print=20, strip_dirs=True)
    def _train_dataloader(self, train_dl):
        # train the network
        best_tr = np.inf
        best_val = np.inf

        self.lambda_history_ = []    
        with trange(int(self.epochs), desc='Loss', position=0, leave=True, disable=not self.verbose) as tri:
            for epoch in tri: # range(int(self.epochs)):
                # self.model.train()
                for i, (xb, yb) in enumerate(train_dl):
                    pred = self.model(xb)
                    loss = self.loss_fun_(pred, yb)
                    loss.backward()
                    # ic(loss.grad)
                    self.opt.step()
                    self.opt.zero_grad()
                # self.model.eval()
                # with torch.no_grad():
                #     tr_error = sum(self.loss_fun_(self.model(xb, tb), yb) for xb, tb, yb in train_dl)
                # current_error = tr_error



    # @profile(output_file='train_numpy_simple.prof', sort_by='cumulative', lines_to_print=20, strip_dirs=True)
    def _train_numpy(self, X_train, y_train):
        # train the network sklearn
        n_samples = X_train.shape[0]
        sample_idx = np.arange(n_samples, dtype=int)
        
        sample_idx = shuffle(sample_idx, random_state=self._random_state)

        with trange(int(self.epochs), desc='Loss', position=0, leave=True, disable=not self.verbose) as tri:
            for epoch in tri: # range(int(self.epochs)):

                accumulated_loss = 0.0
                for batch_slice in gen_batches(n_samples, self.batch_size):
                    xb = X_train[sample_idx[batch_slice]]
                    yb = y_train[sample_idx[batch_slice]]
                    pred = self.model(xb)
                    loss = self.loss_fun_(pred, yb)
                    loss.backward()
                    self.opt.step()
                    self.opt.zero_grad()
    
    
    
    # # @profile(output_file='fit_simple.prof', sort_by='cumulative', lines_to_print=10, strip_dirs=True)
    def fit(self, X, y, task_info=None, verbose=False, X_val=None, y_val=None, X_test=None, y_test=None, **kwargs):
        """Fit the model to data matrix X and target(s) y.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : returns a trained MLP model.
        """
        self.task_info = task_info

        n, m = X.shape
        if task_info is not None:
            task_col = task_info
            self.unique, self.groups_idx = npi.group_by(X[:, task_col],
                                                        np.arange(n))
            X_data = np.delete(X, task_col, axis=1).astype(float)
            X_task = X[:, task_col]
            self.map_dic = dict(zip(self.unique, range(len(self.unique))))
            X_task_map = np.array([self.map_dic[x] for x in X_task])
        else:
            X_data = X

        #if not hasattr(self, 'model'):
        self.model = self.create_model(X_data)
        # if not hasattr(self, 'opt'):
        self.opt = self.get_opt_()

        # self.reg_ = self._get_reg()
        self.loss_fun_ = self._get_loss_fun(**self.loss_kwargs)

        if y.ndim == 1:
            y_2d = y[:, None]
        else:
            y_2d = y

        X_train, y_train = map(torch.tensor, (X_data, y_2d))
        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        # if (X_val is not None) and (y_val is not None):
        #     if y_val.ndim == 1:
        #         y_val_2d = y_val[:, None]
        #     else:
        #         y_val_2d = y_val
        #     X_val, y_val = map(torch.tensor, (X_val, y_val_2d))
        #     valid_ds = TensorDataset(X_val, y_val)
        #     valid_dl = DataLoader(valid_ds, batch_size=self.batch_size * 2)
        #     val = True
        # else:
        #     val = False
        
        # train the network
        if self.train_mode == 'lightning':
            self._train_lightning(train_dl)
        elif self.train_mode == 'numpy':
            self._train_numpy(X_train, y_train)
        elif self.train_mode == 'dataloader':
            self._train_dataloader(train_dl)
        else:
            raise AttributeError("{} is not a valid train_mode".format(self.train_mode))
        return self