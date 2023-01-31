
import torch
from zmq import EVENT_CLOSE_FAILED
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

from convexmtl_torch.model.HardSharingTorchCombinator import HardSharingTorchCombinator

from pytorch_lightning import Trainer

from icecream import ic

from convexmtl_torch.model.utils import *

import matplotlib.pyplot as plt


class HardSharingMTLPytorchModel(BaseEstimator):
    """
    Abstract class from which HardSharingMTLPytorchClassifier and HardSharingMTLPytorchRegressor inherit.
    Attributes
    --------------
    - layers: list containing the number of neurons in each one of the common hidden layers 
    of the network. The input and output layers sizes are determined by the problem.
    - learning_rate: the 
    """
    _opt_keys = {'weight_decay'}
    _loss_keys = {}

    @kwargs_decorator(
        {"weight_decay": 0,
        "val_size": 0.1,
        "min_delta": 1e-3,
        "patience": 25,
        "random_state": 42,
        "early_stopping": True,
        "lr_scheduler": True})
    def __init__(self,
                 loss_fun: str,
                 common_module = None,
                 specific_modules: dict = None,
                 epochs: int=100,
                 learning_rate=0.001,
                 batch_size=128,
                 verbose: int=1,
                 **kwargs):
                 
        super(HardSharingMTLPytorchModel, self).__init__()
        self.common_module = common_module
        self.specific_modules = specific_modules
        self.loss_fun = loss_fun
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.opt_kwargs = {}
        self.loss_kwargs = {}
        for k, v in kwargs.items():
            if k in HardSharingMTLPytorchModel._opt_keys:
                self.opt_kwargs[k] = v
            elif k in HardSharingMTLPytorchModel._loss_keys:
                self.loss_kwargs[k] = v
        # self.weight_decay = kwargs["weight_decay"]
        self.val_size = kwargs["val_size"]
        self.min_delta = kwargs["min_delta"]
        self.patience = kwargs["patience"]
        self.metrics = kwargs["metrics"]
        self._random_state = kwargs["random_state"]
        self.train_mode = kwargs["train_mode"]
        self.early_stopping = kwargs["early_stopping"]
        self.lr_scheduler = kwargs["lr_scheduler"]
        
         
    
    
    def predict_(self, X, task_info=-1, **kwargs):
        n, m = X.shape


        task_col = task_info
        self.unique, self.groups_idx = npi.group_by(X[:, task_col],
                                                    np.arange(n))
        X_data = np.delete(X, task_col, axis=1).astype(float)
        X_task = X[:, task_col]
        X_task_map = np.array([self.map_dic[x] for x in X_task])

        if 'new_shape' in kwargs and kwargs['new_shape'] is not None:
            ic(X_data.shape)
            new_shape = [X_data.shape[0]] + list(kwargs['new_shape'])
            X_data = np.reshape(X_data, tuple(new_shape))
        else:
            if self.new_shape is not None:
                new_shape = [n] + list(self.new_shape)
                X_data = np.reshape(X_data, tuple(new_shape))

        X_tensor_data = torch.tensor(X_data)
        X_tensor_task = torch.tensor(X_task_map)
        pred_tensor = self.model(X_tensor_data, X_tensor_task)
        return pred_tensor


    def predict(self, X, task_info=-1, **kwargs):
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

        return pred_tensor.detach().numpy()

    def get_opt_(self, **opt_kwargs):
        # ic([(n, p) for n, p in self.model.named_parameters()])
        param_list = torch.nn.ParameterList(
                      [p for n, p in self.model.named_parameters()])

        ic(param_list)
        ic(self.model)
        # namedparam_list = [n for n, p in self.model.named_parameters() if p.requires_grad]
        # ic(namedparam_list)
        # namedparam_list = [n for n, p in self.model.named_parameters() if p.requires_grad]
        if hasattr(self.model, 'output_layers'):
            for name, module in self.model.output_layers.items():
                param_list.extend(
                    torch.nn.ParameterList(
                        [p for n, p in module.named_parameters() if p.requires_grad])
                )

        ic(param_list)

        opt = optim.AdamW(param_list, **opt_kwargs)#, lr=self.learning_rate)
        
        return opt

    def get_opt(self):
        assert hasattr(self, 'model')
        if not hasattr(self, 'opt'):
             self.opt = self.get_opt_(**self.opt_kwargs)
        return self.opt

    def get_model(self, X, y):
        if not hasattr(self, 'model'):
             self.model = self.create_model_(X, y, **self.model_kwargs)
        return self.model

    def _get_reg(self, **reg_kwargs):
        raise NotImplementedError()
    
    def _get_loss_fun(self, **loss_kwargs):
        switcher = {
            'mse': nn.MSELoss,
            'l1': nn.L1Loss,
            'cross_entropy': nn.CrossEntropyLoss
        }
        return switcher.get(self.loss_fun, "Invalid Loss Function.")(**loss_kwargs)

    def _get_metric(self, metric):
        scorer = get_scorer(metric)
        return scorer
    
    
    # @profile(output_file='train_lightning_convexmtl.prof', sort_by='cumulative', lines_to_print=20, strip_dirs=True)
    def _train_lightning(self, train_dl, valid_dl):
        trainer = Trainer(max_epochs=self.epochs)
        if self.val:
            trainer.fit(self.model, train_dl, valid_dl)
        else:
            trainer.fit(self.model, train_dl)

    # @profile(output_file='train_dataloader_convexmtl.prof', sort_by='cumulative', lines_to_print=20, strip_dirs=True)
    def _train_dataloader(self, train_dl, valid_dl):
        best_tr = np.inf
        best_val = np.inf
        old_val_loss = np.inf
        early_stopping_count = 0
        with trange(int(self.epochs), desc='Loss', position=0, leave=True, disable=not self.verbose) as tri:
                for epoch in tri:
                    self.model.train()
                    for i, (xb, tb, yb) in enumerate(train_dl):
                        pred = self.model(xb, tb)
                        loss = self.loss_fun_(pred, yb)
                        loss.backward()
                        # ic(loss.grad)
                        self.opt.step()
                        self.opt.zero_grad()
                
                    self.model.eval()
                    with torch.no_grad():
                        tr_loss = sum(self.loss_fun_(self.model(xb, tb), yb) for xb, tb, yb in train_dl)

                    if tr_loss <= best_tr:
                        best_tr = tr_loss
                    
                    if self.val:                        
                        with torch.no_grad():
                            val_loss = sum(self.loss_fun_(self.model(xb, tb), yb) for xb, tb, yb in valid_dl)
                        
                        if val_loss >= best_val - self.min_delta: # old_val_loss - self.min_delta:
                            early_stopping_count += 1
                        else:
                            early_stopping_count = 0

                        old_val_loss = val_loss

                        if self.patience > 0 and early_stopping_count >= self.patience:
                            break


                        if val_loss <= best_val:
                            best_val = val_loss             

                    # store loss and metrics history
                    self.history['loss'].append(tr_loss.detach().item())
                    if self.val:
                        self.history['val_loss'].append(val_loss.detach().item())

                    
                    # for m in self.metrics:
                    #     scoring_fun = metric_scorers[m]._score_func
                    #     score = scoring_fun(y_train_, pred_train)
                    #     self.history[m].append(score)
                    #     if self.val:
                    #         self.history['val_{}'.format(m)].append(scoring_fun(y_val_, pred_val))

                    tr_text = f"{tr_loss:4.2f}({best_tr:4.2f})"
                    if self.val:
                        val_text = f"{val_loss:4.2f}({best_val:4.2f})"
                    else:
                        val_text = "N/A"

                    tri.set_description(
                            f"Loss: {tr_loss:6.4e}, Tr:{tr_text}, V:{val_text}")


    # @profile(output_file='train_numpy_convexmtl.prof', sort_by='cumulative', lines_to_print=20, strip_dirs=True)
    def _train_numpy(self, X_train, t_train, y_train, X_val, t_val, y_val):

        best_tr = np.inf
        best_val = np.inf
        old_val_loss = np.inf
        early_stopping_count = 0
        cont = 0
        with trange(int(self.epochs), desc='Loss', position=0, leave=True, disable=not self.verbose) as tri:
            for epoch in tri:
                self.model.train()
                n_train = X_train.shape[0]
                sample_idx = np.arange(n_train, dtype=int)                        
                sample_idx = shuffle(sample_idx, random_state=self._random_state)
                accumulated_loss = 0
                for batch_slice in gen_batches(n_train, self.batch_size):
                    xb = X_train[sample_idx[batch_slice]]
                    tb = t_train[sample_idx[batch_slice]]
                    yb = y_train[sample_idx[batch_slice]]
                    pred = self.model(xb, tb)
                    loss = self.loss_fun_(pred, yb)
                    accumulated_loss += loss
                    loss.backward()
                    self.opt.step()
                    self.opt.zero_grad()

                    # print(tb)

                    # # Print self.model's state_dict
                    # print("Model's state_dict:")
                    # for param_tensor in self.model.state_dict():
                    #     if '6' in param_tensor:
                    #         print(param_tensor, "\t", self.model.state_dict()[param_tensor])   

                    # cont += 1
                    # if cont == 2:
                    #     exit()                
                
                tr_loss = accumulated_loss

                if tr_loss <= best_tr:
                        best_tr = tr_loss

                if self.val:
                    n_val = X_val.shape[0]   
                    sample_idx = np.arange(n_val, dtype=int)                      
                    with torch.no_grad():
                        val_loss = sum(self.loss_fun_(self.model(X_val[sample_idx[batch_slice]], t_val[sample_idx[batch_slice]]), y_val[sample_idx[batch_slice]]) 
                                        for batch_slice in gen_batches(n_val, self.batch_size))
                    
                    # if val_loss >= best_val - self.min_delta: # old_val_loss - self.min_delta:
                    #     early_stopping_count += 1
                    # else:
                    #     early_stopping_count = 0

                    # old_val_loss = val_loss

                    # if self.patience > 0 and early_stopping_count >= self.patience:
                    #     break                   

                    # if val_loss <= best_val:
                    #     best_val = val_loss  

                    if self.early_stopping:                    
                        self._early_stopping(val_loss, epoch, self.model, self.opt, self.loss_fun_)

                        if self._early_stopping.early_stop:
                            # # Print self.model's state_dict
                            # print("Model's state_dict:")
                            # for param_tensor in self.model.state_dict():
                            #     print(param_tensor, "\t", self.model.state_dict()[param_tensor])
                            # # Print self.model's state_dict
                            # print("Model's state_dict:")
                            # for param_tensor in self._early_stopping.best_model_state_dict:
                            #     print(param_tensor, "\t", self._early_stopping.best_model_state_dict[param_tensor])
                            
                            self.model.load_state_dict(self._early_stopping.best_model_state_dict)

                            # # Print self.model's state_dict
                            # print("Model's state_dict:")
                            # for param_tensor in self.model.state_dict():
                            #     print(param_tensor, "\t", self.model.state_dict()[param_tensor])
                            
                            break
                    
                    best_val = self._early_stopping.best_loss

                    if self.lr_scheduler:
                        self._lr_scheduler(val_loss)
                          

                # store loss and metrics history
                self.history['loss'].append(tr_loss.detach().item())
                if self.val:
                    self.history['val_loss'].append(val_loss.detach().item())


                tr_text = f"{tr_loss:4.2f}({best_tr:4.2f})"
                if self.val:
                    val_text = f"{val_loss:4.2f}({best_val:4.2f})"
                else:
                    val_text = "N/A"

                tri.set_description(
                        f"Loss: {tr_loss:6.4e}, Tr:{tr_text}, V:{val_text}")

            self.tot_epochs = epoch + 1

    
    def fit(self, X, y, task_info=-1, verbose=False, X_val=None, y_val=None, X_test=None, y_test=None, **kwargs):
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
        if isinstance(task_info, int):
            task_col = task_info
            self.unique, self.groups_idx = npi.group_by(X[:, task_col],
                                                        np.arange(n))
            X_data = np.delete(X, task_col, axis=1).astype(float)
            X_task = X[:, task_col]
        else:
            t = task_info
            self.unique, self.groups_idx = npi.group_by(t,
                                                        np.arange(n))
            X_data = X
            X_task = t
        
        if 'new_shape' in kwargs and kwargs['new_shape'] is not None:
            self.new_shape = kwargs['new_shape']
            new_shape = [n] + list(self.new_shape)
            X_data = np.reshape(X_data, tuple(new_shape))
        elif hasattr(self, 'new_shape'):
            new_shape = [n] + list(self.new_shape)
            X_data = np.reshape(X_data, tuple(new_shape))
        
        self.map_dic = dict(zip(self.unique, range(len(self.unique))))
        X_task_map = np.array([self.map_dic[x] for x in X_task])


        ic(X_data.shape)

        #if not hasattr(self, 'model'):
        self.model = self.create_model(X_data, X_task, y)
        # if not hasattr(self, 'opt'):
        self.opt = self.get_opt_()

        # self.reg_ = self._get_reg()
        self.loss_fun_ = self._get_loss_fun(**self.loss_kwargs)


        # Generate validation sets
        if (X_val is not None) and (y_val is not None):
            if y_val.ndim == 1:
                y_val_2d = y_val[:, None]
            else:
                y_val_2d = y_val
            Xval_data = np.delete(X_val, task_col, axis=1).astype(float)
            Xval_task = X_val[:, task_col]
            Xval_task = np.array([self.map_dic[x] for x in Xval_task])
            
            X_val = Xval_data
            t_val = Xval_task
            y_val = y_val_2d

            X_train = X_data
            t_train = X_task_map
            y_train = y

            self.val = True
        else:
            if self.val_size == 0:
                X_train = X_data
                t_train = X_task_map
                y_train = y
                X_val, t_val, y_val = None, None, None
                self.val = False
            else:
                X_train, X_val, t_train, t_val, y_train, y_val = train_test_split(X_data, X_task_map, y, 
                                                                                shuffle=True, test_size=self.val_size)            
                self.val = True
        
        # Numpy Arrays to Pytorch Tensors
        X_train, t_train, y_train = self.generate_tensors(X_train, t_train, y_train)
        # t_train = t_train.int()
        train_ds = TensorDataset(X_train, t_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        if self.val:
            X_val, t_val, y_val = self.generate_tensors(X_val, t_val, y_val)
            valid_ds = TensorDataset(X_val, t_val, y_val)
            valid_dl = DataLoader(valid_ds, batch_size=self.batch_size * 2)

            if self.early_stopping:
                self._early_stopping = EarlyStopping(patience=self.patience,
                                                     min_delta=self.min_delta)

            # lr scheduler
            if self.lr_scheduler:
                self._lr_scheduler = LRScheduler(self.opt, patience=self.patience)


        # initialize metrics and loss variables
        metric_scorers = {m: self._get_metric(m) for m in self.metrics}
        self.history = {}
        self.history['loss'] = []
        if self.val:
            self.history['val_loss'] = []
        for m in self.metrics:
            self.history[m] = []
            if self.val:
                self.history['val_{}'.format(m)] = []


        # train the network
        if self.train_mode == 'lightning':
            self._train_lightning(train_dl, valid_dl)
        elif self.train_mode == 'numpy':
            self._train_numpy(X_train, t_train, y_train, X_val, t_val, y_val)
        elif self.train_mode == 'dataloader':
            self._train_dataloader(train_dl, valid_dl)
        else:
            raise AttributeError("{} is not a valid train_mode".format(self.train_mode))
        
        
        return self

    def _create_model(self, input_dim, n_outputs=1, n_channels=1, name=None, **model_kwargs):
        tasks = self.map_dic.values()
        
        model = HardSharingTorchCombinator(n_features=input_dim,
                                      n_outputs=n_outputs,
                                      n_channels=n_channels,
                                      tasks=tasks,
                                      common_module=self.common_module,
                                      **model_kwargs)
        # model = HardSharingMTLNetwork(n_features=input_dim,
        #                               n_outputs=n_outputs,
        #                               n_channels=n_channels,
        #                               tasks=tasks,
        #                               common_module=self.common_module,
        #                               **model_kwargs)        
        return model
    
    def generate_tensors(self, X, t, y):
        # if t.ndim == 1:
        #     t_2d = t[:, None]
        # else:
        #     t_2d = t
        X, t, y = map(torch.tensor, (X, t, y))
        # t = t.int()
        return X, t, y
    
    def plot_history(self, val=True, ax=None, include_metrics=False, **kwargs):
        if ax is None:
            fig = plt.figure(**kwargs)
            ax = plt.gca()
        x = range(len(self.history['loss']))
        # loss
        y = self.history['loss']        
        ax.plot(x, y, label='loss')
        if self.val:
            y = self.history['val_loss']        
            ax.plot(x, y, label='val_loss')
        
        # metrics
        if include_metrics:
            for m in self.metrics:
                y = self.history[m]
                ax.plot(x, y, label=m)
                if self.val:
                    y = self.history['val_{}'.format(m)]
                    ax.plot(x, y, label='val_{}'.format(m))

        plt.legend()

        return ax
        


    def score(self, X, y, sample_weight=None):
        pass

class HardSharingMTLPytorchClassifier(HardSharingMTLPytorchModel):
    """Multi-layer Perceptron classifier using Keras.
    """
    @kwargs_decorator(
        {"weight_decay": 0,
        "min_delta": 1e-4,
        "random_state": 42,
        "metrics": [],
        "train_mode": "numpy",
        })
    def __init__(self,
                loss_fun: str='cross_entropy',
                common_module=None,
                specific_modules: dict=None,
                epochs: int=100,
                learning_rate: float=0.1,
                batch_size: int=128,
                verbose: int=1,
                weight_decay=0.1,
                val_size = 0.2,
                patience = 20,
                **kwargs):
        # kwargs = {**kwargs, **{'weight_decay': weight_decay}}
        self.weight_decay = weight_decay
        kwargs = {**kwargs, **{"val_size": val_size, "patience": patience}}
        super(HardSharingMTLPytorchClassifier, self).__init__(common_module=common_module,
                                                    specific_modules=specific_modules,
                                                    loss_fun=loss_fun,
                                                    epochs=epochs,
                                                    learning_rate=learning_rate,
                                                    batch_size=batch_size,
                                                    verbose=verbose,
                                                    **kwargs)
        
    
    
    def fit(self, X, y, task_info=-1, verbose=False, X_val=None, y_val=None, X_test=None, y_test=None, **kwargs):
        y_int = y.astype(int)
        
        super(HardSharingMTLPytorchClassifier, self).fit(X, y, task_info, verbose, X_val, y_val, X_test, y_test, **kwargs)
    
    def predict(self, X, task_info=-1, **kwargs):
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
        
        outputs = super(HardSharingMTLPytorchClassifier, self).predict_(X, task_info, **kwargs)
        
        _, pred = torch.max(outputs, 1)

        return pred.numpy()

    def predict_proba(self, X, task_info=-1, **kwargs):
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
        
        outputs = super(HardSharingMTLPytorchClassifier, self).predict_(X, task_info, **kwargs)

        proba = outputs.detach().numpy()

        ic(proba.shape)

        return proba

    def score(self, X, y, sample_weight=None):
        pass

    def generate_tensors(self, X, t, y):
        X, t, y = map(torch.tensor, (X, t, y))
        y = y.long()
        # t = t.int()
        return X, t, y


    def create_model(self, X_data, X_task, y, task_info=-1, name=None):
        self.task_info = task_info
        task_col = self.task_info
        n, m = X_data.shape[:2]
        input_dim = m

        unique = np.unique(y)
        n_outputs = len(unique)
        ic(n_outputs)
        if self.new_shape is None:
            n_channels = 1
        else:
            n_channels = self.new_shape[0]

        self.model = self._create_model(input_dim, n_outputs=n_outputs, n_channels=n_channels, name=name)
        return self.model


class HardSharingMTLPytorchRegressor(HardSharingMTLPytorchModel):
    """Multi-layer Perceptron Regressor using Keras.
    """
    @kwargs_decorator(
        {"weight_decay": 0,
        "min_delta": 1e-4,
        "random_state": 42,
        "metrics": [],
        "train_mode": "numpy",
        })
    def __init__(self,
                loss_fun: str='mse',
                common_module=None,
                specific_modules: dict=None,
                epochs: int=100,
                learning_rate: float=0.1,
                batch_size: int=128,
                verbose: int=1,
                weight_decay=0.1,
                val_size = 0.2,
                patience = 20,
                **kwargs):
        # kwargs = {**kwargs, **{'weight_decay': weight_decay}}
        self.weight_decay = weight_decay
        kwargs = {**kwargs, **{"val_size": val_size, "patience": patience}}
        super(HardSharingMTLPytorchRegressor, self).__init__(common_module=common_module,
                                                    specific_modules=specific_modules,
                                                    loss_fun=loss_fun,
                                                    epochs=epochs,
                                                    learning_rate=learning_rate,
                                                    batch_size=batch_size,
                                                    verbose=verbose,
                                                    **kwargs)

    def fit(self, X, y, task_info=-1, verbose=False, X_val=None, y_val=None, X_test=None, y_test=None, **kwargs):
        if y.ndim == 1:
            y_2d = y[:, None]
        else:
            y_2d = y
        
        super(HardSharingMTLPytorchRegressor, self).fit(X, y_2d, task_info, verbose, X_val, y_val, X_test, y_test, **kwargs)

    def create_model(self, X_data, X_task, y, task_info=-1, name=None):
        self.task_info = task_info
        task_col = self.task_info
        n, m = X_data.shape[:2]
        input_dim = m
        unique = np.unique(X_task)
        n_outputs = len(unique)
        self.model = self._create_model(input_dim, n_outputs=n_outputs, name=name)
        return self.model

                        