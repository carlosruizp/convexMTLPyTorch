
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

from convexmtl_torch.model.ConvexTorchCombinator import ConvexTorchCombinator

from pytorch_lightning import Trainer

from icecream import ic

from convexmtl_torch.model.utils import *

import matplotlib.pyplot as plt


class ConvexMTLPytorchModel(BaseEstimator):
    """
    Abstract class from which ConvexMTLPytorchClassifier and ConvexMTLPytorchRegressor inherit.
    Attributes
    --------------
    - layers: list containing the number of neurons in each one of the common hidden layers 
    of the network. The input and output layers sizes are determined by the problem.
    - lr: the 
    """
    _opt_keys = {'weight_decay', 'lr'}
    _loss_keys = {}

    def __init__(self,
                 loss_fun: str,
                 common_module,
                 specific_modules: dict,
                 epochs: int,
                 lr: float,
                 batch_size: int,
                 n_hidden_common: int,
                 n_hidden_specific: int,
                 verbose: int,
                 lamb: float ,
                 lambda_lr: float,
                 lambda_trainable: bool,
                 specific_lambda: bool,
                 weight_decay: float,
                 val_size: float,
                 min_delta: float,
                 patience: int,
                 random_state: int,
                 early_stopping: bool,
                 lr_scheduler: bool,
                 metrics: list,
                 train_mode: str,
                 enable_progress_bar: True
                ):
                 
        super(ConvexMTLPytorchModel, self).__init__()
        self.common_module = common_module
        self.specific_modules = specific_modules
        self.loss_fun = loss_fun
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.n_hidden_common = n_hidden_common
        self.n_hidden_specific = n_hidden_specific
        self.verbose = verbose
        self.lamb = lamb
        self.lambda_lr = lambda_lr
        self.lambda_trainable = lambda_trainable
        self.specific_lambda = specific_lambda
        if not lambda_trainable and specific_lambda:
            raise ValueError("specific_lambda=True requires lambda_trainable=True")
        self.weight_decay = weight_decay
        self.val_size = val_size
        self.min_delta = min_delta
        self.patience = patience
        self.metrics = metrics
        self.random_state = random_state
        self.train_mode = train_mode
        self.early_stopping = early_stopping
        self.lr_scheduler = lr_scheduler
        self.enable_progress_bar = enable_progress_bar

    
    def predict_(self, X, task_info=-1, **kwargs):
        n, m = X.shape
        task_col = task_info
        self.unique, self.groups_idx = npi.group_by(X[:, task_col],
                                                    np.arange(n))
        X_data = np.delete(X, task_col, axis=1).astype(float)
        X_task = X[:, task_col]
        X_task_map = np.array([self.map_dic[x] for x in X_task])


        if 'new_shape' in kwargs and kwargs['new_shape'] is not None:
            # ic(X_data.shape)
            new_shape = [X_data.shape[0]] + list(kwargs['new_shape'])
            X_data = np.reshape(X_data, tuple(new_shape))
        elif hasattr(self, 'new_shape'):
            if self.new_shape is not None:
                new_shape = [n] + list(self.new_shape)
                X_data = np.reshape(X_data, tuple(new_shape))

        X_tensor_data, X_tensor_task = self.generate_tensors(X_data, X_task_map)
        # X_tensor_data = torch.tensor(X_data)
        # X_tensor_task = torch.tensor(X_task_map)
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

    def predict_common(self, X):
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
        X_data = X[:, :-1][:, None]
        X_tensor = torch.tensor(X_data)
        y_pred = self.model.common_module(X_tensor).detach().numpy()

        pred_flatten = y_pred.flatten()

        return pred_flatten

    def predict_task(self, X, task=None):
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
        assert task in self.model.specific_modules_.keys()
        X_data = X[:, :-1][:, None]
        X_tensor = torch.tensor(X_data)
        y_pred = self.model.specific_modules_[task](X_tensor).detach().numpy()

        pred_flatten = y_pred.flatten()

        return pred_flatten


    

    # def set_model(self, model):      
    #     self.model = model
    #     return

    # def set_model_fname(self, model_fname):      
    #     self.model_fname = model_fname
    #     return
    
    # def load_model(self, model_fname=None):      
    #     self.model = keras.models.load_model(model_fname)
    #     return self

    # def _load_model(self, model_fname=None):      
    #     self.model = keras.models.load_model(self.model_fname)
    #     return self.model

    # def save_model(self, model_fname):        
    #     self.model.save(model_fname)
    #     self.model_fname = model_fname
    #     return

    # def get_opt_(self, **opt_kwargs):
    #     params = self.model.get_params()        
    #     # ic(params)
    #     ic(self.lr, self.lambda_lr, self.weight_decay)
    #     opt = optim.AdamW(params, **opt_kwargs)#, lr=self.lr)
        
    #     return opt

    def get_opt(self):
        assert hasattr(self, 'model')
        return self.model.configure_optimizers()

    def _get_reg(self, **reg_kwargs):
        raise NotImplementedError()
    
    def _get_loss_fun(self, **loss_kwargs):
        switcher = {
            'mse': nn.MSELoss,
            'l1': nn.L1Loss,
            'cross_entropy': nn.CrossEntropyLoss
        }
        loss_kwargs = {}
        for k in self._loss_keys:
            if hasattr(self, k):
                loss_kwargs[k] = getattr(self, k)
        return switcher.get(self.loss_fun, "Invalid Loss Function.")(**loss_kwargs)

    def _get_metric(self, metric):
        scorer = get_scorer(metric)
        return scorer

    def plot_lambda_hist(self, **kwargs):
        plt.figure(figsize=(15, 10))
        if not self.lambda_trainable:
            return
        hist = self.lambda_history_
        l = len(hist)
        # plt.scatter(range(l), hist)
        plt.plot(range(l), hist)
    
    
    # @profile(output_file='train_lightning_convexmtl.prof', sort_by='cumulative', lines_to_print=20, strip_dirs=True)
    def _train_lightning(self, train_dl, valid_dl):
        trainer = Trainer(max_epochs=self.epochs, enable_progress_bar=self.enable_progress_bar)
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
        with trange(int(self.epochs), desc='Loss', position=0, leave=True, disable=not self.enable_progress_bar) as tri:
                for epoch in tri:
                    self.model.train()
                    if self.lambda_trainable:
                            z = self.model.lamb.detach().numpy()
                            z_sigmoid = 1/(1 + np.exp(-z))
                            self.lambda_history_.append(z_sigmoid)
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
                    # pred_train, y_train_ = map(lambda x: x[0].detach().numpy(),zip(*[(self.model(xb, tb), yb) for xb, tb, yb in train_dl]))
                    if self.val:
                        self.history['val_loss'].append(val_loss.detach().item())
                        # pred_val, y_val_ = list(map(lambda x: x[0].detach().numpy(), zip(*[(self.model(xb, tb), yb) for xb, tb, yb in valid_dl])))

                    
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

        # ic(self.model)
        # ic(self.model.common_module_)
        best_tr = np.inf
        best_val = np.inf
        old_val_loss = np.inf
        early_stopping_count = 0
        with trange(int(self.epochs), desc='Loss', position=0, leave=True, disable=not self.enable_progress_bar) as tri:
            for epoch in tri:
                self.model.train()
                self.log_lambda()
                n_train = X_train.shape[0]
                sample_idx = np.arange(n_train, dtype=int)                        
                sample_idx = shuffle(sample_idx, random_state=self.random_state)
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

                        # ic(self._early_stopping.early_stop)

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
                # pred_train, y_train_ = map(lambda x: x[0].detach().numpy(),zip(*[(self.model(xb, tb), yb) for xb, tb, yb in train_dl]))
                if self.val:
                    self.history['val_loss'].append(val_loss.detach().item())
                    # pred_val, y_val_ = list(map(lambda x: x[0].detach().numpy(), zip(*[(self.model(xb, tb), yb) for xb, tb, yb in valid_dl])))

                
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

            self.tot_epochs = epoch + 1


    def log_lambda(self):
        if self.lambda_trainable:
            if not self.specific_lambda:
                z = self.model.lamb.detach().numpy()
                z_sigmoid = 1/(1 + np.exp(-z))
                self.lambda_history_.append(z_sigmoid)
            else:
                for t, lamb_ in self.model.lamb_dic.items():
                    z = lamb_.detach().numpy()
                    z_sigmoid = 1/(1 + np.exp(-z))
                    self.lambda_history_[self.inv_map_dic[t]].append(z_sigmoid)
    
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
        self.inv_map_dic =  {v: k for k, v in self.map_dic.items()}
        X_task_map = np.array([self.map_dic[x] for x in X_task])



        #if not hasattr(self, 'model'):
        self.model = self.create_model(X_data, X_task, y)
        # if not hasattr(self, 'opt'):
        self.opt = self.get_opt()

        # self.reg_ = self._get_reg()
        self.loss_fun_ = self._get_loss_fun()


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
        X_train, t_train = self.generate_tensors(X_train, t_train)
        y_train = self.generate_tensors_target(y_train)
        # t_train = t_train.int()
        train_ds = TensorDataset(X_train, t_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        if self.val:
            X_val, t_val = self.generate_tensors(X_val, t_val)
            y_val = self.generate_tensors_target(y_val)
            valid_ds = TensorDataset(X_val, t_val, y_val)
            valid_dl = DataLoader(valid_ds, batch_size=self.batch_size * 2)

            if self.early_stopping:
                self._early_stopping = EarlyStopping(patience=self.patience,
                                                     min_delta=self.min_delta)

            # lr scheduler
            if self.lr_scheduler:
                self._lr_scheduler = LRScheduler(self.opt, patience=self.patience)

        else:
            valid_dl = None


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
        # lambda history
        if self.specific_lambda:
            # self.lambda_history_ = dict(zip(self.unique, [[]]*len(self.unique)))
            self.lambda_history_ = {}
            for t in self.unique:
                self.lambda_history_[t] = []
        else:
            self.lambda_history_ = []


        # ic(self.train_mode)
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

    def _create_model(self, input_dim, n_output=1, n_channel=1, name=None, **model_kwargs):
        tasks = list(self.map_dic.values())
        # ic(list(self.map_dic.items()))
        # ic(tasks)
        if self.n_hidden_specific is None:
            n_hidden_specific = self.n_hidden_common
        else:
            n_hidden_specific = self.n_hidden_specific
        kwargs = {
            "lambda_trainable": self.lambda_trainable,
            "specific_lambda": self.specific_lambda,
            "n_hidden_common": self.n_hidden_common,
            "n_hidden_specific": n_hidden_specific,
        }
        opt_kwargs = {}
        for k in self._opt_keys:
            if hasattr(self, k):
                opt_kwargs[k] = getattr(self, k)
        model_kwargs = {**model_kwargs, **opt_kwargs, **kwargs}
        model = ConvexTorchCombinator(n_features=input_dim,
                                      n_output=n_output,
                                      n_channel=n_channel,
                                      tasks=tasks,
                                      lamb=self.lamb,
                                      common_module=self.common_module,
                                      specific_modules=self.specific_modules,
                                      **model_kwargs)        
        return model
    
    def generate_tensors(self, *args):
        # if t.ndim == 1:
        #     t_2d = t[:, None]
        # else:
        #     t_2d = t
        tensor_args = map(lambda z: torch.tensor(z).float(), args)
        # t = t.int()
        return tensor_args
    
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

    def plot_lambda_history(self, val=True, ax=None, include_metrics=False, t=None, **kwargs):
        if ax is None:
            fig = plt.figure(**kwargs)
            ax = plt.gca()
        if not self.specific_lambda:
            x = range(self.tot_epochs)
            # loss
            y = self.lambda_history_        
            ax.plot(x, y, label=r'$\lambda$')
        elif t is not None:
            x = range(self.tot_epochs)                
            y = self.lambda_history_[t]      
            ax.plot(x, y, label=r'$\lambda$_{}'.format(t))
        else:
            x = range(self.tot_epochs)
            for t in self.unique:
                
                # loss
                y = self.lambda_history_[t]      
                ax.plot(x, y, label=r'$\lambda$_{}'.format(t))

        plt.legend()

        return ax
        


    def score(self, X, y, sample_weight=None):
        pass

    def get_lamb_history(self):
        if not self.lambda_trainable:
            raise AttributeError()
        return self.lambda_history_


class ConvexMTLPytorchClassifier(ConvexMTLPytorchModel):
    """Multi-layer Perceptron classifier using Keras.
    """
    def __init__(self,
                loss_fun: str='cross_entropy',
                common_module=None,
                specific_modules: dict=None,
                epochs: int=100,
                lr: float=1e-3,
                batch_size: int=128,
                n_hidden_common: int=16,
                n_hidden_specific: int=16,
                verbose: int=1,
                lamb: float = 0.5,
                weight_decay: float=0.01,
                val_size = 0.2,
                patience = 20,
                lambda_trainable = False,
                specific_lambda = False,
                lambda_lr = 0.001,
                min_delta = 1e-4,
                random_state = 42,
                metrics = [],
                train_mode='numpy',
                early_stopping=True,
                lr_scheduler = True,
                enable_progress_bar = True):
        super().__init__(loss_fun=loss_fun,
                         common_module=common_module,
                         specific_modules=specific_modules,
                         epochs=epochs,
                         lr=lr,
                         batch_size=batch_size,
                         n_hidden_common=n_hidden_common, 
                         n_hidden_specific=n_hidden_specific,
                         verbose=verbose,
                         lamb=lamb,
                         weight_decay=weight_decay,
                         val_size=val_size,
                         patience=patience,
                         lambda_trainable=lambda_trainable,
                         specific_lambda=specific_lambda,
                         lambda_lr=lambda_lr,
                         min_delta=min_delta,
                         random_state=random_state,
                         metrics=metrics,
                         train_mode=train_mode,
                         early_stopping=early_stopping,
                         lr_scheduler=lr_scheduler,
                         enable_progress_bar=enable_progress_bar)
        
    
    
    def fit(self, X, y, task_info=-1, verbose=False, X_val=None, y_val=None, X_test=None, y_test=None, **kwargs):
        y_flat = y.astype(int).flatten()

        # if y.ndim == 1:
        #     y_2d = y_int[:, None]
        # else:
        #     y_2d = y_int
        
        super(ConvexMTLPytorchClassifier, self).fit(X, y_flat, task_info, verbose, X_val, y_val, X_test, y_test, **kwargs)
    
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
        
        outputs = super(ConvexMTLPytorchClassifier, self).predict_(X, task_info, **kwargs)
        
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
        
        outputs = super(ConvexMTLPytorchClassifier, self).predict_(X, task_info, **kwargs)

        proba = outputs.detach().numpy()

        return proba

    def score(self, X, y, sample_weight=None):
        pass

    def generate_tensors_target(self, target):
        tensor_target = torch.tensor(target).long()
        return tensor_target

    def create_model(self, X_data, X_task, y, task_info=-1, name=None):
        self.task_info = task_info
        task_col = self.task_info
        n, m = X_data.shape[:2]
        input_dim = m

        unique = np.unique(y)
        n_outputs = len(unique)
        if not hasattr(self, 'new_shape') or self.new_shape is None:
            n_channel = 1
        else:
            n_channel = self.new_shape[0]

        self.model = self._create_model(input_dim, n_output=n_outputs, n_channel=n_channel, name=name)
        return self.model


class ConvexMTLPytorchRegressor(ConvexMTLPytorchModel):
    """Multi-layer Perceptron Regressor using Keras.
    """
    def __init__(self,
                 loss_fun: str='mse',
                 common_module=None,
                 specific_modules: dict=None,
                 epochs: int=100,
                 lr: float=1e-3,
                 batch_size: int=128,
                 n_hidden_common: int=16,
                 n_hidden_specific: int=16,
                 verbose: int=1,
                 lamb: float = 0.5,
                 weight_decay: float = 0.01,
                 val_size = 0.2,
                 patience = 20,
                 lambda_trainable = False,
                 specific_lambda = False,
                 lambda_lr = 0.001,
                 min_delta = 1e-4,
                 random_state = 42,
                 metrics = [],
                 train_mode='numpy',
                 early_stopping=True,
                 lr_scheduler = True,
                 enable_progress_bar = True):
        # kwargs = {**kwargs, **{'weight_decay': weight_decay}}
        super().__init__(loss_fun=loss_fun,
                         common_module=common_module,
                         specific_modules=specific_modules,
                         epochs=epochs,
                         lr=lr,
                         batch_size=batch_size,
                         n_hidden_common=n_hidden_common, 
                         n_hidden_specific=n_hidden_specific,
                         verbose=verbose,
                         lamb=lamb,
                         weight_decay=weight_decay,
                         val_size=val_size,
                         patience=patience,
                         lambda_trainable=lambda_trainable,
                         specific_lambda=specific_lambda,
                         lambda_lr=lambda_lr,
                         min_delta=min_delta,
                         random_state=random_state,
                         metrics=metrics,
                         train_mode=train_mode,
                         early_stopping=early_stopping,
                         lr_scheduler=lr_scheduler,
                         enable_progress_bar=enable_progress_bar)

    def fit(self, X, y, task_info=-1, verbose=False, X_val=None, y_val=None, X_test=None, y_test=None, **kwargs):
        if y.ndim == 1:
            y_2d = y[:, None]
        else:
            y_2d = y
        
        super(ConvexMTLPytorchRegressor, self).fit(X, y_2d, task_info, verbose, X_val, y_val, X_test, y_test, **kwargs)

    def create_model(self, X_data, X_task, y, task_info=-1, name=None):
        self.task_info = task_info
        task_col = self.task_info
        n, m = X_data.shape[:2]
        input_dim = m
        self.model = self._create_model(input_dim, n_output=1, name=name)
        return self.model
    
    def generate_tensors_target(self, target):
        tensor_target = torch.tensor(target).float()
        return tensor_target

                        