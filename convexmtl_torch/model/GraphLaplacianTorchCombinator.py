
import torch
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

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer

from icecream import ic

from convexmtl_torch.model.utils import *

import matplotlib.pyplot as plt

from typing import Union

import seaborn as sns






class GraphLaplacianMTLNetwork(LightningModule):
    def __init__(self, n_features, n_hidden=64, n_outputs=1, **kwargs):
        super(GraphLaplacianMTLNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
        self.n_outputs = n_outputs
        self.output_layers = {i: nn.Linear(n_hidden, 1).double() for i in range(n_outputs)}
        # self.double()

    def forward(self, x, t):
        feat = self.linear_relu_stack(x)
        # logits = torch.zeros((x.shape[0], 1), requires_grad=True)
        logits = 0 * self.output_layers[0](feat)
        for k, module in self.output_layers.items():
            k_idx = (t==k)
            logits_task = module(feat[k_idx])
            logits[k_idx] = logits_task
        # ic(x)
        # ic(t)
        # ic(logits)
        # exit()
        return logits



class GraphLaplacianConvNet(LightningModule):
    # L1 ImgIn shape=(?, 28, 28, 1)
    # Conv -> (?, 28, 28, 10)
    def __init__(self, n_channels=1, n_outputs=1, **kwargs):
        super(GraphLaplacianConvNet, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, n_outputs)
        self.n_outputs = n_outputs
        self.output_layers = {i: nn.Linear(50, 1).double() for i in range(n_outputs)}

        # self.double()

    def forward(self, x, t):

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)

        logits = 0 * self.output_layers[0](x)
        for k, module in self.output_layers.items():
            k_idx = (t==k)
            logits_task = module(x[k_idx])
            logits[k_idx] = logits_task


        return logits # F.log_softmax(x, dim=1)



class GraphLaplacianTorchCombinator(LightningModule):

    @kwargs_decorator(
        {
        "loss_fun": "mse",
        "adj_trainable": True,
        "adj_lr": 1e-2,
        "log_matrix_freq": 50
        }
        )
    def __init__(self,
                 n_features: int,
                 tasks: Union[list, np.ndarray],
                 n_outputs: int = 1,
                 n_channel: int = 1,
                 n_last_hidden: int = 64,
                 common_module=None,
                 nu=1,
                 mu=1e-3,
                 **kwargs):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super().__init__()
        self.tasks = tasks
        self.nu = nu
        self.mu = mu

        self.loss_fun = kwargs["loss_fun"]
        self.adj_trainable = kwargs["adj_trainable"]
        self.adj_lr = kwargs["adj_lr"]
        self.log_matrix_freq = kwargs["log_matrix_freq"]

        self.n_outputs = n_outputs
        self.output_layers = {r: nn.Linear(n_last_hidden, n_outputs).float() for r in tasks}

        adj_ = {r: {s: np.log(1e-9) if r!=s else np.log(1) for s in tasks} for r in tasks}
        self.adj = {r: {s: None for s in tasks} for r in tasks}
        for r in tasks:
            for s in tasks:
                self.adj[r][s] = torch.nn.Parameter(torch.full([n_outputs, n_outputs], adj_[r][s], dtype=torch.float),
                                                     requires_grad=self.adj_trainable)
       
        if tasks is None:
            if common_module is None:
                common_module = NeuralNetworkFeatLearn
            self.common_module_ = common_module(n_features=n_features, n_channels=n_channel, n_last_hidden=n_last_hidden)
        else:
            if common_module is None:
                common_module = NeuralNetworkFeatLearn
            self.common_module_ = common_module(n_features=n_features, n_channels=n_channel, n_last_hidden=n_last_hidden)
        # # self.double()

    def forward(self, x_data, x_task):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # x_data = x[:, :-1]
        # x_task = x[:, -1]
        feat = self.common_module_(x_data).float()
        zero_torch = torch.tensor(0).float()
        # ic(zero_torch.dtype)
        # ic(self.output_layers[0](feat).float().dtype)
        logits = zero_torch * self.output_layers[0](feat).float()

        for k, module in self.output_layers.items():
            k_idx = (x_task==k).flatten().bool()
            logits_task = module(feat[k_idx])
            logits[k_idx] = logits_task
        # ic(x)
        # ic(t)
        # ic(logits)
        # exit()
        return logits

    def _get_loss_fun(self, **loss_kwargs):
        switcher = {
            'mse': nn.MSELoss,
            'l1': nn.L1Loss,
            'cross_entropy': nn.CrossEntropyLoss
        }
        return switcher.get(self.loss_fun, "Invalid Loss Function.")(**loss_kwargs)
    
    
    def get_params(self):
        params = []
        param_list = torch.nn.ParameterList(
                      [p for n, p in self.named_parameters() if p.requires_grad])
        params.append({'params': param_list})
        if hasattr(self, 'output_layers'):
            for name, module in self.output_layers.items():
                param_list = torch.nn.ParameterList(
                        [p for n, p in module.named_parameters() if p.requires_grad])
                params.append({'params': param_list})

        if self.adj_trainable:
            for r in self.tasks:
                for s in self.tasks:
                    param_list = torch.nn.ParameterList([self.adj[r][s]])
                    params.append({'params': param_list, 'lr': self.adj_lr})
        return params
    
    def configure_optimizers(self):
        params = self.get_params()
        return optim.AdamW(params)


    def training_step(self, batch, batch_idx, **kwargs):
        x, t, y = batch
        logits = self(x, t)
        loss_fun = self._get_loss_fun()
        loss = loss_fun(logits, y)

        lap_reg = 0
        entropy = 0

        self.entropy = {}
        for r, module_r in self.output_layers.items():
            w_r = module_r.weight
            sum_r = torch.sum(torch.tensor([torch.exp(self.adj[r][s]) for s in self.tasks]))
            entropy_r = 0# torch.sum(-softmax_r * torch.log(softmax_r))
            entropy_r_dic = {}
            for j, (s, module_s) in enumerate(self.output_layers.items()):
                A_rs = torch.exp(self.adj[r][s])/sum_r
                # ic(A_rs)
                self.log('A_({},{})'.format(r, s), A_rs, on_step=False, on_epoch=True)
                w_s = module_s.weight
                # ic(s)
                prod_rr = w_r @ w_r.T 
                prod_ss = w_s @ w_s.T 
                prod_rs = w_r @ w_s.T 
                dist_rs = (prod_rr + prod_ss - 2 * prod_rs)
                self.log('dist_({},{})'.format(r, s), dist_rs, on_step=False, on_epoch=True)
                self.log('A_rs * dist_({},{})'.format(r, s), A_rs * dist_rs, on_step=False, on_epoch=True)
                # ic(self.adj[r][s])
                entropy_r += (-A_rs * torch.log(A_rs))
                entropy_r_dic[s] = (-A_rs * torch.log(A_rs))
                lap_reg += torch.sum(A_rs * dist_rs)
                # ic(lap_reg)
            self.log('entropy {}'.format(r), entropy_r, on_step=False, on_epoch=True)
            entropy += entropy_r
            self.entropy[r] = entropy_r_dic

        T = len(self.tasks)
        self.log('lap_reg', lap_reg, on_step=False, on_epoch=True)
        self.log('entropy', entropy, on_step=False, on_epoch=True)
        self.log('loss', loss, on_step=False, on_epoch=True)
        self.log('nu*lap_reg-mu*entr', (self.nu * lap_reg)/T**2 - (self.mu * entropy)/T**2, on_step=False, on_epoch=True)

        self.log('loss+nu*lap_reg-mu*entr', loss + (self.nu * lap_reg)/T**2 - (self.mu * entropy)/T**2, on_step=False, on_epoch=True)
       
        return loss + (self.nu * lap_reg)/T**2 - (self.mu * entropy)/T**2
    
    def training_epoch_end(self, training_step_outputs):
        self._log_adjMatrix()
        return None
    
    def get_adjMatrix(self):
        adjMatrix_ = np.zeros((len(self.tasks), len(self.tasks)))
        for i, r in enumerate(self.tasks):
            sum_r = torch.sum(torch.tensor([torch.exp(self.adj[r][s]) for s in self.tasks]))
            for j, s in enumerate(self.tasks):
                adjMatrix_[i, j] = (torch.exp(self.adj[r][s]) / sum_r).detach().numpy()
        # ic(adjMatrix_)
        return adjMatrix_
    
    def get_fig_adjMatrix(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = plt.gca()
        adjMatrix = self.get_adjMatrix()    
        sns.heatmap(adjMatrix, xticklabels=self.tasks, yticklabels=self.tasks, vmin=0, vmax=1, ax=ax)
        return fig

    def _log_adjMatrix(self):
        if self.current_epoch % self.log_matrix_freq == 0:
            ic(self.get_adjMatrix())
            ic(self.entropy)
            tensorboard = self.logger.experiment
            fig = self.get_fig_adjMatrix()
            ic(self.current_epoch)
            tensorboard.add_figure(tag='adj_matrix', 
                                   figure=fig, global_step=self.current_epoch)
    

                

