
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





# class HardSharingMTLNetwork(LightningModule):
#     def __init__(self, n_features, n_hidden=64, n_output=1, **kwargs):
#         super(HardSharingMTLNetwork, self).__init__()
#         # self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(n_features, n_hidden),
#             nn.ReLU(),
#             nn.Linear(n_hidden, n_hidden),
#             nn.ReLU(),
#             nn.Linear(n_hidden, n_hidden),
#             nn.ReLU(),
#         )
#         self.n_output = n_output
#         self.output_layers = {i: nn.Linear(n_hidden, 1).double() for i in range(n_output)}
#         # self.double()

#     def forward(self, x, t):
#         feat = self.linear_relu_stack(x)
#         # logits = torch.zeros((x.shape[0], 1), requires_grad=True)
#         logits = 0 * self.output_layers[0](feat)
#         for k, module in self.output_layers.items():
#             k_idx = (t==k).flatten().bool()
#             logits_task = module(feat[k_idx])
#             logits[k_idx] = logits_task
#         # ic(x)
#         # ic(t)
#         # ic(logits)
#         # exit()
#         return logits



# class HardSharingConvNet(LightningModule):
#     # L1 ImgIn shape=(?, 28, 28, 1)
#     # Conv -> (?, 28, 28, 10)
#     def __init__(self, n_channels=1, n_output=1, **kwargs):
#         super(HardSharingConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(n_channels, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         # self.fc2 = nn.Linear(50, n_output)
#         self.n_output = n_output
#         self.output_layers = {i: nn.Linear(50, 1).double() for i in range(n_output)}

#         # self.double()

#     def forward(self, x, t):

#         # Perform the usual forward pass
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)

#         logits = 0 * self.output_layers[0](x)
#         for k, module in self.output_layers.items():
#             k_idx = (t==k)
#             logits_task = module(x[k_idx])
#             logits[k_idx] = logits_task


#         return logits # F.log_softmax(x, dim=1)



class HardSharingTorchCombinator(LightningModule):

    @kwargs_decorator(
        {"lr": 1e-3,
         "lambda_lr": 1e-3,
         "weight_decay": 1e-2,
         "n_hidden_common": 16,
         })
    def __init__(self,
                 n_features: int,
                 tasks: Union[list, np.ndarray],
                 n_output: int = 1,
                 n_channel: int = 1,
                 n_last_hidden: int = 64,
                 common_module=None,
                 loss_fun='mse',
                 **kwargs):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super().__init__()
        self.loss_fun = loss_fun

        self.n_output = n_output
        ic(n_output)
        self.output_layers = {r: nn.Linear(n_last_hidden, n_output).float() for r in tasks}

        net_common_kwargs = {}
        if common_module is None or common_module == NeuralNetwork:
            net_common_kwargs['n_hidden_common'] = kwargs['n_hidden_common']
       
        if tasks is None:
            if common_module is None:
                common_module = NeuralNetworkFeatLearn
            self.common_module_ = common_module(n_features=n_features, n_channels=n_channel, n_last_hidden=n_last_hidden, **net_common_kwargs)
        else:
            if common_module is None:
                common_module = NeuralNetworkFeatLearn
            self.common_module_ = common_module(n_features=n_features, n_channels=n_channel, n_last_hidden=n_last_hidden, **net_common_kwargs)
        # # self.double()

    def forward(self, x_data, x_task):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        feat = self.common_module_(x_data).float()

        logits = torch.zeros((x_data.shape[0], self.n_output))
        for k, module in self.output_layers.items():
            k_idx = (x_task==k).flatten().bool()
            logits_task = module(feat[k_idx])
            logits[k_idx] = logits_task
        return logits

    def _get_loss_fun(self, **loss_kwargs):
        switcher = {
            'mse': nn.MSELoss,
            'l1': nn.L1Loss,
            'cross_entropy': nn.CrossEntropyLoss
        }
        return switcher.get(self.loss_fun, "Invalid Loss Function.")(**loss_kwargs)
    
    def configure_optimizers(self):
        param_list = torch.nn.ParameterList(
                      [p for n, p in self.named_parameters() if p.requires_grad])
        if hasattr(self, 'output_layers'):
            for name, module in self.output_layers.items():
                param_list.extend(
                    torch.nn.ParameterList(
                        [p for n, p in module.named_parameters() if p.requires_grad])
                )
        return optim.AdamW(param_list)

    def training_step(self, batch, batch_idx, **kwargs):
        x, t, y = batch
        logits = self(x, t)
        loss_fun = self._get_loss_fun()
        loss = loss_fun(logits, y)
        self.log('loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx, **kwargs):
        x, t, y = batch
        logits = self(x, t)
        loss_fun = self._get_loss_fun()
        loss = loss_fun(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

