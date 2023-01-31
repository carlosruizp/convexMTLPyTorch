import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from pytorch_lightning import LightningModule

import numpy as np

from icecream import ic

from convexmtl_torch.model.utils import *

from typing import Union


class ConvexTorchCombinator(LightningModule):

    @kwargs_decorator(
        {"lambda_trainable": True,
         "loss_fun": "mse"})
    def __init__(self,
                 n_features: int,
                 n_output: int = 1,
                 n_channel: int = 1,
                 tasks: Union[list, np.ndarray] = None,
                 lamb: float=0.5,
                 common_module=None,
                 specific_modules: dict=None,
                 specific_lambda = False,
                 **kwargs):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ConvexTorchCombinator, self).__init__()
        self.lambda_trainable = kwargs["lambda_trainable"]
        self.loss_fun = kwargs["loss_fun"]
        assert (0 <= lamb and lamb <= 1)
        eps = 1e-4 # for computational purposes
        invsigmoid_lamb = -np.log((1 / (lamb + eps)) - 1 + eps)
        self.specific_lambda = specific_lambda
        if self.specific_lambda:
            self.lamb_dic = {}
            for t in tasks:
                self.lamb_dic[t] = torch.nn.Parameter(
                torch.tensor([invsigmoid_lamb], dtype=torch.double), requires_grad=self.lambda_trainable)
        else:
            self.lamb = torch.nn.Parameter(
                torch.tensor([invsigmoid_lamb], dtype=torch.double), requires_grad=self.lambda_trainable)
        self.specific_modules_ = {}
        if tasks is None:
            if common_module is None:
                common_module = NeuralNetwork
            self.common_module_ = common_module(n_features=n_features, n_output=n_output, n_channels=n_channel)
            for t, module in specific_modules.items():
                self.specific_modules_[t] = module(n_features=n_features, n_output=n_output, n_channels=n_channel)
                # self.add_module('Module Task {}'.format(t), self.specific_modules_[t])
        else:
            if common_module is None:
                common_module = NeuralNetwork
            self.common_module_ = common_module(n_features=n_features, n_output=n_output, n_channels=n_channel)
            for t in tasks:
                self.specific_modules_[t] = common_module(n_features=n_features, n_output=n_output, n_channels=n_channel)
                # if unique.dtype == 'float':
                #     tasks = ['{:.0f}'.format(u) for u in unique]
                # else:       
                #     tasks = unique.astype(str)
                # self.add_module('task{}_module'.format(t), self.specific_modules_[t])
        # self.double()

    def forward(self, x_data, x_task):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # x_data = x[:, :-1]
        # x_task = x[:, -1]
        pred = self.common_module_(x_data)
        # pred = torch.zeros(pred_common_.shape, requires_grad=True).double()
        for k, module in self.specific_modules_.items():
            k_idx = (x_task==k).flatten().bool()
            # k_idx = k_idx.repeat((1, x_data.shape[1]))
            # k_idx_int = np.argwhere(k_idx)
            # ic(k_idx_int)
            if not self.specific_lambda:
                lamb = self.lamb
            else:
                lamb = self.lamb_dic[k]
            # ic(torch.index_select(x_data, 0, torch.tensor(k_idx_int)))
            pred_specific = module(x_data[k_idx]).float()
            pred[k_idx] = torch.sigmoid(lamb).float() * pred[k_idx] \
                          + (1 - torch.sigmoid(lamb).float()) * pred_specific

        return pred

    def _get_loss_fun(self, **loss_kwargs):
        switcher = {
            'mse': nn.MSELoss,
            'l1': nn.L1Loss,
            'cross_entropy': nn.CrossEntropyLoss
        }
        return switcher.get(self.loss_fun, "Invalid Loss Function.")(**loss_kwargs)
    
    def configure_optimizers(self):
        param_list = torch.nn.ParameterList(
                      [p for n, p in self.named_parameters() if p.requires_grad and n != "lamb"])
        if hasattr(self, 'specific_modules_'):
            for name, module in self.specific_modules_.items():
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
        return loss




    
   
