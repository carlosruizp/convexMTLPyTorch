import torch 
from icecream import ic
from copy import deepcopy

from torch import nn
import torch.nn.functional as F
from torch import optim
from pytorch_lightning import LightningModule
from functools import wraps
import numpy as np

def kwargs_decorator(dict_kwargs):
    def wrapper(f):
        @wraps(f)
        def inner_wrapper(*args, **kwargs):
            new_kwargs = {**dict_kwargs, **kwargs}
            return f(*args, **new_kwargs)

        return inner_wrapper

    return wrapper



class SpatialTransformerNet(LightningModule):
    # L1 ImgIn shape=(?, 28, 28, 1)
    # Conv -> (?, 28, 28, 10)
    def __init__(self, **kwargs):
        super(SpatialTransformerNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # self.double()

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x # F.log_softmax(x, dim=1)


class ConvNet(LightningModule):
    # L1 ImgIn shape=(?, 28, 28, 1)
    # Conv -> (?, 28, 28, 10)
    def __init__(self, n_channels=1, **kwargs):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # self.double()

    def forward(self, x):

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x # F.log_softmax(x, dim=1)



class NeuralNetwork(LightningModule):
    def __init__(self, n_features, n_hidden=64, n_output=1, loss_fun='mse', **kwargs):
        super(NeuralNetwork, self).__init__()
        self.loss_fun = loss_fun
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )
        # # self.double()

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def _get_loss_fun(self, **loss_kwargs):
        switcher = {
            'mse': nn.MSELoss,
            'l1': nn.L1Loss,
            'cross_entropy': nn.CrossEntropyLoss
        }
        return switcher.get(self.loss_fun, "Invalid Loss Function.")(**loss_kwargs)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())

    def training_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        logits = self(x)
        loss_fun = self._get_loss_fun()
        loss = loss_fun(logits, y)
        return loss


class TaskNeuralNetwork(LightningModule):
    def __init__(self, n_features, n_hidden=64, n_output=1):
        super(TaskNeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )
        # self.double()

    def forward(self, x, t):
        logits = self.linear_relu_stack(x)
        return logits



class SpatialTransformerNetFeatLearn(LightningModule):
    # L1 ImgIn shape=(?, 28, 28, 1)
    # Conv -> (?, 28, 28, 10)
    def __init__(self, **kwargs):
        super(SpatialTransformerNetFeatLearn, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # self.double()

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x # F.log_softmax(x, dim=1)


class ConvNetFeatLearn(LightningModule):
    # L1 ImgIn shape=(?, 28, 28, 1)
    # Conv -> (?, 28, 28, 10)
    def __init__(self, n_channels=1, n_last_hidden=10, **kwargs):
        super(ConvNetFeatLearn, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, n_last_hidden)
        # self.fc2 = nn.Linear(50, 10)

        # self.double()

    def forward(self, x):

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        return x # F.log_softmax(x, dim=1)


class NeuralNetworkFeatLearn(LightningModule):
    def __init__(self, n_features, n_hidden=64, n_last_hidden=64, loss_fun='mse', **kwargs):
        super(NeuralNetworkFeatLearn, self).__init__()
        self.loss_fun = loss_fun
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_last_hidden),
            nn.ReLU(),
        )
        # self.double()

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def _get_loss_fun(self, **loss_kwargs):
        switcher = {
            'mse': nn.MSELoss,
            'l1': nn.L1Loss,
            'cross_entropy': nn.CrossEntropyLoss
        }
        return switcher.get(self.loss_fun, "Invalid Loss Function.")(**loss_kwargs)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())

    def training_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        logits = self(x)
        loss_fun = self._get_loss_fun()
        loss = loss_fun(logits, y)
        return loss





class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                # factor=self.factor,
                # min_lr=self.min_lr,
                # verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss, epoch, model, optimizer, criterion):
        if self.best_loss == None:
            self.best_loss = val_loss
        else:
            if self.best_loss - val_loss > self.min_delta:
                self.best_loss = val_loss
                # reset counter if validation loss improves
                self.counter = 0
                # save best model
                # print(f"\nBest validation loss: {self.best_loss}")
                # print(f"\nSaving best model for epoch: {epoch+1}\n")
                self.best_model_state_dict = deepcopy(model.state_dict())
                self.best_optimizer_state_dict = deepcopy(optimizer.state_dict())
                # torch.save({
                #     'epoch': epoch+1,
                #     'model_state_dict': model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'loss': criterion,
                #     }, 'outputs/best_model.pth')
            else:
                self.counter += 1
                # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
                if self.counter >= self.patience:
                    if not hasattr(self, 'best_model_state_dict'):
                        self.best_model_state_dict = deepcopy(model.state_dict())
                        self.best_optimizer_state_dict = deepcopy(optimizer.state_dict())
                    # print('INFO: Early stopping')
                    self.early_stop = True


# lt.style.use('ggplot')
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'outputs/best_model.pth')
            

def invsigmoid(s, eps=1e-7):
    return -np.log((1 / (s + eps)) - 1 + eps)