import unittest
import torch
from torch.nn import MSELoss
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from convexmtl_torch.model.ConvexTorchCombinator import ConvexTorchCombinator
from convexmtl_torch.model.utils import NeuralNetwork
from pytorch_lightning import Trainer

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from icecream import ic


class TestConvexTorchCombinator(unittest.TestCase):
    def setUp(self):
        self.n_features = 3
        self.n_output = 1
        self.n_channel = 1
        self.tasks = [0, 1]
        self.lamb = 0.5
        self.common_module = NeuralNetwork
        self.specific_modules = {0: NeuralNetwork, 1: NeuralNetwork}
        self.specific_lambda = True
        self.epochs = 10
        self.batch_size = 32

        self.model = ConvexTorchCombinator(self.n_features, self.tasks, self.n_output, self.n_channel, self.lamb, self.common_module, self.specific_modules, self.specific_lambda)
        ic(self.model.get_lamb())

    def test_forward(self):
        x_data = torch.randn(10, self.n_features).float()
        x_task = torch.randint(0, 2, (10,))
        output = self.model(x_data, x_task)
        self.assertEqual(output.shape, (10, self.n_output))

    def test_lambda_trainable(self):
        self.assertTrue(self.model.lambda_trainable)

    def test_loss_fun(self):
        self.assertEqual(self.model.loss_fun, "mse")

    def test_train(self):
        X_data = torch.randn(1000, self.n_features).double()
        y = X_data[:, 0] ** 2 + torch.sin(X_data[:, 1]) - 2 * X_data[:, 2]
        if y.ndim != 2:
            y_2d = y[:, None]
        else:
            y_2d = y
        X_task = torch.randint(0, 2, (1000, 1))

        X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(X_data, X_task, y_2d)

        X_train, t_train, y_train = map(torch.tensor, (X_train, t_train, y_train))
        X_test, t_test, y_test = map(torch.tensor, (X_test, t_test, y_test))
        X_train, t_train, y_train = map(lambda obj: obj.float(), [X_train, t_train, y_train])
        X_test, t_test, y_test = map(lambda obj: obj.float(), [X_test, t_test, y_test])
        train_ds = TensorDataset(X_train, t_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        
        trainer = Trainer(max_epochs=self.epochs)
        ic(self.model)
        trainer.fit(self.model, train_dl)

        pred = self.model(X_test, t_test)

        loss = MSELoss()

        score = loss(y_test, pred)
        ic(score)
        ic(self.model.get_lamb())

        self.assertLessEqual(score, 1)


if __name__ == '__main__':
    unittest.main()