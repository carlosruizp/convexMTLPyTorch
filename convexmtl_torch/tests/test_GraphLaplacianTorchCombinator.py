import unittest
import torch
from torch.nn import MSELoss
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from convexmtl_torch.model.GraphLaplacianTorchCombinator import GraphLaplacianTorchCombinator
from convexmtl_torch.model.utils import NeuralNetwork
from pytorch_lightning import Trainer

from icecream import ic

from sklearn.model_selection import train_test_split

class TestGraphLaplacianTorchCombinator(unittest.TestCase):

    def setUp(self):
        self.epochs=50
        self.n_features = 10
        self.n_outputs = 1
        self.n_channel = 1
        self.tasks = [0, 1]
        self.n_last_hidden = 64
        self.batch_size = 32
        self.model = GraphLaplacianTorchCombinator(
            n_features=self.n_features,
            n_outputs=self.n_outputs,
            n_channel=self.n_channel,
            n_last_hidden=self.n_last_hidden,
            tasks=self.tasks
        )
        ic(self.model.get_adjMatrix())
        self.dummy_input = torch.randn(self.batch_size, self.n_features).float()
        self.dummy_task = torch.randint(len(self.tasks), size=(self.batch_size,)).float()
        ic(self.dummy_task)
        self.dummy_target = torch.randn(self.batch_size, self.n_outputs).float()

    def test_forward(self):
        output = self.model(self.dummy_input, self.dummy_task)
        self.assertEqual(output.shape, (self.batch_size, self.n_outputs))

    def test_configure_optimizers(self):
        optimizer = self.model.configure_optimizers()
        self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_training_step(self):
        loss = self.model.training_step((self.dummy_input, self.dummy_task, self.dummy_target), 0)
        self.assertIsInstance(loss, torch.Tensor)

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
        ic(self.model.get_adjMatrix())
        self.assertLessEqual(score, 1)
    

if __name__ == '__main__':
    unittest.main()