import unittest
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from convexmtl_torch.model.HardSharingTorchCombinator import HardSharingTorchCombinator
from convexmtl_torch.model.utils import NeuralNetworkFeatLearn

class TestHardSharingTorchCombinator(unittest.TestCase):

    def setUp(self):
        self.n_features = 10
        self.n_outputs = 3
        self.n_channel = 1
        self.n_last_hidden = 64
        self.batch_size = 32
        self.model = HardSharingTorchCombinator(
            n_features=self.n_features,
            n_outputs=self.n_outputs,
            n_channel=self.n_channel,
            n_last_hidden=self.n_last_hidden
        )
        self.dummy_input = torch.randn(self.batch_size, self.n_features).float()
        self.dummy_task = torch.randint(self.n_outputs, size=(self.batch_size,)).float()
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

if __name__ == '__main__':
    unittest.main()