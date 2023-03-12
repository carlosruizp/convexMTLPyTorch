import unittest
from convexmtl_torch.data.DataLoader import DataLoader
import numpy as np

from icecream import ic

class TestDataLoader(unittest.TestCase):
    def test_variations_mnist_dataset(self):
        dataloader = DataLoader('my_data')
        df_data, df_target, _, _, task_info = dataloader.load_dataset(dataset_name='variations_mnist')
        ic(df_data)
        self.assertEqual(df_data.shape, (70000, 785))
        self.assertEqual(df_target.shape, (70000,))
        uniq = np.unique(df_data[:, -1])
        self.assertEqual(len(uniq), 3)

    def test_variations_mnist_dataset(self):
        dataloader = DataLoader('my_data')
        df_data, df_target, _,  _,  task_info = dataloader.load_dataset(dataset_name='school')
        ic(df_data)
        self.assertEqual(df_data.shape, (15362, 28))
        self.assertEqual(df_target.shape, (15362,1))
        uniq = np.unique(df_data[:, -1])
        ic(len(uniq))
        self.assertEqual(len(uniq), 139)
        
    def test_invalid_dataset(self):
        with self.assertRaises(AttributeError):
            dataloader = DataLoader('my_data')
            df_data, df_target, _, _ = dataloader.load_dataset(dataset_name='invalid')


if __name__ == '__main__':
    unittest.main()