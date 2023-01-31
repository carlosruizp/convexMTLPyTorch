import unittest
from mtlskl.data.DataLoader import DataLoader
import numpy as np

from icecream import ic

class TestDataLoader(unittest.TestCase):
    def test_school_dataset(self):
        data_loader = DataLoader(dataset_name='school')
        self.assertEqual(data_loader.data.shape, (15362, 28))
        self.assertEqual(data_loader.target.shape, (15362,))
        uniq = np.unique(data_loader.data[:, data_loader.task_info])
        self.assertEqual(len(uniq), 139)

    def test_computer_dataset(self):
        data_loader = DataLoader(dataset_name='computer')
        self.assertEqual(data_loader.data.shape, (3800, 14))
        self.assertEqual(data_loader.target.shape, (3800,))
        uniq = np.unique(data_loader.data[:, data_loader.task_info])
        self.assertEqual(len(uniq), 190)

    def test_compas_dataset(self):
        data_loader = DataLoader(dataset_name='compas')
        self.assertEqual(data_loader.data.shape, (3987, 8))
        self.assertEqual(data_loader.target.shape, (3987,))
        uniq = np.unique(data_loader.data[:, data_loader.task_info])
        self.assertEqual(len(uniq), 8)
        
    def test_invalid_dataset(self):
        with self.assertRaises(ValueError):
            data_loader = DataLoader(dataset_name='invalid')


if __name__ == '__main__':
    unittest.main()