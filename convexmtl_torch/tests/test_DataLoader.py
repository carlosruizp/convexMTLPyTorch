import unittest
from convexmtl_torch.data.DataLoader import DataLoader
import numpy as np

from icecream import ic

class TestDataLoader(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    

    # def test_variations_mnist_dataset(self):
    #     dataloader = DataLoader('my_data')
    #     df_data, df_target, _, _, task_info = dataloader.load_dataset(dataset_name='variations_mnist')
    #     ic(df_data)
    #     self.assertEqual(df_data.shape, (70000, 785))
    #     self.assertEqual(df_target.shape, (70000,))
    #     uniq = np.unique(df_data[:, task_info])
    #     self.assertEqual(len(uniq), 3)

    # def test_rotated_mnist_dataset(self):
    #     dataloader = DataLoader('my_data')
    #     df_data, df_target, _, _, task_info = dataloader.load_dataset(dataset_name='rotated_mnist')
    #     ic(df_data)
    #     self.assertEqual(df_data.shape, (70000, 785))
    #     self.assertEqual(df_target.shape, (70000,))
    #     uniq = np.unique(df_data[:, task_info])
    #     self.assertEqual(len(uniq), 6)

    # def test_variations_fashionmnist_dataset(self):
    #     dataloader = DataLoader('my_data')
    #     df_data, df_target, _, _, task_info = dataloader.load_dataset(dataset_name='variations_fashionmnist')
    #     ic(df_data)
    #     self.assertEqual(df_data.shape, (70000, 785))
    #     self.assertEqual(df_target.shape, (70000,))
    #     uniq = np.unique(df_data[:, task_info])
    #     self.assertEqual(len(uniq), 3)

    # def test_rotated_fashionmnist_dataset(self):
    #     dataloader = DataLoader('my_data')
    #     df_data, df_target, _, _, task_info = dataloader.load_dataset(dataset_name='rotated_fashionmnist')
    #     ic(df_data)
    #     self.assertEqual(df_data.shape, (70000, 785))
    #     self.assertEqual(df_target.shape, (70000,))
    #     uniq = np.unique(df_data[:, task_info])
    #     self.assertEqual(len(uniq), 6)

    # def test_school_mnist_dataset(self):
    #     dataloader = DataLoader('my_data')
    #     df_data, df_target, _,  _,  task_info = dataloader.load_dataset(dataset_name='school')
    #     ic(df_data)
    #     self.assertEqual(df_data.shape, (15362, 28))
    #     self.assertEqual(df_target.shape, (15362,1))
    #     uniq = np.unique(df_data[:, task_info])
    #     ic(len(uniq))
    #     self.assertEqual(len(uniq), 139)

    # def test_school_mnist_dataset(self):
    #     dataloader = DataLoader('my_data')
    #     df_data, df_target, _,  _,  task_info = dataloader.load_dataset(dataset_name='school')
    #     ic(df_data)
    #     self.assertEqual(df_data.shape, (15362, 28))
    #     self.assertEqual(df_target.shape, (15362,1))
    #     uniq = np.unique(df_data[:, task_info])
    #     ic(len(uniq))
    #     self.assertEqual(len(uniq), 139)

    # def test_sarcos_dataset(self):
    #     dataloader = DataLoader('my_data')
    #     df_data, df_target, cv_out, _, task_info = dataloader.load_dataset(dataset_name='sarcos')
    #     ic(df_data)
    #     self.assertEqual(df_data.shape, (342531, 22))
    #     self.assertEqual(df_target.shape, (342531,1))
    #     uniq = np.unique(df_data[:, -1])
    #     ic(len(uniq))
    #     self.assertEqual(len(uniq), 7)
    
    # def test_computer_dataset(self):
    #     dataloader = DataLoader('my_data')
    #     df_data, df_target, _, _, task_info = dataloader.load_dataset(dataset_name='computer')
    #     ic(df_data)
    #     self.assertEqual(df_data.shape, (3800, 14))
    #     self.assertEqual(df_target.shape, (3800,1))
    #     uniq = np.unique(df_data[:, -1])
    #     ic(len(uniq))
    #     self.assertEqual(len(uniq), 190)

    # def test_parkinson_dataset(self):
    #     dataloader = DataLoader('my_data')
    #     df_data, df_target, _, _, task_info = dataloader.load_dataset(dataset_name='parkinson')
    #     ic(df_data)
    #     self.assertEqual(df_data.shape, (5875, 20))
    #     self.assertEqual(df_target.shape, (5875,1))
    #     uniq = np.unique(df_data[:, -1])
    #     ic(len(uniq))
    #     self.assertEqual(len(uniq), 42)

    def test_ClustersAReg_dataset(self):
        dataloader = DataLoader('my_data')
        m_r = 10
        T = 6
        df_data, df_target, _, _, task_info = dataloader.load_dataset(dataset_name='clustersA-reg', synthetic=True, n_samples_per_task=m_r)
        self.assertEqual(df_data.shape, (T * m_r, 2))
        self.assertEqual(df_target.shape, (T * m_r,1))
        uniq = np.unique(df_data[:, -1])
        ic(len(uniq))
        self.assertEqual(len(uniq), T)



        
    def test_invalid_dataset(self):
        with self.assertRaises(AttributeError):
            dataloader = DataLoader('my_data')
            df_data, df_target, _, _ = dataloader.load_dataset(dataset_name='invalid')


if __name__ == '__main__':
    unittest.main()