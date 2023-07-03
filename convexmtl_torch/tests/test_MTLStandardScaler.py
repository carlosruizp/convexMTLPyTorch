import unittest
from convexmtl_torch.data.DataLoader import DataLoader
from convexmtl_torch.preprocessing import MTLStandardScaler
import numpy as np

from icecream import ic

class TestStandardScaler(unittest.TestCase):

    def test_school_dataset(self):
        dataloader = DataLoader('my_data')
        df_data, df_target, _,  _,  task_info = dataloader.load_dataset(dataset_name='school')
        ic(df_data)
        task_col_ini = df_data[:,-1]
        mean_cols = np.mean(df_data[:, :-1], axis=0)
        std_cols = np.std(df_data[:, :-1], axis=0)

        scaler = MTLStandardScaler()
        df_data = scaler.fit_transform(df_data)
        mean_cols = np.mean(df_data[:, :-1], axis=0)
        for m in mean_cols:
            self.assertAlmostEqual(m, 0)

        std_cols = np.std(df_data[:, :-1], axis=0)
        for m in std_cols:
            self.assertAlmostEqual(m, 1)

        task_col = df_data[:,-1]
        self.assertListEqual(list(task_col), list(task_col_ini))


if __name__ == '__main__':
    unittest.main()