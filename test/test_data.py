"""
Some basic end-to-end testing to assert that things are not breaking after refactoring/optimization attempts
"""

import unittest
import random
from os.path import dirname, abspath, join

import numpy as np
import torch

from src.data import FloorplanGraphDataset

BASE_PATH = abspath(dirname(__file__))
SEED = 10

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.set_printoptions(profile="full")


class TestFloorplanGraphDataset(unittest.TestCase):
    data = np.load(join(BASE_PATH, 'resources/housegan_clean_data_1024_samples.npy'), allow_pickle=True)
    dataset = FloorplanGraphDataset(data, augment=True)

    def test_getitem(self):
        indices = [0, 7]

        for idx in indices:
            t = torch.load(join(BASE_PATH, f'resources/dataset_{idx}.pt'))
            d = self.dataset[idx]

            self.assertTrue(torch.allclose(d[0], t[0]))
            self.assertTrue(torch.allclose(d[1], t[1]))
            self.assertTrue(torch.allclose(d[2], t[2]))


if __name__ == "__main__":
    unittest.main()
