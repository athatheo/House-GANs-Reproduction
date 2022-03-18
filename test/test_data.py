"""
Some basic end-to-end testing to assert that things are not breaking after refactoring/optimization attempts
"""

import unittest
import random
from os.path import dirname, abspath, join

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import FloorplanGraphDataset, collate


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

    def test_collation(self):
        mks_expected = torch.load(join(BASE_PATH, f'resources/mks1.pt'))
        nds_expected = torch.load(join(BASE_PATH, f'resources/nds1.pt'))
        eds_expected = torch.load(join(BASE_PATH, f'resources/eds1.pt'))
        nd_to_sample_expected = torch.load(join(BASE_PATH, f'resources/nd_to_sample1.pt'))
        ed_to_sample_expected = torch.load(join(BASE_PATH, f'resources/ed_to_sample1.pt'))

        loader = DataLoader(self.dataset, batch_size=5, shuffle=False, num_workers=1, collate_fn=collate)

        batch = next(iter(loader))
        mks, nds, eds, nd_to_sample, ed_to_sample = batch
        self.assertTrue(torch.allclose(mks, mks_expected))
        self.assertTrue(torch.allclose(nds, nds_expected))
        self.assertTrue(torch.allclose(eds, eds_expected))
        self.assertTrue(torch.allclose(nd_to_sample, nd_to_sample_expected))
        self.assertTrue(torch.allclose(ed_to_sample, ed_to_sample_expected))


if __name__ == "__main__":
    unittest.main()
