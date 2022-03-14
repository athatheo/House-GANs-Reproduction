import random

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np

# bounding box limits
MIN_H = 0.03
MIN_W = 0.03


def floorplan_collate_fn():
    raise NotImplementedError


def create_loaders(path, train_batch_size, test_batch_size, loader_threads, n_rooms=(10, 12)):
    data = np.load(path, allow_pickle=True)

    # filter the data
    train_data = []
    test_data = []
    for floorplan in data:
        rooms_types = floorplan[0]
        rooms_bbs = floorplan[1]  # bounding boxes

        # discard malformed samples
        if not type or any(i == 0 for i in rooms_types) or any(i is None for i in rooms_bbs):
            continue

        # discard small rooms
        types_filtered = []
        bbs_filtered = []
        for t, bb in zip(rooms_types, rooms_bbs):
            if bb[2] - bb[0] > MIN_H and bb[3] - bb[1] > MIN_W:
                types_filtered.append(t)
                bbs_filtered.append(bb)

        # trainset has samples outside the target range for number of rooms, and testset only those inside (?)
        # also cap the number of eval samples to 5k for some reason
        # TODO remove this stupid limit placement here after confirming that all is ok without it
        if n_rooms[0] <= len(rooms_types) <= n_rooms[1] and len(train_data) <= 5000:
            test_data.append([types_filtered, bbs_filtered])
        else:
            train_data.append([types_filtered, bbs_filtered])

    # create datasets
    train_dataset = FloorplanGraphDataset(train_data, train=True)
    test_dataset = FloorplanGraphDataset(test_data, train=False)

    # create loaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=loader_threads,
                              collate_fn=floorplan_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=loader_threads,
                             collate_fn=floorplan_collate_fn)

    return train_loader, test_loader


class FloorplanGraphDataset(Dataset):

    def __init__(self, data, train):
        self.data = data
        self.train = train
        self.transform = transforms.Normalize(mean=[0.5], std=[0.5])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError


























