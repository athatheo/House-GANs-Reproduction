""" Data loaders
This file provides:
  1. A `create_loaders` function that creates two data loaders, one for the training set and one for the test set.
  2. A `FloorplanGraphDataset` class which extends torch's Dataset.

Raw data schema:
  - dataset is a python list of floorplans
  - each floorplan if a python list containing rooms_types, rooms_bounding_boxes, and 4 other elements (which we drop)
  - the rooms_types is a list of ints, each signifying the type of each room in the floorplan. The mapping is:
    ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "missing": 5,
    "closet": 6, "balcony": 7, "corridor": 8, "dining_room": 9, "laundry_room": 10}
  - the rooms_bounding_boxes is a list of numpy arrays. Each of these arrays has 4 integer elements.
    The first two correspond to the coordinates of the point to the upper left of the room's box, and the last two
    to the bottom right respectively. The coordinates' values are in the range [0, 256]
  - an example of floorplan is the following:
    ```
    [
        [6.0, 2.0, 4.0],
        [array([array([132,   6, 148,  65]), array([110,  68, 208, 130]), array([132,  91, 160, 130])],
        ...(the rest are dropped)
    ]
    ```

Transformations that the data undergo sequentially:
  1. malformed data are dropped
  2. (very) small rooms are dropped
  3. if the set is for training, the floorplans are augmented, that is, the bounding boxes of the rooms are randomly
     flipped and rotated.
  4. the bounding boxes are scaled to 0-1 range
  5. the bounding box of each room is moved so that its center is identical to the center of the image
  6. a graph is created:
    1. the nodes are the rooms
    2. the edges are tuples of 3 elements
  7. the images are created, by rescaling and the bounding boxes and then using them to create 32x32 arrays containing
    -1 and 1 in the places of those bounding boxes of the rooms (-1: no room area, 1: room area)

Output schema:
  Each FloorplanGraphDataset access (__getitem__ or []) returns:
    1. a FloatTensor (why floats?) of shape (n_rooms, 32, 32), 32 being the dimension of the output image (showing the
      bounding box for each room),
    2. a FloatTensor (why floats?) of shape (n_rooms, 10) which is the onehot encoding of the room types,
    3. and a LongTensor of (n_edges, 3), denoting the relationships between the rooms.
  Example:
    For the example given previously:
    ```
    [
        [6.0, 2.0, 4.0],
        [array([array([132,   6, 148,  65]), array([110,  68, 208, 130]), array([132,  91, 160, 130])],
        ...(the rest are dropped)
    ]
    ```
    we get the following 3 tensors:
    1. https://imgur.com/a/HsQkeXg . The 1's are not in the exact locations of the starting bounding boxes because
      when training, the data undergoes augmentation randomly (random flips and multiples of 90deg rotations of the
      rooms)
    2. ```
        tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])
       ```
    3. ```
        tensor([[ 0,  1,  1],
               [ 0, -1,  2],
               [ 1,  1,  2]])
       ```
      The edges tensor (3.) means the following: Room with index 0 (0) is adjacent (1) to room w/ index 1 (1), room with
      index 0 (0) is not adjacent (-1) to room w/ index 2 (2) etc. Rooms overlapping in some way are considered
      adjacent.

Dataset collation:
  When the Dataset is loaded in batches, the collation function floorplan_collate_fn is called. This concatenates
  "vertically" (stacks) all the tensors that __getitem__ returns (so now we get tensors of sum(n_i) sizes,
  i={1 to batch size} and n_i the number of rooms in each floorplan of the batch, same with edges you get the idea),
  and also returns two more tensors, which __are only used when training in parallel__.
  TODO elabolarate further here.
"""

import random

import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

IMAGE_SIZE_IN = 256  # 256x256
IMAGE_SIZE_OUT = 32  # 32x32

# bounding box limits
MIN_H = 0.03
MIN_W = 0.03
ADJACENCY_THRESHOLD = 0.03


def collate(batch):
    n_rooms_total = sum([len(b[0]) for b in batch])
    n_edges_total = sum([len(b[2]) for b in batch])

    # preallocate tensors to speed up
    all_rooms_mks = torch.empty((n_rooms_total, IMAGE_SIZE_OUT, IMAGE_SIZE_OUT), dtype=torch.float)
    all_nodes = torch.empty((n_rooms_total, 10), dtype=torch.float)
    all_edges = torch.empty((n_edges_total, 3), dtype=torch.long) if n_edges_total > 0 else torch.LongTensor([])
    all_node_to_sample = torch.empty(n_rooms_total, dtype=torch.long)
    all_edges_to_sample = torch.empty(n_edges_total, dtype=torch.long)

    node_offset = 0
    edge_offset = 0
    for i, (rooms_mks, nodes, edges) in enumerate(batch):
        n_nodes = len(nodes)
        n_edges = len(edges)

        all_rooms_mks[node_offset:node_offset + n_nodes] = rooms_mks
        all_nodes[node_offset:node_offset + n_nodes] = nodes
        if edges.shape[0] > 0:
            all_edges[edge_offset:edge_offset + n_edges] = edges
            all_edges[edge_offset:edge_offset + n_edges, 0] += node_offset
            all_edges[edge_offset:edge_offset + n_edges, 2] += node_offset
        all_node_to_sample[node_offset:node_offset + n_nodes] = torch.LongTensor(n_nodes).fill_(i)
        all_edges_to_sample[edge_offset:edge_offset + n_edges] = torch.LongTensor(n_edges).fill_(i)


        node_offset += n_nodes
        edge_offset += n_edges

    return all_rooms_mks, all_nodes, all_edges, all_node_to_sample, all_edges_to_sample


def create_loaders(path, train_batch_size=32, test_batch_size=64, loader_threads=8, n_rooms=(10, 12)):
    data = np.load(path, allow_pickle=True)

    # filter the data
    train_data = []
    test_data = []
    for floorplan in data:
        rooms_types = floorplan[0]
        rooms_bbs = floorplan[1]  # bounding boxes

        # discard malformed samples
        # TODO create a version of the dataset that is pre-cleaned
        if not rooms_types or any(i == 0 for i in rooms_types) or any(i is None for i in rooms_bbs):
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
        # TODO remove this stupid 5k limit here after confirming that all is ok without it
        if n_rooms[0] <= len(rooms_types) <= n_rooms[1] and len(train_data) <= 5000:
            test_data.append([types_filtered, bbs_filtered])
        else:
            train_data.append([types_filtered, bbs_filtered])

    # create datasets
    train_dataset = FloorplanGraphDataset(train_data, augment=True)
    test_dataset = FloorplanGraphDataset(test_data)

    # create loaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=loader_threads,
                              collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=loader_threads,
                             collate_fn=collate)

    return train_loader, test_loader


class FloorplanGraphDataset(Dataset):

    def __init__(self, data, augment=False):
        self.data = data
        self.augment = augment
        self.image_shape = (IMAGE_SIZE_IN, IMAGE_SIZE_IN)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        floorplan = self.data[index]

        rooms_type = floorplan[0]
        rooms_bbs = floorplan[1]  # bounding boxes

        if self.augment:
            angle = random.randint(0, 3) * 90.0
            flip = random.randint(0, 1) == 1
            rooms_bbs = [self.augment_bounding_box(bb, angle, flip) for bb in rooms_bbs]
            # mutate per room or per floorplan for all rooms (like orig)??
            # can I move angle/flip inside augment_bounding_box() then?
            # intuitively per room mutation seems better, TODO discuss this
            # rooms_bbs = list(map(self.augment_bounding_box, rooms_bbs))

        rooms_bbs = np.stack(rooms_bbs) / IMAGE_SIZE_IN  # "normalize"

        # find the boundary box and centralize all bounding boxes according to it
        # i.e. move them so that the center of their boundary box matches the
        # center of the (now normalized to [0, 1]) image
        # think in terms of vectors to understand how this works
        top_left = np.min(rooms_bbs[:, :2], axis=0)
        bottom_right = np.max(rooms_bbs[:, 2:], axis=0)
        shift = (top_left + bottom_right) / 2.0 - 0.5  # shifting vector
        # subtract the shifting vector from all points to centralize them
        rooms_bbs[:, :2] -= shift
        rooms_bbs[:, 2:] -= shift

        # create graph (nodes, edges)
        edges = []
        for k in range(len(rooms_type)):
            for l in range(len(rooms_type)):
                if l > k:
                    nd0, bb0 = rooms_type[k], rooms_bbs[k]
                    nd1, bb1 = rooms_type[l], rooms_bbs[l]

                    if is_adjacent(bb0, bb1):
                        edges.append([k, 1, l])
                    else:
                        edges.append([k, -1, l])

        nodes = torch.LongTensor(list(map(int, rooms_type)))
        edges = torch.LongTensor(edges)
        rooms_mks = np.zeros((len(nodes), IMAGE_SIZE_OUT, IMAGE_SIZE_OUT))
        for k, bb in enumerate(rooms_bbs):
            bb *= IMAGE_SIZE_OUT
            x0, y0, x1, y1 = bb.astype(int)
            rooms_mks[k, x0:x1 + 1, y0:y1 + 1] = 1.0

        # onehot encode and drop class 0 because the rooms' classes are 1-10
        nodes = one_hot(nodes, num_classes=11)[:, 1:]
        nodes = nodes.float()

        rooms_mks = torch.FloatTensor(rooms_mks)
        normalize = transforms.Normalize(mean=[0.5],
                                         std=[0.5])  # basically transforms zeros to -1... don't know why this is useful
        rooms_mks = normalize(rooms_mks)

        return rooms_mks, nodes, edges

    def augment_bounding_box(self, bb, angle, flip):
        angle_rad = np.deg2rad(angle)
        x0, y0 = self.flip_and_rotate(np.array([bb[0], bb[1]]), flip, angle_rad)
        x1, y1 = self.flip_and_rotate(np.array([bb[2], bb[3]]), flip, angle_rad)

        xmin, ymin = min(x0, x1), min(y0, y1)
        xmax, ymax = max(x0, x1), max(y0, y1)

        return np.array([xmin, ymin, xmax, ymax]).astype(int)

    def flip_and_rotate(self, vector, flip, angle):
        image_bounds = np.array(self.image_shape)
        center = (image_bounds - 1) / 2

        vector = vector - center
        rot_matrix = np.array([[np.cos(angle), np.sin(angle)],
                               [- np.sin(angle), np.cos(angle)]])
        # clockwise rotation by angle rads
        vector = np.dot(rot_matrix, vector)
        vector = vector + center

        x, y = vector

        if flip:
            # mirror around the x = shape/2 line (around the middle of the x dim)
            mid = IMAGE_SIZE_IN / 2
            dist = abs(mid - x)
            x = mid - dist if x > mid else mid + dist

        return x, y


def is_adjacent(box_a, box_b, threshold=ADJACENCY_THRESHOLD):
    # returns true if two bounding boxes are overlapping in some way or are adjacent
    x0, y0, x1, y1 = box_a
    x2, y2, x3, y3 = box_b

    h1, h2 = x1 - x0, x3 - x2
    w1, w2 = y1 - y0, y3 - y2

    xc1, xc2 = (x0 + x1) / 2.0, (x2 + x3) / 2.0
    yc1, yc2 = (y0 + y1) / 2.0, (y2 + y3) / 2.0

    delta_x = np.abs(xc2 - xc1) - (h1 + h2) / 2.0
    delta_y = np.abs(yc2 - yc1) - (w1 + w2) / 2.0

    delta = max(delta_x, delta_y)

    return delta < threshold
