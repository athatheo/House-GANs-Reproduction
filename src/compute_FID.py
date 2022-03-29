import numpy as np
import argparse
import os
import torchvision.transforms as transforms

import torchvision.transforms as transforms
# from torchvision.utils import save_image

# todo: added this to import staff from nikos script
from data import create_loaders

from floorplan_dataset_maps import FloorplanGraphDataset, floorplan_collate_fn

# import torch.nn as nn
# import torch.nn.functional as F
# import torch.autograd as autograd
import torch

# from PIL import Image, ImageDraw
# from reconstruct import reconstructFloorplan
# import svgwrite

# from utils import bb_to_img, bb_to_vec, bb_to_seg, mask_to_bb, remove_junctions, ID_COLOR, bb_to_im_fid
from utils import mask_to_bb, ID_COLOR
from models import Generator

# from collections import defaultdict
# import matplotlib.pyplot as plt
# import networkx as nx


# TODO : remomve redundant args
parser = argparse.ArgumentParser()
parser.add_argument("--n-cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent-dim", type=int, default=128, help="dimensionality of the latent space")
# parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--num-variations", type=int, default=10, help="number of variations")
parser.add_argument("--exp-folder", type=str, default='exp', help="destination folder")

parser.add_argument('--data', type=str, default='../data/train.npy', help='Training data path')
parser.add_argument('--train-batch-size', type=int, default=32, help='Training batch size')
parser.add_argument('--test-batch-size', type=int, default=64, help='Testing batch size')
parser.add_argument('--loader-threads', type=int, default=8, help='Number of threads of the data loader')
parser.add_argument("--target_set", type=str, default='A', help="which split to remove")
parser.add_argument("--phase", type=str, default='eval', help="phase split")

args = parser.parse_args()
print(args)

device = "cuda" if torch.cuda.is_available() else "cpu"

# create folders to place generated and real figures in
exp_name = 'exp_with_graph_global_new'
path_real = './FID/{}_{}/real'.format(exp_name, args.target_set)
path_fake = './FID/{}_{}/fake'.format(exp_name, args.target_set)
os.makedirs(path_real, exist_ok=True)
os.makedirs(path_fake, exist_ok=True)

# TODO : update checkpoint path 
checkpoint = '../housegan/exp_demo_D_500000.pth'

# TODO : probably redundant
room_path = args.data


generator = Generator()
# todo : check line below for chckpoint
# todo : removed line below ( check again)
# generator.load_state_dict(torch.load(checkpoint,map_location = torch.device('cpu')))
# generator.to(device)

generator.load_state_dict(torch.load(checkpoint,map_location = torch.to(device))


# fp_dataset_test

# use command below to normalize the floorplans

# import test_loader from create_loaders of data.py that niko created
# after importing it , normalize it

train_loader, test_loader = create_loaders(args.data, args.train_batch_size, args.test_batch_size, args.loader_threads,
                                           n_rooms=(0, 3))
                                           
# TODO: remove lines below if normalization is done in loop
#normalize = transforms.Normalize(mean=[0.5], std=[0.5])
#test_loader = normalize(test_loader)


# ================================================ #
#                     Vectorize                    #
# ================================================ #

# initialize
globalIdxReal = 0
globalIdxFake = 0
finalImages = []


for idx, minibatch in enumerate(test_loader):
        # Unpack the minibatch
        masks, nds, eds, nds_to_sample, eds_to_sample = minibatch
        # TODO: Remove line below
        #   indices = nds_to_sample, eds_to_sample

        real_masks = Variable(masks.type(Tensor))
        given_nds = Variable(nds.type(Tensor))
        given_eds = eds

	for var in range(args.num-variations):
	z = Variable(Tensor(np.random.normal(0, 1, (real_masks.shape[0], opt.latent_dim))))
	with torch.no_grad():
		gen_masks = generator(z, given_nds, given_eds)
            	gen_bbs = np.array([np.array(mask_to_bb(mk)) for mk in gen_masks.detach().to(device)])
            	real_bbs = np.array([np.array(mask_to_bb(mk)) for mk in real_mks.detach().to(device)])
            	real_nodes = np.where(given_nds.detach().to(device)==1)[-1]
            	
	if k == 0:
	        real_bbs = real_bbs[np.newaxis, :, :]/32.0
	        real_im = bb_to_im_fid(real_bbs, real_nodes)
        	real_im.save('{}/{}.jpg'.format(path_real, globalIndexReal))
            	globalIndexReal += 1
        
        # draw vector
        gen_bbs = gen_bbs[np.newaxis, :, :]/32.0
        fake_im = bb_to_im_fid(gen_bbs, real_nodes)
        fake_im.save('{}/{}.jpg'.format(path_fake, globalIndexFake))
        globalIndexFake += 1


