"""Things to do before running this file:
        set shuffle=True test_loader on data.py in order to be consistant with what is done in the paper
        set correct arguments and checkpoint path
"""

# before running this set shuffle=True test_loader on data.py

import math
import sys
import random
import numpy as np
import argparse
import os

from PIL import Image, ImageDraw

import torchvision.transforms as transforms
from torchvision.utils import save_image

# todo: added this to import staff from nikos script
from data import create_loaders

from torch.autograd import Variable
import torch.nn as nn
import torch
from utils import mask_to_bb, ID_COLOR, bb_to_im_fid
from models import Generator


import svgwrite
from collections import defaultdict
import matplotlib.pyplot as plt



# TODO : remomve redundant args
parser = argparse.ArgumentParser()
parser.add_argument("--n-cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument('--data', type=str, default='./data/train_data.npy', help='Training data path')
parser.add_argument('--train-batch-size', type=int, default=1, help='Training batch size')
parser.add_argument('--test-batch-size', type=int, default=1, help='Testing batch size')
parser.add_argument("--latent-dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--num-variations", type=int, default=10, help="number of variations")
parser.add_argument("--exp-folder", type=str, default='exp', help="destination folder")
parser.add_argument('--loader-threads', type=int, default=4, help='Number of threads of the data loader')
parser.add_argument("--target_set", type=str, default='C', help="which split to remove")
parser.add_argument("--phase", type=str, default='eval', help="phase split")
args = parser.parse_args()
print(args)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO : update checkpoint path 
# checkpoint = '../src/exp_D_224250.pth'
checkpoint = './exp_D_224250.pth'


# create folders to place generated and real figures in
exp_name = 'exp_with_graph_global_new'
path_real = './FID/{}_{}/real'.format(exp_name, args.target_set)
path_fake = './FID/{}_{}/fake'.format(exp_name, args.target_set)
os.makedirs(path_real, exist_ok=True)
os.makedirs(path_fake, exist_ok=True)


generator = Generator()
generator.load_state_dict(torch.load(checkpoint, map_location = torch.device('cpu'))['gen_state_dict'])



Tensor = torch.FloatTensor

# substitute line above with line below before pushing and remove if below
# Tensor = Tensor.to(device)

if device.type == "cuda":
    generator.cuda()
    Tensor = torch.cuda.FloatTensor
#     Tensor = Tensor.to(device)


_ , test_loader = create_loaders(args.data, args.train_batch_size, args.test_batch_size, args.loader_threads,
                                           n_rooms=(7, 9))
                                           

# ================================================ #
#                     Vectorize                    #
# ================================================ #

# initialize
globalIndexReal = 0
globalIndexFake = 0
finalImages = []


for idx, minibatch in enumerate(test_loader):
        # Unpack the minibatch
        masks, nds, eds, nds_to_sample, eds_to_sample = minibatch

        real_masks = Variable(masks.type(Tensor))
        given_nds = Variable(nds.type(Tensor))
        given_eds = eds

        for var in range(args.num_variations):
                z = Variable(Tensor(np.random.normal(0, 1, (real_masks.shape[0], args.latent_dim))))
                with torch.no_grad():
                        gen_masks = generator(z, given_nds, given_eds)
                        gen_bbs = np.array([np.array(mask_to_bb(mk)) for mk in gen_masks.detach().to(device)])
                        real_bbs = np.array([np.array(mask_to_bb(mk)) for mk in real_masks.detach().to(device)])
                        real_nodes = np.where(given_nds.detach().to(device)==1)[-1]
                
                # here we generate multiple fake images for each graph as it is a better practice for diversity calculation. 
                # so the ration between fake and real images is num-variations	
                if var == 0:
                        real_bbs = real_bbs[np.newaxis, :, :]/32.0		# brink the room masksback to range 0-1 
                        real_im = bb_to_im_fid(real_bbs, real_nodes)
                        real_im.save('{}/{}.jpg'.format(path_real, globalIndexReal))
                        globalIndexReal += 1
                
                # draw vector
                gen_bbs = gen_bbs[np.newaxis, :, :]/32.0
                fake_im = bb_to_im_fid(gen_bbs, real_nodes)
                fake_im.save('{}/{}.jpg'.format(path_fake, globalIndexFake))
                globalIndexFake += 1


