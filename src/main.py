#!/usr/bin/env python

from argparse import ArgumentParser

import numpy as np

from data import create_loaders
import torch
from torch.autograd import Variable

from src.models import Generator, Discriminator

parser = ArgumentParser()
parser.add_argument('--data', type=str, default='../data/train.npy', help='Training data path')
parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size')
parser.add_argument('--test_batch_size', type=int, default=64, help='Testing batch size')
parser.add_argument('--loader_threads', type=int, default=8, help='Number of threads of the data loader')
parser.add_argument("--n_epochs", type=int, default=1000000, help="number of epochs of training")

args = parser.parse_args()

train_loader, test_loader = create_loaders(args.data, args.train_batch_size, args.test_batch_size, args.loader_threads,
                                           n_rooms=(10, 12))

# ----------
#  Training
# ----------
generator = Generator()
discriminator = Discriminator()
adversarial_loss = torch.nn.BCEWithLogitsLoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

batches_done = 0
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
Tensor = torch.FloatTensor
for epoch in range(args.n_epochs):
    for i, batch in enumerate(train_loader):
        # Unpack batch
        mks, nds, eds, nd_to_sample, ed_to_sample = batch
        indices = nd_to_sample, ed_to_sample

        # Adversarial ground truths
        batch_size = torch.max(nd_to_sample) + 1
        valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_mks = Variable(mks.type(Tensor))
        given_nds = Variable(nds.type(Tensor))
        given_eds = eds

        for p in discriminator.parameters():
            p.requires_grad = True

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Generate a batch of images
        z_shape = [real_mks.shape[0], 128]
        z = Variable(Tensor(np.random.normal(0, 1, tuple(z_shape))))
        gen_mks = generator(z, given_nds, given_eds)

        real_validity = discriminator(real_mks, given_nds, given_eds, nd_to_sample)
        fake_validity = discriminator(gen_mks.detach(), given_nds.detach(), \
                                          given_eds.detach(), nd_to_sample.detach())
        d_loss = adversarial_loss(real_validity, fake_validity)

        # Update discriminator
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Set grads off
        for p in discriminator.parameters():
            p.requires_grad = False

        # Train the generator every n_critic steps
        if i % 1 == 0:

            # Generate a batch of images
            z = Variable(Tensor(np.random.normal(0, 1, tuple(z_shape))))
            gen_mks = generator(z, given_nds, given_eds)
            fake_validity = discriminator(gen_mks, given_nds, given_eds, nd_to_sample)

            # Update generator
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            optimizer_G.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                  % (epoch, args.n_epochs, i, len(train_loader), d_loss.item(), g_loss.item()))