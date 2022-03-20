# import libraries
import torch
import numpy as np
from torch.autograd import Variable

import os

from argparse import ArgumentParser

from data import create_loaders

parser = ArgumentParser()
parser.add_argument('--data', type=str, default='../data/train.npy', help='Training data path')
parser.add_argument('--train-batch-size', type=int, default=32, help='Training batch size')
parser.add_argument('--test-batch-size', type=int, default=64, help='Testing batch size')
parser.add_argument('--loader-threads', type=int, default=8,
                    help='Number of threads of the data loader')
args = parser.parse_args()

# set important training parameters
# generator
lr_gen = 0.0001
betas_gen = (0.5, 0.999)
# discriminator
lr_dis = 0.0001
betas_dis = (0.5, 0.999)
# training
epochs = 1000000
noise_dim = 128
lambda_gp = 10
sample_interval = 50000

# Check for gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create folders to put result files and checkpoints
target_set = "A"  # TODO - See how this is added as an argument during evaluation!
exp_folder = "exp+" + target_set
os.makedirs("./exps/" + exp_folder, exist_ok=True)
os.makedirs("./checkpoints/", exist_ok=True)
os.makedirs("./temp/", exist_ok=True)

# Initialize generator and discriminator and their optimizers
# TODO - Update generator/discriminator models
generator = "generator"  # Generator()
opti_gen = torch.optim.Adam(generator.parameters(), lr=lr_gen, betas=betas_gen)

discriminator = "discriminator"  # Discriminator()
opti_dis = torch.optim.Adam(discriminator.parameters(), lr=lr_dis, betas=betas_dis)

Tensor = torch.FloatTensor

# If there is a gpu, put everything in it
if device == "cuda":
    generator.cuda()
    discriminator.cuda()
    Tensor = torch.cuda.FloatTensor

# TODO - Create functions to run everything on parallel (graph_scatter, data_parallel)
multi_gpu = False
data_parallel = False

# TODO - Visualize a single batch

# Load train and test data
train_loader, test_loader = create_loaders(args.data, args.train_batch_size, args.test_batch_size,
                                           args.loader_threads, n_rooms=(10, 12))


def compute_gradient_penalty(dis, real, fake, given_nds=None, given_eds=None, indices=None,
                             data_parallel=None):
    batch_size = torch.max(nds_to_sample) + 1
    dtype, device = real.dtype, real.device
    alpha = torch.FloatTensor(real.shape[0], 1, 1).to(device)
    alpha.data.resize_(real.shape[0], 1, 1)
    alpha.uniform_(0, 1)

    # TODO - what about these instead of the above?
    # batch_size, C, L = real.shape
    # alpha = torch.rand((batch_size, 1, 1)).repeat(1, C, L)

    x_both = real.data * alpha + fake.data * (1 - alpha)
    x_both = x_both.to(device)
    x_both = Variable(x_both, requires_grad=True)
    grad_outputs = torch.ones(batch_size, 1).to(device)
    # TODO - why not the below?
    # grad_outputs = torch.ones_like(x_both)
    if data_parallel:
        dis_out = data_parallel(dis, (x_both, given_nds, given_eds, indices[0]), indices)
    else:
        dis_out = dis(x_both, given_nds, given_eds, indices[0])
    grad = torch.autograd.grad(outputs=dis_out,
                               inputs=x_both,
                               grad_outputs=grad_outputs,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]
    gradient_penalty = ((grad.norm(2, 1).norm(2, 1) - 1) ** 2).mean()
    # TODO - Why not the below lines?
    # gradient = grad.view(grad.shape[0], -1)
    # gradient_norm = gradient.norm(2, dim=1)
    # gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty


# Training
for epoch in range(epochs):
    for idx, minibatch in enumerate(train_loader):

        # Unpack the minibatch
        masks, nds, eds, nds_to_sample, eds_to_sample = minibatch
        indices = nds_to_sample, eds_to_sample
        # TODO : What happens with nds_to_sample, eds_to_sample

        # Not needed!
        # Adversarial ground truths
        # batch_size = torch.max(nds_to_sample) + 1
        # valid = Variable(torch.Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
        # fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # TODO - update names
        real_masks = Variable(masks.type(Tensor))
        given_nds = Variable(nds.type(Tensor))
        given_eds = eds

        # Train the discriminator
        # Set grads on # TODO - why??
        for p in discriminator.parameters():
            p.requires_grad = True
        opti_dis.zero_grad()

        # Generate masks for each room
        z = Variable(Tensor(np.random.normal(0, 1, (given_nds.shape[0], noise_dim))))
        # if multi_gpu:
        if multi_gpu:
            pass
            # gen_masks = data_parallel(generator, (z, given_nds, given_eds), indices)
        else:
            gen_masks = generator(z, given_nds, given_eds)

        # Evaluate with discriminator
        # Real masks
        if multi_gpu:
            pass
            # real_validity = data_parallel(discriminator, \
            #                               (real_mks, given_nds, \
            #                                given_eds, nd_to_sample), \
            #                               indices)
        else:
            real_validity = discriminator(real_masks, given_nds, given_eds, indices)

        # Generated images
        if multi_gpu:
            pass
            # fake_validity = data_parallel(discriminator, \
            #                               (gen_mks.detach(), given_nds.detach(), \
            #                                given_eds.detach(), nd_to_sample.detach()), \
            #                               indices)
        else:
            fake_validity = discriminator(gen_masks.detach(), given_nds.detach(), \
                                          given_eds.detach(), nds_to_sample.detach())

        # Measure the discriminator gradient penalty
        if multi_gpu:
            pass
            # gradient_penalty = compute_gradient_penalty(discriminator, real_masks.data,
            #                                             gen_masks.data,
            #                                             given_nds=given_nds.data,
            #                                             given_eds=given_eds.data,
            #                                             indices=indices,
            #                                             data_parallel=data_parallel)
        else:
            gradient_penalty = compute_gradient_penalty(discriminator, real_masks.data,
                                                        gen_masks.data,
                                                        given_nds=given_nds.data,
                                                        given_eds=given_eds.data,
                                                        indices=indices)

        # Compute the discriminator loss with gradient penalty
        dis_loss = - torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * \
                   gradient_penalty

        # Update the discriminator weights and perform one step in the optimizer
        discriminator.backward()
        opti_dis.step()

        # Set grads off
        for p in discriminator.parameters():
            p.requires_grad = False

        # ==========================================================================================
        # Train the generator
        # ==========================================================================================
        opti_gen.zero_grad()

        # TODO - Here, a new mask is generated from the generator network! Why does this happen??
        # Also, no parallel
        # Commenting out for now
        # Generate a batch of images
        # z = Variable(Tensor(np.random.normal(0, 1, (given_nds.shape[0], noise_dim))))
        # gen_masks = generator(z, given_nds, given_eds)
        #
        # # Score fake images
        # if multi_gpu:
        #     fake_validity = data_parallel(discriminator, \
        #                                   (gen_masks, given_nds, \
        #                                    given_eds, nds_to_sample), \
        #                                   indices)
        # else:
        #     fake_validity = discriminator(gen_masks, given_nds, given_eds, nds_to_sample)

        # TODO - If the above is redundant, then this is relevant
        if multi_gpu:
            fake_validity = data_parallel(discriminator, \
                                          (gen_masks.detach(), given_nds.detach(), \
                                           given_eds.detach(), nds_to_sample.detach()), \
                                          indices)
        else:
            fake_validity = discriminator(gen_masks.detach(), given_nds.detach(), \
                                          given_eds.detach(), nds_to_sample.detach())

        # Compute the generator loss
        gen_loss = -torch.mean(fake_validity)

        # Update the generator weights and perform one step in the optimizer
        generator.backward()
        opti_gen.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
              % (epoch, epochs, idx, len(train_loader), dis_loss.item(), gen_loss.item()))

        # Save a checkpoint, if the epoch is over
        batches_done = epoch * len(train_loader) + idx
        if (batches_done % sample_interval == 0) and batches_done:
            torch.save(generator.state_dict(), './checkpoints/{}_{}.pth'.format(exp_folder, batches_done))
            # TODO - Visualize batch
            # visualizeSingleBatch(test_loader, opt)
        # batches_done += opt.n_critic
