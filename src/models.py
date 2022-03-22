import torch
from torch import cat
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ConvTranspose2d
from torch.nn import LeakyReLU
from torch.nn import Tanh
from torch.nn import MaxPool2d
from torch import zeros_like


class ConvMPN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3*16, out_channels=2*16, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = Conv2d(in_channels=2*16, out_channels=2*16, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = Conv2d(in_channels=2*16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def get_nodes(self, feature_vectors, edges, include_neighbours=True):
        device = torch.cuda.current_device()#torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nodes = zeros_like(feature_vectors, device=device)
        if include_neighbours:
            index = torch.where(edges[:, 1] > 0)
        else:
            index = torch.where(edges[:, 1] < 0)

        src = torch.cat([edges[index[0], 0], edges[index[0], 2]]).long()
        dst = torch.cat([edges[index[0], 2], edges[index[0], 0]]).long()
        src = feature_vectors[src.contiguous()]
        dst = dst.view(-1, 1, 1, 1).expand_as(src)
        return nodes.scatter_add(0, dst, src)

    def cat_nodes(self, feature_vectors, edges):
        neighbouring_nodes = self.get_nodes(feature_vectors, edges, include_neighbours=True, )
        non_neighbouring_nodes = self.get_nodes(feature_vectors, edges, include_neighbours=False)

        encoding = torch.cat([feature_vectors, neighbouring_nodes, non_neighbouring_nodes], 1)
        return encoding

    def forward(self, x, edges):
        x = self.cat_nodes(x, edges)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class Generator(Module):
    def __init__(self):
        super().__init__()
        self.linear_reshape_1 = Linear(138, 1024)
        self.conv_mpn_1 = ConvMPN()
        self.upsample_1 = ConvTranspose2d(16, 16, 4, 2, 1)
        self.conv_mpn_2 = ConvMPN()
        self.upsample_2 = ConvTranspose2d(16, 16, 4, 2, 1)
        self.conv_1 = Conv2d(16, 256, 3, 1, 1)
        self.leaky_relu = LeakyReLU(0.1)
        self.conv_2 = Conv2d(256, 128, 3, 1, 1)
        self.conv_3 = Conv2d(128, 1, 3, 1, 1)
        self.tanh = Tanh()

    def forward(self, z, t, edges):
        z = z.view(-1, 128)#
        t = t.view(-1, 10) #
        x = cat([z, t], 1)
        x = self.linear_reshape_1(x)
        x = x.view(-1, 16, 8, 8)
        x = self.conv_mpn_1(x, edges).view(-1, *x.shape[1:])
        x = self.upsample_1(x)
        x = self.conv_mpn_2(x, edges).view(-1, *x.shape[1:])
        x = self.upsample_2(x)
        x = self.conv_1(x.view(-1, x.shape[1], *x.shape[2:]))
        x = self.leaky_relu(x)
        x = self.conv_2(x)
        x = self.leaky_relu(x)
        x = self.conv_3(x)
        x = self.tanh(x)
        x = x.view(-1, *x.shape[2:])
        return x


class Discriminator(Module):
    def __init__(self):
        super().__init__()
        self.linear_reshape_1 = Linear(10, 8192)
        self.leaky_relu = LeakyReLU(0.1)
        self.conv_1 = Conv2d(9, 16, 3, 1, 1, bias=True)
        self.conv_2 = Conv2d(16, 16, 3, 1, 1)
        self.conv_3 = Conv2d(16, 16, 3, 1, 1)
        self.conv_mpn_1 = ConvMPN()
        self.downsample_1 = Conv2d(16, 16, 3, 2, 1)
        self.conv_mpn_2 = ConvMPN()
        self.downsample_2 = Conv2d(16, 16, 3, 2, 1)
        self.dec_conv_1 = Conv2d(16, 256, 3, 2, 1)
        self.dec_conv_2 = Conv2d(256, 128, 3, 2, 1)
        self.dec_conv_3 = Conv2d(128, 128, 3, 2, 1)
        self.pool_reshape_linear = Linear(128, 1)

    def add_pool(self, x, nd_to_sample):
        dtype, device = x.dtype, x.device
        batch_size = torch.max(nd_to_sample) + 1
        pooled_x = torch.zeros(batch_size, x.shape[-1], device=device).float()
        pool_to = nd_to_sample.view(-1, 1).expand_as(x)
        pooled_x = pooled_x.scatter_add(0, pool_to, x)
        return pooled_x

    def forward(self, x, t, edges, nd_to_sample):
        x = x.view(-1, 1, 32, 32)
        t = self.linear_reshape_1(t)
        t = t.view(-1, 8, 32, 32)
        x = cat([x, t], 1)
        x = self.conv_1(x)
        x = self.leaky_relu(x)
        x = self.conv_2(x)
        x = self.leaky_relu(x)
        x = self.conv_3(x)
        x = self.leaky_relu(x)
        x = self.conv_mpn_1(x, edges)
        x = self.downsample_1(x)
        x = self.leaky_relu(x)
        x = self.conv_mpn_2(x, edges)
        x = self.downsample_2(x)
        x = self.leaky_relu(x)
        x = self.dec_conv_1(x)
        x = self.leaky_relu(x)
        x = self.dec_conv_2(x)
        x = self.leaky_relu(x)
        x = self.dec_conv_3(x)
        x = self.leaky_relu(x)
        x = x.view(-1, x.shape[1])
        x = self.add_pool(x, nd_to_sample)
        x = self.pool_reshape_linear(x)
        return x
